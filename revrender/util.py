"""Helpers every module shares: digests, create-only writes, the frozen-path guard and run manifests."""
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42                        # the only seed this project runs (user rule)
FROZEN_PATH = re.compile(r'frozen|holdout|冻结|封存|(?:^|[_-])m6(?:$|[_-])', re.I)


class RefusedError(RuntimeError):
    """A declared contract was violated; the run stops before producing anything."""


def require(condition, message, error=RefusedError):
    if not condition:
        raise error(message)


def schema_is(value, pattern):
    """Whether a data file's schema tag matches a regular expression (tags of existing data carry a historical id)."""
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_path(path):
    """Resolve a path, refusing anything that names a frozen or held-out location."""
    expanded = Path(path).expanduser()
    for candidate in (expanded.absolute(), expanded.resolve()):
        for part in candidate.parts:
            require(not FROZEN_PATH.search(part), f'frozen/held-out path refused: {path}')
    return expanded.resolve()


def write_json(path, value, indent=2):
    """Create-only: an existing file is an error, never overwritten."""
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=indent, sort_keys=True, allow_nan=False)
        f.write('\n')


def make_output_dir(path, parents=True):
    """Create-only directory: an existing path (or symlink) is an error."""
    path = Path(path).expanduser()
    require(not path.exists() and not path.is_symlink(), f'create-only output exists: {path}', FileExistsError)
    path.mkdir(parents=parents, exist_ok=False)
    return path


def git_state(root=ROOT):
    """The commit the code was built from; a deployed snapshot carries it in deployment_origin.json."""
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return {'commit': commit, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        origin = Path(root)/'deployment_origin.json'
        if origin.is_file():
            record = json.loads(origin.read_text())
            return {'commit': record['source_commit'], 'dirty': False, 'deployed_snapshot': True}
        return {'commit': None, 'dirty': None}


def input_hashes(paths):
    """{path: sha256} of the files a run read, for its manifest (directories are named by their run_manifest.json)."""
    out = {}
    for path in paths:
        path = Path(path)
        target = path/'run_manifest.json' if path.is_dir() else path
        out[str(path)] = digest(target)
    return out


def provenance(config, inputs=None, seed=SEED):
    """What a run records about itself: the configuration, the seed, the git commit and the hashes of its input files."""
    import numpy
    record = {'config': config, 'seed': seed, 'git': git_state(), 'inputs': dict(inputs or {}),
              'environment': {'python': platform.python_version(), 'numpy': numpy.__version__,
                              'platform': platform.platform(), 'argv': list(sys.argv)},
              'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        import torch
        record['environment'].update(torch=torch.__version__, cuda=torch.version.cuda)
    except ImportError:
        pass
    return record


def write_manifest(out_dir, name='run_manifest.json', *, config, inputs=None, seed=SEED, **extra):
    """One manifest per run, create-only: provenance (configuration, seed, git commit, input hashes) plus what the run adds."""
    record = {**provenance(config, inputs, seed), **extra}
    write_json(Path(out_dir)/name, record)
    return record


def to_jsonable(namespace):
    """An argparse namespace as plain JSON values (paths, alone or in a list, as strings)."""
    text = lambda v: str(v) if isinstance(v, Path) else v
    return {k: [text(x) for x in v] if isinstance(v, list) else text(v) for k, v in vars(namespace).items()}
