"""Run a recorded command again on other inputs: the command line in an output's run_manifest.json, with paths remapped, a labmate's
page folders left out and a new output directory.  Used to rebuild what the labmate's data reached (user 2026-10-21) from the commands
that actually ran, instead of retyping them.

    python -m revrender.tools.replay --from OLD_OUT --to NEW_OUT [--map OLD=NEW ...] [--set --option=VALUE ...] [--print]

``--map`` replaces a path prefix in every argument (longest first); ``--set`` gives an option a new value (the option must be there);
``--pages`` / ``--images`` folders of the labmate's scans (``full1026``) are dropped, with their option.  The module is the one the
manifest ran, from this checkout; it writes its own run_manifest.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from ..util import ROOT, require

DROPPED = 'full1026'                 # the folder of the labmate's scans (data.allowed)
FOLDERS = ('--pages', '--images')


def recorded(old):
    """The argv the output's run_manifest.json recorded: [script path, arguments...]."""
    manifest = json.loads((Path(old)/'run_manifest.json').read_text())
    argv = (manifest.get('environment') or {}).get('argv')
    require(argv and argv[0].endswith('.py') and '/revrender/' in argv[0], f'{old}: no recorded revrender command')
    return argv


def module_of(script):
    """revrender.x.y from …/revrender/x/y.py."""
    parts = Path(script).with_suffix('').parts
    return '.'.join(parts[len(parts)-1-parts[::-1].index('revrender'):])


def rewrite(argv, new, maps=(), sets=()):
    """(module, arguments) of a recorded argv for the new output."""
    maps = sorted(maps, key=lambda m: -len(m[0]))
    def mapped(value):
        for old, to in maps:
            if value == old or value.startswith(old.rstrip('/')+'/'):
                return to+value[len(old):]
        return value
    args, out, k = argv[1:], [], 0
    while k < len(args):
        a = args[k]
        if a in FOLDERS and k+1 < len(args) and DROPPED in args[k+1]:
            k += 2
            continue
        out.append(mapped(a))
        k += 1
    for option, value in sets:
        require(option in out, f'{option} is not in the recorded command')
        out[out.index(option)+1] = value
    require('--output-dir' in out, 'the recorded command has no --output-dir')
    out[out.index('--output-dir')+1] = str(new)
    require(not any(DROPPED in a for a in out), 'a labmate path is still in the command')
    return module_of(argv[0]), out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument('--from', dest='old', type=Path, required=True)
    p.add_argument('--to', type=Path, required=True)
    p.add_argument('--map', action='append', default=[], help='OLD=NEW path prefix')
    p.add_argument('--set', action='append', default=[], help='--option=VALUE')
    p.add_argument('--print', action='store_true', help='print the command and stop')
    args = p.parse_args(argv)
    module, rest = rewrite(recorded(args.old), args.to, [m.split('=', 1) for m in args.map], [s.split('=', 1) for s in args.set])
    command = [sys.executable, '-m', module, *rest]
    print(json.dumps(command), flush=True)
    if args.print:
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
