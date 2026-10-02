"""Inference speed of RevRender and the native instance-segmentation baselines, one validation line at a time.

  ours     RevRender at its frozen checkpoint
  native   one native baseline at its frozen checkpoint (in the native environment)
  collect  (CPU) the readings side by side, in the paper's order

Every model runs under the same settings: one line per call (batch 1), FP32 with TF32 off, cuDNN autotuning off and
deterministic algorithms off (strict determinism, which RevRender trained under, would slow it, and the baselines never
ran under it), the line already on the GPU, and the clock read after torch.cuda.synchronize() on both sides of the call, so
a reading is the model's own forward pass and post-processing on the GPU, without host transfers.  ``WARMUP`` lines are run
first and not timed; then every validation line is timed ``PASSES`` times.  The per-call times, their median and mean,
lines per second (from the mean), the parameter count and the peak GPU memory are kept.

Nothing here changes a model, its checkpoint or its operating point; the launcher runs it on an idle card and the reading
records which.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from ..data.bundle import load_bundle
from ..util import SEED, digest, provenance, require, to_jsonable
from .freeze import is_freeze
from .matching import validation_order

SCHEMA = 'hme-revision-inference-speed-v1'
WARMUP, PASSES = 20, 3
ORDER = ('rvr', 'maskrcnn', 'pointrend', 'condinst', 'solov2', 'mask2former', 'maskdino', 'dimaskdino')
NATIVE = ORDER[1:]


def summarize(times_ms):
    """Median, mean, 10th and 90th percentiles of the per-call times, and lines per second from the mean."""
    require(len(times_ms) > 0, 'no timed calls')
    ordered = sorted(float(t) for t in times_ms)
    n = len(ordered)

    def quantile(q):
        position = q*(n-1)
        low = int(position)
        high = min(low+1, n-1)
        return ordered[low]+(ordered[high]-ordered[low])*(position-low)

    mean = sum(ordered)/n
    return {'calls': n, 'median_ms': quantile(.5), 'mean_ms': mean, 'p10_ms': quantile(.1), 'p90_ms': quantile(.9),
            'lines_per_second': 1000./mean}


def common_settings(torch):
    """The one setting every model is timed under."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {'batch': 1, 'precision': 'fp32', 'tf32': False, 'cudnn_benchmark': False, 'deterministic_algorithms': False,
            'input': 'already on the GPU', 'clock': 'perf_counter between cuda synchronizations',
            'warmup_lines': WARMUP, 'passes': PASSES}


def environment(torch):
    return {'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'device': torch.cuda.get_device_name(0), 'python': platform.python_version(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def time_calls(torch, call, inputs):
    """Warm up on the first WARMUP inputs, then time every input PASSES times; returns the times and the peak memory."""
    with torch.no_grad():
        for x in inputs[:WARMUP]:
            call(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(PASSES):
            for x in inputs:
                torch.cuda.synchronize()
                start = time.perf_counter()
                call(x)
                torch.cuda.synchronize()
                times.append(1000.*(time.perf_counter()-start))
    return times, torch.cuda.max_memory_allocated()/2**20


def frozen_record(path, model, bundle_sha):
    frozen = json.loads(Path(path).read_text())
    require(is_freeze(frozen.get('schema')) and frozen.get('test_touched') is False and frozen.get('bundle_sha256') == bundle_sha,
            f'not the validation freeze of {model}')
    require(digest(frozen['checkpoint']) == frozen['checkpoint_sha256'], 'the checkpoint is not the frozen one')
    return frozen


def ours(args, rows):
    import torch
    from .. import determinism
    from ..infer import load_checkpoint
    frozen = frozen_record(args.freeze, 'rvr', args.bundle_sha)
    determinism.configure()
    determinism.install()
    torch.manual_seed(SEED)
    model, _ = load_checkpoint(frozen['checkpoint'])
    settings = common_settings(torch)
    inputs = [torch.from_numpy(np.asarray(r['image'][0])[None, None].astype(np.float32)/127.5-1).cuda() for r in rows]
    times, peak = time_calls(torch, model, inputs)
    return frozen, settings, environment(torch), sum(p.numel() for p in model.parameters()), times, peak


def native(args, rows):
    import torch
    from ..baselines.native import factory
    frozen = frozen_record(args.freeze, args.method, args.bundle_sha)
    require(frozen.get('model') == args.method, f'the freeze is of {frozen.get("model")}, not {args.method}')
    kind = factory.get_pretraining_kind(args.method)
    weight = args.weights_dir/f'R-50-{"MSRA" if kind == "msra" else kind}.pkl'
    torch.manual_seed(SEED)
    model = factory.build_model(args.method, args.upstream_root, weight, device='cuda', seed=SEED, smoke=False)
    model.load_state_dict(torch.load(frozen['checkpoint'], map_location='cpu', weights_only=True)['model'])
    model.eval()
    settings = common_settings(torch)
    inputs = [{'image': torch.from_numpy(np.asarray(r['image'], dtype=np.float32).copy()).cuda(),
               'height': r['mask'].shape[0], 'width': r['mask'].shape[1]} for r in rows]
    times, peak = time_calls(torch, lambda item: model([item]), inputs)
    return frozen, settings, environment(torch), sum(p.numel() for p in model.parameters()), times, peak


def collect(paths):
    readings = {}
    for path in paths:
        reading = json.loads(Path(path).read_text())
        require(reading.get('schema') == SCHEMA and reading.get('stage') in ('ours', 'native'), f'{path}: not a speed reading')
        require(reading.get('test_touched') is False, f'{path}: touched the test split')
        require(reading['model'] not in readings, f'{reading["model"]} read twice')
        readings[reading['model']] = reading
    require(set(readings) == set(ORDER), f'missing readings: {sorted(set(ORDER)-set(readings))}')
    devices = {r['environment']['device'] for r in readings.values()}
    require(len(devices) == 1, f'readings come from different cards: {sorted(devices)}')
    rows = [dict(model=k, parameters_m=readings[k]['parameters']/1e6, peak_memory_mib=readings[k]['peak_memory_mib'],
                 **{f: readings[k]['summary'][f] for f in ('median_ms', 'mean_ms', 'p10_ms', 'p90_ms', 'lines_per_second')})
            for k in ORDER]
    return {'schema': SCHEMA, 'stage': 'collect', 'test_touched': False, 'device': devices.pop(), 'rows': rows,
            'inputs_sha256': {str(p): digest(p) for p in paths}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('stage', choices=('ours', 'native', 'collect'))
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--bundle-sha')
    parser.add_argument('--freeze', type=Path, help='ours, native: the frozen record')
    parser.add_argument('--method', choices=NATIVE, help='native: which baseline')
    parser.add_argument('--upstream-root', type=Path)
    parser.add_argument('--weights-dir', type=Path)
    parser.add_argument('--reading', type=Path, action='append', default=[], help='collect: the eight readings')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'create-only output exists', FileExistsError)
    if args.stage == 'collect':
        result = collect(args.reading)
    else:
        require(args.bundle and args.bundle_sha and args.freeze is not None and (args.stage == 'ours' or (
            args.method and args.upstream_root and args.weights_dir)),
            'a timing stage needs --bundle, --bundle-sha, --freeze (and --method, --upstream-root, --weights-dir)')
        bundle = load_bundle(args.bundle, args.bundle_sha)
        rows, _, _ = validation_order(bundle['dev'])
        require(not any(r.get('split') == 'test' for r in rows), 'a test row reached the validation split')
        frozen, settings, env, parameters, times, peak = (ours if args.stage == 'ours' else native)(args, rows)
        result = {'schema': SCHEMA, 'stage': args.stage, 'model': 'rvr' if args.stage == 'ours' else args.method,
                  'test_touched': False, 'bundle_sha256': args.bundle_sha, 'lines': len(rows), 'settings': settings,
                  'environment': env, 'parameters': parameters, 'peak_memory_mib': peak, 'summary': summarize(times),
                  'per_call_ms': times, 'freeze_sha256': digest(args.freeze), 'checkpoint_sha256': frozen['checkpoint_sha256'],
                  'provenance': provenance(to_jsonable(args), {str(args.freeze): digest(args.freeze), str(args.bundle): args.bundle_sha})}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=1, sort_keys=True)+'\n')
    shown = result['rows'] if args.stage == 'collect' else {result['model']: result['summary']}
    print(json.dumps(shown, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
