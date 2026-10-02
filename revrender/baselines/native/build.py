"""Build one native baseline the way every stage (training, threshold re-run, timing) needs it.

``method`` is one of the seven native models; ``transfiner_pin`` swaps Mask R-CNN's mask pooler and head for Mask
Transfiner's (the pin file lists the SHA-256 of the authors' files); ``dense`` adds the per-pixel head of dense.py.
Use one method per Python process: the official packages register overlapping backbone names.
"""
import json
from pathlib import Path

METHODS = ('maskdino', 'pointrend', 'mask2former', 'condinst', 'maskrcnn', 'solov2', 'dimaskdino')
UPSTREAM_V2_METHODS = ('solov2', 'dimaskdino')


def weights_path(method, weights_dir):
    from . import factory
    kind = factory.get_pretraining_kind(method)
    return Path(weights_dir)/f'R-50-{"MSRA" if kind == "msra" else kind}.pkl', kind


def build(method, upstream_root, weights_dir, seed=42, smoke=False, transfiner_pin=None, dense=False, device='cuda'):
    from . import factory
    weight, _ = weights_path(method, weights_dir)
    model = factory.build_model(method, upstream_root, weight, device=device, seed=seed, smoke=smoke)
    if transfiner_pin is not None:
        from . import transfiner
        if method != 'maskrcnn':
            raise ValueError('Transfiner builds on the Mask R-CNN')
        model = transfiner.install(model, expected=json.loads(Path(transfiner_pin).read_text()))
    if dense:
        from . import dense as dense_head
        model = dense_head.attach_dense(model, method)
    return model
