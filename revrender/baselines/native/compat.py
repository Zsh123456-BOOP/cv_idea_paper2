"""Audited, idempotent build compatibility patches to pinned upstream sources.

No architecture substitutions. CondInst imports only its dependencies; the
unrelated BezierAlign/DefROIAlign operators are not imported or compiled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PINS = {
    'detectron2': 'a2f4a8771ab77e8411c26b27f24f9489a28a2453',
    'Mask2Former': '9b0651c6c1d5b3af2e6da0589b719c514ec0d69a',
    'MaskDINO': '3831d8514a3728535ace8d4ecc7d28044c42dd14',
    'AdelaiDet': '5e19cb172b8363820b409ed1a2754fb19ad3acb8',
}


def apply(root):
    changes = []
    for name, revision in PINS.items():
        repo = root / name
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise ValueError(f'{name}: wrong upstream revision {actual}')
        edits = {}
        if name == 'detectron2':
            # The upstream PointRend sampler concatenates an empty label list
            # for an all-clean batch. Its point loss has no targets in this
            # case: return a differentiable zero; retain RPN/box negative loss.
            rel = 'projects/PointRend/point_rend/mask_head.py'
            original = subprocess.check_output(['git', '-C', str(repo), 'show', f'HEAD:{rel}'])
            before = b'            point_coords, point_labels = self._sample_train_points(coarse_mask, instances)'
            after = (b'            if sum(len(item) for item in instances) == 0:\n'
                     b'                losses["loss_mask_point"] = coarse_mask.sum() * 0.0\n'
                     b'                return losses\n\n' + before)
            if original.count(before) != 1:
                raise ValueError('upstream PointRend empty-target patch anchor changed')
            edits[rel] = (original, original.replace(before, after))
        elif name in ('Mask2Former', 'MaskDINO'):
            package = 'mask2former' if name == 'Mask2Former' else 'maskdino'
            base = f'{package}/modeling/pixel_decoder/ops/src/cuda/'
            for file in ('ms_deform_attn_cuda.cu', 'ms_deform_im2col_cuda.cuh'):
                rel = base + file
                original = subprocess.check_output(['git', '-C', str(repo), 'show', f'HEAD:{rel}'])
                patched = original.replace(b'value.type(),', b'value.scalar_type(),')
                patched = patched.replace(b'#include <THC/THCAtomics.cuh>', b'#include <ATen/cuda/Atomic.cuh>')
                edits[rel] = (original, patched)
        elif name == 'AdelaiDet':
            contents = {
                'adet/modeling/__init__.py': 'from .fcos import FCOS\nfrom .backbone import build_fcos_resnet_fpn_backbone\nfrom .condinst import CondInst\n',
                'adet/modeling/backbone/__init__.py': 'from .fpn import build_fcos_resnet_fpn_backbone\n',
                'adet/layers/__init__.py': 'from .deform_conv import DFConv2d\nfrom .ml_nms import ml_nms\nfrom .iou_loss import IOULoss\nfrom .conv_with_kaiming_uniform import conv_with_kaiming_uniform\nfrom .naive_group_norm import NaiveGroupNorm\n',
            }
            for rel, contents_text in contents.items():
                original = subprocess.check_output(['git', '-C', str(repo), 'show', f'HEAD:{rel}'])
                edits[rel] = (original, contents_text.encode())
        for rel, (original, patched) in edits.items():
            path = repo / rel
            if path.read_bytes() not in (original, patched):
                raise ValueError(f'refusing to overwrite unrelated upstream edit: {path}')
            path.write_bytes(patched)
            changes.append({'repository': name, 'path': rel,
                            'original_sha256': hashlib.sha256(original).hexdigest(),
                            'patched_sha256': hashlib.sha256(patched).hexdigest()})
    (root / 'compatibility_manifest.json').write_text(json.dumps({'pins': PINS, 'changes': changes}, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', type=Path, required=True)
    apply(parser.parse_args(argv).upstream_root.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
