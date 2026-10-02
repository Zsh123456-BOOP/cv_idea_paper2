"""Mask Transfiner (Ke et al., CVPR 2022) as a native baseline: its official RoI pooler and mask head on our Mask R-CNN.

The authors release Transfiner as a whole fork of an older detectron2 (SysCV/transfiner, commit 5b61fb5,
Apache-2.0).  Against the detectron2 every other baseline here runs on, the fork changes two files that carry the
method -- modeling/poolers.py (RoI features pooled at 14, 28, 56 and 112 px with a semantic branch on P2) and
modeling/roi_heads/mask_head.py (incoherent-region detection, the point quadtree and the three-layer refinement
transformer, with their losses) -- plus layers/boundary.py, which the mask head uses for its boundary loss.  This
module builds the factory's Mask R-CNN (ImageNet R-50, as every baseline) and replaces its mask pooler and mask head with the
authors' classes, loaded unmodified from the pinned files (SHA-256 checked); roi_heads.py is the same in both.  The
fork's two kornia calls sit behind vis_period == 100, which stays 0: kornia (not in our environment) is a stub that
refuses if ever called.  The bundled pytorch_toolbelt supplies its boundary loss.  The fork's data mapper adds
gt_masks_bit (each instance's full-image BitMasks) beside gt_masks; ours already are BitMasks, so they fill it.
Its mask loss assumes every image holds objects (COCO training drops empty images); our protocol keeps clean lines in
every batch, so the loss is given the lines of each micro-batch that hold revisions (on_lines_with_revisions).

Training and the mask-threshold freeze run through the common native protocol with ``--transfiner-pin`` (train.py,
threshold.py); the run is recorded as maskrcnn + Transfiner.  The pin file records the SHA-256 of the authors' three
files (``write_pin`` creates it from the pinned checkout).

This module deliberately does NOT: change the authors' pooler or mask head, the common losses, optimiser, schedule or
augmentation; download anything; read the test split; or contact the network.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

SOURCE = Path('/home/zsh/hme/work/native_baselines_sources_transfiner_v1/transfiner')
COMMIT = '5b61fb53d8df5484f44c8b7d8415f398fd283ddc'
FILES = {'detectron2/modeling/poolers.py': None, 'detectron2/modeling/roi_heads/mask_head.py': None,
         'detectron2/layers/boundary.py': None}


class TransfinerError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise TransfinerError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_file(path, name, package_path=None):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=package_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def kornia_stub():
    """kornia.morphology.dilation and kornia.filters.blur_pool2d, which the fork calls only when vis_period == 100."""
    def refuse(*args, **kwargs):
        raise TransfinerError('a kornia visualisation path was reached; vis_period must stay 0')
    root, morphology, filters = (types.ModuleType(n) for n in ('kornia', 'kornia.morphology', 'kornia.filters'))
    morphology.dilation, filters.blur_pool2d = refuse, refuse
    root.morphology, root.filters = morphology, filters
    sys.modules.update({'kornia': root, 'kornia.morphology': morphology, 'kornia.filters': filters})


def authors_modules(source=SOURCE, expected=None):
    """The fork's pooler, mask head and boundary helper, loaded as they are, against our detectron2."""
    import detectron2.layers
    hashes = {rel: digest(source/rel) for rel in FILES}
    if expected is not None:
        require(hashes == expected, 'the pinned Transfiner files changed')
    kornia_stub()
    if 'pytorch_toolbelt' not in sys.modules:
        load_file(source/'pytorch_toolbelt/__init__.py', 'pytorch_toolbelt', [str(source/'pytorch_toolbelt')])
    boundary = load_file(source/'detectron2/layers/boundary.py', 'transfiner_boundary')
    detectron2.layers.get_instances_contour_interior = boundary.get_instances_contour_interior
    poolers = load_file(source/'detectron2/modeling/poolers.py', 'transfiner_poolers')
    heads = load_file(source/'detectron2/modeling/roi_heads/mask_head.py', 'transfiner_mask_head')
    return poolers, heads, hashes


def on_lines_with_revisions(original):
    """The authors' mask loss over the lines of the micro-batch that hold revisions.

    It walks the per-image instances, skips an image with none, and yet pairs its semantic branch (one map per image)
    with the targets of the others; COCO training drops empty images, our protocol keeps clean lines in every batch.
    So it is given exactly the lines it would see without the clean ones; with none, its five losses are an exact zero
    that keeps the graph (the detector's own losses still train on the clean lines).
    """
    def loss(pred, uncertain, boundary, x_hr, x_hr_l, x_hr_ll, x_c, x_p2_s, encoder, instances, vis_period=0):
        keep = [i for i, inst in enumerate(instances) if len(inst)]
        if not keep:
            zero = pred.sum()*0+uncertain.sum()*0+boundary.sum()*0+x_p2_s.sum()*0
            return zero, zero, zero, zero, zero
        return original(pred, uncertain, boundary, x_hr, x_hr_l, x_hr_ll, x_c, x_p2_s[keep], encoder,
                        [instances[i] for i in keep], vis_period)
    loss._clean_lines = True
    return loss


def install(model, source=SOURCE, expected=None):
    """The factory's Mask R-CNN with the authors' mask pooler and mask head in place of its own."""
    from detectron2.layers import ShapeSpec
    poolers, heads, hashes = authors_modules(source, expected)
    if not getattr(heads.mask_rcnn_loss, '_clean_lines', False):
        heads.mask_rcnn_loss = on_lines_with_revisions(heads.mask_rcnn_loss)
    native, cfg = model.native_model, model.native_cfg
    require(cfg.VIS_PERIOD == 0 and cfg.MODEL.ROI_MASK_HEAD.NAME == 'MaskRCNNConvUpsampleHead'
            and cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE == 'ROIAlignV2', 'not the Mask R-CNN configuration Transfiner builds on')
    roi = native.roi_heads
    in_features = list(roi.mask_in_features)
    shapes = native.backbone.output_shape()
    resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
    device = next(native.parameters()).device
    roi.mask_pooler = poolers.ROIPooler(output_size=resolution, scales=tuple(1./shapes[k].stride for k in in_features),
                                        sampling_ratio=cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO,
                                        pooler_type='ROIAlignV2').to(device)
    channels = {shapes[k].channels for k in in_features}
    require(len(channels) == 1, 'the mask features differ in width')
    roi.mask_head = heads.build_mask_head(cfg, ShapeSpec(channels=channels.pop(), width=resolution, height=resolution)).to(device)
    require(type(roi.mask_head).__module__ == 'transfiner_mask_head', "the mask head is not the authors'")
    forward = model.forward

    def with_bit_masks(batch):
        # the fork's data mapper gives each instance gt_masks_bit (full-image BitMasks) next to gt_masks; ours are those
        if model.training:
            for row in batch:
                if not row['instances'].has('gt_masks_bit'):
                    row['instances'].gt_masks_bit = row['instances'].gt_masks
        return forward(batch)
    model.forward = with_bit_masks
    model.baseline_metadata = dict(getattr(model, 'baseline_metadata', {}), transfiner={
        'repository': 'SysCV/transfiner', 'commit': COMMIT, 'files_sha256': hashes, 'license': 'Apache-2.0',
        'replaced': ['roi_heads.mask_pooler', 'roi_heads.mask_head'], 'kornia': 'stub (vis_period 0)',
        'data_contract': 'gt_masks_bit = the full-image BitMasks, as the fork\'s detection_utils sets it',
        'clean_lines': "the authors' mask loss reads the micro-batch's lines with revisions; none -> exact zero"})
    return model


def pinned(source=SOURCE):
    return {rel: digest(Path(source)/rel) for rel in FILES}


def write_pin(path, source=SOURCE):
    """The SHA-256 record of the authors' files; refuses a checkout that is not the pinned commit."""
    path = Path(path)
    require(not path.exists(), 'create-only output exists')
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == COMMIT, f'the checkout is {head}, not the pinned commit')
    path.write_text(json.dumps(pinned(source), indent=1, sort_keys=True)+'\n')
    return path
