"""Give the native baselines the same mask-threshold rule as RevRender: 0.3, 0.4 or 0.5, chosen on validation.

RevRender chooses its mask threshold on validation among 0.3 / 0.4 / 0.5 together with NMS and cut; the native
baselines archive binary masks at the threshold their own upstream code applies (0.5).  This re-runs a frozen native
checkpoint on the validation split at each of the three thresholds, the threshold applied where the upstream code
applies its own:

  MaskDINO / DI-MaskDINO   instance_inference binarises ``mask_pred > 0`` (probability 0.5) and scores each mask by its
                           class score times the mean probability inside the binary mask; here the cut is logit(t), and
                           the score follows the upstream formula on that mask
  Mask2Former              the same rule in its own instance_inference (softmax class score)
  PointRend / Mask R-CNN   detector_postprocess pastes the soft ROI masks with ``mask_threshold=0.5``; here with t
  CondInst                 CondInst.postprocess binarises the upsampled global masks with ``mask_threshold=0.5``; here with t
  SOLOv2                   its two upstream mask comparisons are rewritten (checked: exactly two) to t

Each threshold's predictions are archived in the training run's format.  The 0.5 archive must equal the frozen validation
archive mask for mask and score on every line -- the wiring proof -- before anything is read.  The frontier (NMS and cut)
is then read at each threshold and the rule of eval.freeze picks one: best F1, then fewer false positives, then the lower
threshold.

This does NOT train or change a weight, change the upstream model beyond the one threshold, read the test split, or
contact the network.
"""
import argparse
import ast
import functools
import inspect
import json
import math
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from ...data.bundle import load_bundle
from ...eval import matching
from ...eval.freeze import BUDGETS, SCHEMA, budget_table, is_freeze, pick, read_candidates
from ...util import SEED, digest, make_output_dir, provenance, to_jsonable, write_json
from .build import METHODS, build
from .train import archive_prediction, predict_record


class NativeThresholdError(RuntimeError):
    pass


def require_(condition, message):
    if not condition:
        raise NativeThresholdError(message)


def logit_cut(threshold):
    """The logit a probability threshold corresponds to; exactly 0.0 at 0.5, as upstream compares."""
    return math.log(threshold/(1.-threshold))


def maskdino_inference(threshold):
    """MaskDINO's instance_inference (maskdino/maskdino.py) with the binarisation at logit(threshold)."""
    import torch
    from detectron2.structures import Boxes, Instances
    cut = logit_cut(threshold)

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        require_(not self.panoptic_on, 'panoptic MaskDINO is not what the training run builds')
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(
            self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        result = Instances(image_size)
        result.pred_masks = (mask_pred > cut).float()
        result.pred_boxes = Boxes(mask_box_result[topk_indices])
        mask_scores_per_image = ((mask_pred.sigmoid().flatten(1)*result.pred_masks.flatten(1)).sum(1)
                                 / (result.pred_masks.flatten(1).sum(1)+1e-6))
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image*mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
    return instance_inference


def mask2former_inference(threshold):
    """Mask2Former's instance_inference (mask2former/maskformer_model.py) with the binarisation at logit(threshold)."""
    import torch
    from torch.nn import functional as F
    from detectron2.structures import Boxes, Instances
    cut = logit_cut(threshold)

    def instance_inference(self, mask_cls, mask_pred):
        require_(not self.panoptic_on, 'panoptic Mask2Former is not what the training run builds')
        image_size = mask_pred.shape[-2:]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(
            self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        result = Instances(image_size)
        result.pred_masks = (mask_pred > cut).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        mask_scores_per_image = ((mask_pred.sigmoid().flatten(1)*result.pred_masks.flatten(1)).sum(1)
                                 / (result.pred_masks.flatten(1).sum(1)+1e-6))
        result.scores = scores_per_image*mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
    return instance_inference


def solov2_inference(original, threshold):
    """Change only the two upstream mask comparisons; leave NMS arguments intact.

    This does not edit source files, losses, scoring, NMS or the instance head.
    Fail closed if the pinned method no longer contains exactly two comparisons.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))

    class Binarization(ast.NodeTransformer):
        count = 0

        def visit_Compare(self, node):
            for index, right in enumerate(node.comparators):
                if (isinstance(right, ast.Attribute) and right.attr == 'mask_threshold'
                        and isinstance(right.value, ast.Name) and right.value.id == 'self'
                        and isinstance(node.ops[index], ast.Gt)):
                    node.comparators[index] = ast.copy_location(ast.Constant(float(threshold)), right)
                    self.count += 1
            return node

    patch = Binarization()
    tree = patch.visit(tree)
    require_(patch.count == 2, 'SOLOv2 must have exactly two upstream mask binarisations')
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(original), 'exec'), namespace)
    return namespace[original.__name__]


def set_threshold(method, model, threshold, originals):
    """Point the upstream binarisation at `threshold`; `originals` holds what was there first."""
    if method in ('maskdino', 'mask2former', 'dimaskdino'):
        cls = type(model.native_model)
        name = {'maskdino': 'MaskDINO', 'mask2former': 'MaskFormer', 'dimaskdino': 'MaskDINO'}[method]
        require_(cls.__name__ == name and 'instance_inference' in vars(cls), f'not the {name} meta-architecture')
        originals.setdefault(method, vars(cls)['instance_inference'])
        cls.instance_inference = (maskdino_inference if method in ('maskdino', 'dimaskdino') else mask2former_inference)(threshold)
    elif method == 'solov2':
        cls = type(model.native_model)
        require_(cls.__name__ == 'SOLOv2' and 'inference_single_image' in vars(cls), 'not the SOLOv2 meta-architecture')
        originals.setdefault(method, vars(cls)['inference_single_image'])
        cls.inference_single_image = solov2_inference(originals[method], threshold)
    elif method == 'condinst':
        cls = type(model.native_model)
        require_(cls.__name__ == 'CondInst' and 'postprocess' in vars(cls), 'not the CondInst meta-architecture')
        originals.setdefault('condinst', vars(cls)['postprocess'])
        cls.postprocess = functools.partialmethod(originals['condinst'], mask_threshold=threshold)
    elif method in ('pointrend', 'maskrcnn'):
        rcnn = sys.modules['detectron2.modeling.meta_arch.rcnn']
        originals.setdefault(method, rcnn.detector_postprocess)
        rcnn.detector_postprocess = functools.partial(originals[method], mask_threshold=threshold)
    else:
        raise NativeThresholdError(f'unsupported method: {method}')


def archives_equal(first, second):
    """Per line id: masks and scores identical in two archives of the training run's format; returns (equal, total)."""
    import numpy as np

    def index(directory):
        out = {}
        for path in sorted(Path(directory).glob('*.npz')):
            payload = np.load(path, allow_pickle=True)
            out[str(payload['sample_id'])] = (payload['masks_packed'].tobytes(), tuple(payload['shape'].tolist()),
                                              np.asarray(payload['scores'], np.float32).tobytes())
        return out
    a, b = index(first), index(second)
    require_(set(a) == set(b), 'the two archives cover different lines')
    return sum(a[k] == b[k] for k in a), len(a)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--freeze', type=Path, required=True, help="the native run's freeze record (mask threshold 0.5)")
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--transfiner-pin', type=Path, help='Mask Transfiner: the pin record of the authors\' files')
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    try:
        require_(not args.output_dir.exists(), 'create-only output exists')
        frozen = json.loads(args.freeze.read_text())
        require_(is_freeze(frozen.get('schema'), native_only=True) and frozen['model'] == args.method
                 and frozen['test_touched'] is False and frozen['bundle_sha256'] == args.bundle_sha
                 and float(frozen['mask_threshold']) == .5, "not the validation freeze of this native model")
        require_(digest(frozen['checkpoint']) == frozen['checkpoint_sha256'], 'the checkpoint is not the frozen one')
        bundle = load_bundle(args.bundle, args.bundle_sha)
        rows = bundle['dev']
        require_(not any(r.get('split') == 'test' for r in rows), 'a test row reached the validation split')
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = build(args.method, args.upstream_root, args.weights_dir, seed=SEED, transfiner_pin=args.transfiner_pin)
        model.load_state_dict(torch.load(frozen['checkpoint'], map_location='cpu', weights_only=True)['model'])
        model.eval()
        make_output_dir(args.output_dir)
        originals, archives = {}, {}
        for threshold in (.5, .4, .3):          # 0.5 first: nothing is read before the wiring proof
            set_threshold(args.method, model, threshold, originals)
            archive = args.output_dir/f'dev_instances_{threshold:.1f}'
            archive.mkdir()
            with torch.no_grad():
                for index, row in enumerate(rows):
                    archive_prediction(archive, index, row, predict_record(model, row, 'cuda'))
            archives[threshold] = archive
            if threshold == .5:
                equal, total = archives_equal(archive, frozen['validation_archive'])
                require_(equal == total, f'the 0.5 re-run differs from the frozen archive on {total-equal} of {total} lines')
                print(json.dumps({'wiring': 'threshold 0.5 reproduces the frozen archive', 'lines': total}), flush=True)
        dev, order, truths = matching.validation_order(rows)
        readings, frontiers = read_candidates(lambda level: matching.native_provider(archives[level]), order, truths,
                                              candidates=(.3, .4, .5))
        for best in readings:
            print(json.dumps({'threshold': best['mask_threshold'], 'f1': round(best['f1'], 4),
                              'tp_fp': [best['true_positive'], best['false_positive']],
                              'pixel_iou': round(best['pixel_iou'], 4), 'pixel_p': round(best['pixel_precision'], 4)}), flush=True)
        at_half = next(r for r in readings if r['mask_threshold'] == .5)
        require_(abs(at_half['f1']-frozen['validation']['own_best']['f1']) < 1e-12, "the 0.5 reading is not the frozen F1")
        best = pick(readings)
        points = frontiers[best['mask_threshold']]
        result = {'schema': SCHEMA, 'split': 'validation', 'model': args.method, 'candidate': frozen['candidate'], 'test_touched': False,
                  'common_test_sample_ids_sha256': frozen['common_test_sample_ids_sha256'], 'bundle_sha256': args.bundle_sha,
                  'chosen_step': frozen['chosen_step'], 'checkpoint': frozen['checkpoint'],
                  'checkpoint_sha256': frozen['checkpoint_sha256'], 'selection_record_sha256': frozen['selection_record_sha256'],
                  'refrozen_from_sha256': digest(args.freeze),
                  'validation_archive': str(archives[best['mask_threshold']].resolve()), 'mask_threshold': best['mask_threshold'],
                  'mask_threshold_rule': 'chosen on validation among 0.3/0.4/0.5 (user, 2026-09-27); applied at the upstream binarisation',
                  'score_threshold': best['score_threshold'], 'nms': best['nms'],
                  'validation': {'gold_components': sum(map(len, truths)), 'own_best': best,
                                 'by_mask_threshold': {str(r['mask_threshold']): r for r in readings},
                                 'tp_within_fp_budget': budget_table(points, BUDGETS)},
                  'wiring': {'threshold_0.5_equals_frozen_archive': True, 'lines': len(rows)},
                  'provenance': provenance(to_jsonable(args), {str(args.freeze): digest(args.freeze), str(args.bundle): args.bundle_sha}),
                  'frozen_at': datetime.now(timezone.utc).isoformat()}
        write_json(args.output_dir/'freeze.json', result)
    except NativeThresholdError as exc:
        raise SystemExit('threshold refused: '+str(exc))
    print(json.dumps({'method': args.method, 'mask_threshold': best['mask_threshold'], 'f1': round(best['f1'], 4),
                      'tp_fp': [best['true_positive'], best['false_positive']], 'pixel_iou': round(best['pixel_iou'], 4)},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
