"""RevRender (needs torch and segmentation_models_pytorch; skipped where they are missing, run on the server)."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('segmentation_models_pytorch')

from revrender import determinism  # noqa: E402
from revrender.models import losses, ops, quality  # noqa: E402
from revrender.models.revrender import VARIANTS, build_model  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent/'golden/model_parameters.json').read_text())
RESNET50 = os.environ.get('REVRENDER_RESNET50')


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return build_model('control')


def test_the_model_has_the_paper_one_parameters_in_the_paper_one_order(model):
    state = model.state_dict()
    entries = [[k, list(v.shape), str(v.dtype)] for k, v in state.items()]
    assert len(entries) == GOLDEN['entries']
    assert hashlib.sha256(json.dumps(entries).encode()).hexdigest() == GOLDEN['names_shapes_dtypes_sha256']
    assert sum(p.numel() for p in model.parameters()) == GOLDEN['parameters_incl_buffers_excluded'] == 27743113
    assert list(dict(model.named_children())) == ['see', 'find', 'draw']


@pytest.mark.skipif(not RESNET50, reason='set REVRENDER_RESNET50 to the ImageNet R-50 .pkl to compare initial weights')
def test_initial_weights_equal_the_recorded_paper_one_initialisation():
    from revrender.train import state_digest
    import random
    random.seed(42), np.random.seed(42), torch.manual_seed(42)
    built = build_model('control', RESNET50)
    got = {name: state_digest(part) for name, part in built.named_children()}
    assert got == {'see': '36bba2d289143f46f5a65b14a81d25423b0101917f22114977597f0e2597e5fd',
                   'find': '2d5816bb74f97cf72786449187f6f4809a03c0d29322fbea3a12e476432f5cae',
                   'draw': '2546b553bb6415f0acad7af041d0710b800b1f6bee432f0d37830850fdfc0e2e'}


def test_the_variants_change_exactly_what_they_declare():
    torch.manual_seed(0)
    nocontext = build_model('nocontext')
    assert not nocontext.draw.alignment_residual.contextual and not hasattr(nocontext.draw.alignment_residual, 'query')
    assert not any(k.startswith('draw.alignment_residual.query') for k in nocontext.state_dict())
    torch.manual_seed(0)
    nofine = build_model('nofine')
    assert nofine.see.pixel_decoder.blank_image and set(nofine.state_dict()) == set(build_model('control').state_dict())
    torch.manual_seed(0)
    nodense = build_model('nodense')
    assert not nodense.dense_supervision
    with pytest.raises(ValueError):
        build_model('tune_the_backbone')
    assert VARIANTS == ('control', 'nofine', 'nocontext', 'nodense')


def lines():
    image = torch.full((1, 1, 96, 768), 1.)
    image[:, :, 30:40, 100:400] = -1.
    gold = torch.zeros(96, 768)
    gold[28:42, 98:402] = 1.
    return image, [[gold]]


def test_inference_returns_rendered_logits_and_a_delivered_score_per_query(model):
    image, _ = lines()
    model.eval()
    with torch.no_grad():
        rendered, scores, boxes = model(image)
    assert rendered.shape == (1, 32, 96, 768) and scores.shape == (1, 32) and boxes.shape == (1, 32, 4)
    assert torch.isfinite(rendered).all() and torch.isfinite(scores).all()
    assert model.cache['mask_iou'].shape == (1, 32)


def test_the_training_loss_has_every_term_and_backpropagates_to_every_part(model):
    image, targets = lines()
    model.train()
    prediction, objects, _ = model(image, targets)
    loss, pieces = losses.compute_loss(model, prediction, targets)
    assert torch.isfinite(loss)
    assert list(pieces) == ['stage_0', 'stage_1', 'stage_2', 'stage_3', 'stage_4', 'denoising', 'dense', 'roi_point',
                            'mask_iou', 'mask_iou_target_mean']
    loss.backward()
    for name, part in model.named_children():
        assert any(p.grad is not None and bool((p.grad != 0).any()) for p in part.parameters()), name
    assert model.find.mask_iou.score[-1].weight.grad is not None
    floats = losses.materialize(pieces)
    assert all(isinstance(v, float) for v in floats.values())
    model.zero_grad()


def test_nodense_zeroes_the_per_pixel_term_and_nothing_else():
    torch.manual_seed(0)
    full, silent = build_model('control'), build_model('nodense')
    silent.load_state_dict(full.state_dict())
    image, targets = lines()
    out = {}
    for name, m in (('full', full), ('silent', silent)):
        m.train()
        torch.manual_seed(5)
        prediction, _, _ = m(image, targets)
        loss, pieces = losses.compute_loss(m, prediction, targets)
        out[name] = (float(loss.detach()), {k: float(v.detach()) for k, v in pieces.items()})
    assert out['silent'][1]['dense'] == 0. and out['full'][1]['dense'] > 0.
    assert out['full'][0] - out['silent'][0] == pytest.approx(out['full'][1]['dense'], rel=1e-5)


def test_clean_lines_train_with_empty_targets(model):
    image, _ = lines()
    model.train()
    prediction, _, _ = model(image, [[]])
    loss, pieces = losses.compute_loss(model, prediction, [[]])
    assert torch.isfinite(loss) and 'denoising' not in pieces
    model.zero_grad()


def test_more_targets_than_queries_are_refused(model):
    image, _ = lines()
    model.train()
    prediction, _, _ = model(image, [[]])
    with pytest.raises(ValueError, match='query budget'):
        losses.compute_loss(model, prediction, [[torch.zeros(96, 768)]*33])


def test_the_clipper_clips_the_quality_head_on_its_own(model):
    clipper = losses.Clipper(model, ['find.mask_iou'], 5.)
    assert len(clipper.groups) == 2
    names = {id(p): n for n, p in model.named_parameters()}
    assert all(names[id(p)].startswith('find.mask_iou') for p in clipper.groups[0])
    assert not any(names[id(p)].startswith('find.mask_iou') for p in clipper.groups[1])
    with pytest.raises(ValueError):
        losses.Clipper(model, ['no.such.prefix'])


def test_delivered_score_is_the_geometric_mean_of_objectness_and_quality():
    objectness, predicted = torch.tensor([2., -1., 0.5]), torch.tensor([1., 3., -2.])
    got = quality.delivered_score(objectness, predicted).sigmoid()
    expected = (objectness.sigmoid()*predicted.sigmoid()).sqrt()
    assert torch.allclose(got, expected, atol=1e-5)
    with pytest.raises(ValueError):
        quality.delivered_score(objectness, predicted, 'sum')


def test_overlap_targets_are_the_best_iou_against_the_gold():
    gold = torch.zeros(1, 8, 8)
    gold[0, 2:6, 2:6] = 1
    logits = torch.full((1, 3, 8, 8), -10.)
    logits[0, 0, 2:6, 2:6] = 10.                         # perfect
    logits[0, 1, 2:6, 2:4] = 10.                         # half
    out = quality.overlap_targets(logits, [[gold[0]]])
    assert out[0].tolist() == pytest.approx([1., .5, 0.])
    assert quality.overlap_targets(logits, [[]]).sum() == 0


def test_learned_sampling_matches_native_bilinear_inside_the_image():
    torch.manual_seed(0)
    features = torch.randn(2, 3, 9, 13)
    coordinates = torch.rand(2, 50, 2)*.8+.1
    native = torch.nn.functional.grid_sample(features, coordinates[:, None]*2-1, mode='bilinear', padding_mode='zeros',
                                             align_corners=False)[:, :, 0]
    assert torch.allclose(ops.sample_learned(features, coordinates), native, atol=1e-5)
    coordinates.requires_grad_()
    ops.sample_learned(features, coordinates).sum().backward()
    assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()


def test_boxes_and_rois():
    mask = torch.zeros(2, 10, 20, dtype=torch.bool)
    mask[0, 2:6, 4:12] = True
    boxes = ops.mask_boxes(mask)
    assert boxes[0].tolist() == pytest.approx([.4, .4, .4, .4]) and boxes[1].tolist() == [.5, .5, 1., 1.]    # empty -> whole image
    roi = ops.expanded_roi(torch.tensor([[.5, .5, .2, .2]]), (96, 768))
    assert roi[0].tolist() == pytest.approx([.35, .35, .65, .65])
    assert ops.giou(ops.xyxy(boxes[:1]), ops.xyxy(boxes[:1])).item() == pytest.approx(1.)


class Deterministic:
    """The deterministic operators return the native forward values and the native gradients."""

    @staticmethod
    def setup_method():
        torch.use_deterministic_algorithms(True)          # the adjoints refuse to run otherwise
        if not determinism.installed():
            determinism.install()

    @staticmethod
    def teardown_method():
        determinism.uninstall()
        torch.use_deterministic_algorithms(False)


class TestDeterministicOperators(Deterministic):
    def test_resize_gradient_equals_the_native_one(self):
        x = torch.randn(1, 2, 6, 10, requires_grad=True)
        native = determinism.NATIVE_RESIZE(x, size=(12, 20), mode='bilinear', align_corners=False)
        mine = torch.nn.functional.interpolate(x, size=(12, 20), mode='bilinear', align_corners=False)
        assert torch.equal(native, mine)
        weight = torch.randn_like(native)
        g_native, = torch.autograd.grad((native*weight).sum(), x)
        g_mine, = torch.autograd.grad((mine*weight).sum(), x)
        assert torch.allclose(g_native, g_mine, atol=1e-5)

    def test_pool_and_grid_gradients_equal_the_native_ones(self):
        x = torch.randn(1, 2, 8, 12, requires_grad=True)
        weight = torch.randn(1, 2, 4, 3)
        g_mine, = torch.autograd.grad((torch.nn.functional.adaptive_avg_pool2d(x, (4, 3))*weight).sum(), x)
        g_native, = torch.autograd.grad((determinism.NATIVE_POOL(x, (4, 3))*weight).sum(), x)
        assert torch.allclose(g_mine, g_native, atol=1e-6)
        coordinates = torch.rand(1, 20, 2)
        for mode in ('bilinear', 'nearest'):
            mine = ops.sample_points(x, coordinates, mode)
            native = determinism.NATIVE_GRID(x, coordinates[:, None]*2-1, mode=mode, padding_mode='border', align_corners=False)[:, :, 0]
            assert torch.equal(mine, native)
            w = torch.randn_like(mine)
            gm, = torch.autograd.grad((mine*w).sum(), x)
            gn, = torch.autograd.grad((native*w).sum(), x)
            assert torch.allclose(gm, gn, atol=1e-5)
        with pytest.raises(ValueError):
            ops.sample_points(x, coordinates.requires_grad_(), 'bilinear')

    def test_install_twice_is_refused(self):
        with pytest.raises(RuntimeError, match='already installed'):
            determinism.install()


def test_predict_lines_delivers_instances_per_strip(model):
    from revrender import infer
    strips = [np.full((96, 768), 250, np.uint8), np.full((96, 768), 200, np.uint8)]
    out = infer.predict_lines(model, strips, {'mask_threshold': .4, 'nms': .5, 'score_threshold': 0.})
    assert len(out) == 2
    for instances in out:
        assert all(mask.shape == (96, 768) and mask.dtype == bool and 0. <= score <= 1. for mask, score in instances)
        assert [s for _, s in instances] == sorted((s for _, s in instances), reverse=True)
