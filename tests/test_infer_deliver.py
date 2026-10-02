import numpy as np

from revrender import infer


def test_delivery_applies_mask_threshold_nms_score_cut_and_drops_empty_masks():
    p = np.zeros((4, 8, 16))
    p[0, 2:5, 2:10] = .9
    p[1, 2:5, 3:11] = .45          # overlaps query 0; at mask .4 it is a duplicate, at .5 it is empty
    p[2, 6:8, 12:15] = .8
    p[3, 0:2, 0:3] = .8            # low score
    scores = np.array([.9, .8, .7, .2])
    point = {'mask_threshold': .4, 'nms': .5, 'score_threshold': .5}
    out = infer.deliver(p, scores, point)
    assert [round(s, 2) for _, s in out] == [.9, .7]                   # 1 suppressed, 3 below the cut
    assert out[0][0].sum() == 24 and out[1][0].sum() == 6
    out = infer.deliver(p, scores, {**point, 'mask_threshold': .5})
    assert [round(s, 2) for _, s in out] == [.9, .7]                   # query 1 is empty at .5 and dropped
    assert len(infer.deliver(p, scores, {**point, 'nms': 1., 'mask_threshold': .4})) == 3   # NMS off keeps the duplicate


def test_the_packed_archive_format_round_trips(tmp_path):
    rows = [{'sample_id': 'a:b'}, {'sample_id': 'a:c'}]
    bits = np.zeros((7, 2, 32, 9216), np.uint8)
    scores = np.zeros((2, 32), np.float32)
    infer.save_predictions(tmp_path/'p.npz', bits, scores, rows)
    with np.load(tmp_path/'p.npz') as a:
        assert a['mask_shape'].tolist() == [2, 32, 96, 768] and a['sample_ids'].tolist() == ['a:b', 'a:c']
        assert np.allclose(a['mask_thresholds'], [.2, .3, .4, .5, .6, .7, .8])


def test_a_checkpoint_names_its_variant_in_either_profile_format():
    assert infer.variant_of({'variant': 'nofine'}) == 'nofine'
    assert infer.variant_of({'configuration': {'tuning880': {'mode': 'nocontext'}}}) == 'nocontext'
    assert infer.variant_of({}) == 'control' and infer.variant_of({'variant': 'nofine'}, 'nodense') == 'nodense'
