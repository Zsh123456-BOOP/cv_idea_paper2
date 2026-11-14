"""Selective scores and the router of readings."""
import numpy as np

from revrender import router
from revrender.eval import selective
from revrender.recognize_lines import UNREADABLE


def test_risk_falls_with_coverage_when_confidence_is_right():
    confidence, loss = [.9, .8, .3, .2], [0, 0, 1, 1]
    coverage, risk = selective.risk_coverage(confidence, loss)
    assert coverage.tolist() == [.25, .5, .75, 1.] and risk.tolist() == [0, 0, 1/3, .5]
    assert selective.risk_at(confidence, loss, .5) == 0 and selective.risk_at(confidence, loss, 1.) == .5
    assert selective.aurc(confidence, loss) < selective.aurc(confidence[::-1], loss)
    assert selective.auroc([.9, .8, .1], [True, True, False]) == 1. and selective.auroc([.5, .5], [True, False]) == .5
    assert selective.auroc([1, 2], [True, True]) is None


def test_a_reading_is_right_as_the_label_or_as_a_refusal():
    assert router.right({'text': 'x=5'}, 'x = 5') and not router.right({'text': 'x=5'}, 'x=3')
    assert router.right({'text': '', 'readable': False}, UNREADABLE) and not router.right({'text': '', 'readable': False}, 'x')
    assert not router.right({'text': 'x=5'}, UNREADABLE)


def test_critics_learn_which_source_to_believe():
    rng = np.random.default_rng(42); rows, a, b = {}, {}, {}
    for i in range(200):                                              # source a is right when it is sure, b when a is not
        k = f'w{i % 20}_{i}'; sure = rng.random() < .5; rows[k] = {'text': 'abc', 'fold': i % 5, 'readable': True}
        a[k] = {'text': 'abc' if sure else 'xyz', 'confidence': .9 if sure else .3}
        b[k] = {'text': 'xyz' if sure else 'abc', 'confidence': .5}
    inks = {k: (.4, .1) for k in rows}
    chosen, trusted = router.route(rows, {'a': a, 'b': b}, inks)
    picked_right = np.mean([router.right(rows[k], {'a': a, 'b': b}[chosen[k][0]][k]['text']) for k in rows])
    assert picked_right > .95                                         # either alone is right half the time
    assert len(router.features('a', {'a': a['w0_0'], 'b': b['w0_0']}, rows['w0_0'], (.4, .1))) == len(router.FEATURES)


def test_views_with_the_evidence_taken_apart():
    from revrender.data.struck_pack import intervened
    scan = np.array([[255, 40, 40, 40, 255]], np.uint8); strike = np.array([[False, True, True, False, False]]); share = np.array([[0, 255, 255, 0, 0]], np.uint8)
    strike_off, writing_off, left = intervened(scan, strike, share)
    assert strike_off.tolist() == [[255, 255, 255, 40, 255]]          # the struck writing outside the strike stays
    assert writing_off.tolist() == [[255, 40, 40, 255, 255]]          # only the strike is left
    assert left.tolist() == [[255, 255, 255, 255, 255]]               # and the separator takes that away too
