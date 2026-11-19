from revrender.eval import erase_gap


def test_thirds_split_by_value_then_id():
    values = [0, 0, 0, 1, 2, 3]; ids = ['f', 'e', 'd', 'c', 'b', 'a']
    assert erase_gap.thirds(values, ids) == ['middle', 'low', 'low', 'middle', 'high', 'high']


def test_kind_of_a_wrong_reading():
    row = lambda reading, s, d: {'E-NAFNet': {'reading': reading, 's': s, 'd': d, 'i': 0}}
    assert erase_gap.kind_of(row('[读不出]', 0, 3)) == 'refused'
    assert erase_gap.kind_of(row('ab', 0, 1)) == 'missing' and erase_gap.kind_of(row('ax', 1, 1)) == 'wrong'


def test_verdict_needs_both_the_interval_and_two_points_of_ar():
    a = lambda lo, hi, ar: {'exact': {'interval': [lo, hi]}, 'ar': {'difference': ar}}
    assert erase_gap.verdict(a(.01, .1, .03)) == 'clearly_better' and erase_gap.verdict(a(-.1, -.01, -.03)) == 'clearly_worse'
    assert erase_gap.verdict(a(.01, .1, .01)) == 'cannot_tell' and erase_gap.verdict(a(-.01, .1, .05)) == 'cannot_tell'


def test_finite_turns_nan_into_none():
    assert erase_gap.finite({'a': float('nan'), 'b': [1., float('nan')], 'c': 'x'}) == {'a': None, 'b': [1., None], 'c': 'x'}
