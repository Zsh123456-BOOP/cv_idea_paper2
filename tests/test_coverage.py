"""How much of the struck, rewritten and valid text of a labelled line a reading holds."""
from revrender.eval import coverage as C


def test_a_line_is_split_into_what_the_strike_removed_what_the_rewrite_added_and_what_stayed():
    struck, rewritten, valid = C.split('x=3,y=5', 'x=7,y=5')
    assert struck == {2} and rewritten == {2} and valid == {0, 1, 3, 4, 5, 6}
    struck, rewritten, valid = C.split('abc', '')
    assert struck == {0, 1, 2} and rewritten == set() and valid == set()


def test_reading_the_final_text_the_struck_text_or_nothing_scores_what_the_labels_allow():
    assert C.coverage('-x-1=2', '-x-1=0', '-x-1=2') == {'leak': 0., 'rewrite': 1., 'valid': 1.}
    assert C.coverage('-x-1=0', '-x-1=0', '-x-1=2') == {'leak': 1., 'rewrite': 0., 'valid': 1.}
    assert C.coverage('', '-x-1=0', '-x-1=2') == {'leak': 0., 'rewrite': 0., 'valid': 0.}
    assert C.coverage('黄瓜和茄子', '黄瓜和茄子', '') == {'leak': 1., 'rewrite': None, 'valid': None}      # the whole line struck, nothing after
    assert C.coverage('a=2', 'a=2', 'a=20') == {'leak': None, 'rewrite': 0., 'valid': 1.}                 # nothing was struck, a character was added


def test_a_share_counts_characters_held_in_order_and_ignores_spelling_and_the_annotators_ocr_prefix():
    assert C.coverage('x = - 3', 'x=3', 'x=-3')['rewrite'] == 1.
    assert C.coverage('3=x', 'x=3', 'x=-3')['valid'] < 1.                                                   # the same characters in another order are not held
    assert C.coverage('∴ ∠ABC = 60°', '\\therefore \\angle ABC = 60^\\circ', 'OCR：\\therefore \\angle ABC = 60^\\circ x')['valid'] == 1.
    assert C.prepare('OCR：y = 2') == C.prepare('y=2') and C.prepare(None) == ''


def test_the_alignment_keeps_equal_characters_in_order():
    assert C.alignment('abc', 'axc') == ({0, 2}, {0, 2})
    assert C.alignment('', 'abc') == (set(), set()) and C.alignment('ab', 'ab') == ({0, 1}, {0, 1})
    assert C.similarity('x = 3', 'x=3') == 1. and C.similarity('x=3', 'q') < .5


def test_a_stutter_in_the_labels_is_held_under_one_alignment_and_not_the_other():
    got = C.counts('b=85', 'b=8585', 'b=85')                       # a reading of the final text: which copy of "85" was struck?
    assert got['leak'] == {'total': 2, 'held': 2, 'strict': 0, 'lenient': 2} and got['valid']['strict'] == got['valid']['total'] == 4
    got = C.counts('b=8585', 'b=8585', 'b=85')
    assert got['leak']['strict'] == 2 and 'rewrite' not in got
