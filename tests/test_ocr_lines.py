"""Cutting a page into lines the way the catalogue's lines were cut from the OCR model's text.

``tests/golden/ocr_lines.json`` holds what the paper-1 parser returned for hand-made OCR texts."""
import json
from pathlib import Path

from revrender.data import ocr_lines as O

GOLDEN = json.loads((Path(__file__).parent/'golden'/'ocr_lines.json').read_text())


def test_the_ported_parser_reproduces_the_paper_one_cut_of_every_golden_text():
    for case in GOLDEN:
        lines = O.reorder(O.parse_regions(case['text']))
        boxed = [x for x in case['all_lines'] if x[1] is not None]        # the old parser also kept boxless lines of unlabelled text
        assert [[l['ocr_type'], l['box'], l['inferred']] for l in lines] == boxed
        assert [l['box'] for l in O.strict_lines(case['text'])] == case['strict_boxes']


def test_a_region_with_several_lines_shares_its_box_in_equal_heights_and_is_not_strict():
    text = '<|det|>equation [100, 100, 500, 400]<|/det|>\\[\na = 1 \\\\\nb = 2 \\\\\nc = 3\n\\]\n<|det|>text [110, 420, 520, 500]<|/det|>so x = 2'
    lines = O.parse_regions(text)
    assert [l['inferred'] for l in lines] == [True, True, True, False]
    assert [l['box'][1] for l in lines[:3]] == [100., 200., 300.] and lines[2]['box'][3] == 400.
    assert [l['box'] for l in O.strict_lines(text)] == [[110.0, 420.0, 520.0, 500.0]]


def test_printed_furniture_degenerate_boxes_and_repeated_boxes_are_not_lines():
    text = ('<|det|>page_number [41, 956, 64, 999]<|/det|>20\n<|det|>image [600, 100, 900, 400]<|/det|>fig 1\n'
            '<|det|>text [100, 300, 400, 300]<|/det|>flat line x=1\n<|det|>text [100, 100, 400, 160]<|/det|>a=b\n'
            '<|det|>equation [100, 100, 400, 160]<|/det|>\\( a = b \\)')
    assert [l['ocr_text'] for l in O.strict_lines(text)] == ['a=b']
    assert O.strict_lines('') == [] and O.strict_lines('text with no regions x=1') == []


def test_reading_order_puts_the_second_column_after_the_first_and_bands_left_to_right():
    text = ('<|det|>text [560, 80, 900, 160]<|/det|>r1 x=1\n<|det|>text [60, 200, 400, 280]<|/det|>l2 y=2\n'
            '<|det|>text [60, 80, 400, 160]<|/det|>l1 z=3\n<|det|>text [560, 200, 900, 280]<|/det|>r2 w=4')
    assert [l['ocr_text'] for l in O.strict_lines(text)] == ['l1 z=3', 'l2 y=2', 'r1 x=1', 'r2 w=4']
    band = '<|det|>text [300, 100, 500, 140]<|/det|>second m=2\n<|det|>text [50, 105, 250, 145]<|/det|>first n=1'
    assert [l['ocr_text'] for l in O.strict_lines(band)] == ['first n=1', 'second m=2']


def test_the_normaliser_drops_markup_so_only_real_symbols_count():
    assert O.normalise(r'\( \angle A = {36}^{\circ} \)') == '∠A=36°'
    assert O.normalise(r'\frac {1}{2}') == '(1)/(2)' and O.normalise('') == ''
    assert O.split_region('x\n\\\\\n=') == []               # single symbols and bare separators are not lines


def test_the_answer_length_is_read_from_the_first_non_empty_transcription_of_a_page(tmp_path):
    (tmp_path/'a.jsonl').write_text('\n'.join(json.dumps(r) for r in (
        {'image_id': 'p1', 'text': ''}, {'image_id': 'p2', 'text': 'x = 3'}, {'image_id': 'p3', 'text': None})) + '\n')
    (tmp_path/'b.jsonl').write_text('\n'.join(json.dumps(r) for r in (
        {'image_id': 'p1', 'text': r'\( \angle A = 36^{\circ} \)'}, {'image_id': 'p2', 'text': 'later entry'})) + '\n')
    texts = O.answer_texts([tmp_path/'a.jsonl', tmp_path/'b.jsonl'])
    assert texts == {'p1': r'\( \angle A = 36^{\circ} \)', 'p2': 'x = 3', 'p3': ''}
    assert [O.answer_chars(texts[k]) for k in ('p1', 'p2', 'p3')] == [6, 3, 0]


def test_regions_list_every_boxed_region_and_a_split_line_knows_its_region():
    text = ('<|det|>image [600, 100, 900, 400]<|/det|>\n<|det|>equation [100, 100, 500, 400]<|/det|>\\[\na = 1 \\\\\nb = 2\n\\]\n'
            '<|det|>text [100, 300, 400, 300]<|/det|>flat x=1')
    assert [t for t, _ in O.regions(text)] == ['image', 'equation']                    # the flat region has no area
    lines = [l for l in O.parse_regions(text) if l['inferred']]
    assert [l['parts'] for l in lines] == [2, 2] and all(l['region'] == [100., 100., 500., 400.] for l in lines)


def test_braces_arrays_and_the_spellings_of_congruence_and_similarity_normalise_alike():
    braced = O.normalise(r'$\left\{ \begin{array}{l} AB = AC \\ \angle B = \angle C \end{array} \right.$')
    assert braced == O.normalise(r'$\begin{cases}AB=AC\\ \angle B=\angle C\end{cases}$') == 'AB=AC∠B=∠C'
    assert O.normalise('△ABC≌△DEF') == O.normalise(r'$\triangle ABC\cong\triangle DEF$') == O.normalise(r'$\triangle ABC\text{≌}\triangle DEF$')
    assert O.normalise('△A∽△B') == O.normalise(r'$\triangle A\backsim\triangle B$') == O.normalise(r'$\triangle A\sim\triangle B$')
    assert O.normalise('①') == '1'
