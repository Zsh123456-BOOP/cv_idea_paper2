import json

import numpy as np

from revrender.data.struck_pack import validation_writer, window
from revrender.eval.struck_reader import scores_of


def test_the_crop_is_the_polygon_box_and_six_pixels_inside_the_page():
    assert window([[10, 20], [50, 20], [50, 40], [10, 40]], (100, 45)) == (4, 14, 56, 45)


def test_about_a_tenth_of_the_writers_validate():
    share = np.mean([validation_writer(f'{k:06x}-w') for k in range(3000)])
    assert .07 < share < .13


def test_each_model_and_condition_is_scored_on_the_lines_it_read(tmp_path):
    (tmp_path/'pack.json').write_text(json.dumps({'rows': [{'id': 'dev|a_q#1', 'line_id': 'a_q#1', 'kind': 'line', 'text': 'abcd'},
                                                           {'id': 'dev|b_q#1', 'line_id': 'b_q#1', 'kind': 'line', 'text': 'xy'}]}))
    (tmp_path/'r').mkdir()
    (tmp_path/'r'/'predictions.jsonl').write_text('\n'.join(json.dumps(x) for x in [
        {'id': 'dev|a_q#1', 'model': 'M', 'condition': 'c', 'text': 'abcd'}, {'id': 'dev|b_q#1', 'model': 'M', 'condition': 'c', 'text': 'x'},
        {'id': 'dev|a_q#1', 'model': 'M', 'condition': 'd', 'text': 'ab'}]))
    shares, exact = scores_of(tmp_path, [tmp_path/'r'])
    assert shares['M|c'] == {'a_q#1': 1., 'b_q#1': .5} and exact['M|c'] == {'a_q#1': 1., 'b_q#1': 0.} and shares['M|d'] == {'a_q#1': .5}
