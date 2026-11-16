import numpy as np

from revrender.data.pixel_masks import region_pixels
from revrender.tools.annotate.tight_region import bounds, tight_polygon


def test_the_polygon_hugs_the_struck_word_and_leaves_the_neighbours_out():
    grey = np.full((80, 200), 255, np.uint8)
    grey[30:45, 40:60] = 0; grey[30:45, 70:90] = 0          # two struck characters
    grey[36:38, 35:95] = 0                                  # the strike through them
    grey[30:45, 130:150] = 0                                # a neighbour the strike does not touch
    grey[60:75, 40:90] = 0                                  # the line below
    strike = np.zeros(grey.shape, bool); strike[36:38, 35:95] = True
    box = [20, 20, 170, 78]                                 # a detection reaching both neighbours
    polygon = tight_polygon(grey, box, strike)
    l, t, r, b = bounds(polygon); inside = np.zeros(grey.shape, bool); inside[t:b, l:r] = region_pixels(polygon, [l, t, r, b])
    ink = grey < 200
    assert inside[30:45, 40:60].all() and inside[30:45, 70:90].all() and inside[36:38, 35:95].all()
    assert not (inside & ink)[:, 120:].any() and not (inside & ink)[55:].any()


def test_no_strike_keeps_the_box():
    grey = np.full((40, 40), 255, np.uint8)
    assert tight_polygon(grey, [5, 5, 30, 30], np.zeros(grey.shape, bool)) == [[5, 5], [30, 5], [30, 30], [5, 30]]


def test_the_library_keeps_confirmed_records_with_the_line_page_corrections_and_refills_moved_boxes(tmp_path):
    import json
    from revrender.tools.annotate.pixel_library import hint, manual_records
    rec = lambda i, **k: {'id': f'p_1_1#{i}', 'box': [0, 0, 10, 10], 'runs': [], 'ops': 0, **k}
    (tmp_path/'a.json').write_text(json.dumps({'schema': 'paper2-pixel-review-v1', 'pages': {'p_1_1': {'revisions': [
        rec('det0000', confirmed=True, text='old'), rec('det0001', ops=3), rec('det0002')], 'deleted_revisions': [rec('det0003', deleted=True)]}}}))
    (tmp_path/'d.json').write_text(json.dumps([
        {'id': 'p_1_1#det0000', 'page': 'p_1_1', 'box': [1, 1, 9, 9], 'region': None, 'runs': [], 'text': 'new', 'readable': True, 'pixels_confirmed': True, 'text_confirmed': True, 'mode': 'erase'},
        {'id': 'p_1_1#add1', 'page': 'p_1_1', 'box': [2, 2, 5, 5], 'region': None, 'runs': [], 'text': '', 'readable': False, 'pixels_confirmed': True, 'text_confirmed': True, 'mode': 'erase'}]))
    pages = manual_records([tmp_path/'a.json'], tmp_path/'d.json')['p_1_1']
    assert set(pages) == {'p_1_1#det0000', 'p_1_1#det0001', 'p_1_1#det0003', 'p_1_1#add1'}
    assert pages['p_1_1#det0000']['text'] == 'new' and pages['p_1_1#det0000']['box'] == [1, 1, 9, 9] and pages['p_1_1#add1']['unreadable']
    assert [k for k, r in pages.items() if hint(r)] == ['p_1_1#det0001']
