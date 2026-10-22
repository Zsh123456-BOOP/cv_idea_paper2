import numpy as np

from revrender.eval.restoration_errors import measure, zone_counts


def _crop():
    image=np.full((40,120),255,np.uint8)
    for x in range(10,110,20):image[8:32,x:x+4]=60       # five upright strokes of writing
    stroke=np.zeros(image.shape,bool);stroke[19:22,5:115]=True;image[stroke]=20
    return image,stroke


def test_a_strike_left_where_it_crosses_writing_counts_near_and_erased_writing_away_counts_away():
    image,stroke=_crop();domain=np.ones(image.shape,bool)
    restored=image.copy();restored[stroke]=255;restored[19:22,10:14]=image[19:22,10:14]   # one crossing kept
    restored[8:12,90:94]=255                                                               # writing erased far from the strike
    counts=zone_counts(image,stroke,domain,restored)
    assert counts['stroke_near_writing'][1]>0 and counts['stroke_away'][1]==0
    assert counts['writing_away'][1]==16 and counts['writing_near_stroke'][1]==0


def test_a_long_thin_straight_strike_measures_as_one():
    image,stroke=_crop();m=measure(image,stroke)
    assert 2<=m['stroke_width']<=4 and m['straightness']<.05 and m['length_in_heights']>4 and m['darker_than_writing']==40
