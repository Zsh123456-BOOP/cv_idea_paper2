import random
import numpy as np
from revrender.data.restoration_pairs import draw_pair, host_canvas, placement, SIZE


def test_clean_target_preserves_ink_hidden_by_synthetic_stroke():
    clean=np.full((32,40),255,np.uint8);clean[4:28,18:21]=80
    alpha=np.zeros(clean.shape,np.float32);alpha[13:17,:]=1
    image,target=draw_pair(clean,alpha,10)
    assert np.array_equal(target,clean)
    assert image[14,19]==10 and target[14,19]==80
    assert image[14,5]==10 and target[14,5]==255
    assert not np.shares_memory(target,clean)


def test_host_crop_and_placement_keep_native_aspect_and_intersect_ink():
    grey=np.full((40,360),255,np.uint8)
    for x in range(10,350,20):grey[8:32,x:x+4]=30
    result=host_canvas(grey,random.Random(42));assert result is not None
    canvas,box=result;assert canvas.shape==(SIZE,SIZE)
    l,t,w,h=placement(canvas,random.Random(42))
    assert (canvas[t:t+h,l:l+w]<200).any()
    assert 0<=l<l+w<=SIZE and 0<=t<t+h<=SIZE


def test_lighter_overlay_never_bleaches_existing_darker_writing():
    clean=np.array([[0,40,255],[100,200,255]],np.uint8)
    x,y=draw_pair(clean,np.full(clean.shape,.8,np.float32),80)
    assert np.all(x<=clean) and np.array_equal(y,clean)
    assert x[0,0]==0 and x[0,1]==40 and x[0,2]<255
