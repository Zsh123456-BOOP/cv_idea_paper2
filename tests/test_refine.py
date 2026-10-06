"""Correction guidance and locality contracts; no real student labels are used."""
import numpy as np
import pytest

from revrender.refine import click_maps, simulate, refine_crop


def test_review_training_accepts_confirmed_positive_and_negative_crops_only(tmp_path):
    import json
    from PIL import Image
    from revrender.train_refiner import reviewed
    from revrender.data.pixel_masks import SCHEMA
    scans=tmp_path/'scans';scans.mkdir()
    image=np.full((30,70),255,np.uint8);image[10:13,10:60]=30
    Image.fromarray(image).save(scans/'a_5_1.png')
    mapping=tmp_path/'mapping.json';pairs=tmp_path/'pairs.json';selection=tmp_path/'selection.json';export=tmp_path/'export.json'
    mapping.write_text(json.dumps({'rows':[{'group_uuid':'a','split':'train'}]}));pairs.write_text('{"pairs":[]}')
    selection.write_text(json.dumps({'pages':[{'image_id':'a_5_1','split':'train'}]}))
    region={'box':[5,5,65,20],'runs':[[310,20]],'confirmed':True}
    state={'done':True,'revisions':[region,{**region,'not_revision':True}]}
    payload={'schema':SCHEMA,'pages':{'a_5_1':state,'unfinished_5_1':{'done':False,'revisions':[]}}}
    export.write_text(json.dumps(payload))
    images,labels=reviewed([export],selection,[scans],mapping,pairs)
    assert images.shape==labels.shape==(2,96,768)
    assert labels[0].any() and not labels[1].any()
    with pytest.raises(RuntimeError,match='held-out writer'):
        reviewed([export],selection,[scans],mapping,pairs,{'a'})
    pairs.write_text('{"pairs":[{"x":"a","y":"b"}]}')
    with pytest.raises(RuntimeError,match='held-out writer'):
        reviewed([export],selection,[scans],mapping,pairs,{'b'})
    mapping.write_text(json.dumps({'rows':[{'group_uuid':'a','split':'train'},{'group_uuid':'b','split':'dev'}]}))
    with pytest.raises(RuntimeError,match='held-out writer'):
        reviewed([export],selection,[scans],mapping,pairs)
    pairs.write_text('{"pairs":[]}')
    state['revisions'][0]['confirmed']=False;export.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError,match='draft'):
        reviewed([export],selection,[scans],mapping,pairs)
    state['revisions'][0]['confirmed']=True;export.write_text(json.dumps(payload))
    selection.write_text(json.dumps({'pages':[{'image_id':'a_5_1','split':'test'}]}))
    with pytest.raises(RuntimeError,match='non-training'):
        reviewed([export],selection,[scans],mapping,pairs)


def test_add_and_keep_clicks_are_distinct_and_simulation_is_reproducible():
    maps=click_maps((24,40),[(10,8,True),(30,18,False)])
    assert maps[0,8,10]==1 and maps[1,18,30]==1 and maps[1,8,10]<.01
    grey=np.full((24,40),255,np.uint8); grey[10:13,4:35]=30
    target=grey<100
    a=simulate(grey,target,np.random.default_rng(42))
    b=simulate(grey,target,np.random.default_rng(42))
    assert np.array_equal(a[0],b[0]) and a[2]==b[2]
    assert all(bool(target[y,x])==add for x,y,add in a[2])


def test_suggestions_cannot_change_pixels_far_from_clicks_and_respect_keep_points():
    torch=pytest.importorskip('torch')
    class AllErase(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.dummy=torch.nn.Parameter(torch.zeros(1))
        def forward(self,x):return torch.ones_like(x[:,0:1])*10+self.dummy
    grey=np.full((40,120),255,np.uint8); grey[18:22,10:110]=20
    prior=np.zeros_like(grey,bool);prior[18:22,80:100]=True
    result,_=refine_crop(AllErase(),grey,prior,[(30,20,True),(35,20,False)],radius=8)
    assert np.array_equal(result[:,50:],prior[:,50:])
    assert result[20,30] and not result[20,35] and not result[2].any()
