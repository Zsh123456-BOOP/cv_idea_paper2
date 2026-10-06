"""Pixel review stays lossless, training-only and explicitly human-confirmed."""
import json
from argparse import Namespace

import numpy as np
import pytest
from PIL import Image

from revrender.data import pixel_masks as M
from revrender.data.page_labels import merge_exports, stroke_pixels
from revrender.tools.annotate.pixel_review import crop_request, pack
from revrender import pages


def test_native_mask_round_trip_and_invalid_runs():
    image=np.zeros((11,23),bool);image[4,2:20]=True;image[4,7:10]=False;image[7:10,15]=True
    assert np.array_equal(M.decode(M.encode(image),image.shape),image)
    for bad in ([[2,0]],[[2,4],[3,2]],[[-1,2]],[[252,2]],[[1.5,1]]):
        with pytest.raises(RuntimeError):M.decode(bad,image.shape)


def test_unconfirmed_predictions_never_become_gold_and_user_mask_is_not_redilated(tmp_path):
    r={'id':'r1','box':[3,4,13,9],'region':[[3,4],[13,4],[13,9],[3,9]],'runs':[[12,4]],'confirmed':False}
    assert not M.geometry({'done':False,'revisions':[r]})['strokes']
    with pytest.raises(RuntimeError):M.geometry({'done':True,'revisions':[r]})
    r['confirmed']=True
    export=tmp_path/'export.json';export.write_text(json.dumps({'schema':M.SCHEMA,'pages':{'p':{'done':True,'revisions':[r]}}}))
    state=merge_exports([export])['p']
    ink=np.ones((20,30),bool)
    result=stroke_pixels(state['strokes'][0],ink,ink,np.ones_like(ink),(30,20))
    assert result.sum()==4 and result[5,5:9].all()


def test_pack_refuses_nontraining_questions_and_crop_is_bounded(tmp_path):
    pre=tmp_path/'pre';pre.mkdir();images=tmp_path/'images';images.mkdir();masks=tmp_path/'masks';masks.mkdir()
    grey=np.full((50,100),255,np.uint8);grey[20:23,10:80]=30
    Image.fromarray(grey).save(images/'a_5_1.png')
    q={'id':'a_5_1','question':'5','split':'train','revisions':[{'id':'r1','kind':'line','region':[[10,10],[80,10],[80,30],[10,30]]}]}
    (pre/'prefill.json').write_text(json.dumps({'questions':[q]}))
    mask=grey<100
    pages.write_page(masks,q['id'],(100,50),[],[{'line':0,'score':1.,'box':[0,0,100,50],'mask':mask}])
    (masks/'run_manifest.json').write_text('{}')
    args=Namespace(prefill=pre,masks=masks,pages=[images],first=5,name='fixture',output_dir=tmp_path/'pack')
    pack(args)
    payload=json.loads((args.output_dir/'pack.json').read_text())
    r=payload['rows'][0]
    crop,got=crop_request(args.output_dir,payload,r)
    assert got.sum()==mask.sum() and crop.shape==got.shape
    with pytest.raises(RuntimeError):crop_request(args.output_dir,payload,{**r,'box':[-1,0,20,20]})
    q['split']='test';(pre/'prefill.json').write_text(json.dumps({'questions':[q]}))
    args.output_dir=tmp_path/'refused'
    with pytest.raises(RuntimeError):pack(args)
