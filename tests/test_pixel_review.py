"""Pixel review stays lossless, training-only and explicitly human-confirmed."""
import json
from argparse import Namespace

import numpy as np
import pytest
from PIL import Image

from revrender.data import pixel_masks as M
from revrender.data.page_labels import merge_exports, stroke_pixels
from revrender.tools.annotate.pixel_review import crop_request, pack, refresh
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


def test_refresh_uses_clean_images_and_preserves_identity_and_native_coordinates(tmp_path):
    old=tmp_path/'old';old.mkdir();clean=tmp_path/'clean';clean.mkdir()
    grey=np.full((20,30),255,np.uint8);grey[7,9:16]=20
    Image.fromarray(grey).save(clean/'p.png');(clean/'run_manifest.json').write_text('{}')
    mask=np.zeros((8,12),bool);mask[2,2:9]=True;mask[5,10]=True
    row={'id':'p#r1','page':0,'box':[7,5,19,13],'region':[[7,5],[19,5],[19,13],[7,13]],'runs':M.encode(mask)}
    payload={'id':'kept-id','pages':[{'id':'p','size':[30,20],'image':'pages/0.png'}],'rows':[row]}
    source=json.dumps(payload);(old/'pack.json').write_text(source)
    args=Namespace(pack=old,clean_pages=clean,output_dir=tmp_path/'new')
    refresh(args)
    new=json.loads((args.output_dir/'pack.json').read_text())
    assert new['id']=='kept-id' and new['rows'][0]['id']==row['id']
    assert new['rows'][0]['box']==row['box'] and new['rows'][0]['region']==row['region']
    assert new['image_revision'] and new['image_source']==str(clean)
    assert np.array_equal(np.asarray(Image.open(args.output_dir/'pages/0.png').convert('L')),grey)
    got=M.decode(new['rows'][0]['runs'],mask.shape)
    assert got.sum()==7 and not got[5,10]
    # Even a browser with stale pre-clean pixels cannot feed white noise to the refiner.
    crop,filtered=crop_request(args.output_dir,new,row)
    assert np.array_equal(got,filtered) and crop[5,10]==255
    assert (old/'pack.json').read_text()==source
    with pytest.raises(FileExistsError):refresh(args)
    Image.new('L',(31,20),255).save(clean/'p.png')
    args.output_dir=tmp_path/'bad'
    with pytest.raises(RuntimeError,match='dimensions'):refresh(args)
    assert not args.output_dir.exists()


def test_offline_pack_embeds_human_progress_and_preserves_original(tmp_path):
    from revrender.tools.annotate.pixel_offline import standalone
    old=tmp_path/'old';old.mkdir()
    grey=np.full((20,30),255,np.uint8);grey[7,9:16]=20;grey[8,9:16]=225;grey[2,3]=225
    Image.fromarray(grey).save(old/'p.png')
    row={'id':'p#r1','page':0,'box':[0,0,30,20],'region':[[0,0],[30,0],[30,20],[0,20]],'runs':M.encode(grey<245)}
    payload={'id':'fixture','pages':[{'id':'p','size':[30,20],'image':'p.png','question':'5'}],'rows':[row]}
    (old/'pack.json').write_text(json.dumps(payload))
    export={'schema':M.SCHEMA,'pack':'fixture','pages':{'p':{'done':False,'revisions':[
        {**row,'confirmed':True,'note':'human </script> note','seconds':25},
        {**row,'id':'p#manual','runs':[[219,3]],'confirmed':True,'note':'manual'}]}}}
    source=tmp_path/'human.json';source.write_text(json.dumps(export))
    args=Namespace(pack=old,review_export=source,output_dir=tmp_path/'offline')
    standalone(args)
    out=args.output_dir;html=(out/'标注.html').read_text()
    assert '<script src=' not in html and 'data:image/png;base64,' in html
    assert 'id="review-draft"' in html and '\\u003c/script>' in html
    assert (out/'source_annotations.json').read_bytes()==source.read_bytes()
    initial=json.loads((out/'imported_annotations.json').read_text())
    r,manual=initial['pages']['p']['revisions']
    assert not r['confirmed'] and r['cleaned_pixels']==1 and r['note']=='human </script> note'
    assert manual['confirmed'] and manual['runs']==[[219,3]]
    clean=np.asarray(Image.open(out/'images/p.png').convert('L'))
    assert clean[2,3]==255 and (clean[grey<200]==grey[grey<200]).all()
    assert (clean[8,9:16]==225).all() and clean.shape==grey.shape
    assert initial['image_source']==str(out/'images') and initial['image_revision']
    assert not initial['pages']['p']['done']
    with pytest.raises(FileExistsError):standalone(args)


def test_html_asset_update_keeps_images_storage_identity_and_embedded_progress(tmp_path):
    from revrender.tools.annotate.pixel_offline import render_html, update_html, EmbeddedReview
    payload={'id':'existing','standalone':True,'image_revision':'keep-key','pages':[{'image':'data:image/png;base64,abc'}]}
    draft={'savedAt':1234,'rows':{'r':{'note':'human </script> data','runs':[[2,4]],'deleted':True}}}
    source=tmp_path/'existing.html';source.write_text(render_html(payload,draft))
    old=source.read_bytes();out=tmp_path/'update'
    update_html(Namespace(source=source,output_dir=out))
    values=EmbeddedReview((out/'标注.html').read_text()).values
    assert json.loads(values['review-pack'])==payload
    assert json.loads(values['review-draft'])==draft
    assert source.read_bytes()==old
    assert '自动清理：灰度 ≥100' in (out/'标注.html').read_text()


def test_click_refiner_receives_white_outside_polygon(tmp_path):
    grey=np.full((10,10),20,np.uint8);Image.fromarray(grey).save(tmp_path/'p.png')
    domain=np.zeros((10,10),bool);domain[:3,:]=True;domain[:, :3]=True
    payload={'pages':[{'image':'p.png','size':[10,10]}]}
    crop,mask=crop_request(tmp_path,payload,{'page':0,'box':[0,0,10,10],
        'runs':[[0,100]],'domain_runs':M.encode(domain)})
    assert (crop[~domain]==255).all() and (crop[domain]==20).all()
    assert np.array_equal(mask,domain)


def test_exact_batch_preserves_whole_questions_and_writer_diversity():
    from revrender.tools.annotate.pixel_batch import exact_questions
    questions=[{'id':str(i),'revisions':[{}]*n} for i,n in enumerate([6,3,2,2,3])]
    chosen=exact_questions(questions,10)
    assert sum(len(q['revisions']) for q in chosen)==10
    assert len(chosen)==4 and len({q['id'] for q in chosen})==4
    with pytest.raises(RuntimeError):exact_questions(questions,1)


def test_standalone_without_human_export_keeps_model_prefill_unconfirmed(tmp_path):
    from revrender.tools.annotate.pixel_offline import standalone, EmbeddedReview
    source=tmp_path/'pack';source.mkdir()
    grey=np.full((10,20),255,np.uint8);grey[5,3:15]=20
    Image.fromarray(grey).save(source/'p.png')
    row={'id':'r','page':0,'box':[0,0,20,10],'region':[[0,0],[20,0],[20,10],[0,10]],'runs':M.encode(grey<200)}
    (source/'pack.json').write_text(json.dumps({'id':'new-batch','pages':[{'id':'p','size':[20,10],'image':'p.png'}],'rows':[row]}))
    standalone(Namespace(pack=source,review_export=None,output_dir=tmp_path/'offline'))
    result=json.loads(EmbeddedReview((tmp_path/'offline/标注.html').read_text()).values['review-pack'])
    assert result['rows'][0]['runs']==row['runs'] and not result['rows'][0].get('confirmed')
    assert result['initial_export']['pages']=={}


def test_reviewed_training_crop_uses_polygon_and_ignores_archived_pixels():
    page=Image.fromarray(np.full((20,30),20,np.uint8))
    row={'box':[10,5,20,15],'region':[[10,5],[20,5],[20,8],[13,8],[13,15],[10,15]],
         'runs':[[0,100]],'outsideMask':{'box':[10,5,20,15],'runs':[[88,1]]}}
    image,label=M.reviewed_crop(page,row)
    assert label.sum()==51 and image[8,8]==255 and label[8,8]==0
    assert image[1,8]==20 and label[1,8]==1
    _,negative=M.reviewed_crop(page,{**row,'not_revision':True})
    assert not negative.any()


def test_unknown_pixels_are_not_silently_converted_to_binary_training_labels():
    page=Image.fromarray(np.full((10,10),20,np.uint8))
    row={'id':'r','box':[0,0,10,10],'region':[[0,0],[10,0],[10,3],[3,3],[3,10],[0,10]],
         'runs':[[25,1]],'confirmed':True,'uncertainMask':{'box':[0,0,10,10],'runs':[[23,1]]}}
    with pytest.raises(RuntimeError,match='ignore-mask'):
        M.reviewed_crop(page,row)
    with pytest.raises(RuntimeError,match='uncertain'):
        M.geometry({'done':False,'revisions':[row]})
    # Shrinking the polygon parks older unknown annotations without discarding them.
    row['uncertainMask']['runs']=[[88,1]]
    _,mask=M.reviewed_crop(page,row)
    assert mask.sum()==1 and mask[2,5]==1
    assert len(M.geometry({'done':False,'revisions':[row]})['strokes'])==1
