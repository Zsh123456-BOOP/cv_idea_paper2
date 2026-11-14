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


def _pairs(tmp_path):
    """Two restoration pairs as data.restoration_pairs writes them: a struck one and a clean one."""
    import json
    from revrender.util import digest, write_manifest
    from revrender.data.restoration_pairs import draw_pair
    d=tmp_path/'pairs';d.mkdir();clean=np.full((SIZE,SIZE),255,np.uint8)
    for x in range(40,200,24):clean[110:140,x:x+12]=60
    alpha=np.zeros(clean.shape,np.float32);alpha[124:127,30:210]=1
    struck,target=draw_pair(clean,alpha,20)
    np.save(d/'gold_pixels.npy',np.stack([struck,clean,struck]));np.save(d/'targets.npy',np.stack([target,clean,target]))
    rows=[dict(id='w1_1:L1#1',line_id='w1_1:L1',group_uuid='w1',split='train',clean=False,seed=7),
          dict(id='w1_1:L1#0',line_id='w1_1:L1',group_uuid='w1',split='train',clean=True,seed=8),
          dict(id='w9_1:L1#1',line_id='w9_1:L1',group_uuid='w9',split='val',clean=False,seed=9)]
    (d/'records.json').write_text(json.dumps(rows))
    write_manifest(d,config={},inputs={},files={n:digest(d/n) for n in ('gold_pixels.npy','targets.npy','records.json')})
    return d,struck,target


def test_pairs_cut_as_box_strips_keep_the_strike_as_one_instance_and_the_clean_pair_as_crops(tmp_path):
    import json
    from revrender.data.restoration_pairs import as_boxes
    from revrender.data.synthetic_set import read_set
    d,_,_=_pairs(tmp_path);as_boxes(d,'gold_pixels',tmp_path/'boxes')
    records,images,labels=read_set(tmp_path/'boxes')
    assert images.shape[1:]==(96,768)
    struck=[i for i,r in enumerate(records) if r['revisions']]
    assert [records[i]['pair_index'] for i in struck]==[0,2] and all(labels[i].max()==1 for i in struck)
    clean=[r for r in records if not r['revisions']]
    assert clean and all(r['pair_index']==1 for r in clean) and all(labels[i].max()==0 for i,r in enumerate(records) if not r['revisions'])


def test_native_tiles_whiten_only_the_pixels_to_remove_and_cover_the_whole_crop():
    from revrender.data.restoration_pairs import tiles
    image=np.full((40,600),255,np.uint8);image[10:30,::7]=50;mask=np.zeros(image.shape,bool);mask[19:21,:]=True;image[mask]=30
    out=tiles(image,mask)
    assert len(out)==3 and all(t.shape==(SIZE,SIZE) and g.shape==(SIZE,SIZE) and (w==1).all() for t,g,w in out)
    t,g,_=out[0];changed=t!=g
    assert changed.any() and (g[changed]==255).all() and (t[changed]==30).all()


def test_human_pairs_join_the_regions_tiles_to_the_replayed_pairs_and_keep_validation_writers_apart(tmp_path):
    import json
    from revrender.data.restoration_pairs import human_pairs
    d,_,_=_pairs(tmp_path);s=tmp_path/'set';s.mkdir()
    (s/'run_manifest.json').write_text('{}')
    image=np.full((30,100),255,np.uint8);image[10:20,10:90]=40;mask=np.zeros(image.shape,bool);mask[14:16,:]=True
    np.savez(s/'training_native.npz',image_0=image,mask_0=mask)
    (s/'training_native.json').write_text(json.dumps([dict(id='h1',group_uuid='hw')]))
    (s/'records.json').write_text(json.dumps([dict(id='h1',group_uuid='hw',split='train',source='human_confirmed',revisions=1),
        dict(id='w1_1:L1#1',group_uuid='w1',split='train',source='synthetic_replay',revisions=1,pair_index=0),
        dict(id='x',group_uuid='z',split='val',source='synthetic_replay',revisions=0,pair_index=1)]))
    human_pairs(s,d,'gold_pixels',tmp_path/'out')
    rows=json.loads((tmp_path/'out'/'records.json').read_text())
    assert [r['source'] for r in rows]==['human_confirmed','synthetic_replay','synthetic_validation']
    assert rows[2]['group_uuid']=='w9' and np.load(tmp_path/'out'/'gold_pixels.npy').shape==(3,SIZE,SIZE)

