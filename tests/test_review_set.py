import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from revrender.data import review_set as R
from revrender.util import RefusedError


def test_only_confirmed_regions_enter_tuning_without_inventing_page_completion(tmp_path,monkeypatch):
    images=tmp_path/'images';images.mkdir()
    Image.fromarray(np.full((10,10),30,np.uint8)).save(images/'w_1.png')
    base={'id':'a','box':[0,0,10,10],'region':[[0,0],[10,0],[10,3],[3,3],[3,10],[0,10]],'runs':[[22,1],[88,1]],'confirmed':True}
    path=tmp_path/'export.json';data={'schema':R.SCHEMA,'pages':{'w_1':{'done':False,'revisions':[base,{**base,'id':'draft','confirmed':False}], 'deleted_revisions':[{**base,'id':'deleted'}]}}}
    path.write_text(json.dumps(data));mapping=tmp_path/'map.json';mapping.write_text('{}')
    monkeypatch.setattr(R,'writer_splits',lambda *a:{'w':'train'});monkeypatch.setattr(R,'twins',lambda *a:{})
    rows,_,skipped=R.confirmed([path],[images],mapping,mapping)
    assert len(rows)==1 and skipped==1 and rows[0]['mask'].sum()==1 and rows[0]['image'][8,8]==255
    assert not json.loads(path.read_text())['pages']['w_1']['done']
    monkeypatch.setattr(R,'writer_splits',lambda *a:{'w':'test'})
    with pytest.raises(RefusedError,match='held-out'):R.confirmed([path],[images],mapping,mapping)


def test_human_validation_writers_and_twins_are_excluded_from_replay(tmp_path,monkeypatch):
    rows=[{'id':str(i),'group_uuid':f'w{i}','page':f'w{i}_1','image':np.full((10,10),20,np.uint8),'mask':np.ones((10,10),np.uint8),'box':[0,0,10,10],'region':[[0,0],[10,0],[10,10],[0,10]]} for i in range(5)]
    monkeypatch.setattr(R,'confirmed',lambda *a:(rows,{},2));monkeypatch.setattr(R,'twins',lambda *a:{'w0':'z0','z0':'w0'})
    monkeypatch.setattr(R,'writer_splits',lambda *a:{})
    records=[{'id':f's{i}','group_uuid':f's{i}','split':'train' if i<10 else 'val','revisions':2 if i<5 else 0} for i in range(14)]
    records.append({'id':'twin','group_uuid':'z0','split':'train','revisions':2})
    x=np.full((15,96,768),255,np.uint8);y=np.zeros_like(x);y[:5,20:23,20:30]=1;y[:5,25:28,20:30]=2;y[14]=y[0]
    monkeypatch.setattr(R,'read_set',lambda p:(records,x,y))
    replay=tmp_path/'replay';replay.mkdir();(replay/'run_manifest.json').write_text('{}')
    args=SimpleNamespace(review_export=[],pages=[],mapping=None,same_student=None,validation_writers=1,replay=replay,replay_clean=4,validation_clean=4,output_dir=tmp_path/'out')
    R.build(args);result=json.loads((args.output_dir/'records.json').read_text())
    assert 'twin' not in {r['id'] for r in result}
    train={r['group_uuid'] for r in result if r['split']=='train'};val={r['group_uuid'] for r in result if r['split']=='val'}
    assert not train&val
    assert sum(r['source']=='human_confirmed' for r in result)==5


def test_incremental_tuning_keeps_fixed_validation_and_previous_replay(tmp_path,monkeypatch):
    base=tmp_path/'base';base.mkdir()
    (base/'validation_native.json').write_text(json.dumps([{'group_uuid':'held','id':'v'}]))
    np.savez(base/'validation_native.npz',image_0=np.full((4,4),20,np.uint8),mask_0=np.ones((4,4),np.uint8))
    (base/'run_manifest.json').write_text('{}')
    records=[{'id':'old','group_uuid':'oldwriter','split':'train','source':'human_confirmed','revisions':1},
             {'id':'replay','group_uuid':'synthetic','split':'train','source':'synthetic_replay','revisions':0},
             {'id':'v','group_uuid':'held','split':'val','source':'human_confirmed','revisions':1}]
    x=np.full((3,96,768),200,np.uint8);y=np.zeros_like(x);y[0,2,2]=1;y[2,3,3]=1
    monkeypatch.setattr(R,'read_set',lambda p:([dict(r) for r in records],x,y))
    monkeypatch.setattr(R,'twins',lambda *a:{'held':'heldTwin','heldTwin':'held'})
    rows=[{'id':'new','group_uuid':'newwriter','image':np.full((10,10),20,np.uint8),'mask':np.eye(10,dtype=np.uint8)}]
    monkeypatch.setattr(R,'confirmed',lambda *a,**kw:(rows,{},0))
    args=SimpleNamespace(base_set=base,review_export=[],pages=[],mapping=None,same_student=None,output_dir=tmp_path/'out')
    R.extend(args)
    result=json.loads((args.output_dir/'records.json').read_text())
    assert result[:3]==records and result[3]['split']=='train'
    assert np.array_equal(np.load(args.output_dir/'images.npy')[:3],x)
    assert np.array_equal(np.load(args.output_dir/'labels.npy')[:3],y)
    for name in ('validation_native.json','validation_native.npz'):
        assert (base/name).read_bytes()==(args.output_dir/name).read_bytes()
    rows[0]['group_uuid']='held'
    with pytest.raises(RefusedError,match='fixed validation'):R.extend(args)


def test_unused_training_writer_is_explicitly_allowed_but_test_twin_is_not(tmp_path,monkeypatch):
    mapping=tmp_path/'mapping';mapping.write_text('{}')
    Image.new('L',(10,10),20).save(tmp_path/'new_1.png')
    row={'id':'r','confirmed':True,'box':[0,0,10,10],'region':[[0,0],[10,0],[10,10],[0,10]],'runs':[[0,1]]}
    export=tmp_path/'export.json';export.write_text(json.dumps({'schema':R.SCHEMA,'pages':{'new_1':{'revisions':[row]}}}))
    monkeypatch.setattr(R,'writer_splits',lambda *a:{})
    monkeypatch.setattr(R,'twins',lambda *a:{})
    assert len(R.confirmed([export],[tmp_path],mapping,mapping,allow_unused=True)[0])==1
    monkeypatch.setattr(R,'writer_splits',lambda *a:{'twin':'test'})
    monkeypatch.setattr(R,'twins',lambda *a:{'new':'twin','twin':'new'})
    with pytest.raises(RefusedError,match='held-out'):R.confirmed([export],[tmp_path],mapping,mapping,allow_unused=True)
