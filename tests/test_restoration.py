import numpy as np
from revrender.baselines.restoration import restore_native, removal_counts, removal_metrics


def test_native_tiling_identity_preserves_geometry_and_all_edge_pixels():
    rng=np.random.default_rng(42)
    for shape in [(36,99),(290,533),(256,256)]:
        x=rng.random(shape,dtype=np.float32)
        assert np.array_equal(restore_native(lambda a:a,x),x)


def test_removal_counts_do_not_reward_all_white_or_identity():
    image=np.ones((30,40),np.float32);image[4:20,8:12]=0;image[10:13,3:35]=0
    gold=np.zeros(image.shape,bool);gold[10:13,3:35]=True;gold[10:13,8:12]=False
    perfect=image.copy();perfect[gold]=1
    result=removal_metrics(removal_counts(image,perfect,gold))
    assert result['f1']==1 and result['valid_ink_damage']==0
    white=removal_metrics(removal_counts(image,np.ones_like(image),gold))
    assert white['f1']<1 and white['valid_ink_damage']==1
    assert removal_metrics(removal_counts(image,image,gold))['f1']==0


def test_added_ink_is_reported_separately_from_removal():
    image=np.ones((8,8),np.float32);pred=image.copy();pred[0,0]=0
    result=removal_metrics(removal_counts(image,pred,np.zeros((8,8),bool)))
    assert result['new_dark_pixels']==1 and result['tp']==0


def test_export_merge_deduplicates_only_identical_scans(tmp_path):
    import json
    from argparse import Namespace
    from PIL import Image
    from revrender.baselines.restoration_export import merge
    dirs=[]
    for name in ['NAFNet','DocDiff']:
        directory=tmp_path/name;(directory/'pack').mkdir(parents=True);dirs.append(directory)
        Image.fromarray(np.zeros((8,8),np.uint8)).save(directory/'pack/image.png')
        rows=[{'id':f'{model}|sample','model':model,'images':{'erased':'image.png'}} for model in ['scan',name]]
        (directory/'pack/pack.json').write_text(json.dumps({'rows':rows}))
        (directory/'run_manifest.json').write_text('{}')
    merge(Namespace(inputs=dirs,output_dir=tmp_path/'merged'))
    rows=json.loads((tmp_path/'merged/pack.json').read_text())['rows']
    assert len(rows)==3 and all(__import__('pathlib').Path(r['images']['erased']).is_file() for r in rows)


def test_export_serializes_clipped_region_sizes(tmp_path,monkeypatch):
    import json
    from argparse import Namespace
    from PIL import Image
    from revrender.baselines import restoration_export as module
    run=tmp_path/'run';run.mkdir();geometry=tmp_path/'geometry';geometry.mkdir();scans=tmp_path/'scans';scans.mkdir()
    checkpoint=run/'checkpoint';checkpoint.write_bytes(b'fixture')
    for name in ['run_manifest.json','selection.json']:(run/name).write_text('{}')
    metadata={'config':{'model':'NAFNet','arm':'gold_pixels'}}
    monkeypatch.setattr(module,'selected_predictor',lambda *a:(lambda x:x,checkpoint,metadata))
    image_id='sample_1_1';Image.fromarray(np.full((32,40),255,np.uint8)).save(scans/(image_id+'.png'))
    labels=tmp_path/'labels.json';labels.write_text(json.dumps({'schema':'paper2-revision-box-v1','pages':{image_id:{'done':True,'revisions':[
        {'id':'r1','kind':'line','text':'x','region':[[1.5,2.5],[38.5,2.5],[38.5,28.5],[1.5,28.5]]}]}}}))
    selection=tmp_path/'selection.json';selection.write_text(json.dumps({'pages':[{'image_id':image_id,'split':'test'}]}))
    module.export(Namespace(run=run,upstream=tmp_path,selection=selection,geometry=geometry,pages=[scans],labels=[labels],output_dir=tmp_path/'out'))
    rows=json.loads((tmp_path/'out/pack/pack.json').read_text())['rows']
    assert len(rows)==2 and rows[0]['size']==[40,32]
