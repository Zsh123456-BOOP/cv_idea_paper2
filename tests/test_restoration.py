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


def test_unpaired_domains_have_independent_order_but_reproducible_batches():
    from revrender.baselines.restoration import Experiment
    from types import SimpleNamespace
    def make():
        e=Experiment.__new__(Experiment);e.args=SimpleNamespace(batch_size=4)
        e.train=np.arange(100);e.x=np.arange(100,dtype=np.uint8);e.y=e.x.copy()
        e.rng=np.random.default_rng(42)
        e.unpaired_rng=np.random.default_rng(np.random.SeedSequence(42).spawn(1)[0])
        return e
    e=make();clone=make();paired=make()
    for _ in range(5):
        x,y=e.batch(unpaired=True);xx,yy=clone.batch(unpaired=True);px,py=paired.batch()
        assert np.array_equal(x,xx) and np.array_equal(y,yy)
        assert np.array_equal(x,px) and np.array_equal(px,py)
        assert not np.array_equal(x,y)


def test_two_weight_candidates_select_validation_best_and_count_one_plateau(tmp_path):
    from revrender.baselines.restoration import Experiment
    from types import SimpleNamespace
    import time,json
    e=Experiment.__new__(Experiment)
    e.args=SimpleNamespace(batch_size=1,steps=100,min_steps=1,patience=2)
    e.out=tmp_path;e.native_rows=[{}]
    im=np.full((12,12),255,np.uint8);im[4:8,4:8]=0
    gold=im==0;e.native={'image_0':im,'mask_0':gold}
    e.x=im[None];e.y=np.full_like(e.x,255);e.val=np.array([0])
    e.best=(-1,None,'');e.start=time.monotonic();e.stale=0;e.stop=False;e.candidate_improved=False
    save=lambda path:path.mkdir()
    # Raw is perfect; bad EMA must neither replace it nor count as a plateau on this evaluation.
    e.evaluate(lambda x:np.ones_like(x),1,save,variant='raw',final_candidate=False)
    e.evaluate(lambda x:x,1,save,variant='ema')
    assert e.best==(1.,1,'_raw') and e.stale==0 and not e.stop
    for step in (2,3):
        e.evaluate(lambda x:x,step,save,variant='raw',final_candidate=False)
        e.evaluate(lambda x:x,step,save,variant='ema')
    assert e.stale==2 and e.stop
    e.finish();selection=json.loads((tmp_path/'selection.json').read_text())
    assert selection['checkpoint']=='step_1_raw' and selection['evaluation']=='evaluation_1_raw.json'
    assert selection['actual_steps']==3


def test_report_reads_selected_ema_or_raw_evaluation_and_variable_model_count(tmp_path):
    from revrender.tools.restoration_report import build
    from types import SimpleNamespace
    from PIL import Image
    import json
    root=tmp_path/'evaluation';root.mkdir();(root/'COMPLETED').touch()
    run=tmp_path/'training';run.mkdir();pack=root/'pack';pack.mkdir();ocr=root/'qwen8b';ocr.mkdir()
    exported=root/'NAFNet_gold_pixels';exported.mkdir()
    def write(path,value):path.write_text(json.dumps(value))
    image=tmp_path/'sample.png';Image.fromarray(np.zeros((3,3),np.uint8)).save(image)
    rows=[{'id':m+'|q#r','line_id':'q#r','image_id':'q','model':m,'kind':'line','text':'x',
           'images':{'erased':str(image)}} for m in ('scan','NAFNet_gold_pixels')]
    write(pack/'pack.json',{'rows':rows})
    (ocr/'predictions.jsonl').write_text('\n'.join(json.dumps({'id':r['id'],'text':'x','finish_reason':'stop'}) for r in rows))
    write(root/'read_qwen8b.json',{'held':{m+'/erased_neutral/line':{'held':1.,'exact':1.,'revisions':1} for m in ('scan','NAFNet_gold_pixels')}})
    write(exported/'pixels.json',{'model':'NAFNet_gold_pixels'})
    write(exported/'run_manifest.json',{'config':{'run':str(run)}})
    write(run/'run_manifest.json',{'config':{'steps':100}})
    write(run/'selection.json',{'step':42,'evaluation':'evaluation_42_raw.json'})
    write(run/'evaluation_42_raw.json',{'human_removal':{'f1':.5},'variant':'raw'})
    out=tmp_path/'report';build(SimpleNamespace(evaluation=root,output_dir=out,expected_models=1))
    summary=json.loads((out/'summary.json').read_text())
    assert summary['models'][0]['validation']['variant']=='raw'
    assert summary['models'][0]['training']['steps']==100
    assert (out/'index.html').is_file()



def test_a_version_is_kept_when_its_blocks_to_fix_fall_though_the_score_is_negative(tmp_path):
    from revrender.baselines.restoration import Experiment
    from types import SimpleNamespace
    import time
    e=Experiment.__new__(Experiment);e.args=SimpleNamespace(batch_size=1,steps=100,min_steps=1,patience=0,select='blocks')
    e.out=tmp_path;e.native_rows=[{}]
    im=np.full((12,12),255,np.uint8);im[4:8,4:8]=0;e.native={'image_0':im,'mask_0':im==0}
    e.x=im[None];e.y=np.full_like(e.x,255);e.val=np.array([],int)
    e.best=(-np.inf,None,'');e.start=time.monotonic();e.stale=0;e.stop=False;e.candidate_improved=False
    e.evaluate(lambda x:x,1,lambda path:path.mkdir())
    assert e.best[1]==1 and (tmp_path/'step_1').is_dir()
