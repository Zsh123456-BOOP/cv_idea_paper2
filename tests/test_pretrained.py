"""Protect validation-only selection and official checkpoint geometry."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from revrender.baselines.pretrained import candidates, checkpoint_parameters, report, restore, validation_revisions
from revrender.util import RefusedError


def test_author_checkpoints_unwrap_metadata_but_preserve_parameter_keys():
    parameters={'module.layer.weight':[1,2],'module.layer.bias':[3]}
    for key in ('model_state_dict','params','state_dict'):
        assert checkpoint_parameters({'epoch':42,key:parameters})=={'layer.weight':[1,2],'layer.bias':[3]}
    assert checkpoint_parameters({'layer.weight':[1]})=={'layer.weight':[1]}
    with pytest.raises(RefusedError):checkpoint_parameters({})


def test_only_completed_readable_validation_revisions_are_selected():
    revision={'id':'r','text':'x','kind':'line','region':[[0,0],[10,10]]}
    labels={k:{'done':True,'revisions':[revision]} for k in ('train','test','val','absent')}
    labels['unfinished']={'done':False,'revisions':[revision]}
    labels['val']['revisions']=[revision,{**revision,'id':'deleted','deleted':True},
        {**revision,'id':'unreadable','unreadable':True},{**revision,'id':'unknown','text':'?'}]
    selection={'pages':[{'image_id':k,'split':k} for k in ('train','test','val')]+[{'image_id':'unfinished','split':'val'}]}
    assert [(i,r['id']) for i,r in validation_revisions(labels,selection)]==[('val','r')]


def test_catalogue_does_not_silently_use_attribute_ground_truth(tmp_path):
    base=tmp_path/'CycleGAN/checkpoints/strikethrough_removal'
    for kind in ('ORIGINAL','ATTR_GUIDED'):
        path=base/kind/'author-run';path.mkdir(parents=True)
        (path/'genStrikeToClean_best_fmeasure.pth').write_bytes(b'checkpoint')
        (path/'config.cfg').write_text('[DEFAULT]\nblockcount=1\nmodel=dense\ninvert_images=True\n')
    result=candidates(tmp_path)
    assert len(result)==1 and result[0]['name']=='CycleGAN-Original-01'
    assert result[0]['layout']=='cycle-official' and len(result[0]['sha256'][0])==64


def test_official_geometry_restores_original_extent_without_white_padding():
    for shape in ((38,73),(37,381)):
        image=np.zeros(shape,np.float32)
        result=restore(lambda x:x,image,'cycle-official')
        assert result.shape==shape
        assert np.max(result[:,:-2])<.03
    with pytest.raises(RefusedError):restore(lambda x:x,np.zeros((5,5)),'unknown')


def test_stain_release_registers_three_distinct_task_weights(tmp_path):
    source=tmp_path/'StainRestorer';source.mkdir()
    for name in ('StainDoc.pth','StainDoc_mark.pth','StainDoc_seal.pth'):(source/name).write_bytes(name.encode())
    rows=candidates(tmp_path)
    assert {r['name'] for r in rows}=={'StainRestorer-stain','StainRestorer-watermark','StainRestorer-seal'}
    assert len({r['sha256'][0] for r in rows})==3
    assert all(r['layout']=='native256' and r['model']=='StainRestorer' for r in rows)


def test_selection_uses_transcription_validation_and_checks_complete_readings(tmp_path):
    from PIL import Image
    pack=tmp_path/'pack';pack.mkdir();picture=pack/'crop.png';Image.new('L',(8,8),255).save(picture)
    rows=[dict(id=m+'|q#r',line_id='q#r',model=m,text='x',kind='line',images={'erased':str(picture)})
          for m in ('scan','text-best','pixel-best')]
    def write(path,value):path.write_text(json.dumps(value))
    write(pack/'pack.json',{'rows':rows});scores=tmp_path/'scores.json';readings=tmp_path/'readings.jsonl'
    write(scores,{'held':{m+'/erased_neutral/line':dict(held=v,exact=v,revisions=1)
                         for m,v in [('scan',.5),('text-best',.8),('pixel-best',.7)]}})
    readings.write_text('\n'.join(json.dumps({'id':r['id'],'text':'x','finish_reason':'stop'}) for r in rows))
    runs=[]
    for m,f1 in [('text-best',.4),('pixel-best',.9)]:
        run=tmp_path/m;run.mkdir();runs.append(run)
        write(run/'pixels.json',dict(model=m,f1=f1,valid_ink_damage=.1))
        write(run/'run_manifest.json',{'candidate':{'model':'family'}})
    args=SimpleNamespace(pack=pack,scores=scores,readings=readings,runs=runs,output_dir=tmp_path/'report')
    report(args);summary=json.loads((args.output_dir/'summary.json').read_text())
    assert summary['selected']=='text-best' and summary['selected_improves_raw']
    readings.write_text('');args.output_dir=tmp_path/'incomplete'
    with pytest.raises(RefusedError,match='missing or extra'):report(args)
