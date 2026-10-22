import numpy as np
import pytest

from revrender.baselines.restoration import removal_counts
from revrender.eval.unified_restoration import extend_inputs, freeze_specs, restore_case, verify_files
from revrender.tools.unified_restoration_report import audit
from revrender.util import RefusedError, digest


def test_mask_and_restored_image_use_identical_dark_ink_domain():
    source=np.array([[30,100,199,200,230,255]],np.uint8)
    gold=np.array([[1,0,1,1,1,0]],bool)
    mask=np.ones_like(gold)
    raw=source.astype(np.float32)/255
    restored=np.where(mask,1.,raw)
    # The target and valid-ink denominators both exclude light gray/background.
    assert removal_counts(raw,restored,gold).tolist()==[2,1,0,1,0]
    assert removal_counts(raw,raw,gold).tolist()==[0,0,2,1,0]


def test_polygon_only_pasting_and_overlap_are_counted_once():
    source=np.full((3,5),30,np.uint8)
    windows=[[0,0,4,3],[1,0,5,3]]
    a=np.zeros((3,4),bool);a[1,1:]=True
    b=np.zeros((3,4),bool);b[1,:3]=True
    result=restore_case(lambda x:np.ones_like(x),source,windows,[a,b])
    domain=np.zeros_like(source,bool);domain[1,1:4]=True
    assert (result==1).sum()==3
    assert np.array_equal(result[~domain],(source.astype(np.float32)/255)[~domain])
    assert removal_counts(source[domain]/255,result[domain],domain[domain]).tolist()==[3,0,0,0,0]


def test_frozen_checkpoint_change_is_rejected(tmp_path):
    checkpoint=tmp_path/'weights.pt';checkpoint.write_bytes(b'first')
    spec=freeze_specs([dict(name='Ours',type='own',checkpoint=str(checkpoint))])[0]
    verify_files(spec['sha256'])
    checkpoint.write_bytes(b'changed')
    with pytest.raises(RefusedError,match='frozen input changed'):verify_files(spec['sha256'])


def test_report_rejects_duplicate_or_incomplete_readings():
    rows=[dict(id='a')]
    with pytest.raises(RefusedError,match='missing or extra'):audit(rows,[],[],{})
    with pytest.raises(RefusedError,match='duplicate'):audit(rows,[dict(id='a'),dict(id='a')],[],{})


def test_report_requires_same_labels_even_when_counts_match():
    rows=[]
    for split,count in [('val',41),('test',81)]:
        for name in ('scan','candidate'):
            rows.extend(dict(id=f'{split}|{name}|{i}',split=split,model=split+'__'+name,
                             line_id=str(i),kind='line',text='x',size=[20,10]) for i in range(count))
    readings=[dict(id=r['id'],condition='erased_neutral') for r in rows]
    audit(rows,readings,[dict(name='candidate')],{'val':41,'test':81})
    audit([r for r in rows if r['split']=='val'],[r for r in readings if r['id'].startswith('val')],[dict(name='candidate')],{'val':41})
    rows[-1]['text']='different'
    with pytest.raises(RefusedError,match='different OCR examples'):audit(rows,readings,[dict(name='candidate')],{'val':41,'test':81})
import json
from types import SimpleNamespace



def test_extend_preserves_inputs_and_reuses_only_matching_cached_inference(tmp_path):
    base=tmp_path/'base'; source=base/'inputs'; source.mkdir(parents=True)
    (base/'COMPLETED').touch(); (source/'crops').mkdir(); (source/'crops/image.png').write_bytes(b'image')
    for name in ('protocol.json','pack.json','pixels.json','counts.json'): (source/name).write_text('{}')
    (source/'pixels.npz').write_bytes(b'arrays')
    checkpoint=tmp_path/'checkpoint.pt'; checkpoint.write_bytes(b'weights')
    prior=freeze_specs([dict(name='Existing',type='own',checkpoint=str(checkpoint))])
    (source/'models.json').write_text(json.dumps(prior))
    hashes={str(p):digest(p) for p in source.rglob('*') if p.is_file()}
    (source/'prepared_sha256.json').write_text(json.dumps(hashes))
    run=base/'runs/Existing'; run.mkdir(parents=True)
    manifest=dict(model=prior[0],inputs={str(source/'prepared_sha256.json'):digest(source/'prepared_sha256.json')})
    (run/'run_manifest.json').write_text(json.dumps(manifest)); (run/'pixels.json').write_text('{}')
    candidates=tmp_path/'models.json';candidates.write_text(json.dumps([dict(name='Candidate',type='own',checkpoint=str(checkpoint))]))
    out=tmp_path/'extension'
    extend_inputs(SimpleNamespace(base=base,models=candidates,output_dir=out))
    assert (out/'inputs/pixels.npz').read_bytes()==b'arrays'
    assert (out/'runs/Existing/run_manifest.json').read_bytes()==(run/'run_manifest.json').read_bytes()
    assert len(json.loads((out/'inputs/models.json').read_text()))==2
    verify_files(hashes)
    manifest['inputs'][str(source/'prepared_sha256.json')]='changed'
    (run/'run_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RefusedError,match='different inputs'):
        extend_inputs(SimpleNamespace(base=base,models=candidates,output_dir=tmp_path/'mismatched'))


def _evaluation(tmp_path,versions):
    """An evaluation directory as eval_unified_restoration.sh leaves it: versions = [(name, family, step, f1, [read texts])]."""
    root=tmp_path/'eval';(root/'inputs').mkdir(parents=True);(root/'pack').mkdir();(root/'qwen8b').mkdir()
    specs=[];rows=[];readings=[]
    for name,family,step,f1,texts in versions:
        specs.append(dict(name=name,family=family,step=step))
        (root/'runs'/name).mkdir(parents=True)
        (root/'runs'/name/'pixels.json').write_text(json.dumps({'splits':{'val':{'f1':f1,'tp':f1*1000,'fp':(1-f1)*1000,'fn':(1-f1)*1000}}}))
        for i,text in enumerate(texts):
            rid=f'val|{name}|w{i}_1#r';rows.append(dict(id=rid,split='val',model='val__'+name,kind='line',line_id=f'w{i}_1#r',text='abcd'))
            readings.append(json.dumps(dict(id=rid,text=text,condition='erased_neutral')))
    (root/'inputs'/'models.json').write_text(json.dumps(specs));(root/'pack'/'pack.json').write_text(json.dumps({'rows':rows}))
    (root/'qwen8b'/'predictions.jsonl').write_text('\n'.join(readings))
    return root


def test_choose_reads_back_only_among_pixel_improvements_and_the_last_version(tmp_path):
    from revrender.eval.unified_restoration import choose
    root=_evaluation(tmp_path,[('A1','A',1,.5,['ab','ab']),('A2','A',2,.4,['abcd','abcd']),('A3','A',3,.6,['abc','abc']),('A4','A',4,.55,['a','a'])])
    choose(SimpleNamespace(evaluation=root,output_dir=tmp_path/'chosen',splits=['val']))
    chosen=json.loads((tmp_path/'chosen'/'selection.json').read_text())['A']
    assert [v['candidate'] for v in chosen['versions']]==[True,False,True,True]
    assert chosen['chosen']['name']=='A3'      # A2 reads best but never improved the pixel F1


def test_candidates_list_a_remover_every_n_steps_and_a_restorers_kept_versions(tmp_path):
    from revrender.eval.unified_restoration import candidates
    own=tmp_path/'own';own.mkdir()
    for s in (500,2000,2500,4000,4300):(own/f'step_{s}.pt').write_text('')
    run=tmp_path/'run';run.mkdir();(run/'run_manifest.json').write_text(json.dumps({'config':{'model':'DocDiff'}}))
    for d in ('step_5000_raw','step_100000_raw','step_100000_ema'):(run/d).mkdir()
    candidates(SimpleNamespace(own=[f'Ours={own}'],every=2000,run=[f'DocDiff={run}'],upstream=tmp_path,stain_upstream=None,output=tmp_path/'c.json'))
    specs=json.loads((tmp_path/'c.json').read_text())
    assert [s['name'] for s in specs]==['Ours-s2000','Ours-s4000','Ours-s4300','DocDiff-s5000','DocDiff-s100000','DocDiff-s100000ema']


def test_the_development_half_gives_pixel_cases_and_readable_ocr_crops_of_confirmed_regions_only(tmp_path):
    from PIL import Image
    from revrender.data.pixel_masks import encode
    from revrender.eval.unified_restoration import development
    page=np.full((40,80),255,np.uint8);page[10:30,10:70]=40;Image.fromarray(page).save(tmp_path/'q1_1.png')
    Image.fromarray(page).save(tmp_path/'q2_1.png')
    mask=np.zeros((20,60),bool);mask[9:11,:]=True
    base=dict(box=[10,10,70,30],region=[[10,10],[70,10],[70,30],[10,30]],runs=encode(mask),text='$x+1$')
    export={'pages':{'q1_1':{'revisions':[dict(base,id='a',confirmed=True,text_confirmed=True),dict(base,id='b',confirmed=False,text_confirmed=True,unreadable=True)]},
                     'q2_1':{'revisions':[dict(base,id='c',confirmed=True,text_confirmed=True)]}}}
    (tmp_path/'export.json').write_text(json.dumps(export));(tmp_path/'halves.json').write_text(json.dumps({'development':['q1_1'],'sealed':['q2_1']}))
    out=tmp_path/'inputs';(out/'crops').mkdir(parents=True);rows=[];cases=[];arrays={}
    development(SimpleNamespace(dev_export=tmp_path/'export.json',dev_pages=tmp_path,dev_questions=tmp_path/'halves.json'),out,rows,cases,arrays)
    assert [c['id'] for c in cases]==['q1_1#a'] and arrays['gold_0'].sum()==mask.sum()
    assert [r['line_id'] for r in rows]==['q1_1#a'] and rows[0]['split']=='dev'      # the sealed question is never read
