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
