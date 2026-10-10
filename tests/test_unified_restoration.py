import numpy as np
import pytest

from revrender.baselines.restoration import removal_counts
from revrender.eval.unified_restoration import freeze_specs, restore_case, verify_files
from revrender.tools.unified_restoration_report import audit
from revrender.util import RefusedError


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
    with pytest.raises(RefusedError,match='missing or extra'):audit(rows,[],[])
    with pytest.raises(RefusedError,match='duplicate'):audit(rows,[dict(id='a'),dict(id='a')],[])


def test_report_requires_same_labels_even_when_counts_match():
    rows=[]
    for split,count in [('val',41),('test',81)]:
        for name in ('scan','candidate'):
            rows.extend(dict(id=f'{split}|{name}|{i}',split=split,model=split+'__'+name,
                             line_id=str(i),kind='line',text='x',size=[20,10]) for i in range(count))
    readings=[dict(id=r['id'],condition='erased_neutral') for r in rows]
    audit(rows,readings,[dict(name='candidate')])
    rows[-1]['text']='different'
    with pytest.raises(RefusedError,match='different OCR examples'):audit(rows,readings,[dict(name='candidate')])
