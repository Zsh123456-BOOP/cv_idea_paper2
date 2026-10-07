import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from revrender import infer
from revrender.eval import review_removal as E
from revrender.tools.annotate import pixel_regions as R
from revrender.util import RefusedError


def test_prefill_respects_human_polygon_and_never_overwrites_pixel_labels(tmp_path,monkeypatch):
    root=tmp_path/'pack';root.mkdir();Image.fromarray(np.full((10,10),30,np.uint8)).save(root/'p.png')
    pack={'id':'frames','pages':[{'id':'p','image':'p.png','question':1,'size':[10,10]}],'rows':[]}
    (root/'pack.json').write_text(json.dumps(pack));checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'model')
    region={'id':'p#human','box':[0,0,10,10],'region':[[0,0],[10,0],[10,3],[3,3],[3,10],[0,10]],'runs':[],'confirmed':False}
    export={'schema':R.SCHEMA,'pack':'frames','task':'region_selection','pages':{'p':{'done':True,'revisions':[region]}}}
    path=tmp_path/'frames.json';path.write_text(json.dumps(export))
    monkeypatch.setattr(infer,'load_checkpoint',lambda p:(None,{}))
    seen=[]
    def predict(model,images):
        seen.extend(images);return [im<245 for im in images]
    monkeypatch.setattr(E,'predict_crops',predict)
    args=SimpleNamespace(pack=root,regions_export=path,checkpoint=checkpoint,output_dir=tmp_path/'out')
    R.prefill(args)
    result=json.loads((args.output_dir/'pack.json').read_text());r=result['rows'][0]
    assert seen[0][8,8]==255 and seen[0][1,1]==30
    assert r['region']==region['region'] and r['id']==region['id'] and not r['confirmed']
    assert sum(n for _,n in r['runs'])==51
    assert result['task']=='strike_only' and result['id']!='frames'
    export['pages']['p']['revisions'][0]['runs']=[[1,1]];path.write_text(json.dumps(export));args.output_dir=tmp_path/'refused'
    with pytest.raises(RefusedError,match='overwrite'):R.prefill(args)


def test_native_metrics_separate_missed_strikes_from_erased_valid_ink():
    image=np.array([[20,30,40,255]],np.uint8);gold=np.array([[1,1,0,0]],bool);mask=np.array([[1,0,1,0]],bool)
    metrics=E.metrics([E.counts(mask,gold,image)])
    assert metrics['precision']==metrics['recall']==metrics['f1']==.5
    assert metrics['valid_ink_erased']==1
