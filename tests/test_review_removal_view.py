import base64
import io
import json
import re

import numpy as np
from PIL import Image

from revrender.tools.review_removal_view import build


def test_comparison_uses_native_masks_and_labels_a_failed_replacement(tmp_path):
    data=tmp_path/'data';data.mkdir();evaluation=tmp_path/'evaluation';evaluation.mkdir()
    scan=np.full((3,5),30,np.uint8);gold=np.zeros_like(scan);gold[0,0]=1
    old=gold.copy();old[1,1]=1;new=gold.copy()
    np.savez(data/'validation_native.npz',image_0=scan,mask_0=gold)
    (data/'validation_native.json').write_text(json.dumps([{'id':'example'}]))
    np.savez(evaluation/'baseline.npz',mask_0=old);np.savez(evaluation/'step_200.npz',mask_0=new)
    metrics={'precision':.9,'recall':.8,'f1':.85,'valid_ink_erased':.1}
    (evaluation/'summary.json').write_text(json.dumps({'selected':'baseline','models':{'baseline':{'metrics':metrics},'step_200':{'metrics':metrics}}}))
    out=tmp_path/'view';build(data,evaluation,out)
    html=(out/'index.html').read_text();assert '未通过替换规则' in html and 'step_200' in html
    images=[np.asarray(Image.open(io.BytesIO(base64.b64decode(s)))) for s in re.findall(r'data:image/png;base64,([^\"]+)',html)]
    assert len(images)==4 and all(im.shape==scan.shape for im in images)
    assert np.array_equal(images[0],scan)
    assert images[1][1,1]==255 and images[2][1,1]==30
    assert np.array_equal(images[2],images[3])
