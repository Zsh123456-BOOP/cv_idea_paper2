"""Check selected inference reload against the saved native validation prediction."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..util import digest, require, write_json
from .restoration import HINTED, restore_native
from .restoration_export import selected_predictor


def verify(run,upstream):
    predict,checkpoint,manifest=selected_predictor(run,upstream)
    selection=json.loads((run/'selection.json').read_text())
    data=np.load(Path(manifest['config']['human_validation']).with_suffix('.npz'))
    key=selection['checkpoint'].replace('step_','native_',1)+'.npz'
    expected=np.load(run/key)['image_0']
    config=manifest['config'];hint=None
    if config['model'] in HINTED and config['hint']:hint=np.load(config['human_validation_hints'])['hint_0']
    actual=np.rint(restore_native(predict,data['image_0'].astype(np.float32)/255,hint)*255).astype(np.uint8)
    error=int(np.abs(actual.astype(int)-expected.astype(int)).max())
    require(error<=1,f'selected checkpoint inference differs: {error} grayscale levels')
    write_json(run/'inference_verified.json',{'checkpoint':str(checkpoint),'sha256':digest(checkpoint),
               'validation_image':0,'max_uint8_error':error,'test_data_read':False})
    print(json.dumps({'status':'verified','max_uint8_error':error}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    a=p.parse_args();verify(a.run,a.upstream)


if __name__=='__main__':main()
