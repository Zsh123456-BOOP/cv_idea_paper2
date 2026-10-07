"""Compare box-remover checkpoints on untouched native human validation crops."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .. import infer
from ..data.gold import letterbox, unletterbox
from ..data.synthetic_set import read_set
from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

POINT={'mask_threshold':.4,'nms':.5,'score_threshold':0.}


def predict_crops(model,images):
    strips=[np.asarray(letterbox(Image.fromarray(im),768,96,255)) for im in images]
    delivered=infer.predict_lines(model,strips,POINT)
    masks=[]
    for image,items in zip(images,delivered):
        mask=unletterbox(items[0][0],image.shape[1],image.shape[0]) if items else np.zeros(image.shape,bool)
        masks.append(mask.astype(bool)&(image<245))
    return masks


def counts(mask,gold,image):
    target=gold.astype(bool);pred=mask.astype(bool)
    return {'tp':int((pred&target).sum()),'fp':int((pred&~target).sum()),'fn':int((~pred&target).sum()),
            'valid_ink':int(((image<245)&~target).sum())}


def metrics(rows):
    totals={k:sum(r[k] for r in rows) for k in ('tp','fp','fn','valid_ink')}
    tp,fp,fn=totals['tp'],totals['fp'],totals['fn']
    return {**totals,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn),
            'valid_ink_erased':fp/max(1,totals['valid_ink'])}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    read_set(args.dataset)  # Validate the complete data manifest, including native crops.
    records=json.loads((args.dataset/'validation_native.json').read_text())
    archive=np.load(args.dataset/'validation_native.npz',allow_pickle=False)
    images=[archive[f'image_{i}'] for i in range(len(records))];gold=[archive[f'mask_{i}'] for i in range(len(records))]
    candidates=sorted(args.checkpoints.glob('step_*.pt'),key=lambda p:int(p.stem.split('_')[-1]));require(candidates,'no checkpoints')
    out=make_output_dir(args.output_dir);readings={};inputs={str(args.dataset/'run_manifest.json'):digest(args.dataset/'run_manifest.json')}
    for name,path in [('baseline',args.baseline),*[(p.stem,p) for p in candidates]]:
        model,payload=infer.load_checkpoint(path);predictions=predict_crops(model,images)
        per_row=[{'id':r['id'],**counts(mask,g,image)} for r,mask,g,image in zip(records,predictions,gold,images)]
        readings[name]={'checkpoint':str(path),'step':payload.get('step'),'metrics':metrics(per_row),'regions':per_row}
        np.savez_compressed(out/(name+'.npz'),**{f'mask_{i}':m for i,m in enumerate(predictions)})
        inputs[str(path)]=digest(path);print(json.dumps({'model':name,**readings[name]['metrics']}),flush=True)
        del model
    base=readings['baseline']['metrics']
    eligible=[n for n,r in readings.items() if n!='baseline' and r['metrics']['precision']>=base['precision']-.02 and r['metrics']['f1']>base['f1']]
    chosen=max(eligible,key=lambda n:(readings[n]['metrics']['f1'],-readings[n]['step'])) if eligible else 'baseline'
    summary={'selected':chosen,'checkpoint':readings[chosen]['checkpoint'],'rule':'native pixel F1 improvement with precision no more than .02 below baseline; otherwise retain baseline',
             'operating_point':POINT,'top':1,'auto_cleanup_applied':False,'validation_regions':len(records),'models':readings,'test_touched':False}
    write_json(out/'summary.json',summary);write_manifest(out,config=to_jsonable(args),inputs=inputs,selected=chosen,test_touched=False)


if __name__=='__main__':main()
