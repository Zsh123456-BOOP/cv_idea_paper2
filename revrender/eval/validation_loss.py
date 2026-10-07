"""Evaluate saved remover checkpoints on their unaugmented validation set.

This is eval-mode loss: training-only denoising and sampled ROI loss are absent. The mean is weighted by
batch size, at a fixed batch size and seed for all checkpoints. It does not select
the real-data checkpoint; revision_boxes supplies that reading separately.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..data.synthetic_set import read_set
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest


def main(argv=None):
    import torch
    from ..infer import load_checkpoint
    from ..models.losses import compute_loss, materialize
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--synthetic',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,required=True,help='formal run directory containing step_*.pt')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    require(args.batch_size>0,'positive batch size required')
    records,images,labels=read_set(args.synthetic)
    indices=[i for i,r in enumerate(records) if r['split']=='val']
    require(indices,'no validation rows')
    checkpoints=sorted(args.checkpoints.glob('step_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    require(checkpoints,'no saved steps')
    out=make_output_dir(args.output_dir)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    readings={}
    for checkpoint in checkpoints:
        model,_=load_checkpoint(checkpoint)
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        total=0.;terms={}
        with torch.no_grad():
            for start in range(0,len(indices),args.batch_size):
                ids=indices[start:start+args.batch_size]
                x=torch.from_numpy(np.asarray(images[ids])[:,None].astype(np.float32)/127.5-1).cuda()
                targets=[[torch.from_numpy((labels[i]==k).astype(np.float32)).cuda()
                          for k in sorted(set(np.unique(labels[i]))-{0})] for i in ids]
                rendered,_,_=model(x,targets)
                loss,pieces=compute_loss(model,rendered,targets,point_supervision=False)
                require(bool(torch.isfinite(loss)),'nonfinite validation loss')
                total+=float(loss)*len(ids)
                for name,value in materialize(pieces).items():
                    terms[name]=terms.get(name,0.)+value*len(ids)
        readings[checkpoint.stem]={'loss':total/len(indices),'terms':{k:v/len(indices) for k,v in terms.items()}}
        print(json.dumps({'checkpoint':checkpoint.stem,'samples':len(indices),'loss':readings[checkpoint.stem]['loss']}),flush=True)
        del model
    write_json(out/'summary.json',{'scope':'held-out validation, eval mode without training-only denoising or sampled ROI loss',
                                  'sources':sorted({records[i].get('source','synthetic') for i in indices}),
                                  'samples':len(indices),'batch_size':args.batch_size,'seed':SEED,'checkpoints':readings})
    write_manifest(out,config=to_jsonable(args),inputs={str(p):digest(p) for p in [args.synthetic/'run_manifest.json',*checkpoints]})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
