"""Shared, writer-isolated task adaptation of official restoration baselines.

This is a budget-limited strike-restoration experiment, not a claim to reproduce
the papers' original benchmark scores. Native human labels supervise removal
only; they cannot certify invented/reconstructed intersection pixels.
"""
import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest


def verified_source(path):
    record=json.loads((path/'source_manifest.json').read_text())
    for name, value in record['files'].items():
        require(digest(path/name)==value, f'changed upstream source: {name}')
    return record


def module_file(name, path):
    import sys
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def removal_counts(image, restored, gold):
    ink=image<200/255
    removed=(restored-image>.2)&ink
    truth=gold.astype(bool)&ink
    return np.array([(removed&truth).sum(),(removed&~truth).sum(),(~removed&truth).sum(),
                     (~truth&ink).sum(),((image-restored>.2)&(image>=200/255)).sum()],np.int64)


def removal_metrics(count):
    tp,fp,fn,valid,added=map(int,count)
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn),valid_ink_damage=fp/max(1,valid),
                new_dark_pixels=added,threshold=.2)


def restore_native(predict, grey):
    """Native resolution tiling; centre short regions on the same white 256 canvas."""
    h,w=grey.shape;ph=max(256,h);pw=max(256,w);oy=(ph-h)//2;ox=(pw-w)//2
    padded=np.ones((ph,pw),np.float32);padded[oy:oy+h,ox:ox+w]=grey
    positions=lambda n: sorted(set(list(range(0,n-255,192))+[n-256]))
    total=np.zeros_like(padded);weight=np.zeros_like(padded)
    for y in positions(ph):
        for x in positions(pw):
            value=predict(padded[None,y:y+256,x:x+256])[0]
            total[y:y+256,x:x+256]+=value;weight[y:y+256,x:x+256]+=1
    return (total/weight)[oy:oy+h,ox:ox+w]


class Experiment:
    def __init__(self,args):
        self.args=args;self.out=make_output_dir(args.output_dir)
        self.rows=json.loads((args.data/'records.json').read_text())
        self.x=np.load(args.data/(args.arm+'.npy'),mmap_mode='r');self.y=np.load(args.data/'targets.npy',mmap_mode='r')
        self.train=np.array([i for i,r in enumerate(self.rows) if r['split']=='train'])
        self.val=np.array([i for i,r in enumerate(self.rows) if r['split']=='val'])
        rng=np.random.default_rng(SEED);self.val=rng.choice(self.val,min(args.validation_limit,len(self.val)),replace=False)
        self.rng=np.random.default_rng(SEED)
        self.native=np.load(args.human_validation.with_suffix('.npz'))
        self.native_rows=json.loads(args.human_validation.read_text())
        require(not {r['group_uuid'] for r in self.native_rows}&{r['group_uuid'] for r in self.rows},'human validation writer leaked')
        self.best=(-1,None);self.start=time.monotonic()
        self.source=verified_source(args.upstream/args.model)
        write_manifest(self.out,config=to_jsonable(args),inputs={str(p):digest(p) for p in [
            args.data/'run_manifest.json',args.data/(args.arm+'.npy'),args.data/'targets.npy',args.data/'records.json',
            args.human_validation,args.human_validation.with_suffix('.npz'),args.upstream/args.model/'source_manifest.json']},
            upstream=self.source,selection='maximum removal F1 on 16 independent human regions; threshold 0.2 fixed',
            synthetic_validation_indices=self.val.tolist(),limitations='human labels do not provide true clean images or hidden-stroke truth',
            adaptation='256 square grayscale (RGB repeated for DocDiff/NAFNet); same sampled batches across models/arms; shortened training budget')

    def batch(self):
        ids=self.rng.choice(self.train,self.args.batch_size,replace=False)
        return self.x[ids].astype(np.float32)/255,self.y[ids].astype(np.float32)/255

    def log(self,record):
        record={**record,'elapsed_seconds':round(time.monotonic()-self.start,2)}
        with (self.out/'events.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)

    def evaluate(self,predict,step,save):
        counts=np.zeros(5,np.int64);native_out={};tiles=[]
        for i,_ in enumerate(self.native_rows):
            im=self.native[f'image_{i}'].astype(np.float32)/255;gold=self.native[f'mask_{i}']
            pred=restore_native(predict,im);require(np.isfinite(pred).all(),'nonfinite restoration')
            counts+=removal_counts(im,pred,gold);native_out[f'image_{i}']=np.rint(pred*255).astype(np.uint8)
            canvas=Image.new('L',(768,256),255)
            for j,a in enumerate((im,pred,np.where(gold,1,im))):
                tile=Image.fromarray(np.rint(a*255).astype(np.uint8));tile.thumbnail((252,252));canvas.paste(tile,(j*256+(256-tile.width)//2,(256-tile.height)//2))
            tiles.append(np.asarray(canvas))
        sse=0.;base=0.;n=0;changed=[];preserve=[]
        for start in range(0,len(self.val),self.args.batch_size):
            ids=self.val[start:start+self.args.batch_size];x=self.x[ids].astype(np.float32)/255;y=self.y[ids].astype(np.float32)/255
            p=predict(x);require(np.isfinite(p).all(),'nonfinite synthetic restoration')
            sse+=float(((p-y)**2).sum());base+=float(((x-y)**2).sum());n+=y.size
            changed.extend(np.abs(p-y)[np.abs(x-y)>.05].tolist())
            preserve.extend(np.abs(p-y)[(y<200/255)&(np.abs(x-y)<=.05)].tolist())
        result={'step':step,'human_removal':removal_metrics(counts),'synthetic_psnr':-10*np.log10(max(sse/n,1e-12)),
                'identity_synthetic_psnr':-10*np.log10(max(base/n,1e-12)),
                'synthetic_changed_mae':float(np.mean(changed)) if changed else 0.,
                'synthetic_preserved_ink_mae':float(np.mean(preserve)) if preserve else 0.}
        score=result['human_removal']['f1']
        improved=score>self.best[0]
        # Always retain the final checkpoint; intermediate checkpoints only when validation improves.
        if improved or step==self.args.steps:
            save(self.out/f'step_{step}')
            np.savez_compressed(self.out/f'native_{step}.npz',**native_out)
            Image.fromarray(np.concatenate(tiles)).save(self.out/f'native_{step}.png')
        if improved:self.best=(score,step)
        result['selected_step']=self.best[1];write_json(self.out/f'evaluation_{step}.json',result);self.log(result)
        return result

    def finish(self):
        write_json(self.out/'selection.json',{'step':self.best[1],'human_removal_f1':self.best[0],
            'checkpoint':f'step_{self.best[1]}','no_test_data_used':True})
        self.log({'status':'completed','selected_step':self.best[1]})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['NAFNet','DocDiff','DE-GAN'],required=True)
    p.add_argument('--arm',choices=['gold_pixels','predicted_pixels','predicted_traced'],required=True)
    for name in ('data','upstream','human-validation','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--validation-limit',type=int,default=64)
    args=p.parse_args(argv);require(args.steps>0 and args.batch_size>0 and args.eval_every>0,'invalid budget')
    random.seed(SEED);np.random.seed(SEED)
    experiment=Experiment(args)
    if args.model=='DE-GAN':from .restoration_tf import train
    else:from .restoration_torch import train
    train(experiment)


if __name__=='__main__':main()
