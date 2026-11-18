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
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image

from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

MODELS = ['NAFNet', 'DocDiff', 'DE-GAN', 'COR-Generator', 'Strikethrough-CycleGAN', 'Restormer', 'StainRestorer',
          'Strikethrough-CycleGAN-A', 'VCGAN', 'DocRes']
SOURCE = {'Strikethrough-CycleGAN-A': 'Strikethrough-CycleGAN'}   # the attribute-guided CycleGAN is built from the same release


def upstream_of(upstream, model):
    """The official source a model is built from."""
    return upstream/SOURCE.get(model, model)


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
            tile=padded[None,y:y+256,x:x+256]
            value=predict(tile)[0]
            total[y:y+256,x:x+256]+=value;weight[y:y+256,x:x+256]+=1
    return (total/weight)[oy:oy+h,ox:ox+w]


SCALE=(.8,1.25)     # --augment (plan of 2026-11-16, fixed): each training tile independently scaled, turned and its ink darkened or lightened
TURN=5.             # degrees either way
GAMMA=(.8,1.25)     # grey ** gamma: white stays white


def augmented(x,y,w,rng):
    """The tiles (grey on [0, 1]), their targets and loss weights under one random scale, turn and gamma per tile, the same for all
    three; what the turn or a shrink brings in from outside is white with no weight.  No mirror: a mirrored character does not exist."""
    from scipy import ndimage
    x,y,w=x.copy(),y.copy(),w.copy();n=x.shape[-1];centre=np.array([(n-1)/2]*2)
    for i in range(len(x)):
        s=rng.uniform(*SCALE);a=np.deg2rad(rng.uniform(-TURN,TURN));g=rng.uniform(*GAMMA)
        m=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])/s;offset=centre-m@centre
        x[i]=ndimage.affine_transform(x[i],m,offset,order=1,cval=1.)**g;y[i]=ndimage.affine_transform(y[i],m,offset,order=1,cval=1.)**g
        w[i]=ndimage.affine_transform(w[i],m,offset,order=0,cval=0.)
    return x,y,w


def inside(im,pred,gold,inner):
    """A native crop scored only inside ``inner`` (a crop with context: its region; ``None``: all of it): outside it white."""
    if inner is None:return im,pred,gold
    keep=inner.astype(bool);return np.where(keep,im,1.),np.where(keep,pred,1.),gold.astype(bool)&keep


class Experiment:
    w=None          # per-pixel loss weights of the data (weights.npy)
    ex=None         # --extra tiles: (images, targets, training indices)

    def __init__(self,args):
        self.args=args;self.out=make_output_dir(args.output_dir)
        self.rows=json.loads((args.data/'records.json').read_text())
        self.x=np.load(args.data/(args.arm+'.npy'),mmap_mode='r');self.y=np.load(args.data/'targets.npy',mmap_mode='r')
        # Per-pixel loss weights (data.pseudo_regions: 0 where the teachers disagree); honoured by NAFNet's loss.
        self.w=np.load(args.data/'weights.npy',mmap_mode='r') if (args.data/'weights.npy').exists() else None
        # Per-sample loss weights (data.pseudo_regions --pseudo-weight: a pseudo-labelled region's against a human one's).
        self.s=np.array([float(r.get('weight',1.)) for r in self.rows],np.float32)
        self.train=np.array([i for i,r in enumerate(self.rows) if r['split']=='train'])
        self.val=np.array([i for i,r in enumerate(self.rows) if r['split']=='val'])
        rng=np.random.default_rng(SEED);self.val=rng.choice(self.val,min(args.validation_limit,len(self.val)),replace=False)
        self.rng=np.random.default_rng(SEED)
        self.native=np.load(args.human_validation.with_suffix('.npz'))
        self.native_rows=json.loads(args.human_validation.read_text())
        require(not {r['group_uuid'] for r in self.native_rows}&{r['group_uuid'] for r in self.rows},'human validation writer leaked')
        self.best=(-np.inf,None,'');self.start=time.monotonic()   # the best score may be negative (--select edits or blocks: fewer the better)
        self.stale=0;self.stop=False;self.last_step=0
        self.unpaired_rng=np.random.default_rng(np.random.SeedSequence(SEED).spawn(1)[0])
        self.augment_rng=np.random.default_rng(np.random.SeedSequence(SEED).spawn(2)[1])   # its own stream: the batches drawn stay the same
        # --extra (synthesis plan of 2026-11-16, section six): each sample of a batch is one of these with --extra-share's chance, its
        # weights 1; the steps stay the data's own (its passes), so a run with and without them takes as many steps.
        if getattr(args,'extra',None):
            ex_rows=json.loads((args.extra/'records.json').read_text())
            require(not {r['group_uuid'] for r in ex_rows}&{r['group_uuid'] for r in self.native_rows},'human validation writer among the extra hosts')
            self.ex=(np.load(args.extra/(args.arm+'.npy'),mmap_mode='r'),np.load(args.extra/'targets.npy',mmap_mode='r'),
                     np.array([i for i,r in enumerate(ex_rows) if r['split']=='train']))
            self.extra_rng=np.random.default_rng(np.random.SeedSequence(SEED).spawn(3)[2])
        self.candidate_improved=False
        self.source=verified_source(upstream_of(args.upstream,args.model))
        inputs=[
            args.data/'run_manifest.json',args.data/(args.arm+'.npy'),args.data/'targets.npy',args.data/'records.json',
            args.human_validation,args.human_validation.with_suffix('.npz'),upstream_of(args.upstream,args.model)/'source_manifest.json']
        if args.protocol:inputs.append(args.protocol)
        if args.init:inputs+=[p for p in sorted(args.init.iterdir()) if p.is_file()]
        if getattr(args,'extra',None):inputs+=[args.extra/'run_manifest.json',args.extra/'records.json']
        require(set(r['split'] for r in self.rows)<= {'train','val'},'test row in training bundle')
        require(not {r['group_uuid'] for r in self.rows if r['split']=='train'} &
                {r['group_uuid'] for r in self.rows if r['split']=='val'},'synthetic validation writer leaked')
        write_manifest(self.out,config=to_jsonable(args),inputs={str(p):digest(p) for p in inputs},
            upstream=self.source,selection='maximum removal F1 on 16 independent human regions; threshold 0.2 fixed',weighted=self.w is not None,
            synthetic_validation_indices=self.val.tolist(),limitations='human labels do not provide true clean images or hidden-stroke truth',
            adaptation='native 256 square tiles; image-only; synthetic-only; paired supervision except independent clean-domain sampling for CycleGAN; task adaptation, not original benchmark scores')

    def sample(self):
        return self.rng.choice(self.train,self.args.batch_size,replace=False)

    def batch(self,unpaired=False,weights=False):
        ids=self.sample();self.last_ids=ids
        targets=self.unpaired_rng.choice(self.train,self.args.batch_size,replace=False) if unpaired else ids
        x,y=self.x[ids].astype(np.float32)/255,self.y[targets].astype(np.float32)/255
        w=self.w[ids].astype(np.float32) if self.w is not None and weights else np.ones_like(x);s=self.s[ids].copy() if weights else None
        if self.ex is not None:
            require(not unpaired,'--extra with unpaired targets')
            pick=self.extra_rng.random(len(ids))<self.args.extra_share;n=int(pick.sum())
            if n:
                ex=np.sort(self.extra_rng.choice(self.ex[2],n,replace=False))
                x[pick]=self.ex[0][ex].astype(np.float32)/255;y[pick]=self.ex[1][ex].astype(np.float32)/255;w[pick]=1.
                if s is not None:s[pick]=1.
        if not weights:return x,y
        if getattr(self.args,'augment',False):x,y,w=augmented(x,y,w,self.augment_rng)
        return x,y,w,s

    def log(self,record):
        record={**record,'elapsed_seconds':round(time.monotonic()-self.start,2)}
        with (self.out/'events.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)

    def evaluate(self,predict,step,save,variant='',final_candidate=True):
        suffix=('_'+variant) if variant else ''
        from ..eval.prefill import edits as edit_counts
        counts=np.zeros(5,np.int64);native_out={};tiles=[];edits=[];blocks=[]
        for i,_ in enumerate(self.native_rows):
            im=self.native[f'image_{i}'].astype(np.float32)/255;gold=self.native[f'mask_{i}']
            pred=restore_native(predict,im);require(np.isfinite(pred).all(),'nonfinite restoration')
            im,pred,gold=inside(im,pred,gold,self.native[f'inner_{i}'] if f'inner_{i}' in self.native else None)
            count=removal_counts(im,pred,gold);counts+=count;native_out[f'image_{i}']=np.rint(pred*255).astype(np.uint8)
            edits.append((count[1]+count[2])/max(1,count[0]+count[2]))   # pixels to fix (missed + writing erased) over the strike's
            blocks.append(edit_counts(im,gold,pred)['blocks'])            # blocks to fix: about one stroke of the eraser or brush each
            canvas=Image.new('L',(768,256),255)
            for j,a in enumerate((im,pred,np.where(gold,1,im))):
                tile=Image.fromarray(np.rint(a*255).astype(np.uint8));tile.thumbnail((252,252));canvas.paste(tile,(j*256+(256-tile.width)//2,(256-tile.height)//2))
            tiles.append(np.asarray(canvas))
        sse=0.;base=0.;n=0;changed=[];preserve=[]
        for start in range(0,len(self.val),self.args.batch_size):
            ids=self.val[start:start+self.args.batch_size];xu=self.x[ids];yu=self.y[ids]
            x=xu.astype(np.float32)/255;y=yu.astype(np.float32)/255
            p=predict(x);require(np.isfinite(p).all(),'nonfinite synthetic restoration')
            sse+=float(((p-y)**2).sum());base+=float(((x-y)**2).sum());n+=y.size
            changed.extend(np.abs(p-y)[np.abs(x-y)>.05].tolist())
            preserve.extend(np.abs(p-y)[(y<200/255)&(np.abs(x-y)<=.05)].tolist())
        result={'step':step,'human_removal':removal_metrics(counts),'human_edits':float(np.mean(edits)),'human_blocks':float(np.mean(blocks)),
                'synthetic_psnr':-10*np.log10(max(sse/n,1e-12)) if n else None,
                'identity_synthetic_psnr':-10*np.log10(max(base/n,1e-12)) if n else None,
                'synthetic_changed_mae':float(np.mean(changed)) if changed else 0.,
                'synthetic_preserved_ink_mae':float(np.mean(preserve)) if preserve else 0.}
        # by the removal F1 of the human validation, or (the prefill model) by its pixels or blocks to fix, fewer the better
        select=getattr(self.args,'select','f1')
        score={'edits':-result['human_edits'],'blocks':-result['human_blocks']}.get(select,result['human_removal']['f1'])
        # a version is kept when it beats the best so far; it counts against the patience unless it beats it by --min-delta
        improved=score>self.best[0]
        self.last_step=step
        self.candidate_improved|=score>self.best[0]+getattr(self.args,'min_delta',0.)
        if final_candidate:
            self.stale=0 if self.candidate_improved else self.stale+1
            self.candidate_improved=False
        self.stop=bool(self.args.patience and step>=self.args.min_steps and self.stale>=self.args.patience)
        # Always retain the final checkpoint; intermediate checkpoints only when validation improves.
        if improved or step==self.args.steps or self.stop or getattr(self.args,'keep_all',False):
            save(self.out/f'step_{step}{suffix}')
            np.savez_compressed(self.out/f'native_{step}{suffix}.npz',**native_out)
            Image.fromarray(np.concatenate(tiles)).save(self.out/f'native_{step}{suffix}.png')
        if improved:self.best=(score,step,suffix)
        result.update(selected_step=self.best[1],variant=variant,plateau_stop=self.stop,stale_validations=self.stale)
        write_json(self.out/f'evaluation_{step}{suffix}.json',result);self.log(result)
        return result

    def finish(self):
        write_json(self.out/'selection.json',{'step':self.best[1],'human_removal_f1':self.best[0],'selected_by':getattr(self.args,'select','f1'),
            'checkpoint':f'step_{self.best[1]}{self.best[2]}','evaluation':f'evaluation_{self.best[1]}{self.best[2]}.json',
            'no_test_data_used':True,'actual_steps':self.last_step,
            'stop_reason':'validation_plateau' if self.stop else 'declared_budget',
            'selected_at_last_evaluation':self.best[1]==self.last_step,
            'training_examples_seen':self.last_step*self.args.batch_size})
        self.log({'status':'completed','selected_step':self.best[1]})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--arm',choices=['gold_pixels','predicted_pixels','predicted_traced'],required=True)
    for name in ('data','upstream','human-validation','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--validation-limit',type=int,default=64)
    p.add_argument('--min-steps',type=int,default=0)
    p.add_argument('--patience',type=int,default=0)
    p.add_argument('--min-delta',type=float,default=0.,help='a validation counts against the patience unless it beats the best by this much')
    p.add_argument('--keep-all',action='store_true',help='keep the weights of every validation, not only the improving ones')
    p.add_argument('--ema',action='store_true')
    p.add_argument('--augment',action='store_true',help='each weighted training tile scaled 0.8-1.25, turned 5 degrees either way, ink gamma 0.8-1.25')
    p.add_argument('--select',choices=['f1','edits','blocks'],default='f1',
                   help="keep a version when the human validation's removal F1 rises, or its pixels (edits) or blocks to fix fall")
    p.add_argument('--extra',type=Path,help='more training tiles (synthetic): each sample drawn from them with --extra-share chance; steps unchanged')
    p.add_argument('--extra-share',type=float,default=.5)
    p.add_argument('--init',type=Path,help="fine-tune from this checkpoint directory (the protocol's [finetune] budget)")
    p.add_argument('--budget',help="a budget section of the protocol (passes over the data, evaluations, learning-rate scale): "
                   "default [finetune] with --init, the model's own [training] budget without")
    pre,_=p.parse_known_args(argv)
    if pre.protocol:
        protocol=tomllib.loads(pre.protocol.read_text())
        p.set_defaults(**protocol['training'][pre.model])
    args=p.parse_args(argv)
    args.learning_rate_scale=1.
    budget=args.budget or ('finetune' if args.init else None)
    if budget:
        require(pre.protocol is not None and budget in protocol,f'the protocol has no [{budget}] budget')
        rule=protocol[budget];rows=json.loads((args.data/'records.json').read_text())
        args.steps=max(rule['evaluations'],round(rule['passes']*sum(r['split']=='train' for r in rows)/args.batch_size))
        args.eval_every=max(1,args.steps//rule['evaluations'])
        # a budget may stop early: no sooner than its min_passes (a whole number of validations), after `patience` validations short of the
        # best by min_delta
        args.min_steps=args.eval_every*round(rule['evaluations']*rule.get('min_passes',0)/rule['passes'])
        args.patience=rule.get('patience',0);args.min_delta=rule.get('min_delta',0.)
        args.learning_rate_scale=rule.get('learning_rate_scale',1.)
    require(args.steps>0 and args.batch_size>0 and args.eval_every>0,'invalid budget')
    random.seed(SEED);np.random.seed(SEED)
    experiment=Experiment(args)
    if args.model=='DE-GAN':from .restoration_tf import train
    elif args.model in ('Strikethrough-CycleGAN','Strikethrough-CycleGAN-A'):from .restoration_cycle import train
    elif args.model=='VCGAN':from .restoration_vcgan import train
    else:from .restoration_torch import train
    train(experiment)


if __name__=='__main__':main()
