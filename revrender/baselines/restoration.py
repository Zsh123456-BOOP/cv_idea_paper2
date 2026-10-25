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

MODELS = ['NAFNet', 'DocDiff', 'DE-GAN', 'COR-Generator', 'Strikethrough-CycleGAN', 'Restormer', 'StainRestorer', 'RevSep']
HINTED = ('RevSep',)    # told where the revision is (data.separation): ours (models.separation), not a baseline


def upstream_of(upstream, model):
    """The official source a model is built from: RevSep's body is NAFNet's."""
    return upstream/('NAFNet' if model == 'RevSep' else model)


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


def restore_native(predict, grey, hint=None):
    """Native resolution tiling; centre short regions on the same white 256 canvas (a hinted model's hint on zeros)."""
    h,w=grey.shape;ph=max(256,h);pw=max(256,w);oy=(ph-h)//2;ox=(pw-w)//2
    padded=np.ones((ph,pw),np.float32);padded[oy:oy+h,ox:ox+w]=grey
    if hint is not None:marks=np.zeros((ph,pw),np.float32);marks[oy:oy+h,ox:ox+w]=hint
    positions=lambda n: sorted(set(list(range(0,n-255,192))+[n-256]))
    total=np.zeros_like(padded);weight=np.zeros_like(padded)
    for y in positions(ph):
        for x in positions(pw):
            tile=padded[None,y:y+256,x:x+256]
            value=(predict(tile) if hint is None else predict(tile,marks[None,y:y+256,x:x+256]))[0]
            total[y:y+256,x:x+256]+=value;weight[y:y+256,x:x+256]+=1
    return (total/weight)[oy:oy+h,ox:ox+w]


class Experiment:
    hinted=False    # a model told where the revision is (HINTED)
    w=None          # per-pixel loss weights of the data (weights.npy)

    def __init__(self,args):
        self.args=args;self.out=make_output_dir(args.output_dir)
        self.rows=json.loads((args.data/'records.json').read_text())
        self.x=np.load(args.data/(args.arm+'.npy'),mmap_mode='r');self.y=np.load(args.data/'targets.npy',mmap_mode='r')
        # Per-pixel loss weights (data.pseudo_regions: 0 where the teachers disagree); honoured by NAFNet's loss.
        self.w=np.load(args.data/'weights.npy',mmap_mode='r') if (args.data/'weights.npy').exists() else None
        self.train=np.array([i for i,r in enumerate(self.rows) if r['split']=='train'])
        self.val=np.array([i for i,r in enumerate(self.rows) if r['split']=='val'])
        rng=np.random.default_rng(SEED);self.val=rng.choice(self.val,min(args.validation_limit,len(self.val)),replace=False)
        self.rng=np.random.default_rng(SEED)
        self.native=np.load(args.human_validation.with_suffix('.npz'))
        self.native_rows=json.loads(args.human_validation.read_text())
        require(not {r['group_uuid'] for r in self.native_rows}&{r['group_uuid'] for r in self.rows},'human validation writer leaked')
        self.best=(-1,None,'');self.start=time.monotonic()
        self.stale=0;self.stop=False;self.last_step=0
        self.unpaired_rng=np.random.default_rng(np.random.SeedSequence(SEED).spawn(1)[0])
        self.candidate_improved=False
        self.source=verified_source(upstream_of(args.upstream,args.model))
        inputs=[
            args.data/'run_manifest.json',args.data/(args.arm+'.npy'),args.data/'targets.npy',args.data/'records.json',
            args.human_validation,args.human_validation.with_suffix('.npz'),upstream_of(args.upstream,args.model)/'source_manifest.json']
        self.hinted=args.model in HINTED
        if self.hinted:
            from ..data.separation import HUMAN
            self.human=np.array([r.get('source')==HUMAN for r in self.rows])
            self.stored_hints=np.load(args.data/'hint.npy',mmap_mode='r') if (args.data/'hint.npy').exists() else None
            require(self.stored_hints is not None or not self.human.any(),'human tiles without their detector hints (restoration_pairs human --detector)')
            require(args.human_validation_hints is not None or not args.hint,'the hints of the human validation are needed (data.separation hints)')
            self.native_hints=np.load(args.human_validation_hints) if args.hint else None
            self.hint_rng=np.random.default_rng(np.random.SeedSequence(SEED).spawn(2)[1])
            self.strokes=None
            if args.stroke and args.strokes:
                self.strokes=[np.load(args.strokes/f'{k}.npy',mmap_mode='r') for k in ('stroke','centre','width')]
                require(len(self.strokes[0])==len(self.rows) and json.loads((args.strokes/'run_manifest.json').read_text())['inputs'][str(args.data/'records.json')]
                        ==digest(args.data/'records.json'),'stroke geometry of other pairs')
                inputs.append(args.strokes/'run_manifest.json')
            require(not args.stroke or self.strokes is not None or self.human[self.train].all(),'synthetic pairs without their stroke geometry (data.separation strokes)')
            inputs+=[p for p in (args.data/'hint.npy',args.human_validation_hints) if p is not None and p.exists()]
        if args.protocol:inputs.append(args.protocol)
        if args.init:inputs+=[p for p in sorted(args.init.iterdir()) if p.is_file()]
        if getattr(args,'reader',None):inputs.append(args.reader)
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
        ids=self.sample()
        targets=self.unpaired_rng.choice(self.train,self.args.batch_size,replace=False) if unpaired else ids
        x,y=self.x[ids].astype(np.float32)/255,self.y[targets].astype(np.float32)/255
        if not weights:return x,y
        return x,y,(self.w[ids].astype(np.float32) if self.w is not None else np.ones_like(x))

    def hints_of(self,ids,x,y,rng):
        """The hints of pairs ``ids`` (their uint8 x, y): a human tile's detector hint jittered, a synthetic one's simulated
        (``data.separation``); zeros with --no-hint."""
        from ..data.separation import jittered_hint, simulated_hint
        if not self.args.hint:return np.zeros(x.shape,np.float32)
        return np.stack([jittered_hint(self.stored_hints[i],a,rng) if self.human[i] else
                         simulated_hint(a,b,rng,self.args.hint_reach,self.args.hint_drop) for i,a,b in zip(ids,x,y)]).astype(np.float32)

    def labelled_batch(self):
        """A batch for RevSep: grey, target and hint on [0, 1], the erase and cross labels (``data.separation.pair_labels``), its rows,
        and with --stroke the (mask, centreline, half-width) labels (``data.separation.stroke_labels``)."""
        from ..data.separation import pair_labels
        from ..data.separation import stroke_labels
        ids=self.sample();x=self.x[ids];y=self.y[ids];labels=[pair_labels(a,b,self.human[i]) for i,a,b in zip(ids,x,y)]
        strokes=None
        if self.args.stroke:
            made=[stroke_labels(a,b,self.human[i],None if self.human[i] else [s[i] for s in self.strokes]) for i,a,b in zip(ids,x,y)]
            strokes=tuple(np.stack([m[k] for m in made]) for k in range(3))
        return (x.astype(np.float32)/255,y.astype(np.float32)/255,self.hints_of(ids,x,y,self.hint_rng),
                np.stack([e for e,_ in labels]),np.stack([c for _,c in labels]),ids,strokes)

    def native_hint(self,i):
        if not self.hinted:return None
        return self.native_hints[f'hint_{i}'] if self.args.hint else np.zeros(self.native[f'image_{i}'].shape,np.uint8)

    def log(self,record):
        record={**record,'elapsed_seconds':round(time.monotonic()-self.start,2)}
        with (self.out/'events.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)

    def evaluate(self,predict,step,save,variant='',final_candidate=True):
        suffix=('_'+variant) if variant else ''
        counts=np.zeros(5,np.int64);native_out={};tiles=[]
        for i,_ in enumerate(self.native_rows):
            im=self.native[f'image_{i}'].astype(np.float32)/255;gold=self.native[f'mask_{i}']
            pred=restore_native(predict,im,self.native_hint(i));require(np.isfinite(pred).all(),'nonfinite restoration')
            counts+=removal_counts(im,pred,gold);native_out[f'image_{i}']=np.rint(pred*255).astype(np.uint8)
            canvas=Image.new('L',(768,256),255)
            for j,a in enumerate((im,pred,np.where(gold,1,im))):
                tile=Image.fromarray(np.rint(a*255).astype(np.uint8));tile.thumbnail((252,252));canvas.paste(tile,(j*256+(256-tile.width)//2,(256-tile.height)//2))
            tiles.append(np.asarray(canvas))
        sse=0.;base=0.;n=0;changed=[];preserve=[];hint_rng=np.random.default_rng(SEED)    # the same hints at every evaluation
        for start in range(0,len(self.val),self.args.batch_size):
            ids=self.val[start:start+self.args.batch_size];xu=self.x[ids];yu=self.y[ids]
            x=xu.astype(np.float32)/255;y=yu.astype(np.float32)/255
            p=predict(x,self.hints_of(ids,xu,yu,hint_rng)) if self.hinted else predict(x);require(np.isfinite(p).all(),'nonfinite synthetic restoration')
            sse+=float(((p-y)**2).sum());base+=float(((x-y)**2).sum());n+=y.size
            changed.extend(np.abs(p-y)[np.abs(x-y)>.05].tolist())
            preserve.extend(np.abs(p-y)[(y<200/255)&(np.abs(x-y)<=.05)].tolist())
        result={'step':step,'human_removal':removal_metrics(counts),'synthetic_psnr':-10*np.log10(max(sse/n,1e-12)),
                'identity_synthetic_psnr':-10*np.log10(max(base/n,1e-12)),
                'synthetic_changed_mae':float(np.mean(changed)) if changed else 0.,
                'synthetic_preserved_ink_mae':float(np.mean(preserve)) if preserve else 0.}
        score=result['human_removal']['f1']
        improved=score>self.best[0]
        self.last_step=step
        self.candidate_improved|=improved
        if final_candidate:
            self.stale=0 if self.candidate_improved else self.stale+1
            self.candidate_improved=False
        self.stop=bool(self.args.patience and step>=self.args.min_steps and self.stale>=self.args.patience)
        # Always retain the final checkpoint; intermediate checkpoints only when validation improves.
        if improved or step==self.args.steps or self.stop:
            save(self.out/f'step_{step}{suffix}')
            np.savez_compressed(self.out/f'native_{step}{suffix}.npz',**native_out)
            Image.fromarray(np.concatenate(tiles)).save(self.out/f'native_{step}{suffix}.png')
        if improved:self.best=(score,step,suffix)
        result.update(selected_step=self.best[1],variant=variant,plateau_stop=self.stop,stale_validations=self.stale)
        write_json(self.out/f'evaluation_{step}{suffix}.json',result);self.log(result)
        return result

    def finish(self):
        write_json(self.out/'selection.json',{'step':self.best[1],'human_removal_f1':self.best[0],
            'checkpoint':f'step_{self.best[1]}{self.best[2]}','evaluation':f'evaluation_{self.best[1]}{self.best[2]}.json',
            'no_test_data_used':True,'actual_steps':self.last_step,
            'stop_reason':'validation_plateau' if self.stop else 'declared_budget',
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
    p.add_argument('--ema',action='store_true')
    p.add_argument('--init',type=Path,help="fine-tune from this checkpoint directory (the protocol's [finetune] budget)")
    p.add_argument('--human-validation-hints',type=Path,help='RevSep: the detector hints of the human validation crops (data.separation hints)')
    p.add_argument('--hint',action=argparse.BooleanOptionalAction,default=True,help='RevSep: the hint channel (off: zeros)')
    p.add_argument('--reader',type=Path,help='RevSep: a frozen reader (reader_ctc) whose CTC loss on the output of the one-tile human regions joins the loss')
    p.add_argument('--reading-weight',type=float,default=0.,help='RevSep: the weight of that reading term')
    p.add_argument('--stroke',action=argparse.BooleanOptionalAction,default=False,help='RevSep: the stroke head A and its drawn mask (models.separation)')
    p.add_argument('--hint-from',choices=['detector','stroke','both'],default='detector',
                   help="RevSep --stroke: the second pass's range: none (detector: one pass), the drawn mask, or both side by side")
    p.add_argument('--strokes',type=Path,help='RevSep --stroke: the stroke geometry of the synthetic pairs (data.separation strokes)')
    p.add_argument('--heads',action=argparse.BooleanOptionalAction,default=True,
                   help='RevSep: the erase and cross heads and the composition (off: H1 alone, NAFNet with the hint channel)')
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
        args.eval_every=max(1,args.steps//rule['evaluations']);args.min_steps=0;args.patience=0
        args.learning_rate_scale=rule.get('learning_rate_scale',1.)
    require(args.steps>0 and args.batch_size>0 and args.eval_every>0,'invalid budget')
    random.seed(SEED);np.random.seed(SEED)
    experiment=Experiment(args)
    if args.model=='DE-GAN':from .restoration_tf import train
    elif args.model=='Strikethrough-CycleGAN':from .restoration_cycle import train
    else:from .restoration_torch import train
    train(experiment)


if __name__=='__main__':main()
