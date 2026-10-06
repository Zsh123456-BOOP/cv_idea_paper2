"""Train a click-conditioned correction model on synthetic erase targets only."""
import argparse
import json
import random
import time
import tomllib
from pathlib import Path

import numpy as np

from .data.synthetic_set import read_set
from .data.prior_annotations import twins, writer_splits
from .refine import simulate
from .util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest


def dataset(path, mapping, same_student):
    records, images, labels = read_set(path)
    splits = writer_splits(mapping, same_student)
    other = twins(same_student)
    require(all(splits.get(w,'unused') in ('train','unused') and splits.get(other.get(w),'unused') in ('train','unused')
                for w in {r['group_uuid'] for r in records}), 'validation or test writer in training corpus')
    train = [i for i,r in enumerate(records) if r['split'] == 'train']
    val = [i for i,r in enumerate(records) if r['split'] == 'val']
    require(train and val, 'need synthetic training and validation writers')
    held={records[i]['group_uuid'] for i in val}
    held|={other[w] for w in list(held) if w in other}
    require(not ({records[i]['group_uuid'] for i in train} & held), 'synthetic writer overlap')
    return images, labels, train, val


def reviewed(paths, selection, folders, mapping, same_student, excluded=()):
    """Completed, explicitly confirmed training crops, including rejected detections.

    Keep native masks until the same strip transform used by synthetic training.
    Synthetic validation host writers stay excluded from this training supplement.
    """
    from PIL import Image
    from . import pages as pages_module
    from .data.pixel_masks import SCHEMA, decode
    from .data.synthetic_set import strip
    selected={p['image_id']:p for p in json.loads(selection.read_text())['pages']}
    splits=writer_splits(mapping,same_student)
    other=twins(same_student)
    excluded=set(excluded)
    excluded|={other[w] for w in list(excluded) if w in other}
    states={}
    for path in paths:
        payload=json.loads(path.read_text())
        require(payload.get('schema')==SCHEMA,'review training requires the pixel editor export')
        states.update(payload['pages'])
    states={key:q for key,q in states.items() if q.get('done')}
    require(states,'finish and confirm at least one training question')
    for key,q in states.items():
        writer=key.split('_')[0]
        require(selected.get(key,{}).get('split')=='train','review contains a non-training question')
        require(splits.get(writer,'unused') in ('train','unused') and splits.get(other.get(writer),'unused') in ('train','unused')
                and writer not in excluded,'held-out writer in review training')
        require(all(r.get('confirmed') for r in q['revisions']),'reviewed question contains a draft')
    files=pages_module.page_files(folders,list(states))
    images,labels=[],[]
    for key,q in states.items():
        with Image.open(files[key]) as opened:
            page=opened.convert('L')
        for r in q['revisions']:
            l,t,right,bottom=r['box']
            require(all(type(v) is int for v in r['box']) and 0<=l<right<=page.width and 0<=t<bottom<=page.height,'review crop outside scan')
            mask=decode(r['runs'],(bottom-t,right-l)).astype(np.uint8)
            if r.get('not_revision'):
                mask[:]=0
            image,target=strip(page.crop(tuple(r['box'])),mask)
            images.append(image);labels.append(target)
    require(images,'completed questions contain no reviewed regions')
    return np.stack(images),np.stack(labels)


def batch(images, labels, indices, rng, device):
    import torch
    items = [simulate(np.asarray(images[i]), np.asarray(labels[i]) > 0, rng)[0] for i in indices]
    x = torch.from_numpy(np.stack(items)).to(device)
    y = torch.from_numpy((np.asarray(labels[indices]) > 0).astype(np.float32)[:,None]).to(device)
    return x, y


def losses(logits, target, inputs):
    import torch
    from torch.nn import functional as F
    weight = 1+3*target+2*(inputs[:,0:1] < .8)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*weight).mean()
    p = logits.sigmoid()
    dice = 1-((2*(p*target).sum((1,2,3))+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1)).mean()
    return bce+dice


def validate(model, images, labels, indices, batch_size, device):
    import torch
    rng = np.random.default_rng(SEED)
    counts = np.zeros(5, np.float64)
    loss_sum=0.
    model.eval()
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            x,y = batch(images,labels,indices[start:start+batch_size],rng,device)
            logits=model(x)
            loss_sum+=float(losses(logits,y,x))*len(x)
            p = (logits.sigmoid() >= .5) & (x[:,0:1] < .93)
            truth = y.bool()
            prior = x[:,1:2] > .5
            counts += [int((p & truth).sum()), int(p.sum()), int(truth.sum()), int((prior != truth).sum()), int((p != truth).sum())]
    hit, predicted, gold, before, after = counts
    return {'loss':loss_sum/max(1,len(indices)), 'precision': hit/max(1,predicted), 'recall': hit/max(1,gold), 'f1': 2*hit/max(1,predicted+gold),
            'error_pixels_before': int(before), 'error_pixels_after': int(after), 'samples': len(indices)}


def main(argv=None):
    import torch
    from .models.stroke_refiner import StrokeRefiner
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--synthetic',type=Path,required=True)
    parser.add_argument('--mapping',type=Path,required=True)
    parser.add_argument('--same-student',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,default=Path('configs/stroke_refiner.toml'))
    parser.add_argument('--init',type=Path,help='continue from a click-refiner checkpoint; optimizer starts fresh')
    parser.add_argument('--review-export',type=Path,action='append',default=[],help='confirmed pixel review export of training questions')
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--pages',type=Path,action='append',default=[])
    parser.add_argument('--review-share',type=float,default=.25,help='fraction of each batch from reviewed real crops')
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(argv)
    recipe = tomllib.loads(args.recipe.read_text())
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.cuda.is_available(),'CUDA is required for training')
    images,labels,train,val = dataset(args.synthetic,args.mapping,args.same_student)
    real=None
    if args.review_export:
        require(args.init and args.selection and args.pages,'review fine-tuning needs --init, --selection and --pages')
        require(0<args.review_share<1 and recipe['batch_size']>=2,'review batches must mix real and synthetic crops')
        records=json.loads((args.synthetic/'records.json').read_text())
        excluded={records[i]['group_uuid'] for i in val}
        real=reviewed(args.review_export,args.selection,args.pages,args.mapping,args.same_student,excluded)
    rng = np.random.default_rng(SEED)
    chosen = np.random.default_rng(SEED).choice(val,size=min(recipe['validation_samples'],len(val)),replace=False).tolist()
    out = make_output_dir(args.output_dir)
    inputs=[args.recipe,args.mapping,args.same_student,args.synthetic/'run_manifest.json',*args.review_export]
    inputs += [p for p in [args.init,args.selection] if p]
    write_manifest(out,config={**to_jsonable(args),'recipe_values':recipe},inputs={str(p):digest(p) for p in inputs},
                   training_samples=len(train),reviewed_samples=len(real[0]) if real else 0,validation_samples=len(chosen),real_evaluation=False)
    model=StrokeRefiner().cuda()
    if args.init:
        checkpoint=torch.load(args.init,map_location='cpu',weights_only=False)
        require(checkpoint.get('model')=='StrokeRefiner','--init must be a click-refiner checkpoint')
        model.load_state_dict(checkpoint['state_dict'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=1e-4)
    best=-1.; started=time.monotonic()
    with (out/'log.jsonl').open('x') as log:
        for step in range(1,recipe['steps']+1):
            model.train()
            ids=rng.choice(train,size=recipe['batch_size'],replace=True)
            x,y=batch(images,labels,ids,rng,'cuda')
            if real:
                count=min(recipe['batch_size']-1,max(1,round(recipe['batch_size']*args.review_share)))
                rx,ry=batch(*real,rng.choice(len(real[0]),size=count,replace=True),rng,'cuda')
                x[:count],y[:count]=rx,ry
            optimizer.zero_grad(set_to_none=True)
            loss=losses(model(x),y,x)
            require(bool(torch.isfinite(loss)),'nonfinite refinement loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
            if step % 50 == 0:
                entry={'step':step,'loss':float(loss.detach()),'seconds':round(time.monotonic()-started,1)}
                log.write(json.dumps(entry)+'\n'); log.flush()
                print(json.dumps(entry),flush=True)
            if step % recipe['validate_every'] == 0 or step == recipe['steps']:
                metrics=validate(model,images,labels,chosen,recipe['batch_size'],'cuda')
                entry={'step':step,'synthetic_validation':metrics}
                log.write(json.dumps(entry)+'\n');log.flush()
                payload={'state_dict':model.state_dict(),'step':step,'metrics':metrics,'model':'StrokeRefiner','seed':SEED}
                torch.save(payload,out/f'step_{step}.pt')
                if metrics['f1']>best:
                    best=metrics['f1']; best_step=step; best_metrics=metrics
        write_json(out/'complete.json',{'best_step':best_step,'synthetic_validation':best_metrics,
                    'seconds':round(time.monotonic()-started,1),'real_quality':'requires human review'})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
