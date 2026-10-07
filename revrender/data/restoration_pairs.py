"""Matched clean/struck image pairs and prediction-derived stroke material for restoration baselines.

Original clean hosts are targets. Model predictions are material candidates, never clean-image gold.
All arms share host, crop, placement, split and seed; only the stroke bank differs.
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .. import pages
from ..tighten import non_white
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .clean_pages import review_clean
from .handwriting import group_key
from .lines import crop_box
from .paste_strikes import Bank, gold_piece
from .prior_annotations import twins, writer_splits
from .strike_bank import strike, window, ink_level, sheet

ARMS = ('gold_pixels', 'predicted_pixels', 'predicted_traced')
SIZE = 256


def canonical(writer, other):
    return min(writer, other.get(writer, writer))


def draw_pair(clean, alpha, level):
    """Alpha compositing with the untouched clean host as ground truth, including crossings."""
    x = np.minimum(clean,np.rint(clean.astype(np.float32)*(1-alpha)+float(level)*alpha).clip(0,255).astype(np.uint8))
    return x, clean.copy()


def prediction_material(args, excluded, other, splits, out):
    """Filter once; keep exactly the same source IDs and canvas in pixel/traced banks."""
    accepted, rejected = [], Counter()
    arrays = {k:{} for k in ('pixels','traced')}
    paths = sorted(p for p in args.detections.glob('*.json') if p.stem not in ('run_manifest','summary'))
    allowed = [p for p in paths if splits.get(p.stem.split('_')[0], 'unused') in ('train','unused')
               and splits.get(other.get(p.stem.split('_')[0]), 'unused') in ('train','unused')
               and canonical(p.stem.split('_')[0], other) not in excluded]
    scans = pages.page_files(args.pages, [p.stem for p in allowed])
    inputs = {}
    for path in allowed:
        record, instances = pages.read_output(args.detections,path.stem)
        with Image.open(scans[path.stem]) as im: grey = review_clean(np.asarray(im.convert('L')))
        inputs[str(path)] = digest(path)
        inputs[str(path.with_suffix('.npz'))] = digest(path.with_suffix('.npz'))
        inputs[str(scans[path.stem])] = digest(scans[path.stem])
        for k, inst in enumerate(instances):
            if inst['score'] < args.min_score:
                rejected['low_score'] += 1; continue
            made = window(grey,inst)
            if made is None: continue
            part, mask, origin = made
            line = record['lines'][inst['line']]['box']; height=line[3]-line[1]
            pixel, inside, measure = strike(part,mask,height,'pixels')
            if pixel is None:
                rejected[measure['reason']] += 1; continue
            # The proposed method: predicted mask intersected with ink, not rectangular copying.
            intersect = mask & non_white(part)
            fill = float(intersect.sum())/max(1,measure['length']*measure['width'])
            if fill > 1.6:
                rejected['whole_mask_contains_extra_ink'] += 1; continue
            traced, ti, tm = strike(part,mask,height,'traced')
            if traced is None: continue
            px=np.where(intersect,part,255).astype(np.uint8)
            redraw=np.full(part.shape,255,np.uint8)
            tx,ty=tm['at'];redraw[ty:ty+traced.shape[0],tx:tx+traced.shape[1]]=traced
            support=(px<255)|(redraw<255);ys,xs=np.nonzero(support)
            if not len(xs): continue
            l,t,r,b=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
            # This horizontal-strike bank must not stretch vertical fragments into
            # fake cancellation strokes (observed in the material contact sheet).
            if r-l < 2.5*(b-t):
                rejected['not_horizontal_elongated']+=1;continue
            px=px[t:b,l:r];redraw=redraw[t:b,l:r]
            name=f'piece_{len(accepted)}';arrays['pixels'][name]=px;arrays['traced'][name]=redraw
            accepted.append({'id':name,'image_id':path.stem,'instance':k,'writer':canonical(path.stem.split('_')[0],other),
                             'kind':'line','size':[r-l,b-t],'line_height':height,'score':inst['score'],
                             'ink_level':ink_level(px,px<255),'pixels':int((px<255).sum()),
                             'box':[origin[0]+l,origin[1]+t,origin[0]+r,origin[1]+b],**{k:v for k,v in measure.items() if k!='at'}})
    require(len(accepted)>=30,'too few prediction-derived stroke candidates')
    for mode in arrays:
        dest=make_output_dir(out/f'bank_predicted_{mode}');np.savez_compressed(dest/'bank.npz',**arrays[mode]);write_json(dest/'pieces.json',accepted)
        chosen=random.Random(SEED).sample(accepted,min(48,len(accepted)))
        sheet([arrays[mode][p['id']] for p in chosen],[f"s{p['score']:.2f} f{p['fill']}" for p in chosen],dest/'sheet.png')
        write_manifest(dest,config={'mode':mode,'min_score':args.min_score},inputs=inputs,accepted=len(accepted),rejected=dict(rejected),excluded_writers=sorted(excluded))
    return {'accepted':len(accepted),'rejected':dict(rejected),'writers':len({p['writer'] for p in accepted})}


def host_canvas(grey, rng):
    """A high-resolution local crop; letterbox without aspect distortion and keep a context margin."""
    ink=grey<200;ys,xs=np.nonzero(ink)
    if not len(xs): return None
    height=max(1,int(ys.max()-ys.min()+1));cx=int(rng.choice(xs))
    width=min(grey.shape[1],max(height*4,64))
    left=max(0,min(grey.shape[1]-width,cx-width//2))
    top=max(0,int(ys.min())-4);bottom=min(grey.shape[0],int(ys.max())+5)
    crop=Image.fromarray(grey[top:bottom,left:left+width])
    scale=min((SIZE-24)/crop.width,(SIZE-24)/crop.height,2.)
    crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.BILINEAR)
    canvas=np.full((SIZE,SIZE),255,np.uint8);ox=(SIZE-crop.width)//2;oy=(SIZE-crop.height)//2
    canvas[oy:oy+crop.height,ox:ox+crop.width]=np.asarray(crop)
    return canvas, [left,top,left+width,bottom]


def placement(clean, rng):
    ys,xs=np.nonzero(clean<200);height=int(ys.max()-ys.min()+1)
    # Intersect actual writing, never hide unused marks in empty page margins.
    index=rng.randrange(len(xs));cx,cy=int(xs[index]),int(ys[index])
    w=min(SIZE-12,max(24,round(height*rng.uniform(1.2,3.5))))
    h=max(3,min(height,round(height*rng.uniform(.12,.45))))
    return [max(0,min(SIZE-w,cx-w//2)),max(0,min(SIZE-h,cy-h//2)),w,h]


def material_alpha(bank, rng, place):
    p=bank.pieces[rng.randrange(len(bank.pieces))];l,t,w,h=place
    # Fit identical placement canvas in paired pixel/traced material arms.
    a=gold_piece(bank,p,rng,(w,h));alpha=np.zeros((SIZE,SIZE),np.float32);alpha[t:t+h,l:l+w]=a
    return alpha,p['id']


def build(args):
    other=twins(args.same_student);splits=writer_splits(args.mapping,args.same_student)
    recipes=[json.loads(s) for s in args.corpus.read_text().splitlines() if s.strip()]
    recipes=[r for r in recipes if r['kind']=='clean']
    templates=dict(np.load(args.templates)); removed_print=0
    excluded={canonical(r['group_uuid'],other) for r in json.loads(args.human_validation.read_text())}
    hosts=[]
    for r in recipes:
        w=canonical(r['writer'],other)
        if w in excluded: continue
        require(splits.get(r['writer'])=='train' and splits.get(other.get(r['writer']),'train')=='train','host belongs to held-out paper-1 writer')
        hosts.append({**r,'writer':w})
    val={r['writer'] for r in hosts if r['split']=='val'}
    train={r['writer'] for r in hosts if r['split']=='train'}
    require(train and val and not train&val,'invalid writer split')
    out=make_output_dir(args.output_dir)
    bank_info=prediction_material(args,excluded|val,other,splits,out)
    gold=Bank(args.gold_bank)
    gold.pieces=[p for p in gold.pieces if canonical(p['writer'],other) not in excluded|val]
    require(gold.pieces,'no eligible gold stroke material')
    banks={'gold_pixels':gold,**{f'predicted_{k}':Bank(out/f'bank_predicted_{k}') for k in ('pixels','traced')}}
    files=pages.page_files(args.pages,sorted({r['image_id'] for r in hosts}))
    records=[];targets=[];data={a:[] for a in ARMS};opened={};host_inputs={}
    for r in sorted(hosts,key=lambda r:r['line_id']):
        if r['image_id'] not in opened:
            opened.clear();path=files[r['image_id']]
            with Image.open(path) as im: opened[r['image_id']]=review_clean(np.asarray(im.convert('L')))
            host_inputs[str(path)]=digest(path)
        page=opened[r['image_id']];box=crop_box(r['box_1000'],page.shape[1],page.shape[0]);grey=page[box[1]:box[3],box[0]:box[2]]
        template=templates.get(group_key(r['image_id'],(page.shape[1],page.shape[0])))
        require(template is not None,'missing printed template for host')
        printed=template[box[1]:box[3],box[0]:box[2]]
        if ((grey<200)&printed).sum()/max(1,(grey<200).sum()) > .02:
            removed_print+=1;continue
        for copy in range(args.copies):
            seed=int(hashlib.sha256(f'{SEED}:{r["line_id"]}:{copy}'.encode()).hexdigest()[:8],16)
            rng=random.Random(seed);result=host_canvas(grey,rng)
            if result is None:continue
            clean,crop=result;place=placement(clean,rng);is_clean=(copy==0);level=float(np.percentile(clean[clean<200],30))
            ids={}
            for arm,bank in banks.items():
                # Validation is identical for all arms and uses procedural, separately generated strokes.
                if r['split']=='val':
                    lay=Image.new('L',(SIZE,SIZE),0);draw=ImageDraw.Draw(lay);l,t,w,h=place
                    draw.line([(l,t+h//2),(l+w//2,t),(l+w-1,t+h-1)],fill=255,width=max(2,h//4))
                    alpha=np.asarray(lay,dtype=np.float32)/255;pid='procedural_validation'
                else:alpha,pid=material_alpha(bank,random.Random(seed),place)
                if is_clean:alpha[:]=0
                image,target=draw_pair(clean,alpha,level);data[arm].append(image);ids[arm]=pid
            targets.append(target);records.append({'id':f'{r["line_id"]}#{copy}','line_id':r['line_id'],'image_id':r['image_id'],
                'group_uuid':r['writer'],'split':r['split'],'clean':is_clean,'crop':crop,'placement':place,'pieces':ids,'seed':seed})
    require(records,'empty paired dataset')
    for arm in ARMS: np.save(out/(arm+'.npy'),np.stack(data[arm]))
    np.save(out/'targets.npy',np.stack(targets));write_json(out/'records.json',records)
    samples=random.Random(SEED).sample([i for i,r in enumerate(records) if not r['clean'] and r['split']=='train'],min(12,len(records)))
    montage=np.concatenate([np.concatenate([targets[i],*[data[a][i] for a in ARMS]],axis=1) for i in samples],axis=0)
    Image.fromarray(montage).save(out/'comparison.png')
    summary={'samples':len(records),'printed_hosts_rejected':removed_print,'counts':dict(Counter(r['split'] for r in records)),'bank':bank_info,'training_writers':sorted(train),
             'validation_writers':sorted(val),'human_validation_excluded':sorted(excluded),'arms':list(ARMS),'target':'untouched clean host',
             'validation':'identical procedural strikes across arms; real human validation measured separately'}
    write_json(out/'summary.json',summary)
    inputs={str(p):digest(p) for p in [args.corpus,args.mapping,args.same_student,args.human_validation,args.templates,args.gold_bank/'run_manifest.json']}
    files_out={p.name:digest(p) for p in out.glob('*.npy')};files_out['records.json']=digest(out/'records.json')
    write_manifest(out,config=to_jsonable(args),inputs={**inputs,**host_inputs},files=files_out,**summary)
    print(json.dumps(summary),flush=True)


def darken_pairs(source,destination):
    """Exactly repair the compositor without changing hosts, material or RNG draws."""
    out=make_output_dir(destination);target=np.load(source/'targets.npy',mmap_mode='r')
    inputs={str(source/p):digest(source/p) for p in ['run_manifest.json','records.json','targets.npy',*[a+'.npy' for a in ARMS]]}
    source_manifest=json.loads((source/'run_manifest.json').read_text())
    for name,value in source_manifest['files'].items():require(digest(source/name)==value,'source paired data changed')
    shutil.copyfile(source/'targets.npy',out/'targets.npy');shutil.copyfile(source/'records.json',out/'records.json')
    counts={}
    for arm in ARMS:
        before=np.load(source/(arm+'.npy'),mmap_mode='r')
        after=np.lib.format.open_memmap(out/(arm+'.npy'),mode='w+',dtype=np.uint8,shape=before.shape)
        repaired=0
        for start in range(0,len(before),64):
            x=before[start:start+64];y=target[start:start+64];repaired+=int((x>y).sum());after[start:start+64]=np.minimum(x,y)
        after.flush();counts[arm]=repaired
    records=json.loads((out/'records.json').read_text());ids=random.Random(SEED).sample([i for i,r in enumerate(records) if not r['clean'] and r['split']=='train'],12)
    arrays=[np.load(out/(arm+'.npy'),mmap_mode='r') for arm in ARMS]
    Image.fromarray(np.concatenate([np.concatenate([target[i],*[x[i] for x in arrays]],axis=1) for i in ids])).save(out/'comparison.png')
    summary={**json.loads((source/'summary.json').read_text()),'compositing':'darken-only; original clean target preserved',
             'source_pairs':str(source),'lightening_pixels_corrected':counts}
    write_json(out/'summary.json',summary)
    write_manifest(out,config={'source':str(source),'operation':'elementwise minimum(input, untouched target)'},inputs=inputs,
                   files={p.name:digest(p) for p in out.glob('*.npy')},**summary)
    print(json.dumps({'counts':summary['counts'],'corrected_pixels':counts}),flush=True)


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv[:1]==['darken']:
        p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
        a=p.parse_args(argv[1:]);darken_pairs(a.source,a.output_dir);return
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','mapping','same-student','human-validation','gold-bank','detections','templates','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pages',type=Path,action='append',required=True);p.add_argument('--copies',type=int,default=4);p.add_argument('--min-score',type=float,default=.8)
    build(p.parse_args(argv))


if __name__=='__main__':main()
