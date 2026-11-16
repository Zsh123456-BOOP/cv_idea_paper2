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


def real_placement(clean, rng, bank, stretch=1., passes=1, wave=0., thicken=1.):
    """(placement, opacity, piece id) of a bank piece at its own size against the writing it struck (its line's height to this host's
    writing), ``stretch`` times as long and ``thicken`` times as thick, drawn back and forth up to ``passes`` times a little apart, its
    path bent by a wave of up to ``wave`` writing heights, across the middle of the host's writing round a point of its ink."""
    ys,xs=np.nonzero(clean<200);height=int(ys.max()-ys.min()+1)
    piece=bank.pieces[rng.randrange(len(bank.pieces))];scale=height/max(1,piece['line_height'])
    w=int(np.clip(round(piece['size'][0]*scale*stretch),8,SIZE-12));h=int(np.clip(round(piece['size'][1]*scale*thicken),2,SIZE-12))
    patch=gold_piece(bank,piece,rng,(w,h))
    amplitude=int(round(rng.uniform(0,wave)*height))
    if amplitude:
        period=rng.uniform(.6,1.5)*height;phase=rng.uniform(0,2*np.pi);bent=np.zeros((h+2*amplitude,w),np.float32)
        for x in range(w):
            dy=amplitude+int(round(amplitude*np.sin(2*np.pi*x/period+phase)));bent[dy:dy+h,x]=patch[:,x]
        patch=bent
    count=rng.randint(1,max(1,passes))
    if count>1:
        gap=max(2,int(round(rng.uniform(.15,.35)*height)));ph,pw=patch.shape;many=np.zeros((ph+gap*(count-1),pw),np.float32)
        for k in range(count):many[k*gap:k*gap+ph]=np.maximum(many[k*gap:k*gap+ph],patch[:, ::-1] if k%2 else patch)
        patch=many
    patch=patch[:SIZE-12,:SIZE-12];h,w=patch.shape
    # through the middle of the writing under it, as a student strikes a word out, not along its top or bottom edge
    index=rng.randrange(len(xs));cx=int(xs[index]);under=(xs>=cx-w//2)&(xs<cx+w-w//2);cy=int(np.median(ys[under]))
    l,t=max(0,min(SIZE-w,cx-w//2)),max(0,min(SIZE-h,cy-h//2))
    alpha=np.zeros((SIZE,SIZE),np.float32);alpha[t:t+h,l:l+w]=patch
    return [l,t,w,h],alpha,piece['id']


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
            clean,crop=result;is_clean=(copy==0)
            if args.real_geometry:      # v2: real size and a level lighter than the writing (data.restoration_pairs --real-geometry)
                place,real_alpha,real_id=real_placement(clean,rng,gold,args.stretch,args.passes,args.wave,args.thicken)
                level=float(np.clip(np.median(clean[clean<200])+args.lighter,0,250))
            else:
                place=placement(clean,rng);level=float(np.percentile(clean[clean<200],30))
            ids={}
            for arm,bank in banks.items():
                # Validation is identical for all arms and uses procedural, separately generated strokes.
                if r['split']=='val':
                    lay=Image.new('L',(SIZE,SIZE),0);draw=ImageDraw.Draw(lay);l,t,w,h=place
                    draw.line([(l,t+h//2),(l+w//2,t),(l+w-1,t+h-1)],fill=255,width=max(2,h//4))
                    alpha=np.asarray(lay,dtype=np.float32)/255;pid='procedural_validation'
                elif args.real_geometry and arm=='gold_pixels':alpha,pid=real_alpha.copy(),real_id
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


REMOVED = 51        # a pixel the strike darkened by more than 0.2 of the range: the evaluation's "removed" (baselines.restoration)


def canvas_tiles(array, fill):
    """The 256 tiles ``baselines.restoration.restore_native`` reads of a native crop: centred on ``fill`` when shorter, stride 192."""
    h, w = array.shape; ph, pw = max(SIZE, h), max(SIZE, w); oy, ox = (ph-h)//2, (pw-w)//2
    canvas = np.full((ph, pw), fill, array.dtype); canvas[oy:oy+h, ox:ox+w] = array
    at = lambda n: sorted(set(list(range(0, n-SIZE+1, 192))+[n-SIZE]))
    return [canvas[y:y+SIZE, x:x+SIZE] for y in at(ph) for x in at(pw)]


def tiles(image, mask, weight=None, clean=None):
    """The tiles of a native crop (``canvas_tiles`` on white), each with its target (the crop with the pixels to remove made white, or
    the given ``clean`` image) and its per-pixel loss weight (1 where none is given)."""
    if clean is None: target = image.copy(); target[mask.astype(bool)] = 255
    else: target = np.asarray(clean, np.uint8)
    weights = np.ones(image.shape, np.uint8) if weight is None else np.asarray(weight, np.uint8)
    return list(zip(canvas_tiles(image, 255), canvas_tiles(target, 255), canvas_tiles(weights, 1)))


def question_gold(page, revisions):
    """(grey, strike mask, region mask) of a whole question: the confirmed pixels of every region (``data.pixel_masks.reviewed_crop``)
    and the union of the regions' polygons, both in the question's own coordinates."""
    from .pixel_masks import region_pixels, reviewed_crop
    grey = np.asarray(page.convert('L')); gold = np.zeros(grey.shape, bool); regions = np.zeros(grey.shape, bool)
    for r in revisions:
        if r.get('deleted') or not r.get('confirmed'): continue
        _, mask = reviewed_crop(page, r); l, t, rr, b = r['box']
        gold[t:b, l:rr] |= mask.astype(bool); regions[t:b, l:rr] |= region_pixels(r.get('region'), r['box'])
    return grey, gold, regions


def question_pairs(export, pages, ids, review_set, source, arm, whole, destination):
    """The human training data of a whole-question test (user 2026-10-23): the review set's native training crops (``human_pairs``
    without replay), and the questions ``ids`` of an export whose every region is confirmed -- whole (``whole``: every 256 tile of the
    question, the ink outside its regions kept) or as the crops of their regions only -- with the pairs' validation split."""
    from .pixel_masks import reviewed_crop
    data = json.loads(export.read_text()); chosen = Path(ids).read_text().split()
    native = json.loads((review_set/'training_native.json').read_text()); arrays = np.load(review_set/'training_native.npz')
    pairs = json.loads((source/'records.json').read_text())
    x = np.load(source/(arm+'.npy'), mmap_mode='r'); y = np.load(source/'targets.npy', mmap_mode='r')
    images, targets, rows = [], [], []
    def add(items, writer, name, kind):
        for n, (tile, target) in enumerate(items):
            images.append(tile); targets.append(target)
            rows.append({'id': f'{name}~{n}', 'group_uuid': writer, 'split': 'train', 'source': kind, 'clean': bool((tile == target).all())})
    for k, r in enumerate(native):
        add([t[:2] for t in tiles(arrays[f'image_{k}'], arrays[f'mask_{k}'])], r['group_uuid'], r['id'], 'human_confirmed')
    for image_id in chosen:
        revisions = data['pages'][image_id]['revisions']
        require(revisions and all(r.get('confirmed') for r in revisions if not r.get('deleted')), f'{image_id}: a region is not confirmed')
        with Image.open(pages/(image_id+'.png')) as page:
            if whole:
                grey, gold, _ = question_gold(page, revisions); add([t[:2] for t in tiles(grey, gold)], image_id.split('_')[0], image_id, 'whole_question')
            else:
                for r in revisions:
                    if r.get('confirmed') and not r.get('deleted'):
                        image, mask = reviewed_crop(page, r); add([t[:2] for t in tiles(image, mask)], image_id.split('_')[0], image_id+'#'+r['id'], 'question_region')
    train = {r['group_uuid'] for r in rows}
    for i, r in enumerate(pairs):
        if r['split'] == 'val' and r['group_uuid'] not in train:
            images.append(np.asarray(x[i])); targets.append(np.asarray(y[i]))
            rows.append({'id': r['id'], 'group_uuid': r['group_uuid'], 'split': 'val', 'source': 'synthetic_validation', 'clean': r['clean']})
    out = make_output_dir(destination)
    np.save(out/(arm+'.npy'), np.stack(images)); np.save(out/'targets.npy', np.stack(targets)); write_json(out/'records.json', rows)
    summary = {'counts': dict(Counter(r['source']+'/'+r['split'] for r in rows)), 'questions': len(chosen), 'whole': whole}
    write_json(out/'summary.json', summary)
    inputs = {str(p): digest(p) for p in [export, Path(ids), review_set/'run_manifest.json', review_set/'training_native.npz', source/'run_manifest.json']}
    write_manifest(out, config={'export': str(export), 'pages': str(pages), 'ids': str(ids), 'review_set': str(review_set), 'whole': whole},
                   inputs=inputs, files={p.name: digest(p) for p in out.glob('*.npy')} | {'records.json': digest(out/'records.json')}, **summary)
    print(json.dumps(summary), flush=True)


def as_boxes(source, arm, destination):
    """The same pairs as the box remover's training set (``data.synthetic_set --crops``): a crop round each pair's strike -- the
    pixels it darkened by more than ``REMOVED``, the box of the writing they void, a random margin -- letterboxed onto the strip, one
    instance; a clean pair gives ``CLEAN_CROPS`` crops of its words.  Each record keeps the index of the pair it was cut from."""
    from .paste_strikes import word_spans
    from .synthetic_set import CLEAN_CROPS, cropped, region_of, rows_round, sheet
    out = make_output_dir(destination); manifest = json.loads((source/'run_manifest.json').read_text())
    for name, value in manifest['files'].items(): require(digest(source/name) == value, 'source paired data changed')
    pairs = json.loads((source/'records.json').read_text())
    x = np.load(source/(arm+'.npy'), mmap_mode='r'); y = np.load(source/'targets.npy', mmap_mode='r')
    images, labels, records, skipped = [], [], [], 0
    for i, r in enumerate(pairs):
        struck, clean = np.asarray(x[i]), np.asarray(y[i]); rng = random.Random(r['seed'])
        numbers = (((clean.astype(np.int16)-struck) > REMOVED) & (struck < 200)).astype(np.uint8)
        ink = struck < 200; image = Image.fromarray(struck)
        base = {'line_id': r['line_id'], 'group_uuid': r['group_uuid'], 'split': r['split'], 'pair_index': i,
                'styles': ['procedural' if r['split'] == 'val' else 'gold_line']}
        if not r['clean']:
            if not numbers.any():
                skipped += 1; continue            # its strike lies wholly on writing: nothing to remove
            grey, strip_numbers, crop = cropped(image, numbers, region_of(numbers, 1, ink), rng)
            images.append(grey); labels.append(strip_numbers)
            records.append({**base, 'id': r['id'], 'kinds': ['paste'], 'revisions': 1, 'crop': crop}); continue
        spans = word_spans(ink, int(np.ptp(np.nonzero(ink)[0]))+1) if ink.any() else []
        for n in range(CLEAN_CROPS if spans else 0):
            first = rng.randrange(len(spans)); last = min(len(spans), first+rng.randint(1, 3))-1
            columns = (spans[first][0], spans[last][1]); rows = np.flatnonzero(ink[:, columns[0]:columns[1]].any(1))
            top, bottom = rows_round(ink, columns, int(np.median(rows)))
            grey, strip_numbers, crop = cropped(image, np.zeros_like(numbers), [columns[0], top, columns[1], bottom], rng)
            images.append(grey); labels.append(strip_numbers)
            records.append({**base, 'id': f"{r['id']}#c{n}", 'kinds': ['clean'], 'revisions': 0, 'crop': crop})
    np.save(out/'images.npy', np.stack(images)); np.save(out/'labels.npy', np.stack(labels)); write_json(out/'records.json', records)
    sheet(np.stack(images), np.stack(labels), records, out/'sheet.png')
    summary = {'samples': len(records), 'by_split': dict(Counter(r['split'] for r in records)),
               'positive': sum(r['revisions'] > 0 for r in records), 'struck_wholly_on_writing': skipped, 'source_pairs': str(source), 'arm': arm}
    write_json(out/'summary.json', summary)
    write_manifest(out, config={'source': str(source), 'arm': arm, 'removed_above': REMOVED}, inputs={str(source/'run_manifest.json'): digest(source/'run_manifest.json')},
                   files={n: digest(out/n) for n in ('images.npy', 'labels.npy', 'records.json')}, **summary)
    print(json.dumps(summary), flush=True)


def human_pairs(review_set, source, arm, destination, no_replay=False):
    """The box remover's fine-tuning set (``data.review_set``) for the restorers, on their own native tiles: its human training regions
    tiled (``tiles``), its synthetic replay as the very pairs those strips were cut from (``as_boxes``'s ``pair_index``), and the pairs'
    own validation split (writers apart from the training ones) for their synthetic check.  The 16 fixed validation regions stay the
    restorers' human validation, as in their synthetic training.  ``no_replay``: the human regions alone.  A review set with per-pixel
    weights (``weight_<k>``, ``data.pseudo_regions``) gives ``weights.npy`` beside the tiles, 1 on every other row; a region with its
    own ``target_<k>`` image (the teachers' mean) is tiled against that instead of the whitened mask; a region keeps the source and the
    sample weight its review set's record gives it.  Without ``source`` (no synthetic pairs: the prefill model) there is neither replay
    nor a synthetic validation."""
    out = make_output_dir(destination)
    records = json.loads((review_set/'records.json').read_text())
    native = json.loads((review_set/'training_native.json').read_text()); arrays = np.load(review_set/'training_native.npz')
    pairs = json.loads((source/'records.json').read_text()) if source else []
    if source: x = np.load(source/(arm+'.npy'), mmap_mode='r'); y = np.load(source/'targets.npy', mmap_mode='r')
    data, targets, rows, weights = [], [], [], []
    weighted = any(name.startswith('weight_') for name in arrays.files); source_of = {r['id']: r['source'] for r in records}
    weight_of = {r['id']: r['weight'] for r in records if 'weight' in r}
    for k, r in enumerate(native):
        made = tiles(arrays[f'image_{k}'], arrays[f'mask_{k}'], arrays[f'weight_{k}'] if weighted else None,
                     arrays[f'target_{k}'] if f'target_{k}' in arrays.files else None)
        for n, (tile, target, weight) in enumerate(made):
            data.append(tile); targets.append(target); weights.append(weight)
            rows.append({'id': f"{r['id']}~{n}", 'group_uuid': r['group_uuid'], 'split': 'train', 'source': source_of.get(r['id'], 'human_confirmed'),
                         'clean': bool((tile == target).all()), **({'weight': weight_of[r['id']]} if r['id'] in weight_of else {})})
    replay = [] if no_replay or not source else [r for r in records if r['source'] == 'synthetic_replay' and r['split'] == 'train']
    require(all('pair_index' in r for r in replay), 'the replay was not cut from these pairs')
    for r in replay:
        i = r['pair_index']; data.append(np.asarray(x[i])); targets.append(np.asarray(y[i]))
        rows.append({'id': pairs[i]['id'], 'group_uuid': r['group_uuid'], 'split': 'train', 'source': 'synthetic_replay', 'clean': pairs[i]['clean']})
    train = {r['group_uuid'] for r in rows}
    for i, r in enumerate(pairs):
        if r['split'] == 'val' and r['group_uuid'] not in train:
            data.append(np.asarray(x[i])); targets.append(np.asarray(y[i]))
            rows.append({'id': r['id'], 'group_uuid': r['group_uuid'], 'split': 'val', 'source': 'synthetic_validation', 'clean': r['clean']})
    np.save(out/(arm+'.npy'), np.stack(data)); np.save(out/'targets.npy', np.stack(targets)); write_json(out/'records.json', rows)
    if weighted: np.save(out/'weights.npy', np.concatenate([np.stack(weights), np.ones((len(data)-len(weights), SIZE, SIZE), np.uint8)]))
    summary = {'counts': dict(Counter(r['source']+'/'+r['split'] for r in rows)), 'human_regions': len(native), 'weighted': weighted,
               'train_tiles': sum(r['split'] == 'train' for r in rows), 'review_set': str(review_set), 'source_pairs': str(source) if source else None}
    write_json(out/'summary.json', summary)
    inputs = {str(p): digest(p) for p in [review_set/'run_manifest.json', review_set/'training_native.npz', *([source/'run_manifest.json'] if source else [])]}
    write_manifest(out, config={'review_set': str(review_set), 'source': str(source) if source else None, 'arm': arm, 'no_replay': no_replay}, inputs=inputs,
                   files={p.name: digest(p) for p in out.glob('*.npy')} | {'records.json': digest(out/'records.json')}, **summary)
    print(json.dumps(summary), flush=True)


# ---- synthesis materials and their checks (plan of 2026-11-16, step one: no training) -----------------------------------------------
SOURCES = ('H', 'P', 'D')
LOOK = 200                    # materials per source checked (seed 42)
NOMINAL = 40                  # a drawn strike (D) is drawn for a line this high
DRAWN = {'line': .2, 'double': .1, 'wave': .1, 'zigzag': .2, 'cross': .15, 'scribble': .25}   # D's styles, heavy ones often
SHARE_CUTS = (.23, .35)       # light, middle, heavy by the strike's share of the ink (eval.prefill's tiers)
TINY = .5                     # or whose strike spans less than this share of the writing it struck (a speck the eraser took)
LINE_COVER = .8               # or with a straight part still less covered than this once closed (check 2's line)
GAPPY = .6                    # or with fewer pixels than this share of one pen stroke as long as it spans (dots left between writing)
HOLES = .3                    # or with more writing in its holes than this share of its pixels (``strike_bank.holes``)
FRAGMENTS = .4                # a human strike with more of its pixels in specks than this, closed or not, is left out


def tier(share):
    return 'light' if share < SHARE_CUTS[0] else 'middle' if share < SHARE_CUTS[1] else 'heavy'


def excluded_writers(set_dir, mapping, same_student):
    """The writers no material or host may come from: the held-out and fixed-validation writers of the prefill set, paper 1's test
    writers, and each one's other id."""
    summary = json.loads((set_dir/'summary.json').read_text()); other = twins(same_student); splits = writer_splits(mapping, same_student)
    out = set(summary['validation_writers'])|set(summary['fixed_writers'])|{w for w, s in splits.items() if s == 'test'}
    return out|{other[w] for w in out if w in other}


def specks(mask):
    labels, n = ndimage.label(mask, np.ones((3, 3)))
    if not n: return 1.
    sizes = np.bincount(labels.ravel())[1:]; width = max(2., 2*float(np.median(ndimage.distance_transform_edt(mask)[mask])))
    return float(sizes[sizes < max(12, 2*width*width)].sum())/max(1, sizes.sum())


def line_height_at(lines, page, box):
    """The height of the catalogue line on the page that holds most of a box (None if none does)."""
    best, height = 0, None
    for line in lines.get(page, []):
        l, t, r, b = line['px']; over = max(0, min(r, box[2])-max(l, box[0]))*max(0, min(b, box[3])-max(t, box[1]))
        if over > best: best, height = over, b-t
    return height


def human_materials(set_dir, readable, lines, excluded):
    """Source H: the strikes the annotator erased in the prefill set's training regions, with their gaps closed (``strike_bank.completed``).
    Left out (counted): a region read as unreadable (a blackout), of an excluded writer, with too few strike pixels, or in specks."""
    from .strike_bank import human_material
    from .paste_strikes import writing_height
    rows = json.loads((set_dir/'training_native.json').read_text()); arrays = np.load(set_dir/'training_native.npz')
    made, left = [], Counter()
    for k, r in enumerate(rows):
        if r['group_uuid'] in excluded: left['excluded_writer'] += 1; continue
        if readable.get(r['id']) is False: left['unreadable'] += 1; continue
        grey, strike = arrays[f'image_{k}'], arrays[f'mask_{k}'].astype(bool)
        if strike.sum() < 12: left['too_few_pixels'] += 1; continue
        before, after, level, holed, cover = human_material(grey, strike)
        if cover < LINE_COVER: left['line_still_broken'] += 1; continue
        if specks(after >= .3) > FRAGMENTS: left['specks'] += 1; continue
        if holed > HOLES: left['holes'] += 1; continue
        writing = (grey < 200) & ~strike; wy, wx = np.nonzero(writing); sy, sx = np.nonzero(after >= .3)
        tall = writing_height(writing) or grey.shape[0]
        span = max(np.ptp(sy), np.ptp(sx))+1
        if span < TINY*tall: left['tiny'] += 1; continue
        stroke = max(2., 2*float(np.median(ndimage.distance_transform_edt(grey < 200)[grey < 200])))      # the pen's width
        if len(sy) < GAPPY*stroke*span: left['mostly_gaps'] += 1; continue
        rel_y = float(np.clip((sy.mean()-wy.min())/max(1, wy.max()-wy.min()), 0, 1)) if len(wy) else .5
        height = line_height_at(lines, r['page'], r['box']) or (int(wy.max()-wy.min()+1) if len(wy) else grey.shape[0])
        made.append({'id': f"H:{r['id']}", 'source': 'H', 'writer': r['group_uuid'], 'page': r['page'], 'box': r['box'], 'line_height': int(height),
                     'writing_height': tall,
                     'rel_y': rel_y, 'level': level, 'origin_share': float(strike.sum())/max(1, strike.sum()+writing.sum()),
                     'arrays': {'before': before, 'after': after, 'origin': grey}})
    return made, dict(left)


def bank_materials(bank_dir, scans, excluded, other):
    """Source P: paper 1's bank lines of writers not excluded, trimmed (``strike_bank.trimmed_line``).  Left out (counted): a line with
    too little of its length inked (dots, pieces) or that is not one line once trimmed (letters, loops, several strokes)."""
    from .strike_bank import trimmed_line, line_measures
    from .paste_strikes import writing_height
    pieces = json.loads((bank_dir/'pieces.json').read_text()); arrays = np.load(bank_dir/'bank.npz')
    made, left, opened = [], Counter(), {}
    for p in pieces:
        if canonical(p['writer'], other) in excluded or p['writer'] in excluded: left['excluded_writer'] += 1; continue
        before, after, measures = trimmed_line(arrays[p['id']], p['ink_level'])
        if measures['broken']: left['broken'] += 1; continue
        if not line_measures(after)['linear']: left['not_one_line'] += 1; continue
        if p['image_id'] not in opened:
            opened.clear()
            with Image.open(scans[p['image_id']]) as im: opened[p['image_id']] = np.asarray(im.convert('L'))
        page = opened[p['image_id']]; l, t, r, b = p['box']; m = p['line_height']//2
        L, T, R, B = max(0, l-m), max(0, t-m), min(page.shape[1], r+m), min(page.shape[0], b+m)
        window = page[T:B, L:R]; mine = np.zeros(window.shape, bool); mine[t-T:b-T, l-L:r-L] = arrays[p['id']] < 255
        writing = (window < 200) & ~mine; wy = np.flatnonzero(writing[:, l-L:r-L].any(1))
        rel_y = float(np.clip(((t+b)/2-T-wy.min())/max(1, wy.max()-wy.min()), 0, 1)) if len(wy) else .5
        made.append({'id': f"P:{p['id']}", 'source': 'P', 'writer': canonical(p['writer'], other), 'page': p['image_id'], 'box': p['box'],
                     'line_height': p['line_height'], 'writing_height': writing_height(writing[:, l-L:r-L]) or p['line_height'], 'rel_y': rel_y, 'level': p['ink_level'],
                     'origin_share': float(mine.sum())/max(1, mine.sum()+writing[:, l-L:r-L].sum()),
                     'arrays': {'before': before, 'after': after, 'origin': window}})
    return made, dict(left)


def drawn_material(rng, n):
    """Source D: a strike drawn by ``paste_strikes.strokes`` (no blackouts) for a line ``NOMINAL`` high, in one of ``DRAWN``'s styles."""
    from .paste_strikes import strokes, STROKE
    style = rng.choices(list(DRAWN), weights=list(DRAWN.values()))[0]
    w, h = int(NOMINAL*rng.uniform(1., 4.)), int(NOMINAL*rng.uniform(.3, .8))
    _, alpha = strokes((w, h), style, rng, (0, 0, 0), max(2., rng.uniform(*STROKE)*NOMINAL), 0)
    return {'id': f'D:{n}', 'source': 'D', 'writer': None, 'style': style, 'line_height': NOMINAL, 'writing_height': NOMINAL, 'rel_y': rng.uniform(.4, .6), 'level': 0,
            'arrays': {'before': alpha, 'after': alpha}}


def host_lines(catalogue, files, excluded, other):
    """Lines without a revision (the catalogue's ``has_edit`` false) of allowed writers outside ``excluded``, on pages there are."""
    return [r for r in catalogue if not r.get('has_edit') and canonical(r['group_uuid'], other) not in excluded
            and r['group_uuid'] not in excluded and r['image_id'] in files]


PRINTED = .02                 # a host line with more of its ink on the page's printed template than this is printed (a question number)


def usable_host(page, line, templates, image_id):
    """Whether a host line is handwriting: little of its ink on its question's printed template, and written at least two heights wide."""
    l, t, r, b = line['px']; ink = page[t:b, l:r] < 200
    columns = np.flatnonzero(ink.any(0))
    if not len(columns) or columns[-1]-columns[0] < 2*(b-t): return False
    template = templates.get(group_key(image_id, (page.shape[1], page.shape[0])))
    return template is None or (ink & template[t:b, l:r]).sum() <= PRINTED*ink.sum()


def rendered(alpha, level=40):
    return np.rint(255-(255-level)*np.clip(alpha, 0, 1)).astype(np.uint8)


def png64(array, scale):
    import base64, io
    image = Image.fromarray(array); image = image.resize((max(1, image.width*scale), max(1, image.height*scale)), Image.NEAREST)
    buffer = io.BytesIO(); image.save(buffer, 'PNG'); return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def look_page(chosen, path):
    """Check 4's page: per material its original place, before and after cleaning, and put on a host; two boxes to tick (writing
    carried, gaps) kept in the browser and counted at the top, exportable."""
    import html
    rows = []
    for m in chosen:
        a = m['arrays']; shown = [x for x in (a.get('origin'), a['after'], m.get('pasted')) if x is not None]
        scale = max(1, min(4, 100//max(x.shape[0] for x in shown)))
        cells = [png64(a['origin'], scale) if a.get('origin') is not None else None, png64(rendered(a['before']), scale), png64(rendered(a['after']), scale),
                 png64(m['pasted'], scale) if m.get('pasted') is not None else None]
        info = ' '.join(f'{k}={v}' for k, v in m['shown'].items())
        rows.append(f'<tr data-id="{html.escape(m["id"])}"><td class="id">{html.escape(m["id"][:40])}<br><small>{html.escape(info)}</small></td>'
                    + ''.join(f'<td>{f"<img src={chr(34)}{c}{chr(34)}>" if c else "—"}</td>' for c in cells)
                    + '<td><label><input type="checkbox" data-k="writing">带字</label><br><label><input type="checkbox" data-k="gap">缺口</label></td></tr>')
    page = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>合成素材检查</title>
<style>body{{font:14px sans-serif;margin:16px;background:#fff;color:#111}}table{{border-collapse:collapse}}td{{border:1px solid #ddd;padding:4px;vertical-align:middle}}
td.id{{width:200px;min-width:200px;word-break:break-all;font-size:11px}}img{{image-rendering:pixelated;max-width:340px;max-height:200px}}#top{{position:sticky;top:0;background:#fff;padding:6px 0;border-bottom:1px solid #ccc}}
h2{{margin-top:28px}}</style>
<div id="top">勾选：带字 / 缺口。计数（按来源）：<span id="count"></span> <button id="export">导出勾选</button></div>
<p>每行：原处｜清洗前｜清洗后｜贴到宿主上。D 是程序画的，没有原处、不清洗。</p>"""
    for source in SOURCES:
        part = [r for r, m in zip(rows, chosen) if m['source'] == source]
        page += f'<h2>来源 {source}（{len(part)} 条）</h2><table><tr><th>素材</th><th>原处</th><th>清洗前</th><th>清洗后</th><th>贴到宿主上</th><th>看出</th></tr>{"".join(part)}</table>'
    page += """<script>const K='synth-materials-check-20261116';let s={};try{s=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
function count(){const c={};for(const tr of document.querySelectorAll('tr[data-id]')){const src=tr.dataset.id[0];c[src]=c[src]||{n:0,writing:0,gap:0,any:0};c[src].n++;
let any=0;for(const b of tr.querySelectorAll('input')){if(b.checked){c[src][b.dataset.k]++;any=1}}c[src].any+=any}
document.getElementById('count').textContent=Object.entries(c).map(([k,v])=>`${k}: 带字 ${v.writing}、缺口 ${v.gap}、任一 ${v.any}/${v.n}（${(100*v.any/v.n).toFixed(1)}%）`).join('；')}
for(const tr of document.querySelectorAll('tr[data-id]'))for(const b of tr.querySelectorAll('input')){b.checked=!!(s[tr.dataset.id]||{})[b.dataset.k];
b.onchange=()=>{(s[tr.dataset.id]=s[tr.dataset.id]||{})[b.dataset.k]=b.checked;try{localStorage.setItem(K,JSON.stringify(s))}catch(e){}count()}}
document.getElementById('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(s,null,1)],{type:'application/json'}));a.download='synth_materials_check.json';a.click()};count()</script>"""
    path.write_text(page)


def materials(args):
    """The three sources of strikes (H human, P paper 1's bank, D drawn), cleaned; checks 1 and 2 on ``LOOK`` of each (before and after),
    check 4's page and check 3's reader pack (each material alone on white) on the same ones."""
    from .strike_bank import line_measures
    other = twins(args.same_student); excluded = excluded_writers(args.set, args.mapping, args.same_student)
    readable = {r['id']: r.get('readable') for r in json.loads(args.dataset.read_text())}
    from .lines import read_catalogue
    catalogue = read_catalogue(args.catalogue); files = pages.page_files(args.pages)
    hosts = host_lines(catalogue, files, excluded, other)
    lines = {}
    for r in catalogue:
        if r['image_id'] in files: lines.setdefault(r['image_id'], []).append(r)
    sizes = {}
    def size(page):
        if page not in sizes:
            with Image.open(files[page]) as im: sizes[page] = im.size
        return sizes[page]
    for page, rows in lines.items():
        for r in rows: r['px'] = crop_box(r['box_1000'], *size(page))
    human, human_left = human_materials(args.set, readable, lines, excluded)
    bank, bank_left = bank_materials(args.bank, files, excluded, other)
    rng = random.Random(SEED); drawn = [drawn_material(rng, n) for n in range(LOOK)]
    out = make_output_dir(args.output_dir)
    store = {}
    for m in human+bank:
        for key in ('before', 'after'): store[f"{m['id']}|{key}"] = np.rint(m['arrays'][key]*255).astype(np.uint8)
    np.savez_compressed(out/'materials.npz', **store)
    write_json(out/'materials.json', [{k: v for k, v in m.items() if k != 'arrays'} for m in human+bank])
    chosen = [m for source, pool in (('H', human), ('P', bank), ('D', drawn)) for m in random.Random(SEED).sample(pool, min(LOOK, len(pool)))]
    by_page = {}
    for r in hosts: by_page.setdefault(r['image_id'], []).append(r)
    host_pages = sorted(by_page); opened = {}
    checks = {s: {'before': [], 'after': [], 'tiers': Counter(), 'shares': []} for s in SOURCES}
    templates = dict(np.load(args.templates))
    pack_rows = []; (out/'reader'/'assets').mkdir(parents=True)
    from .paste_strikes import on_writing
    for n, m in enumerate(chosen):
        draw = random.Random(f'{SEED}:{m["id"]}')
        put = None
        for _ in range(30):
            page = draw.choice(host_pages)
            if page not in opened:
                if len(opened) > 64: opened.clear()
                with Image.open(files[page]) as im: opened[page] = review_clean(np.asarray(im.convert('L')))
            line = draw.choice(by_page[page])
            if not usable_host(opened[page], line, templates, page): continue
            put = on_writing(opened[page], line['px'], m['arrays']['after'], m, draw)
            if put is not None: break
        mb, ma = line_measures(m['arrays']['before']), line_measures(m['arrays']['after'])
        checks[m['source']]['before'].append(mb); checks[m['source']]['after'].append(ma)
        m['pasted'] = put['image'] if put else None
        if put: checks[m['source']]['tiers'][tier(put['strike_share'])] += 1; checks[m['source']]['shares'].append(put['strike_share'])
        m['shown'] = {'线状': ma.get('linear'), 'fill': f"{mb.get('fill')}→{ma.get('fill')}", 'cover': f"{mb.get('cover')}→{ma.get('cover')}",
                      **({'share': round(put['strike_share'], 2)} if put else {}), **({'style': m['style']} if 'style' in m else {})}
        for key in ('before', 'after') if m['source'] != 'D' else ('after',):
            name = f"{m['source']}_{key}_{n}.png"
            Image.fromarray(np.pad(rendered(m['arrays'][key]), 8, constant_values=255)).convert('RGB').save(out/'reader'/'assets'/name)
            pack_rows.append({'id': f"{m['id']}|{key}", 'line_id': f"{m['id']}|{key}", 'source': m['source'], 'cleaning': key, 'images': {'raw': f'assets/{name}'}})
    write_json(out/'reader'/'pack.json', {'schema': 'revrender-line-pack', 'seed': SEED, 'rows': pack_rows})
    look_page(chosen, out/'check_page.html')
    def summary_of(measures):
        lined = [x for x in measures if x.get('linear')]
        if not lined: return {'linear': 0, 'of': len(measures)}
        fill = np.array([x['fill'] for x in lined]); cover = np.array([x['cover'] for x in lined])
        return {'linear': len(lined), 'of': len(measures), 'fill_median': round(float(np.median(fill)), 3), 'fill_p95': round(float(np.percentile(fill, 95)), 3),
                'cover_median': round(float(np.median(cover)), 3), 'cover_below_0.8': round(float((cover < .8).mean()), 3)}
    result = {s: {'before': summary_of(c['before']), 'after': summary_of(c['after']), 'pasted_tiers': dict(c['tiers']),
                  'pasted_share_median': round(float(np.median(c['shares'])), 3) if c['shares'] else None} for s, c in checks.items()}
    kept = {'H': human, 'P': bank}
    produced = {s: {'materials': len(pool), 'writers': len({m['writer'] for m in pool}),
                    'origin_tiers': dict(Counter(tier(m['origin_share']) for m in pool))} for s, pool in kept.items()}
    summary = {'checks_1_2': result, 'produced': produced, 'left_out': {'H': human_left, 'P': bank_left}, 'hosts': {'lines': len(hosts),
               'writers': len({canonical(r['group_uuid'], other) for r in hosts}), 'pages': len(host_pages)}, 'excluded_writers': len(excluded),
               'looked': {s: sum(m['source'] == s for m in chosen) for s in SOURCES}}
    write_json(out/'summary.json', summary)
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [args.set/'training_native.npz', args.bank/'bank.npz', args.dataset,
                   args.catalogue, args.mapping, args.same_student, args.templates]}, **summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def read_check(pack, predictions, output):
    """Check 3: the share of each source's materials (alone on white) a reader that was never taught read at least one letter, digit or
    Chinese character in (not the refusal, not empty; a LaTeX command's name is not read writing: ``\cancel{}`` reads nothing), against
    the drawn strikes (D)."""
    import re
    rows = {r['id']: r for r in json.loads((pack/'pack.json').read_text())['rows']}
    readings = [json.loads(s) for s in predictions.read_text().splitlines() if s.strip()]
    found = Counter(); total = Counter()
    for p in readings:
        r = rows[p['id']]; key = f"{r['source']}_{r['cleaning']}"; total[key] += 1
        text = re.sub(r'\\[A-Za-z]+', '', (p.get('text') or '').replace('[读不出]', '').replace('[空]', ''))
        found[key] += bool(re.search(r'[0-9A-Za-z\u4e00-\u9fff]', text))
    shares = {k: round(found[k]/total[k], 3) for k in total}
    out = {'read_share': shares, 'counts': dict(total), 'control': shares.get('D_after'),
           'within_5_points_of_control': {k: shares[k] <= shares.get('D_after', 0)+.05 for k in shares if k.endswith('_after') and not k.startswith('D')}}
    write_json(output, out); print(json.dumps(out), flush=True)


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv[:1]==['darken']:
        p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
        a=p.parse_args(argv[1:]);darken_pairs(a.source,a.output_dir);return
    if argv[:1]==['boxes']:
        p=argparse.ArgumentParser(description=as_boxes.__doc__);p.add_argument('--source',type=Path,required=True)
        p.add_argument('--arm',choices=ARMS,default='gold_pixels');p.add_argument('--output-dir',type=Path,required=True)
        a=p.parse_args(argv[1:]);as_boxes(a.source,a.arm,a.output_dir);return
    if argv[:1]==['questions']:
        p=argparse.ArgumentParser(description=question_pairs.__doc__)
        for name in ('export','pages','ids','review-set','source','output-dir'):p.add_argument('--'+name,type=Path,required=True)
        p.add_argument('--arm',choices=ARMS,default='gold_pixels');p.add_argument('--whole',action='store_true',help='every tile of each question, not only its regions')
        a=p.parse_args(argv[1:]);question_pairs(a.export,a.pages,a.ids,a.review_set,a.source,a.arm,a.whole,a.output_dir);return
    if argv[:1]==['materials']:
        p=argparse.ArgumentParser(description=materials.__doc__)
        for name in ('set','bank','dataset','catalogue','mapping','same-student','templates','output-dir'):p.add_argument('--'+name,type=Path,required=True)
        p.add_argument('--pages',type=Path,action='append',required=True,help='clean pages (hosts) and the scans the bank was cut from')
        materials(p.parse_args(argv[1:]));return
    if argv[:1]==['read-check']:
        p=argparse.ArgumentParser(description=read_check.__doc__)
        for name in ('pack','predictions','output'):p.add_argument('--'+name,type=Path,required=True)
        a=p.parse_args(argv[1:]);read_check(a.pack,a.predictions,a.output);return
    if argv[:1]==['human']:
        p=argparse.ArgumentParser(description=human_pairs.__doc__);p.add_argument('--review-set',type=Path,required=True)
        p.add_argument('--source',type=Path,help='the synthetic pairs replayed and checked against (none: the human regions alone)');p.add_argument('--arm',choices=ARMS,default='gold_pixels')
        p.add_argument('--no-replay',action='store_true',help='the human regions alone, without the synthetic replay')
        p.add_argument('--output-dir',type=Path,required=True)
        a=p.parse_args(argv[1:]);human_pairs(a.review_set,a.source,a.arm,a.output_dir,a.no_replay);return
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','mapping','same-student','human-validation','gold-bank','detections','templates','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pages',type=Path,action='append',required=True);p.add_argument('--copies',type=int,default=4);p.add_argument('--min-score',type=float,default=.8)
    p.add_argument('--real-geometry',action='store_true',help='v2: gold pieces at their real size against the writing, not squeezed into a strip')
    p.add_argument('--lighter',type=float,default=0.,help='with --real-geometry: the strike this many grey levels above the writing\'s median')
    for name,help_ in (('stretch','length'),('thicken','thickness')):
        p.add_argument('--'+name,type=float,default=1.,help=f'with --real-geometry: times the piece\'s real {help_}')
    p.add_argument('--passes',type=int,default=1,help='with --real-geometry: drawn back and forth up to this many times')
    p.add_argument('--wave',type=float,default=0.,help='with --real-geometry: its path bent by up to this many writing heights')
    build(p.parse_args(argv))


if __name__=='__main__':main()
