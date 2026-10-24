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


def tiles(image, mask):
    """The tiles of a native crop (``canvas_tiles`` on white), each with its target: the crop with the pixels to remove made white."""
    target = image.copy(); target[mask.astype(bool)] = 255
    return list(zip(canvas_tiles(image, 255), canvas_tiles(target, 255)))


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
        add(tiles(arrays[f'image_{k}'], arrays[f'mask_{k}']), r['group_uuid'], r['id'], 'human_confirmed')
    for image_id in chosen:
        revisions = data['pages'][image_id]['revisions']
        require(revisions and all(r.get('confirmed') for r in revisions if not r.get('deleted')), f'{image_id}: a region is not confirmed')
        with Image.open(pages/(image_id+'.png')) as page:
            if whole:
                grey, gold, _ = question_gold(page, revisions); add(tiles(grey, gold), image_id.split('_')[0], image_id, 'whole_question')
            else:
                for r in revisions:
                    if r.get('confirmed') and not r.get('deleted'):
                        image, mask = reviewed_crop(page, r); add(tiles(image, mask), image_id.split('_')[0], image_id+'#'+r['id'], 'question_region')
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


def human_pairs(review_set, source, arm, destination, no_replay=False, detector=None):
    """The box remover's fine-tuning set (``data.review_set``) for the restorers, on their own native tiles: its human training regions
    tiled (``tiles``), its synthetic replay as the very pairs those strips were cut from (``as_boxes``'s ``pair_index``), and the pairs'
    own validation split (writers apart from the training ones) for their synthetic check.  The 16 fixed validation regions stay the
    restorers' human validation, as in their synthetic training.  ``no_replay``: the human regions alone.  ``detector``: a detection
    run whose masks hint each human tile (``data.separation.detector_hint``, hint.npy; a synthetic tile's hint is simulated)."""
    from .separation import detector_hint
    out = make_output_dir(destination)
    records = json.loads((review_set/'records.json').read_text())
    native = json.loads((review_set/'training_native.json').read_text()); arrays = np.load(review_set/'training_native.npz')
    pairs = json.loads((source/'records.json').read_text())
    x = np.load(source/(arm+'.npy'), mmap_mode='r'); y = np.load(source/'targets.npy', mmap_mode='r')
    data, targets, rows, hints = [], [], [], []
    for k, r in enumerate(native):
        if detector:
            hints += canvas_tiles(detector_hint(pages.read_output(detector, r['page'])[1], r['box'], arrays[f'image_{k}']), 0)
        for n, (tile, target) in enumerate(tiles(arrays[f'image_{k}'], arrays[f'mask_{k}'])):
            data.append(tile); targets.append(target)
            rows.append({'id': f"{r['id']}~{n}", 'group_uuid': r['group_uuid'], 'split': 'train', 'source': 'human_confirmed',
                         'clean': bool((tile == target).all())})
    replay = [] if no_replay else [r for r in records if r['source'] == 'synthetic_replay' and r['split'] == 'train']
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
    if detector: np.save(out/'hint.npy', np.stack(hints + [np.zeros((SIZE, SIZE), np.uint8)]*(len(rows)-len(hints))))
    summary = {'counts': dict(Counter(r['source']+'/'+r['split'] for r in rows)), 'human_regions': len(native),
               'train_tiles': sum(r['split'] == 'train' for r in rows), 'review_set': str(review_set), 'source_pairs': str(source)}
    write_json(out/'summary.json', summary)
    inputs = {str(p): digest(p) for p in [review_set/'run_manifest.json', review_set/'training_native.npz', source/'run_manifest.json',
                                          *([detector/'run_manifest.json'] if detector else [])]}
    write_manifest(out, config={'review_set': str(review_set), 'source': str(source), 'arm': arm, 'no_replay': no_replay,
                                'detector': str(detector) if detector else None}, inputs=inputs,
                   files={p.name: digest(p) for p in out.glob('*.npy')} | {'records.json': digest(out/'records.json')}, **summary)
    print(json.dumps(summary), flush=True)


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
    if argv[:1]==['human']:
        p=argparse.ArgumentParser(description=human_pairs.__doc__);p.add_argument('--review-set',type=Path,required=True)
        p.add_argument('--source',type=Path,required=True);p.add_argument('--arm',choices=ARMS,default='gold_pixels')
        p.add_argument('--no-replay',action='store_true',help='the human regions alone, without the synthetic replay')
        p.add_argument('--detector',type=Path,help="a detection run: its masks hint the human tiles (hint.npy, for RevSep)")
        p.add_argument('--output-dir',type=Path,required=True)
        a=p.parse_args(argv[1:]);human_pairs(a.review_set,a.source,a.arm,a.output_dir,a.no_replay,a.detector);return
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
