"""Offline polygon selection followed by batch model prefill in those exact regions."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ...data.clean_pages import review_clean
from ...data.pixel_masks import SCHEMA, encode, reviewed_crop
from ...data.prior_annotations import twins, writer_splits
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest


def select(args):
    from .pixel_review import ASSETS,HERE,write_assets
    splits=writer_splits(args.mapping,args.same_student);other=twins(args.same_student)
    canonical=lambda w:min(w,other.get(w,w))
    excluded=set()
    for path in args.exclude_export:
        excluded.update(canonical(k.split('_')[0]) for k in json.loads(path.read_text())['pages'])
    selected=[]
    for q in json.loads(args.selection.read_text())['pages']:
        writer=q['image_id'].split('_')[0];group=canonical(writer)
        if q['split']!='train' or splits.get(writer)!='train' or splits.get(other.get(writer,writer),'train')!='train' or group in excluded:continue
        if not (args.pages/(q['image_id']+'.png')).exists():continue
        selected.append(q);excluded.add(group)
        if len(selected)==args.count:break
    require(selected,'no new training questions available')
    out=make_output_dir(args.output_dir);(out/'images').mkdir()
    inputs={str(p):digest(p) for p in [args.selection,args.mapping,args.same_student,*args.exclude_export,*(HERE/n for n in ASSETS)]}
    pages=[]
    for q in selected:
        path=args.pages/(q['image_id']+'.png')
        with Image.open(path) as im:clean=review_clean(np.asarray(im.convert('L')))
        name='images/'+q['image_id']+'.png';Image.fromarray(clean).save(out/name);inputs[str(path)]=digest(path)
        pages.append({'id':q['image_id'],'question':q['question'],'size':[clean.shape[1],clean.shape[0]],'image':name})
    write_json(out/'pack.json',{'id':args.name,'schema':'paper2-pixel-review-pack-v1','task':'region_selection','pages':pages,'rows':[]})
    write_json(out/'selection.json',{'pages':selected});write_assets(out)
    write_manifest(out,config=to_jsonable(args),inputs=inputs,questions=len(pages),regions=0,human_labels=False)
    print(json.dumps({'questions':len(pages),'regions':0}))


def prefill(args):
    from ... import infer
    from ...eval.review_removal import predict_crops
    from .pixel_review import ASSETS,HERE,write_assets
    payload=json.loads((args.pack/'pack.json').read_text());export=json.loads(args.regions_export.read_text())
    require(export.get('schema')==SCHEMA and export.get('pack')==payload['id'] and export.get('task')=='region_selection','export must be the matching region-selection pack')
    known={p['id'] for p in payload['pages']};require(set(export['pages'])<=known,'unknown pages in region export')
    require(any(p.get('revisions') for p in export['pages'].values()),'no selected regions')
    model,_=infer.load_checkpoint(args.checkpoint)
    out=make_output_dir(args.output_dir);(out/'images').mkdir();rows=[];pages=[];seen=set()
    inputs={str(p):digest(p) for p in [args.pack/'pack.json',args.regions_export,args.checkpoint,*(HERE/n for n in ASSETS)]}
    for p in payload['pages']:
        revisions=[r for r in export['pages'].get(p['id'],{}).get('revisions',[]) if not r.get('deleted')]
        if not revisions:continue
        path=args.pack/p['image'];inputs[str(path)]=digest(path)
        with Image.open(path) as im:
            crops=[]
            for r in revisions:
                require(r['id'] not in seen,'duplicate region id');seen.add(r['id'])
                require(not r.get('confirmed') and not r.get('runs'),'refuse to overwrite pixel annotations')
                crop,_=reviewed_crop(im,{**r,'runs':[]});crops.append(crop)
            name='images/'+p['id']+'.png';im.save(out/name)
        masks=predict_crops(model,crops);pi=len(pages);pages.append({**p,'image':name})
        for r,mask in zip(revisions,masks):
            rows.append({'id':r['id'],'page':pi,'box':r['box'],'region':r['region'],'runs':encode(mask),
                         'text':r.get('text',''),'note':r.get('note',''),'confirmed':False})
    write_json(out/'pack.json',{'id':payload['id']+'-prefilled','schema':'paper2-pixel-review-pack-v1','task':'strike_only','pages':pages,'rows':rows})
    write_assets(out);write_manifest(out,config=to_jsonable(args),inputs=inputs,questions=len(pages),regions=len(rows),human_labels=False)
    print(json.dumps({'questions':len(pages),'regions':len(rows),'checkpoint':str(args.checkpoint)}))
