"""Full detector coverage with model prefill, stable IDs and an offline review library."""
import argparse
import base64
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ...data.clean_pages import review_clean
from ...data.pixel_masks import SCHEMA, encode
from ...data.prior_annotations import twins, writer_splits
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .page_prefill import corners, region_of
from .pixel_offline import render_html, script_json


def manual_records(exports):
    """Only human-confirmed/edited/deleted rows protect work across model updates."""
    pages={}
    for path in exports:
        value=json.loads(path.read_text());require(value.get('schema')==SCHEMA,'pixel annotation export required')
        for page,q in value['pages'].items():
            target=pages.setdefault(page,{})
            for r in [*q.get('revisions',[]),*q.get('deleted_revisions',[])]:
                if r.get('confirmed') or r.get('ops',0)>0 or r.get('deleted'):
                    target[r['id']]=copy.deepcopy(r)
    return pages


def overlap(a,b):
    l,t,r,z=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    area=max(0,r-l)*max(0,z-t)
    return area/max(1,min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])))


def infer_all(args):
    from ... import infer, pages as page_io
    from ...eval.review_removal import predict_crops
    selected=json.loads(args.selected.read_text());checkpoint=Path(selected['checkpoint'])
    model,_=infer.load_checkpoint(checkpoint)
    split=writer_splits(args.mapping,args.same_student);other=twins(args.same_student)
    canon=lambda w:min(w,other.get(w,w))
    held={canon(r['group_uuid']) for r in json.loads(args.validation.read_text())}
    manual=manual_records(args.review_export)
    ids=sorted(p.stem for p in args.detector.glob('*.json') if p.stem!='run_manifest' and (args.pages/(p.stem+'.png')).exists())
    require(ids,'no detected pages');out=make_output_dir(args.output_dir);(out/'images').mkdir()
    inputs={str(p):digest(p) for p in [args.selected,checkpoint,args.mapping,args.same_student,args.validation,*args.review_export,args.detector/'run_manifest.json',Path(__file__)]}
    pages=[];rows=[];coverage=[];scanned=0
    for page in ids:
        record,instances=page_io.read_output(args.detector,page);scanned+=1
        if not instances and not manual.get(page):continue
        path=args.pages/(page+'.png')
        with Image.open(path) as im:grey=review_clean(np.asarray(im.convert('L')))
        h,w=grey.shape;require(record['size']==[w,h],'page dimensions differ')
        human=list(manual.get(page,{}).values());fresh=[];crops=[]
        for n,inst in enumerate(instances):
            l,t,r,b=inst['box'];seeds=np.zeros_like(grey,bool);seeds[t:b,l:r]=inst['mask']
            box=region_of(seeds,grey<245) or inst['box']
            # Keep the existing optimized line height where available, without extending
            # a localized revision across the entire line horizontally.
            line=next((v for v in record['lines'] if v.get('index')==inst['line']),None)
            if line:
                a,c,z,d=line['box'];box=[box[0],min(box[1],c),box[2],max(box[3],d)]
            box=[int(v) for v in box]
            covered=next((r for r in human if overlap(box,r['box'])>=.5),None)
            identity=f'{page}#det{n:04d}'
            coverage.append({'page':page,'detector_index':n,'row_id':covered['id'] if covered else identity,'preserved_manual':covered is not None})
            if covered:continue
            l,t,r,b=box;fresh.append({'id':identity,'box':box,'region':corners(box),'detector_index':n,'score':inst['score'],'confirmed':False})
            crops.append(grey[t:b,l:r])
        masks=predict_crops(model,crops) if crops else []
        pi=len(pages);writer=page.split('_')[0]
        role=split.get(writer,'train')
        if role=='dev' or split.get(other.get(writer,writer)) in ('val','dev'):role='val'
        if split.get(other.get(writer,writer))=='test':role='test'
        if canon(writer) in held:role='val'
        source='images/'+page+'.png';Image.fromarray(grey).save(out/source)
        pages.append({'id':page,'question':page.split('_')[-2],'image':source,'size':[w,h],'split':role,'writer_group':canon(writer)})
        for row,mask in zip(fresh,masks):rows.append({**row,'page':pi,'runs':encode(mask),'model':selected['selected']})
        for row in human:rows.append({**row,'page':pi,'preserved_manual':True})
        for f in (path,args.detector/(page+'.json'),args.detector/(page+'.npz')):inputs[str(f)]=digest(f)
        if len(pages)%100==0:print(json.dumps({'scanned':scanned,'pages':len(pages),'regions':len(rows)}),flush=True)
    require(set(manual)<=set(ids),'manual pages missing from detector/image sources')
    pack={'id':args.name,'schema':'paper2-pixel-review-pack-v1','task':'strike_only','pages':pages,'rows':rows,
          'model_revision':digest(checkpoint),'image_revision':'review-clean-v1','image_source':str(out/'images'),'selected_checkpoint':str(checkpoint),
          'selection':selected,'protected_validation_writers':sorted(held),'coverage':coverage}
    write_json(out/'pack.json',pack)
    summary={'scanned_pages':scanned,'pages':len(pages),'detected_masks':len(coverage),'regions':len(rows),
             'preserved_manual':sum(bool(r.get('preserved_manual')) for r in rows),'confirmed':sum(bool(r.get('confirmed')) for r in rows),
             'splits':dict(Counter(p['split'] for p in pages)),'zero_masks':sum(not r['runs'] for r in rows),'selected':selected['selected']}
    write_json(out/'summary.json',summary);write_manifest(out,config=to_jsonable(args),inputs=inputs,**summary)
    require(all(digest(Path(p))==h for p,h in inputs.items()),'inputs changed during inference')
    (out/'COMPLETE').touch();print(json.dumps(summary),flush=True)


def bundle(args):
    from .pixel_review import HERE
    source=json.loads((args.pack/'pack.json').read_text());out=make_output_dir(args.output_dir)
    (out/'parts').mkdir();parts=[];initial={};lookup={}
    for start in range(0,len(source['pages']),args.pages_per_part):
        batch=source['pages'][start:start+args.pages_per_part];number=len(parts);identity=source['id']+'-'+hashlib.sha256('|'.join(p['id'] for p in batch).encode()).hexdigest()[:12]
        rows=[{**r,'page':r['page']-start} for r in source['rows'] if start<=r['page']<start+len(batch)]
        pages=copy.deepcopy(batch)
        for page in pages:
            page['image']='data:image/png;base64,'+base64.b64encode((args.pack/page['image']).read_bytes()).decode()
            lookup[page['id']]=number
        originals={}
        for i,p in enumerate(pages):
            records=[r for r in rows if r['page']==i and r.get('preserved_manual')]
            if records:
                originals[p['id']]={'revisions':[r for r in records if not r.get('deleted')],'deleted_revisions':[r for r in records if r.get('deleted')],'done':False}
                initial[p['id']]=originals[p['id']]
        payload={k:source[k] for k in ('schema','task','model_revision','image_revision','image_source')}
        payload.update(id=identity,pages=pages,rows=rows,standalone=True,
                       library={'id':source['id'],'part':number,'parts':(len(source['pages'])+args.pages_per_part-1)//args.pages_per_part},
                       initial_export={'schema':SCHEMA,'pack':identity,'pages':originals})
        html=render_html(payload).replace('<script id="review-pack"','<script src="../library_store.js"></script><script id="review-pack"',1)
        (out/'parts'/f'{number:03d}.html').write_text(html)
        parts.append({'id':identity,'file':f'parts/{number:03d}.html','pages':len(pages),'regions':sum(not r.get('deleted') for r in rows),'confirmed':sum(bool(r.get('confirmed')) for r in rows),'splits':dict(Counter(p['split'] for p in pages))})
    meta={'id':source['id'],'parts':parts,'page_to_part':lookup,'initial_pages':initial,
          'page_splits':{p['id']:p['split'] for p in source['pages']},'image_source':source['image_source'],
          'selected_checkpoint':source['selected_checkpoint'],'validation':source['selection'],
          'protected_validation_writers':source['protected_validation_writers']}
    (out/'library_data.js').write_text('window.REVIEW_LIBRARY='+script_json(meta)+';')
    for name in ('library_store.js','pixel_library.html','pixel_library.js'):
        (out/('index.html' if name.endswith('.html') else name)).write_text((HERE/name).read_text())
    write_manifest(out,config=to_jsonable(args),inputs={str(args.pack/'pack.json'):digest(args.pack/'pack.json')},parts=len(parts),pages=len(source['pages']),regions=len(source['rows']))
    print(json.dumps({'parts':len(parts),'pages':len(source['pages']),'regions':len(source['rows']),'entry':str(out/'index.html')}))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    i=sub.add_parser('infer')
    for name in ('selected','pages','detector','mapping','same-student','validation','output-dir'):i.add_argument('--'+name,type=Path,required=True)
    i.add_argument('--review-export',type=Path,action='append',default=[]);i.add_argument('--name',default='paper2-full-review')
    b=sub.add_parser('bundle');b.add_argument('--pack',type=Path,required=True);b.add_argument('--output-dir',type=Path,required=True)
    b.add_argument('--pages-per-part',type=int,default=20)
    args=p.parse_args();infer_all(args) if args.command=='infer' else bundle(args)


if __name__=='__main__':main()
