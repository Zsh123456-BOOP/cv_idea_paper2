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
from ...data.pixel_masks import SCHEMA, decode, encode
from ...data.prior_annotations import twins, writer_splits
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from ...data.annotation_set import allowed_writers, refuse_outside
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


def duplicate_groups(rows):
    """Merge near-identical automatic crops, never neighbouring lines or human rows.

    Complete-link grouping avoids joining a chain of different regions. Original
    rows remain in the pack; the editor archives aliases only if nobody edited them.
    """
    def matches(a,b):
        x,y=a['box'],b['box']
        height=max(x[3]-x[1],y[3]-y[1])
        return (a['page']==b['page'] and overlap(x,y)>=.9
                and max(abs(x[1]-y[1]),abs(x[3]-y[3]))<=.2*height)
    by_page={}
    for row in rows:
        if not row.get('preserved_manual') and row.get('detector_index') is not None:
            by_page.setdefault(row['page'],[]).append(row)
    result=[]
    for candidates in by_page.values():
        groups=[]
        for row in candidates:
            group=next((g for g in groups if all(matches(row,r) for r in g)),None)
            if group is None:groups.append([row])
            else:group.append(row)
        for group in groups:
            if len(group)<2:continue
            box=[min(r['box'][0] for r in group),min(r['box'][1] for r in group),
                 max(r['box'][2] for r in group),max(r['box'][3] for r in group)]
            l,t,r,b=box;mask=np.zeros((b-t,r-l),bool)
            for row in group:
                a,c,z,d=row['box']
                mask[c-t:d-t,a-l:z-l] |= decode(row['runs'],(d-c,z-a))
            result.append({'ids':[r['id'] for r in group],'box':box,'region':corners(box),'runs':encode(mask)})
    return result


def infer_all(args):
    from ... import infer, pages as page_io
    from ...eval.review_removal import predict_crops
    selected=json.loads(args.selected.read_text());checkpoint=Path(selected['checkpoint'])
    model,_=infer.load_checkpoint(checkpoint)
    split=writer_splits(args.mapping,args.same_student);other=twins(args.same_student)
    canon=lambda w:min(w,other.get(w,w))
    held={canon(r['group_uuid']) for r in json.loads(args.validation.read_text())}
    manual=manual_records(args.review_export)
    # A page is read from the first detector folder that has it and the first page folder with its cleaned png.
    detector={};folder={}
    for d in args.detector:
        for f in d.glob('*.json'):
            if f.stem!='run_manifest':detector.setdefault(f.stem,d)
    for d in args.pages:
        for f in d.glob('*.png'):folder.setdefault(f.stem,d)
    wanted=set(args.ids.read_text().split()) if args.ids else None
    ids=sorted(p for p in detector if p in folder and (wanted is None or p in wanted))
    require(ids,'no detected pages');require(wanted is None or set(ids)==wanted,f'pages without detection or cleaned image: {sorted(wanted-set(ids))[:3]}')
    refuse_outside(ids,allowed_writers(args.allowed_writers))
    out=make_output_dir(args.output_dir);(out/'images').mkdir()
    inputs={str(p):digest(p) for p in [args.selected,checkpoint,args.mapping,args.same_student,args.validation,args.allowed_writers,*args.review_export,
                                       *(d/'run_manifest.json' for d in args.detector),*([args.ids] if args.ids else []),Path(__file__)]}
    pages=[];rows=[];coverage=[];scanned=0
    for page in ids:
        record,instances=page_io.read_output(detector[page],page);scanned+=1
        if not instances and not manual.get(page):continue
        path=folder[page]/(page+'.png')
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
        for f in (path,detector[page]/(page+'.json'),detector[page]/(page+'.npz')):inputs[str(f)]=digest(f)
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
    from .pixel_review import HERE, ASSETS
    source=json.loads((args.pack/'pack.json').read_text());out=make_output_dir(args.output_dir)
    (out/'parts').mkdir();parts=[];initial={};lookup={}
    for start in range(0,len(source['pages']),args.pages_per_part):
        batch=source['pages'][start:start+args.pages_per_part];number=len(parts);identity=source['id']+'-'+hashlib.sha256('|'.join(p['id'] for p in batch).encode()).hexdigest()[:12]
        rows=[{**r,'page':r['page']-start} for r in source['rows'] if start<=r['page']<start+len(batch)]
        duplicates=duplicate_groups(rows)
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
                       duplicate_groups=duplicates,
                       library={'id':source['id'],'part':number,'parts':(len(source['pages'])+args.pages_per_part-1)//args.pages_per_part},
                       initial_export={'schema':SCHEMA,'pack':identity,'pages':originals})
        html=render_html(payload)
        (out/'parts'/f'{number:03d}.html').write_text(html)
        parts.append({'id':identity,'file':f'parts/{number:03d}.html','pages':len(pages),'regions':sum(not r.get('deleted') for r in rows)-sum(len(g['ids'])-1 for g in duplicates),'confirmed':sum(bool(r.get('confirmed')) for r in rows),'splits':dict(Counter(p['split'] for p in pages))})
    meta={'id':source['id'],'title':args.title,'parts':parts,'page_to_part':lookup,'initial_pages':initial,
          'page_splits':{p['id']:p['split'] for p in source['pages']},'image_source':source['image_source'],
          'selected_checkpoint':source['selected_checkpoint'],'validation':source['selection'],
          'protected_validation_writers':source['protected_validation_writers']}
    (out/'library_data.js').write_text('window.REVIEW_LIBRARY='+script_json(meta)+';')
    for name in ('library_store.js','pixel_library.html','pixel_library.js'):
        (out/('index.html' if name.endswith('.html') else name)).write_text((HERE/name).read_text())
    inputs=[args.pack/'pack.json',Path(__file__),*(HERE/n for n in (*ASSETS,'pixel_library.js','pixel_library.html'))]
    write_manifest(out,config=to_jsonable(args),inputs={str(p):digest(p) for p in inputs},parts=len(parts),pages=len(source['pages']),regions=len(source['rows']),visible_regions=sum(p['regions'] for p in parts))
    print(json.dumps({'parts':len(parts),'pages':len(source['pages']),'regions':len(source['rows']),'entry':str(out/'index.html')}))


def prefill_export(args):
    """A pack's regions as a review export: the model's prefill, unconfirmed, with near-identical crops merged as the library merges
    them (``duplicate_groups``: the first id keeps the union), and the preserved human records as they were."""
    source=json.loads((args.pack/'pack.json').read_text())
    groups=duplicate_groups(source['rows']);alias={i for g in groups for i in g['ids'][1:]};merged={g['ids'][0]:g for g in groups}
    pages={p['id']:{'revisions':[],'deleted_revisions':[],'done':False} for p in source['pages']}
    for row in source['rows']:
        if row['id'] in alias:continue
        page=pages[source['pages'][row['page']]['id']];record={k:v for k,v in row.items() if k!='page'}
        if row['id'] in merged:
            g=merged[row['id']];record.update(box=g['box'],region=g['region'],runs=g['runs'],merged_detection_ids=g['ids'])
        page['deleted_revisions' if record.get('deleted') else 'revisions'].append(record)
    out=make_output_dir(args.output_dir)
    write_json(out/'export.json',{'schema':SCHEMA,'pack':source['id'],'task':source['task'],'model_revision':source['model_revision'],
                                  'selected_checkpoint':source['selected_checkpoint'],'pages':pages})
    summary={'pages':len(pages),'regions':sum(len(p['revisions']) for p in pages.values()),'merged_aliases':len(alias),
             'human':sum(bool(r.get('preserved_manual')) for r in source['rows'])}
    write_manifest(out,config=to_jsonable(args),inputs={str(args.pack/'pack.json'):digest(args.pack/'pack.json')},**summary)
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    i=sub.add_parser('infer')
    for name in ('selected','mapping','same-student','validation','allowed-writers','output-dir'):i.add_argument('--'+name,type=Path,required=True)
    i.add_argument('--pages',type=Path,action='append',required=True,help='a folder of cleaned pages (repeatable)')
    i.add_argument('--detector',type=Path,action='append',required=True,help='a folder of detections (repeatable)')
    i.add_argument('--ids',type=Path,help='only these pages, one id per line')
    i.add_argument('--review-export',type=Path,action='append',default=[]);i.add_argument('--name',default='paper2-full-review')
    b=sub.add_parser('bundle');b.add_argument('--pack',type=Path,required=True);b.add_argument('--output-dir',type=Path,required=True)
    b.add_argument('--pages-per-part',type=int,default=20)
    b.add_argument('--title',default='全部涂改 · 筛选与精修')
    e=sub.add_parser('prefill-export');e.add_argument('--pack',type=Path,required=True);e.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();{'infer':infer_all,'bundle':bundle,'prefill-export':prefill_export}[args.command](args)


if __name__=='__main__':main()
