"""Confirmed native review crops plus a small synthetic replay set for box-remover tuning."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .pixel_masks import SCHEMA, reviewed_crop
from .prior_annotations import twins, writer_splits
from .synthetic_set import read_set, strip
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest


def confirmed(paths, folders, mapping, same_student):
    """Explicit region confirmations are sufficient; whole-question completion is not invented."""
    splits=writer_splits(mapping,same_student);other=twins(same_student)
    rows=[];inputs={str(p):digest(p) for p in [*paths,mapping,same_student]};seen=set();skipped=0
    for path in paths:
        data=json.loads(path.read_text());require(data.get('schema')==SCHEMA,'pixel review export required')
        for key,page in data['pages'].items():
            chosen=[r for r in page.get('revisions',[]) if r.get('confirmed') and not r.get('deleted')]
            skipped+=len(page.get('revisions',[]))-len(chosen)
            if not chosen:continue
            writer=key.split('_')[0];group=min(writer,other.get(writer,writer))
            require(splits.get(writer)=='train' and splits.get(other.get(writer,writer),'train')=='train','held-out or unknown writer in review')
            candidates=[p/(key+'.png') for p in folders if (p/(key+'.png')).exists()]
            require(len(candidates)==1,'need exactly one cleaned scan per confirmed question')
            scan=candidates[0];inputs[str(scan)]=digest(scan)
            with Image.open(scan) as im:
                for r in chosen:
                    require(r['id'] not in seen,'duplicate reviewed region');seen.add(r['id'])
                    image,mask=reviewed_crop(im,r)
                    require(mask.any() or r.get('not_revision'),'empty positive confirmation')
                    rows.append({'id':r['id'],'group_uuid':group,'page':key,'image':image,'mask':mask,'region':r['region'],'box':r['box']})
    require(rows,'no confirmed regions')
    return rows,inputs,skipped


def held_writers(rows,count):
    writers=sorted({r['group_uuid'] for r in rows});random.Random(SEED).shuffle(writers)
    require(0<count<len(writers),'need separate review training and validation writers')
    return set(writers[:count])


def build(args):
    rows,inputs,skipped=confirmed(args.review_export,args.pages,args.mapping,args.same_student)
    held=held_writers(rows,args.validation_writers);other=twins(args.same_student)
    canonical=lambda w:min(w,other.get(w,w))
    # Also exclude the review validation writers from every replay row.
    records,images,labels=read_set(args.replay)
    splits=writer_splits(args.mapping,args.same_student)
    all_human={r['group_uuid'] for r in rows}
    candidates=[i for i,r in enumerate(records) if r['split']=='train' and canonical(r['group_uuid']) not in all_human
                and splits.get(r['group_uuid'],'unused') in ('train','unused')
                and splits.get(other.get(r['group_uuid']),'unused') in ('train','unused')]
    rng=random.Random(SEED);rng.shuffle(candidates)
    positive=[i for i in candidates if records[i]['revisions']>0]
    multi=[i for i in positive if records[i].get('source_instances',records[i]['revisions'])>=2]
    clean=[i for i in candidates if records[i]['revisions']==0]
    require(len(multi)>=2 and len(clean)>=args.replay_clean,'not enough replay learning strata')
    # Negatives for validation use the replay corpus's own held-out writers, disjoint from every human writer.
    validation=[i for i,r in enumerate(records) if r['split']=='val' and r['revisions']==0 and canonical(r['group_uuid']) not in all_human]
    rng.shuffle(validation);require(len(validation)>=args.validation_clean,'not enough held-out clean replay')
    count=sum(r['group_uuid'] not in held and r['mask'].any() for r in rows)
    selected=multi[:2]+[i for i in positive if i not in multi[:2]][:max(2,count)-2]+clean[:args.replay_clean]+validation[:args.validation_clean]
    data_images=[];data_labels=[];data_records=[];native={};val_records=[]
    for r in rows:
        image,mask=strip(Image.fromarray(r['image']),r['mask'])
        split='val' if r['group_uuid'] in held else 'train'
        data_images.append(image);data_labels.append(mask)
        data_records.append({'id':r['id'],'line_id':r['id'],'group_uuid':r['group_uuid'],'split':split,
                             'revisions':int(mask.any()),'source':'human_confirmed'})
        if split=='val':
            i=len(val_records);native[f'image_{i}']=r['image'];native[f'mask_{i}']=r['mask']
            val_records.append({k:v for k,v in r.items() if k not in ('image','mask')})
    for i in selected:
        r=records[i];data_images.append(np.asarray(images[i]));data_labels.append(np.asarray(labels[i]))
        data_records.append({**r,'group_uuid':canonical(r['group_uuid']),'source':'synthetic_replay'})
    # Check canonical identities, including replay's validation negatives, before writing.
    train={r['group_uuid'] for r in data_records if r['split']=='train'}
    val={r['group_uuid'] for r in data_records if r['split']=='val'}
    require(not train&val,'writer leakage between fine-tuning and validation')
    out=make_output_dir(args.output_dir)
    np.save(out/'images.npy',np.stack(data_images));np.save(out/'labels.npy',np.stack(data_labels))
    np.savez_compressed(out/'validation_native.npz',**native)
    write_json(out/'validation_native.json',val_records);write_json(out/'records.json',data_records)
    inputs[str(args.replay/'run_manifest.json')]=digest(args.replay/'run_manifest.json')
    summary={'confirmed':len(rows),'skipped_unconfirmed':skipped,'human_train':sum(r['group_uuid'] not in held for r in rows),
             'human_validation':len(val_records),'validation_writers':sorted(held),
             'counts':dict(Counter(r['source']+'/'+r['split']+('/positive' if r['revisions'] else '/clean') for r in data_records))}
    write_json(out/'summary.json',summary)
    files={name:digest(out/name) for name in ['images.npy','labels.npy','records.json','validation_native.npz','validation_native.json']}
    write_manifest(out,config=to_jsonable(args),inputs=inputs,files=files,**summary)
    print(json.dumps(summary),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-export',type=Path,action='append',required=True)
    p.add_argument('--pages',type=Path,action='append',required=True)
    p.add_argument('--mapping',type=Path,required=True);p.add_argument('--same-student',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--validation-writers',type=int,default=3)
    p.add_argument('--replay-clean',type=int,default=96);p.add_argument('--validation-clean',type=int,default=16)
    p.add_argument('--output-dir',type=Path,required=True)
    build(p.parse_args(argv))


if __name__=='__main__':main()
