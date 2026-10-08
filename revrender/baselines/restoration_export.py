"""Export selected restoration checkpoints on fixed labelled test regions.

Oracle region crops isolate restoration from detection. They are not an end-to-end
page-detection score. Test results never select weights or thresholds.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages
from ..data.clean_pages import review_clean
from ..eval.revision_boxes import read_labels, prepare, unpacked
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .restoration import module_file, removal_counts, removal_metrics, restore_native, verified_source


def selected_predictor(run,upstream):
    manifest=json.loads((run/'run_manifest.json').read_text());model=manifest['config']['model']
    selection=json.loads((run/'selection.json').read_text());weights=run/selection['checkpoint']
    source=verified_source(upstream/model);require(source==manifest['upstream'],'upstream mismatch')
    if model=='DE-GAN':
        os.environ['TF_USE_LEGACY_KERAS']='1'
        import tensorflow as tf
        import random
        from tf_keras.src import backend
        class IntegerBoundsRandom(random.Random):
            def randint(self,a,b):return super().randint(int(a),int(b))
        tf.keras.utils.set_random_seed(SEED);backend._SEED_GENERATOR.generator=IntegerBoundsRandom(SEED)
        gpu=tf.config.list_physical_devices('GPU');require(len(gpu)==1,'one assigned GPU required')
        tf.config.experimental.set_memory_growth(gpu[0],True)
        m=module_file('paper2_degan_export',upstream/model/'models/models.py')
        network=m.generator_model(biggest_layer=1024);network.load_weights(str(weights/'generator.h5'))
        predict=lambda x:np.clip(network(x[...,None],training=False).numpy()[...,0],0,1)
        checkpoint=weights/'generator.h5'
    else:
        import torch
        from .restoration_torch import build, predict_image
        torch.manual_seed(SEED);torch.set_num_threads(4)
        network,diffusion,_=build(model,upstream/model);checkpoint=weights/'weights.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);network.load_state_dict(state['model']);network.eval()
        @torch.no_grad()
        def predict(x):
            return predict_image(network,diffusion,x,model)
    return predict,checkpoint,manifest


def export(args):
    out=make_output_dir(args.output_dir);predict,checkpoint,manifest=selected_predictor(args.run,args.upstream)
    name=manifest['config']['model']+'_'+manifest['config']['arm']
    labels=read_labels(args.labels);selection={r['image_id']:r for r in json.loads(args.selection.read_text())['pages']}
    ids=sorted(i for i,q in labels.items() if q.get('done') and selection.get(i,{}).get('split')=='test')
    require(ids,'no confirmed test questions')
    geometry_ids=sorted(p.stem for p in args.geometry.glob('*.json') if p.stem not in ('summary','run_manifest'))
    files=pages.page_files(args.pages,sorted(set(ids+geometry_ids)));images={}
    def scan(i):
        if i not in images:
            with Image.open(files[i]) as im:images[i]=review_clean(np.asarray(im.convert('L')))
        return images[i]
    pack=make_output_dir(out/'pack');(pack/'crops').mkdir();rows=[]
    for image_id in ids:
        grey=scan(image_id)
        for r in labels[image_id]['revisions']:
            if r.get('deleted') or r.get('unreadable') or not prepare(r.get('text','')).replace('?',''):continue
            # Same annotated box + 6px context as the earlier crop OCR experiment.
            points=np.asarray(r['region']);l,t=(int(v) for v in np.maximum(0,points.min(0)-6))
            rr,b=(int(v) for v in np.minimum([grey.shape[1],grey.shape[0]],points.max(0)+6))
            raw=grey[t:b,l:rr].astype(np.float32)/255;restored=restore_native(predict,raw)
            for model,pixels in [('scan',raw),(name,restored)]:
                path=f'crops/{len(rows):05d}.png';Image.fromarray(np.rint(pixels*255).astype(np.uint8)).save(pack/path)
                rows.append({'id':f'{model}|{image_id}#{r["id"]}','line_id':f'{image_id}#{r["id"]}',
                    'model':model,'image_id':image_id,'kind':r['kind'],'size':[rr-l,b-t],
                    'boxes':[],'images':{'erased':path},'text':r['text']})
    write_json(pack/'pack.json',{'schema':'revrender-line-pack','rows':rows})
    total=np.zeros(5,np.int64);regions=0
    for image_id in geometry_ids:
        grey=scan(image_id);record=json.loads((args.geometry/f'{image_id}.json').read_text())
        arrays=np.load(args.geometry/f'{image_id}.npz');domain=np.zeros(grey.shape,bool);gold=np.zeros(grey.shape,bool)
        restored=grey.astype(np.float32)/255
        for n,r in enumerate(record['revisions']):
            if r['kind']!='line':continue
            region=unpacked(arrays[f'region_{n}'],grey.shape[1]);ys,xs=np.nonzero(region)
            if not len(xs):continue
            l=max(0,int(xs.min())-6);t=max(0,int(ys.min())-6);rr=min(grey.shape[1],int(xs.max())+7);b=min(grey.shape[0],int(ys.max())+7)
            raw=grey[t:b,l:rr].astype(np.float32)/255;value=restore_native(predict,raw)
            local=region[t:b,l:rr];restored[t:b,l:rr][local]=value[local]
            domain|=region;gold|=unpacked(arrays[f'revision_{n}'],grey.shape[1]);regions+=1
        total+=removal_counts(grey[domain].astype(np.float32)/255,restored[domain],gold[domain])
    write_json(out/'pixels.json',{'questions':len(geometry_ids),'regions':regions,'model':name,**removal_metrics(total)})
    inputs={str(p):digest(p) for p in [args.run/'selection.json',args.run/'run_manifest.json',checkpoint,args.selection,*args.labels,
            *files.values(),*args.geometry.glob('*.json'),*args.geometry.glob('*.npz')]}
    write_manifest(out,config=to_jsonable(args),inputs=inputs,model=name,rows=len(rows),
        protocol='oracle labelled test-region crops; fixed validation-selected checkpoint; no test tuning')
    print(json.dumps({'model':name,'ocr_crops':len(rows)//2,'pixel_regions':regions}),flush=True)


def merge(args):
    out=make_output_dir(args.output_dir);rows={}
    for directory in args.inputs:
        for r in json.loads((directory/'pack/pack.json').read_text())['rows']:
            row={**r,'images':{k:str((directory/'pack'/v).resolve()) for k,v in r['images'].items()}}
            if r['id'] in rows:
                old=rows[r['id']];require(r['model']=='scan' and digest(old['images']['erased'])==digest(row['images']['erased']),'different duplicate reading input')
            else:rows[r['id']]=row
    write_json(out/'pack.json',{'schema':'revrender-line-pack','rows':list(rows.values())})
    write_manifest(out,config=to_jsonable(args),inputs={str(p/'run_manifest.json'):digest(p/'run_manifest.json') for p in args.inputs},rows=len(rows))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export')
    for name in ('run','upstream','selection','geometry','output-dir'):e.add_argument('--'+name,type=Path,required=True)
    for name in ('labels','pages'):e.add_argument('--'+name,type=Path,action='append',required=True)
    m=sub.add_parser('merge');m.add_argument('--inputs',type=Path,nargs='+',required=True);m.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv);(export if args.command=='export' else merge)(args)


if __name__=='__main__':main()
