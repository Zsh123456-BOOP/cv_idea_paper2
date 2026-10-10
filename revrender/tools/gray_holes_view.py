"""Read-only 100-region comparison of first-paper mask erasure and gray holes."""
import argparse
import base64
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

from .. import pages
from ..data.pixel_masks import SCHEMA, decode, region_pixels
from ..gray_holes import Settings, process
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .tighten_view import window

ARMS={'direct':'第一篇掩膜直接擦除','global':'① 整体墨迹深浅','joint':'② 整体＋局部深浅',
      'structure':'③ 深浅＋交叉结构','tight':'旧：收缩 1 像素＋去浅灰',
      'gray':'旧：局部灰度挖洞','linked':'旧：灰度＋两侧支持'}
HOLE_BASES={'global':'direct','joint':'direct','structure':'direct','gray':'tight','linked':'tight'}
TEMPLATE=Path(__file__).with_suffix('.html')


def png(array):
    stream=BytesIO();Image.fromarray(array).save(stream,format='PNG',optimize=True)
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


def erase(grey,mask):
    out=grey.copy();out[mask]=255;return out


def overlay(grey,mask,holes):
    rgb=np.repeat(grey[:,:,None],3,axis=2).astype(float)
    for select,color in ((mask,(222,40,60)),(holes,(0,160,100))):
        rgb[select]=.35*rgb[select]+.65*np.array(color)
    return rgb.astype(np.uint8)


def confusion(pred,gold,ink):
    return {'tp':int((pred&gold&ink).sum()),'fp':int((pred&~gold&ink).sum()),
            'fn':int((~pred&gold&ink).sum()),'valid':int((~gold&ink).sum())}


def metrics(counts):
    tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
    return {**counts,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'valid_damage':fp/counts['valid'] if counts['valid'] else None}


def load_rows(exports,frames,count):
    rows=[];seen=set()
    for file in exports:
        d=json.loads(file.read_text());require(d.get('schema')==SCHEMA,'pixel review export required')
        for page,q in d['pages'].items():
            for r in q.get('revisions',[]):
                if not r.get('confirmed') or r.get('deleted'):continue
                require((page,r['id']) not in seen,'duplicate confirmed region');seen.add((page,r['id']))
                rows.append({'page':page,'row':r,'confirmed':True,'source':str(file)})
    require(len(rows)<=count,'requested count would drop confirmed examples')
    data=json.loads(frames.read_text());candidates=[]
    for page,q in data['pages'].items():
        for r in q.get('revisions',[]):
            if not r.get('deleted') and (page,r['id']) not in seen:
                candidates.append({'page':page,'row':r,'confirmed':False,'source':str(frames)})
    random.Random(SEED).shuffle(candidates)
    rows+=candidates[:count-len(rows)]
    require(len(rows)==count,'not enough distinct regions')
    return rows


def run(args):
    selected=load_rows(args.review_export,args.frames_export,args.count)
    frame_pack=json.loads((args.frame_pack/'pack.json').read_text())
    frame_images={p['id']:args.frame_pack/p['image'] for p in frame_pack['pages']}
    inputs={str(p):digest(p) for p in [*args.review_export,args.frames_export,args.frame_pack/'pack.json',
                                       args.detector/'run_manifest.json',Path(__file__),TEMPLATE,
                                       Path(__file__).parents[1]/'gray_holes.py']}
    cache={};cards=[];native={};totals={a:dict(tp=0,fp=0,fn=0,valid=0) for a in ARMS}
    useful={a:dict(base=base,saved_valid=0,retained_strike=0) for a,base in HOLE_BASES.items()}
    ink_groups={'erase':[],'keep':[]};thumbs=[]
    for n,item in enumerate(selected):
        page=item['page'];r=item['row']
        if page not in cache:
            image=args.images/(page+'.png') if item['confirmed'] else frame_images[page]
            record,instances=pages.read_output(args.detector,page)
            with Image.open(image) as im:grey=np.asarray(im.convert('L'))
            require(list(grey.shape[::-1])==record['size'],'scan and detector coordinates differ')
            cache[page]=(grey,instances)
            for p in [image,args.detector/(page+'.json'),args.detector/(page+'.npz')]:inputs[str(p)]=digest(p)
        whole,instances=cache[page];l,t,rr,b=r['box'];h,w=whole.shape
        box=[max(0,l-12),max(0,t-12),min(w,rr+12),min(h,b+12)]
        crop=whole[box[1]:box[3],box[0]:box[2]].copy()
        domain=region_pixels(r['region'],box);prior=np.zeros(crop.shape,bool)
        for inst in instances:prior |= window(inst,box)
        result=process(crop,prior,domain)
        card={'n':n+1,'id':r['id'],'page':page,'confirmed':item['confirmed'],'source':item['source'],
              'box':box,'region':r['region'],'statistics':result['statistics'],'metrics':{},
              'images':{'original':png(crop),'prior':png(overlay(crop,prior&domain,np.zeros_like(prior)))}}
        for arm in ARMS:
            card['images'][arm]=png(erase(crop,result[arm]))
            native[f'{n}_{arm}']=np.packbits(result[arm],axis=-1)
        card['protected']={}
        for arm,base in HOLE_BASES.items():
            holes=result[base]&~result[arm]
            card['protected'][arm]=int(holes.sum())
            card['images'][arm+'_overlay']=png(overlay(crop,result[arm],holes))
            native[f'{n}_{arm}_holes']=np.packbits(holes,axis=-1)
        if item['confirmed']:
            gold=window({'box':r['box'],'mask':decode(r['runs'],(b-t,rr-l))},box)&domain&(crop<245)
            if r.get('not_revision'):gold[:]=False
            known=domain&(crop<245)
            require(not r.get('uncertainMask',{}).get('runs'),'unknown pixels require separate evaluation support')
            card['images']['human']=png(erase(crop,gold))
            for arm in ARMS:
                c=confusion(result[arm],gold,known);card['metrics'][arm]=metrics(c)
                for key in c:totals[arm][key]+=c[key]
            for arm,base in HOLE_BASES.items():
                holes=result[base]&~result[arm]
                useful[arm]['saved_valid']+=int((holes&~gold&known).sum())
                useful[arm]['retained_strike']+=int((holes&gold&known).sum())
            ink_groups['erase'].append(crop[result['direct']&gold])
            ink_groups['keep'].append(crop[result['direct']&~gold])
        native[f'{n}_shape']=np.array(crop.shape)
        cards.append(card)
        if n<12:
            samples=[crop,erase(crop,result['direct']),erase(crop,result['structure']),
                     erase(crop,gold) if item['confirmed'] else crop]
            strips=[]
            for a in samples:
                im=Image.fromarray(a).convert('RGB');im.thumbnail((350,125));canvas=Image.new('RGB',(350,145),'white');canvas.paste(im,(0,18));strips.append(canvas)
            sheet=Image.new('RGB',(1400,145),'#ddd')
            for col,im in enumerate(strips):sheet.paste(im,(col*350,0))
            ImageDraw.Draw(sheet).text((4,2),f'{n+1}: original | direct | structure | human',fill='black');thumbs.append(sheet)
        print(f'{n+1}/{len(selected)}',flush=True)
    distributions={}
    for name,arrays in ink_groups.items():
        values=np.concatenate(arrays)
        distributions[name]={'pixels':len(values),'median_gray':float(np.median(values)) if len(values) else None,
                             'gray_le_10':float((values<=10).mean()) if len(values) else None}
    summary={'count':len(cards),'confirmed':sum(c['confirmed'] for c in cards),'framed_only':sum(not c['confirmed'] for c in cards),
             'settings':asdict(Settings()),'arms':{a:metrics(c) for a,c in totals.items()},'holes_effect':useful,
             'gray_distributions_inside_detector':distributions,'zero_detection':sum(c['statistics']['prior_ink']==0 for c in cards),
             'protocol':'Exploratory comparison; fixed parameters, no fitting or model selection on these labels. Metrics only on confirmed regions; these are 50 prior training + 16 prior validation regions, not an independent test set.'}
    out=make_output_dir(args.output_dir)
    write_json(out/'summary.json',summary);write_json(out/'selection.json',[{k:v for k,v in c.items() if k!='images'} for c in cards])
    np.savez_compressed(out/'proposals.npz',**native)
    payload=json.dumps({'summary':summary,'arms':ARMS,'hole_bases':HOLE_BASES,'cards':cards},ensure_ascii=False).replace('<','\\u003c')
    (out/'对比.html').write_text(TEMPLATE.read_text().replace('__DATA__',payload))
    sheet=Image.new('RGB',(1400,145*len(thumbs)),'white')
    for i,im in enumerate(thumbs):sheet.paste(im,(0,i*145))
    sheet.save(out/'contact.png')
    write_manifest(out,config=to_jsonable(args),inputs=inputs,settings=asdict(Settings()),human_annotations_modified=False,
                   regions=len(cards),confirmed=summary['confirmed'])
    require(all(digest(Path(p))==sha for p,sha in inputs.items()),'input file changed during read-only run')
    print(json.dumps(summary,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-export',type=Path,action='append',required=True)
    p.add_argument('--frames-export',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--frame-pack',type=Path,required=True)
    p.add_argument('--detector',type=Path,required=True)
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
