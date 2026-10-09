"""Validation-only evaluation of released restoration weights; no optimization.

The candidate catalogue is frozen before inference. Human pixel validation and
transcription validation remain separate, and test questions cannot enter the
transcription input pack. All downloaded weights and derived images stay outside
Git. Attribute-guided CycleGANs requiring oracle labels are not eligible here.
"""
import argparse
import configparser
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages
from ..data.clean_pages import review_clean
from ..eval.revision_boxes import prepare, read_labels
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .restoration import module_file, removal_counts, removal_metrics, restore_native, verified_source


def validation_revisions(labels, selection):
    """Only completed, readable validation regions, never train or test."""
    selected={r['image_id']:r for r in selection['pages']}
    for image_id,q in sorted(labels.items()):
        if selected.get(image_id,{}).get('split')!='val' or not q.get('done'):continue
        for r in q['revisions']:
            if r.get('deleted') or r.get('unreadable') or not prepare(r.get('text','')).replace('?',''):continue
            yield image_id,r


def prepare_inputs(args):
    out=make_output_dir(args.output_dir);(out/'crops').mkdir()
    labels=read_labels(args.labels);selection=json.loads(args.selection.read_text())
    revisions=list(validation_revisions(labels,selection));require(revisions,'no readable validation regions')
    files=pages.page_files(args.pages,sorted({i for i,_ in revisions}));cache={};rows=[]
    for image_id,r in revisions:
        if image_id not in cache:
            with Image.open(files[image_id]) as im:cache[image_id]=review_clean(np.asarray(im.convert('L')))
        grey=cache[image_id];points=np.asarray(r['region'])
        l,t=np.maximum(0,points.min(0)-6).astype(int)
        rr,b=np.minimum([grey.shape[1],grey.shape[0]],points.max(0)+6).astype(int)
        require(rr>l and b>t,'empty validation crop')
        rel=f'crops/{len(rows):05d}.png';Image.fromarray(grey[t:b,l:rr]).save(out/rel)
        key=image_id+'#'+r['id']
        rows.append(dict(id='scan|'+key,line_id=key,image_id=image_id,model='scan',kind=r['kind'],
                         text=r['text'],boxes=[],size=[int(rr-l),int(b-t)],images={'erased':rel}))
    write_json(out/'pack.json',{'schema':'revrender-line-pack','split':'val','rows':rows})
    write_manifest(out,config=to_jsonable(args),inputs={str(p):digest(p) for p in [*args.labels,args.selection,*files.values()]},
                   protocol='validation only; completed questions; cleaned PNG; annotated bounding box plus six pixels',
                   questions=len(files),regions=len(rows),line_regions=sum(r['kind']=='line' for r in rows))
    print(json.dumps({'questions':len(files),'regions':len(rows),'line_regions':sum(r['kind']=='line' for r in rows)}))


def candidates(weights):
    """Author checkpoint names, dimensions and task variants; no fitted settings."""
    result=[]
    def add(name,model,paths,**kw):
        paths=[weights/p for p in paths]
        if not all(p.is_file() for p in paths):return
        result.append(dict(name=name,model=model,weights=[str(p.relative_to(weights)) for p in paths],
                           sha256=[digest(p) for p in paths],layout='native256',**kw))
    for group,short in [('ORIGINAL','Original'),('ORIGINAL_WITH_AUX_DISCR','Aux')]:
        pattern='CycleGAN/checkpoints/strikethrough_removal/'+group+'/*/genStrikeToClean_best_fmeasure.pth'
        for i,p in enumerate(sorted(weights.glob(pattern)),1):
            cfg=configparser.ConfigParser();cfg.read(p.parent/'config.cfg')
            require(cfg['DEFAULT']['blockcount']=='1' and cfg['DEFAULT']['model']=='dense','unsupported CycleGAN architecture')
            require(cfg['DEFAULT'].getboolean('invert_images'),'unexpected CycleGAN polarity')
            add(f'CycleGAN-{short}-{i:02d}','Strikethrough-CycleGAN',[p.relative_to(weights)],source_config=str((p.parent/'config.cfg').relative_to(weights)))
            result[-1]['layout']='cycle-official'
    add('DocDiff-deblur','DocDiff',['DocDiff/init.pth','DocDiff/denoiser.pth'])
    add('DocDiff-seal','DocDiff',['DocDiff/seal_init.pth','DocDiff/seal_denoiser.pth'])
    add('NAFNet-GoPro32','NAFNet',['NAFNet-GoPro-width32.pth'])
    add('NAFNet-SIDD32','NAFNet',['NAFNet-SIDD-width32.pth'],architecture=dict(type='NAFNet',img_channel=3,width=32,
        enc_blk_nums=[2,2,4,8],middle_blk_num=12,dec_blk_nums=[2,2,2,2]))
    add('Restormer-motion','Restormer',['restormer_motion_deblurring.pth'])
    add('Restormer-denoise','Restormer',['restormer_real_denoising.pth'],architecture={'LayerNorm_type':'BiasFree'})
    for task,file,big in [('watermark','watermark_rem_weights.h5',512),('deblur','deblur_weights.h5',1024),
                          ('binarize','binarization_generator_weights.h5',1024)]:
        found=list(weights.glob('DE-GAN/**/'+file))
        require(len(found)<=1,'ambiguous DE-GAN checkpoint')
        if found:add('DE-GAN-'+task,'DE-GAN',[found[0].relative_to(weights)],biggest_layer=big,binarize=task=='binarize')
    require(result,'no downloaded checkpoints found')
    return result


def catalogue(args):
    out=make_output_dir(args.output_dir);rows=candidates(args.weights)
    write_json(out/'candidates.json',rows)
    write_manifest(out,config=to_jsonable(args),inputs={str(args.weights/p):d for r in rows for p,d in zip(r['weights'],r['sha256'])},
        selection='maximum validation line character readback; tie: strict exact, lower valid-ink damage, lexical name',
        preprocessing='CycleGAN author pad-scale to 128x512 then inverse map; others native256 stride192; no UI residual cleanup',
        exclusions='COR: no verified public checkpoint; CycleGAN attribute-guided models require oracle types and are excluded',
        limitations='checkpoint selection on small validation sets; not a held-out test claim; inference seed42, author training seeds retained')
    print(json.dumps({'candidates':len(rows),'names':[r['name'] for r in rows]}))


def released_predictor(spec,weights,upstream):
    model=spec['model'];source=verified_source(upstream/model)
    paths=[weights/p for p in spec['weights']]
    require([digest(p) for p in paths]==spec['sha256'],'downloaded checkpoint changed')
    if model=='DE-GAN':
        import os,random
        os.environ['TF_USE_LEGACY_KERAS']='1'
        import tensorflow as tf
        from tf_keras.src import backend
        class IntegerBoundsRandom(random.Random):
            def randint(self,a,b):return super().randint(int(a),int(b))
        tf.keras.utils.set_random_seed(SEED);backend._SEED_GENERATOR.generator=IntegerBoundsRandom(SEED)
        devices=tf.config.list_physical_devices('GPU');require(len(devices)==1,'one GPU required')
        tf.config.experimental.set_memory_growth(devices[0],True)
        m=module_file('official_degan_released',upstream/model/'models/models.py')
        network=m.generator_model(biggest_layer=spec['biggest_layer']);network.load_weights(str(paths[0]))
        def predict(x):
            value=np.clip(network(x[...,None],training=False).numpy()[...,0],0,1)
            return (value>.95).astype(np.float32) if spec.get('binarize') else value
    else:
        import torch
        from .restoration_torch import build,predict_image
        torch.manual_seed(SEED);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        network,diffusion,_=build(model,upstream/model,spec.get('architecture'))
        def state(path):
            value=torch.load(path,map_location='cpu',weights_only=True)
            if isinstance(value,dict) and 'params' in value:value=value['params']
            if all(k.startswith('module.') for k in value):value={k[7:]:v for k,v in value.items()}
            return value
        if model=='DocDiff':
            network.init_predictor.load_state_dict(state(paths[0]),strict=True)
            network.denoiser.load_state_dict(state(paths[1]),strict=True)
        else:network.load_state_dict(state(paths[0]),strict=True)
        network.eval()
        def predict(x):return predict_image(network,diffusion,x,model)
    return predict,source


def restore(predict,grey,layout):
    if layout=='native256':return restore_native(predict,grey)
    require(layout=='cycle-official','unknown input layout')
    # Exact author PadToSize(256,1024), Resize(128,512), with inverse geometry.
    h,w=grey.shape;iw=min(1024,math.ceil(w*256/h))
    im=Image.fromarray(np.rint(grey*255).astype(np.uint8)).resize((iw,256),Image.Resampling.BICUBIC)
    padded=Image.new('L',(1024,256),255);padded.paste(im)
    xx=np.asarray(padded.resize((512,128),Image.Resampling.BILINEAR),np.float32)/255
    value=predict(xx[None])[0]
    # Float interpolation avoids quantizing network predictions before scoring.
    restored=Image.fromarray(value.astype(np.float32)).resize((1024,256),Image.Resampling.BILINEAR)
    return np.asarray(restored.crop((0,0,iw,256)).resize((w,h),Image.Resampling.BILINEAR))


def infer(args):
    specs=json.loads(args.catalogue.read_text());spec=next(s for s in specs if s['name']==args.candidate)
    out=make_output_dir(args.output_dir);predict,source=released_predictor(spec,args.weights,args.upstream)
    native_rows=json.loads(args.human_validation.read_text());native=np.load(args.human_validation.with_suffix('.npz'))
    counts=np.zeros(5,np.int64);outputs={};per_region=[]
    for i,r in enumerate(native_rows):
        im=native[f'image_{i}'].astype(np.float32)/255;gold=native[f'mask_{i}']
        value=restore(predict,im,spec['layout']);require(np.isfinite(value).all(),'nonfinite restoration')
        count=removal_counts(im,value,gold);counts+=count
        outputs[f'image_{i}']=np.rint(value*255).astype(np.uint8)
        per_region.append({'id':r['id'],**removal_metrics(count)})
    np.savez_compressed(out/'native.npz',**outputs)
    pixels=dict(model=spec['name'],regions=len(native_rows),per_region=per_region,**removal_metrics(counts))
    write_json(out/'pixels.json',pixels)
    source_pack=json.loads((args.input_pack/'pack.json').read_text());require(source_pack.get('split')=='val','validation pack required')
    pack=make_output_dir(out/'pack');(pack/'crops').mkdir();rows=[]
    for r in source_pack['rows']:
        path=(args.input_pack/r['images']['erased']).resolve()
        with Image.open(path) as im:raw=np.asarray(im.convert('L'),np.float32)/255
        value=restore(predict,raw,spec['layout']);require(np.isfinite(value).all(),'nonfinite OCR crop')
        local=f'crops/{len(rows):05d}.png';Image.fromarray(np.rint(value*255).astype(np.uint8)).save(pack/local)
        rows.extend([{**r,'images':{'erased':str(path)}},
            {**r,'id':spec['name']+'|'+r['line_id'],'model':spec['name'],'images':{'erased':local}}])
    write_json(pack/'pack.json',{'schema':'revrender-line-pack','split':'val','rows':rows})
    inputs=[args.catalogue,args.human_validation,args.human_validation.with_suffix('.npz'),args.input_pack/'run_manifest.json']
    write_manifest(out,config=to_jsonable(args),inputs={str(p):digest(p) for p in inputs},candidate=spec,upstream=source,
                   protocol='released weights; zero optimization; validation only; no test access')
    print(json.dumps({'model':spec['name'],'pixel_f1':pixels['f1'],'damage':pixels['valid_ink_damage'],'ocr_regions':len(rows)//2}),flush=True)


def report(args):
    import base64
    out=make_output_dir(args.output_dir)
    pack=json.loads((args.pack/'pack.json').read_text())
    readings={r['id']:r for r in (json.loads(s) for s in args.readings.read_text().splitlines())}
    require(set(readings)=={r['id'] for r in pack['rows']},'missing or extra OCR readings')
    scores=json.loads(args.scores.read_text())['held'];models=[]
    for path in args.runs:
        pixels=json.loads((path/'pixels.json').read_text())
        manifest=json.loads((path/'run_manifest.json').read_text());name=pixels['model']
        score=scores[name+'/erased_neutral/line']
        models.append(dict(name=name,family=manifest['candidate']['model'],pixels=pixels,ocr=score,candidate=manifest['candidate']))
    models.sort(key=lambda r:(-r['ocr']['held'],-(r['ocr']['exact'] or 0),r['pixels']['valid_ink_damage'],r['name']))
    baseline=scores['scan/erased_neutral/line']
    require(models and all(m['ocr']['revisions']==baseline['revisions'] for m in models),'unmatched validation denominators')
    summary=dict(models=models,baseline=baseline,selected=models[0]['name'],
        selected_improves_raw=models[0]['ocr']['held']>baseline['held'],
        truncated=sum(r.get('finish_reason')=='length' for r in readings.values()),
        selection='validation line readback; tie strict exact, lower damage, lexical name',
        limitations=['validation selection only, not final test performance','small validation sample; multiple released candidates',
                     'human pixel masks cannot establish hidden-stroke correctness','no finetuning or UI residual cleanup'])
    write_json(out/'summary.json',summary)
    samples={}
    for r in pack['rows']:
        sample=samples.setdefault(r['line_id'],dict(id=r['line_id'],truth=r['text'],kind=r['kind'],variants={}))
        path=Path(r['images']['erased']);path=path if path.is_absolute() else args.pack/path
        sample['variants'][r['model']]=dict(read=readings[r['id']]['text'],
            image='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode())
    payload=json.dumps(dict(summary=summary,samples=list(samples.values())),ensure_ascii=False).replace('<','\\u003c')
    (out/'index.html').write_text(REPORT_HTML.replace('__DATA__',payload),encoding='utf-8')
    write_manifest(out,config=to_jsonable(args),inputs={str(p):digest(p) for p in
        [args.pack/'pack.json',args.readings,args.scores,*[r/'run_manifest.json' for r in args.runs]]})
    print(json.dumps({'selected':summary['selected'],'held':models[0]['ocr']['held'],'raw_held':baseline['held']}))


REPORT_HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>官方权重 · 验证集对比</title>
<style>body{font:16px/1.65 system-ui;max-width:1200px;margin:28px auto;padding:0 20px;background:#f5f7fb;color:#182637}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{padding:7px;border-bottom:1px solid #dde3eb;text-align:left}.card{padding:16px;background:white;border:1px solid #dce2eb;border-radius:8px;margin:14px 0}img{display:block;width:100%;height:auto;max-height:440px;object-fit:contain;image-rendering:auto}.text{white-space:pre-wrap;overflow-wrap:anywhere}select,button{font:inherit;padding:8px;margin:3px}.bar{position:sticky;top:0;background:#e9eff8;padding:8px}small{color:#54647a}</style>
<h1>官方预训练权重 · 验证集对比</h1><p>直接加载作者权重，无重新训练。只使用验证集；这些数值不能作为最终测试成绩。</p><p id="summary"></p><div id="table"></div>
<p><small>按被划字符读回率选择；同分看严格全对，再看文字误擦。读回率不惩罚所有多余字符，需结合原图和完整转写检查。像素验证与转写验证是两套标注，不代表能确认完全遮住的笔画。</small></p>
<div class="bar"><select id="model"></select><button id="prev">上一处</button><select id="sample"></select><button id="next">下一处</button></div>
<div class="card"><b>人工转写</b><div id="truth" class="text"></div></div><div class="card"><b>原图</b><img id="raw"><div id="rawread" class="text"></div></div><div class="card"><b id="restitle"></b><img id="restored"><div id="restread" class="text"></div></div>
<script>const D=__DATA__;const $=id=>document.getElementById(id);const pct=x=>(x*100).toFixed(2)+'%';
$('summary').textContent='转写验证 '+D.summary.baseline.revisions+' 处划线；原图读回 '+pct(D.summary.baseline.held)+'；本轮选中 '+D.summary.selected+(D.summary.selected_improves_raw?'。':'，但读回未超过原图。');
const table=document.createElement('table');let row=table.insertRow();for(const v of ['权重版本','读回↑','严格全对↑','像素 F1↑','划线去除↑','正常文字误擦↓']){const c=document.createElement('th');c.textContent=v;row.append(c)}
for(const m of D.summary.models){row=table.insertRow();for(const v of [m.name,pct(m.ocr.held),pct(m.ocr.exact||0),pct(m.pixels.f1),pct(m.pixels.recall),pct(m.pixels.valid_ink_damage)])row.insertCell().textContent=v;$('model').add(new Option(m.name,m.name))}$('table').append(table);
D.samples.forEach((s,i)=>$('sample').add(new Option((i+1)+' · '+s.truth.slice(0,45),i)));function draw(){const s=D.samples[+$('sample').value],m=$('model').value,a=s.variants.scan,b=s.variants[m];$('truth').textContent=s.truth;$('raw').src=a.image;$('rawread').textContent=a.read;$('restitle').textContent=m;$('restored').src=b.image;$('restread').textContent=b.read}
$('prev').onclick=()=>{$('sample').selectedIndex=Math.max(0,$('sample').selectedIndex-1);draw()};$('next').onclick=()=>{$('sample').selectedIndex=Math.min(D.samples.length-1,$('sample').selectedIndex+1);draw()};$('model').onchange=$('sample').onchange=draw;draw();</script></html>'''


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    for k in ('labels','pages'):prep.add_argument('--'+k,type=Path,action='append',required=True)
    prep.add_argument('--selection',type=Path,required=True)
    cat=sub.add_parser('catalogue');cat.add_argument('--weights',type=Path,required=True)
    inf=sub.add_parser('infer');inf.add_argument('--candidate',required=True)
    for k in ('catalogue','weights','upstream','human-validation','input-pack'):inf.add_argument('--'+k,type=Path,required=True)
    rep=sub.add_parser('report')
    for k in ('pack','readings','scores'):rep.add_argument('--'+k,type=Path,required=True)
    rep.add_argument('--runs',type=Path,nargs='+',required=True)
    for parser in (prep,cat,inf,rep):parser.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    {'prepare':prepare_inputs,'catalogue':catalogue,'infer':infer,'report':report}[args.command](args)


if __name__=='__main__':main()
