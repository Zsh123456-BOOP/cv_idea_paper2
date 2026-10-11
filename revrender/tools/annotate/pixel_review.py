"""Native-pixel stroke review pack and loopback-only click-refinement service.

pack: prefill from a box remover on training questions; serve: a trained conditional
refiner provides local suggestions. Browser exports are read by data.page_labels.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from ... import pages as pages_module
from ...box_remover import crop_of
from ...data.pixel_masks import encode, decode
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

HERE=Path(__file__).parent
ASSETS = ('pixel_editor.html', 'pixel_editor.js', 'pixel_rules.js', 'pixel_overview.js', 'pixel_viewports.js', 'page_rules.js',
          'library_store.js', 'symbol_picker.js', 'pixel_text.js')


def write_assets(out):
    for name in ASSETS:
        (out/('index.html' if name.endswith('.html') else name)).write_text((HERE/name).read_text())


def pack(args):
    prefill=json.loads((args.prefill/'prefill.json').read_text())
    selected=prefill['questions'][:args.first or None]
    require(selected and all(q['split']=='train' for q in selected),'this review pilot accepts training questions only')
    paths=pages_module.page_files(args.pages,[q['id'] for q in selected])
    checkpoint=getattr(args,'checkpoint',None)
    if checkpoint:
        from ... import infer
        from ...box_remover import read_boxes
        model,_=infer.load_checkpoint(checkpoint)
    out=make_output_dir(args.output_dir);(out/'pages').mkdir()
    pages=[];rows=[]
    for qi,q in enumerate(selected):
        with Image.open(paths[q['id']]) as image:
            grey=np.asarray(image.convert('L'))
            if getattr(args,'review_clean',False):
                from ...data.clean_pages import review_clean
                grey=review_clean(grey)
            Image.fromarray(grey).convert('RGB').save(out/'pages'/f'{qi}.png')
        if checkpoint:
            boxes=[crop_of(r['region'],(grey.shape[1],grey.shape[0])) for r in q['revisions'] if r['kind']=='line']
            instances=read_boxes(model,{'mask_threshold':.4,'nms':.5,'score_threshold':0},grey,boxes,top=1) if boxes else []
        else:
            record,instances=pages_module.read_output(args.masks,q['id'])
            require(tuple(record['size'])==(grey.shape[1],grey.shape[0]),'prediction and scan size differ')
        union=np.zeros(grey.shape,bool)
        for item in instances:
            l,t,r,b=item['box'];union[t:b,l:r]|=item['mask']
        union &= grey < 245
        pages.append({'id':q['id'],'question':q['question'],'image':f'pages/{qi}.png','size':[grey.shape[1],grey.shape[0]]})
        for revision in q['revisions']:
            if revision['kind']!='line':
                continue
            box=crop_of(revision['region'],(grey.shape[1],grey.shape[0]));l,t,r,b=box
            rows.append({'id':f'{q["id"]}#{revision["id"]}','page':qi,'box':box,'region':revision['region'],
                         'runs':encode(union[t:b,l:r]),'text':revision.get('text','')})
    require(rows,'no struck-line regions in the pilot')
    payload={'id':args.name,'schema':'paper2-pixel-review-pack-v1','pages':pages,'rows':rows}
    write_json(out/'pack.json',payload)
    write_assets(out)
    write_manifest(out,config=to_jsonable(args),inputs={str(p):digest(p) for p in
                   [args.prefill/'prefill.json',checkpoint or args.masks/'run_manifest.json',*(HERE/n for n in ASSETS)]},
                   questions=len(pages),revisions=len(rows),human_labels=False)
    print(json.dumps({'questions':len(pages),'revisions':len(rows),'url':'http://127.0.0.1:8769/'}))


def refresh(args):
    """Keep row IDs and native coordinates while moving an existing pack onto cleaned scans.

    The old pack remains intact. Keeping its ID lets the editor migrate browser
    corrections (with a recoverable backup) instead of replacing them by predictions.
    """
    source = args.pack/'pack.json'
    payload = json.loads(source.read_text())
    clean_manifest = args.clean_pages/'run_manifest.json'
    revision = digest(clean_manifest)[:16]
    inputs = {str(p): digest(p) for p in [source, clean_manifest, *(HERE/n for n in ASSETS)]}
    images = []
    removed = 0
    for pi, page in enumerate(payload['pages']):
        path = args.clean_pages/(page['id']+'.png')
        with Image.open(path) as image:
            require(list(image.size) == page['size'], 'clean page dimensions changed')
            grey = np.asarray(image.convert('L')).copy()
        images.append(grey)
        inputs[str(path)] = digest(path)
        for row in payload['rows']:
            if row['page'] != pi:
                continue
            l,t,r,b = row['box']
            mask = decode(row['runs'], (b-t,r-l))
            kept = mask & (grey[t:b,l:r] < 245)
            removed += int((mask & ~kept).sum())
            row['runs'] = encode(kept)
        page['image'] = f'pages/{pi}.png'
    out = make_output_dir(args.output_dir)
    (out/'pages').mkdir()
    for pi, grey in enumerate(images):
        Image.fromarray(grey).convert('RGB').save(out/'pages'/f'{pi}.png')
    payload['image_revision'] = revision
    payload['image_source'] = str(args.clean_pages)
    write_json(out/'pack.json', payload)
    write_assets(out)
    write_manifest(out, config=to_jsonable(args), inputs=inputs, questions=len(images),
                   revisions=len(payload['rows']), removed_prefill_pixels=removed, human_labels=False)
    print(json.dumps({'questions':len(images), 'revisions':len(payload['rows']), 'removed_prefill_pixels':removed}))


def crop_request(root, payload, body):
    page=body['page']; box=body['box']
    require(type(page) is int and 0<=page<len(payload['pages']),'invalid page')
    require(len(box)==4 and all(type(v) is int for v in box),'invalid crop')
    info=payload['pages'][page];w,h=info['size'];l,t,r,b=box
    require(0<=l<r<=w and 0<=t<b<=h and (r-l)*(b-t)<=2000000,'crop outside page or too large')
    with Image.open(root/info['image']) as source:
        grey=np.asarray(source.convert('L').crop(tuple(box)))
    if 'domain_runs' in body:
        domain=decode(body['domain_runs'],grey.shape)
        grey=grey.copy();grey[~domain]=255
    return grey,decode(body['runs'],grey.shape) & (grey < 245)


def serve(args):
    import torch
    from ...models.stroke_refiner import StrokeRefiner
    from ...refine import refine_crop
    torch.set_num_threads(4)
    root=args.pack.resolve(); payload=json.loads((root/'pack.json').read_text())
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    require(checkpoint.get('model')=='StrokeRefiner','not a click-refiner checkpoint')
    model=StrokeRefiner().eval();model.load_state_dict(checkpoint['state_dict'],strict=True)

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control','no-store')
            super().end_headers()

        def do_POST(self):
            if self.path!='/refine':
                self.send_error(404);return
            try:
                origin=self.headers.get('Origin')
                require(not origin or urlsplit(origin).hostname in ('localhost','127.0.0.1'),'cross-origin request refused')
                require(self.headers.get('Content-Type','').startswith('application/json'),'JSON required')
                size=int(self.headers.get('Content-Length','0'));require(0<size<=4000000,'request too large')
                body=json.loads(self.rfile.read(size))
                grey,mask=crop_request(root,payload,body)
                result,_=refine_crop(model,grey,mask,body['clicks'],radius=min(60,max(4,int(body.get('radius',28)))))
                data=json.dumps({'runs':encode(result),'model_step':checkpoint['step']}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
                self.end_headers();self.wfile.write(data)
            except (ValueError,KeyError,TypeError,RuntimeError) as error:
                data=json.dumps({'error':str(error)}).encode()
                self.send_response(400);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)

    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(root)))
    print(f'review service on 127.0.0.1:{args.port}, refiner step {checkpoint["step"]}',flush=True)
    server.serve_forever()


def main(argv=None):
    from .pixel_offline import standalone, update_html
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('pack');p.add_argument('--prefill',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--masks',type=Path);source.add_argument('--checkpoint',type=Path)
    p.add_argument('--pages',type=Path,action='append',required=True);p.add_argument('--first',type=int,default=5)
    p.add_argument('--review-clean',action='store_true',help='use connected-ink cleaned pages for both model input and display')
    p.add_argument('--name',required=True);p.add_argument('--output-dir',type=Path,required=True)
    r=sub.add_parser('refresh',help='preserve pack identity and use existing cleaned pages, writing a new directory')
    r.add_argument('--pack',type=Path,required=True);r.add_argument('--clean-pages',type=Path,required=True)
    r.add_argument('--output-dir',type=Path,required=True)
    o=sub.add_parser('standalone',help='clean scans and embed imported progress in one offline HTML')
    o.add_argument('--pack',type=Path,required=True);o.add_argument('--review-export',type=Path)
    o.add_argument('--output-dir',type=Path,required=True)
    u=sub.add_parser('update-html',help='update an offline editor without changing images or draft identity')
    u.add_argument('--source',type=Path,required=True);u.add_argument('--output-dir',type=Path,required=True)
    f=sub.add_parser('select-regions',help='blank clean questions for human polygon selection')
    f.add_argument('--selection',type=Path,required=True);f.add_argument('--pages',type=Path,required=True)
    f.add_argument('--mapping',type=Path,required=True);f.add_argument('--same-student',type=Path,required=True)
    f.add_argument('--exclude-export',type=Path,action='append',default=[]);f.add_argument('--count',type=int,default=30)
    f.add_argument('--name',required=True);f.add_argument('--output-dir',type=Path,required=True)
    f=sub.add_parser('prefill-regions',help='model prefill inside human-selected polygons')
    f.add_argument('--pack',type=Path,required=True);f.add_argument('--regions-export',type=Path,required=True)
    f.add_argument('--checkpoint',type=Path,required=True);f.add_argument('--output-dir',type=Path,required=True)
    s=sub.add_parser('serve');s.add_argument('--pack',type=Path,required=True);s.add_argument('--checkpoint',type=Path,required=True)
    s.add_argument('--port',type=int,default=8769)
    args=parser.parse_args(argv)
    if args.command in ('select-regions','prefill-regions'):
        from . import pixel_regions
        return (pixel_regions.select if args.command=='select-regions' else pixel_regions.prefill)(args)
    return {'pack':pack, 'refresh':refresh, 'serve':serve, 'standalone':standalone, 'update-html':update_html}[args.command](args)


if __name__=='__main__':
    main()
