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
            image.convert('RGB').save(out/'pages'/f'{qi}.png')
            grey=np.asarray(image.convert('L'))
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
    for name in ['pixel_editor.html','pixel_editor.js']:
        (out/('index.html' if name.endswith('.html') else name)).write_text((HERE/name).read_text())
    write_manifest(out,config=to_jsonable(args),inputs={str(p):digest(p) for p in
                   [args.prefill/'prefill.json',checkpoint or args.masks/'run_manifest.json',HERE/'pixel_editor.html',HERE/'pixel_editor.js']},
                   questions=len(pages),revisions=len(rows),human_labels=False)
    print(json.dumps({'questions':len(pages),'revisions':len(rows),'url':'http://127.0.0.1:8769/'}))


def crop_request(root, payload, body):
    page=body['page']; box=body['box']
    require(type(page) is int and 0<=page<len(payload['pages']),'invalid page')
    require(len(box)==4 and all(type(v) is int for v in box),'invalid crop')
    info=payload['pages'][page];w,h=info['size'];l,t,r,b=box
    require(0<=l<r<=w and 0<=t<b<=h and (r-l)*(b-t)<=2000000,'crop outside page or too large')
    with Image.open(root/info['image']) as source:
        grey=np.asarray(source.convert('L').crop(tuple(box)))
    return grey,decode(body['runs'],grey.shape)


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
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('pack');p.add_argument('--prefill',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--masks',type=Path);source.add_argument('--checkpoint',type=Path)
    p.add_argument('--pages',type=Path,action='append',required=True);p.add_argument('--first',type=int,default=5)
    p.add_argument('--name',required=True);p.add_argument('--output-dir',type=Path,required=True)
    s=sub.add_parser('serve');s.add_argument('--pack',type=Path,required=True);s.add_argument('--checkpoint',type=Path,required=True)
    s.add_argument('--port',type=int,default=8769)
    args=parser.parse_args(argv)
    return pack(args) if args.command=='pack' else serve(args)


if __name__=='__main__':
    main()
