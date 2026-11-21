"""Read the regions of a pack with a dedicated OCR model, untaught (plan of 2026-11-21, 专用OCR对照, its appendix): each region's crop,
byte for byte the pack's file, with one prompt of the model card; nothing else goes into the model.

    python -m revrender.recognize_ocr --pack PACK --model-dir DIR --kind glm_ocr|minicpm_ocr|unlimited_ocr [--prompt TEXT]
        --image raw|erased [--limit N] [--max-new-tokens 256] --output-dir OUT

Kinds (the appendix): ``glm_ocr`` (GLM-OCR) and ``minicpm_ocr`` (MiniCPM-OCR-1.0) through their processors' chat templates and
``generate`` (greedy), the prompt one of the card's (``Text Recognition:``, ``Formula Recognition:``); ``unlimited_ocr``
(Unlimited-OCR-2, its bundled code) through the card's ``model.infer`` with ``save_results=False`` and an output folder inside OUT,
the prompt the model's own page prompt.  The shell is taken off (``bare``): the special tokens the first two end with, the page and
block marks Unlimited-OCR-2 wraps its text in (``<PAGE>``, ``<BLOCK>kind[x1, y1, x2, y2]<FORK>``, blocks joined in order); formula
delimiters and spaces are left to ``eval.coverage.prepare``.  A reading left empty (or ``[空]``) is written as the reader's refusal
(``recognize_lines.UNREADABLE``) and counted.  The model's code runs with two safeguards set before anything loads (``safeguards``):
``eval`` called from model code (a model folder, or any path holding ``transformers_modules``) fails, and any network connection
fails; the hub is offline.  OUT/predictions.jsonl as ``recognize_lines`` writes it (with the raw output, why it stopped, and the
sha256 of the image file given); OUT/run_manifest.json the call, the counts and the safeguards.
"""
import argparse
import builtins
import hashlib
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

MODEL_CODE = ('/home/zsh/hme/models/', 'transformers_modules')
BLOCK = re.compile(r'<BLOCK>[^<\[]*\[[^\]]*\]<FORK>')


def safeguards():
    """Refuse eval from model code and every network connection in this process; the hub offline."""
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    real = builtins.eval

    def guarded(*a, **k):
        frame = sys._getframe(1); caller = frame.f_code.co_filename
        if caller.startswith(MODEL_CODE[0]) or MODEL_CODE[1] in caller:
            raise RuntimeError(f'SAFEGUARD: eval called from model code {caller}')
        if len(a) == 1 and not k:                       # no namespaces given: the caller's, as eval itself would take
            return real(a[0], frame.f_globals, frame.f_locals)
        return real(*a, **k)
    builtins.eval = guarded

    def refuse(*a, **k):
        raise RuntimeError('SAFEGUARD: network connection attempted')
    socket.socket.connect = refuse; socket.create_connection = refuse


def bare(kind, raw):
    """The reading without the model's shell (see the module's words)."""
    if kind == 'unlimited_ocr':
        text = raw.replace('<PAGE>', '')
        parts = [p.strip() for p in BLOCK.split(text)]
        return '\n'.join(p for p in parts if p).strip()
    return raw.strip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--pack', type=Path, required=True); p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--kind', choices=('glm_ocr', 'minicpm_ocr', 'unlimited_ocr'), required=True)
    p.add_argument('--prompt', default='', help="the card's prompt (glm_ocr, minicpm_ocr); unlimited_ocr builds its own")
    p.add_argument('--image', choices=('raw', 'erased'), required=True); p.add_argument('--limit', type=int)
    p.add_argument('--max-new-tokens', type=int, default=256); p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    safeguards()
    import torch
    from PIL import Image
    from .recognize_lines import UNREADABLE
    from .util import digest, make_output_dir, to_jsonable, write_manifest
    rows = json.loads((args.pack/'pack.json').read_text())['rows'][:args.limit]
    out = make_output_dir(args.output_dir); started = time.time()
    if args.kind == 'unlimited_ocr':
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
        model = AutoModel.from_pretrained(str(args.model_dir), trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16).eval().cuda()
        scratch = out/'model_scratch'; scratch.mkdir()

        def read(path):
            text = model.infer(tokenizer, image_file=str(path), output_path=str(scratch), save_results=False, max_page_tokens=args.max_new_tokens)
            return text, 'stop'
        call = f"model.infer(tokenizer, image_file, output_path=OUT/model_scratch, save_results=False, max_page_tokens={args.max_new_tokens})"
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if not args.prompt.strip():
            raise SystemExit('--prompt is needed for this kind')
        processor = AutoProcessor.from_pretrained(str(args.model_dir))
        model = AutoModelForImageTextToText.from_pretrained(str(args.model_dir), dtype=torch.bfloat16, device_map={'': 0}).eval()
        stop = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])

        def read(path):
            msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': Image.open(path).convert('RGB')}, {'type': 'text', 'text': args.prompt}]}]
            batch = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(0)
            batch.pop('token_type_ids', None)
            with torch.no_grad():
                ids = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
            new = ids[0, batch['input_ids'].shape[1]:].tolist()
            return processor.decode(new, skip_special_tokens=True), 'stop' if new and new[-1] in stop else 'length'
        call = f"processor.apply_chat_template([image, {args.prompt!r}]) -> model.generate(max_new_tokens={args.max_new_tokens}, greedy), decoded without special tokens"
    condition = f'{args.kind}|{args.prompt or "page"}|{args.image}'; refusals = 0
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for r in rows:
            path = args.pack/r['images'][args.image]
            raw, finish = read(path)
            text = bare(args.kind, raw)
            if not text or text == '[空]':
                text, refusals = UNREADABLE, refusals+1
            stream.write(json.dumps({'condition': condition, 'id': r['id'], 'line_id': r.get('line_id', r['id']), 'model': args.model_dir.name, 'text': text,
                                     'raw': raw, 'finish_reason': finish, 'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, ensure_ascii=False)+'\n')
            stream.flush()
    import transformers
    write_manifest(out, config=to_jsonable(args), inputs={str(args.pack/'pack.json'): digest(args.pack/'pack.json')}, call=call, condition=condition,
                   regions=len(rows), empty_written_as_refusal=refusals, transformers=transformers.__version__,
                   safeguards='eval from model code refused; network connections refused; HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set',
                   seconds=round(time.time()-started))
    print(json.dumps({'condition': condition, 'regions': len(rows), 'empty_written_as_refusal': refusals}), flush=True)


if __name__ == '__main__':
    main()
