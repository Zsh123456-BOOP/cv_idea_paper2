"""Teach a Qwen3-VL to give the final valid writing of a line struck out in places: LoRA on synthetic revisions.

    python -m revrender.train_reader --corpus corpus.jsonl --pages DIR [--pages DIR ...] --model MODEL_DIR --output-dir OUT
        [--kinds clean whole paste stack] [--blank] [--steps N] [--batch-size B] [--accumulate A] [--rank R] [--lr LR]

The samples are the recipes of ``data.reader_corpus`` drawn when they are read (``data.paste_strikes``): the line, with writing struck out and
put in or the whole line struck, and the text it should be read as -- the host's own, or nothing (``[空]``).  The words of the request are
those of ``recognize_lines`` (``--words``); the answer is ``<transcription>text</transcription>`` and only the answer carries the loss.  The
language model's linear layers get a low-rank adapter; the vision tower and everything else stay as they are.  ``--kinds clean`` trains on
the clean lines alone, the control that says how much of a gain is the writer's hand and the answer's style and how much is the strikes.
``--blank`` whites out the columns under the strokes' mask (the way ``recognize_lines`` conditions ``blanked_*`` do with a detector's masks) before
the sample is read, so the reader learns to read what is left around a gap; leave out ``stack``, whose column holds writing that is valid.
Seed 42; OUT/adapter is the adapter (``recognize_lines --backend hf --adapter OUT/adapter`` reads with it), OUT/train_log.jsonl the loss.

``--pack`` (in place of ``--corpus``) teaches on a line pack of real crops instead (``data.struck_pack train``: each row's ``images.raw``
and its ``text``, rows of split ``val`` the validation), asked in the words of ``--condition`` (e.g. ``raw_struck``: read only what is
struck out).  A pack of ``data.struck_pack folds`` is taught with ``--fold K``: the writers of fold K are left out altogether (they are
read afterwards, ``recognize_lines --fold K``), the rows marked ``val`` among the rest are the validation, and the answer is the row's
``target``.  The ``*_revision`` conditions show the images they name (``pair_revision`` two); the mapped ones need
``--layer-map``: the row's layer map goes in beside the scan through a projection of its own (``models.vlm_tokens``), taught at
``--map-lr`` and saved as OUT/layer_map.pt.  ``--keep-best`` keeps the adapter of the lowest validation loss instead of the last.
``--erased DIR`` takes every row's erased crop from DIR (``<row's place in the pack>.png``) instead of the pack.

``--separator RUN`` (with ``pair_revision``): the erased crop is made while training by that separator (``models.hint_separator``: a
NAFNet of ``baselines.restoration``, its hint never darker than the scan), and from step ``--separator-from`` on the separator learns
too, at ``--separator-lr`` (0: never), from the loss on the reading and ``--anchor`` times its hints' distance from the crops a person
erased (``--anchor-pack``).  OUT/separator.pt is the separator kept, OUT/hints its hint of every row of the pack (``--erased`` for the
reading and for another reader's teaching).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import pages as pages_module
from .data import lines as lines_module
from .data.line_pack import band_mask
from .data.paste_strikes import compose
from .recognize_lines import MAPPED, SYSTEM, image_keys, prompt
from .util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

EMPTY = '[空]'
TARGET = '<transcription>{}</transcription>'


def target_of(text):
    """The answer a request is taught to get: the text between the tags, ``[空]`` for nothing."""
    return TARGET.format(text.strip() if text.strip() else EMPTY)


def read_rows(corpus, kinds=None, split=None):
    rows = [json.loads(line) for line in Path(corpus).read_text(encoding='utf-8').splitlines() if line.strip()]
    return [r for r in rows if (not kinds or r['kind'] in kinds) and (not split or r['split'] == split)]


class Crops:
    """Line crops of the pages, by line id, as RGB arrays (a page is opened once per crop; the pages are big, the crops small)."""

    def __init__(self, folders, rows):
        self.files = pages_module.page_files(folders, sorted({r['image_id'] for r in rows}))
        self.boxes = {r['line_id']: r['box_1000'] for r in rows}

    def get(self, line_id, box_1000=None):
        image_id = line_id.split(':')[0]
        with Image.open(self.files[image_id]) as page:
            return np.asarray(page.convert('RGB').crop(lines_module.crop_box(box_1000 or self.boxes[line_id], *page.size)))


def render(row, crops, blank=False):
    """(PIL image, the text it should be read as) for a recipe row; with ``blank`` the columns under the strokes' mask are whited out."""
    host = crops.get(row['line_id'], row['box_1000'])
    donors = [crops.get(d) for d in row['donors']]
    made = compose(host, donors, random.Random(row['seed']), row['kind'])
    image = made['image'].convert('RGB')
    if blank and made['boxes']:
        array = np.array(image)
        array[band_mask(made['boxes'], image.size)] = 255
        image = Image.fromarray(array)
    return image, '' if made['kind'] == 'whole' else row['host_text']      # what could not be drawn comes out clean: the host's text


class Samples:
    def __init__(self, rows, crops, blank=False):
        self.rows, self.crops, self.blank = rows, crops, blank

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return render(self.rows[i], self.crops, self.blank)


class PackSamples:
    """The rows of a line pack: (the images a condition shows, the text to give, the layer map of a mapped condition or None, the crop
    a person erased or None).  ``erased``: a folder the erased crops are taken from instead, by the row's place in the pack (``places``);
    ``anchors``: the folder of a pack whose ``erased`` is a person's, with its rows by id."""

    def __init__(self, root, rows, keys=('raw',), mapped=False, erased=None, places=None, anchors=None):
        self.root, self.rows, self.keys, self.mapped, self.erased, self.places, self.anchors = root, rows, keys, mapped, erased, places, anchors

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = lambda key: self.erased/f"{self.places[row['id']]:05d}.png" if key == 'erased' and self.erased else self.root/row['images'][key]
        opened = lambda key: Image.open(path(key)).convert('RGB')
        anchor = Image.open(self.anchors[0]/self.anchors[1][row['id']]['images']['erased']).convert('L') if self.anchors else None
        return [opened(key) for key in self.keys], row.get('target', row['text']), opened('map') if self.mapped else None, anchor


class Collate:
    """A batch for the model: the chat with the answer, the loss only on the answer."""

    def __init__(self, processor, words, scans=False):
        self.processor, self.words, self.scans = processor, words, scans
        self.start = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')

    def __call__(self, batch):
        batch = [item if len(item) == 4 else ([item[0]], item[1], None, None) for item in batch]      # (images, text, layer map, anchor)
        anchors = [np.asarray(anchor) if anchor is not None else None for *_, anchor in batch]
        batch = [item[:3] for item in batch]
        conversations = [[{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
                          {'role': 'user', 'content': [*({'type': 'image', 'image': image} for image in images), {'type': 'text', 'text': self.words}]},
                          {'role': 'assistant', 'content': [{'type': 'text', 'text': target_of(text)}]}] for images, text, _ in batch]
        out = self.processor.apply_chat_template(conversations, tokenize=True, return_dict=True, return_tensors='pt', padding=True)
        out = {k: out[k] for k in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')}
        if batch[0][2] is not None:
            from .models.vlm_tokens import layer_maps, with_layer_maps
            out['pixel_values'] = with_layer_maps(out['pixel_values'], [m for images, _, layer in batch for m in layer_maps(images, layer)], self.processor.image_processor)
        labels = out['input_ids'].clone()
        for b in range(labels.shape[0]):
            marks = (out['input_ids'][b] == self.start).nonzero().flatten()
            labels[b, :int(marks[-1])+3] = -100                        # the system and user turns and "<|im_start|>assistant\n"
        labels[out['attention_mask'] == 0] = -100
        out['labels'] = labels
        if self.scans:                                             # for a separator inside the training (models.hint_separator.lay_hints)
            out['scans'], out['anchors'] = [np.asarray(images[0].convert('L')) for images, _, _ in batch], anchors
        return out


def lora_model(model_dir, rank, alpha, dropout, layer_map=False):
    """The model with a low-rank adapter on the language model's linear layers; with ``layer_map`` also the (trainable, zero) projection
    of a layer map into the patch embedding, added after the adapter froze everything else."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    from .models.vlm_tokens import patch_embed_as_linear, patch_embed_with_layer_map
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.bfloat16, attn_implementation='sdpa')
    patch_embed_as_linear(model.model.visual.patch_embed)
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias='none',
                        target_modules=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)')
    adapted = get_peft_model(model, config)
    if layer_map:
        patch_embed_with_layer_map(model.model.visual.patch_embed)
    return adapted


def layer_map_state(model):
    """The layer-map projection's weights, as ``models.vlm_tokens.Reader`` loads them."""
    return {name.rsplit('layer_map.', 1)[1]: q.detach().cpu() for name, q in model.named_parameters() if 'layer_map.' in name}


def cycle(loader):
    """Go through a loader again and again."""
    while True:
        yield from loader


def on_device(batch, device):
    """The batch's tensors on the device; what is not a tensor stays as it is."""
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}


def evaluate(model, loader, device, lay=None):
    import torch
    model.eval()
    total, count = 0., 0
    with torch.inference_mode():
        for batch in loader:
            batch = on_device(batch, device)
            if lay:
                lay(batch)
            batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
            tokens = int((batch['labels'] != -100).sum())
            total += float(model(**batch).loss)*tokens
            count += tokens
    model.train()
    return total/max(1, count)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--corpus', type=Path); source.add_argument('--pack', type=Path, help='a line pack of real crops and their text (data.struck_pack)')
    p.add_argument('--condition', help='with --pack: the recognize_lines condition whose words ask (e.g. raw_struck)')
    p.add_argument('--fold', type=int, help='with --pack: leave this fold of the pack out; the rows marked val among the rest validate')
    p.add_argument('--layer-map', action='store_true', help='with a mapped condition: teach the projection of the layer map')
    p.add_argument('--map-lr', type=float, default=2e-5, help='the learning rate of the layer-map projection: Adam moves each of a patch\'s '
                   '1,536 weights by about this a step, so a patch with strike in it shifts by this times its strike pixels (1e-3 wrecked the image in two steps)')
    p.add_argument('--keep-best', action='store_true', help='keep the adapter of the lowest validation loss, not the last')
    p.add_argument('--erased', type=Path, help="with --pack: a folder of erased crops (<the row's place in the pack>.png) in place of the pack's")
    p.add_argument('--separator', type=Path, help='with pair_revision: a NAFNet run of baselines.restoration that makes the erased crop while training')
    p.add_argument('--upstream', type=Path, help="--separator: the restorers' official sources")
    p.add_argument('--separator-lr', type=float, default=1e-4, help='--separator: its learning rate once it learns (0: it never does)')
    p.add_argument('--separator-from', type=int, default=40, help='--separator: the step from which it learns (the reader first learns to read)')
    p.add_argument('--anchor-pack', type=Path, help="--separator: a pack of the same rows whose erased crop is a person's (data.struck_pack folds)")
    p.add_argument('--anchor', type=float, default=10., help="--separator: the weight of its hints' distance from the anchor crops")
    p.add_argument('--pages', type=Path, action='append', help='with --corpus: a folder of scans <image_id>.jpg (repeatable)')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--kinds', nargs='+', choices=('clean', 'whole', 'paste', 'stack'), default=['clean', 'whole', 'paste', 'stack'])
    p.add_argument('--blank', action='store_true', help='white out the columns under the strokes before reading (not with stack)')
    p.add_argument('--words', choices=('neutral', 'final'), default='final', help='the words of the request (recognize_lines conditions raw_neutral / raw_final)')
    p.add_argument('--steps', type=int, default=600, help='optimizer steps')
    p.add_argument('--batch-size', type=int, default=8, help='samples per forward pass')
    p.add_argument('--accumulate', type=int, default=4, help='forward passes per optimizer step')
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--dropout', type=float, default=.05)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--val-limit', type=int, default=200, help='validation recipes (writers held out of training)')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--checkpointing', action='store_true', help='gradient checkpointing (the 8B model)')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    import torch
    from transformers import AutoProcessor
    if args.pack:
        require(args.condition is not None, '--pack needs --condition')
        require(args.layer_map == (args.condition in MAPPED), 'a mapped condition, and only it, is taught with --layer-map')
        require(not args.separator or (args.condition == 'pair_revision' and args.upstream and not args.erased), '--separator goes with pair_revision and --upstream, without --erased')
        pack_rows = json.loads((args.pack/'pack.json').read_text())['rows']
        if args.fold is not None:
            rest = [r for r in pack_rows if r['fold'] != args.fold]
            train_rows, val_rows = [r for r in rest if not r['val']], [r for r in rest if r['val']][:args.val_limit]
        else:
            train_rows, val_rows = [r for r in pack_rows if r['split'] == 'train'], [r for r in pack_rows if r['split'] == 'val'][:args.val_limit]
    else:
        require(args.pages, '--corpus needs --pages')
        train_rows, val_rows = read_rows(args.corpus, args.kinds, 'train'), read_rows(args.corpus, args.kinds, 'val')[:args.val_limit]
    require(train_rows, 'no training recipe of these kinds')
    require(not (args.blank and 'stack' in args.kinds), '--blank and the stack kind do not go together: the blanked column holds valid writing')
    out = make_output_dir(args.output_dir)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    processor = AutoProcessor.from_pretrained(str(args.model))
    processor.tokenizer.padding_side = 'right'
    if args.pack:
        collate = Collate(processor, prompt(args.condition, {'boxes': []}), scans=bool(args.separator))
        anchors = (args.anchor_pack, {r['id']: r for r in json.loads((args.anchor_pack/'pack.json').read_text())['rows']}) if args.anchor_pack else None
        shown = dict(keys=image_keys(args.condition), mapped=args.condition in MAPPED, erased=args.erased, places={r['id']: i for i, r in enumerate(pack_rows)}, anchors=anchors)
        train_set, val_set = PackSamples(args.pack, train_rows, **shown), PackSamples(args.pack, val_rows, **shown)
    else:
        crops = Crops(args.pages, read_rows(args.corpus))
        collate = Collate(processor, prompt('raw_final' if args.words == 'final' else 'raw_neutral', {'boxes': []}))
        train_set, val_set = Samples(train_rows, crops, args.blank), Samples(val_rows, crops, args.blank)
    generator = torch.Generator().manual_seed(SEED)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.workers,
                                         generator=generator, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate, num_workers=0) if val_rows else None
    model = lora_model(args.model, args.rank, args.alpha, args.dropout, args.layer_map).to('cuda')
    if args.checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    model.train()
    params = [q for q in model.parameters() if q.requires_grad]
    layer = [q for name, q in model.named_parameters() if q.requires_grad and 'layer_map.' in name]
    groups = [{'params': [q for q in params if all(q is not m for m in layer)], 'base': args.lr}, *([{'params': layer, 'base': args.map_lr}] if layer else [])]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0., betas=(.9, .999))
    separator = separator_optimizer = None
    if args.separator:
        from .models.hint_separator import hint, lay_hints, load
        separator, separator_weights = load(args.separator, args.upstream); separator.eval()       # NAFNet has no dropout: one forward for teaching, validating and the hints
        if args.separator_lr > 0:
            separator_optimizer = torch.optim.AdamW(separator.parameters(), lr=args.separator_lr, weight_decay=0.)
        lay = lambda batch: lay_hints(batch, separator, processor.image_processor)
    schedule = lambda step: min(1., (step+1)/args.warmup)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
    source = args.pack/'pack.json' if args.pack else args.corpus
    write_manifest(out, config=to_jsonable(args), inputs={str(source): digest(source), **({str(separator_weights): digest(separator_weights)} if args.separator else {})},
                   train_recipes=len(train_rows), val_recipes=len(val_rows),
                   trainable_parameters=sum(q.numel() for q in params), kinds=args.kinds)
    stream = (out/'train_log.jsonl').open('x', encoding='utf-8')
    started, iterator, kept = time.monotonic(), cycle(loader), (math.inf, None)

    def save():
        model.save_pretrained(out/'adapter')
        if args.layer_map:
            torch.save(layer_map_state(model), out/'layer_map.pt')
        if separator is not None:
            torch.save({'model': separator.state_dict()}, out/'separator.pt')
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group['lr'] = group['base']*schedule(step)
        learning = separator_optimizer is not None and step >= args.separator_from       # the separator learns from this step on
        total, tokens, away = 0., 0, 0.
        for _ in range(args.accumulate):
            batch = on_device(next(iterator), 'cuda')
            if separator is not None:
                with torch.set_grad_enabled(learning):
                    distance = lay(batch)
            batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
            loss = model(**batch).loss
            ((loss+args.anchor*distance if learning else loss)/args.accumulate).backward()
            total += float(loss)
            away += float(distance) if separator is not None else 0.
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if learning:
            torch.nn.utils.clip_grad_norm_(separator.parameters(), .01)                   # the clip of its own training (baselines.restoration_torch)
            separator_optimizer.step()
            separator_optimizer.zero_grad(set_to_none=True)
        row = {'step': step+1, 'loss': total/args.accumulate, 'lr': optimizer.param_groups[0]['lr'], 'seconds': round(time.monotonic()-started, 1)}
        if separator is not None:
            row.update(separator_learning=learning, anchor_distance=away/args.accumulate)
        if val_loader is not None and ((step+1) % args.eval_every == 0 or step+1 == args.steps):
            row['val_loss'] = evaluate(model, val_loader, 'cuda', lay if separator is not None else None)
            if args.keep_best and row['val_loss'] < kept[0]:
                kept = (row['val_loss'], step+1)
                save()
        stream.write(json.dumps(row)+'\n')
        stream.flush()
        print(json.dumps(row), flush=True)
    if kept[1] is None:
        save()
    stream.close()
    if separator is not None:                                                             # the kept separator's hint of every row of the pack
        separator.load_state_dict(torch.load(out/'separator.pt', map_location='cuda')['model']); (out/'hints').mkdir()
        with torch.no_grad():
            for place, r in enumerate(pack_rows):
                with Image.open(args.pack/r['images']['raw']) as raw, Image.open(args.pack/r['images']['erased']) as erased:
                    made = hint(separator, torch.from_numpy(np.asarray(raw.convert('L'))).to('cuda', torch.float32)/255)
                    (erased.convert('L') if made is None else Image.fromarray((made*255).round().byte().cpu().numpy())).save(out/'hints'/f'{place:05d}.png')
    write_json(out/'complete.json', {'steps': args.steps, 'seconds': round(time.monotonic()-started, 1), 'final_loss': row['loss'], 'final_val_loss': row.get('val_loss'),
                                     'kept_step': kept[1] or args.steps, 'kept_val_loss': kept[0] if kept[1] else row.get('val_loss')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
