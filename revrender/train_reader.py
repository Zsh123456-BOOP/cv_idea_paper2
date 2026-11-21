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
``target``; with ``--all-rest`` every row of the other folds is taught and none validates.  The ``*_revision`` conditions show the images they name (``pair_revision`` two); the mapped ones need
``--layer-map``: the row's layer map goes in beside the scan through a projection of its own (``models.vlm_tokens``), taught at
``--map-lr`` and saved as OUT/layer_map.pt.  ``--keep-best`` keeps the adapter of the lowest validation loss instead of the last.

``--answer latex`` teaches a readable row of a folds pack its text in LaTeX (``data.struck_pack.latex_answer``, the spelling the words
ask for) in place of its ``target`` (the scores' flattened spelling; the two agree once ``eval.coverage.prepare`` has been applied).

``--init-adapter`` starts from an earlier run's adapter and teaches it on (a schedule of its own: warm-up and cosine over ``--steps``).

``--view-students PATTERN`` (with ``--fold K``; plan of 2026-11-20, 第五轮_X与V结合, section 4.1): the reader learns on what the
vision-tower adapters of ``train_clean_view`` make of its images.  PATTERN names a fold's adapter (``{fold}``); the images at the places
``--student-images`` names (1-based) of a training row of fold j are seen through fold j's adapter -- the one that never trained on fold
j -- and fold K's is never loaded (the reader of fold K reads fold K through it, ``recognize_lines --view-student``).  The adapters
are frozen and in eval, as when reading (``models.vlm_tokens.install_students``); only the language model's adapter learns.
``--student-off``: those images go the adapters' way with the adapters disabled (the trajectory check: it must learn as the plain
reader).  ``--stop-after N``: stop after N steps of the schedule of ``--steps`` (the trajectory check).

``--counterfactual S W`` (a pack of ``data.struck_pack counterfactual``) teaches what a change of the evidence should change: a
training sample of a readable region is shown, with probability S, with its strike taken off the scan (the reading stays the same:
the strike is not the writing) and, with probability W, with the struck writing taken off instead (the reading becomes
``recognize_lines.UNREADABLE``: with the evidence gone the reader must not know).  The validation rows are shown as they are.
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
from .models.internvl import processing
from .recognize_lines import INTERVENED, MAPPED, SYSTEM, UNREADABLE, image_keys, prompt
from .util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

EMPTY = '[空]'
# every linear layer of the language model's blocks: attention and MLP, and the linear-attention projections of Qwen3.5's hybrid layers
LANGUAGE_TARGETS = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)'
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
    """The rows of a line pack: (the images a condition shows, the text to give, the layer map of a mapped condition or None).
    ``counterfactual`` (strike off, writing off): the probabilities a readable row is shown with one of those interventions
    (``INTERVENED``: which image stands in for which) and what it is then taught to give."""

    def __init__(self, root, rows, keys=('raw',), mapped=False, counterfactual=(0., 0.), condition=None, latex=False):
        self.root, self.rows, self.keys, self.mapped, self.counterfactual = root, rows, keys, mapped, counterfactual
        self.condition, self.latex = condition, latex

    def __len__(self):
        return len(self.rows)

    def answer(self, row):
        if self.latex and row['readable']:
            from .data.struck_pack import latex_answer
            return latex_answer(row['text'])
        return row.get('target', row['text'])

    def __getitem__(self, i):
        row = self.rows[i]; text = self.answer(row); swap = {}
        draw = random.random() if row.get('readable', True) and sum(self.counterfactual) else 1.
        if draw < self.counterfactual[0]:
            swap = INTERVENED['strike_off']
        elif draw < sum(self.counterfactual):
            swap, text = INTERVENED['writing_off'], UNREADABLE
        opened = lambda key: Image.open(self.root/row['images'][swap.get(key, key)]).convert('RGB')
        return ([opened(key) for key in self.keys], text, opened('map') if self.mapped else None, prompt(self.condition, row) if self.condition else None,
                row.get('fold'))


class Collate:
    """A batch for the model: the chat with the answer, the loss only on the answer; the words are a sample's own when it carries them.
    ``students``: the places (1-based) of a sample's images its own fold's vision adapter sees; the batch then carries ``students``,
    per image in order, that adapter's name (``fold<j>``) or None."""

    def __init__(self, processor, words, students=()):
        self.processor, self.words, self.students = processor, words, tuple(students)
        from .models.vlm_tokens import THINK_OPEN, answer_prefill
        self.start = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
        prefill = answer_prefill(processor)                            # the template writes an empty thinking block before a taught answer
        self.header = len(processor.tokenizer('<|im_start|>assistant\n'+(THINK_OPEN+prefill if prefill else ''), add_special_tokens=False)['input_ids'])

    def __call__(self, batch):
        batch = [item if len(item) == 5 else ([item[0]], item[1], None, None, None) for item in batch]      # (images, text, layer map, words, fold)
        conversations = [[{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
                          {'role': 'user', 'content': [*({'type': 'image', 'image': image} for image in images), {'type': 'text', 'text': words or self.words}]},
                          {'role': 'assistant', 'content': [{'type': 'text', 'text': target_of(text)}]}] for images, text, _, words, _ in batch]
        out = self.processor.apply_chat_template(conversations, tokenize=True, return_dict=True, return_tensors='pt', padding=True)
        out = {k: out[k] for k in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'mm_token_type_ids') if k in out}
        if batch[0][2] is not None:
            from .models.vlm_tokens import layer_maps, with_layer_maps
            out['pixel_values'] = with_layer_maps(out['pixel_values'], [m for images, _, layer, _, _ in batch for m in layer_maps(images, layer)], self.processor.image_processor)
        labels = out['input_ids'].clone()
        for b in range(labels.shape[0]):
            marks = (out['input_ids'][b] == self.start).nonzero().flatten()
            labels[b, :int(marks[-1])+self.header] = -100              # the system and user turns, "<|im_start|>assistant\n" and an empty thinking block
        labels[out['attention_mask'] == 0] = -100
        out['labels'] = labels
        if self.students:
            out['students'] = [f'fold{fold}' if n+1 in self.students else None for images, _, _, _, fold in batch for n in range(len(images))]
        return out


def lora_model(model_dir, rank, alpha, dropout, layer_map=False, init=None):
    """The model with a low-rank adapter on the language model's linear layers; with ``layer_map`` also the (trainable, zero) projection
    of a layer map into the patch embedding, added after the adapter froze everything else; with ``init`` the adapter starts as that one
    (a run of this module) and goes on learning."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    from .models.internvl import is_internvl, training_model
    from .models.vlm_tokens import patch_embed_as_linear, patch_embed_with_layer_map
    if is_internvl(model_dir):                                           # InternVL3.5: its bundled model (models.internvl)
        require(not layer_map, 'layer maps are not supported on InternVL')
        model = training_model(model_dir)
    else:
        model = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.bfloat16, attn_implementation='sdpa')
        patch_embed_as_linear(model.model.visual.patch_embed)
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias='none',
                        target_modules=LANGUAGE_TARGETS)
    if init:
        from peft import PeftModel
        adapted = PeftModel.from_pretrained(model, str(init), is_trainable=True)
    else:
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


def answer_loss(model, batch):
    """The mean cross-entropy over the answer tokens, as transformers computes it (the logits at each position against the next token,
    -100 left out), from the logits of the positions from the one before the first answer token on only: the whole sequence's logits over
    the whole vocabulary do not fit on a card beside three images and a question."""
    import torch
    labels = batch['labels']
    first = int((labels != -100).int().argmax(1).min())                 # the earliest answer token of the batch
    keep = labels.shape[1]-first+1
    logits = model(**{k: v for k, v in batch.items() if k != 'labels'}, logits_to_keep=keep).logits
    return torch.nn.functional.cross_entropy(logits[:, :-1].float().flatten(0, 1), labels[:, labels.shape[1]-keep+1:].flatten(), ignore_index=-100)


def as_list(items):
    """The samples of a group as they are (``--one-at-a-time`` collates each alone)."""
    return items


def one_at_a_time(model, singles, accumulate, students=None):
    """A group's loss as ``answer_loss`` of the group in one batch -- the cross-entropy summed over every answer token of the group over
    the group's answer-token count -- its gradients taken one sample at a time (``--one-at-a-time``): each sample's summed
    cross-entropy over the whole group's token count, back-propagated (divided by ``accumulate``, as a batch is).  ``singles``: the group's
    samples each collated alone.  Returns the group's loss."""
    import torch
    counts = [int((b['labels'] != -100).sum()) for b in singles]; total = 0.
    for b, n in zip(singles, counts):
        if students is not None:
            students.flags = b.pop('students')
        b = {k: v.to('cuda') for k, v in b.items() if k != 'students'}
        b['pixel_values'] = b['pixel_values'].to(torch.bfloat16)
        share = answer_loss(model, b)*n/sum(counts)
        (share/accumulate).backward(); total += float(share)
    return total


def evaluate(model, loader, device):
    import torch
    model.eval()
    total, count = 0., 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'students'}
            batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
            tokens = int((batch['labels'] != -100).sum())
            total += float(answer_loss(model, batch))*tokens
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
    p.add_argument('--answer', choices=('target', 'latex'), default='target', help="with --pack: teach the row's target, or its text in LaTeX (data.struck_pack.latex_answer)")
    p.add_argument('--keep-best', action='store_true', help='keep the adapter of the lowest validation loss, not the last')
    p.add_argument('--init-adapter', type=Path, help='start from this adapter (OUT/adapter of an earlier run) and go on teaching it, with a schedule of its own')
    p.add_argument('--all-rest', action='store_true', help='with --fold: teach on every row of the other folds, the val marks ignored (no validation)')
    p.add_argument('--view-students', help="with --pack and --fold: a fold's vision adapter of train_clean_view, {fold} for the fold (see above)")
    p.add_argument('--student-images', type=int, nargs='+', default=[], help='with --view-students: the places (1-based) of the images the adapters see')
    p.add_argument('--student-off', action='store_true', help='with --view-students: those images go the adapters\' way with the adapters disabled')
    p.add_argument('--stop-after', type=int, help='stop after this many steps of the --steps schedule (the trajectory check)')
    p.add_argument('--one-at-a-time', action='store_true', help='each group of --batch-size samples one sample at a time, the same loss and gradients (for memory)')
    p.add_argument('--counterfactual', type=float, nargs=2, default=(0., 0.), metavar=('STRIKE_OFF', 'WRITING_OFF'),
                   help='with --pack: the probabilities a readable training row is shown with its strike, or its struck writing, taken off')
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
    from .models.vlm_tokens import processor_for
    if args.pack:
        require(args.condition is not None, '--pack needs --condition')
        require(args.layer_map == (args.condition in MAPPED), 'a mapped condition, and only it, is taught with --layer-map')
        pack_rows = json.loads((args.pack/'pack.json').read_text())['rows']
        if args.fold is not None:
            rest = [r for r in pack_rows if r['fold'] != args.fold]
            train_rows, val_rows = ([r for r in rest if not r['val']], [r for r in rest if r['val']][:args.val_limit]) if not args.all_rest else (rest, [])
        else:
            train_rows, val_rows = [r for r in pack_rows if r['split'] == 'train'], [r for r in pack_rows if r['split'] == 'val'][:args.val_limit]
    else:
        require(args.pages, '--corpus needs --pages')
        train_rows, val_rows = read_rows(args.corpus, args.kinds, 'train'), read_rows(args.corpus, args.kinds, 'val')[:args.val_limit]
    require(train_rows, 'no training recipe of these kinds')
    require(bool(args.view_students) == bool(args.student_images) and (not args.view_students or (args.pack and args.fold is not None and not args.layer_map)),
            '--view-students and --student-images go together, with --pack and --fold')
    folds_used = sorted({r['fold'] for r in train_rows + val_rows}) if args.view_students else []
    require(args.fold not in folds_used, 'a fold read by this reader would be seen through an adapter of its own')
    require(not (args.blank and 'stack' in args.kinds), '--blank and the stack kind do not go together: the blanked column holds valid writing')
    out = make_output_dir(args.output_dir)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    processor = processor_for(args.model)
    processor.tokenizer.padding_side = 'right'
    if args.pack:
        collate = Collate(processor, prompt(args.condition, {'boxes': [], 'problem': ''}), args.student_images)      # each pack sample carries its own words
        shown = image_keys(args.condition), args.condition in MAPPED
        latex = args.answer == 'latex'
        train_set = PackSamples(args.pack, train_rows, *shown, tuple(args.counterfactual), args.condition, latex)
        val_set = PackSamples(args.pack, val_rows, *shown, condition=args.condition, latex=latex)
    else:
        crops = Crops(args.pages, read_rows(args.corpus))
        collate = Collate(processor, prompt('raw_final' if args.words == 'final' else 'raw_neutral', {'boxes': []}))
        train_set, val_set = Samples(train_rows, crops, args.blank), Samples(val_rows, crops, args.blank)
    generator = torch.Generator().manual_seed(SEED)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=as_list if args.one_at_a_time else collate,
                                         num_workers=args.workers, generator=generator, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1 if args.one_at_a_time else args.batch_size, collate_fn=collate, num_workers=0) if val_rows else None
    require(not (args.init_adapter and args.layer_map), '--init-adapter with --layer-map is not supported')
    from .models import internvl
    model = lora_model(args.model, args.rank, args.alpha, args.dropout, args.layer_map, args.init_adapter)
    model = model if internvl.SPLIT and internvl.is_internvl(args.model) else model.to('cuda')     # a split InternVL is placed already
    students, adapters = None, {f'fold{j}': Path(args.view_students.format(fold=j)) for j in folds_used}
    if args.view_students:
        from .models.vlm_tokens import install_students
        state = torch.get_rng_state(), torch.cuda.get_rng_state_all()      # loading the adapters draws their (overwritten) initial weights
        students = install_students(model.get_base_model().model, adapters, args.student_off)
        torch.set_rng_state(state[0]); torch.cuda.set_rng_state_all(state[1])
    if args.checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    model.train()
    params = [q for q in model.parameters() if q.requires_grad]
    layer = [q for name, q in model.named_parameters() if q.requires_grad and 'layer_map.' in name]
    groups = [{'params': [q for q in params if all(q is not m for m in layer)], 'base': args.lr}, *([{'params': layer, 'base': args.map_lr}] if layer else [])]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0., betas=(.9, .999))
    schedule = lambda step: min(1., (step+1)/args.warmup)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
    source = args.pack/'pack.json' if args.pack else args.corpus
    write_manifest(out, config=to_jsonable(args), inputs={str(source): digest(source), **({str(w): digest(w) for w in sorted(args.init_adapter.glob('*.safetensors'))} if args.init_adapter else {})},
                   train_recipes=len(train_rows), val_recipes=len(val_rows),
                   trainable_parameters=sum(q.numel() for q in params), kinds=args.kinds, lora_targets=LANGUAGE_TARGETS,
                   lora_modules=sorted({n.split('.lora_')[0].rsplit('.', 1)[-1] for n, q in model.named_parameters() if q.requires_grad and '.lora_' in n}),
                   model_processing=processing(args.model),
                   **({'one_at_a_time': f'the reader taught one sample at a time, mathematically {args.batch_size} a batch x {args.accumulate} accumulated (the group loss over the group token count)'} if args.one_at_a_time else {}),
                   **({'students': {k: str(v) for k, v in adapters.items()}, 'student_off': args.student_off,
                       'students_used': {f'fold{j}': sum(r['fold'] == j for r in train_rows) for j in folds_used},
                       'student_adapter_files': {str(w): digest(w) for v in adapters.values() for w in sorted(v.glob('*.safetensors'))}} if students else {}))
    stream = (out/'train_log.jsonl').open('x', encoding='utf-8')
    started, iterator, kept = time.monotonic(), cycle(loader), (math.inf, None)

    def save():
        # with --view-students the vision adapters sit inside the model too; only the language model's adapter is the reader's
        model.save_pretrained(out/'adapter', **({'state_dict': {k: v for k, v in model.state_dict().items() if '.visual.' not in k}} if students else {}))
        if args.layer_map:
            torch.save(layer_map_state(model), out/'layer_map.pt')
    for step in range(args.stop_after or args.steps):
        for group in optimizer.param_groups:
            group['lr'] = group['base']*schedule(step)
        total, tokens = 0., 0
        for _ in range(args.accumulate):
            if args.one_at_a_time:                       # the same group as a batch would be, its samples one at a time
                total += one_at_a_time(model, [collate([item]) for item in next(iterator)], args.accumulate, students)
                continue
            batch = next(iterator)
            if students is not None:
                students.flags = batch.pop('students')
            batch = {k: v.to('cuda') for k, v in batch.items()}
            batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
            loss = answer_loss(model, batch)
            (loss/args.accumulate).backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row = {'step': step+1, 'loss': total/args.accumulate, 'lr': optimizer.param_groups[0]['lr'], 'seconds': round(time.monotonic()-started, 1)}
        if val_loader is not None and ((step+1) % args.eval_every == 0 or step+1 == args.steps):
            row['val_loss'] = evaluate(model, val_loader, 'cuda')
            if args.keep_best and row['val_loss'] < kept[0]:
                kept = (row['val_loss'], step+1)
                save()
        stream.write(json.dumps(row)+'\n')
        stream.flush()
        print(json.dumps(row), flush=True)
    if kept[1] is None:
        save()
    stream.close()
    write_json(out/'complete.json', {'steps': args.steps, 'seconds': round(time.monotonic()-started, 1), 'final_loss': row['loss'], 'final_val_loss': row.get('val_loss'),
                                     'kept_step': kept[1] or args.steps, 'kept_val_loss': kept[0] if kept[1] else row.get('val_loss'),
                                     **({'stopped_after': args.stop_after} if args.stop_after else {}),
                                     **({'students_encoded': students.encoded} if students else {})})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
