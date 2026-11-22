"""B of 插件结构优化 (plan of 2026-11-22, section two, as revised in section eight): the fold's reader as the judge of a tower adapter.

    python -m revrender.reading_signal gap --model MODEL --reader-adapter READER --pack VIEWS --scan-pack RAW_VIEWS --fold K --output OUT.json
    python -m revrender.reading_signal grads --model MODEL --reader-adapter READER --pack VIEWS --scan-pack RAW_VIEWS --fold K --output OUT.json

``ReadingSignal`` (``train_clean_view train --reader-adapter``): the fold's reader (``train_reader``'s adapter merged, frozen, every
block of the language model recomputed in the backward pass) whose own vision tower takes the adapter being taught.  A region is
read as V reads it -- the scan pack's raw crop, its erased window (the scan) and its context line, the second and third through the
adapter, the first through the frozen tower (``vlm_tokens.picked`` keeping the adapter's graph), words of ``context_revision`` --
and its answer is what the reader was taught to give (``train_reader --answer latex``: ``PackSamples.answer``; an unreadable region
the refusal).  A region's loss is the mean cross-entropy over its answer tokens (``train_reader.answer_loss``); a step's reading loss
the mean of its real regions' (none: no reading loss); the gradient reaches the adapter alone.

``gap`` (the plan's check 5): on the real training regions of fold K's adapter (the other folds without the validation writers, as
``train_clean_view train`` takes them), the reader's mean answer loss when it sees the scan (the scan pack's three images, no
adapter: the adapter at step 0) and when it sees the annotator's erased images (the pack's three: what it was taught on); per region
and their means.  The rule: a difference under 0.05 over the five folds' mean means the reader gives no signal, and B is not run.
"""
import argparse
import json
from pathlib import Path

from .util import require, write_json

KEYS = ('raw', 'erased', 'context')


def frozen_reader(model_dir, reader_adapter):
    """(the processor, the reader merged and frozen on the card, its language model checkpointed and in training mode -- it has no
    dropout -- so gradients can pass through it to the tower)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from .models.vlm_tokens import patch_embed_as_linear
    processor = AutoProcessor.from_pretrained(str(model_dir)); processor.tokenizer.padding_side = 'right'
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.bfloat16, attn_implementation='sdpa')
    model = PeftModel.from_pretrained(model, str(reader_adapter)).merge_and_unload()
    for q in model.parameters():
        q.requires_grad_(False)
    model = model.to('cuda')
    patch_embed_as_linear(model.model.visual.patch_embed)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.train()
    return processor, model


def region_samples(pack, scan_pack, source):
    """{id: (the three image paths, the answer, the words)}: the images of ``source`` ('scan': the scan pack's; 'clean': the pack's),
    the answer the reader was taught (from the pack's row), the words of ``context_revision`` (of the scan pack's row)."""
    from .recognize_lines import prompt
    from .train_reader import PackSamples
    clean = {r['id']: r for r in json.loads((pack/'pack.json').read_text())['rows']}
    scans = {r['id']: r for r in json.loads((scan_pack/'pack.json').read_text())['rows']}
    answers = PackSamples(pack, [], latex=True); root, rows = (scan_pack, scans) if source == 'scan' else (pack, clean)
    return {k: ([root/rows[k]['images'][key] for key in KEYS], answers.answer(clean[k]), prompt('context_revision', scans[k])) for k in clean if k in scans}


class ReadingSignal:
    def __init__(self, model_dir, reader_adapter, pack, scan_pack):
        from .train_reader import Collate
        self.processor, self.model = frozen_reader(model_dir, reader_adapter)
        self.tower = self.model.model.visual
        self.samples = region_samples(pack, scan_pack, 'scan'); self.collate = Collate(self.processor, '')
        self.students = None

    def route(self, view, name='default'):
        """The reader's image features through ``view`` (the adapter being taught, put into this reader's tower) for the flagged images."""
        from .models.vlm_tokens import Students, student_features
        self.students = Students(view, [name]); self.name = name
        inner = self.model.model
        inner.get_image_features = student_features(inner.get_image_features, self.students, False, grad=True)

    def loss(self, k):
        """Region ``k``'s mean answer-token cross-entropy, read as V reads it."""
        import torch
        from PIL import Image
        from .train_reader import answer_loss
        paths, answer, words = self.samples[k]
        batch = self.collate([([Image.open(p).convert('RGB') for p in paths], answer, None, words, None)])
        self.students.flags = [None, self.name, self.name]
        batch = {key: v.to('cuda') for key, v in batch.items()}; batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
        return answer_loss(self.model, batch)

    def backward(self, view, ids, weight):
        """``weight`` x the mean of the regions' losses back-propagated, a region at a time; returns that mean (None for no region)."""
        if not ids:
            return None
        total = 0.
        for k in ids:
            one = self.loss(k); (weight*one/len(ids)).backward(); total += float(one)
        return total/len(ids)

    def mean_loss(self, view, ids):
        """The regions' mean loss, nothing learned (validation; recorded only)."""
        import torch
        with torch.no_grad():
            return sum(float(self.loss(k)) for k in ids)/max(1, len(ids))

    def record(self):
        return {'regions_with_answers': len(self.samples), 'images': "the scan pack's raw, erased, context; the adapter on the second and third",
                'words': 'context_revision', 'answers': 'train_reader --answer latex (PackSamples.answer); unreadable: the refusal',
                'region_loss': 'the mean cross-entropy over its answer tokens', 'step_loss': 'the mean over the real regions of the step (none: 0)'}


def gap(args):
    import numpy as np
    import torch
    from PIL import Image
    from .train_clean_view import regions_of, validation_writers
    from .train_reader import Collate, answer_loss
    real = regions_of(args.pack, args.scan_pack, lambda r: r['fold'] != args.fold)
    held = validation_writers(real); ids = sorted(r[0] for r in real if r[1] not in held)
    processor, model = frozen_reader(args.model, args.reader_adapter); model.eval(); collate = Collate(processor, '')
    seen = {source: region_samples(args.pack, args.scan_pack, source) for source in ('scan', 'clean')}
    per = {}
    with torch.no_grad():
        for k in ids:
            per[k] = {}
            for source, samples in seen.items():
                paths, answer, words = samples[k]
                batch = collate([([Image.open(p).convert('RGB') for p in paths], answer, None, words, None)])
                batch = {key: v.to('cuda') for key, v in batch.items()}; batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
                per[k][source] = float(answer_loss(model, batch))
    means = {source: float(np.mean([v[source] for v in per.values()])) for source in seen}
    require(not args.output.exists(), f'{args.output} exists')
    write_json(args.output, {'fold': args.fold, 'regions': len(ids), 'validation_writers_left_out': sorted(held), 'mean_scan': means['scan'],
                             'mean_clean': means['clean'], 'difference': means['scan']-means['clean'], 'per_region': per})
    print(json.dumps({'fold': args.fold, 'regions': len(ids), **means}), flush=True)


def grads(args):
    """Check 3 for B: a new low-rank adapter in fold K's reader, one region's alignment loss and then its reading loss back-propagated:
    which parameters hold gradients (the adapter's alone), whether the reader's are all frozen, and whether the reading loss moved the
    adapter's gradients (the signal reaches it)."""
    import torch
    from PIL import Image
    from .train_clean_view import encode, levels, pair_loss, regions_of, validation_writers, with_student
    signal = ReadingSignal(args.model, args.reader_adapter, args.pack, args.scan_pack)
    torch.manual_seed(42); view = with_student(signal.tower); signal.route(view); view.train()
    real = regions_of(args.pack, args.scan_pack, lambda r: r['fold'] != args.fold); held = validation_writers(real)
    k, _, pairs = next(r for r in real if r[1] not in held)
    scan, clean = (encode(signal.processor, [Image.open(p[i]).convert('RGB') for p in pairs], 'cuda') for i in (0, 1))
    with torch.no_grad(), view.disable_adapter():
        teacher = levels(view, *clean)
    pair_loss(levels(view, *scan), teacher).backward()
    adapter = {n: q for n, q in signal.model.named_parameters() if 'lora_' in n}
    before = {n: q.grad.detach().clone() if q.grad is not None else None for n, q in adapter.items()}
    reading = signal.backward(view, [k], .1)
    moved = sum(float((q.grad-(before[n] if before[n] is not None else 0)).abs().sum()) for n, q in adapter.items() if q.grad is not None)
    graded = [n for n, q in signal.model.named_parameters() if q.grad is not None]
    result = {'region': k, 'reading_loss': reading, 'parameters_with_gradients': sum(adapter[n].numel() for n in graded if n in adapter),
              'outside_the_adapter': [n for n in graded if n not in adapter], 'adapter_parameters': sum(q.numel() for q in adapter.values()),
              'reader_parameters_trainable': sum(q.numel() for n, q in signal.model.named_parameters() if q.requires_grad and n not in adapter),
              'tower_parameters_trainable': sum(q.numel() for n, q in signal.tower.named_parameters() if q.requires_grad and 'lora_' not in n),
              'reading_moves_the_adapter_gradients': moved}
    result['passed'] = not result['outside_the_adapter'] and result['reader_parameters_trainable'] == 0 and result['tower_parameters_trainable'] == 0 and moved > 0
    require(not args.output.exists(), f'{args.output} exists'); write_json(args.output, result)
    print(json.dumps({key: v for key, v in result.items() if key != 'outside_the_adapter'}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('gap')
    for key in ('model', 'reader-adapter', 'pack', 'scan-pack', 'output'): g.add_argument('--'+key, type=Path, required=True)
    g.add_argument('--fold', type=int, required=True)
    r = sub.add_parser('grads')
    for key in ('model', 'reader-adapter', 'pack', 'scan-pack', 'output'): r.add_argument('--'+key, type=Path, required=True)
    r.add_argument('--fold', type=int, required=True)
    args = p.parse_args(argv)
    {'gap': gap, 'grads': grads}[args.command](args)


if __name__ == '__main__':
    main()
