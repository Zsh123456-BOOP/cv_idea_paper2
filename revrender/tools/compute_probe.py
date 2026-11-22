"""How a part of the plug-in study is computed on a card, measured before anything is read (plan 换模型验证插件, revisions two, four and
five): the worst batch of one part -- the regions whose three images make the most image tokens -- then the time of a batch of the
median size.

    python -m revrender.tools.compute_probe --model DIR --clean-pack VIEWS --scan-pack RAW_VIEWS --part reader|plugin|read
        [--one-at-a-time] [--read-batch 8] [--plugin-pieces 1] [--checkpointing] --output-dir OUT

``reader``: one group of two regions of H's training (context_revision, checkpointing, as ``train_reader``), as one batch or
``--one-at-a-time`` (``train_reader.one_at_a_time``, the same loss); an update is eight groups.  ``plugin``: one step of the tower
adapter (eight regions' windows and lines, in ``--plugin-pieces`` pieces with ``train_clean_view.accumulated``, ``--checkpointing``),
and a validation pass's time per region.  ``read``: one batch of ``--read-batch`` regions read as V (an untaught adapter on the second
and third images).  InternVL takes ``INTERNVL_MAX_TILES`` and ``INTERNVL_SPLIT`` (the whole model over two cards).  OUT/result.json:
the peak memory on each card, whether it fitted, the times.
"""
import argparse
import json
import time
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ('model', 'clean-pack', 'scan-pack', 'output-dir'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--part', choices=('reader', 'plugin', 'read'), required=True); p.add_argument('--one-at-a-time', action='store_true')
    p.add_argument('--read-batch', type=int, default=8); p.add_argument('--plugin-pieces', type=int, default=1); p.add_argument('--checkpointing', action='store_true')
    args = p.parse_args(argv)
    import torch
    from PIL import Image
    from ..models import internvl
    from ..models.vlm_tokens import processor_for, reader_for
    from ..util import make_output_dir, write_json
    out = make_output_dir(args.output_dir); started = time.time()
    clean, scan = args.clean_pack, args.scan_pack
    rows = {r['id']: r for r in json.loads((clean/'pack.json').read_text())['rows']}; scans = {r['id']: r for r in json.loads((scan/'pack.json').read_text())['rows']}
    processor = processor_for(args.model)
    opened = lambda pack, row, key: Image.open(pack/row['images'][key]).convert('RGB')
    def tokens(images):                                                  # patches, tiles, Gemma's patches that are not padding, LLaVA's tiles
        batch = processor.image_processor(images=images, return_tensors='pt')
        if 'image_grid_thw' in batch:
            return int(batch['image_grid_thw'].prod(-1).sum())
        if 'image_sizes' in batch:
            from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches
            return sum(image_size_to_num_patches(s, processor.image_processor.image_grid_pinpoints, 336) for s in batch['image_sizes'].tolist())
        return int((batch['image_position_ids'] != -1).all(-1).sum())
    size = {k: tokens([opened(clean, r, key) for key in ('raw', 'erased', 'context')]) for k, r in rows.items()}
    worst = sorted(rows, key=lambda k: -size[k]); middle = worst[len(worst)//2-8:len(worst)//2+8]
    fitted, error, timing = True, None, {}

    def timed(run, repeats):
        torch.cuda.synchronize(); t = time.time()
        for _ in range(repeats): run()
        torch.cuda.synchronize(); return (time.time()-t)/repeats
    try:
        if args.part == 'reader':
            from ..recognize_lines import prompt
            from ..models import gemma
            from ..train_reader import Collate, answer_loss, lora_model, one_at_a_time
            processor.tokenizer.padding_side = 'right'; collate = Collate(processor, '')
            model = lora_model(args.model, 16, 32, .05)
            if gemma.is_gemma(args.model):
                gemma.on_card(model)                                     # the audio tower stays off the card
            else:
                model = model if internvl.SPLIT and internvl.is_internvl(args.model) else model.to('cuda')
            timing['load_seconds'] = round(time.time()-started, 1)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.enable_input_require_grads(); model.train()
            item = lambda r: ([opened(clean, r, key) for key in ('raw', 'erased', 'context')], r['text'] or '', None, prompt('context_revision', r), None)

            def group(ks):
                items = [item(rows[k]) for k in ks]
                if args.one_at_a_time:
                    one_at_a_time(model, [collate([i]) for i in items], 1)
                else:
                    batch = {k: v.to('cuda') for k, v in collate(items).items()}; batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
                    answer_loss(model, batch).backward()
                model.zero_grad(set_to_none=True)
            group(worst[:2])
            g = timed(lambda: group(middle[:2]), 3)
            timing.update({'group_seconds': round(g, 2), 'update_seconds': round(8*g, 1)})
        elif args.part == 'plugin':
            from ..train_clean_view import accumulated, checkpoint_layers, encode, levels, regions_of, validation_writers, vision_tower, with_student
            student = with_student(vision_tower(args.model).to('cuda')); student.train(); timing['load_seconds'] = round(time.time()-started, 1)
            if args.checkpointing:
                checkpoint_layers(student.base_model.model)

            def views(piece):
                pairs = [(opened(scan, scans[k], key), opened(clean, rows[k], key)) for k in piece for key in ('erased', 'context')]
                xs, gs = encode(processor, [a for a, _ in pairs], 'cuda'); xc, gc = encode(processor, [b for _, b in pairs], 'cuda'); return xs, gs, xc, gc

            def teacher_of(piece):
                _, _, xc, gc = views(piece)
                with student.disable_adapter():
                    return levels(student, xc, gc)

            def step(chosen):
                n = -(-len(chosen)//args.plugin_pieces)
                accumulated(lambda piece: levels(student, *views(piece)[:2]), teacher_of, [chosen[i:i+n] for i in range(0, len(chosen), n)])
                student.zero_grad(set_to_none=True)
            pair_size = {k: tokens([opened(clean, rows[k], 'erased'), opened(clean, rows[k], 'context')]) for k in rows}
            step(sorted(rows, key=lambda k: -pair_size[k])[:8])
            timing['step_seconds'] = round(timed(lambda: step(middle[:8]), 2), 2)

            def validate_one():
                with torch.no_grad():
                    for k in middle[:8]: teacher_of([k]); levels(student, *views([k])[:2])
            timing['validation_seconds_per_region'] = round(timed(validate_one, 1)/8, 3)
            real = regions_of(clean, scan, lambda r: r['fold'] != 0); held = validation_writers(real)
            timing['validation_regions'] = sum(r[1] in held for r in real)
        else:
            from ..recognize_lines import SYSTEM, prompt
            from ..train_clean_view import vision_tower, with_student
            fresh = out/'untaught_adapter'; with_student(vision_tower(args.model)).save_pretrained(str(fresh))
            reader = reader_for(args.model, student=fresh); timing['load_seconds'] = round(time.time()-started, 1)

            def read(ks):
                chosen = [scans[k] for k in ks]
                reader.read([([opened(scan, r, key) for key in ('raw', 'erased', 'context')], prompt('context_revision', r), None, 0.) for r in chosen], SYSTEM, 256,
                            loop_limit=4, students=[[False, True, True] for _ in chosen])
            read(worst[:args.read_batch])
            timing['seconds_per_region'] = round(timed(lambda: read(middle[:args.read_batch]), 1)/args.read_batch, 2)
    except torch.cuda.OutOfMemoryError as e:
        fitted, error = False, str(e).splitlines()[0]
    gib = lambda b: round(b/2**30, 2)
    result = {'part': args.part, 'model': args.model.name, 'processing': internvl.processing(args.model), 'cards': 2 if internvl.SPLIT and internvl.is_internvl(args.model) else 1,
              'one_at_a_time': args.one_at_a_time, 'read_batch': args.read_batch, 'plugin_pieces': args.plugin_pieces, 'checkpointing': args.checkpointing,
              'fitted': fitted, 'error': error, 'peak_allocated_gib': [gib(torch.cuda.max_memory_allocated(d)) for d in range(torch.cuda.device_count())],
              'card_gib': gib(torch.cuda.get_device_properties(0).total_memory), 'seconds': round(time.time()-started, 1), **timing,
              'worst_image_units_per_region': [size[k] for k in worst[:8]], 'median_image_units_per_region': size[middle[0]]}
    write_json(out/'result.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
