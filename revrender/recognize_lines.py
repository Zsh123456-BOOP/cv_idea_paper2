"""Read the lines of a ``data.line_pack`` pack with one VLM under several conditions, the images of a condition in each request.

    python -m revrender.recognize_lines --pack PACK --model MODEL_DIR [--conditions NAME ...] [--backend vllm|hf] [--adapter DIR] [--tp N] --output-dir OUT

A condition is an image of the line and the words that go with it (``CONDITIONS``).  The words are one skeleton -- transcribe, LaTeX for
formulas, □ for what cannot be read, no solving or correcting -- and what a condition adds to it:

  raw_neutral      the scan, nothing more (the model reads what it sees)
  raw_final        the scan, and the instruction to give only the final valid work: what is struck out, blacked out or voided is left out
  erased_neutral   the revision erased (the masks of the pack), nothing more
  erased_final     the revision erased, and the instruction above
  blanked_neutral  the columns under each mask whited out (the revised words gone), nothing more
  blanked_final    the same, and the instruction above
  overlay_final    the revision tinted red in the one image, said to be a marker and not writing, and the instruction above
  hint_final       the scan, the places of the masks in words (thousandths of the width and height), and the instruction above
  raw_struck       the scan, and the instruction to give only what is struck out or blacked out (the annotation page's starting text)
  boxed_read       an image with a blue box round one revision, and the instruction to give what the student wrote in the box (under
                   a strike if one is there): eval.struck_diagnostics
  boxed_marked     the same, the strike drawn red and said to be the strike
  from_pack        the words (and the system prompt) each row of the pack carries: ``prompt``, ``system`` (eval.revision_judgment)

Conditions of one revision (a pack of ``data.struck_pack folds``: a row is a struck region; the mapped ones ``--backend hf``): the struck
text is asked for, or ``[读不出]`` when it cannot be read.  With vLLM a condition of several images sends them all, and ``--loop-limit``
cuts the greedy reading where the hf reader would have stopped it (``stopped``).

  raw_revision     the scan
  erased_revision  the crop with the strike erased (said to be erased, perhaps not cleanly)
  pair_revision    the scan and, as a second image, the erased crop, said to be a reference that may be wrong
  mapped_revision  the scan; the row's layer map (``images.map``: which ink is the strike) goes into the patch embedding beside it
                   (``models.vlm_tokens.patch_embed_with_layer_map``, the weights of ``--layer-map``)
  pair_mapped_revision  the pair, and the layer map into the scan's patch embedding as well
  context_revision      the pair and, as a third image, the line the region stands in, the region boxed in blue (``images.context``)
  problem_revision      the pair, and the statement of the question the page answers (the row's ``problem``) in the words, to tell
                        letters and signs apart by, not to copy
  context_problem_revision  the pair, the line and the question
  raw_context_revision  the scan and, as a second image, its line with the scan laid back in (a reader taught without an erased crop):
                        the words of context_revision without the sentence on the pair, the line called the second image, the box
                        said to hold the scan

Conditions of a whole line (a pack of ``data.line_set pack``: a row is a line, its struck regions erased in ``before`` and taken out in
``after``; a row without the condition's image is not read under it):

  before_line      ``before``, nothing more than the skeleton
  after_line       ``after``, nothing more than the skeleton
  difference_line  ``before`` and, as a second image, ``after``: give only what the first has and the second has not

``--intervention strike_off | writing_off`` (a pack of ``data.struck_pack counterfactual``) reads the rows with their strike, or their
struck writing, taken off the images (``INTERVENED``): what a reader that reads from the evidence should keep reading, and should no
longer read.  With ``--backend hf`` every reading carries its ``confidence``, the geometric mean of the probabilities of its tokens.

Token conditions (``--backend hf``, Qwen3-VL; ``models.vlm_tokens``) leave the image alone and hide visual tokens from the text, each
token keeping its position:

  hide_band_neutral          the scan; the tokens with at least ``--cell-threshold`` of their area in the columns of the masks are hidden
  hide_ink_neutral           the scan; the tokens with at least ``--ink-threshold`` of their area on masked ink are hidden
  soft_band_neutral          the scan; the same tokens as hide_band, their attention logits lowered by ``--soft`` instead of removed
  blanked_hide_band_neutral  the columns whited out and the tokens over them hidden as well

``--view-student`` (``--backend hf``, Qwen3-VL): a vision-tower adapter of ``train_clean_view``; the images of a condition at the places
``--student-images`` names are seen through it, the others through the frozen tower (``models.vlm_tokens.student_features``).
``--mix-pack`` (Qwen3-VL): those images take features on the chord from the frozen tower's to the frozen tower's of the mix pack's same
images instead, ``--mix-alpha`` or ``--mix-matched`` (``models.feature_mix``); OUT/mix_stats.jsonl per row.

Greedy decoding, seed 42; every request carries one image, so nothing in the words can be read back as text.  The backend is vLLM (the
default, for the plain conditions) or transformers (``hf``, for all of them: the plain conditions read the same to the precision of another
kernel).  OUT/predictions.jsonl has a row per (condition, line): the reading (the text inside <transcription>), the raw output, its
length and why it stopped.
"""
import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

from PIL import Image

from .data.line_pack import band_mask, ink_mask
from .models.internvl import processing
from .util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

SYSTEM = 'You are an OCR transcription assistant.'
SKELETON = ('请转写图中的手写数学作答。保留原来的换行；公式使用 LaTeX。无法辨认的字符写 □，不猜测、不解题、不纠错、不补全。'
            '没有可转写的内容时输出 [空]。仅返回 <transcription>转写结果</transcription>，不附带解释。')
FINAL = '只转写学生最终保留的有效作答：被划掉、涂黑或作废的内容不要输出，保留替换后的内容。'
STRUCK = '只转写被划线划掉或涂黑的那部分内容，能认出多少写多少；没有被划掉的内容不要输出。公式写在 $ $ 之间。'
OVERLAY = '图中的红色是已检测出的涂改位置，是辅助标记，不是学生的笔迹。'
BOX = ('只转写蓝色框里学生原来写的内容，框外的内容不要输出。框里的字上可能有划线，也可能划线已经被去掉：有划线时，读出划线下面原来写的字，'
       '不要输出划线本身。')
MARKED = '图中红色的笔画是划线（作废记号），不是学生写的字；黑色的笔画是学生写的字。'
UNREADABLE = '[读不出]'
ILLEGIBLE = f'被划掉的内容完全认不出时，只输出 {UNREADABLE}。'
CLEANED = '图中原来被划线划掉的内容，划线已经被去掉（可能没有去干净，也可能擦掉了一部分字迹）。转写这部分原来写的内容。'
PAIR = '第一张是原图；第二张是去掉划线后的参考图，参考图可能有错，以原图为准。'
CONTEXT = '第三张是这一处所在的那一行，蓝色框标出的就是这一处（框里是去掉划线后的样子）。只转写蓝色框里原来被划掉的内容，框外的内容不要输出，也不要照抄框外写的字。'
CONTEXT_RAW = CONTEXT.replace('第三张', '第二张').replace('（框里是去掉划线后的样子）', '（框里是原图）')
PROBLEM = '这道题的题目如下，只用来帮你辨认字母和符号；只转写图里被划掉的内容，不要照抄题目。\n【题目】{problem}'
DIFFERENCE = ('两张图是同一行手写作答。第一张比第二张多出一段或几段内容，其余部分相同。只转写第一张里有、第二张里没有的内容，两张都有的不要输出。'
              f'公式写在 $ $ 之间。多出的内容完全认不出时，只输出 {UNREADABLE}。')
CONDITIONS = {
    'raw_neutral': ('raw', (), None),
    'raw_final': ('raw', (FINAL,), None),
    'erased_neutral': ('erased', (), None),
    'erased_final': ('erased', (FINAL,), None),
    'blanked_neutral': ('blanked', (), None),
    'blanked_final': ('blanked', (FINAL,), None),
    'overlay_final': ('overlay', (OVERLAY, FINAL), None),
    'hint_final': ('raw', ('hint', FINAL), None),
    'raw_struck': ('raw', (STRUCK,), None),
    'boxed_read': ('raw', (BOX,), None),
    'boxed_marked': ('raw', (MARKED, BOX), None),
    'from_pack': ('raw', None, None),
    'raw_revision': ('raw', (STRUCK, ILLEGIBLE), None),
    'erased_revision': ('erased', (CLEANED, ILLEGIBLE), None),
    'pair_revision': (('raw', 'erased'), (PAIR, STRUCK, ILLEGIBLE), None),
    'mapped_revision': ('raw', (STRUCK, ILLEGIBLE), 'map'),
    'pair_mapped_revision': (('raw', 'erased'), (PAIR, STRUCK, ILLEGIBLE), 'map'),
    'context_revision': (('raw', 'erased', 'context'), (PAIR, CONTEXT, STRUCK, ILLEGIBLE), None),
    'problem_revision': (('raw', 'erased'), (PAIR, STRUCK, ILLEGIBLE, 'problem'), None),
    'context_problem_revision': (('raw', 'erased', 'context'), (PAIR, CONTEXT, STRUCK, ILLEGIBLE, 'problem'), None),
    'raw_context_revision': (('raw', 'context'), (CONTEXT_RAW, STRUCK, ILLEGIBLE), None),
    'before_line': ('before', (), None),
    'after_line': ('after', (), None),
    'difference_line': (('before', 'after'), (DIFFERENCE,), None),
    'hide_band_neutral': ('raw', (), 'band'),
    'hide_ink_neutral': ('raw', (), 'ink'),
    'soft_band_neutral': ('raw', (), 'soft_band'),
    'blanked_hide_band_neutral': ('blanked', (), 'band'),
}
TOKEN_RULES = {c for c, (_, _, rule) in CONDITIONS.items() if rule and rule != 'map'}
REVISION = {c for c in CONDITIONS if c.endswith('_revision')}
MAPPED = {c for c, (_, _, rule) in CONDITIONS.items() if rule == 'map'}
LINE = {c for c in CONDITIONS if c.endswith('_line')}
# an intervention on the evidence: the image that stands in for each image of a row
INTERVENED = {'strike_off': {'raw': 'raw_strike_off'}, 'writing_off': {'raw': 'raw_writing_off', 'erased': 'erased_writing_off'}}


def image_keys(condition):
    """The images of a condition, as the keys of a row's ``images``."""
    keys = CONDITIONS[condition][0]
    return (keys,) if isinstance(keys, str) else keys


def hint_text(boxes):
    """The places of the masks in words: boxes (left, top, right, bottom) in thousandths of the crop's width and height."""
    if not boxes:
        return '没有检测到涂改区域。'
    places = '；'.join(f'[{l}, {t}, {r}, {b}]' for l, t, r, b in boxes)
    return f'已检测出的涂改区域（相对图像宽高的千分比，[左, 上, 右, 下]）：{places}。'


def prompt(condition, row):
    """The words of a condition for a row of the pack."""
    _, parts, _ = CONDITIONS[condition]
    if parts is None:
        return row['prompt']
    said = {'hint': lambda: hint_text(row['boxes']), 'problem': lambda: PROBLEM.format(problem=row['problem'])}
    return '\n'.join([SKELETON, *(said[part]() if part in said else part for part in parts)])


def messages(image, text, system=SYSTEM):
    """A chat request of one image or several (in order, before the words)."""
    def url(one):
        buffer = io.BytesIO()
        one.save(buffer, format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
    shown = image if isinstance(image, (list, tuple)) else [image]
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': [*({'type': 'image_url', 'image_url': {'url': url(one)}} for one in shown), {'type': 'text', 'text': text}]}]


def stopped(tokens, stop, loop_limit):
    """A greedy decode cut where the hf reader's loop check (``models.vlm_tokens.looped``) stops it: the tokens up to the first point
    that ends in one block written ``loop_limit`` times, the block kept once, and whether it was cut.  The decode being greedy, the tokens
    before that point are the ones the hf loop would have written."""
    from .models.vlm_tokens import looped
    for n in range(1, len(tokens)+1):
        if tokens[n-1] in stop:
            return tokens[:n], False
        cut = looped(tokens[:n], loop_limit)
        if cut:
            return tokens[:n-cut], True
    return tokens, False


def reading_of(raw):
    """The text inside <transcription> (all of the output when the tags are missing); [空] is no text.  A thinking model's
    reasoning (up to </think>) is left out: it may quote the tags."""
    raw = raw.rsplit('</think>', 1)[-1]
    found = re.search(r'<transcription>(.*?)(?:</transcription>|$)', raw, re.S)
    text = (found.group(1) if found else raw).strip()
    return '' if text in ('[空]', '空', '【空】') else text


def hidden_flags(reader, rule, row, image, root, args):
    """The bool flags of the image's tokens a token rule hides in a row (None when the rule hides nothing there)."""
    from .models.vlm_tokens import cell_shares, hidden_tokens
    rows, cols = reader.grid(image)
    if rule == 'ink':
        with Image.open(root/row['images']['raw']) as raw, Image.open(root/row['images']['erased']) as erased:
            mask, threshold = ink_mask(raw, erased), args.ink_threshold
    else:
        mask, threshold = band_mask(row['boxes'], row['size']), args.cell_threshold
    flags = hidden_tokens(cell_shares(mask, rows, cols), threshold)
    return flags if flags.any() else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--conditions', nargs='+', choices=tuple(CONDITIONS), default=[c for c in CONDITIONS if c not in TOKEN_RULES | REVISION | LINE | {'raw_struck', 'boxed_read', 'boxed_marked', 'from_pack'}])
    p.add_argument('--backend', choices=('vllm', 'hf'), default='vllm')
    p.add_argument('--adapter', type=Path, help='hf: a LoRA adapter of train_reader, merged into the model')
    p.add_argument('--layer-map', type=Path, help='hf: the layer-map weights of train_reader --layer-map (mapped conditions)')
    p.add_argument('--fold', type=int, help='only the rows of this fold of the pack (the writers a reader of train_reader --fold never trained on)')
    p.add_argument('--intervention', choices=tuple(INTERVENED), help='read the rows with their strike, or their struck writing, taken off (a counterfactual pack)')
    p.add_argument('--tp', type=int, default=1, help='vllm: tensor parallel size (the 32B model needs two cards)')
    p.add_argument('--gpu-memory', type=float, default=.90, help='vllm: share of each card the engine may take (lower it when others hold memory there)')
    p.add_argument('--batch-size', type=int, default=64, help='requests per progress checkpoint (hf: lines per forward pass, keep it small)')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--loop-limit', type=int, default=0, help='stop a reading that ends in one block written this many times, keeping the block once (0: never)')
    p.add_argument('--max-model-len', type=int, default=4096, help='vllm: context length (whole-question images need more)')
    p.add_argument('--cell-threshold', type=float, default=.25, help='token rules: share of a token cell under the band that hides it')
    p.add_argument('--ink-threshold', type=float, default=.05, help='token rules: share of a token cell on masked ink that hides it')
    p.add_argument('--soft', type=float, default=4., help='soft_band: bias subtracted from the attention logits of the hidden tokens')
    p.add_argument('--view-student', type=Path, help='hf: a vision-tower adapter of train_clean_view; the images --student-images names are seen through it')
    p.add_argument('--student-images', type=int, nargs='+', default=[], help='with --view-student: the places (1-based) of a condition\'s images the student sees')
    p.add_argument('--student-off', action='store_true', help='with --view-student: those images go the student\'s way with its adapter disabled (the wiring check)')
    p.add_argument('--mix-pack', type=Path, help='hf, Qwen3-VL: the images --student-images names take features on the chord to this pack\'s same images (models.feature_mix)')
    p.add_argument('--mix-alpha', type=float, help='with --mix-pack: z = (1 - A) x + A c')
    p.add_argument('--mix-matched', choices=('x', 's'), help='with --mix-pack and --view-student: per token as close to c as the student; s farther than x -> x (M) or s (M\')')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='only the first N lines (a smoke test)')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    require(len(set(args.conditions)) == len(args.conditions), 'duplicate condition')
    require(args.backend == 'hf' or not (TOKEN_RULES | MAPPED) & set(args.conditions), 'token and mapped conditions need --backend hf')
    require(bool(args.layer_map) == bool(MAPPED & set(args.conditions)) and (not args.layer_map or set(args.conditions) <= MAPPED), 'mapped conditions, and only they, read with --layer-map')
    require(args.backend == 'hf' or not args.adapter, '--adapter needs --backend hf')
    require(args.backend == 'vllm' or 'from_pack' not in args.conditions, 'from_pack needs --backend vllm')
    require(args.backend == 'hf' or not args.view_student, '--view-student needs --backend hf')
    require(not args.mix_pack or ((args.mix_alpha is None) != (args.mix_matched is None) and bool(args.student_images) and args.backend == 'hf'
                                  and bool(args.view_student) == bool(args.mix_matched) and not args.student_off),
            '--mix-pack takes one of --mix-alpha / --mix-matched and --student-images (the mixed places); --mix-matched, and only it, a --view-student')
    require(bool(args.view_student) == bool(args.student_images) or args.mix_pack, '--view-student and --student-images go together')
    pack = json.loads((args.pack/'pack.json').read_text())
    require(pack.get('schema') == 'revrender-line-pack', 'not a line pack')
    rows = [r for r in pack['rows'] if args.fold is None or r['fold'] == args.fold][:args.limit or None]
    require(rows, 'no row to read')
    jobs = [(c, row) for c in args.conditions for row in rows if all(key in row['images'] for key in image_keys(c))]
    require(jobs, 'no row has the images of the conditions')
    out = make_output_dir(args.output_dir); reader = None
    if args.backend == 'hf':
        from .models.vlm_tokens import reader_for
        reader = reader_for(args.model, adapter=args.adapter, layer_map=args.layer_map, student=args.view_student, student_off=args.student_off)
        if args.mix_pack:
            from .models.feature_mix import install
            require(type(reader).__module__.endswith('vlm_tokens'), '--mix-pack reads with a Qwen3-VL reader')
            mix, mixed_rows = install(reader, args.mix_alpha, args.mix_matched), {r['id']: r for r in json.loads((args.mix_pack/'pack.json').read_text())['rows']}
    else:
        from vllm import LLM, SamplingParams
        model = LLM(model=str(args.model), dtype='bfloat16', seed=SEED, enforce_eager=True, max_model_len=args.max_model_len, max_num_seqs=16, gpu_memory_utilization=args.gpu_memory,
                    limit_mm_per_prompt={'image': max(len(image_keys(c)) for c in args.conditions)}, tensor_parallel_size=args.tp, trust_remote_code=args.trust_remote_code, disable_log_stats=True)
        sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=SEED)
        tokenizer = model.get_tokenizer(); special = set(tokenizer.all_special_ids)
    write_manifest(out, config=to_jsonable(args), inputs={str(args.pack/'pack.json'): digest(args.pack/'pack.json'),
                                                           **{str(w): digest(w) for w in sorted(args.model.glob('*.safetensors'))},
                                                           **({str(w): digest(w) for w in sorted(args.adapter.glob('*.safetensors'))} if args.adapter else {}),
                                                           **({str(args.layer_map): digest(args.layer_map)} if args.layer_map else {}),
                                                           **({str(w): digest(w) for w in sorted(args.view_student.glob('*.safetensors'))} if args.view_student else {}),
                                                           **({str(args.mix_pack/'pack.json'): digest(args.mix_pack/'pack.json')} if args.mix_pack else {})},
                   backend=args.backend, rows=len(jobs), lines=len(rows), model_processing=processing(args.model),
                   conditions={c: {'image': image_keys(c), 'token_rule': CONDITIONS[c][2], 'example_prompt': prompt(c, rows[0]), 'rows': sum(j[0] == c for j in jobs)} for c in args.conditions},
                   system=SYSTEM, generation={'temperature': 0, 'max_new_tokens': args.max_new_tokens, 'loop_limit': args.loop_limit, 'dtype': 'bfloat16'})
    name = args.model.name+(f'+{args.adapter.parent.name}' if args.adapter else '')+(f'+{args.view_student.parent.name}' if args.view_student else '')      # the reading of a model with an adapter is another model's
    started, done = time.monotonic(), 0
    mix_stream = (out/'mix_stats.jsonl').open('x', encoding='utf-8') if args.mix_pack else None
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start+args.batch_size]
            swap = INTERVENED.get(args.intervention, {})
            opened = lambda row, key: Image.open(args.pack/row['images'][swap.get(key, key)]).convert('RGB')
            images = [[opened(row, key) for key in image_keys(condition)] for condition, row in batch]
            if args.backend == 'hf':
                requests = [(shown if len(shown) > 1 else shown[0], prompt(c, row),
                             hidden_flags(reader, CONDITIONS[c][2], row, shown[0], args.pack, args) if c in TOKEN_RULES else None,
                             args.soft if CONDITIONS[c][2] == 'soft_band' else 0.) for (c, row), shown in zip(batch, images)]
                places = [[n+1 in args.student_images for n in range(len(shown))] for shown in images]
                results = reader.read(requests, SYSTEM, args.max_new_tokens, maps=[opened(row, 'map') for _, row in batch] if args.layer_map else None,      # of each job's first image
                                      loop_limit=args.loop_limit, students=places if args.view_student or args.mix_pack else None,
                                      mix_images=[[Image.open(args.mix_pack/mixed_rows[row['id']]['images'][key]).convert('RGB') for key in image_keys(c)] for c, row in batch] if args.mix_pack else None)
                if mix_stream:
                    at = 0
                    for (c, row), shown in zip(batch, images):
                        mix_stream.write(json.dumps({'id': row['id'], 'images': {str(n+1): mix.stats[at+n] for n in range(len(shown)) if mix.stats[at+n] is not None}})+'\n')
                        at += len(shown)
                    mix_stream.flush()
            else:
                generated = model.chat([messages(shown, prompt(c, row), row.get('system', SYSTEM) if c == 'from_pack' else SYSTEM)
                                        for (c, row), shown in zip(batch, images)], sampling, use_tqdm=False)
                results = []
                for g in generated:
                    kept, cut = stopped(list(g.outputs[0].token_ids), special, args.loop_limit) if args.loop_limit else (g.outputs[0].token_ids, False)
                    results.append({'text': tokenizer.decode(kept, skip_special_tokens=True) if cut else g.outputs[0].text, 'tokens': len(kept),
                                    'finish': 'loop' if cut else g.outputs[0].finish_reason})
            require(len(results) == len(batch), 'missing generation results')
            for (condition, row), result in zip(batch, results):
                stream.write(json.dumps({'condition': condition, 'id': row['id'], 'line_id': row['line_id'], 'model': name, 'text': reading_of(result['text']), **({'confidence': result['confidence']} if 'confidence' in result else {}),
                                         'raw': result['text'], 'generated_tokens': result['tokens'], 'finish_reason': result['finish'],
                                         **({'thinking_tokens': result['thinking_tokens']} if 'thinking_tokens' in result else {})}, ensure_ascii=False)+'\n')
            stream.flush()
            done += len(batch)
            print(json.dumps({'completed': done, 'total': len(jobs), 'seconds': round(time.monotonic()-started, 1)}), flush=True)
    if mix_stream:
        mix_stream.close()
    audio = getattr(reader, 'audio_calls', None)                              # Gemma: the audio tower's calls (0)
    write_json(out/'complete.json', {'rows': done, 'seconds': round(time.monotonic()-started, 1), 'predictions_sha256': digest(out/'predictions.jsonl'),
                                     **({'audio_tower_calls': len(audio)} if audio is not None else {})})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
