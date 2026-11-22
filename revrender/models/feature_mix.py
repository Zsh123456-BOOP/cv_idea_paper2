"""The features a Qwen3-VL reader takes for some images replaced by points on the chord from the scan's features to the erased
crop's (T2 of the plan of 2026-11-22, 三个不训练的检验, revision one): ``recognize_lines --mix-pack PACK (--mix-alpha A | --mix-matched
x|s)``.  Per batch the frozen tower sees the read pack's images (x; every image not mixed keeps these) and the same jobs' images of
``PACK`` (c, in the same order, so a mixed image at A = 1 is that pack's reading bit for bit); per mixed image, per level (the merged
tokens and the three deepstack levels), per token:

- ``A``: z = (1 - A) x + A c, in float32, then the model's dtype (A = 0 is x, A = 1 is c, bit for bit).
- ``matched``: the plug-in's features s (the reader's ``--view-student``) as well; the A that gives cos(z, c) = cos(s, c)
  (``matched_alpha``: along the chord the cosine is monotone, solved in float64); a token where s is farther from c than x
  (cos(s, c) < cos(x, c)) takes A = 0 (``x``, arm M) or keeps s itself (``s``, arm M').

``Mix.stats``: per image of the last batch, None (not mixed) or per level the tokens, the tokens taken back (s farther than x), and the
largest |cos(z, c) - cos(s, c)| of the others in float32 before the cast (check 5) and whether the kept s tokens are s bit for bit."""
import contextlib

LEVELS = ('merged', 'deepstack1', 'deepstack2', 'deepstack3')


def chord_cosine(p, q, a2, b, d2, cn, alpha):
    """cos((1 - alpha) x + alpha c, c) from the per-token scalars: p = <x, c>, q = <c - x, c>, a2 = |x|^2, b = <x, c - x>,
    d2 = |c - x|^2, cn = |c|."""
    import torch
    return (p+alpha*q)/(cn*torch.sqrt((a2+2*alpha*b+alpha*alpha*d2).clamp_min(1e-300))).clamp_min(1e-300)


def matched_alpha(x, c, s, halvings=60):
    """Per token (rows of [tokens, width]), the alpha in [0, 1] with cos(z, c) = cos(s, c) along the chord z = (1 - alpha) x + alpha c,
    and the tokens where s is farther from c than x (alpha 0).  In float64; the cosine along the chord is monotone in alpha, so the
    root is bracketed by [0, 1] and halved ``halvings`` times on the closed form (``chord_cosine``)."""
    import torch
    x, c, s = (t.double() for t in (x, c, s))
    d = c-x; cn = c.norm(dim=-1).clamp_min(1e-300)
    p, q, a2, b, d2 = (x*c).sum(-1), (d*c).sum(-1), (x*x).sum(-1), (x*d).sum(-1), (d*d).sum(-1)
    target = (s*c).sum(-1)/(s.norm(dim=-1).clamp_min(1e-300)*cn)
    base = chord_cosine(p, q, a2, b, d2, cn, torch.zeros_like(p))
    farther = target < base
    lo, hi = torch.zeros_like(p), torch.ones_like(p)
    for _ in range(halvings):
        mid = (lo+hi)/2
        up = chord_cosine(p, q, a2, b, d2, cn, mid) < target
        lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
    alpha = torch.where(farther, torch.zeros_like(p), (lo+hi)/2)
    return alpha, farther


def mixed_level(x, c, s, alpha, matched):
    """One level of one image: (z in x's dtype, its stats)."""
    import torch
    stats = {'tokens': int(x.shape[0])}
    if not matched:
        a = torch.full((x.shape[0],), float(alpha), device=x.device, dtype=torch.float32); farther = None
    else:
        a64, farther = matched_alpha(x, c, s); a = a64.float()
    z32 = (1-a)[:, None]*x.float()+a[:, None]*c.float()
    if matched:
        cos = lambda u, v: torch.nn.functional.cosine_similarity(u.double(), v.double(), dim=-1)
        solved = ~farther
        stats.update(taken_back=int(farther.sum()), alpha_mean=float(a[solved].mean()) if solved.any() else None,
                     largest_cos_gap=float((cos(z32, c)-cos(s, c))[solved].abs().max()) if solved.any() else 0.)
    z = z32.to(x.dtype)
    if matched == 's':
        z = torch.where(farther[:, None], s, z)
        stats['kept_s_bit_for_bit'] = bool(torch.equal(z[farther], s[farther]))
    return z, stats


class Mix:
    """What ``mixed_features`` replaces: ``flags`` per image of the next batch (True: mixed), ``companion`` the processed images of
    the mix pack for the same batch (pixel values, grid), ``alpha`` or ``matched`` ('x' or 's'; with ``view``, the reader's student)."""

    def __init__(self, alpha=None, matched=None, view=None):
        if (alpha is None) == (matched is None) or (matched and view is None) or matched not in (None, 'x', 's'):
            raise ValueError('a mix is one alpha or matched x / s with a student')
        self.alpha, self.matched, self.view = alpha, matched, view
        self.flags = self.companion = None; self.stats = []


def mixed_features(features, mix):
    """``get_image_features`` of a Qwen3-VL with the images ``mix.flags`` names replaced (see the module's words)."""
    import torch
    from .vlm_tokens import image_levels

    def chosen(pixel_values, image_grid_thw=None, **kwargs):
        frozen = mix.view.disable_adapter() if mix.view is not None else contextlib.nullcontext()
        with torch.no_grad():
            with frozen:
                out = features(pixel_values, image_grid_thw, **kwargs); xs = image_levels(out)
                pixels, grid = mix.companion
                if not torch.equal(grid.to(image_grid_thw.device), image_grid_thw):
                    raise ValueError('the mix pack gives other grids')
                cs = image_levels(features(pixels.to(pixel_values.device, pixel_values.dtype), grid.to(image_grid_thw.device), **kwargs))
            ss = None
            if mix.matched:
                mix.view.set_adapter('default', inference_mode=True)
                ss = image_levels(features(pixel_values, image_grid_thw, **kwargs))
            flags = mix.flags or [False]*len(xs)
            if len(flags) != len(xs):
                raise ValueError(f'{len(flags)} mix flags for {len(xs)} images')
            chosen_, mix.stats = [], []
            for i, f in enumerate(flags):
                if not f:
                    chosen_.append(xs[i]); mix.stats.append(None); continue
                pairs = [mixed_level(x, c, ss[i][n] if ss else None, mix.alpha, mix.matched) for n, (x, c) in enumerate(zip(xs[i], cs[i]))]
                chosen_.append([z for z, _ in pairs]); mix.stats.append({name: st for name, (_, st) in zip(LEVELS, pairs)})
        embeds, deep = tuple(p[0] for p in chosen_), [tuple(p[n] for p in chosen_) for n in range(1, len(chosen_[0]))]
        if isinstance(out, (tuple, list)):                                  # 4.57: the levels each one tensor
            return embeds, [torch.cat(level) for level in deep]
        out.pooler_output = embeds
        if getattr(out, 'deepstack_features', None) is not None:
            out.deepstack_features = deep
        return out
    return chosen


def install(reader, alpha=None, matched=None):
    """Route a Qwen3-VL ``vlm_tokens.Reader``'s image features through a ``Mix`` (its student, if any, gives s); returns the mix."""
    inner = reader.model.model
    if reader.students is not None:                                     # undo the student route: the mix calls the tower itself
        del inner.get_image_features
    mix = Mix(alpha, matched, reader.students.view if reader.students is not None else None)
    inner.get_image_features = mixed_features(inner.get_image_features, mix)
    return mix


def grids(args):
    """Check 4: every row's mixed images give one grid in the read pack and the mix pack."""
    import json
    from transformers import AutoProcessor
    from PIL import Image
    from ..recognize_lines import image_keys
    processor = AutoProcessor.from_pretrained(str(args.model)); keys = image_keys('context_revision')
    read, mixed = ({r['id']: r for r in json.loads((p/'pack.json').read_text())['rows']} for p in (args.pack, args.mix_pack))
    grid = lambda pack, row, key: processor.image_processor(images=[Image.open(pack/row['images'][key]).convert('RGB')], return_tensors='pt')['image_grid_thw'][0].tolist()
    differ = [{'id': k, 'image': n} for k in sorted(read) for n in args.places if grid(args.pack, read[k], keys[n-1]) != grid(args.mix_pack, mixed[k], keys[n-1])]
    return {'rows': len(read), 'images': args.places, 'differ': differ, 'passed': not differ and set(read) == set(mixed)}


def routes(args):
    """Check 3: on the first ``--regions`` rows of a fold, in batches as read, a place set's features beside V-修's (2 3) and the frozen
    tower's, bit for bit: the places it names are V-修's, the others the frozen tower's."""
    import json
    import torch
    from PIL import Image
    from ..recognize_lines import image_keys
    from .vlm_tokens import Reader, image_levels
    reader = Reader(args.model, student=args.student); inner = reader.model.model; keys = image_keys('context_revision')
    rows = sorted((r for r in json.loads((args.pack/'pack.json').read_text())['rows'] if r['fold'] == args.fold), key=lambda r: r['id'])[:args.regions]
    sets = {'V-修': (2, 3), 'V-只2': (2,), 'V-只3': (3,)}; out = {name: {'images': 0, 'same_as_student': 0, 'same_as_frozen': 0, 'differ': []} for name in sets if name != 'V-修'}
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        processed = reader.processor.image_processor(images=[Image.open(args.pack/r['images'][k]).convert('RGB') for r in batch for k in keys], return_tensors='pt')
        pv, grid = processed['pixel_values'].to(reader.device, reader.model.dtype), processed['image_grid_thw'].to(reader.device)

        def features(places):
            reader.students.flags = [('default' if n+1 in places else None) for _ in batch for n in range(len(keys))] if places else None
            with torch.inference_mode():
                return image_levels(inner.get_image_features(pv, grid))
        frozen, student = features(()), features(sets['V-修'])
        for name in out:
            got = features(sets[name])
            for i, levels in enumerate(got):
                place = i % len(keys)+1; want = student[i] if place in sets[name] else frozen[i]
                same = all(torch.equal(a, b) for a, b in zip(levels, want)); kind = 'same_as_student' if place in sets[name] else 'same_as_frozen'
                out[name]['images'] += 1; out[name][kind] += same
                if not same:
                    out[name]['differ'].append({'id': batch[i//len(keys)]['id'], 'image': place})
    return {'regions': len(rows), 'sets': out, 'passed': all(not v['differ'] for v in out.values())}


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    from ..util import make_output_dir, write_json, write_manifest
    p = argparse.ArgumentParser(description='The checks of T2 / T1 (三个不训练的检验): grids (check 4), routes (check 3).')
    p.add_argument('check', choices=('grids', 'routes'))
    p.add_argument('--model', type=Path, required=True); p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--mix-pack', type=Path); p.add_argument('--places', type=int, nargs='+', default=[2, 3])
    p.add_argument('--student', type=Path); p.add_argument('--fold', type=int, default=0); p.add_argument('--regions', type=int, default=20)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    out = make_output_dir(args.output_dir); write_manifest(out, config={k: str(v) for k, v in vars(args).items()})
    result = grids(args) if args.check == 'grids' else routes(args)
    write_json(out/'result.json', result); print(json.dumps({k: v for k, v in result.items() if k != 'differ'}, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
