"""MAA's three-branch adapter (Li et al., ICML 2026; its public code, ``adapters.py`` and ``modeling.py``) in a Qwen3-VL vision tower
(plan of 2026-11-22, 插件结构优化, S1 / S2; revision of section eight).

Each block of the tower gets an adapter on its input, as MAA wraps each layer of its CLIP encoder: a bottleneck MLP (LayerNorm ->
256 -> GELU -> width) plus a 7 x 7 local window self-attention (128 wide, 8 heads, relative position bias, scaled by ``beta_attn``
from 1) together scaled by ``alpha_mlp``, and a 3 x 3 max pool scaled by ``alpha_pool``; the two alphas start at 0, so the tower
starts as it was.  What MAA's code does is kept, a window smaller than 7 at an image's edge included (``window_h = min(7, rows)``:
then that image's windows carry no relative position bias).  What had to change: Qwen's tower has no CLS token (every token is a
patch); an image's rows and columns are the processor's (``grid_thw``), not guessed from a square root; the images of a batch, laid
end to end in one sequence, are adapted one at a time; and Qwen lays an image's patches out by merge group (``merge x merge``
patches together), so they are put in row order for the window and the pool (``raster``) and back (``merge_order``).  The adapter's
weights are float32; it runs under bf16 autocast on a card (MAA trains under autocast), its residuals added in float32 and the sum
cast back, as MAA's forward does.

``MaaView`` stands where a peft model stands for the low-rank adapter: ``disable_adapter()`` is the frozen tower, ``set_adapter``,
``load_adapter``, ``save_pretrained`` / ``MaaView.from_pretrained`` (a folder with ``CONFIG`` and ``WEIGHTS``), ``base_model.model``
the tower.  The block wrappers take the image grid and the adapter in use as keyword arguments from the tower's forward, so a block
recomputed in the backward pass (``train_clean_view.checkpoint_layers``) recomputes what its forward did.
"""
import contextlib
import json
import types
from pathlib import Path

MLP_HIDDEN, ATTN_DIM, HEADS, WINDOW, POOL = 256, 128, 8, 7, 3          # adapters.py's constants and kernel_size
CONFIG, WEIGHTS = 'maa_config.json', 'maa_adapter.safetensors'


def raster(tokens, t, h, w, merge):
    """[t, h, w, width] in row order from an image's tokens [t*h*w, width] in Qwen's merge-group order."""
    width = tokens.shape[-1]
    return tokens.reshape(t, h//merge, w//merge, merge, merge, width).permute(0, 1, 3, 2, 4, 5).reshape(t, h, w, width)


def merge_order(grid, merge):
    """[t*h*w, width] in Qwen's merge-group order from [t, h, w, width] in row order (``raster`` undone)."""
    t, h, w, width = grid.shape
    return grid.reshape(t, h//merge, merge, w//merge, merge, width).permute(0, 1, 3, 2, 4, 5).reshape(-1, width)


def modules():
    """The adapter's two modules (built on first use, so importing this needs no torch)."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    class WindowSelfAttention(nn.Module):
        """adapters.py's ``WindowSelfAttention``, line for line."""

        def __init__(self, dim, attn_dim, num_heads, window_size):
            super().__init__()
            self.dim, self.attn_dim, self.num_heads = int(dim), int(attn_dim), int(num_heads)
            self.head_dim = self.attn_dim//self.num_heads; self.scale = self.head_dim**-.5
            self.window_h = self.window_w = int(window_size)
            self.qkv = nn.Linear(self.dim, 3*self.attn_dim); self.proj = nn.Linear(self.attn_dim, self.dim)
            self.beta_attn = nn.Parameter(torch.tensor(1.))
            window_h, window_w = self.window_h, self.window_w
            self.relpos_table = nn.Parameter(torch.zeros(self.num_heads, (2*window_h-1)*(2*window_w-1)))
            coords = torch.stack(torch.meshgrid(torch.arange(window_h), torch.arange(window_w), indexing='ij'))
            flat = torch.flatten(coords, 1); rel = flat[:, :, None]-flat[:, None, :]
            rel[0] += window_h-1; rel[1] += window_w-1; rel[0] *= 2*window_w-1
            self.register_buffer('relpos_index', rel[0]+rel[1], persistent=False)

        def forward(self, patches_2d):
            batch, dim, height, width = patches_2d.shape
            window_h, window_w = min(self.window_h, height), min(self.window_w, width)
            pad_h, pad_w = (window_h-height % window_h) % window_h, (window_w-width % window_w) % window_w
            if pad_h or pad_w:
                patches_2d = F.pad(patches_2d, (0, pad_w, 0, pad_h), mode='replicate'); padded_h, padded_w = height+pad_h, width+pad_w
            else:
                padded_h, padded_w = height, width
            nh, nw = padded_h//window_h, padded_w//window_w
            tokens = patches_2d.permute(0, 2, 3, 1).contiguous().reshape(batch, nh, window_h, nw, window_w, dim).permute(0, 1, 3, 2, 4, 5).contiguous()
            tokens = tokens.reshape(batch*nh*nw, window_h*window_w, dim)
            query, key, value = self.qkv(tokens).chunk(3, dim=-1)
            heads = lambda z: z.reshape(tokens.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
            query, key, value = heads(query), heads(key), heads(value)
            attn = (query*self.scale) @ key.transpose(-2, -1)
            if window_h == self.window_h and window_w == self.window_w:
                bias = self.relpos_table[:, self.relpos_index.view(-1)].view(self.num_heads, window_h*window_w, window_h*window_w)
                attn = attn+bias.unsqueeze(0)
            out = (attn.softmax(dim=-1) @ value).transpose(1, 2).contiguous().reshape(tokens.shape[0], window_h*window_w, self.attn_dim)
            out = self.beta_attn*self.proj(out)
            out = out.reshape(batch, nh, nw, window_h, window_w, dim).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch, padded_h, padded_w, dim)
            if pad_h or pad_w:
                out = out[:, :height, :width, :]
            return out.reshape(batch, height*width, dim)

    class Adapter(nn.Module):
        """adapters.py's ``MAAAdapter`` without the CLS token, on one image's tokens in row order [t, h, w, width] (t frames as the batch)."""

        def __init__(self, dim, kernel_size=POOL):
            super().__init__()
            self.dim = int(dim)
            self.norm = nn.LayerNorm(self.dim)
            self.mlp = nn.Sequential(nn.Linear(self.dim, MLP_HIDDEN), nn.GELU(), nn.Linear(MLP_HIDDEN, self.dim))
            self.window_attn = WindowSelfAttention(self.dim, max((ATTN_DIM//HEADS)*HEADS, HEADS), HEADS, WINDOW)
            self.pool = nn.MaxPool2d(kernel_size=int(kernel_size), stride=1, padding=int(kernel_size)//2)
            self.alpha_mlp = nn.Parameter(torch.zeros(1)); self.alpha_pool = nn.Parameter(torch.zeros(1))

        def forward(self, grid):
            batch, height, width, dim = grid.shape
            x = grid.reshape(batch, height*width, dim); input_dtype = x.dtype; adapter_dtype = self.alpha_mlp.dtype
            autocast = torch.autocast('cuda', dtype=torch.bfloat16) if x.is_cuda else contextlib.nullcontext()
            with autocast:
                mlp_residual = self.mlp(self.norm(x.to(adapter_dtype)))
                patches_2d = x.transpose(1, 2).reshape(batch, dim, height, width).to(adapter_dtype)
                fused = mlp_residual+self.window_attn(patches_2d)
                pool_residual = self.pool(patches_2d).reshape(batch, dim, height*width).transpose(1, 2)
            out = x.to(torch.float32)+self.alpha_mlp.to(torch.float32)*fused.to(torch.float32)
            out = out+self.alpha_pool.to(torch.float32)*pool_residual.to(torch.float32)
            return out.to(input_dtype).reshape(batch, height, width, dim)
    return Adapter


def adapter_set(width, depth):
    from torch import nn
    Adapter = modules()
    return nn.ModuleList([Adapter(width) for _ in range(depth)])


class MaaView:
    """A Qwen3-VL vision tower with named sets of MAA adapters, one adapter per block on the block's input; ``active`` the set in use
    (None: the frozen tower)."""

    def __new__(cls, tower, names=('default',), sets=None):
        import torch
        from torch import nn

        class View(nn.Module):
            def __init__(self):
                super().__init__()
                for q in tower.parameters():                         # the tower stays frozen; only the adapters learn (as peft freezes its base)
                    q.requires_grad_(False)
                self.tower = tower
                self.sets = nn.ModuleDict(sets or {n: adapter_set(tower.config.hidden_size, len(tower.blocks)) for n in names})
                self.sets.to(next(tower.parameters()).device)
                self.active = next(iter(self.sets), None)
                self.base_model = types.SimpleNamespace(model=tower)

            def forward(self, *args, **kwargs):
                return self.tower(*args, **kwargs)

            @contextlib.contextmanager
            def disable_adapter(self):
                kept = self.active; self.active = None
                try:
                    yield
                finally:
                    self.active = kept

            def set_adapter(self, name, inference_mode=True):
                self.active = name
                if inference_mode:
                    for q in self.sets[name].parameters(): q.requires_grad_(False)

            def adapt(self, index, hidden, grid, name):
                merge = self.tower.spatial_merge_size; out, start = [], 0
                for t, h, w in grid.tolist():
                    n = t*h*w
                    out.append(merge_order(self.sets[name][index](raster(hidden[start:start+n], t, h, w, merge)), merge)); start += n
                if start != hidden.shape[0]:
                    raise ValueError(f'{hidden.shape[0]} tokens but the grid holds {start}')
                return torch.cat(out)

            def alphas(self, name=None):
                """The set's two alphas over its blocks (mean and largest magnitude) and beta_attn's mean: how far it has moved."""
                one = self.sets[name or self.active or next(iter(self.sets))]
                pick = lambda f: torch.cat([f(a).detach().float().reshape(-1) for a in one])
                mlp, pool, beta = pick(lambda a: a.alpha_mlp), pick(lambda a: a.alpha_pool), pick(lambda a: a.window_attn.beta_attn)
                return {'alpha_mlp_mean_abs': float(mlp.abs().mean()), 'alpha_mlp_max_abs': float(mlp.abs().max()),
                        'alpha_pool_mean_abs': float(pool.abs().mean()), 'alpha_pool_max_abs': float(pool.abs().max()), 'beta_attn_mean': float(beta.mean())}

            def save_pretrained(self, folder, name=None):
                from safetensors.torch import save_file
                folder = Path(folder); folder.mkdir(parents=True)
                name = name or self.active or next(iter(self.sets))
                save_file({k: v.detach().cpu().contiguous() for k, v in self.sets[name].state_dict().items()}, str(folder/WEIGHTS))
                (folder/CONFIG).write_text(json.dumps({'structure': 'maa', 'width': tower.config.hidden_size, 'depth': len(tower.blocks),
                                                       'mlp_hidden': MLP_HIDDEN, 'attn_dim': ATTN_DIM, 'heads': HEADS, 'window': WINDOW, 'pool': POOL,
                                                       'placement': 'the input of each block'}, indent=1))

            def load_adapter(self, folder, adapter_name='default'):
                from safetensors.torch import load_file
                one = adapter_set(tower.config.hidden_size, len(tower.blocks))
                one.load_state_dict(load_file(str(Path(folder)/WEIGHTS)))
                self.sets[adapter_name] = one.to(next(tower.parameters()).device)
        view = View()
        wire(tower, view)
        return view

    @staticmethod
    def from_pretrained(tower, folder, adapter_name='default'):
        view = MaaView(tower, names=())
        view.load_adapter(folder, adapter_name); view.active = adapter_name
        return view


def student_view(tower, adapters):
    """The tower with the vision adapters ``{name: folder}`` loaded, the first active: a peft model, or a ``MaaView`` when the folders
    hold MAA adapters (all of one kind)."""
    names = list(adapters); kinds = {is_maa(adapters[n]) for n in names}
    if len(kinds) != 1:
        raise ValueError('the adapters mix MAA and low-rank ones')
    if kinds == {True}:
        view = MaaView.from_pretrained(tower, adapters[names[0]], names[0])
    else:
        from peft import PeftModel
        view = PeftModel.from_pretrained(tower, str(adapters[names[0]]), adapter_name=names[0])
    for name in names[1:]:
        view.load_adapter(str(adapters[name]), adapter_name=name)
    return view


def is_maa(folder):
    return folder is not None and (Path(folder)/CONFIG).exists()


def wire(tower, view):
    """The tower's forward passes its grid and the set in use to every block (``maa_grid``, ``maa_set``); each block adapts its input
    with that set's adapter of its own index first (nothing when the set is None)."""
    plain_tower = tower.forward

    def tower_forward(hidden_states, grid_thw, **kwargs):
        return plain_tower(hidden_states, grid_thw, maa_grid=grid_thw, maa_set=view.active, **kwargs)
    tower.forward = tower_forward
    for index, block in enumerate(tower.blocks):
        plain = block.forward

        def block_forward(hidden_states, *args, maa_grid=None, maa_set=None, _index=index, _plain=plain, **kwargs):
            if maa_set is not None:
                hidden_states = view.adapt(_index, hidden_states, maa_grid, maa_set)
            return _plain(hidden_states, *args, **kwargs)
        block.forward = block_forward


def checks(args):
    """The plan's checks 1-3 for S1 / S2 (section four, with section eight's two of check 2) on ``--regions`` regions of the scan
    pack, three images each as V reads them: 1 a new set (alphas 0) gives every level bit for bit as the frozen tower; 2 per image the
    adapter's rows x columns are the processor's and make its token count, ``raster`` then ``merge_order`` gives the tokens back bit
    for bit, and the image's own patches laid out in ``raster`` order make the picture again (the processor's input, resized to the
    grid beforehand so it is not resized, its normalisation undone and rounded: equal to the 8-bit picture); 3 after one backward of
    the alignment loss, only adapter parameters have gradients, and how many there are."""
    import numpy as np
    import torch
    from PIL import Image
    from ..train_clean_view import encode, levels, pair_loss, vision_tower
    from .vlm_tokens import processor_for
    rows = json.loads((args.scan_pack/'pack.json').read_text())['rows'][:args.regions]
    processor = processor_for(args.model); image_processor = processor.image_processor
    torch.manual_seed(42)
    view = MaaView(vision_tower(args.model).to('cuda')); view.eval(); merge = view.tower.spatial_merge_size; patch = view.tower.patch_size
    opened = lambda path: Image.open(path).convert('RGB')
    result = {'regions': len(rows), 'check1': [], 'check2': [], 'parameters': sum(q.numel() for q in view.sets.parameters())}
    with torch.no_grad():
        for r in rows:
            x, g = encode(processor, [opened(args.scan_pack/r['images'][k]) for k in ('raw', 'erased', 'context')], 'cuda')
            with view.disable_adapter():
                frozen = levels(view, x, g)
            adapted = levels(view, x, g)
            result['check1'].append({'id': r['id'], 'images': int(g.shape[0]), 'bitwise': all(torch.equal(a, b) for a, b in zip(frozen, adapted))})
            for key in ('raw', 'erased', 'context'):
                image = opened(args.scan_pack/r['images'][key])
                t, h, w = (int(v) for v in image_processor(images=[image], return_tensors='pt')['image_grid_thw'][0])
                exact = image.resize((w*patch, h*patch), Image.BICUBIC)
                batch = image_processor(images=[exact], do_resize=False, return_tensors='pt')
                pixels, grid = batch['pixel_values'].float(), [int(v) for v in batch['image_grid_thw'][0]]
                tokens = torch.randn(t*h*w, 8)
                laid = raster(pixels, t, h, w, merge)                                   # [t, h, w, C*tp*p*p], a patch laid out channel first
                channels = len(image_processor.image_mean); temporal = laid.shape[-1]//(channels*patch*patch)
                picture = laid.reshape(t, h, w, channels, temporal, patch, patch)[0, :, :, :, 0].permute(0, 3, 1, 4, 2).reshape(h*patch, w*patch, channels)
                mean, std = torch.tensor(image_processor.image_mean), torch.tensor(image_processor.image_std)
                back = torch.round((picture*std+mean)*255).clamp(0, 255).to(torch.uint8).numpy()
                result['check2'].append({'id': r['id'], 'image': key, 'grid': [t, h, w], 'grid_unresized': grid, 'tokens': int(pixels.shape[0]),
                                         'rows_x_cols_is_tokens': t*h*w == pixels.shape[0] and grid == [t, h, w],
                                         'round_trip_bitwise': bool(torch.equal(merge_order(raster(tokens, t, h, w, merge), merge), tokens)),
                                         'picture_bitwise': bool(np.array_equal(back, np.asarray(exact, np.uint8)))})
    view.train()
    r = rows[0]; scan, clean = (encode(processor, [opened(p/r['images'][k]) for k in ('erased', 'context')], 'cuda') for p in (args.scan_pack, args.pack))
    with torch.no_grad(), view.disable_adapter():
        teacher = levels(view, *clean)
    pair_loss(levels(view, *scan), teacher).backward()
    adapter = {id(q) for q in view.sets.parameters()}
    graded = [(n, q) for n, q in view.named_parameters() if q.grad is not None]
    result['check3'] = {'parameters_with_gradients': sum(q.numel() for _, q in graded), 'outside_the_adapters': [n for n, q in graded if id(q) not in adapter],
                        'nonzero_gradient_tensors': [n for n, q in graded if bool(q.grad.abs().sum() > 0)],
                        'tower_parameters_trainable': sum(q.numel() for q in view.tower.parameters() if q.requires_grad)}
    result['passed'] = {'check1': all(c['bitwise'] for c in result['check1']),
                        'check2': all(c['rows_x_cols_is_tokens'] and c['round_trip_bitwise'] and c['picture_bitwise'] for c in result['check2']),
                        'check3': not result['check3']['outside_the_adapters'] and result['check3']['tower_parameters_trainable'] == 0}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({'parameters': result['parameters'], 'tower_parameters_trainable': result['check3']['tower_parameters_trainable'], 'passed': result['passed']}), flush=True)
    return result


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    for key in ('model', 'pack', 'scan-pack', 'output'): c.add_argument('--'+key, type=Path, required=True)
    c.add_argument('--regions', type=int, default=20)
    args = p.parse_args(argv)
    result = {'check': checks}[args.command](args)
    raise SystemExit(0 if all(result['passed'].values()) else 'a check failed')


if __name__ == '__main__':
    main()
