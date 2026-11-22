"""Reading a line with a vision-language model whose visual tokens can be hidden from the text: the training-free token masking.

A Qwen3-VL image is a grid of merged tokens, each the area of ``merge x merge`` patches (32 x 32 pixels of the resized image) in
row-major order.  ``cell_shares`` gives the share of each token's area a pixel mask covers; the tokens over a threshold are hidden:
no later position -- the other visual tokens, the prompt, the reading being written -- may attend to them, and every token keeps the
position it had (M-RoPE), as if the visual tokens had been dropped from the sequence and the gap left in place.  A soft mask adds a
negative bias to the attention logits of the hidden keys instead of removing them.

``Reader.read`` runs a greedy decode over a batch of (image, words, hidden tokens) with its own loop: the mask of a request is an
additive 4D attention mask, the position ids are the model's own (``get_rope_index``), and a decode step carries the same bias over the
cache.  With nothing hidden it reproduces ``generate`` token for token (tests/test_vlm_tokens.py, on a model).  Qwen3-VL and Qwen3.5
(plan of 2026-11-21, 换模型验证插件) are supported, under transformers 4.57 and 5.x alike (``tower_levels``, ``rope_index``,
``language_mask``): Qwen3.5 has no deepstack levels, half its language layers are linear attention (they take the padding mask, not
the 4D one, so they see hidden tokens -- hide tokens on Qwen3-VL only), and its template opens a thinking block, closed empty before
the answer (``answer_prefill``: the 27B's own words for not thinking, and what its template writes before a taught answer).  The
vision tower's own attention still mixes a hidden token's patches into its neighbours before the text sees any of them, so hiding
tokens alone does not erase what was under them (blank the pixels as well to do that).
"""
import numpy as np
from PIL import Image

NEG = -1e9                 # an additive mask value that leaves no probability (softmax of the rest is exact in bf16)
THINK_OPEN, THINK_CLOSED = '<think>\n', '\n</think>\n\n'


def answer_prefill(processor):
    """What goes after the generation prompt before the answer: the closing of the empty thinking block when the template opens one
    (Qwen3.5), else nothing."""
    probe = [{'role': 'user', 'content': [{'type': 'text', 'text': '.'}]}]
    return THINK_CLOSED if processor.apply_chat_template(probe, add_generation_prompt=True, tokenize=False).endswith(THINK_OPEN) else ''


def tower_levels(out):
    """(merged, [deepstack levels]) of a vision tower's output or of ``get_image_features``'s, whatever the library version: 4.57
    gives a pair (the merged tokens -- one tensor, or a tuple per image -- and the deepstack levels, each one tensor), 5.x an output
    whose ``pooler_output`` is the merged tokens and ``deepstack_features`` the levels (each one tensor, or a tuple per image; none for
    Qwen3.5)."""
    if isinstance(out, (tuple, list)):
        return out[0], list(out[1] or [])
    return out.pooler_output, list(getattr(out, 'deepstack_features', None) or [])


def image_levels(out):
    """Per image, [merged, deepstack 1..]: the tokens of ``get_image_features``'s output, one image at a time."""
    import torch
    embeds, deep = tower_levels(out)
    sizes = [e.shape[0] for e in embeds]
    deep = [level if isinstance(level, (tuple, list)) else level.split(sizes) for level in deep]
    return [[e, *(level[i] for level in deep)] for i, e in enumerate(embeds)] if torch.is_tensor(embeds[0]) else []


def rope_index(inner, ids, grid, valid, image_token):
    """The model's M-RoPE positions [3, B, L] of a batch of images and text: 5.x asks which tokens are image tokens, 4.57 does not."""
    import inspect
    if 'mm_token_type_ids' in inspect.signature(inner.get_rope_index).parameters:
        return inner.get_rope_index(ids, (ids == image_token).int(), grid, None, attention_mask=valid.long())[0]
    return inner.get_rope_index(ids, grid, None, attention_mask=valid.long())[0]


def language_mask(config, mask, valid):
    """The attention mask the language model takes: the additive 4D mask, and for a model with linear-attention layers (Qwen3.5) the
    padding mask beside it for those layers (None in a decode step: a written token is never padding)."""
    if 'linear_attention' not in (getattr(config.get_text_config(), 'layer_types', None) or ()):
        return mask
    return {'full_attention': mask, 'linear_attention': valid.long() if valid is not None and not bool(valid.all()) else None}


def cell_shares(mask, rows, cols):
    """The share of each of ``rows x cols`` token cells a boolean pixel mask covers (area average over the whole image), row-major."""
    image = Image.fromarray(np.asarray(mask, np.float32), mode='F').resize((cols, rows), Image.BOX)
    return np.asarray(image, np.float32).reshape(-1)


def hidden_tokens(shares, threshold):
    """Which tokens are hidden: a cell with at least ``threshold`` of its area under the mask (and some of it)."""
    return (shares >= threshold) & (shares > 0)


def blocked_keys(input_ids, valid, image_token, hidden):
    """[B, L] bool: the positions of hidden visual tokens.  ``hidden`` is a list, per row, of the bool flags of that row's image tokens
    in order (None for a row with nothing hidden)."""
    blocked = np.zeros(input_ids.shape, bool)
    for b, flags in enumerate(hidden):
        if flags is None or not np.any(flags):
            continue
        places = np.flatnonzero((input_ids[b] == image_token) & valid[b])
        if len(places) != len(flags):
            raise ValueError(f'row {b}: {len(places)} image tokens but {len(flags)} flags')
        blocked[b, places[np.asarray(flags, bool)]] = True
    return blocked


def key_bias(valid, blocked, soft):
    """[B, L] additive bias on every key: 0 for a position the text may attend to, ``NEG`` for padding and for the hidden tokens of a
    row whose ``soft`` is 0, ``-soft`` for the hidden tokens of a row whose ``soft`` is a positive bias."""
    soft = np.asarray(soft, np.float32).reshape(-1, 1)
    bias = np.where(blocked, np.where(soft > 0, -soft, NEG), 0.).astype(np.float32)
    bias[~valid] = NEG
    return bias


def prefill_mask(bias):
    """[B, 1, L, L] additive mask: causal, plus the key bias; every position attends to itself so no row is empty."""
    length = bias.shape[1]
    causal = np.triu(np.full((length, length), NEG, np.float32), 1)
    mask = np.maximum(causal[None, None] + bias[:, None, None, :], NEG)
    index = np.arange(length)
    mask[:, :, index, index] = 0.
    return mask


def patch_embed_as_linear(embed):
    """Run a Qwen3-VL patch embedding (a Conv3d whose stride is its kernel) as the matrix product it is.  torch 2.9 with cuDNN 9.10-9.14
    runs that Conv3d in bf16 about 1.4 seconds per image; the product gives the same numbers and runs in milliseconds."""
    import torch
    weight, bias = embed.proj.weight, embed.proj.bias

    def forward(hidden_states):
        return torch.nn.functional.linear(hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype), weight.reshape(weight.shape[0], -1), bias)
    embed.forward = forward


def patch_embed_with_layer_map(embed):
    """A Qwen3-VL patch embedding that also takes a layer map of the image -- a second picture of the same size saying, pixel by pixel,
    which ink is the strike -- beside the scan: ``[patches, 2 x width]``, the scan's patches then the map's, cut alike.  The map has a
    projection of its own, added to the scan's embedding before the tower mixes anything; it starts at zero, so the model reads as it
    did until the projection is taught (the zero-initialised extra input of Alpha-CLIP, CVPR 2024).  A patch holds strike and writing
    together, so the map has to arrive here, finer than a token.  Without the map's half the scan is embedded as ever
    (``patch_embed_as_linear``).  Returns the projection (float32; ``embed.layer_map``)."""
    import torch
    weight, bias = embed.proj.weight, embed.proj.bias
    width = weight[0].numel()
    embed.layer_map = torch.nn.Linear(width, weight.shape[0], bias=False).to(weight.device)
    torch.nn.init.zeros_(embed.layer_map.weight)

    def forward(hidden_states):
        scan = lambda patches: torch.nn.functional.linear(patches.reshape(-1, width).to(weight.dtype), weight.reshape(weight.shape[0], -1), bias)
        if hidden_states.dim() == 2 and hidden_states.shape[1] == 2*width:
            return scan(hidden_states[:, :width]) + embed.layer_map(hidden_states[:, width:].float()).to(weight.dtype)
        return scan(hidden_states)
    embed.forward = forward
    return embed.layer_map


def looped(tokens, repeats, shortest=12):
    """How many tokens to drop from the end of a decode caught in a loop: when the tokens end in one block written ``repeats`` times
    over at least ``shortest`` tokens, all but the block's first writing; 0 when they do not."""
    n = len(tokens)
    for period in range(1, n//repeats+1):
        if period*repeats >= shortest and tokens[n-period*repeats:] == tokens[n-period:]*repeats:
            return period*(repeats-1)
    return 0


def layer_maps(images, layer):
    """A layer map for each image of a request: the given one for the first (the scan), nothing for the others."""
    return [layer, *(Image.new('RGB', image.size) for image in images[1:])]


def with_layer_maps(pixel_values, maps, image_processor):
    """The scans' patches with their layer maps' beside them: the maps go through the same image processor, so they are cut alike, and
    its normalisation is taken off again -- a map is 0 where it says nothing and 1 on the strike, so a patch without strike adds
    nothing to its embedding whatever the projection has become."""
    import torch
    layers = image_processor(images=maps, return_tensors='pt')['pixel_values']
    if layers.shape != pixel_values.shape:
        raise ValueError(f'layer maps cut to {tuple(layers.shape)}, scans to {tuple(pixel_values.shape)}: a map must have the size of its scan')
    spread = lambda values: torch.tensor(values, dtype=layers.dtype).repeat_interleave(layers.shape[1]//len(values))     # a patch is laid out channel first
    layers = layers*spread(image_processor.image_std)+spread(image_processor.image_mean)
    return torch.cat([pixel_values, layers.to(pixel_values.dtype)], dim=1)


class Students:
    """The vision-tower adapters of ``train_clean_view`` installed in a Qwen3-VL (``install_students``): ``flags`` says, per image of
    the next batch in order, which adapter sees it (its name) or the frozen tower (None); ``encoded`` counts the images each adapter saw."""

    def __init__(self, view, names):
        self.view, self.names, self.flags, self.encoded = view, names, None, {n: 0 for n in names}


def install_students(inner, adapters, off=False, tower=None, attribute='get_image_features', route=None):
    """Load the vision-tower adapters ``{name: folder}`` into ``tower`` (default ``inner.visual``, ``inner`` the Qwen3VLModel) and
    route ``inner``'s image features (its ``attribute``) through them (``route``, default ``student_features``; InternVL:
    ``models.internvl.Reader``); the tower and its parts are pinned in eval (no dropout, whatever ``train()`` later says) and nothing
    of it learns.  ``off``: the flagged images go the adapters' way with the adapters disabled (the wiring and trajectory checks)."""
    import types
    import torch
    from .maa import student_view
    tower = inner.visual if tower is None else tower
    names = list(adapters)
    view = student_view(tower, adapters)                                # low-rank adapters, or MAA's (models.maa)
    for q in view.parameters():
        q.requires_grad_(False)
    for part in (tower, *tower.children()):
        part.train = types.MethodType(lambda self, mode=True: torch.nn.Module.train(self, False), part)
        part.eval()
    students = Students(view, names)
    setattr(inner, attribute, (route or student_features)(getattr(inner, attribute), students, off))
    return students


def picked(students, run, off, grad=False):
    """Per image, what ``run()`` gives per image (a list) with, for each image ``students.flags`` names an adapter for, that adapter
    switched on, and the adapters disabled for the others; no graph is kept.  ``off``: the flagged images the adapters' way with the
    adapters disabled too.  ``grad`` (B of 插件结构优化: the adapter learns from the reader's answer loss): the adapter's run keeps its
    graph, its adapter (the one in use, which learns) left as it is.  Returns (the frozen run's whole output, the choice per image, or
    None when no image is flagged)."""
    import contextlib
    import torch
    view = students.view; grad = grad and torch.is_grad_enabled()          # no graph for a run without gradients (validation)
    with torch.no_grad():
        with view.disable_adapter():
            whole, frozen = run()
        flags = students.flags or [None]*len(frozen)
        if len(flags) != len(frozen):
            raise ValueError(f'{len(flags)} student flags for {len(frozen)} images')
        used = [n for n in students.names if n in flags]
        if not used:
            return whole, None
        chosen = list(frozen)
        for name in used:
            if not off and not grad:
                view.set_adapter(name, inference_mode=True)
            with view.disable_adapter() if off else contextlib.nullcontext(), torch.enable_grad() if grad else contextlib.nullcontext():
                _, taught = run()
            for i, f in enumerate(flags):
                if f == name:
                    chosen[i] = taught[i]
            students.encoded[name] += flags.count(name)
        return whole, chosen


def student_features(features, students, off, grad=False):
    """The model's image features (Qwen) with each image through the adapter ``students.flags`` names for it (``picked``): the merged
    tokens and the deepstack levels alike."""
    import torch

    def chosen(pixel_values, image_grid_thw=None, **kwargs):
        def run():
            out = features(pixel_values, image_grid_thw, **kwargs)
            return out, image_levels(out)
        out, choice = picked(students, run, off, grad)
        if choice is None:
            return out
        embeds, deep = tuple(p[0] for p in choice), [tuple(p[n] for p in choice) for n in range(1, len(choice[0]))]
        if isinstance(out, (tuple, list)):                                  # 4.57: the levels each one tensor
            return embeds, [torch.cat(level) for level in deep]
        out.pooler_output = embeds
        if getattr(out, 'deepstack_features', None) is not None:
            out.deepstack_features = deep
        return out
    return chosen


def greedy(first, advance, stop, max_new_tokens, loop_limit, tokenizer, watch=()):
    """The greedy decode of a batch from the prefill's output ``first``: ``advance(token, step)`` runs one step of the written tokens
    and gives its output.  ``loop_limit``: a reading that ends in one block written that many times (``looped``) stops there and keeps
    the block once (finish ``loop``; 0: never).  A list of dicts (text, tokens, confidence -- the geometric mean of the written tokens'
    probabilities --, finish); ``watch``: token ids whose writings are counted (``thinking_tokens``; Gemma's thinking channel)."""
    n = first.logits.shape[0]; out = first
    written, done, loops, logged = [[] for _ in range(n)], [False]*n, set(), [[] for _ in range(n)]
    for step in range(max_new_tokens):
        chances = out.logits[:, -1].float().log_softmax(-1); token = chances.argmax(-1); chance = chances.gather(1, token[:, None])[:, 0].tolist()
        for b, t in enumerate(token.tolist()):
            if not done[b]:
                written[b].append(t); logged[b].append(chance[b])
                done[b] = t in stop
                cut = looped(written[b], loop_limit) if loop_limit and not done[b] else 0
                if cut:
                    del written[b][-cut:], logged[b][-cut:]
                    done[b] = True
                    loops.add(b)
        if all(done) or step == max_new_tokens-1:
            break
        out = advance(token, step)
    return [{'text': tokenizer.decode([t for t in w if t not in stop], skip_special_tokens=True), 'tokens': len(w),
             'confidence': float(np.exp(np.mean(logged[b]))) if logged[b] else 0.,
             'finish': 'loop' if b in loops else 'stop' if w and w[-1] in stop else 'length', **({'thinking_tokens': sum(t in watch for t in w)} if watch else {})}
            for b, w in enumerate(written)]


def processor_for(model_dir):
    """The processor of a model folder: transformers' own, or ``models.internvl.Processor``."""
    from .gemma import Processor as GemmaProcessor, is_gemma
    from .internvl import Processor as InternProcessor, is_internvl
    from .llava import Processor as LlavaProcessor, is_llava
    if is_internvl(model_dir):
        return InternProcessor(model_dir)
    if is_gemma(model_dir):
        return GemmaProcessor(model_dir)
    if is_llava(model_dir):
        return LlavaProcessor(model_dir)
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(model_dir))


def reader_for(model_dir, **kwargs):
    """The reader of a model folder: ``Reader`` (Qwen3-VL, Qwen3.5) or ``models.internvl.Reader``."""
    from .gemma import is_gemma
    from .internvl import is_internvl
    if is_internvl(model_dir):
        from .internvl import Reader as InternReader
        return InternReader(model_dir, **kwargs)
    if is_gemma(model_dir):
        from .gemma import Reader as GemmaReader
        return GemmaReader(model_dir, **kwargs)
    from .llava import is_llava
    if is_llava(model_dir):
        from .llava import Reader as LlavaReader
        return LlavaReader(model_dir, **kwargs)
    return Reader(model_dir, **kwargs)


class Reader:
    """A Qwen3-VL or Qwen3.5 model (with a LoRA adapter, if given) and its processor on one device, reading batches of lines with hidden
    visual tokens."""

    def __init__(self, model_dir, device='cuda', adapter=None, layer_map=None, student=None, student_off=False):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.processor.tokenizer.padding_side = 'left'
        self.model = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.bfloat16, attn_implementation='sdpa')
        if adapter:                                                      # a LoRA adapter of ``train_reader``, merged into the weights
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter)).merge_and_unload()
        self.model = self.model.to(device).eval()
        if layer_map:                                                    # the layer-map projection of ``train_reader --layer-map``
            patch_embed_with_layer_map(self.model.model.visual.patch_embed).load_state_dict(torch.load(str(layer_map), map_location=device))
        else:
            patch_embed_as_linear(self.model.model.visual.patch_embed)
        self.students = self.mix = None                                  # mix: models.feature_mix.install (or models.transplant.install)
        if student:                                                      # a vision-tower adapter of ``train_clean_view``: see ``install_students``
            self.students = install_students(self.model.model, {'default': student}, student_off)
        self.prefill = answer_prefill(self.processor)
        self.merge = self.model.config.vision_config.spatial_merge_size
        self.image_token = self.model.config.image_token_id
        eos = self.model.generation_config.eos_token_id
        self.stop = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def grid(self, image):
        """(rows, cols) of merged tokens the processor makes of an image."""
        thw = self.processor.image_processor(images=[image], return_tensors='pt')['image_grid_thw'][0]
        return int(thw[1])//self.merge, int(thw[2])//self.merge

    def read(self, jobs, system, max_new_tokens=256, maps=None, loop_limit=0, students=None, mix_images=None):
        """Greedy readings of (image, words, hidden, soft) jobs: ``image`` one image or several, ``hidden`` the bool flags of the image's
        tokens (or None), ``soft`` 0 to remove the hidden tokens from every attention or a positive bias subtracted from their attention
        logits; ``maps`` the layer map of each job's first image (a reader made with ``layer_map``).  ``loop_limit``: a reading that
        ends in one block written that many times (``looped``) stops there and keeps the block once (finish ``loop``; 0: never).  A list
        of dicts (text, tokens, confidence -- the geometric mean of the written tokens' probabilities --, finish).  ``students``: per job,
        which of its images the student sees (a reader made with ``student``); with a mix (``models.feature_mix``), which are mixed, and
        ``mix_images`` per job the mix pack's images."""
        torch = self.torch
        if students is not None and self.students is None and self.mix is None:
            raise ValueError('student images without a student')
        if self.mix is not None:
            self.mix.flags = [bool(f) for job_flags in students for f in job_flags]
            mixed = self.processor.image_processor(images=[one for images in mix_images for one in images], return_tensors='pt')
            self.mix.companion = (mixed['pixel_values'], mixed['image_grid_thw'])
        elif self.students is not None:
            self.students.flags = [('default' if f else None) for job_flags in students for f in job_flags] if students is not None else None
        shown = lambda image: [{'type': 'image', 'image': one} for one in (image if isinstance(image, (list, tuple)) else [image])]
        conversations = [[{'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                          {'role': 'user', 'content': [*shown(image), {'type': 'text', 'text': words}]}] for image, words, _, _ in jobs]
        texts = [self.processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False)+self.prefill for c in conversations]
        images = [one['image'] for c in conversations for one in c[1]['content'] if one['type'] == 'image']
        batch = self.processor(text=texts, images=images, return_tensors='pt', padding=True)
        ids, valid = batch['input_ids'], batch['attention_mask'].bool()
        blocked = blocked_keys(ids.numpy(), valid.numpy(), self.image_token, [job[2] for job in jobs])
        bias = key_bias(valid.numpy(), blocked, [job[3] for job in jobs])
        mask = torch.from_numpy(prefill_mask(bias)).to(self.device, self.model.dtype)
        ids, valid = ids.to(self.device), valid.to(self.device)
        return self.run(jobs, batch, ids, valid, bias, mask, maps, max_new_tokens, loop_limit)

    def run(self, jobs, batch, ids, valid, bias, mask, maps, max_new_tokens, loop_limit):
        """The prefill and the decode of a prepared batch (the model's own part of ``read``)."""
        torch, dtype = self.torch, self.model.dtype
        positions = rope_index(self.model.model, ids, batch['image_grid_thw'].to(self.device), valid, self.image_token)
        several = lambda image: list(image) if isinstance(image, (list, tuple)) else [image]
        pixels = batch['pixel_values'] if maps is None else with_layer_maps(
            batch['pixel_values'], [m for job, layer in zip(jobs, maps) for m in layer_maps(several(job[0]), layer)], self.processor.image_processor)
        pixel_values, grid = pixels.to(self.device, dtype), batch['image_grid_thw'].to(self.device)
        from transformers import DynamicCache
        config = self.model.config; cache = DynamicCache(config=config)
        with torch.inference_mode():
            first = self.model(input_ids=ids, attention_mask=language_mask(config, mask, valid), position_ids=positions, pixel_values=pixel_values,
                               image_grid_thw=grid, past_key_values=cache, use_cache=True, logits_to_keep=1)
            last = positions.amax(dim=(0, 2))                                  # the last position each row used
            keys = [torch.from_numpy(bias).to(self.device, dtype)]

            def advance(token, step):
                keys[0] = torch.cat([keys[0], torch.zeros(len(jobs), 1, device=self.device, dtype=dtype)], dim=1)
                position = (last+1+step).view(1, -1, 1).expand(3, -1, 1)
                return self.model(input_ids=token.view(-1, 1), attention_mask=language_mask(config, keys[0].view(len(jobs), 1, 1, -1), None), position_ids=position,
                                  past_key_values=cache, use_cache=True)
            return greedy(first, advance, self.stop, max_new_tokens, loop_limit, self.processor.tokenizer)
