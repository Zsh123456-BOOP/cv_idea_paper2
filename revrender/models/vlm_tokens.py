"""Reading a line with a vision-language model whose visual tokens can be hidden from the text: the training-free token masking.

A Qwen3-VL image is a grid of merged tokens, each the area of ``merge x merge`` patches (32 x 32 pixels of the resized image) in
row-major order.  ``cell_shares`` gives the share of each token's area a pixel mask covers; the tokens over a threshold are hidden:
no later position -- the other visual tokens, the prompt, the reading being written -- may attend to them, and every token keeps the
position it had (M-RoPE), as if the visual tokens had been dropped from the sequence and the gap left in place.  A soft mask adds a
negative bias to the attention logits of the hidden keys instead of removing them.

``Reader.read`` runs a greedy decode over a batch of (image, words, hidden tokens) with its own loop: the mask of a request is an
additive 4D attention mask, the position ids are the model's own (``get_rope_index``), and a decode step carries the same bias over the
cache.  With nothing hidden it reproduces ``generate`` token for token (tests/test_vlm_tokens.py, on a model).  Only Qwen3-VL is
supported: its vision tower's own attention still mixes a hidden token's patches into its neighbours before the text sees any of
them, so hiding tokens alone does not erase what was under them (blank the pixels as well to do that).
"""
import numpy as np
from PIL import Image

NEG = -1e9                 # an additive mask value that leaves no probability (softmax of the rest is exact in bf16)


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


def install_students(inner, adapters, off=False):
    """Load the vision-tower adapters ``{name: folder}`` into ``inner.visual`` (``inner``: the Qwen3VLModel) and route its image
    features through them (``student_features``); the tower is pinned in eval (no dropout, whatever ``train()`` later says) and nothing
    of it learns.  ``off``: the flagged images go the adapters' way with the adapters disabled (the wiring and trajectory checks)."""
    import types
    import torch
    from peft import PeftModel
    names = list(adapters)
    view = PeftModel.from_pretrained(inner.visual, str(adapters[names[0]]), adapter_name=names[0])
    for name in names[1:]:
        view.load_adapter(str(adapters[name]), adapter_name=name)
    for q in view.parameters():
        q.requires_grad_(False)
    inner.visual.train = types.MethodType(lambda self, mode=True: torch.nn.Module.train(self, False), inner.visual)
    inner.visual.eval()
    students = Students(view, names)
    inner.get_image_features = student_features(inner.get_image_features, students, off)
    return students


def student_features(features, students, off):
    """The model's image features with, for each image ``students.flags`` names an adapter for, that adapter's, and the frozen tower's
    (the adapters disabled) for the others: the merged tokens and the deepstack levels alike.  The tower learns nothing, so no graph is
    kept.  ``off``: the flagged images go the same way with the adapters disabled too (it must read, or learn, as the plain model)."""
    import contextlib
    import torch
    view = students.view

    def chosen(pixel_values, image_grid_thw=None):
        with torch.no_grad():
            with view.disable_adapter():
                embeds, deep = features(pixel_values, image_grid_thw)
            flags = students.flags or [None]*len(embeds)
            if len(flags) != len(embeds):
                raise ValueError(f'{len(flags)} student flags for {len(embeds)} images')
            used = [n for n in students.names if n in flags]
            if not used:
                return embeds, deep
            sizes = [e.shape[0] for e in embeds]
            picked, levels = list(embeds), [list(d.split(sizes)) for d in deep]
            for name in used:
                if not off:
                    view.set_adapter(name, inference_mode=True)
                with view.disable_adapter() if off else contextlib.nullcontext():
                    s_embeds, s_deep = features(pixel_values, image_grid_thw)
                for i, f in enumerate(flags):
                    if f == name:
                        picked[i] = s_embeds[i]
                        for level, d in zip(levels, s_deep):
                            level[i] = d.split(sizes)[i]
                students.encoded[name] += flags.count(name)
            return tuple(picked), [torch.cat(level) for level in levels]
    return chosen


class Reader:
    """A Qwen3-VL model (with a LoRA adapter, if given) and its processor on one device, reading batches of lines with hidden visual tokens."""

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
        self.students = None
        if student:                                                      # a vision-tower adapter of ``train_clean_view``: see ``install_students``
            self.students = install_students(self.model.model, {'default': student}, student_off)
        self.merge = self.model.config.vision_config.spatial_merge_size
        self.image_token = self.model.config.image_token_id
        eos = self.model.generation_config.eos_token_id
        self.stop = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def grid(self, image):
        """(rows, cols) of merged tokens the processor makes of an image."""
        thw = self.processor.image_processor(images=[image], return_tensors='pt')['image_grid_thw'][0]
        return int(thw[1])//self.merge, int(thw[2])//self.merge

    def read(self, jobs, system, max_new_tokens=256, maps=None, loop_limit=0, students=None):
        """Greedy readings of (image, words, hidden, soft) jobs: ``image`` one image or several, ``hidden`` the bool flags of the image's
        tokens (or None), ``soft`` 0 to remove the hidden tokens from every attention or a positive bias subtracted from their attention
        logits; ``maps`` the layer map of each job's first image (a reader made with ``layer_map``).  ``loop_limit``: a reading that
        ends in one block written that many times (``looped``) stops there and keeps the block once (finish ``loop``; 0: never).  A list
        of dicts (text, tokens, confidence -- the geometric mean of the written tokens' probabilities --, finish).  ``students``: per job,
        which of its images the student sees (a reader made with ``student``)."""
        torch = self.torch
        if students is not None and self.students is None:
            raise ValueError('student images without a student')
        if self.students is not None:
            self.students.flags = [('default' if f else None) for job_flags in students for f in job_flags] if students is not None else None
        shown = lambda image: [{'type': 'image', 'image': one} for one in (image if isinstance(image, (list, tuple)) else [image])]
        conversations = [[{'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                          {'role': 'user', 'content': [*shown(image), {'type': 'text', 'text': words}]}] for image, words, _, _ in jobs]
        batch = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt', padding=True)
        ids, valid = batch['input_ids'], batch['attention_mask'].bool()
        blocked = blocked_keys(ids.numpy(), valid.numpy(), self.image_token, [job[2] for job in jobs])
        bias = key_bias(valid.numpy(), blocked, [job[3] for job in jobs])
        dtype = self.model.dtype
        mask = torch.from_numpy(prefill_mask(bias)).to(self.device, dtype)
        ids, valid = ids.to(self.device), valid.to(self.device)
        positions, _ = self.model.model.get_rope_index(ids, batch['image_grid_thw'].to(self.device), None, attention_mask=valid.long())
        several = lambda image: list(image) if isinstance(image, (list, tuple)) else [image]
        pixels = batch['pixel_values'] if maps is None else with_layer_maps(
            batch['pixel_values'], [m for job, layer in zip(jobs, maps) for m in layer_maps(several(job[0]), layer)], self.processor.image_processor)
        pixel_values, grid = pixels.to(self.device, dtype), batch['image_grid_thw'].to(self.device)
        length = ids.shape[1]
        from transformers import DynamicCache
        cache = DynamicCache()
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=positions, pixel_values=pixel_values, image_grid_thw=grid, past_key_values=cache,
                             use_cache=True, cache_position=torch.arange(length, device=self.device), logits_to_keep=1)
            keys = torch.from_numpy(bias).to(self.device, dtype)
            last = positions.amax(dim=(0, 2))                                  # the last position each row used
            written, done, loops, logged = [[] for _ in jobs], [False]*len(jobs), set(), [[] for _ in jobs]
            for step in range(max_new_tokens):
                chances = out.logits[:, -1].float().log_softmax(-1); token = chances.argmax(-1); chance = chances.gather(1, token[:, None])[:, 0].tolist()
                for b, t in enumerate(token.tolist()):
                    if not done[b]:
                        written[b].append(t); logged[b].append(chance[b])
                        done[b] = t in self.stop
                        cut = looped(written[b], loop_limit) if loop_limit and not done[b] else 0
                        if cut:
                            del written[b][-cut:], logged[b][-cut:]
                            done[b] = True
                            loops.add(b)
                if all(done) or step == max_new_tokens-1:
                    break
                keys = torch.cat([keys, torch.zeros(len(jobs), 1, device=self.device, dtype=dtype)], dim=1)
                position = (last+1+step).view(1, -1, 1).expand(3, -1, 1)
                out = self.model(input_ids=token.view(-1, 1), attention_mask=keys.view(len(jobs), 1, 1, -1), position_ids=position, past_key_values=cache,
                                 use_cache=True, cache_position=torch.tensor([length+step], device=self.device))
        return [{'text': self.processor.tokenizer.decode([t for t in w if t not in self.stop], skip_special_tokens=True), 'tokens': len(w),
                 'confidence': float(np.exp(np.mean(logged[b]))) if logged[b] else 0.,
                 'finish': 'loop' if b in loops else 'stop' if w and w[-1] in self.stop else 'length'} for b, w in enumerate(written)]
