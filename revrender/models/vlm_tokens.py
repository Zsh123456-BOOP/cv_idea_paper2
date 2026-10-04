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


class Reader:
    """A Qwen3-VL model (with a LoRA adapter, if given) and its processor on one device, reading batches of lines with hidden visual tokens."""

    def __init__(self, model_dir, device='cuda', adapter=None):
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
        patch_embed_as_linear(self.model.model.visual.patch_embed)
        self.merge = self.model.config.vision_config.spatial_merge_size
        self.image_token = self.model.config.image_token_id
        eos = self.model.generation_config.eos_token_id
        self.stop = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def grid(self, image):
        """(rows, cols) of merged tokens the processor makes of an image."""
        thw = self.processor.image_processor(images=[image], return_tensors='pt')['image_grid_thw'][0]
        return int(thw[1])//self.merge, int(thw[2])//self.merge

    def read(self, jobs, system, max_new_tokens=256):
        """Greedy readings of (image, words, hidden, soft) jobs: ``hidden`` the bool flags of the image's tokens (or None), ``soft`` 0 to
        remove the hidden tokens from every attention or a positive bias subtracted from their attention logits.  A list of dicts
        (text, tokens, finish)."""
        torch = self.torch
        conversations = [[{'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                          {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': words}]}] for image, words, _, _ in jobs]
        batch = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt', padding=True)
        ids, valid = batch['input_ids'], batch['attention_mask'].bool()
        blocked = blocked_keys(ids.numpy(), valid.numpy(), self.image_token, [job[2] for job in jobs])
        bias = key_bias(valid.numpy(), blocked, [job[3] for job in jobs])
        dtype = self.model.dtype
        mask = torch.from_numpy(prefill_mask(bias)).to(self.device, dtype)
        ids, valid = ids.to(self.device), valid.to(self.device)
        positions, _ = self.model.model.get_rope_index(ids, batch['image_grid_thw'].to(self.device), None, attention_mask=valid.long())
        pixel_values, grid = batch['pixel_values'].to(self.device, dtype), batch['image_grid_thw'].to(self.device)
        length = ids.shape[1]
        from transformers import DynamicCache
        cache = DynamicCache()
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=positions, pixel_values=pixel_values, image_grid_thw=grid, past_key_values=cache,
                             use_cache=True, cache_position=torch.arange(length, device=self.device), logits_to_keep=1)
            keys = torch.from_numpy(bias).to(self.device, dtype)
            last = positions.amax(dim=(0, 2))                                  # the last position each row used
            written, done = [[] for _ in jobs], [False]*len(jobs)
            for step in range(max_new_tokens):
                token = out.logits[:, -1].float().argmax(-1)
                for b, t in enumerate(token.tolist()):
                    if not done[b]:
                        written[b].append(t)
                        done[b] = t in self.stop
                if all(done) or step == max_new_tokens-1:
                    break
                keys = torch.cat([keys, torch.zeros(len(jobs), 1, device=self.device, dtype=dtype)], dim=1)
                position = (last+1+step).view(1, -1, 1).expand(3, -1, 1)
                out = self.model(input_ids=token.view(-1, 1), attention_mask=keys.view(len(jobs), 1, 1, -1), position_ids=position, past_key_values=cache,
                                 use_cache=True, cache_position=torch.tensor([length+step], device=self.device))
        return [{'text': self.processor.tokenizer.decode([t for t in w if t not in self.stop], skip_special_tokens=True), 'tokens': len(w),
                 'finish': 'stop' if w and w[-1] in self.stop else 'length'} for w in written]
