"""Gemma 4 E4B (plan of 2026-11-21, 换模型验证插件, revisions seven and eight) in the new env: what differs from the Qwen readers.

- The chat: the local release files (Hugging Face commit 292a7e27; tokenizer.json sha256 cc8d3a0c...).  Its template takes a system
  turn's content as a string alone (a list of parts is printed as the list), so ``Processor.apply_chat_template`` gives the system
  turn its text as a string; the generation prompt is ``<|turn>model\\n``; thinking stays off (no ``<|think|>``), so nothing is
  prefilled.  Images go to the processor nested per conversation (``Processor.__call__`` groups a flat list by each text's
  ``<|image|>`` count); each image is resized to at most 280 soft tokens (the processor's default, the card's).
- The language model: 1-D positions counted over the unpadded tokens, the padding mask (2-D) -- its attention is causal (E4B has no
  bidirectional image blocks), so ``Reader.run`` hands it the mask and the positions and lets it build its own masks; the decode is
  ``vlm_tokens.greedy``.  Hiding tokens and layer maps are not supported.
- The tower: the vision model and ``embed_vision`` (``tower``); the features the language model takes are ``embed_vision``'s, padding
  patches stripped.  Its linear layers are wrapped (``Gemma4ClippableLinear``), so a low-rank adapter goes on their ``.linear``
  (``VISION_TARGETS``).
- The audio tower and its embedder stay off the card (``on_card``, revision eight: compute only); a hook counts the audio tower's calls,
  which must stay 0.
"""
import json
from pathlib import Path

AUDIO = ('audio_tower', 'embed_audio')
VISION_TARGETS = r'.*encoder\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.linear'
ANSWER_HEADER, TURN = '<|turn>model\n', '<|turn>'


def is_gemma(model_dir):
    path = Path(model_dir)/'config.json'
    return path.exists() and json.loads(path.read_text()).get('model_type') == 'gemma4'


def processing(model_dir):
    """What a run records of how Gemma sees and is spoken to."""
    image = json.loads((Path(model_dir)/'processor_config.json').read_text())['image_processor']
    return {'max_soft_tokens': image['max_soft_tokens'], 'patch': image['patch_size'], 'pooling': image['pooling_kernel_size'],
            'files': 'the local release (Hugging Face 292a7e27): tokenizer.json, tokenizer_config.json, chat_template.jinja',
            'system_turn': 'its text as a string (the release template prints a list of parts as it is)', 'thinking': 'off: no <|think|>, nothing prefilled',
            'language_attention': 'sdpa; causal, the padding mask and positions over the unpadded tokens', 'audio': 'off the card, its calls counted (0)'}


class Processor:
    """Gemma's processor as the Qwen code calls a processor: system turns as strings, images grouped per text."""

    def __init__(self, model_dir):
        from transformers import AutoProcessor
        self.inner = AutoProcessor.from_pretrained(str(model_dir))
        self.tokenizer, self.image_processor = self.inner.tokenizer, self.inner.image_processor
        self.image_token = self.inner.image_token
        self.answer_header, self.turn = ANSWER_HEADER, TURN

    @staticmethod
    def plain(conversation):
        return [{**m, 'content': ''.join(p['text'] for p in m['content'] if p['type'] == 'text')} if m['role'] == 'system' and isinstance(m['content'], list) else m
                for m in conversation]

    def apply_chat_template(self, conversations, **kwargs):
        batch = bool(conversations) and isinstance(conversations[0], list)
        fixed = [self.plain(c) for c in conversations] if batch else self.plain(conversations)
        return self.inner.apply_chat_template(fixed, **kwargs)

    def __call__(self, text, images, return_tensors='pt', padding=True):
        counts = [t.count(self.image_token) for t in text]
        if sum(counts) != len(images):
            raise ValueError(f'{sum(counts)} image places for {len(images)} images')
        nested, start = [], 0
        for n in counts:
            nested.append(list(images[start:start+n])); start += n
        return self.inner(text=text, images=nested, return_tensors=return_tensors, padding=padding)


def gemma_inner(model):
    """The Gemma4Model inside a model (peft-wrapped or not)."""
    return next(m for m in model.modules() if hasattr(m, 'audio_tower') and hasattr(m, 'embed_vision'))


def on_card(model, device='cuda'):
    """Everything but the audio tower and its embedder onto ``device`` (they stay where they are); a forward hook on the audio tower
    counts its calls.  Returns the count (a list whose length is the calls)."""
    inner = gemma_inner(model); calls = []

    def place(module):
        if module is inner:
            for name, child in module.named_children():
                if name not in AUDIO: child.to(device)
            return
        if any(m is inner for m in module.modules()):
            for child in module.children(): place(child)
            for key, q in list(module._parameters.items()):
                if q is not None: module._parameters[key] = type(q)(q.data.to(device), q.requires_grad)
            for key, b in list(module._buffers.items()):
                if b is not None: module._buffers[key] = b.to(device)
        else:
            module.to(device)
    place(model)
    if inner.audio_tower is not None:
        inner.audio_tower.register_forward_pre_hook(lambda *a: calls.append(1))
    return calls


def tower(model_dir):
    """The tower alone -- the vision model and ``embed_vision``, bf16 -- as a module whose forward gives what the language model takes:
    an output whose ``pooler_output`` is the soft tokens of every image, padding stripped ([tokens, width]), no deepstack levels."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel
    config = AutoConfig.from_pretrained(str(model_dir)); vision = config.vision_config; vision._attn_implementation = 'sdpa'

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = Gemma4VisionModel._from_config(vision, dtype=torch.bfloat16)
            self.embed_vision = Gemma4MultimodalEmbedder(vision, config.text_config).to(torch.bfloat16)
            self.config = vision; self.config.out_hidden_size = config.text_config.hidden_size

        def forward(self, pixel_values, grid_thw=None):
            hidden = self.vision_tower(pixel_values=pixel_values, pixel_position_ids=grid_thw).last_hidden_state
            return BaseModelOutputWithPooling(last_hidden_state=hidden, pooler_output=self.embed_vision(inputs_embeds=hidden))
    module = Tower(); state = {}
    for path in sorted(Path(model_dir).glob('*.safetensors')):
        with safe_open(str(path), 'pt') as f:
            state.update({k[len('model.'):]: f.get_tensor(k) for k in f.keys() if k.startswith(('model.vision_tower.', 'model.embed_vision.'))})
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f'tower weights: missing {missing[:3]}, unexpected {unexpected[:3]}')
    return module


def view_of(inner):
    """The whole model's own tower parts as a ``tower`` module (shared, not copied): where a tower adapter is put when reading."""
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    class View(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower, self.embed_vision = inner.vision_tower, inner.embed_vision
            self.config = inner.config.vision_config

        def forward(self, pixel_values, grid_thw=None):
            hidden = self.vision_tower(pixel_values=pixel_values, pixel_position_ids=grid_thw).last_hidden_state
            return BaseModelOutputWithPooling(last_hidden_state=hidden, pooler_output=self.embed_vision(inputs_embeds=hidden))
    return View()


def from_pretrained(model_dir, adapter=None):
    """The whole model (bf16, sdpa), a ``train_reader`` adapter merged when given; still on the CPU."""
    import torch
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.bfloat16, attn_implementation='sdpa')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    return model


class Reader:
    """Gemma 4 E4B (with a ``train_reader`` adapter merged; with a ``train_clean_view`` tower adapter as the student) reading batches as
    ``vlm_tokens.Reader`` does: the same words and decode."""

    def __init__(self, model_dir, device='cuda', adapter=None, layer_map=None, student=None, student_off=False):
        import torch
        from .vlm_tokens import answer_prefill, install_students
        if layer_map:
            raise ValueError('layer maps are not supported on Gemma')
        self.torch, self.device = torch, device
        self.processor = Processor(model_dir); self.processor.tokenizer.padding_side = 'left'
        self.model = from_pretrained(model_dir, adapter)
        self.audio_calls = on_card(self.model, device); self.model.eval()
        inner = gemma_inner(self.model)
        self.students = install_students(inner, {'default': student}, student_off, tower=view_of(inner)) if student else None
        self.prefill, self.merge = answer_prefill(self.processor), 1
        self.image_token = self.model.config.image_token_id
        eos = self.model.generation_config.eos_token_id
        self.stop = set(eos if isinstance(eos, (list, tuple)) else [eos])

    from .vlm_tokens import Reader as _Qwen
    read = _Qwen.read

    def run(self, jobs, batch, ids, valid, bias, mask, maps, max_new_tokens, loop_limit):
        from transformers import DynamicCache
        from .vlm_tokens import greedy
        torch = self.torch
        if bool(bias[valid.cpu().numpy()].any()):
            raise ValueError('hidden tokens are not supported on Gemma')
        valid = valid.long(); positions = (valid.cumsum(-1)-1).clamp_min(0)
        cache = DynamicCache(config=self.model.config)
        extra = {k: batch[k].to(self.device) for k in ('image_position_ids', 'mm_token_type_ids') if k in batch}
        with torch.inference_mode():
            first = self.model(input_ids=ids, attention_mask=valid, position_ids=positions, pixel_values=batch['pixel_values'].to(self.device, self.model.dtype),
                               past_key_values=cache, use_cache=True, logits_to_keep=1, **extra)
            last = positions.amax(-1); seen = [valid]

            def advance(token, step):
                seen[0] = torch.cat([seen[0], torch.ones(len(jobs), 1, device=self.device, dtype=seen[0].dtype)], dim=1)
                return self.model(input_ids=token.view(-1, 1), attention_mask=seen[0], position_ids=(last+1+step).view(-1, 1), past_key_values=cache, use_cache=True)
            return greedy(first, advance, self.stop, max_new_tokens, loop_limit, self.processor.tokenizer)
