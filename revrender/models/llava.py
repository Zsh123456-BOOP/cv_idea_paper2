"""LLaVA-1.6-Mistral-7B (llava-hf/llava-v1.6-mistral-7b-hf; plan of 2026-11-21, 换模型验证插件, revision ten) in the hme env: what differs
from the Qwen readers.

- The chat: the model's template, its system turn as it writes one (``<<SYS>>\\n...\\n<</SYS>>\\n\\n`` before ``[INST]``), images first
  (``<image>\\n`` each), the words, `` [/INST]``; no generation prompt.  An assistant turn ends in the template with the literal
  characters ``<\\s> `` (a slip for the end token), so the template is used with ``</s>`` there (``END_FIX``): the reader is taught to
  end its answer with the end token, as Mistral's turns end, and reading stops on it.  Every text is tokenized by the processor with
  its special tokens (the start token ``<s>``), the taught chats as the read ones (``Processor``), so the taught prefix is the read
  prompt token for token; where the answer starts is the length of its prompt alone (``prompt_lengths``).
- The images: the processor's own anyres cutting (a 336 base view and up to four 336 tiles, 576 tokens each, unpadded, a newline token
  per row).  The features the language model takes: the projector's output of layer -2 without its CLS token, packed per image
  (``get_image_features``); a tower adapter is aligned on their image tokens -- the newline tokens, a constant alike on both sides,
  left out (revision eleven: the loss counts image tokens, as on the other models; ``tower``) -- and routed per image when reading,
  the newlines in place (``image_route``).
- The tower adapter: low-rank on the CLIP layers whose output reaches layer -2's features (0-22; ``ADAPTER_TARGETS``): the last layer's
  output is not used, so an adapter there would never learn.
- The language model: 1-D positions over the unpadded tokens and the padding mask, as Gemma (``Reader.run``).  Hiding tokens and layer
  maps are not supported.
"""
import json
from pathlib import Path

ADAPTER_TARGETS = r'.*encoder\.layers\.(\d|1\d|2[0-2])\.(self_attn\.(q|k|v|out)_proj|mlp\.(fc1|fc2))'
END_SLIP, END_FIX = "'<\\s> '", "'</s>'"                                # the template's literal ``<\s> `` and what is used instead


def is_llava(model_dir):
    path = Path(model_dir)/'config.json'
    return path.exists() and json.loads(path.read_text()).get('model_type') == 'llava_next'


def processing(model_dir):
    image = json.loads((Path(model_dir)/'preprocessor_config.json').read_text())
    return {'cutting': image.get('aspect_ratio_setting', 'anyres'), 'grid_pinpoints': image['image_grid_pinpoints'], 'tile': 336, 'tokens_per_tile': 576,
            'features': 'layer -2, CLS dropped, projector, packed per image (unpadded, a newline token per row; the newlines left out of the adapter loss)',
            'system_turn': "the template's own: <<SYS>>\\n...\\n<</SYS>>\\n\\n before [INST]",
            'assistant_end': "the template's literal '<\\s> ' replaced by the end token '</s>' (taught and read alike)",
            'adapter_layers': "CLIP layers 0-22 (the last layer's output is not used)", 'language_attention': 'sdpa; causal, the padding mask and positions over the unpadded tokens'}


class Processor:
    """LLaVA-NeXT's processor as the Qwen code calls a processor: the template with its end mended, every text tokenized with the
    processor's special tokens (so taught and read chats are tokenized alike)."""

    def __init__(self, model_dir):
        from transformers import AutoProcessor
        self.inner = AutoProcessor.from_pretrained(str(model_dir))
        template = self.inner.chat_template
        if template.count(END_SLIP.strip("'")) != 1:
            raise ValueError("the template's assistant end is not the expected '<\\s> '")
        self.template = template.replace(END_SLIP, END_FIX)
        self.tokenizer, self.image_processor = self.inner.tokenizer, self.inner.image_processor

    def render(self, conversation, add_generation_prompt=False):
        return self.inner.apply_chat_template(conversation, chat_template=self.template, add_generation_prompt=add_generation_prompt, tokenize=False)

    @staticmethod
    def images_of(conversation):
        return [p['image'] for m in conversation if isinstance(m['content'], list) for p in m['content'] if p['type'] == 'image']

    def apply_chat_template(self, conversations, add_generation_prompt=False, tokenize=False, return_dict=False, return_tensors=None, padding=False, **kwargs):
        batch = bool(conversations) and isinstance(conversations[0], list)
        if not tokenize:
            return [self.render(c, add_generation_prompt) for c in conversations] if batch else self.render(conversations, add_generation_prompt)
        convs = conversations if batch else [conversations]
        return self([self.render(c, add_generation_prompt) for c in convs], [i for c in convs for i in self.images_of(c)], return_tensors or 'pt', padding)

    def prompt_lengths(self, conversations):
        """Per conversation, the tokens before its answer: its chat without the assistant turn, tokenized as the whole chat is."""
        out = []
        for c in conversations:
            asked = [m for m in c if m['role'] != 'assistant']
            out.append(int(self([self.render(asked)], self.images_of(asked))['input_ids'].shape[1]))
        return out

    def __call__(self, text, images, return_tensors='pt', padding=True):
        return self.inner(text=text, images=images or None, return_tensors=return_tensors, padding=padding)


def image_route(features, students, off, grad=False):
    """LLaVA-NeXT's image features (a list, one packed tensor per image) with each image through the adapter ``students.flags`` names
    for it (``vlm_tokens.picked``)."""
    from .vlm_tokens import picked

    def chosen(pixel_values, image_sizes, **kwargs):
        def run():
            out = features(pixel_values, image_sizes, **kwargs)
            return out, [[f] for f in out]
        out, choice = picked(students, run, off, grad)
        return out if choice is None else [p[0] for p in choice]
    return chosen


def tower_module(vision_tower, projector, newline, config):
    """A module whose forward gives what the language model takes from a batch of images: ``(the packed features of every image
    [tokens, width], [])`` -- ``get_image_features`` of LLaVA-NeXT, its parts given (shared or loaded); ``newline`` None packs them
    without the newline tokens (the tower an adapter is aligned on)."""
    import types
    import torch
    from transformers.models.llava_next.modeling_llava_next import LlavaNextModel

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower, self.multi_modal_projector = vision_tower, projector
            self.image_newline = newline
            self.config = config; self.config.out_hidden_size = config.text_config.hidden_size
            self.adapter_targets = ADAPTER_TARGETS
            self.pack_image_features = types.MethodType(LlavaNextModel.pack_image_features, self)
            self.features = types.MethodType(LlavaNextModel.get_image_features, self)

        def forward(self, pixel_values, grid_thw=None):
            return torch.cat(self.features(pixel_values, grid_thw), dim=0), []
    return Tower()


def tower(model_dir):
    """The tower alone (CLIP and the projector, bf16; no newline), loaded from the checkpoint's own tensors."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, CLIPVisionModel
    from transformers.models.llava_next.modeling_llava_next import LlavaNextMultiModalProjector
    config = AutoConfig.from_pretrained(str(model_dir)); config.vision_config._attn_implementation = 'sdpa'
    vision = CLIPVisionModel._from_config(config.vision_config, dtype=torch.bfloat16)
    projector = LlavaNextMultiModalProjector(config).to(torch.bfloat16)
    module = tower_module(vision, projector, None, config); state = {}           # no newline tokens in what is aligned (revision eleven)
    for path in sorted(Path(model_dir).glob('*.safetensors')):
        with safe_open(str(path), 'pt') as f:
            state.update({k: f.get_tensor(k) for k in f.keys() if k.startswith(('vision_tower.', 'multi_modal_projector.'))})
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f'tower weights: missing {missing[:3]}, unexpected {unexpected[:3]}')
    return module


def view_of(inner):
    """The whole model's own tower parts as a ``tower`` module (shared): where a tower adapter is put when reading."""
    return tower_module(inner.vision_tower, inner.multi_modal_projector, inner.image_newline, inner.config)


def from_pretrained(model_dir, adapter=None):
    import torch
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.bfloat16, attn_implementation='sdpa')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    return model


class Reader:
    """LLaVA-1.6-Mistral-7B (a ``train_reader`` adapter merged; a ``train_clean_view`` tower adapter as the student) reading batches as
    ``vlm_tokens.Reader`` does: the same words and decode."""

    def __init__(self, model_dir, device='cuda', adapter=None, layer_map=None, student=None, student_off=False):
        import torch
        from .vlm_tokens import answer_prefill, install_students
        if layer_map:
            raise ValueError('layer maps are not supported on LLaVA')
        self.torch, self.device = torch, device
        self.processor = Processor(model_dir); self.processor.tokenizer.padding_side = 'left'
        self.model = from_pretrained(model_dir, adapter).to(device).eval()
        inner = self.model.model
        self.students = install_students(inner, {'default': student}, student_off, tower=view_of(inner), route=image_route) if student else None
        self.prefill, self.merge = answer_prefill(self.processor), 1
        self.image_token = self.model.config.image_token_index
        self.stop = {self.processor.tokenizer.eos_token_id}

    from .vlm_tokens import Reader as _Qwen
    read = _Qwen.read

    def run(self, jobs, batch, ids, valid, bias, mask, maps, max_new_tokens, loop_limit):
        from transformers import DynamicCache
        from .vlm_tokens import greedy
        torch = self.torch
        if bool(bias[valid.cpu().numpy()].any()):
            raise ValueError('hidden tokens are not supported on LLaVA')
        valid = valid.long(); positions = (valid.cumsum(-1)-1).clamp_min(0)
        cache = DynamicCache(config=self.model.config)
        with torch.inference_mode():
            first = self.model(input_ids=ids, attention_mask=valid, position_ids=positions, pixel_values=batch['pixel_values'].to(self.device, self.model.dtype),
                               image_sizes=batch['image_sizes'].to(self.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
            last = positions.amax(-1); seen = [valid]

            def advance(token, step):
                seen[0] = torch.cat([seen[0], torch.ones(len(jobs), 1, device=self.device, dtype=seen[0].dtype)], dim=1)
                return self.model(input_ids=token.view(-1, 1), attention_mask=seen[0], position_ids=(last+1+step).view(-1, 1), past_key_values=cache, use_cache=True)
            return greedy(first, advance, self.stop, max_new_tokens, loop_limit, self.processor.tokenizer)
