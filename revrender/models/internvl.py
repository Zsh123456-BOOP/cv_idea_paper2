"""InternVL3.5 (its bundled code, transformers 4.57) behind the interfaces the readers and the vision adapters use for Qwen3-VL (plan of
2026-11-21, 换模型验证插件): a processor (``Processor``), the tower alone (``tower``), a reader (``Reader``) and the training model
(``training_model``).

The model's own preprocessing, as its card gives it: an image is cut into 448-pixel tiles on the grid of at most ``MAX_TILES`` tiles
whose shape is closest to the image's, plus a thumbnail of the whole when there is more than one tile, each tile resized bicubically
and normalised with ImageNet's mean and spread; a tile is 256 tokens after the pixel shuffle and ``mlp1`` (``extract_feature``), the
features the language model takes and the ones the adapters align.  The words: the model's template (``internvl2_5``) with our system
words; an image is ``<image>\\n`` before the words (several: ``Image-1: <image>\\n...``, the card's way), each placeholder written
out as ``<img>`` + 256 x tiles ``<IMG_CONTEXT>`` + ``</img>``.

Two changes from the card's loading, both what the readers of Qwen3-VL have: the language model attends with sdpa (without
flash-attn the bundled code falls back to eager, which holds every attention matrix), and the tower's stochastic depth is off
(``drop_path_rate`` 0.1 would make the frozen teacher and the frozen tower under a training reader random; reading is in eval, where
it is off anyway).  The bundled code runs with ``recognize_ocr.safeguards`` set first (eval from model code and the network refused).

"""
import json
import os
import types
from pathlib import Path

TILE, TOKENS = 448, 256
# at most this many tiles an image: the card's 12, lowered only when the smoke's memory says so, 12 -> 6 -> 4, one value for the
# readers, the adapters and the readings (the planning session's rule, plan 换模型验证插件, revision two)
MAX_TILES = int(os.environ.get('INTERNVL_MAX_TILES', '12'))
if MAX_TILES not in (12, 6, 4):
    raise ValueError('INTERNVL_MAX_TILES is 12, 6 or 4')
# the whole model over the two visible cards (accelerate's balanced map) when one card cannot hold a reader's training or reading:
# how it is computed, not what (the planning session's revision four)
SPLIT = os.environ.get('INTERNVL_SPLIT') == '1'
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def is_internvl(model_dir):
    return json.loads((Path(model_dir)/'config.json').read_text()).get('model_type') == 'internvl_chat'


def processing(model_dir):
    """What a run records of how the model sees and attends (InternVL: the tiles, and the two changes from the card's loading; Gemma:
    ``models.gemma.processing``)."""
    from .gemma import is_gemma, processing as gemma_processing
    from .llava import is_llava, processing as llava_processing
    if is_gemma(model_dir):
        return gemma_processing(model_dir)
    if is_llava(model_dir):
        return llava_processing(model_dir)
    if not is_internvl(model_dir):
        return None
    card = json.loads((Path(model_dir)/'config.json').read_text())['vision_config'].get('drop_path_rate')
    return {'tile': TILE, 'max_tiles': MAX_TILES, 'thumbnail': True, 'tokens_per_tile': TOKENS, 'resize': 'bicubic', 'mean': MEAN, 'std': STD,
            'vision_drop_path_rate': {'card': card, 'used': 0.}, 'whole_model_cards': 2 if SPLIT else 1, 'language_attention': 'sdpa (the bundled code falls back to eager without flash-attn)',
            'images_in_words': '<image> before the words; several: Image-1: <image> ... (the card)',
            'tower_alone_checkpointing': 'off (the bundled encoder hard-codes it on; compute only, the adapter needs it off to learn)'}


def bundled(model_dir, name):
    """A class or function of the model's bundled code (``module.attribute``), loaded with the safeguards set."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from ..recognize_ocr import safeguards
    safeguards()
    return get_class_from_dynamic_module(name, str(model_dir))


def closest_grid(aspect, width, height, most=MAX_TILES, size=TILE):
    """The card's ``find_closest_aspect_ratio`` over its grids of 1..``most`` tiles."""
    grids = sorted({(i, j) for n in range(1, most+1) for i in range(1, n+1) for j in range(1, n+1) if 1 <= i*j <= most}, key=lambda g: g[0]*g[1])
    best, gap = (1, 1), float('inf')
    for g in grids:
        d = abs(aspect-g[0]/g[1])
        if d < gap:
            best, gap = g, d
        elif d == gap and width*height > .5*size*size*g[0]*g[1]:
            best = g
    return best


def tiles(image, most=MAX_TILES, size=TILE):
    """The card's ``dynamic_preprocess`` with the thumbnail: the image's tiles, row by row, and the thumbnail when there are several."""
    cols, rows = closest_grid(image.width/image.height, image.width, image.height, most, size)
    resized = image.resize((size*cols, size*rows))
    out = [resized.crop(((i % cols)*size, (i//cols)*size, (i % cols+1)*size, (i//cols+1)*size)) for i in range(cols*rows)]
    return out+[image.resize((size, size))] if len(out) > 1 else out


class Processor:
    """The model's processing in the shape of a transformers processor: ``tokenizer``, ``image_processor(images=...)`` (pixel values of
    the tiles, and per image ``[tiles, 1, 1]`` as its grid), ``apply_chat_template`` (the template's text, or tokenized with the images)
    and ``__call__(text=, images=)`` (each ``<image>`` written out for its image's tiles, in order)."""

    def __init__(self, model_dir):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=False)
        self.template = bundled(model_dir, 'conversation.get_conv_template')
        self.image_processor = self.images

    def images(self, images, return_tensors='pt'):
        import torch
        from torchvision import transforms as T
        from torchvision.transforms.functional import InterpolationMode
        step = T.Compose([T.Resize((TILE, TILE), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
        cut = [tiles(image.convert('RGB')) for image in images]
        return {'pixel_values': torch.stack([step(t) for c in cut for t in c]), 'image_grid_thw': torch.tensor([[len(c), 1, 1] for c in cut])}

    def text(self, conversation, add_generation_prompt):
        chat = self.template('internvl2_5')
        for message in conversation:
            parts = message['content'] if isinstance(message['content'], list) else [{'type': 'text', 'text': message['content']}]
            shown = sum(p['type'] == 'image' for p in parts); words = ''.join(p['text'] for p in parts if p['type'] == 'text')
            if message['role'] == 'system':
                chat.system_message = words
                continue
            places = '<image>\n' if shown == 1 else ''.join(f'Image-{n+1}: <image>\n' for n in range(shown))
            chat.append_message(chat.roles[0] if message['role'] == 'user' else chat.roles[1], places+words)
        if add_generation_prompt:
            chat.append_message(chat.roles[1], None)
        return chat.get_prompt()

    def apply_chat_template(self, conversations, add_generation_prompt=False, tokenize=False, return_dict=False, return_tensors=None, padding=False):
        several = bool(conversations) and isinstance(conversations[0], list)
        texts = [self.text(c, add_generation_prompt) for c in (conversations if several else [conversations])]
        if not tokenize:
            return texts if several else texts[0]
        images = [p['image'] for c in (conversations if several else [conversations]) for m in c if isinstance(m['content'], list)
                  for p in m['content'] if p['type'] == 'image']
        return self(text=texts, images=images, return_tensors=return_tensors, padding=padding)

    def __call__(self, text, images, return_tensors='pt', padding=True):
        pixels = self.images(images)
        counts = iter(int(n) for n in pixels['image_grid_thw'][:, 0])
        written = []
        for t in text:
            while '<image>' in t:
                t = t.replace('<image>', '<img>'+'<IMG_CONTEXT>'*TOKENS*next(counts)+'</img>', 1)
            written.append(t)
        if next(counts, None) is not None:
            raise ValueError('more images than <image> places')
        batch = self.tokenizer(written, return_tensors=return_tensors, padding=padding)
        return {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], **pixels}


def tower(model_dir):
    """The tower alone -- the vision model and ``mlp1``, bf16, no stochastic depth -- as a module whose forward gives the features the
    language model takes: ``(tokens [tiles x 256, width], [])`` (no deepstack levels), as ``train_clean_view.levels`` reads a tower."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    vision = config.vision_config; vision.drop_path_rate = 0.; vision.use_flash_attn = False
    chat = bundled(model_dir, 'modeling_internvl_chat.InternVLChatModel')
    module = torch.nn.Module()
    module.config = vision
    module.vision_model = bundled(model_dir, 'modeling_intern_vit.InternVisionModel')(vision)
    module.vision_model.encoder.gradient_checkpointing = False   # the bundled encoder hard-codes reentrant checkpointing, which leaves the adapter without gradients
    width = vision.hidden_size*int(1/config.downsample_ratio)**2
    module.mlp1 = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, config.llm_config.hidden_size), torch.nn.GELU(),
                                      torch.nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size))
    module.select_layer, module.downsample_ratio, module.ps_version = config.select_layer, config.downsample_ratio, config.ps_version
    module.pixel_shuffle = types.MethodType(chat.pixel_shuffle, module)
    state = {}
    for path in sorted(Path(model_dir).glob('*.safetensors')):
        with safe_open(str(path), 'pt') as f:
            state.update({k: f.get_tensor(k) for k in f.keys() if k.startswith(('vision_model.', 'mlp1.'))})
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f'tower weights: missing {missing[:3]}, unexpected {unexpected[:3]}')
    module.to(torch.bfloat16)
    module.forward = types.MethodType(lambda self, pixel_values, grid_thw=None: (self.features(pixel_values).flatten(0, 1), []), module)
    module.features = types.MethodType(chat.extract_feature, module)
    return module


def whole(model_dir):
    """The whole model, bf16, the language model on sdpa and the tower without stochastic depth (see the module's words)."""
    import torch
    from transformers import AutoConfig, AutoModel
    bundled(model_dir, 'modeling_internvl_chat.InternVLChatModel')          # the safeguards, before the bundled code loads
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True); config.vision_config.drop_path_rate = 0.
    model = AutoModel.from_pretrained(str(model_dir), config=config, torch_dtype=torch.bfloat16, trust_remote_code=True, use_flash_attn=False, low_cpu_mem_usage=True,
                                      **({'device_map': 'balanced'} if SPLIT else {}))
    model.language_model.config._attn_implementation = 'sdpa'
    return model


def view_of(model):
    """The whole model's own tower and ``mlp1`` under one module, named as in ``tower`` (an adapter of the tower loads into it)."""
    import torch
    module = torch.nn.Module()
    module.config, module.vision_model, module.mlp1 = model.config.vision_config, model.vision_model, model.mlp1
    return module


def embedded(model, input_ids, pixel_values, features=None):
    """The language model's input embeddings with the image tokens' features put in (``features``: the tiles' features if given)."""
    embeds = model.language_model.get_input_embeddings()(input_ids)
    if pixel_values is None:
        return embeds
    vit = (features if features is not None else model.extract_feature(pixel_values)).reshape(-1, embeds.shape[-1])
    place = input_ids == model.img_context_token_id
    if int(place.sum()) != vit.shape[0]:
        raise ValueError(f'{int(place.sum())} image tokens for {vit.shape[0]} features')
    return embeds.masked_scatter(place[..., None], vit.to(embeds.device, embeds.dtype))


def training_model(model_dir):
    """The whole model whose forward takes what ``train_reader.Collate`` makes (``image_grid_thw`` unused) and keeps the last
    ``logits_to_keep`` positions' logits, as the Qwen models' does."""
    model = whole(model_dir)
    from transformers import AutoTokenizer
    model.img_context_token_id = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=False).convert_tokens_to_ids('<IMG_CONTEXT>')

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None, logits_to_keep=0, **kwargs):
        return self.language_model(inputs_embeds=embedded(self, input_ids, pixel_values), attention_mask=attention_mask, logits_to_keep=logits_to_keep, **kwargs)
    model.forward = types.MethodType(forward, model)
    return model


def tile_features(features, students, off):
    """``extract_feature`` with each image's tiles through the adapter ``students.flags`` names for it (``vlm_tokens.picked``); the
    tiles per image are ``students.counts``, set by the reader for the batch."""
    import torch
    from .vlm_tokens import picked

    def chosen(pixel_values):
        def run():
            out = features(pixel_values)
            return out, list(out.split(students.counts))
        out, choice = picked(students, run, off)
        return out if choice is None else torch.cat(choice)
    return chosen


class Reader:
    """InternVL3.5 (with a LoRA adapter of ``train_reader``, merged; with a tower adapter of ``train_clean_view`` as the student) reading
    batches as ``vlm_tokens.Reader`` does: the same words, the same decode (``vlm_tokens.greedy``), positions counted over the unpadded
    tokens (the model's language part uses 1-D rotary positions).  Hiding tokens and layer maps are not supported."""

    def __init__(self, model_dir, device='cuda', adapter=None, layer_map=None, student=None, student_off=False):
        import torch
        from .vlm_tokens import install_students
        if layer_map:
            raise ValueError('layer maps are not supported on InternVL')
        self.torch, self.device = torch, device
        self.processor = Processor(model_dir); self.processor.tokenizer.padding_side = 'left'
        self.model = whole(model_dir)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter)).merge_and_unload()
        self.model = (self.model if SPLIT else self.model.to(device)).eval()
        if SPLIT:
            self.device = 'cuda:0'                                       # the inputs go to the first card; accelerate's hooks carry them on
        self.image_token = self.model.img_context_token_id = self.processor.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.students = install_students(self.model, {'default': student}, student_off, tower=view_of(self.model), attribute='extract_feature',
                                         route=tile_features) if student else None
        self.prefill, self.merge = '', 1
        self.stop = {self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')}

    from .vlm_tokens import Reader as _Qwen
    read = _Qwen.read

    def run(self, jobs, batch, ids, valid, bias, mask, maps, max_new_tokens, loop_limit):
        from transformers import DynamicCache
        from .vlm_tokens import greedy
        torch, dtype = self.torch, self.model.dtype
        if self.students is not None:
            self.students.counts = [int(n) for n in batch['image_grid_thw'][:, 0]]
        positions = (valid.long().cumsum(-1)-1).clamp_min(0)
        language = self.model.language_model; cache = DynamicCache(config=language.config)
        with torch.inference_mode():
            embeds = embedded(self.model, ids, batch['pixel_values'].to(self.device, dtype))
            first = language(inputs_embeds=embeds, attention_mask=mask, position_ids=positions, past_key_values=cache, use_cache=True, logits_to_keep=1)
            last = positions.amax(-1)
            keys = [torch.from_numpy(bias).to(self.device, dtype)]

            def advance(token, step):
                keys[0] = torch.cat([keys[0], torch.zeros(len(jobs), 1, device=self.device, dtype=dtype)], dim=1)
                return language(input_ids=token.view(-1, 1), attention_mask=keys[0].view(len(jobs), 1, 1, -1), position_ids=(last+1+step).view(-1, 1),
                                past_key_values=cache, use_cache=True)
            return greedy(first, advance, self.stop, max_new_tokens, loop_limit, self.processor.tokenizer)
