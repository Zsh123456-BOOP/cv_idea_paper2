"""Hiding visual tokens from the text: which tokens, which keys, which mask -- and, on a model, that hiding nothing changes nothing."""
import os
from pathlib import Path

import numpy as np
import pytest

from revrender.data import line_pack
from revrender.models import vlm_tokens as V


def test_a_token_cell_is_hidden_by_the_share_of_its_area_under_the_mask():
    mask = np.zeros((40, 80), bool)
    mask[:, 20:40] = True                                   # a band over the columns 20..39 of 80
    shares = V.cell_shares(mask, 2, 4)                      # four token columns of 20 pixels each
    assert shares.shape == (8,) and np.allclose(shares.reshape(2, 4), [[0, 1, 0, 0], [0, 1, 0, 0]])
    half = np.zeros((40, 80), bool)
    half[:, 30:50] = True
    shares = V.cell_shares(half, 2, 4)
    assert np.allclose(shares.reshape(2, 4)[0], [0, .5, .5, 0])
    assert V.hidden_tokens(shares, .5).reshape(2, 4)[0].tolist() == [False, True, True, False] and not V.hidden_tokens(shares, .6).any()
    assert not V.hidden_tokens(np.zeros(8), 0.).any()      # a cell the mask does not touch is never hidden, whatever the threshold


def test_the_blocked_keys_are_the_hidden_image_tokens_of_each_row_and_padding_is_never_one():
    ids = np.array([[0, 0, 7, 7, 7, 7, 5], [0, 7, 7, 7, 7, 5, 6]])       # 7 = the image token; the first row is left-padded
    valid = np.array([[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]], bool)
    blocked = V.blocked_keys(ids, valid, 7, [np.array([False, True, False, True]), None])
    assert blocked.tolist() == [[False, False, False, True, False, True, False], [False]*7]
    with pytest.raises(ValueError, match='image tokens'):
        V.blocked_keys(ids, valid, 7, [np.array([True, False]), None])


def test_the_key_bias_removes_padding_and_hard_hidden_tokens_and_lowers_soft_ones():
    valid = np.array([[0, 1, 1, 1], [1, 1, 1, 1]], bool)
    blocked = np.array([[0, 0, 1, 0], [0, 1, 0, 0]], bool)
    bias = V.key_bias(valid, blocked, [0., 3.])
    assert bias[0].tolist() == [V.NEG, 0., V.NEG, 0.] and bias[1].tolist() == [0., -3., 0., 0.]


def test_the_prefill_mask_is_causal_carries_the_key_bias_and_leaves_no_row_empty():
    bias = np.array([[0., V.NEG, 0., 0.]], np.float32)
    mask = V.prefill_mask(bias)
    assert mask.shape == (1, 1, 4, 4)
    allowed = mask[0, 0] > V.NEG/2
    assert allowed.tolist() == [[True, False, False, False], [True, True, False, False], [True, False, True, False], [True, False, True, True]]
    assert allowed.any(axis=1).all()


def test_the_band_and_the_ink_come_back_from_what_the_pack_stores():
    from PIL import Image
    band = line_pack.band_mask([[250, 0, 500, 1000]], (40, 10), pad=2)
    assert band.shape == (10, 40) and band[:, 8:22].all() and not band[:, :8].any() and not band[:, 22:].any()
    raw = np.full((4, 6, 3), 255, np.uint8)
    raw[1, 2] = 10
    raw[3, 5] = 20
    erased = raw.copy()
    erased[3, 5] = 255
    ink = line_pack.ink_mask(Image.fromarray(raw), Image.fromarray(erased))
    assert ink.sum() == 1 and ink[3, 5]


@pytest.mark.skipif(not os.environ.get('RVR_VLM_DIR'), reason='needs a Qwen3-VL folder in RVR_VLM_DIR and a GPU')
def test_with_nothing_hidden_the_loop_reads_what_generate_reads_and_hiding_the_whole_image_changes_it():
    import torch
    from PIL import Image, ImageDraw
    reader = V.Reader(Path(os.environ['RVR_VLM_DIR']))
    embed = reader.model.model.visual.patch_embed
    patches = torch.randn(40, embed.in_channels*embed.temporal_patch_size*embed.patch_size**2, device=reader.device, dtype=reader.model.dtype)
    with torch.backends.cudnn.flags(enabled=False):                           # the convolution as the library defines it
        as_convolution = embed.proj(patches.view(-1, embed.in_channels, embed.temporal_patch_size, embed.patch_size, embed.patch_size)).view(-1, embed.embed_dim)
    assert torch.allclose(embed(patches).float(), as_convolution.float(), atol=2e-2, rtol=2e-2)
    image = Image.new('RGB', (240, 48), 'white')
    ImageDraw.Draw(image).text((10, 12), 'x = 3 + 4', fill='black')
    words = '请转写图中的文字。仅返回 <transcription>转写结果</transcription>'
    rows, cols = reader.grid(image)
    plain = reader.read([(image, words, None, 0.)], 'You are an OCR transcription assistant.', 40)[0]
    conversation = [{'role': 'system', 'content': [{'type': 'text', 'text': 'You are an OCR transcription assistant.'}]},
                    {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': words}]}]
    batch = reader.processor.apply_chat_template([conversation], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(reader.device)
    batch['pixel_values'] = batch['pixel_values'].to(reader.model.dtype)
    out = reader.model.generate(**batch, max_new_tokens=40, do_sample=False)
    assert plain['text'] == reader.processor.tokenizer.decode(out[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
    hidden = reader.read([(image, words, np.ones(rows*cols, bool), 0.)], 'You are an OCR transcription assistant.', 40)[0]
    assert hidden['text'] != plain['text']
