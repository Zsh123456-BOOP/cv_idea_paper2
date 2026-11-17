"""The reader's training samples: what is drawn from a recipe, the answer it is taught and that only the answer carries the loss."""
import json
import os

import numpy as np
import pytest
from PIL import Image

from revrender import train_reader as T


def page(tmp_path):
    """A white page with two lines of 'writing' (blocks of ink) at known places."""
    array = np.full((400, 800, 3), 255, np.uint8)
    for top in (50, 250):
        for left in (40, 200, 380):
            array[top:top+40, left:left+100] = 20
    folder = tmp_path/'pages'
    folder.mkdir()
    Image.fromarray(array).save(folder/'w1_3_1.jpg', quality=95)
    return folder


def row(kind, line, donors, text='x = 3'):
    return {'id': f'{line}#{kind}', 'split': 'train', 'kind': kind, 'line_id': line, 'image_id': 'w1_3_1', 'writer': 'w1', 'text': text if kind != 'whole' else '', 'host_text': text,
            'box_1000': [30, 100, 900, 300] if line.endswith('L1') else [30, 600, 900, 800], 'donors': donors, 'seed': 42}


def test_the_answer_is_the_text_between_the_tags_and_nothing_is_written_as_the_empty_mark():
    assert T.target_of(' x = 3 \n') == '<transcription>x = 3</transcription>' and T.target_of('') == '<transcription>[空]</transcription>'
    assert T.target_of('  ') == '<transcription>[空]</transcription>'


def test_a_recipe_is_drawn_into_an_image_and_the_text_it_should_be_read_as(tmp_path):
    folder = page(tmp_path)
    rows = [row('clean', 'w1_3_1:L1', []), row('whole', 'w1_3_1:L1', []), row('paste', 'w1_3_1:L1', ['w1_3_1:L2']), row('paste', 'w1_3_1:L1', [])]
    crops = T.Crops([folder], rows+[row('clean', 'w1_3_1:L2', [])])
    (clean, text_clean), (whole, text_whole), (paste, text_paste), (alone, text_alone) = (T.render(r, crops) for r in rows)
    assert text_clean == text_paste == text_alone == 'x = 3' and text_whole == ''
    assert paste.width > clean.width and alone.size == clean.size                  # a paste with no donor comes out clean, with the host's text
    assert np.asarray(whole).min() < np.asarray(clean).min()+1 and np.array_equal(np.asarray(T.render(rows[2], crops)[0]), np.asarray(paste))


def test_with_blank_the_columns_under_the_strokes_are_white_and_the_rest_is_as_it_was(tmp_path):
    folder = page(tmp_path)
    rows = [row('paste', 'w1_3_1:L1', ['w1_3_1:L2']), row('whole', 'w1_3_1:L1', []), row('clean', 'w1_3_1:L1', [])]
    crops = T.Crops([folder], rows+[row('clean', 'w1_3_1:L2', [])])
    plain, blanked = T.render(rows[0], crops), T.render(rows[0], crops, blank=True)
    assert plain[1] == blanked[1] and plain[0].size == blanked[0].size
    assert (np.asarray(blanked[0]) < 200).sum() < (np.asarray(plain[0]) < 200).sum() and (np.asarray(blanked[0]) >= np.asarray(plain[0])).all()      # whiter, nothing darker
    assert np.asarray(T.render(rows[1], crops, blank=True)[0]).min() >= 250                               # a whole-line strike whites out the whole line
    assert np.array_equal(np.asarray(T.render(rows[2], crops, blank=True)[0]), np.asarray(T.render(rows[2], crops)[0]))


def test_the_corpus_rows_are_read_by_kind_and_split(tmp_path):
    path = tmp_path/'corpus.jsonl'
    rows = [row('clean', 'w1_3_1:L1', []), {**row('whole', 'w1_3_1:L1', []), 'split': 'val'}, row('paste', 'w1_3_1:L1', ['w1_3_1:L2'])]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert len(T.read_rows(path)) == 3 and [r['kind'] for r in T.read_rows(path, ['clean', 'paste'])] == ['clean', 'paste']
    assert [r['kind'] for r in T.read_rows(path, split='val')] == ['whole']


@pytest.mark.skipif(not os.environ.get('RVR_VLM_DIR'), reason='needs a Qwen3-VL folder in RVR_VLM_DIR')
def test_only_the_answer_carries_the_loss():
    from transformers import AutoProcessor
    from revrender.recognize_lines import prompt
    processor = AutoProcessor.from_pretrained(os.environ['RVR_VLM_DIR'])
    processor.tokenizer.padding_side = 'right'
    batch = T.Collate(processor, prompt('raw_final', {'boxes': []}))([(Image.new('RGB', (200, 40), 'white'), 'x = 3 + 4'), (Image.new('RGB', (120, 50), 'white'), '')])
    for b, answer in enumerate(('<transcription>x = 3 + 4</transcription><|im_end|>\n', '<transcription>[空]</transcription><|im_end|>\n')):
        kept = batch['input_ids'][b][batch['labels'][b] != -100]
        assert processor.tokenizer.decode(kept) == answer


def test_the_answer_loss_is_the_loss_transformers_computes_over_the_whole_sequence():
    torch = pytest.importorskip('torch')
    from transformers.loss.loss_utils import ForCausalLMLoss
    from revrender.train_reader import answer_loss
    torch.manual_seed(0)
    full = torch.randn(2, 9, 11)

    class Model:                       # the logits of the last ``logits_to_keep`` positions, as the model gives them
        def __call__(self, input_ids, logits_to_keep):
            return type('Out', (), {'logits': full[:, -logits_to_keep:]})()
    labels = torch.tensor([[-100]*5+[3, 4, 5, -100], [-100]*4+[1, 2, -100, -100, -100]])      # answers, then right padding
    expected = ForCausalLMLoss(full, labels, vocab_size=11)
    assert torch.allclose(answer_loss(Model(), {'input_ids': labels, 'labels': labels}), expected)
