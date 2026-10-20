"""A recorded command run again on clean inputs (tools.replay)."""
import json

import pytest

from revrender.tools import replay as R
from revrender.util import RefusedError


def test_the_recorded_command_is_remapped_without_the_labmate_folders(tmp_path):
    argv = ['/home/zsh/hme/code_paper2_abc/revrender/data/synthetic_set.py', '--corpus', '/o/paper2_reader_corpus_main/corpus.jsonl',
            '--pages', '/d/unlabeled_v1/images', '--pages', '/d/full1026_v1/images', '--copies', '4', '--output-dir', '/o/paper2_synthetic_strokes']
    (tmp_path/'run_manifest.json').write_text(json.dumps({'environment': {'argv': argv}}))
    module, args = R.rewrite(R.recorded(tmp_path), '/o/new', [('/o/paper2_reader_corpus_main', '/o/clean_corpus')], [('--copies', '2')])
    assert module == 'revrender.data.synthetic_set'
    assert args == ['--corpus', '/o/clean_corpus/corpus.jsonl', '--pages', '/d/unlabeled_v1/images', '--copies', '2', '--output-dir', '/o/new']
    with pytest.raises(RefusedError, match='labmate'):
        R.rewrite(['x/revrender/a.py', '--bank', '/d/full1026_v1/bank', '--output-dir', 'o'], 'n')
