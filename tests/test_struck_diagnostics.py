"""The diagnostic study's versions, scopes, measures and guards (eval.struck_diagnostics)."""
import json

import numpy as np
import pytest

from revrender.data.pixel_masks import SCHEMA, encode
from revrender.eval import struck_diagnostics as D
from revrender.util import RefusedError

COLOURS = {'red': [215, 40, 40], 'blue': [0, 90, 255]}


def page():
    grey = np.full((40, 60), 255, np.uint8)
    grey[18:22, 10:50] = 20          # a strike
    grey[10:30, 20:24] = 30          # a letter it crosses
    region = np.zeros(grey.shape, bool)
    region[8:32, 8:52] = True
    strike = np.zeros(grey.shape, bool)
    strike[18:22, 10:50] = True
    return grey, region, strike


def test_versions_change_only_what_they_name():
    grey, region, strike = page()
    raw = D.version(grey, 'R', region, strike, COLOURS)
    assert (raw[..., 0] == grey).all()
    erased = D.version(grey, 'E*', region, strike, COLOURS)
    assert (erased[strike] == 255).all() and (erased[~strike][..., 0] == grey[~strike]).all()
    marked = D.version(grey, 'M*', region, strike, COLOURS)
    assert (marked[strike] == COLOURS['red']).all() and (marked[~strike][..., 0] == grey[~strike]).all()
    blank = D.version(grey, 'B', region, strike, COLOURS)
    assert (blank[region] == 255).all() and (blank[~region][..., 0] == grey[~region]).all()


def test_the_blue_box_lies_outside_the_region_box():
    grey, region, _ = page()
    rgb = D.draw_box(np.repeat(grey[..., None], 3, 2), [8, 8, 52, 32], gap=3, width=2, colour=COLOURS['blue'])
    blue = (rgb == COLOURS['blue']).all(-1)
    assert blue.any() and not blue[8:32, 8:52].any()


def test_scope_two_takes_the_overlapping_lines_or_a_band():
    lines = [{'box': [0, 0, 60, 10]}, {'box': [5, 12, 55, 30]}, {'box': [0, 31, 60, 40]}]
    crop, rule = D.line_band([20, 14, 30, 26], lines, (60, 40), pad=1, overlap=.5)
    assert rule == '1 line(s)' and crop == [4, 11, 56, 31]
    crop, rule = D.line_band([20, 14, 30, 26], [{'box': [0, 0, 60, 10]}], (60, 40), pad=0, overlap=.5)
    assert rule == 'band' and crop == [0, 8, 60, 32]


def test_measures():
    assert D.measures('x=5', 'x=5') == {'held': 1., 'whole': 1., 'exact': 1., 'extra': 0., 'empty': 0.}
    half = D.measures('ab', 'a')
    assert half['held'] == .5 and half['exact'] == 0. and half['extra'] == 0.
    extra = D.measures('ab', 'abcd')
    assert extra['held'] == 1. and extra['extra'] == 1. and extra['exact'] == 0.
    assert D.measures('a?', 'a')['exact'] is None and D.measures('a?', 'a')['held'] == 1.
    assert D.measures('ab', '[空]')['empty'] == 1. and D.measures('ab', '[空]')['held'] == 0.


def test_bootstrap_resamples_whole_answer_images_and_is_repeatable():
    values, clusters = [1, 1, 0, 0, .5, .5], ['a', 'a', 'b', 'b', 'c', 'c']
    first = D.bootstrap(values, clusters, 2000)
    assert first == D.bootstrap(values, clusters, 2000)
    assert first[0] == pytest.approx(.5) and first[1] < .5 < first[2]


def test_pixel_regions_take_confirmed_readable_strikes(tmp_path):
    revision = {'id': 'w1_2_1#det0001', 'box': [0, 0, 4, 4], 'region': [[0, 0], [4, 0], [4, 4], [0, 4]], 'runs': encode(np.eye(4, dtype=bool)),
                'confirmed': True, 'text_confirmed': True, 'text': 'x=5', 'annotationMode': 'erase'}
    others = [{**revision, 'id': 'a', 'unreadable': True}, {**revision, 'id': 'b', 'not_revision': True}, {**revision, 'id': 'c', 'text': '??'},
              {**revision, 'id': 'd', 'text_confirmed': False}, {**revision, 'id': 'w1_2_1#manual-1'}]
    (tmp_path/'export.json').write_text(json.dumps({'schema': SCHEMA, 'pages': {'w1_2_1': {'revisions': [revision, *others]}}}))
    (tmp_path/'splits.json').write_text(json.dumps([{'page': 'w1_2_1', 'split': 'train'}]))
    regions = D.pixel_regions({'pixel_export': tmp_path/'export.json', 'page_splits': tmp_path/'splits.json'})
    assert [(r['id'], r['manual'], r['split']) for r in regions] == [('w1_2_1#det0001', False, 'train'), ('w1_2_1#manual-1', True, 'train')]
    on_page = D.runs_on_page(revision, (6, 5))
    assert on_page.shape == (5, 6) and on_page.sum() == 4 and on_page[3, 3]


def test_test_writers_are_refused(tmp_path):
    (tmp_path/'mapping.json').write_text(json.dumps({'rows': [{'group_uuid': 'w1', 'split': 'train'}, {'group_uuid': 'w2', 'split': 'test'}]}))
    D.refuse_test_writers([{'page': 'w1_2_1'}], tmp_path/'mapping.json', None)
    with pytest.raises(RefusedError):
        D.refuse_test_writers([{'page': 'w1_2_1'}, {'page': 'w2_3_1'}], tmp_path/'mapping.json', None)


def test_the_config_names_the_primary_models_and_fixed_thresholds():
    cfg = D.load_config(D.Path(__file__).parents[1]/'configs'/'struck_diagnostics.toml')
    assert len(cfg['decisions']['primary']) == 4 and cfg['decisions']['majority'] == 3
    assert cfg['data']['box_remover'].endswith('paper2_strokes_boxes/formal/step_8000.pt')   # synthetic only: never saw these regions


def test_score_reads_predictions_and_takes_the_decisions(tmp_path):
    build = tmp_path/'build'
    regions = [{'id': f'p{k}#r', 'set': 'P', 'page': f'w{k}_2_1', 'split': 'train', 'text': 'x=5', 'mode': 'erase', 'manual': False,
                'strike_share': k/10, 'text_length': 3, 'has_math': False, 'index': k} for k in range(6)]
    rows = [{'id': f'{r["id"]}|{v}|{s}', 'line_id': r['id'], 'variant': v, 'scope': s, 'text': r['text'], 'set': 'P'}
            for r in regions for v in D.VARIANTS for s in D.SCOPES]
    for pack, keep in (('pack_read', lambda r: r['variant'] not in D.MARKED), ('pack_marked', lambda r: r['variant'] in D.MARKED)):
        (build/pack).mkdir(parents=True)
        (build/pack/'pack.json').write_text(json.dumps({'rows': [r for r in rows if keep(r)]}))
    (build/'regions.json').write_text(json.dumps(regions))
    cfg = D.load_config(D.Path(__file__).parents[1]/'configs'/'struck_diagnostics.toml')
    readings = []
    for model in cfg['decisions']['primary']:
        # the raw page is misread, the oracle versions read right
        good = {'E*', 'M*'}
        lines = [json.dumps({'id': r['id'], 'condition': 'boxed_marked' if r['variant'] in D.MARKED else 'boxed_read',
                             'text': 'x=5' if r['variant'] in good else 'y'}) for r in rows]
        (tmp_path/f'{model}.jsonl').write_text('\n'.join(lines))
        readings += ['--readings', f'{model}={tmp_path/model}.jsonl']
    config = tmp_path/'config.toml'
    config.write_text((D.Path(__file__).parents[1]/'configs'/'struck_diagnostics.toml').read_text().replace('bootstrap = 10000', 'bootstrap = 200'))
    D.main(['score', '--config', str(config), '--build', str(build), *readings, '--output-dir', str(tmp_path/'score')])
    summary = json.loads((tmp_path/'score'/'summary.json').read_text())
    rules = summary['decisions']['rules']
    assert rules['D1 problem'] and rules['D2 separation is the bottleneck'] and not rules['D3 keeping (marking) beats erasing']
    assert summary['table'][f'{cfg["decisions"]["primary"][0]}/P/E*/S1']['held'] == 1.
    assert '## 判定' in (tmp_path/'score'/'report.md').read_text()
