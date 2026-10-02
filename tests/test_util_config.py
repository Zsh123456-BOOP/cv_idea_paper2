import json

import pytest

from revrender import config, util


def test_json_and_directories_are_create_only(tmp_path):
    util.write_json(tmp_path/'a.json', {'x': 1})
    with pytest.raises(FileExistsError):
        util.write_json(tmp_path/'a.json', {'x': 2})
    util.make_output_dir(tmp_path/'run')
    with pytest.raises(FileExistsError):
        util.make_output_dir(tmp_path/'run')
    (tmp_path/'link').symlink_to(tmp_path/'run')
    with pytest.raises(FileExistsError):
        util.make_output_dir(tmp_path/'link')


@pytest.mark.parametrize('name', ['frozen_test', 'holdout', 'x/冻结', 'a/封存/b', 'run_m6_x', 'm6'])
def test_paths_that_name_a_frozen_location_are_refused(tmp_path, name):
    with pytest.raises(util.RefusedError):
        util.safe_path(tmp_path/name)


def test_an_ordinary_path_resolves(tmp_path):
    assert util.safe_path(tmp_path/'bundle') == (tmp_path/'bundle').resolve()


def test_a_manifest_records_config_seed_git_commit_and_input_hashes(tmp_path):
    data = tmp_path/'in.txt'
    data.write_text('hello')
    folder = tmp_path/'folder'
    folder.mkdir()
    (folder/'run_manifest.json').write_text('{}')
    record = util.write_manifest(tmp_path, config={'lr': 3e-4}, inputs=util.input_hashes([data, folder]), stage='smoke')
    on_disk = json.loads((tmp_path/'run_manifest.json').read_text())
    assert on_disk['config'] == {'lr': 3e-4} and on_disk['seed'] == 42 and on_disk['stage'] == 'smoke'
    assert on_disk['inputs'][str(data)] == util.digest(data) and on_disk['inputs'][str(folder)] == util.digest(folder/'run_manifest.json')
    assert 'commit' in on_disk['git'] and 'python' in on_disk['environment']
    with pytest.raises(FileExistsError):
        util.write_manifest(tmp_path, config={})


def test_a_deployed_snapshot_names_its_commit_without_git(tmp_path):
    (tmp_path/'deployment_origin.json').write_text(json.dumps({'source_commit': 'abc123', 'archive_sha256': 'x'}))
    assert util.git_state(tmp_path) == {'commit': 'abc123', 'dirty': False, 'deployed_snapshot': True}


def test_the_recipe_is_the_paper_one_and_typos_are_refused(tmp_path):
    recipe = config.load_recipe()
    train = recipe['train']
    assert (train['batch_size'], train['optimizer_steps'], train['learning_rate'], train['weight_decay']) == (8, 28300, 3e-4, 1e-4)
    assert train['early_stop_patience'] == 6 and train['select_from_step'] == 14150 and recipe['stages'] == {'smoke': 4, 'learning': 512}
    text = (config.CONFIGS/'revrender.toml').read_text()
    bad = tmp_path/'bad.toml'
    bad.write_text(text.replace('weight_decay', 'weight_decya'))
    with pytest.raises(util.RefusedError):
        config.load_recipe(bad)
    bad.write_text(text.replace('"control"', '"mystery"'))
    with pytest.raises(util.RefusedError, match='variant'):
        config.load_recipe(bad)


def test_the_paths_example_reads():
    paths = config.load_paths()
    assert 'raw_data' in paths and 'server' in paths
    assert 'bundle' in paths['server']['data']
