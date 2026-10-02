"""Read the TOML files in configs/: where the data lives (paths.toml) and the training recipes."""
import os
import tomllib
from pathlib import Path

from .util import ROOT, require

CONFIGS = ROOT/'configs'


def load_toml(path):
    with Path(path).expanduser().open('rb') as f:
        return tomllib.load(f)


def load_paths(path=None):
    """configs/paths.toml (or $REVRENDER_PATHS); falls back to the example file so a fresh checkout still reads."""
    path = Path(path or os.environ.get('REVRENDER_PATHS') or CONFIGS/'paths.toml')
    if not path.is_file():
        path = CONFIGS/'paths.example.toml'
    return load_toml(path)


def expand(value):
    return Path(os.path.expanduser(str(value)))


def load_recipe(path=None):
    """The RevRender recipe, validated: unknown keys are refused so a typo cannot silently change a run."""
    recipe = load_toml(path or CONFIGS/'revrender.toml')
    expected = {'model': {'variant'},
                'train': {'batch_size', 'negative_ratio', 'optimizer_steps', 'warmup_steps', 'validation_interval',
                          'select_from_step', 'early_validation_multiplier', 'learning_rate', 'weight_decay',
                          'grad_clip', 'clip_groups', 'early_stop_patience'},
                'stages': {'smoke', 'learning'}}
    require(set(recipe) == set(expected), f'recipe sections {sorted(recipe)} differ from {sorted(expected)}')
    for section, keys in expected.items():
        require(set(recipe[section]) == keys, f'recipe [{section}] has {sorted(set(recipe[section]) ^ keys)} wrong')
    require(recipe['model']['variant'] in ('control', 'nofine', 'nocontext', 'nodense'), 'undeclared model variant')
    train = recipe['train']
    require(train['batch_size'] >= 2 and train['batch_size'] % 2 == 0 and train['negative_ratio'] == 1.,
            'positive steps and an even, 1:1 balanced batch are required')
    return recipe


def load_baselines(path=None):
    return load_toml(path or CONFIGS/'baselines.toml')['native']
