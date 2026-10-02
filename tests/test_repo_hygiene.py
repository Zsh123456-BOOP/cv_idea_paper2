"""The repository carries no number of the numbered scripts it was ported from (README's provenance table excepted), no file
named by a number or as a copy of another, and no data, weights or identity files."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = (344, 354, 357, 422, 424, 425, 442, 445, 480, 483, 514, 515, 516, 517, 518, 519, 520, 547, 654, 655, 656, 668, 674, 675,
       685, 687, 699, 701, 702, 703, 750, 767, 772, 775, 782, 788, 793, 816, 831, 832, 833, 834, 835, 853, 854, 861, 865, 866,
       867, 868, 869, 871, 872, 877, 878, 880, 881, 883, 886, 887, 889, 906, 907, 910, 911, 912, 914, 915, 917, 921, 923, 925, 926)
# not inside a hex digest or an identifier, not part of a larger number or a decimal (480.0)
PATTERN = re.compile(r'(?<![0-9A-Za-z_,.])(%s)(?![0-9A-Za-z_,]|\.\d)' % '|'.join(map(str, OLD)))
# One data constant keeps its historical text: the salt that ranks writers for nested training subsets.  It is part of the
# data definition (changing it changes which writers each fraction holds), not a reference to a script.
ALLOWED = ('SALT = ',)
TEXT = ('.py', '.sh', '.toml', '.html', '.md', '.json', '.txt', '.cfg')
# files exempt from the number rule: the provenance table, this test, and the data note, whose image counts (872, 788) equal two script numbers
COUNTS_AS_NUMBERS = ('README.md', 'test_repo_hygiene.py', 'data_location.md')
FORBIDDEN = ('.jpg', '.jpeg', '.png', '.pdf', '.zip', '.pth', '.pt', '.npz', '.npy', '.pkl', '.csv')
MAX_LINES = 800                  # longer than this, split the module (the module boundaries stay)
# a file is not named by a number and is not a second copy of another one (v2, new, old)
NAMED_AS_A_COPY = re.compile(r'^\d|(?:^|[_.-])(?:v\d+|new|old)(?:$|[_.-])', re.I)


def tracked_files():
    try:
        out = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard'],
                             capture_output=True, text=True, check=True).stdout.split('\n')
        return [ROOT/p for p in out if p and (ROOT/p).is_file()]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts]


def test_no_old_script_number_appears_outside_the_readme_and_this_test():
    found = []
    for path in tracked_files():
        if path.name in COUNTS_AS_NUMBERS or path.suffix not in TEXT:
            continue
        for number, line in enumerate(path.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
            if PATTERN.search(line) and not line.lstrip().startswith(ALLOWED):
                found.append(f'{path.relative_to(ROOT)}:{number}: {line.strip()[:100]}')
    assert not found, '\n'.join(found)


def test_no_image_weights_array_or_table_file_is_in_the_repository():
    bad = [str(p.relative_to(ROOT)) for p in tracked_files() if p.suffix.lower() in FORBIDDEN or 'id_mapping' in p.name]
    assert not bad, bad


def test_no_file_is_named_by_a_number_or_as_a_copy_of_another():
    bad = [str(p.relative_to(ROOT)) for p in tracked_files() if NAMED_AS_A_COPY.search(p.name)]
    assert not bad, bad


def test_no_source_file_is_longer_than_about_800_lines():
    long = [f'{p.relative_to(ROOT)}: {n} lines' for p in tracked_files() if p.suffix == '.py'
            and (n := len(p.read_text(encoding='utf-8').splitlines())) > MAX_LINES]
    assert not long, long
