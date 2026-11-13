import json

from revrender.data.pixel_masks import SCHEMA
from revrender.data.struck_dataset import merged, split_writers


def export(path, revisions):
    path.write_text(json.dumps({'schema': SCHEMA, 'pages': {'w1_q1_0': {'revisions': revisions}}}))
    return path


def test_split_writers_fills_test_and_val_to_a_tenth_by_writer():
    counts = {f'w{k:02d}': 1+k % 7 for k in range(60)}
    parts = split_writers(counts); total = sum(counts.values())
    assert parts == split_writers(counts)
    for split in ('test', 'val'):
        n = sum(counts[w] for w in parts if parts[w] == split)
        assert .1*total <= n < .1*total+7
    assert set(parts.values()) == {'train', 'val', 'test'}
    testable = {w for w in counts if int(w[1:]) % 3}
    assert all(w in testable for w, s in split_writers(counts, testable).items() if s == 'test')


def test_merged_keeps_the_latest_confirmed_record_and_honours_deletions(tmp_path):
    a = export(tmp_path/'01.json', [
        {'id': 'p#r1', 'box': [0, 0, 10, 10], 'confirmed': True, 'runs': 'old'},
        {'id': 'p#r2', 'box': [50, 50, 60, 60], 'confirmed': True},
        {'id': 'p#r3', 'box': [80, 80, 90, 90]}])                                        # never confirmed: left out
    b = export(tmp_path/'02.json', [
        {'id': 'q#other', 'box': [1, 0, 11, 10], 'text_confirmed': True, 'text': 'x'},   # same region by IoU: text added
        {'id': 'q#r2', 'box': [50, 50, 60, 60], 'deleted': True}])
    held = merged([a, b])['w1_q1_0']
    assert len(held) == 1
    r = held[0]
    assert r['pixels']['runs'] == 'old' and r['text']['text'] == 'x' and r['export'] == '02.json' and r['first_export'] == '01.json'
