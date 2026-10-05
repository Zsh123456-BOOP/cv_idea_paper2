"""The strike GAN: shapes placed where they are told, the networks' shapes, its sets of lines, a few training steps and the set it draws."""
import json
import math

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip('torch')

from revrender import pages as pages_module                                   # noqa: E402
from revrender.models import strike_gan as G                                 # noqa: E402


def bar():
    shape = torch.zeros(1, 1, 64, 256)
    shape[..., 28:36, :] = 1                  # a horizontal stroke through the middle of its map
    return shape


def test_a_shape_is_placed_at_its_centre_size_and_turn():
    opacity = G.place(bar(), torch.tensor([[[200., 48.]]]), torch.tensor([[[120., 16.]]]), torch.tensor([[0.]]))[0, 0]
    ys, xs = torch.nonzero(opacity > .5, as_tuple=True)
    assert 140 <= int(xs.min()) <= 142 and 258 <= int(xs.max()) <= 260 and 46 <= int(ys.min()) and int(ys.max()) <= 50
    turned = G.place(bar(), torch.tensor([[[200., 48.]]]), torch.tensor([[[80., 16.]]]), torch.tensor([[math.pi/2]]))[0, 0]
    ys, xs = torch.nonzero(turned > .5, as_tuple=True)
    assert int(ys.max())-int(ys.min()) > 60 and int(xs.max())-int(xs.min()) < 6        # standing up now


def test_the_networks_give_a_struck_line_and_a_score():
    torch.manual_seed(0)
    generator, discriminator = G.Generator(), G.Discriminator()
    line = torch.ones(2, *G.CANVAS)
    line[:, 40:56, 100:400] = .1
    shapes = bar().expand(2, G.SLOTS, 64, 256)
    aspect, length = torch.full((2, G.SLOTS), 6.), torch.full((2, G.SLOTS), 3.)
    fake, each = generator(line, torch.randn(2, G.NOISE), shapes, aspect, length, torch.full((2,), 40.))
    assert fake.shape == (2, *G.CANVAS) and each.shape == (2, G.SLOTS, *G.CANVAS) and fake.min() >= 0 and fake.max() <= 1
    assert each[:, 0].amax(dim=(1, 2)).min() > .3                              # the first place is always drawn
    assert discriminator(fake).shape == (2,) and G.augment(fake, G.seeded(1, 'cpu')).shape == fake.shape
    assert float(G.diversity(each, each.flip(0), torch.zeros(2, G.NOISE), torch.ones(2, G.NOISE))) <= 0


def write_world(tmp_path):
    """Two writers' pages (one revised line with a struck word, clean lines), RevRender's instances, a catalogue, a bank."""
    scans, detections, bank = tmp_path/'scans', tmp_path/'det', tmp_path/'bank'
    for d in (scans, detections, bank):
        d.mkdir()
    page = np.full((300, 800, 3), 250, np.uint8)
    for row in range(3):
        for x in range(40, 700, 90):
            page[40+row*90:70+row*90, x:x+60:4] = 20                      # words of thin upright strokes
    page[54:57, 40:160] = 20                                                  # a struck line on the first row
    catalogue = []
    for writer in ('a', 'b'):
        Image.fromarray(page).save(scans/f'{writer}_5_1.png')
        mask = np.zeros((40, 760), bool)
        mask[19:22, 0:120] = True
        pages_module.write_page(detections, f'{writer}_5_1', (800, 300), [{'box': [40, 35, 800, 75], 'source': 'ocr', 'ocr_type': 'text', 'ocr_text': ''}],
                                [{'line': 0, 'score': .9, 'box': [40, 35, 800, 75], 'mask': mask}])
        for row in range(3):
            catalogue.append({'image_id': f'{writer}_5_1', 'group_uuid': writer, 'representative_line_id': f'{writer}_5_1:{row}', 'has_edit': row == 0,
                              'box_1000': [30, round(1000*(30+row*90)/300), 900, round(1000*(80+row*90)/300)]})
    (detections/'run_manifest.json').write_text('{}')
    (tmp_path/'catalogue.jsonl').write_text('\n'.join(json.dumps(r) for r in catalogue))
    stroke = np.full((4, 100), 255, np.uint8)
    stroke[1:3] = 30
    np.savez_compressed(bank/'bank.npz', piece_0=stroke)
    (bank/'pieces.json').write_text(json.dumps([{'id': 'piece_0', 'kind': 'line', 'size': [100, 4], 'line_height': 40, 'ink_level': 30}]))
    (bank/'run_manifest.json').write_text('{}')
    mapping, same = tmp_path/'map.json', tmp_path/'same.json'
    mapping.write_text(json.dumps({'rows': [{'group_uuid': 'a', 'split': 'train'}, {'group_uuid': 'b', 'split': 'test'}]}))
    same.write_text(json.dumps({'pairs': []}))
    return scans, detections, bank, mapping, same


def test_real_lines_are_revised_lines_of_struck_lines_only_and_hosts_clean_ones_of_usable_writers(tmp_path):
    from revrender.data import gan_sets as S
    scans, detections, bank, mapping, same = write_world(tmp_path)
    out = tmp_path/'sets'
    assert S.main(['--detections', str(detections), '--pages', str(scans), '--mapping', str(mapping), '--same-student', str(same),
                   '--catalogue', str(tmp_path/'catalogue.jsonl'), '--output-dir', str(out)]) == 0
    real, hosts = json.loads((out/'real.json').read_text()), json.loads((out/'hosts.json').read_text())
    assert [r['line_id'] for r in real] == ['a_5_1:0'] and [h['line_id'] for h in hosts] == ['a_5_1:1', 'a_5_1:2']   # b is a test writer
    assert np.load(out/'real.npy').shape == (1, *G.CANVAS) and all(30 < h['strip_height'] < 96 for h in hosts)
    missing, none = tmp_path/'missing.txt', tmp_path/'none'
    none.mkdir()
    assert S.main(['--detections', str(none), '--pages', str(scans), '--mapping', str(mapping), '--same-student', str(same),
                   '--catalogue', str(tmp_path/'catalogue.jsonl'), '--missing', str(missing)]) == 0
    assert missing.read_text().split() == ['a_5_1']                            # a revised line, no detections on its page


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_a_few_steps_train_and_the_generator_draws_a_crop_set(tmp_path):
    from revrender import train_gan
    from revrender.data import gan_sets as S
    from revrender.data import gan_strikes as X
    from revrender.data import synthetic_set as T
    scans, detections, bank, mapping, same = write_world(tmp_path)
    sets = tmp_path/'sets'
    S.main(['--detections', str(detections), '--pages', str(scans), '--mapping', str(mapping), '--same-student', str(same),
            '--catalogue', str(tmp_path/'catalogue.jsonl'), '--output-dir', str(sets)])
    gan = tmp_path/'gan'
    assert train_gan.main(['--sets', str(sets), '--bank', str(bank), '--steps', '6', '--batch', '2', '--sample-every', '3', '--output-dir', str(gan)]) == 0
    assert (gan/'generator_last.pt').exists() and (gan/'sheet_000003.png').exists() and len((gan/'log.jsonl').read_text().splitlines()) == 0
    out = tmp_path/'drawn'
    assert X.main(['--generator', str(gan/'generator_last.pt'), '--sets', str(sets), '--bank', str(bank), '--pages', str(scans), '--copies', '3',
                   '--clean-share', '1', '--output-dir', str(out)]) == 0
    records, images, labels = T.read_set(out)
    assert images.shape == labels.shape and any(r['kinds'] == ['clean'] for r in records)
    assert all(labels[i].any() for i, r in enumerate(records) if r['kinds'] == ['gan'])
