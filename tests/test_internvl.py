from PIL import Image

from revrender.models.internvl import closest_grid, tiles
from revrender.models.vlm_tokens import looped


def test_tiles_follow_the_card():
    assert closest_grid(10., 1000, 100) == (10, 1)
    cut = tiles(Image.new('RGB', (1000, 100)))
    assert len(cut) == 11 and all(t.size == (448, 448) for t in cut)          # ten tiles and the thumbnail
    assert len(tiles(Image.new('RGB', (300, 300)))) == 1                      # one tile, no thumbnail
    assert closest_grid(1., 2000, 2000) == (3, 3)                             # a tie on shape goes to the larger grid a big image fills


def test_looped_unchanged():
    assert looped([1, 2, 3]*4, 4) == 9 and looped([1, 2, 3, 4], 4) == 0


def test_accumulated_is_the_batch_loss():
    torch = __import__('pytest').importorskip('torch')
    from revrender.train_clean_view import accumulated, pair_loss
    g = torch.Generator().manual_seed(42)
    scale = torch.nn.Parameter(torch.ones(1))
    data = [(torch.randn(n, 6, generator=g), torch.randn(n, 6, generator=g)) for n in (5, 9, 3, 7)]     # (student input, teacher) per region
    student_of = lambda piece: [torch.cat([s for s, _ in piece])*scale]
    teacher_of = lambda piece: [torch.cat([t for _, t in piece])]
    whole = pair_loss(student_of(data), teacher_of(data)); whole.backward(); grad = scale.grad.clone(); scale.grad = None
    total = accumulated(student_of, teacher_of, [data[:2], data[2:]])
    assert abs(total-float(whole)) < 1e-6 and torch.allclose(scale.grad, grad, atol=1e-6)


def test_one_at_a_time_is_the_group_loss():
    torch = __import__('pytest').importorskip('torch')
    from revrender.train_reader import answer_loss, one_at_a_time

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); torch.manual_seed(42); self.e = torch.nn.Embedding(11, 4); self.o = torch.nn.Linear(4, 11)
        def forward(self, input_ids=None, attention_mask=None, pixel_values=None, logits_to_keep=0):
            return type('O', (), {'logits': self.o(self.e(input_ids))[:, -logits_to_keep:]})()
    model = Toy().cuda() if torch.cuda.is_available() else None
    if model is None:
        __import__('pytest').skip('one_at_a_time moves batches to cuda')
    ids = [torch.tensor([1, 2, 3, 4, 5, 6]), torch.tensor([7, 8, 9, 10])]
    labels = [torch.tensor([-100, -100, 3, 4, 5, 6]), torch.tensor([-100, 8, 9, 10])]
    single = lambda i: {'input_ids': ids[i][None], 'attention_mask': torch.ones(1, len(ids[i]), dtype=torch.long), 'labels': labels[i][None], 'pixel_values': torch.zeros(1)}
    pad = lambda x, v: torch.cat([x, torch.full((6-len(x),), v)])
    batch = {'input_ids': torch.stack([ids[0], pad(ids[1], 0)]).cuda(), 'attention_mask': torch.tensor([[1]*6, [1]*4+[0]*2]).cuda(),
             'labels': torch.stack([labels[0], pad(labels[1], -100)]).cuda(), 'pixel_values': torch.zeros(1).cuda()}
    whole = answer_loss(model, batch); whole.backward(); grads = [q.grad.clone() for q in model.parameters()]; model.zero_grad()
    total = one_at_a_time(model, [single(0), single(1)], 1)
    assert abs(total-float(whole)) < 1e-6 and all(torch.allclose(q.grad, g, atol=1e-6) for q, g in zip(model.parameters(), grads))
