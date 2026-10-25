import pytest

torch = pytest.importorskip('torch')

from revrender.reader_ctc import HEIGHT, Reader, batch_of, ctc, encoded, greedy


def test_crops_of_any_size_are_scaled_to_the_readers_height_and_padded_white():
    batch, lengths = batch_of([torch.zeros(20, 50), torch.zeros(40, 400)])
    assert batch.shape[2] == HEIGHT and batch.shape[3] % 4 == 0 and batch[0, 0, 0, -1] == 1
    assert lengths.tolist() == [round(50*HEIGHT/20)//4, round(400*HEIGHT/40)//4]


def test_texts_are_encoded_in_read_backs_spelling_dropping_the_unknown():
    assert encoded('ab?c', ['a', 'b']) == [1, 2]


def test_the_reading_loss_reaches_the_image_and_the_greedy_reading_uses_the_alphabet():
    torch.manual_seed(42); reader = Reader(3).eval().requires_grad_(False)
    image = torch.rand(30, 90, requires_grad=True)
    loss = ctc(reader, [image], ['ab'], ['a', 'b', 'c']); loss.backward()
    assert torch.isfinite(loss) and image.grad.abs().sum() > 0
    batch, lengths = batch_of([image.detach()])
    assert set(greedy(reader(batch), lengths, ['a', 'b', 'c'])[0]) <= {'a', 'b', 'c'}
