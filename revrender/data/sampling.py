"""Batches for training: balanced shuffled cycles over the revised and clean pools, augmented on a producer thread.

One producer owns the random generator and both cycles, so the sequence of draws is the same whether or not the
consumer is slow.  A batch is 4 revised + 4 clean lines (for batch size 8), shuffled, each warped by ``paired_affine``.
"""
import queue
import random
import threading

import numpy as np
from scipy import ndimage

from ..util import require
from .augment import paired_affine

PREFETCH_DEPTH = 1


class ShuffledCycle:
    """Rows in a fresh random order each pass; ``take_distinct`` never repeats a row inside one request."""

    def __init__(self, rows, rng: random.Random):
        require(rows, 'cannot sample an empty row collection')
        self.rows = rows
        self.rng = rng
        self.order = []
        self.position = 0

    def take_distinct(self, count):
        require(0 <= count <= len(self.rows), 'distinct cycle request exceeds its row pool')
        output = []
        selected = set()
        while len(output) < count:
            if self.position >= len(self.order):
                self.order = list(range(len(self.rows)))
                self.rng.shuffle(self.order)
                self.position = 0
            index = self.order[self.position]
            self.position += 1
            if index in selected:
                continue
            selected.add(index)
            output.append(self.rows[index])
        return output


def take_balanced_batch(positive_cycle, clean_cycle, batch_size, negative_ratio):
    positive_count = int(round(batch_size / (1.0 + negative_ratio)))
    clean_count = batch_size - positive_count
    require(positive_count > 0 and abs(clean_count / positive_count - negative_ratio) <= 1e-9,
            'batch size must exactly express the requested negative ratio')
    rows = positive_cycle.take_distinct(positive_count)+clean_cycle.take_distinct(clean_count)
    line_ids = [str(row['line_id']) for row in rows]
    require(len(line_ids) == len(set(line_ids)), 'balanced batch contains a duplicate line')
    return rows, positive_count, clean_count


def take_mixed_batch(cycles, batch_size, negative_ratio, share):
    """A balanced batch with ``share`` of its revised and of its clean lines from the second pair of cycles (positive, clean, positive,
    clean): the synthetic set's lines first, then the real ones."""
    positive_count = int(round(batch_size / (1.0 + negative_ratio)))
    clean_count = batch_size - positive_count
    real_positive, real_clean = int(round(share*positive_count)), int(round(share*clean_count))
    positive, clean, other_positive, other_clean = cycles
    rows = (positive.take_distinct(positive_count-real_positive)+other_positive.take_distinct(real_positive)
            + clean.take_distinct(clean_count-real_clean)+other_clean.take_distinct(real_clean))
    line_ids = [str(row['line_id']) for row in rows]
    require(len(line_ids) == len(set(line_ids)), 'mixed batch contains a duplicate line')
    return rows


def gold_instances(mask):
    """Connected components (8-neighbour) of one 0..1 gold mask: the targets a line is trained on."""
    labels, count = ndimage.label(np.asarray(mask) > 0.5, structure=np.ones((3, 3), bool))
    return [(labels == index) for index in range(1, count + 1)]


def numbered_instances(numbers):
    """One instance per revision number of a labelled mask (0 none), the numbers a warp left on the strip."""
    return [numbers == k for k in range(1, int(numbers.max())+1) if (numbers == k).any()]


def prepare_batch(images, masks, labelled=None):
    """Strips as an N x 1 x H x W float array in [-1, 1] and each line's gold components as a float block (a ``labelled`` line's
    mask numbers its revisions: one instance per number)."""
    values = np.stack(images)[:, None].astype(np.float32)/127.5-1
    blocks = []
    for mask, numbers in zip(masks, labelled or [False]*len(masks)):
        instances = numbered_instances(mask) if numbers else gold_instances(mask.astype(np.float32)/255)
        blocks.append(np.stack([t.astype(np.float32) for t in instances]) if instances
                      else np.empty((0, *mask.shape), dtype=np.float32))
    return values, blocks


class BatchStream:
    """Single-producer, depth-one prefetcher; ``next()`` returns the prepared (images, gold blocks) of one step."""

    def __init__(self, data, rng: random.Random, steps, batch_size, negative_ratio, mix=None):
        """``mix`` = (pools with fit_positive and fit_clean, share): that share of every batch comes from those pools."""
        self.rng, self.steps, self.batch_size, self.negative_ratio = rng, steps, batch_size, negative_ratio
        self.positive_cycle = ShuffledCycle(data['fit_positive'], rng)
        self.clean_cycle = ShuffledCycle(data['fit_clean'], rng)
        self.mix = None if mix is None else (ShuffledCycle(mix[0]['fit_positive'], rng), ShuffledCycle(mix[0]['fit_clean'], rng), mix[1])
        self.items = queue.Queue(maxsize=PREFETCH_DEPTH)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._produce, name='batch-prefetch', daemon=True)
        self.lines = []          # line ids handed out, in order (kept only when track=True; for tests and audits)
        self.track = False

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.items.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def make_batch(self):
        if self.mix is None:
            rows, _, _ = take_balanced_batch(self.positive_cycle, self.clean_cycle, self.batch_size, self.negative_ratio)
        else:
            rows = take_mixed_batch((self.positive_cycle, self.clean_cycle, *self.mix[:2]), self.batch_size, self.negative_ratio, self.mix[2])
        order = list(range(len(rows)))
        self.rng.shuffle(order)
        rows = [rows[index] for index in order]
        images, masks = [], []
        for row in rows:
            image, mask = paired_affine(row['image'], row['mask'], self.rng)
            images.append(image)
            masks.append(mask)
        if self.track:
            self.lines.append([r['line_id'] for r in rows])
        return prepare_batch(images, masks, [bool(row.get('labelled')) for row in rows]), rows

    def _produce(self):
        try:
            for _ in range(self.steps):
                if self.stop.is_set():
                    return
                batch, _ = self.make_batch()
                if not self._put(('batch', batch)):
                    return
        except BaseException as exc:
            self._put(('error', exc))

    def start(self):
        self.thread.start()

    def next(self):
        kind, value = self.items.get()
        if kind == 'error':
            raise RuntimeError('batch prefetch failed') from value
        return value

    def close(self):
        self.stop.set()
        try:
            while True:
                self.items.get_nowait()
        except queue.Empty:
            pass
        self.thread.join(timeout=5.0)
        require(not self.thread.is_alive(), 'batch prefetch thread did not stop')
