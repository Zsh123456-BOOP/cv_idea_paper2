"""Synthetic revisions on clean lines, with the text the reader should give known by construction.

A clean line (a line of a writer who did not revise it, its text read by two strong models that agree) is turned into one of three
samples; the target text is always the host's own, so no character under a stroke has to be located:

  clean   the line as it is: its text
  paste   a stretch of writing cut from another line of the same writer is struck out and put into the line between two words (or at its
          end): the host's text, the pasted words are not in it
  stack   the same struck writing put above or below one of the host's words, in the same columns, as a correction is written over what
          it replaces: the host's text, and the strokes overlap writing that is valid
  whole   the whole line struck out with strokes across it: nothing (an empty reading)

Strikes are drawn in seven styles (``STYLES``), in the colour of the host's ink, with the stroke width scaled to the line's height; a
stroke may run on over the host's neighbouring writing (``OVERSHOOT``), and then overlaps valid ink.  Every sample returns the image,
the mask of the strokes (ink only, the shape a revision mask has), and the boxes of the masks in thousandths of the image.  All
randomness comes from the ``random.Random`` given: the same recipe draws the same sample.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from ..eval.ink import INK_BELOW

STYLES = ('line', 'double', 'wave', 'zigzag', 'cross', 'scribble', 'blot')
KINDS = ('clean', 'paste', 'stack', 'whole')
SUPER = 3                  # strokes are drawn at this many times the size and brought down: soft edges
GAP = .35                  # words are separated by this many line heights of blank columns (at least MIN_GAP pixels)
MIN_GAP = 8
OVERSHOOT = (0, .5)        # a stroke runs on past the struck writing by this many line heights, from..to
STROKE = (.05, .11)        # stroke width as a share of the line height (at least 2 pixels)
SEGMENT_WIDTH = (.12, .6)  # a pasted stretch is this wide, in widths of the host
STACK_OVERLAP = (.0, .5)   # a stacked stretch shares this much of its height with the host's rows
MASK_ALPHA = .35           # stroke pixels at least this opaque are in the mask


def ink_of(rgb):
    return np.asarray(Image.fromarray(rgb).convert('L')) < INK_BELOW


def ink_rows(ink):
    rows = np.flatnonzero(ink.any(1))
    return (int(rows[0]), int(rows[-1])+1) if len(rows) else None


def ink_colour(rgb, ink, rng):
    """The colour of the line's ink (the median of its darkest pixels), a little varied."""
    pixels = rgb[ink]
    if not len(pixels):
        return (30, 30, 30)
    darkest = pixels[np.argsort(pixels.sum(1))[:max(1, len(pixels)//3)]]
    return tuple(int(np.clip(c+rng.uniform(-12, 12), 0, 255)) for c in np.median(darkest, axis=0))


def word_spans(ink, height):
    """Column spans (left, right) of the writing of a line: runs of ink columns, runs closer than a gap joined."""
    gap = max(MIN_GAP, int(GAP*height))
    columns = np.flatnonzero(ink.any(0))
    if not len(columns):
        return []
    spans, start, last = [], int(columns[0]), int(columns[0])
    for c in columns[1:]:
        if c-last > gap:
            spans.append((start, last+1))
            start = int(c)
        last = int(c)
    spans.append((start, last+1))
    return spans


def take_segment(donor, rng, max_width):
    """A stretch of writing cut from a donor line (array [h, w, 3]): one to three whole words, rows trimmed to the ink.  None if there is none."""
    ink = ink_of(donor)
    rows = ink_rows(ink)
    spans = word_spans(ink, donor.shape[0])
    if rows is None or not spans:
        return None
    first = rng.randrange(len(spans))
    left, right = spans[first]
    for span in spans[first+1:first+rng.choice((1, 1, 2, 3))]:
        if span[1]-left > max_width:
            break
        right = span[1]
    if right-left > max_width:
        right = left+int(max_width)
    pad = 2
    return donor[max(0, rows[0]-pad):rows[1]+pad, max(0, left-pad):right+pad]


def strokes(size, style, rng, colour, width, overshoot):
    """Strokes of one style across a region (width, height) of the page, drawn with ``overshoot`` pixels of room on each side:
    (RGB layer on white, opacity [h, w] of the strokes, both with the region's size plus the room)."""
    w, h = size
    room = int(overshoot)
    big = Image.new('L', ((w+2*room)*SUPER, h*SUPER), 0)
    draw = ImageDraw.Draw(big)
    t = max(2, int(round(width)))*SUPER
    x0, x1, mid = 0, (w+2*room)*SUPER, h*SUPER/2

    def stroke(points, thickness=t):
        draw.line([(x, y) for x, y in points], fill=255, width=int(thickness), joint='curve')
    tilt = rng.uniform(-.12, .12)
    if style == 'line':
        y = mid+rng.uniform(-.15, .15)*h*SUPER
        stroke([(x0, y-tilt*(x1-x0)/2), (x1, y+tilt*(x1-x0)/2)])
    elif style == 'double':
        gap = rng.uniform(.18, .32)*h*SUPER
        for sign in (-1, 1):
            y = mid+sign*gap/2
            stroke([(x0, y-tilt*(x1-x0)/2), (x1, y+tilt*(x1-x0)/2)])
    elif style == 'wave':
        amplitude, wavelength = rng.uniform(.08, .22)*h*SUPER, rng.uniform(.15, .45)*h*SUPER
        phase = rng.uniform(0, 2*math.pi)
        stroke([(x, mid+amplitude*math.sin(2*math.pi*x/wavelength+phase)) for x in np.linspace(x0, x1, max(8, int((x1-x0)/(2*SUPER))))])
    elif style == 'zigzag':
        turns = rng.randint(4, 12)
        xs = np.linspace(x0, x1, turns+1)
        stroke([(x, mid+(-1)**i*rng.uniform(.25, .45)*h*SUPER) for i, x in enumerate(xs)])
    elif style == 'cross':
        for _ in range(rng.choice((1, 2))):
            a, b = rng.uniform(0, .3), rng.uniform(.7, 1)
            stroke([(x0+a*(x1-x0), 0), (x0+b*(x1-x0), h*SUPER)])
            stroke([(x0+a*(x1-x0), h*SUPER), (x0+b*(x1-x0), 0)])
    elif style == 'scribble':
        for _ in range(rng.randint(10, 24)):
            xa, xb = rng.uniform(x0, x1), rng.uniform(x0, x1)
            stroke([(xa, rng.uniform(0, h*SUPER)), (xb, rng.uniform(0, h*SUPER))], t*rng.uniform(.7, 1.3))
    else:                                                              # blot: a filled shape over the writing
        centre_x, centre_y = rng.uniform(.35, .65)*(x1-x0), mid
        rx, ry = rng.uniform(.4, .6)*(x1-x0), rng.uniform(.35, .5)*h*SUPER
        draw.ellipse([centre_x-rx, centre_y-ry, centre_x+rx, centre_y+ry], fill=255)
    opacity = np.asarray(big.resize((w+2*room, h), Image.LANCZOS), np.float32)/255.
    if style == 'blot':
        opacity = np.clip(opacity*rng.uniform(.85, 1.), 0, 1)
    layer = np.full((h, w+2*room, 3), 255, np.float32)
    layer = layer*(1-opacity[..., None])+np.array(colour, np.float32)*opacity[..., None]
    return layer, opacity


def place(canvas, piece, strokes_mask, left, top):
    """Darken a canvas ([h, w, 3] uint8) with a piece (RGB on white) whose top left corner is at (left, top), clipped to the canvas; returns the
    canvas-sized mask of the piece's ``strokes_mask``."""
    h, w = canvas.shape[:2]
    ph, pw = piece.shape[:2]
    x0, y0, x1, y1 = max(0, left), max(0, top), min(w, left+pw), min(h, top+ph)
    mask = np.zeros((h, w), bool)
    if x1 <= x0 or y1 <= y0:
        return mask
    canvas[y0:y1, x0:x1] = np.minimum(canvas[y0:y1, x0:x1], piece[y0-top:y1-top, x0-left:x1-left])
    mask[y0:y1, x0:x1] = strokes_mask[y0-top:y1-top, x0-left:x1-left]
    return mask


def boxes_of(mask):
    """The boxes of the connected pieces of a mask in thousandths of the image (left, top, right, bottom), left to right: one per piece
    of columns that has strokes, joined across gaps of less than a tenth of the width."""
    h, w = mask.shape
    columns = np.flatnonzero(mask.any(0))
    if not len(columns):
        return []
    out, start, last = [], int(columns[0]), int(columns[0])
    for c in list(columns[1:])+[None]:
        if c is None or c-last > max(2, w//10):
            rows = np.flatnonzero(mask[:, start:last+1].any(1))
            out.append([round(1000*start/w), round(1000*int(rows[0])/h), round(1000*(last+1)/w), round(1000*(int(rows[-1])+1)/h)])
            if c is not None:
                start = int(c)
        if c is not None:
            last = int(c)
    return out


def struck_segment(donor, host, rng, style=None, overlap=True):
    """A stretch of the donor, scaled to the host's writing and struck out: (image [h, w, 3] with the room of the overshoot on both sides, mask of
    the strokes in it, the room), or None when the donor has no writing to take."""
    host_ink = ink_of(host)
    host_rows = ink_rows(host_ink)
    if host_rows is None:
        return None
    host_height = host_rows[1]-host_rows[0]
    segment = take_segment(donor, rng, SEGMENT_WIDTH[1]*host.shape[1])
    if segment is None or segment.shape[1] < 8:
        return None
    rows = ink_rows(ink_of(segment))
    scale = float(np.clip(host_height/max(1, rows[1]-rows[0]), .6, 1.6)*rng.uniform(.9, 1.15))
    segment = np.asarray(Image.fromarray(segment).resize((max(8, round(segment.shape[1]*scale)), max(4, round(segment.shape[0]*scale))), Image.LANCZOS))
    room = int(rng.uniform(*OVERSHOOT)*host_height) if overlap else 0
    layer, opacity = strokes((segment.shape[1], segment.shape[0]), style or rng.choice(STYLES), rng, ink_colour(host, host_ink, rng),
                             max(2., rng.uniform(*STROKE)*host_height), room)
    piece = np.full((segment.shape[0], segment.shape[1]+2*room, 3), 255, np.uint8)
    piece[:, room:room+segment.shape[1]] = segment
    piece = np.minimum(piece, layer.round().astype(np.uint8))
    return piece, opacity >= MASK_ALPHA, room


def compose(host, donors, rng, kind, style=None):
    """One sample from a host line (array [h, w, 3] uint8) and donor lines of the same writer (arrays): a dict (image: PIL RGB, mask: bool
    [h, w], boxes, kind, pasted).  ``kind`` is one of ``KINDS``; a ``paste`` or ``stack`` with no donor to take writing from comes out ``clean`` (``kind`` says so)."""
    host = np.asarray(host, np.uint8)
    host_ink = ink_of(host)
    rows = ink_rows(host_ink)
    if kind == 'whole' and rows is not None:
        layer, opacity = strokes((host.shape[1], host.shape[0]), style or rng.choice(STYLES), rng, ink_colour(host, host_ink, rng),
                                 max(2., rng.uniform(*STROKE)*(rows[1]-rows[0])), 0)
        canvas = host.copy()
        mask = place(canvas, layer.round().astype(np.uint8), opacity >= MASK_ALPHA, 0, 0)
        return {'image': Image.fromarray(canvas), 'mask': mask, 'boxes': boxes_of(mask), 'kind': 'whole', 'pasted': 0}
    if kind == 'paste' and rows is not None:
        spans = word_spans(host_ink, host.shape[0])
        cuts = [0, host.shape[1]]+[(a[1]+b[0])//2 for a, b in zip(spans, spans[1:])]
        for donor in rng.sample(list(donors), len(donors)):
            made = struck_segment(np.asarray(donor, np.uint8), host, rng, style, overlap=rng.random() < .4)
            if made is None:
                continue
            piece, piece_mask, room = made
            cut, pad, core = rng.choice(cuts), rng.randint(3, 12), piece.shape[1]-2*room
            height, width = max(host.shape[0], piece.shape[0]), host.shape[1]+core+2*pad
            canvas = np.full((height, width, 3), 255, np.uint8)
            top = (height-host.shape[0])//2
            canvas[top:top+host.shape[0], :cut] = host[:, :cut]
            canvas[top:top+host.shape[0], cut+core+2*pad:] = host[:, cut:]
            top_piece = int(np.clip((height-piece.shape[0])//2+rng.uniform(-.1, .1)*host.shape[0], 0, height-piece.shape[0]))
            mask = place(canvas, piece, piece_mask, cut+pad-room, top_piece)
            return {'image': Image.fromarray(canvas), 'mask': mask, 'boxes': boxes_of(mask), 'kind': 'paste', 'pasted': 1}
    if kind == 'stack' and rows is not None:
        spans = word_spans(host_ink, host.shape[0])
        for donor in rng.sample(list(donors), len(donors)):
            made = struck_segment(np.asarray(donor, np.uint8), host, rng, style, overlap=rng.random() < .4)
            if made is None:
                continue
            piece, piece_mask, room = made
            a, b = rng.choice(spans)
            left = int(a+rng.uniform(-.3, .3)*(b-a))-room
            shared = int(rng.uniform(*STACK_OVERLAP)*min(piece.shape[0], rows[1]-rows[0]))
            above = rng.random() < .5
            height = host.shape[0]+piece.shape[0]-shared
            pad_left = max(0, -left)
            width = max(host.shape[1], left+piece.shape[1])+pad_left
            canvas = np.full((height, width, 3), 255, np.uint8)
            top_host = piece.shape[0]-shared if above else 0
            canvas[top_host:top_host+host.shape[0], pad_left:pad_left+host.shape[1]] = host
            top_piece = 0 if above else host.shape[0]-shared
            mask = place(canvas, piece, piece_mask, left+pad_left, top_piece)
            return {'image': Image.fromarray(canvas), 'mask': mask, 'boxes': boxes_of(mask), 'kind': 'stack', 'pasted': 1}
    return {'image': Image.fromarray(host), 'mask': np.zeros(host.shape[:2], bool), 'boxes': [], 'kind': 'clean', 'pasted': 0}
