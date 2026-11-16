"""Synthetic revisions on clean lines, with the text the reader should give known by construction.

A clean line (a line of a writer who did not revise it, its text read by two strong models that agree) is turned into one of three
samples; the target text is always the host's own, so no character under a stroke has to be located:

  clean   the line as it is: its text
  paste   a stretch of writing cut from another line of the same writer is struck out and put into the line between two words (or at its
          end): the host's text, the pasted words are not in it
  stack   the same struck writing put above or below one of the host's words, in the same columns, as a correction is written over what
          it replaces: the host's text, and the strokes overlap writing that is valid
  whole   the whole line struck out with strokes across it: nothing (an empty reading)
  over    one to three characters of the host's own writing struck out or blacked out where they stand (for the stroke remover: the
          reader's text would have to lose them, so its corpus does not use this kind)

Strikes are drawn in seven styles (``STYLES``), in the colour of the host's ink, with the stroke width scaled to the line's height; a
stroke may run on over the host's neighbouring writing (``OVERSHOOT``), and then overlaps valid ink.  Given a ``Bank`` of real marks cut
from paper 1's gold (``data.strike_bank``), a share of the strikes (``gold_share``) is one of those instead (``GOLD_STYLES``): a struck
line stretched along the stretch it strikes, a blackout over it, scaled by the ratio of the two lines' heights and drawn in the host's ink.
Every sample returns the image, the mask of the strokes (ink only, the shape a revision mask has), the boxes of the masks in thousandths
of the image, and what removing the revision takes (``remove``, its faint edge included): a struck line's pixels that are not writing (the
writing it crosses keeps its pixels, so it stays whole), a blackout's pixels all (what it covers is gone).  All randomness comes from the ``random.Random`` given:
the same recipe draws the same sample, and without a bank the same sample as before the bank existed.
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..eval.ink import INK_BELOW

STYLES = ('line', 'double', 'wave', 'zigzag', 'cross', 'scribble', 'blot')
GOLD_STYLES = ('gold_line', 'gold_blot')
BLOT_STYLES = ('blot', 'gold_blot')    # removing these removes everything they cover
GOLD_SCALE = (.85, 1.15)   # a real mark is scaled by the ratio of the lines' heights times this
FIT = (.5, 2.)             # a struck line is stretched along its place to at most this much off that ratio
KINDS = ('clean', 'paste', 'stack', 'whole', 'over')
OVER_CHARACTERS = (1, 3)    # an ``over`` strikes this many characters (a character as wide as the line's writing is high, give or take)
SUPER = 3                  # strokes are drawn at this many times the size and brought down: soft edges
GAP = .35                  # words are separated by this many line heights of blank columns (at least MIN_GAP pixels)
MIN_GAP = 8
OVERSHOOT = (0, .5)        # a stroke runs on past the struck writing by this many line heights, from..to
STROKE = (.05, .11)        # stroke width as a share of the line height (at least 2 pixels)
SEGMENT_WIDTH = (.12, .6)  # a pasted stretch is this wide, in widths of the host
STACK_OVERLAP = (.0, .5)   # a stacked stretch shares this much of its height with the host's rows
MASK_ALPHA = .35           # stroke pixels at least this opaque are in the mask
REMOVE_ALPHA = .1          # removing a revision takes its pixels at least this opaque (its faint edge too)
WRITING_BELOW = 235        # a pixel of the line without the revision this dark is writing (its soft edge too): a struck line leaves it


def ink_of(rgb):
    return np.asarray(Image.fromarray(rgb).convert('L')) < INK_BELOW


class Bank:
    """The real marks of ``data.strike_bank`` (its output folder), by kind."""

    def __init__(self, folder):
        self.pieces = json.loads((Path(folder)/'pieces.json').read_text())
        with np.load(Path(folder)/'bank.npz') as arrays:
            self.grey = {p['id']: arrays[p['id']] for p in self.pieces}
        self.by_kind = {k: [p for p in self.pieces if p['kind'] == k] for k in ('line', 'blot')}


def choose_style(rng, bank=None, gold_share=0., whole=False, blots=True):
    """A style.  Without a bank one of the drawn ones.  With a bank a real mark with probability ``gold_share`` -- a struck line or a
    blackout as often as the bank holds each -- else a drawn one; a whole line is struck through, never blacked out (a student
    rarely blacks out a whole line), and with ``blots`` False nothing is (blackouts are taken as unreadable, not removed)."""
    if bank is None:
        return rng.choice(STYLES)
    if rng.random() < gold_share:
        lines = len(bank.by_kind['line'])/max(1, len(bank.pieces))
        return 'gold_line' if whole or not blots or rng.random() < lines else 'gold_blot'
    return rng.choice([s for s in STYLES if not ((whole or not blots) and s in BLOT_STYLES)])


def gold_piece(bank, p, rng, size):
    """A bank piece resized to ``size`` (width, height), flipped at random, as the opacity of its ink (its own ink level is 1)."""
    piece = Image.fromarray(bank.grey[p['id']])
    if rng.random() < .5:
        piece = piece.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < .5:
        piece = piece.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    grey = np.asarray(piece.resize(size, Image.LANCZOS), np.float32)
    return np.clip((255.-grey)/(255.-min(p['ink_level'], 200)), 0, 1)


def gold_strokes(size, kind, rng, colour, overshoot, bank, line_height):
    """Real marks of ``kind`` across a region (width, height) of a line ``line_height`` high, with ``overshoot`` pixels of room on each
    side: the same (layer, opacity) as ``strokes``.  A struck line is one mark stretched along the region and its room (by at most ``FIT``
    more than the ratio of the lines' heights); a blackout is real blots at that ratio side by side over the region, as a student blacks
    out one character after another."""
    w, h = size
    room = int(overshoot)
    opacity = np.zeros((h, w+2*room), np.float32)
    if kind == 'line':
        width = w+2*room
        fits = [p for p in bank.by_kind['line'] if FIT[0] <= width*p['line_height']/(p['size'][0]*line_height) <= FIT[1]]
        p = rng.choice(fits) if fits else max(bank.by_kind['line'], key=lambda q: q['size'][0]/q['line_height'])
        down = min(h, max(1, round(p['size'][1]*line_height/p['line_height']*rng.uniform(*GOLD_SCALE))))
        alpha = gold_piece(bank, p, rng, (width, down))
        top = int(np.clip((h-down)/2+rng.uniform(-.15, .15)*h, 0, h-down))
        opacity[top:top+down] = alpha
    else:
        x = room
        while x < room+w:
            left = room+w-x
            fits = [p for p in bank.by_kind['blot'] if p['size'][0]*line_height/p['line_height'] <= 1.3*left] or \
                [min(bank.by_kind['blot'], key=lambda q: q['size'][0]/q['line_height'])]
            p = rng.choice(fits)
            scale = line_height/p['line_height']*rng.uniform(*GOLD_SCALE)
            across, down = max(1, round(p['size'][0]*scale)), min(h, max(1, round(p['size'][1]*scale)))
            alpha = gold_piece(bank, p, rng, (across, down))[:, :room+w-x]
            top = int(np.clip((h-down)/2+rng.uniform(-.15, .15)*h, 0, h-down))
            opacity[top:top+down, x:x+alpha.shape[1]] = np.maximum(opacity[top:top+down, x:x+alpha.shape[1]], alpha)
            x += max(1, int(across*rng.uniform(.7, .95)))
    layer = np.full((h, w+2*room, 3), 255, np.float32)
    layer = layer*(1-opacity[..., None])+np.array(colour, np.float32)*opacity[..., None]
    return layer, opacity


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


def strokes(size, style, rng, colour, width, overshoot, bank=None, line_height=None):
    """Strokes of one style across a region (width, height) of the page, drawn with ``overshoot`` pixels of room on each side:
    (RGB layer on white, opacity [h, w] of the strokes, both with the region's size plus the room).  A real mark (``GOLD_STYLES``) needs
    the bank and the height of the line it goes on."""
    if style in GOLD_STYLES:
        return gold_strokes(size, style[len('gold_'):], rng, colour, overshoot, bank, line_height)
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


def placed(shape, piece_mask, left, top):
    """A mask of a piece whose top left corner is at (left, top) on a canvas of ``shape`` (h, w), clipped to it."""
    h, w = shape[:2]
    ph, pw = piece_mask.shape[:2]
    x0, y0, x1, y1 = max(0, left), max(0, top), min(w, left+pw), min(h, top+ph)
    mask = np.zeros((h, w), bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = piece_mask[y0-top:y1-top, x0-left:x1-left]
    return mask


def place(canvas, piece, strokes_mask, left, top):
    """Darken a canvas ([h, w, 3] uint8) with a piece (RGB on white) whose top left corner is at (left, top), clipped to the canvas; returns the
    canvas-sized mask of the piece's ``strokes_mask``."""
    h, w = canvas.shape[:2]
    ph, pw = piece.shape[:2]
    x0, y0, x1, y1 = max(0, left), max(0, top), min(w, left+pw), min(h, top+ph)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = np.minimum(canvas[y0:y1, x0:x1], piece[y0-top:y1-top, x0-left:x1-left])
    return placed(canvas.shape, strokes_mask, left, top)


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


def struck_segment(donor, host, rng, style=None, overlap=True, bank=None, gold_share=0., blots=True):
    """A stretch of the donor, scaled to the host's writing and struck out: (image [h, w, 3] with the room of the overshoot on both sides, the
    opacity of the strokes in it, the room, the same image without the strokes, the style), or None when the donor has no writing to take."""
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
    style = style or choose_style(rng, bank, gold_share, blots=blots)
    layer, opacity = strokes((segment.shape[1], segment.shape[0]), style, rng, ink_colour(host, host_ink, rng),
                             max(2., rng.uniform(*STROKE)*host_height), room, bank, host.shape[0])
    plain = np.full((segment.shape[0], segment.shape[1]+2*room, 3), 255, np.uint8)
    plain[:, room:room+segment.shape[1]] = segment
    return np.minimum(plain, layer.round().astype(np.uint8)), opacity, room, plain, style


def removal(faint, plain, style):
    """The pixels removing a revision takes (``faint``: its pixels down to ``REMOVE_ALPHA``): a blackout's all, a struck line's where there
    is no writing (``plain``: the image without the revision)."""
    if style in BLOT_STYLES:
        return faint.copy()
    return faint & ~(np.asarray(Image.fromarray(plain).convert('L')) < WRITING_BELOW)


def numbered(labels, remove):
    """The revisions' numbers on the image (0 none) with the new one's ``remove`` numbered next (where no earlier one is)."""
    out = labels.copy()
    out[remove & (out == 0)] = int(labels.max())+1
    return out


def compose(host, donors, rng, kind, style=None, bank=None, gold_share=0., labels=None, blots=True):
    """One sample from a host line (array [h, w, 3] uint8) and donor lines of the same writer (arrays): a dict (image: PIL RGB, mask: bool
    [h, w], boxes, kind, pasted, remove: bool [h, w], style, labels).  ``kind`` is one of ``KINDS``; a ``paste`` or ``stack`` with no donor to
    take writing from comes out ``clean`` (``kind`` says so).  ``bank`` and ``gold_share``: the real marks some strikes are drawn with;
    ``blots`` False: no blackouts, only struck lines.
    ``labels`` (uint8 [h, w]) numbers the revisions the host already has (a sample made from another one): they move with the host's
    pixels, and the new revision's ``remove`` is numbered next in the sample's ``labels``."""
    host = np.asarray(host, np.uint8)
    labels = np.zeros(host.shape[:2], np.uint8) if labels is None else np.asarray(labels, np.uint8)
    host_ink = ink_of(host)
    rows = ink_rows(host_ink)
    if kind == 'whole' and rows is not None:
        style = style or choose_style(rng, bank, gold_share, whole=True, blots=blots)
        layer, opacity = strokes((host.shape[1], host.shape[0]), style, rng, ink_colour(host, host_ink, rng),
                                 max(2., rng.uniform(*STROKE)*(rows[1]-rows[0])), 0, bank, host.shape[0])
        canvas = host.copy()
        mask = place(canvas, layer.round().astype(np.uint8), opacity >= MASK_ALPHA, 0, 0)
        remove = removal(placed(canvas.shape, opacity >= REMOVE_ALPHA, 0, 0), host, style)
        return {'image': Image.fromarray(canvas), 'mask': mask, 'boxes': boxes_of(mask), 'kind': 'whole', 'pasted': 0,
                'remove': remove, 'style': style, 'labels': numbered(labels, remove)}
    if kind == 'paste' and rows is not None:
        spans = word_spans(host_ink, host.shape[0])
        cuts = [0, host.shape[1]]+[(a[1]+b[0])//2 for a, b in zip(spans, spans[1:])]
        for donor in rng.sample(list(donors), len(donors)):
            made = struck_segment(np.asarray(donor, np.uint8), host, rng, style, rng.random() < .4, bank, gold_share, blots)
            if made is None:
                continue
            piece, opacity, room, plain_piece, used = made
            cut, pad, core = rng.choice(cuts), rng.randint(3, 12), piece.shape[1]-2*room
            height, width = max(host.shape[0], piece.shape[0]), host.shape[1]+core+2*pad
            canvas = np.full((height, width, 3), 255, np.uint8)
            top = (height-host.shape[0])//2
            canvas[top:top+host.shape[0], :cut] = host[:, :cut]
            canvas[top:top+host.shape[0], cut+core+2*pad:] = host[:, cut:]
            moved = np.zeros((height, width), np.uint8)
            moved[top:top+host.shape[0], :cut] = labels[:, :cut]
            moved[top:top+host.shape[0], cut+core+2*pad:] = labels[:, cut:]
            top_piece = int(np.clip((height-piece.shape[0])//2+rng.uniform(-.1, .1)*host.shape[0], 0, height-piece.shape[0]))
            plain = canvas.copy()
            place(plain, plain_piece, opacity >= MASK_ALPHA, cut+pad-room, top_piece)
            mask = place(canvas, piece, opacity >= MASK_ALPHA, cut+pad-room, top_piece)
            remove = removal(placed(canvas.shape, opacity >= REMOVE_ALPHA, cut+pad-room, top_piece), plain, used)
            return {'image': Image.fromarray(canvas), 'mask': mask, 'boxes': boxes_of(mask), 'kind': 'paste', 'pasted': 1,
                    'remove': remove, 'style': used, 'labels': numbered(moved, remove)}
    if kind == 'stack' and rows is not None:
        spans = word_spans(host_ink, host.shape[0])
        for donor in rng.sample(list(donors), len(donors)):
            made = struck_segment(np.asarray(donor, np.uint8), host, rng, style, rng.random() < .4, bank, gold_share, blots)
            if made is None:
                continue
            piece, opacity, room, plain_piece, used = made
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
            moved = np.zeros((height, width), np.uint8)
            moved[top_host:top_host+host.shape[0], pad_left:pad_left+host.shape[1]] = labels
            top_piece = 0 if above else host.shape[0]-shared
            plain = canvas.copy()
            place(plain, plain_piece, opacity >= MASK_ALPHA, left+pad_left, top_piece)
            mask = place(canvas, piece, opacity >= MASK_ALPHA, left+pad_left, top_piece)
            remove = removal(placed(canvas.shape, opacity >= REMOVE_ALPHA, left+pad_left, top_piece), plain, used)
            return {'image': Image.fromarray(canvas), 'mask': mask, 'boxes': boxes_of(mask), 'kind': 'stack', 'pasted': 1,
                    'remove': remove, 'style': used, 'labels': numbered(moved, remove)}
    if kind == 'over' and rows is not None:
        spans = [s for s in word_spans(host_ink, host.shape[0]) if s[1]-s[0] >= 4]
        if spans:
            a, b = rng.choice(spans)
            height = rows[1]-rows[0]
            width = int(min(b-a, max(4, rng.randint(*OVER_CHARACTERS)*height*rng.uniform(.6, 1.1))))
            left = a+rng.randint(0, b-a-width)
            columns = host_ink[:, left:left+width].any(1)
            top, bottom = int(np.flatnonzero(columns)[0]), int(np.flatnonzero(columns)[-1])+1
            style = style or choose_style(rng, bank, gold_share, blots=blots)
            room = int(rng.uniform(0, .3)*height)
            layer, opacity = strokes((width, bottom-top), style, rng, ink_colour(host, host_ink, rng), max(2., rng.uniform(*STROKE)*height),
                                     room, bank, host.shape[0])
            canvas = host.copy()
            mask = place(canvas, layer.round().astype(np.uint8), opacity >= MASK_ALPHA, left-room, top)
            remove = removal(placed(canvas.shape, opacity >= REMOVE_ALPHA, left-room, top), host, style)
            return {'image': Image.fromarray(canvas), 'mask': mask, 'boxes': boxes_of(mask), 'kind': 'over', 'pasted': 0,
                    'remove': remove, 'style': style, 'labels': numbered(labels, remove)}
    return {'image': Image.fromarray(host), 'mask': np.zeros(host.shape[:2], bool), 'boxes': [], 'kind': 'clean', 'pasted': 0,
            'remove': np.zeros(host.shape[:2], bool), 'style': None, 'labels': labels.copy()}


# ---- a strike put on a host's own writing at the host's own size (synthesis plan of 2026-11-16) -------------------------------------
SCALE_JITTER = (.85, 1.15)  # a material is scaled by the ratio of the two lines' heights times this
BOX_MARGIN = .1             # a region's box: the strike and the writing it touches, this many host line heights round them ...
BOX_JITTER = .1             # ... each side then moved by up to this share of the box's size, as a detection box is
LEVEL_JITTER = 10           # the host's ink grey varied by up to this much
HOST_INK = 200              # a host pixel this dark is writing
PLACE_TRIES = 20


SCALE_RANGE = (.5, 2.)      # and never by less or more than this


def writing_height(ink):
    """The height of a piece of writing: its ink rows from the 5th to the 95th percentile of its pixels (a neighbour's stray stroke or a
    fraction's far part does not stretch it)."""
    rows = np.nonzero(ink)[0]
    return float(np.percentile(rows, 95)-np.percentile(rows, 5)+1) if len(rows) else 0.


def host_level(grey, rng):
    """The grey of a host's writing (the median of its darkest third), a little varied."""
    values = np.sort(grey[grey < HOST_INK])
    level = float(np.median(values[:max(1, len(values)//3)])) if len(values) else 40.
    return float(np.clip(level+rng.uniform(-LEVEL_JITTER, LEVEL_JITTER), 0, 200))


def on_writing(page, line_box, alpha, material, rng):
    """A material (``alpha``: its opacity, float [h, w]; ``material``: its ``writing_height`` -- the height of the writing it struck --,
    ``rel_y`` -- the height of its middle on that writing, 0 at its top and 1 at its bottom) put on the writing of a host line
    (``line_box`` on the grey ``page``), scaled by the ratio of the two writings' heights (``SCALE_JITTER``; a line's box is often much
    taller than its writing, by a fraction or a neighbour's stroke), its middle on a column of the host's ink with ink under at
    least half its width, at its own height on that writing, in the host's ink (``draw_pair``'s darkening: deeper where it lies on
    writing).  Returns a dict: ``image`` and ``target`` (the host as it is) cut at ``box`` -- the strike and the writing it touches with
    ``BOX_MARGIN`` round them, each side moved by up to ``BOX_JITTER`` --, ``alpha`` (the strike's opacity there), ``remove`` (its pixels
    at least ``REMOVE_ALPHA`` opaque that are not the host's writing), ``strike_share`` (the removed pixels over them and the writing in
    the box), ``scale``; None when the host line has no writing wide enough."""
    from scipy import ndimage
    from .restoration_pairs import draw_pair
    l, t, r, b = line_box; height = max(4, b-t)
    written = writing_height(page[t:b, l:r] < HOST_INK)      # the height of the host's writing (its line box is often taller)
    if not written:
        return None
    scale = float(np.clip(written/max(1, material['writing_height']), *SCALE_RANGE))*rng.uniform(*SCALE_JITTER)
    image = Image.fromarray(np.rint(alpha*255).astype(np.uint8))
    if rng.random() < .5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = max(2, round(image.width*scale)), max(2, round(image.height*scale))
    a = np.asarray(image.resize((w, h), Image.LANCZOS), np.float32)/255.
    ink = page[t:b, l:r] < HOST_INK
    columns = ink.any(0)
    if columns.sum() < 4 or w >= page.shape[1]:
        return None
    for _ in range(PLACE_TRIES):
        cx = l+int(rng.choice(np.flatnonzero(columns)))
        x0 = cx-w//2
        span = columns[max(0, x0-l):max(0, x0-l+w)]
        if len(span) and span.mean() >= .5:
            break
    else:
        return None
    rows = np.flatnonzero(ink[:, max(0, x0-l):max(0, x0-l+w)].any(1))
    top, bottom = t+int(rows[0]), t+int(rows[-1])+1
    cy = top+material.get('rel_y', .5)*(bottom-top)
    y0 = int(round(cy-h/2))
    full = np.zeros(page.shape, np.float32)
    px0, py0, px1, py1 = max(0, x0), max(0, y0), min(page.shape[1], x0+w), min(page.shape[0], y0+h)
    if px1 <= px0 or py1 <= py0:
        return None
    full[py0:py1, px0:px1] = a[py0-y0:py1-y0, px0-x0:px1-x0]
    struck = full >= REMOVE_ALPHA
    near = ndimage.binary_dilation(struck, iterations=2)
    window = np.zeros(page.shape, bool); window[t:b, :] = True
    writing = (page < HOST_INK) & window
    labels, _ = ndimage.label(writing, np.ones((3, 3)))
    touched = np.isin(labels, np.unique(labels[near & writing])) & writing
    ys, xs = np.nonzero(struck | touched)
    margin = BOX_MARGIN*height; box = [xs.min()-margin, ys.min()-margin, xs.max()+1+margin, ys.max()+1+margin]
    bw, bh = box[2]-box[0], box[3]-box[1]
    box = [box[0]+rng.uniform(-1, 1)*BOX_JITTER*bw, box[1]+rng.uniform(-1, 1)*BOX_JITTER*bh,
           box[2]+rng.uniform(-1, 1)*BOX_JITTER*bw, box[3]+rng.uniform(-1, 1)*BOX_JITTER*bh]
    L, T = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    R, B = min(page.shape[1], int(round(box[2]))), min(page.shape[0], int(round(box[3])))
    if R-L < 4 or B-T < 4:
        return None
    clean = page[T:B, L:R]
    level = host_level(page[t:b, l:r], rng)
    struck_image, target = draw_pair(clean, full[T:B, L:R], level)
    remove = (full[T:B, L:R] >= REMOVE_ALPHA) & (clean >= WRITING_BELOW)
    written = int((clean < INK_BELOW).sum())
    return {'image': struck_image, 'target': target, 'alpha': full[T:B, L:R], 'remove': remove, 'box': [L, T, R, B],
            'strike_share': float(remove.sum())/max(1, remove.sum()+written), 'scale': float(scale), 'level': level}
