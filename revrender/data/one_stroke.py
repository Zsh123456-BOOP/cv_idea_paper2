"""A strike as single strokes, repaired, with the place it had on the writing it struck; and put on another writer's writing in the
same place (synthesis plan of 2026-11-17, "按一笔来": it replaces the 2026-11-16 cleaning and pasting).

Units.  A region's strike (the annotator's erased pixels) is cut into strokes: its straight parts (``strike_bank.straight_parts``; a
dashed line, broken where it lay on writing, is one line) taken run by run along each, two runs one stroke where the gap between them
lies on writing or is short (``GAP`` widths); strokes that continue one another (``JOIN`` degrees) are joined (a bowed line).  Strokes
on the same stretch of writing (overlapping across ``OVERLAP`` of the shorter, at about the same height) or touching (a cross, a zigzag)
are one ``group``; a stroke alone is a ``line``.  A region whose strike lies off its strokes for more than ``LOOSE`` of it, or a group
of more than ``MAX_STROKES`` strokes or that fills its box (``DENSE``), is a ``blob`` (a scribble or a smear over writing whose strokes
and writing cannot be told apart): not cut and not straightened, only closed as before (``strike_bank.human_material``).  A bank line
of paper 1 is one line (or several, where it breaks on paper).

Repair (``ribbon``).  A line is straightened along its centre line (a parabola along its main axis fitted to its pixels), a section
across it every pixel.  A section is hidden where writing (ink that is not the strike) lies within ``NEAR`` pixels of its half width,
where the stroke is more than ``WIDE`` times its median width (writing taken with it), darker than its own ink by ``DARKER`` (a
crossing) or without ink over a gap of at most ``GAP`` widths; a longer gap on paper splits it.  Hidden sections are replaced by the
stroke's own clean sections on both sides (interpolated across), so a line is whole and carries no writing; a stroke with less than
``CLEAN`` of its sections clean is left out (nothing to interpolate from).  Put back (``repaired``) a stroke is the scan's own pixels
where its sections were clean and the interpolated ribbon where they were hidden.  A bank line is first found again on its page
(``regrown``: paper 1's band often lies a few pixels off the stroke) and the ink where writing crosses it set apart (``crossed``).

Place (``struck``).  The writing a unit struck: the pieces of writing it touches and those under its length at the height of its
body; characters are those pieces merged where they overlap across (``MERGE``); the body is their rows from the 10th to the 90th
percentile (``BODY``) across the direction of the writing (fitted to the centres of the pieces of the row, ``TILT`` at most).  A
``frame``: that direction, the corner of the body's left and top (``anchor``), its height, its span, the count of characters.
``measures`` of a strike on a frame: the share of its length with writing under it, its mean height where it does (in body heights
from the top) and the share of its pixels on writing.

Put (``put``).  On a host line its characters are found the same way; a run of them as many (or nearly) and as wide in body heights
(to ``MATCH``) is struck; the unit is carried by the map that sends its frame to theirs: scaled by the ratio of the bodies, stretched
along the writing by the ratio of the spans, turned by the difference of the directions (so its angle to the writing, its heights on
it and its overhangs are kept; never mirrored).  It must then be ``PEN`` times the host's pen wide, its measures as at its origin
(``ACCEPT``), and its box free of printed or ruled lines (the page's template, ``ruled``); else ``None`` and why.  Widths on both
sides are ``pen_width``'s.
"""
import numpy as np
from scipy import ndimage

from ..eval.ink import INK_BELOW
from .strike_bank import INK_ALPHA, MIN_WIDTH, completed, holes, ink_level, opacity, straight_parts

GAP = 2.              # a stroke's gap of at most this many widths is closed; a longer one on paper splits it
REACH = 60            # a gap on writing is closed up to this many pixels
JOIN = 15.            # two strokes continuing one another within this many degrees are one
MAX_PARTS = 10        # straight parts looked for in one strike
OVERLAP = .3          # strokes overlapping along the writing across this share of the shorter, at about one height, are a group
LOOSE = .25           # a strike with more than this share of its pixels off its strokes is a blob
MAX_STROKES = 6       # and a group of more strokes
DENSE = .45           # or one whose strokes fill this share of their box (at least ``DENSE_SIDE`` widths each way)
DENSE_SIDE = 4.
NEAR = 1.             # writing within this many pixels of a stroke's half width hides that section
WIDE = 1.4            # a section wider than this many times the stroke's median width (and a pixel) carries writing
DARKER = 30           # or darker than the stroke's own ink by this many grey levels (a crossing)
CLEAN = .3            # a stroke with less of its length clean is left out
ONE_LINE = .85        # a line holds at least this share of its pixels within its ribbon
SPECK = 4             # writing pieces smaller than this are dust
MERGE = .3            # pieces overlapping along the writing across this share of the narrower are one character
BODY = (10, 90)       # the body of writing: these percentiles of its rows across the writing
TILT = 15.            # the writing's direction is at most this many degrees off the horizontal
LONG_PIECE = 2.5      # a piece of struck writing this many body heights long ...
THIN = 5.             # ... and this many times longer than thick is another strike
SMEAR = .45           # struck writing filling this share of its body's box is a smear
TINY = .5             # a unit less long or high than this many body heights is left out
MATCH = .2            # a host's struck run is as wide as the origin's in body heights to this share
PEN = (.7, 1.5)       # the carried stroke is this many times the host's pen wide
ACCEPT = {'under': .15, 'height': .15, 'press': .10, 'press_ratio': (.5, 2.)}
PRINTED = .02         # a box with more of its ink on the page's printed template is printed
RULED = .6            # or with a band of three rows inked in one run across this share of its columns (a printed line) ...
DASHES = 8            # ... or across half of them in this many runs (a dashed one)
BOX_MARGIN = .1       # a region's box: the strike and the writing it touches with this many body heights round ...
BOX_JITTER = .1       # ... each side moved by up to this share of the box's size, as a detection box is


# ---- the writing --------------------------------------------------------------------------------------------------------------
def pieces(writing):
    """{label: (ys, xs)} of the 8-connected pieces of writing of at least ``SPECK`` pixels, and the label image."""
    labels, n = ndimage.label(writing, np.ones((3, 3)))
    out = {}
    for k, box in enumerate(ndimage.find_objects(labels), 1):
        ys, xs = np.nonzero(labels[box] == k)
        if len(ys) >= SPECK:
            out[k] = (ys+box[0].start, xs+box[1].start)
    return out, labels


def across(ys, xs, theta):
    """(along, across) of pixels in the writing's direction ``theta`` (y down: across grows downward)."""
    c, s = np.cos(theta), np.sin(theta)
    return xs*c+ys*s, -xs*s+ys*c


def body(points, theta):
    """(top, height) across ``theta`` of pixels (ys, xs): ``BODY`` percentiles of their rows."""
    _, v = across(points[0], points[1], theta)
    top, bottom = np.percentile(v, BODY)
    return float(top), float(max(3., bottom-top))


def tilt(found, keys):
    """The direction of a row of pieces: the line through their centres (weighted by size), ``TILT`` at most; level for fewer than three."""
    if len(keys) < 3:
        return 0.
    cy = np.array([found[k][0].mean() for k in keys]); cx = np.array([found[k][1].mean() for k in keys])
    if np.ptp(cx) < 1:
        return 0.
    slope = np.polyfit(cx, cy, 1, w=np.sqrt([len(found[k][0]) for k in keys]))[0]
    return float(np.clip(np.arctan(slope), -np.deg2rad(TILT), np.deg2rad(TILT)))


def joined(found, keys):
    return (np.concatenate([found[k][0] for k in keys]), np.concatenate([found[k][1] for k in keys]))


def characters(found, keys, theta):
    """Pieces merged into characters where they overlap along the writing: [(left, right, [keys])] in order."""
    spans = sorted((across(*found[k], theta)[0].min(), across(*found[k], theta)[0].max(), k) for k in keys)
    out = []
    for lo, hi, k in spans:
        if out and min(hi, out[-1][1])-lo >= MERGE*max(1., min(hi-lo, out[-1][1]-out[-1][0])):
            out[-1] = (out[-1][0], max(hi, out[-1][1]), out[-1][2]+[k])
        else:
            out.append((lo, hi, [k]))
    return out


def frame_of(found, keys, theta):
    """The frame of struck writing (``keys`` of ``found``): direction, anchor (x, y), height, span, characters."""
    top, height = body(joined(found, keys), theta)
    u = across(*joined(found, keys), theta)[0]; left = float(u.min())
    c, s = np.cos(theta), np.sin(theta)
    return {'theta': float(theta), 'anchor': [left*c-top*s, left*s+top*c], 'height': height, 'span': float(u.max()-left),
            'chars': len(characters(found, keys, theta))}


def in_frame(ys, xs, frame):
    """(along, across) of pixels in a frame, in its body heights from its anchor."""
    u, v = across(ys-frame['anchor'][1], xs-frame['anchor'][0], frame['theta'])
    return u/frame['height'], v/frame['height']


def struck(writing, strike):
    """The frame of the writing a strike (bool) lies on, and the keys of its pieces; None when it lies on none."""
    found, labels = pieces(writing)
    if not found:
        return None
    near = ndimage.binary_dilation(strike, iterations=2)
    touched = {int(k) for k in np.unique(labels[near & (labels > 0)]) if int(k) in found}
    sy, sx = np.nonzero(strike)
    theta = 0.
    for _ in range(2):
        su = across(sy, sx, theta)[0]; lo, hi = su.min(), su.max()
        def under(k):
            u = across(*found[k], theta)[0]
            return min(hi, u.max())-max(lo, u.min()) >= .5*max(1., np.ptp(u))
        chosen = set(touched)
        if not chosen:                         # a stroke over the writing without touching it: the pieces under it nearest in height
            below = [k for k in found if under(k)]
            if not below:
                return None
            sv = across(sy, sx, theta)[1].mean(); tall = np.median([np.ptp(across(*found[k], theta)[1])+1 for k in below])
            chosen = {k for k in below if abs(across(*found[k], theta)[1].mean()-sv) <= tall}
            if not chosen:
                return None
        top, height = body(joined(found, chosen), theta)
        for k in set(found)-chosen:
            v = across(*found[k], theta)[1]
            if under(k) and ((v >= top) & (v <= top+height)).mean() >= .5:
                chosen.add(k)
        top, height = body(joined(found, chosen), theta)
        row = [k for k in found if ((lambda v: ((v >= top-.25*height) & (v <= top+1.25*height)).mean())(across(*found[k], theta)[1])) >= .5]
        theta = tilt(found, row)
    return frame_of(found, sorted(chosen), theta), sorted(chosen)


def crowded(writing, keys, frame):
    """Why the writing a unit struck is not plain writing, or None: a piece of it is a line longer than ``LONG_PIECE`` body heights
    and ``THIN`` times longer than thick (another strike: paper 1's bank holds one stroke of a mark), or it fills ``SMEAR`` of its body's
    box (a smear or scribble round the line)."""
    found, _ = pieces(writing)
    for k in keys:
        if k not in found:
            continue
        ys, xs = found[k]; u, v = in_frame(ys, xs, frame)
        if len(ys) > 6:
            sd = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(np.stack([u, v]))), 1e-9))
            if np.ptp(u) >= LONG_PIECE and sd[1]/sd[0] >= THIN:
                return 'another_strike'
    ys, xs = joined(found, [k for k in keys if k in found]); u, v = in_frame(ys, xs, frame)
    inside = (v >= 0) & (v <= 1)
    area = max(1., frame['span']*frame['height'])
    return 'smear' if inside.sum()/area >= SMEAR else None


def measures(alpha, writing, frame):
    """Where a strike lies on writing: ``under`` the share of its length (pixel steps along the writing) with writing under it within
    the body, ``height`` its mean height there (body heights from the top), ``press`` the share of its pixels on writing."""
    on = alpha >= INK_ALPHA
    ys, xs = np.nonzero(on)
    if not len(ys):
        return {'under': 0., 'height': None, 'press': 0.}
    u, v = in_frame(ys, xs, frame); step = np.floor(u*frame['height']).astype(int)
    wy, wx = np.nonzero(writing); wu, wv = in_frame(wy, wx, frame)
    covered = np.unique(np.floor(wu[(wv >= -.1) & (wv <= 1.1)]*frame['height']).astype(int))
    steps, where = np.unique(step, return_inverse=True)
    mean_v = np.bincount(where, v)/np.bincount(where)
    hit = np.isin(steps, covered)
    return {'under': float(hit.mean()), 'height': float(mean_v[hit].mean()) if hit.any() else None,
            'press': float((on & writing).sum())/len(ys)}


def agrees(here, there):
    """Whether a strike's measures on a host are those at its origin (``ACCEPT``); and which one is not."""
    if abs(here['under']-there['under']) > ACCEPT['under']:
        return False, 'under'
    if (here['height'] is None) != (there['height'] is None) or (here['height'] is not None and abs(here['height']-there['height']) > ACCEPT['height']):
        return False, 'height'
    ratio = here['press']/there['press'] if there['press'] > 0 else (1. if here['press'] == 0 else np.inf)
    if abs(here['press']-there['press']) > ACCEPT['press'] and not ACCEPT['press_ratio'][0] <= ratio <= ACCEPT['press_ratio'][1]:
        return False, 'press'
    return True, None


# ---- strokes ------------------------------------------------------------------------------------------------------------------
def writing_of(grey, strike):
    """The writing round a strike: ink that is not the strike, without the pieces lying wholly along its edge (its faint fringe)."""
    rest = (grey < INK_BELOW) & ~strike
    labels, _ = ndimage.label(rest, np.ones((3, 3)))
    keep = np.unique(labels[rest & ~ndimage.binary_dilation(strike)])
    return np.isin(labels, keep[keep > 0])


def pen_width(mask):
    """The width of the strokes of a mask: twice its area over its edge pixels (exact for a straight stroke two or more pixels wide,
    and not stuck to whole pixels)."""
    edge = mask & ~ndimage.binary_erosion(mask)
    return max(MIN_WIDTH, 2*float(mask.sum())/edge.sum()) if edge.any() else MIN_WIDTH


def inked_share(ink, x0, y0, x1, y1, half):
    """The share of the steps on the segment (x0, y0)-(x1, y1) with ink within ``half`` across it."""
    n = int(max(abs(x1-x0), abs(y1-y0)))+1; t = np.linspace(0, 1, n)
    d = np.array([x1-x0, y1-y0], float); d /= max(1e-6, np.hypot(*d)); normal = np.array([-d[1], d[0]])
    hit = np.zeros(n, bool)
    for off in np.arange(-half, half+1):
        x = np.rint(x0+t*(x1-x0)+off*normal[0]).astype(int); y = np.rint(y0+t*(y1-y0)+off*normal[1]).astype(int)
        inside = (x >= 0) & (x < ink.shape[1]) & (y >= 0) & (y < ink.shape[0])
        hit[inside] |= ink[y[inside], x[inside]]
    return float(hit.mean())


def ends(mask):
    """(direction (x, y), first end, last end) of a stroke's pixels along its main axis."""
    ys, xs = np.nonzero(mask); pts = np.stack([xs, ys], 1).astype(float); mean = pts.mean(0)
    vectors = np.linalg.eigh(np.cov((pts-mean).T))[1]; a = vectors[:, 1]
    a = a if a[0] > 0 or (a[0] == 0 and a[1] > 0) else -a
    t = (pts-mean) @ a
    return a, pts[np.argmin(t)], pts[np.argmax(t)]


def strokes_of(grey, strike, width):
    """(stroke masks, the strike's pixels on none of them) of a human strike: its straight parts run by run, gaps on writing or short
    closed, strokes continuing one another joined."""
    ink = grey < INK_BELOW; free = strike.copy(); found = []
    ys, xs = np.nonzero(strike)
    for theta, rho in straight_parts(ys, xs, max(3, int(width)), max(6, 3*width), MAX_PARTS):
        fy, fx = np.nonzero(free)
        on = np.abs(fx*np.cos(theta)+fy*np.sin(theta)-rho) <= width/2+1.5
        if on.sum() < 6:
            continue
        py, px = fy[on], fx[on]; along = -px*np.sin(theta)+py*np.cos(theta); order = np.argsort(along, kind='stable')
        py, px, along = py[order], px[order], along[order]
        cuts = np.flatnonzero(np.diff(along) > 1.5)
        runs = np.split(np.arange(len(along)), cuts+1)
        merged = [runs[0]]
        for run in runs[1:]:
            a0, a1 = along[merged[-1][-1]], along[run[0]]; gap = a1-a0
            point = lambda a: (rho*np.cos(theta)-a*np.sin(theta), rho*np.sin(theta)+a*np.cos(theta))
            (x0, y0), (x1, y1) = point(a0), point(a1)
            if gap <= GAP*width or (gap <= REACH and inked_share(ink, x0, y0, x1, y1, int(width//2)) >= .6):
                merged[-1] = np.concatenate([merged[-1], run])
            else:
                merged.append(run)
        for run in merged:
            if along[run[-1]]-along[run[0]] >= 3*width:
                mask = np.zeros_like(strike); mask[py[run], px[run]] = True
                free[py[run], px[run]] = False; found.append(mask)
    changed = True
    while changed:                                # a bowed line found as two straight parts, or two dashes of one line
        changed = False
        for i in range(len(found)):
            for j in range(i+1, len(found)):
                ai, si, ei = ends(found[i]); aj, sj, ej = ends(found[j])
                if np.degrees(np.arccos(min(1., abs(float(ai @ aj))))) > JOIN:
                    continue
                pairs = [(np.hypot(*(p-q)), p, q) for p in (si, ei) for q in (sj, ej)]
                gap, p, q = min(pairs, key=lambda x: x[0])
                link = (q-p)/max(1e-6, gap)
                straight = gap <= 3*width or (np.degrees(np.arccos(min(1., abs(float(link @ ai))))) <= JOIN+5
                                              and np.degrees(np.arccos(min(1., abs(float(link @ aj))))) <= JOIN+5)
                if straight and (gap <= GAP*width or (gap <= REACH and inked_share(ink, *p, *q, int(width//2)) >= .6)):
                    found[i] = found[i] | found[j]; del found[j]; changed = True; break
            if changed:
                break
    return found, free


def groups_of(found, width, tall):
    """Strokes in groups: touching, or over one stretch of writing at about one height (``tall``: the writing's height)."""
    n = len(found); parent = list(range(n))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    spans = []
    for m in found:
        ys, xs = np.nonzero(m); spans.append((xs.min(), xs.max(), ys.mean()))
    for i in range(n):
        near = ndimage.binary_dilation(found[i], iterations=max(1, int(round(2*width))))
        for j in range(i+1, n):
            (l1, r1, y1), (l2, r2, y2) = spans[i], spans[j]
            overlap = min(r1, r2)-max(l1, l2) >= OVERLAP*max(1, min(r1-l1, r2-l2)) and abs(y1-y2) <= .8*tall
            if overlap or (near & found[j]).any():
                parent[root(j)] = root(i)
    out = {}
    for i in range(n):
        out.setdefault(root(i), []).append(found[i])
    return list(out.values())


def ribbon(grey, own, strike, level, width=None):
    """A stroke (``own``) straightened and repaired (see the module); ``strike`` is all of the region's strike (not writing).
    Returns (segments, reasons left out): a segment is the ribbon (sections x offsets across) and how to put it back."""
    ink = grey < INK_BELOW
    writing = ink & ~ndimage.binary_dilation(strike)
    alpha = opacity(grey, level, own)
    strike = strike & ~ndimage.binary_dilation(own)
    ys, xs = np.nonzero(own)
    if len(ys) < 6:
        return [], ['too_few_pixels']
    width = width or pen_width(own)
    pts = np.stack([xs, ys], 1).astype(float); mean = pts.mean(0)
    a = np.linalg.eigh(np.cov((pts-mean).T))[1][:, 1]; a = a if a[0] > 0 or (a[0] == 0 and a[1] > 0) else -a
    c = np.array([-a[1], a[0]])
    t, s = (pts-mean) @ a, (pts-mean) @ c
    deg = 2 if np.ptp(t) > 12*width else 1
    coef = np.polyfit(t, s, deg)
    keep = np.abs(s-np.polyval(coef, t)) <= 2*width+1
    if keep.sum() > 6:
        coef = np.polyfit(t[keep], s[keep], deg)
    K = int(np.ceil(width/2+NEAR+1))
    if (np.abs(s-np.polyval(coef, t)) <= K).mean() < ONE_LINE:
        return [], ['not_one_line']
    ts = np.arange(np.floor(t.min()), np.ceil(t.max())+1); us = np.arange(-K, K+1)
    centre = np.polyval(coef, ts)
    X = mean[0]+ts[:, None]*a[0]+(centre[:, None]+us[None, :])*c[0]; Y = mean[1]+ts[:, None]*a[1]+(centre[:, None]+us[None, :])*c[1]
    sample = lambda image, order, fill: ndimage.map_coordinates(image.astype(np.float32), [Y, X], order=order, cval=fill)
    R = sample(alpha, 1, 0.); O = sample(own, 0, 0.) > .5; G = sample(grey, 1, 255.)
    W = sample(writing | (strike & ~own), 0, 0.) > .5                          # writing, or another stroke crossing this one
    core = np.abs(us) <= width/2+NEAR
    hidden = W[:, core].any(1)
    count = (R >= INK_ALPHA).sum(1); empty = count == 0
    seen = ~hidden & ~empty
    if seen.sum() < 3:
        return [], ['mostly_hidden']
    median = float(np.median(count[seen]))
    wide = count > WIDE*median+1
    darkest = np.where(O, G, 255.).min(1)
    dark = darkest < float(np.median(darkest[seen & ~wide] if (seen & ~wide).any() else darkest[seen]))-DARKER
    bad = hidden | wide | dark | empty
    paper = empty & ~hidden
    breaks = np.zeros(len(ts), bool)                 # gaps on paper longer than GAP widths split the stroke
    edges = np.flatnonzero(np.diff(np.concatenate([[0], paper.astype(int), [0]])))
    for lo, hi in zip(edges[::2], edges[1::2]):
        if hi-lo > GAP*width:
            breaks[lo:hi] = True
    out, why = [], []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (~breaks).astype(int), [0]])))
    for lo, hi in zip(edges[::2], edges[1::2]):
        if hi-lo < 3*width:
            why.append('short'); continue
        clean = ~bad[lo:hi]
        if clean.mean() < CLEAN:
            why.append('mostly_hidden'); continue
        repaired = R[lo:hi].copy(); steps = ts[lo:hi]
        for j in range(len(us)):
            repaired[~clean, j] = np.interp(steps[~clean], steps[clean], repaired[clean, j])
        out.append({'mean': mean, 'a': a, 'c': c, 'coef': coef, 't0': float(steps[0]), 'K': K, 'ribbon': repaired, 'raw': R[lo:hi],
                    'clean': float(clean.mean()), 'bad': ~clean, 'width': median})
    return out, why


SHIFT = 8             # a bank piece's stroke is looked for up to this many pixels off it
REGROWN = 1.5         # and is the ink within this many pixels of its half width there


def regrown(grey, piece):
    """A bank piece's whole stroke on its page: paper 1's gold band, and so the piece cut along it, often lies a few pixels off the
    stroke.  The piece's centre line (fitted) is moved across by up to ``SHIFT`` pixels to where the most of its steps have ink; the
    stroke is the ink within its half width (and a pixel) of it there, its centre fitted again on that ink."""
    ink = grey < INK_BELOW
    ys, xs = np.nonzero(piece)
    if len(ys) < 6:
        return piece
    width = pen_width(piece); pts = np.stack([xs, ys], 1).astype(float); mean = pts.mean(0)
    a = np.linalg.eigh(np.cov((pts-mean).T))[1][:, 1]; c = np.array([-a[1], a[0]])
    t, s = (pts-mean) @ a, (pts-mean) @ c
    coef = np.polyfit(t, s, 2 if np.ptp(t) > 12*width else 1)
    ts = np.arange(t.min(), t.max()+1); centre = np.polyval(coef, ts)
    def inked(d):
        x = np.rint(mean[0]+ts*a[0]+(centre+d)*c[0]).astype(int); y = np.rint(mean[1]+ts*a[1]+(centre+d)*c[1]).astype(int)
        inside = (x >= 0) & (x < grey.shape[1]) & (y >= 0) & (y < grey.shape[0])
        return ink[y[inside], x[inside]].sum()-.01*abs(d)
    shift = max(range(-SHIFT, SHIFT+1), key=inked)
    gy, gx = np.mgrid[:grey.shape[0], :grey.shape[1]]; dx, dy = gx-mean[0], gy-mean[1]; tt = dx*a[0]+dy*a[1]
    across_ = dx*c[0]+dy*c[1]-np.polyval(coef, tt)-shift
    return ink & (np.abs(across_) <= width/2+REGROWN) & (tt >= t.min()-1) & (tt <= t.max()+1)


def crossed(grey, own):
    """The pixels of a regrown bank line where writing crosses it: writing (``writing_of``) lies past its edge on both sides along its
    normal, within its width and ``REGROWN`` and two pixels (that ink is the writing's as much as the line's)."""
    writing = writing_of(grey, own); ys, xs = np.nonzero(own)
    if len(ys) < 6:
        return np.zeros_like(own)
    pts = np.stack([xs, ys], 1).astype(float); a = np.linalg.eigh(np.cov((pts-pts.mean(0)).T))[1][:, 1]; c = np.array([-a[1], a[0]])
    reach = int(np.ceil(pen_width(own)+REGROWN+2)); h, w = own.shape
    def seen(sign):
        hit = np.zeros(len(ys), bool)
        for d in range(1, reach+1):
            x = np.rint(xs+sign*d*c[0]).astype(int); y = np.rint(ys+sign*d*c[1]).astype(int)
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h); hit[inside] |= writing[y[inside], x[inside]]
        return hit
    out = np.zeros_like(own); both = seen(1) & seen(-1); out[ys[both], xs[both]] = True
    return out


def render(segment, shape, zone=False):
    """A ribbon put back where it came from: its opacity on an image of ``shape`` (with ``zone``, also where it was repaired: the
    pixels of its hidden sections)."""
    R, K, t0 = segment['ribbon'], segment['K'], segment['t0']; a, c, mean, coef = segment['a'], segment['c'], segment['mean'], segment['coef']
    ts = t0+np.array([0, len(R)-1]); corners = []
    for t in ts:
        for u in (-K, K):
            corners.append(mean+t*a+(np.polyval(coef, t)+u)*c)
    sag = abs(np.polyval(coef, (ts[0]+ts[1])/2)-(np.polyval(coef, ts[0])+np.polyval(coef, ts[1]))/2)
    corners = np.array(corners); lo = np.floor(corners.min(0)-sag-2).astype(int); hi = np.ceil(corners.max(0)+sag+3).astype(int)
    x0, y0 = max(0, lo[0]), max(0, lo[1]); x1, y1 = min(shape[1], hi[0]), min(shape[0], hi[1])
    out = np.zeros(shape, np.float32); fixed = np.zeros(shape, bool)
    if x1 > x0 and y1 > y0:
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(float)
        dx, dy = gx-mean[0], gy-mean[1]; t = dx*a[0]+dy*a[1]; u = dx*c[0]+dy*c[1]-np.polyval(coef, t)
        out[y0:y1, x0:x1] = ndimage.map_coordinates(R, [t-t0, u+K], order=1, cval=0.)
        k = np.rint(t-t0).astype(int); inside = (k >= 0) & (k < len(R)) & (np.abs(u) <= K)
        fixed[y0:y1, x0:x1] = inside & segment['bad'][np.clip(k, 0, len(R)-1)]
    return (out, fixed) if zone else out


def repaired(segments, alpha, shape):
    """Strokes put back: the scan's own opacity (``alpha``) where their sections were clean, the repaired ribbons where hidden."""
    drawn = [render(g, shape, zone=True) for g in segments]
    out = np.max([d[0] for d in drawn], 0); fixed = np.any([d[1] for d in drawn], 0)
    return np.where(fixed, np.maximum(out, 0), np.where(out > 0, alpha, 0.)).astype(np.float32)


def blob(grey, mask):
    """(before, after, level, holes) of a blob: its pixels as they are, and closed as before (``strike_bank.completed``)."""
    level = ink_level(grey, mask); width = pen_width(mask)
    after, _ = completed(grey, mask, width)
    return opacity(grey, level, mask), opacity(grey, level, after), level, holes(grey, after, width)


def units_of(grey, strike, level):
    """A human strike cut into units: [{'kind', 'repaired', 'before', 'reference', 'segments', 'width', 'holes'}] and the reasons strokes
    were left out (``strike`` bool on ``grey``)."""
    width = pen_width(strike); why = []
    found, free = strokes_of(grey, strike, width)
    writing_pieces, _ = pieces((grey < INK_BELOW) & ~ndimage.binary_dilation(strike))
    tall = float(np.median([np.ptp(ys)+1 for ys, _ in writing_pieces.values()])) if writing_pieces else 4*width
    def as_blob(mask):
        before, after, lvl, holed = blob(grey, mask)
        return {'kind': 'blob', 'repaired': after, 'before': before, 'reference': mask, 'segments': [], 'width': pen_width(mask), 'holes': holed}
    if not found or free.sum() > LOOSE*strike.sum():
        return [as_blob(strike)], why
    out = []
    for group in groups_of(found, width, tall):
        union = np.any(group, 0); ys, xs = np.nonzero(union)
        h, w = np.ptp(ys)+1, np.ptp(xs)+1
        if len(group) > MAX_STROKES or (min(h, w) >= DENSE_SIDE*width and union.sum() >= DENSE*h*w):
            out.append(as_blob(union)); continue
        segments, lost = [], False
        for own in group:
            made, reasons = ribbon(grey, own, strike, level, width)
            segments += made; why += reasons; lost |= not made
        if not segments:
            continue
        why += ['group_partly_kept'] if lost else []      # what is left of it is a line or a smaller group (counted)
        before = opacity(grey, level, union)
        out.append({'kind': 'line' if len(segments) == 1 else 'group', 'repaired': repaired(segments, before, grey.shape), 'before': before,
                    'reference': union, 'segments': segments, 'width': float(np.median([s['width'] for s in segments])), 'holes': 0.})
    return out, why


# ---- carried to a host ----------------------------------------------------------------------------------------------------------
def host_row(grey, line_box):
    """A host line's writing: its pieces, direction and characters (``line_box`` on ``grey``, a window round it); None with too few."""
    ink = grey < INK_BELOW
    found, labels = pieces(ink)
    l, t, r, b = line_box
    mine = [k for k, (ys, xs) in found.items() if ((ys >= t) & (ys < b)).mean() >= .5 and ((xs >= l) & (xs < r)).mean() >= .5]
    if len(mine) < 2:
        return None
    theta = 0.
    for _ in range(2):
        top, height = body(joined(found, mine), theta)
        row = [k for k in mine if ((lambda v: ((v >= top-.25*height) & (v <= top+1.25*height)).mean())(across(*found[k], theta)[1])) >= .5]
        if len(row) < 2:
            return None
        theta = tilt(found, row)
    chars = characters(found, row, theta)
    pen = pen_width(ink & np.isin(labels, row))
    return {'found': found, 'labels': labels, 'theta': theta, 'chars': chars, 'pen': pen, 'ink': ink}


def target(row, frame, rng):
    """A run of a host's characters to strike like ``frame``'s: as many (nearest) and as wide in body heights (to ``MATCH``); its frame."""
    found, chars, theta = row['found'], row['chars'], row['theta']
    want = frame['span']/frame['height']; n = frame['chars']; good = []
    for i in range(len(chars)):
        for j in range(i, min(len(chars), i+n+3)):
            keys = [k for c in chars[i:j+1] for k in c[2]]
            top, height = body(joined(found, keys), theta)
            ratio = (chars[j][1]-chars[i][0])/height/max(1e-6, want)
            if abs(ratio-1) <= MATCH:
                good.append((abs(j-i+1-n), i, j, keys))
    if not good:
        return None
    best = min(g[0] for g in good)
    _, i, j, keys = rng.choice([g for g in good if g[0] == best])
    span = across(*joined(found, keys), theta)[0]
    run_theta = tilt(found, keys) if np.ptp(span) >= 3*body(joined(found, keys), theta)[1] and len(keys) >= 3 else theta
    return frame_of(found, keys, run_theta), keys


def carry(origin, here):
    """The map (2 x 2, origin's anchor, here's anchor) sending a frame to another: scaled by their heights, stretched along by their
    spans, turned by their directions."""
    k = here['height']/origin['height']; stretch = (here['span']/here['height'])/max(1e-6, origin['span']/origin['height'])
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    A = rot(here['theta']) @ np.diag([stretch*k, k]) @ rot(-origin['theta'])
    return A, np.array(origin['anchor'], float), np.array(here['anchor'], float), k, stretch


def warp(alpha, mapping, shape):
    """``alpha`` carried by ``mapping`` (``carry``) onto an image of ``shape``."""
    A, a0, a1 = mapping[:3]
    ys, xs = np.nonzero(alpha > 0)
    out = np.zeros(shape, np.float32)
    if not len(ys):
        return out
    corners = np.array([[x, y] for x in (xs.min()-1, xs.max()+1) for y in (ys.min()-1, ys.max()+1)], float)
    mapped = (A @ (corners-a0).T).T+a1
    x0, y0 = np.maximum(0, np.floor(mapped.min(0)).astype(int)-1); x1, y1 = np.minimum(shape[::-1], np.ceil(mapped.max(0)).astype(int)+2)
    if x1 <= x0 or y1 <= y0:
        return out
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(float)
    back = np.linalg.inv(A) @ np.stack([gx.ravel()-a1[0], gy.ravel()-a1[1]])
    out[y0:y1, x0:x1] = ndimage.map_coordinates(alpha.astype(np.float32), [back[1]+a0[1], back[0]+a0[0]], order=1, cval=0.).reshape(gy.shape)
    return out


def ruled(ink, height):
    """Whether a box holds a printed or ruled line: in a band of three rows, one run of ink across ``RULED`` of its columns and at least
    three body heights, or (dashed) ink across half its columns in ``DASHES`` runs or more; or a column band down all its rows."""
    if ink.shape[0] < 3 or ink.shape[1] < 3:
        return False
    band = ndimage.maximum_filter1d(ink, 3, axis=0)
    for row in band:
        edges = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(int), [0]])))
        runs = edges[1::2]-edges[::2]
        if len(runs) and (runs.max() >= max(RULED*len(row), 3*height) or (row.mean() >= .5 and len(runs) >= DASHES and len(row) >= 4*height)):
            return True
    return bool(ink.shape[0] >= 2*height and ndimage.maximum_filter1d(ink, 3, axis=1).mean(0).max() >= .95)


def put(unit, alpha, host, rng):
    """``alpha`` (the unit's opacity on its origin image) put on a host (``host_row`` of ``host['grey']`` plus its printed ``template``
    window): a dict as ``paste_strikes`` gives (image, target, alpha, remove, box, strike_share) with the map, the run's frame and the
    measures there; or (None, why)."""
    from .paste_strikes import host_level, REMOVE_ALPHA, WRITING_BELOW
    from .restoration_pairs import draw_pair
    row = host['row']; grey = host['grey']
    chosen = target(row, unit['frame'], rng)
    if chosen is None:
        return None, 'no_run_as_wide'
    frame, keys = chosen
    mapping = carry(unit['frame'], frame)
    if not PEN[0] <= unit['width']*mapping[3]/row['pen'] <= PEN[1]:            # both by ``pen_width``
        return None, 'pen'
    full = warp(alpha, mapping, grey.shape)
    if (full >= INK_ALPHA).sum() < 6:
        return None, 'off_the_page'
    here = measures(full, row['ink'], frame)
    ok, why = agrees(here, unit['measures'])
    if not ok:
        return None, why
    strike = full >= REMOVE_ALPHA
    touched = np.isin(row['labels'], np.unique(row['labels'][ndimage.binary_dilation(strike, iterations=2) & row['ink']])) & row['ink']
    ys, xs = np.nonzero(strike | touched)
    margin = BOX_MARGIN*frame['height']; box = [xs.min()-margin, ys.min()-margin, xs.max()+1+margin, ys.max()+1+margin]
    bw, bh = box[2]-box[0], box[3]-box[1]
    box = [box[0]+rng.uniform(-1, 1)*BOX_JITTER*bw, box[1]+rng.uniform(-1, 1)*BOX_JITTER*bh,
           box[2]+rng.uniform(-1, 1)*BOX_JITTER*bw, box[3]+rng.uniform(-1, 1)*BOX_JITTER*bh]
    L, T = max(0, int(round(box[0]))), max(0, int(round(box[1]))); R, B = min(grey.shape[1], int(round(box[2]))), min(grey.shape[0], int(round(box[3])))
    if R-L < 4 or B-T < 4:
        return None, 'box'
    clean = grey[T:B, L:R]; ink = clean < INK_BELOW
    if host.get('template') is not None and (ink & host['template'][T:B, L:R]).sum() > PRINTED*max(1, ink.sum()):
        return None, 'printed'
    if ruled(ink, frame['height']):
        return None, 'ruled'
    level = host_level(grey[row['ink'] & np.isin(row['labels'], keys)], rng)
    image, target_ = draw_pair(clean, full[T:B, L:R], level)
    remove = (full[T:B, L:R] >= REMOVE_ALPHA) & (clean >= WRITING_BELOW)
    written = int((clean < INK_BELOW).sum())
    return {'image': image, 'target': target_, 'alpha': full[T:B, L:R], 'remove': remove, 'box': [L, T, R, B],
            'strike_share': float(remove.sum())/max(1, remove.sum()+written), 'scale': float(mapping[3]), 'stretch': float(mapping[4]),
            'level': level, 'frame': frame, 'measures': here, 'mapping': mapping}, None
