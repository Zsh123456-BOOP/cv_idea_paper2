"""The gold-mask recipe: annotated marks -> raster on the line crop -> NEAREST letterbox onto 96x768 -> pen-width fix.

    gold_mask(marks, width, height) = correct_mask(strip_mask(rasterise(marks, width, height)))

``marks`` are the annotator's boxes and polylines in crop pixels.  The recipe is the one the paper-1 training split and
validation gold v2 share, so a mask drawn on a whole-question crop and mapped to a strip must go through it as well.
"""
import numpy as np
from scipy import ndimage

CANVAS = (96, 768)
MARK_STROKE_PX = 3          # a 3 px median dark run lies under a real revision stroke
THIN_WIDTH_PIXELS = 5.0     # strokes thinner than this (twice the mean distance to the edge) are widened
WIDEN_ITERATIONS = 1


def rasterise(marks, width, height):
    """Draw boxes (filled) and polylines/segments (stroked) onto an L-mode image of the crop's size."""
    from PIL import Image, ImageDraw

    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for mark in marks:
        if mark['shape'] == 'box':
            x0, y0 = float(mark['x']), float(mark['y'])
            draw.rectangle([x0, y0, x0 + float(mark['w']), y0 + float(mark['h'])], fill=255)
        elif mark.get('path') and len(mark['path']) > 1:
            draw.line([tuple(p) for p in mark['path']], fill=255,
                      width=int(mark.get('stroke_width') or MARK_STROKE_PX), joint='curve')
        else:
            draw.line([(float(mark['x1']), float(mark['y1'])), (float(mark['x2']), float(mark['y2']))],
                      fill=255, width=int(mark.get('stroke_width') or MARK_STROKE_PX))
    return mask


def letterbox(image, width, height, fill, mask=False):
    """Scale to fit (aspect kept) and centre on a width x height canvas; NEAREST for masks, BILINEAR for lines."""
    from PIL import Image

    scale = min(width / image.width, height / image.height)
    resized_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    resized = image.resize(resized_size, resample)
    canvas = Image.new('L', (width, height), fill)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    resized.close()
    return canvas


def unletterbox(canvas, width, height):
    """The inverse of ``letterbox`` for a boolean mask: the content area of the canvas resized (NEAREST) back to the crop's size."""
    from PIL import Image

    canvas = np.asarray(canvas, bool)
    scale = min(canvas.shape[1] / width, canvas.shape[0] / height)
    inner_w, inner_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    left, top = (canvas.shape[1] - inner_w) // 2, (canvas.shape[0] - inner_h) // 2
    inner = Image.fromarray(canvas[top:top + inner_h, left:left + inner_w].astype(np.uint8) * 255)
    return np.asarray(inner.resize((width, height), Image.Resampling.NEAREST)) > 127


def strip_mask(raster):
    """The raster letterboxed onto the 96x768 canvas with NEAREST, as a uint8 array."""
    height, width = CANVAS
    return np.asarray(letterbox(raster, width, height, 0, mask=True), np.uint8)


def pen_width(component):
    """Twice the mean distance to the edge: the component's own pen width."""
    if not component.any():
        return 0.0
    return float(2.0*ndimage.distance_transform_edt(component)[component].mean())


def correct_mask(mask):
    """Widen the thin strokes of one gold mask by one pixel; leave blots alone (filled regions already run wide)."""
    structure = np.ones((3, 3), bool)
    binary = np.asarray(mask) > 127
    labels, count = ndimage.label(binary, structure=structure)
    out = np.zeros_like(binary)
    for index in range(1, count + 1):
        component = labels == index
        if pen_width(component) < THIN_WIDTH_PIXELS:
            component = ndimage.binary_dilation(component, structure, iterations=WIDEN_ITERATIONS)
        out |= component
    return out


def gold_mask(marks, width, height):
    return np.asarray(correct_mask(strip_mask(rasterise(marks, width, height))), bool)
