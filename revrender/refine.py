"""Click representation, simulated corrections and local inference for annotation.

The server may only change a neighbourhood of the user's clicks. Manual paint
remains authoritative, and predictions are suggestions until the user confirms.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

from .data.gold import letterbox
from .util import require

CANVAS = (96, 768)


def click_maps(shape, clicks, radius=4.):
    y, x = np.indices(shape)
    maps = np.zeros((2, *shape), np.float32)
    for cx, cy, add in clicks:
        layer = 0 if add else 1
        maps[layer] = np.maximum(maps[layer], np.exp(-((x-cx)**2+(y-cy)**2)/(2*radius**2)))
    return maps


def simulate(grey, target, rng):
    """Noisy previous masks plus error clicks; letters supply plausible false positives."""
    target = np.asarray(target, bool)
    mode = int(rng.integers(4))
    if mode == 0:
        prior = np.zeros_like(target)
    elif mode == 1:
        prior = ndimage.binary_dilation(target, iterations=int(rng.integers(1, 4))) & (grey < 240)
    elif mode == 2:
        prior = ndimage.binary_erosion(target, iterations=1)
    else:
        prior = target.copy()
        for _ in range(3):
            x, y = int(rng.integers(grey.shape[1])), int(rng.integers(grey.shape[0]))
            prior[max(0,y-12):y+12, max(0,x-24):x+24] ^= grey[max(0,y-12):y+12, max(0,x-24):x+24] < 235
    errors = prior != target
    clicks = []
    for _ in range(int(rng.integers(1, 5))):
        if not errors.any():
            break
        distance = ndimage.distance_transform_edt(errors)
        cy, cx = np.unravel_index(int(distance.argmax()), errors.shape)
        clicks.append((int(cx), int(cy), bool(target[cy, cx])))
        errors[max(0,cy-6):cy+7, max(0,cx-6):cx+7] = False
    maps = click_maps(grey.shape, clicks)
    return np.concatenate([(grey.astype(np.float32)/127.5-1)[None], prior[None].astype(np.float32), maps]), prior, clicks


def refine_crop(model, grey, mask, clicks, radius=28):
    """Refine in native crop coordinates; pixels outside click neighbourhoods never move."""
    import torch
    require(grey.shape == mask.shape and grey.ndim == 2, 'image and mask must have equal 2D sizes')
    h, w = grey.shape
    require(0 < len(clicks) <= 200, 'give 1–200 add/keep clicks')
    require(all(len(c) == 3 and 0 <= c[0] < w and 0 <= c[1] < h for c in clicks), 'click outside crop')
    scale = min(CANVAS[1]/w, CANVAS[0]/h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    left, top = (CANVAS[1]-rw)//2, (CANVAS[0]-rh)//2
    mapped = [(x*scale+left, y*scale+top, bool(add)) for x,y,add in clicks]
    image = np.asarray(letterbox(Image.fromarray(grey), CANVAS[1], CANVAS[0], 255))
    previous = np.asarray(letterbox(Image.fromarray(mask.astype(np.uint8)*255), CANVAS[1], CANVAS[0], 0, mask=True))/255.
    inputs = np.concatenate([(image.astype(np.float32)/127.5-1)[None], previous[None], click_maps(CANVAS, mapped)]).astype(np.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        probability = model(torch.from_numpy(inputs[None]).to(device)).sigmoid()[0, 0].cpu().numpy()
    # unletterbox handles masks with nearest sampling; interpolate the probability explicitly.
    probability = np.asarray(Image.fromarray(probability[top:top+rh, left:left+rw]).resize((w,h), Image.Resampling.BILINEAR))
    y, x = np.indices((h,w))
    local = np.zeros((h,w), bool)
    for cx,cy,add in clicks:
        local |= (x-cx)**2+(y-cy)**2 <= radius**2
    result = np.asarray(mask, bool).copy()
    result[local] = (probability[local] >= .5) & (grey[local] < 245)
    # Honour explicit user points, independent of the learned suggestion.
    for cx,cy,add in clicks:
        core = (x-cx)**2+(y-cy)**2 <= 1
        result[core] = bool(add) & (grey[core] < 245)
    return result, probability
