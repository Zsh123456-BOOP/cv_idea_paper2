"""The training augmentation: one small affine warp of a line and its mask, then photometric jitter on the line only.

The draws from ``rng`` (a ``random.Random``) are part of the training recipe: their order fixes every later sample,
so do not reorder them.
"""
import io
import math
import random

import numpy as np


def paired_affine(image, mask, rng: random.Random):
    """With probability 0.2 nothing happens.  Otherwise rotate (+-3 deg), scale (+-6%) and shift (+-4%) image and mask
    together (bilinear / nearest), then contrast, brightness, optional blur and optional JPEG on the image."""
    from PIL import Image, ImageEnhance, ImageFilter

    if rng.random() > 0.8:
        return image, mask
    height, width = image.shape
    angle = rng.uniform(-3.0, 3.0)
    scale = 1.0 + rng.uniform(-0.06, 0.06)
    dx, dy = rng.uniform(-0.04, 0.04) * width, rng.uniform(-0.04, 0.04) * height
    radians = math.radians(angle)
    cosine, sine = math.cos(radians) / scale, math.sin(radians) / scale
    cx, cy = width / 2.0, height / 2.0
    matrix = (cosine, sine, cx - cosine * cx - sine * cy - dx,
              -sine, cosine, cy + sine * cx - cosine * cy - dy)

    def warp(array, resample, fill):
        return Image.fromarray(array).transform(
            (width, height), Image.Transform.AFFINE, matrix, resample=resample, fillcolor=fill)

    picture = warp(image, Image.Resampling.BILINEAR, 255)
    target = warp(mask, Image.Resampling.NEAREST, 0)
    picture = ImageEnhance.Contrast(picture).enhance(1.0 + rng.uniform(-0.12, 0.12))
    picture = ImageEnhance.Brightness(picture).enhance(1.0 + rng.uniform(-0.12, 0.12))
    if rng.random() < 0.20:
        picture = picture.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.8)))
    if rng.random() < 0.15:
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=rng.randint(55, 92))
        buffer.seek(0)
        picture = Image.open(buffer).convert("L")
    return np.asarray(picture, dtype=np.uint8).copy(), np.asarray(target, dtype=np.uint8).copy()
