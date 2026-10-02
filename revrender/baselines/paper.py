"""Two published struck-out methods re-implemented from their papers, on the paper's protocol (ExamRev, gold v2).

  shivakumara2022  Shivakumara et al. (ESWA 2022): stroke-width word proposals, an ImageNet DenseNet-121 struck / clean
                   classifier; a proposal judged struck is delivered as the ink inside its box
  chaudhuri2017    Chaudhuri and Adak (PR 2017): seven morphology features, an RBF-SVM (grid-searched over writers), and a
                   middle-biased skeleton path, dilated to the stroke width, as the strike mask

Both are ported unchanged from the first development protocol (2,834 training lines, 505 validation lines, components
matched at IoU 0.1).  Here they train on the paper's 5,935 training lines (the bundle's training split and the user-corrected
rows, as every model) and are read on the evaluation split: every proposal is an instance scored by its classifier; the NMS
level and score cut are chosen on validation by the frontier (best F1, ties fewer false positives), as every model's
operating point.  A proposal is a positive training example when at least 3 gold pixels lie in its box.  Also recorded: the
proposals' native struck / clean classification on validation.  Seed 42; ``SVM_JOBS`` parallel grid-search workers (same
folds, same result).

Writes an archive and a freeze record.  Neither paper released code, so none is used.  This does NOT read the test split,
choose anything on data other than validation, train with more than one seed, overwrite, or contact the network.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import platform
import random
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from ..util import SEED, digest, input_hashes, make_output_dir, require, to_jsonable, write_json, write_manifest
from .common import freeze_record, read_archive, training_and_validation, write_line_archive

SCHEMA = 'hme-revision-strikeout-paper-baselines-v1'
DENSENET_SHA = 'a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395'
ARMS = ('shivakumara2022', 'chaudhuri2017')
MIN_INK, STEPS, BATCH = 8, 12000, 32
SVM_JOBS = 8
WORKERS = 16


class Paper915Error(RuntimeError):
    """The ported methods' own refusals."""


def otsu_ink(image: np.ndarray) -> np.ndarray:
    """Return dark foreground using a dependency-free Otsu threshold."""
    value = np.asarray(image, dtype=np.uint8)
    histogram = np.bincount(value.ravel(), minlength=256).astype(np.float64)
    count = histogram.sum()
    if count == 0:
        return np.zeros_like(value, dtype=bool)
    probability = histogram / count
    omega = np.cumsum(probability)
    mean = np.cumsum(probability * np.arange(256))
    total = mean[-1]
    denominator = omega * (1.0 - omega)
    between = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    between[valid] = (total * omega[valid] - mean[valid]) ** 2 / denominator[valid]
    threshold = int(np.argmax(between))
    return value <= threshold


def _runs(row: np.ndarray) -> list[int]:
    padded = np.pad(np.asarray(row, dtype=np.int8), (1, 1))
    change = np.flatnonzero(np.diff(padded))
    return [int(end - start) for start, end in zip(change[::2], change[1::2])]


def estimate_stroke_width(ink: np.ndarray) -> int:
    lengths = [length for row in ink for length in _runs(row) if length > 0]
    if not lengths:
        return 1
    values = np.bincount(np.clip(lengths, 1, 31))
    return max(1, int(np.argmax(values[1:]) + 1))


def word_proposals(image: np.ndarray, min_ink: int = 8) -> list[tuple[int, int, int, int]]:
    """Paper-described stroke-width smoothing adapted to a single line crop."""
    ink = otsu_ink(image)
    stroke = estimate_stroke_width(ink)
    # The paper fills horizontal white runs <=5*SW, then applies a SW x 6SW
    # rectangular morphology.  A closing expresses that stated merging goal;
    # the manuscript calls it "opening" despite showing merged word regions.
    merged = ndimage.binary_closing(
        ink, structure=np.ones((1, max(1, 5 * stroke + 1)), dtype=bool))
    merged = ndimage.binary_closing(
        merged, structure=np.ones((max(1, stroke), max(1, 6 * stroke)), dtype=bool))
    labels, count = ndimage.label(merged, np.ones((3, 3), dtype=np.uint8))
    boxes: list[tuple[int, int, int, int]] = []
    height, width = ink.shape
    for index in range(1, count + 1):
        yy, xx = np.nonzero(labels == index)
        if len(xx) == 0:
            continue
        x1, x2 = max(0, int(xx.min()) - stroke), min(width, int(xx.max()) + stroke + 1)
        y1, y2 = max(0, int(yy.min()) - stroke), min(height, int(yy.max()) + stroke + 1)
        if int(ink[y1:y2, x1:x2].sum()) >= min_ink:
            boxes.append((x1, y1, x2, y2))
    if not boxes and int(ink.sum()) >= min_ink:
        yy, xx = np.nonzero(ink)
        boxes.append((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1))
    return sorted(boxes)


def thin(ink: np.ndarray, max_iterations: int = 128) -> np.ndarray:
    """Zhang-Suen thinning, used only by the Chaudhuri graph arm."""
    value = np.asarray(ink, dtype=np.uint8).copy()
    if min(value.shape, default=0) < 3:
        return value.astype(bool)
    for _ in range(max_iterations):
        changed = False
        for first in (True, False):
            p2 = np.roll(value, -1, 0); p3 = np.roll(p2, -1, 1)
            p4 = np.roll(value, -1, 1); p5 = np.roll(p4, 1, 0)
            p6 = np.roll(value, 1, 0); p7 = np.roll(p6, 1, 1)
            p8 = np.roll(value, 1, 1); p9 = np.roll(p8, -1, 0)
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = sum(condition.astype(np.uint8) for condition in (
                (p2 == 0) & (p3 == 1), (p3 == 0) & (p4 == 1),
                (p4 == 0) & (p5 == 1), (p5 == 0) & (p6 == 1),
                (p6 == 0) & (p7 == 1), (p7 == 0) & (p8 == 1),
                (p8 == 0) & (p9 == 1), (p9 == 0) & (p2 == 1)))
            if first:
                condition = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                condition = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (value == 1) & (neighbours >= 2) & (neighbours <= 6) \
                & (transitions == 1) & condition
            remove[[0, -1], :] = False; remove[:, [0, -1]] = False
            if bool(remove.any()):
                value[remove] = 0
                changed = True
        if not changed:
            break
    return value.astype(bool)


def seven_features(crop: np.ndarray) -> np.ndarray:
    """Seven features stated in Chaudhuri's author material."""
    ink = otsu_ink(crop)
    height, width = ink.shape
    eccentricity = max(1e-3, min(height, width) / max(height, width, 1))
    skeleton = thin(ink)
    neighbour_count = ndimage.convolve(
        skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") \
        - skeleton.astype(np.uint8)
    branch = skeleton & (neighbour_count >= 3)
    cross = skeleton & (neighbour_count >= 4)
    thirds = np.array_split(branch, 3, axis=0)
    weighted_branch = sum(weight * int(zone.sum())
                          for weight, zone in zip((1.0, 2.0, 1.0), thirds))
    strip_density = np.array([zone.mean() if zone.size else 0.0
                              for zone in np.array_split(ink, 8, axis=0)])
    holes, hole_count = ndimage.label(~ink, np.ones((3, 3), np.uint8))
    border = set(np.unique(np.concatenate((holes[0], holes[-1], holes[:, 0], holes[:, -1]))))
    interior_holes = [index for index in range(1, hole_count + 1) if index not in border]
    centers = [ndimage.center_of_mass(holes == index) for index in interior_holes]
    aligned_pairs = sum(abs(a[1] - b[1]) <= max(1, width * 0.1)
                        for pos, a in enumerate(centers) for b in centers[pos + 1:])
    return np.asarray([
        branch.sum() / eccentricity,
        weighted_branch / eccentricity,
        cross.sum() / eccentricity,
        ink.mean() / eccentricity,
        strip_density.std() / eccentricity,
        len(interior_holes) / eccentricity,
        aligned_pairs / eccentricity,
    ], dtype=np.float32)


def candidate_label(mask: np.ndarray, box: tuple[int, int, int, int], minimum: int = 3) -> int:
    x1, y1, x2, y2 = box
    return int(np.count_nonzero(mask[y1:y2, x1:x2]) >= minimum)


def _shortest_middle_path(skeleton: np.ndarray) -> np.ndarray:
    output = np.zeros_like(skeleton, dtype=bool)
    yy, xx = np.nonzero(skeleton)
    if not len(xx):
        return output
    neighbours = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8)) \
        - skeleton.astype(np.uint8)
    endpoints = np.argwhere(skeleton & (neighbours <= 1))
    if len(endpoints) < 2:
        endpoints = np.argwhere(skeleton)
    left = endpoints[endpoints[:, 1] <= np.quantile(endpoints[:, 1], 0.2)]
    right = endpoints[endpoints[:, 1] >= np.quantile(endpoints[:, 1], 0.8)]
    targets = {tuple(map(int, point)) for point in right}
    middle = (skeleton.shape[0] - 1) / 2.0
    best: tuple[float, tuple[int, int], dict[tuple[int, int], tuple[int, int]]] | None = None
    for source_array in left[:16]:
        source = tuple(map(int, source_array))
        distance = {source: 0.0}; previous: dict[tuple[int, int], tuple[int, int]] = {}
        queue = [(0.0, source)]
        reached = None
        while queue:
            cost, point = heapq.heappop(queue)
            if cost != distance.get(point):
                continue
            if point in targets:
                reached = point; break
            y, x = point
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dy == 0 and dx == 0) or dx < 0:
                        continue
                    nxt = (y + dy, x + dx)
                    if not (0 <= nxt[0] < skeleton.shape[0] and 0 <= nxt[1] < skeleton.shape[1]) \
                            or not skeleton[nxt]:
                        continue
                    step = math.sqrt(2.0) if dx and dy else 1.0
                    score = cost + step + 0.02 * abs(nxt[0] - middle)
                    if score < distance.get(nxt, float("inf")):
                        distance[nxt] = score; previous[nxt] = point
                        heapq.heappush(queue, (score, nxt))
        if reached is not None and (best is None or distance[reached] < best[0]):
            best = (distance[reached], reached, previous)
    if best is None:
        return output
    point = best[1]
    while True:
        output[point] = True
        if point not in best[2]:
            break
        point = best[2][point]
    return output


def graph_strike_mask(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    ink = otsu_ink(image[y1:y2, x1:x2])
    path = _shortest_middle_path(thin(ink))
    radius = max(1, estimate_stroke_width(ink) // 2)
    path = ndimage.binary_dilation(path, iterations=radius) & ink
    output = np.zeros_like(ink, dtype=np.float32)
    output[path] = 1.0
    return output



def fit_svm(candidates: Sequence[dict[str, Any]], seed: int):
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    labels = np.asarray([item["label"] for item in candidates])
    if set(labels.tolist()) != {0, 1}:
        raise Paper915Error("SVM candidates do not contain both classes")
    model = make_pipeline(StandardScaler(), SVC(
        kernel="rbf", probability=True, class_weight="balanced",
        random_state=seed))
    groups = np.asarray([item["group_uuid"] for item in candidates])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise Paper915Error("fewer than five writers for SVM tuning")
    search = GridSearchCV(
        model,
        {"svc__C": (1.0, 10.0, 100.0),
         "svc__gamma": ("scale", 0.01, 0.1)},
        cv=GroupKFold(5), scoring="balanced_accuracy", n_jobs=SVM_JOBS,
        refit=True)
    search.fit(np.stack([item["features"] for item in candidates]), labels,
               groups=groups)
    return search.best_estimator_



def _resize_crop(image: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]):
    import torch
    from torch.nn import functional as functional

    x1, y1, x2, y2 = box
    value = torch.from_numpy(image[y1:y2, x1:x2].astype(np.float32))[None, None]
    return functional.interpolate(value, size=size, mode="bilinear", align_corners=False)[0, 0]


def make_densenet(weights_path: Path):
    import torch
    import torchvision
    from torch import nn

    model = torchvision.models.densenet121(weights=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Older official torchvision DenseNet checkpoints used keys such as
    # ``denselayer1.norm.1.weight``.  Current torchvision modules expose the
    # same tensor as ``denselayer1.norm1.weight``.  Apply torchvision's
    # historical compatibility rewrite before strict loading so the declared
    # ImageNet initialization is preserved instead of silently dropping most
    # of the encoder.
    legacy = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.([12])\."
        r"(weight|bias|running_mean|running_var)$")
    for key in list(state):
        match = legacy.match(key)
        if match is None:
            continue
        state[match.group(1) + match.group(2) + "." + match.group(3)] = \
            state.pop(key)
    model.load_state_dict(state, strict=True)
    inputs = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(inputs, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 1))
    return model


def fit_densenet(candidates: Sequence[dict[str, Any]], args, seed: int, device):
    # args: densenet121_weights, component_steps, component_batch_size
    import torch
    from torch.nn import functional as functional

    model = make_densenet(args.densenet121_weights).to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    positive = [item for item in candidates if item["label"]]
    negative = [item for item in candidates if not item["label"]]
    if not positive or not negative:
        raise Paper915Error("DenseNet candidates do not contain both classes")
    rng = random.Random(seed)
    for step in range(args.component_steps):
        half = args.component_batch_size // 2
        batch = rng.choices(positive, k=half) + rng.choices(negative, k=args.component_batch_size - half)
        rng.shuffle(batch)
        value = torch.stack([_resize_crop(item["row"]["image"], item["box"], (224, 224))
                             for item in batch]).to(device) / 255.0
        value = value[:, None].repeat(1, 3, 1, 1)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
        value = (value - mean) / std
        target = torch.tensor([item["label"] for item in batch], dtype=torch.float32, device=device)
        loss = functional.binary_cross_entropy_with_logits(model(value).flatten(), target)
        optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step()
        if (step + 1) % 500 == 0:
            print(json.dumps({"event": "train", "step": step + 1,
                              "loss": float(loss.detach().cpu())}), flush=True)
    return model



ROWS = []


def proposals_of(index):
    """One line's word proposals, their labels and, for the SVM, their seven features."""
    row = ROWS[index]
    boxes = word_proposals(row['image'], MIN_INK)
    labels = [candidate_label(row['mask'], box) for box in boxes]
    features = [seven_features(row['image'][y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes] if row['features'] else None
    return index, boxes, labels, features


def all_proposals(rows, features):
    import multiprocessing
    global ROWS
    ROWS = [dict(r, features=features) for r in rows]
    with multiprocessing.get_context('fork').Pool(WORKERS) as pool:
        out = sorted(pool.imap_unordered(proposals_of, range(len(rows)), chunksize=16), key=lambda x: x[0])
    ROWS = []
    return out


def instance_masks(arm, image, boxes):
    """Each proposal as one delivered mask: the ink in its box (Shivakumara), or the skeleton strike path (Chaudhuri)."""
    ink = otsu_ink(image)
    masks = []
    for box in boxes:
        x1, y1, x2, y2 = box
        mask = np.zeros(image.shape, bool)
        if arm == 'shivakumara2022':
            mask[y1:y2, x1:x2] = ink[y1:y2, x1:x2]
        else:
            mask[y1:y2, x1:x2] = graph_strike_mask(image, box) > 0
        masks.append(mask)
    return masks


def densenet_scores(model, image, boxes, device):
    import torch
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(boxes), BATCH):
            chunk = boxes[start:start+BATCH]
            value = torch.stack([_resize_crop(image, box, (224, 224)) for box in chunk]).to(device)/255.0
            value = value[:, None].repeat(1, 3, 1, 1)
            mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
            std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
            scores += torch.sigmoid(model((value-mean)/std).flatten()).cpu().numpy().astype(np.float64).tolist()
    return scores


def main(argv=None):
    global WORKERS, STEPS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--corrected', type=Path, required=True)
    parser.add_argument('--corrected-sha', required=True)
    parser.add_argument('--densenet121-weights', type=Path, help='torchvision DenseNet-121 ImageNet weights (shivakumara2022)')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--smoke', action='store_true', help='4 steps on 200 training and 20 validation lines: checks the pipeline end to end')
    args = parser.parse_args(argv)
    WORKERS = args.workers
    if args.smoke:
        STEPS = 4
    args.component_steps, args.component_batch_size = STEPS, BATCH
    require(not args.output_dir.exists(), 'create-only output exists', FileExistsError)
    if args.arm == 'shivakumara2022':
        require(args.densenet121_weights is not None and digest(args.densenet121_weights) == DENSENET_SHA,
                'the DenseNet-121 weights are missing or changed')
    bundle, train, validation, order, truths, extra_info = training_and_validation(
        args.bundle, args.bundle_sha, args.corrected, args.corrected_sha)
    if args.smoke:
        train = train[:200]
        validation, order, truths = validation[:20], order[:20], truths[:20]
    train_rows = [{'image': r['image'], 'mask': r['mask'], 'group_uuid': r['group_uuid']} for r in train]
    dev_rows = [{'image': r['image'], 'mask': r['mask'], 'group_uuid': r['group_uuid'], 'sample_id': r['sample_id']}
                for r in validation]
    gold = sum(map(len, truths))
    svm = args.arm == 'chaudhuri2017'
    fit = all_proposals(train_rows, svm)
    candidates = [{'box': box, 'label': label, 'row': train_rows[i], 'group_uuid': train_rows[i]['group_uuid'],
                   **({'features': features[k]} if svm else {})}
                  for i, boxes, labels, features in fit for k, (box, label) in enumerate(zip(boxes, labels))]
    print(json.dumps({'training_lines': len(train_rows), 'proposals': len(candidates),
                      'positive': sum(c['label'] for c in candidates)}), flush=True)
    random.seed(SEED)
    np.random.seed(SEED)
    if svm:
        model = fit_svm(candidates, SEED)
        val = all_proposals(dev_rows, True)
        scored = [(boxes, model.predict_proba(np.stack(features))[:, 1].tolist() if boxes else [], labels)
                  for _, boxes, labels, features in val]
        environment = {'sklearn': __import__('sklearn').__version__,
                       'svm_parameters': {k: str(v) for k, v in model.get_params().items() if k.startswith('svc__')}}
        import joblib
        out = make_output_dir(args.output_dir)
        joblib.dump(model, out/'classifier.joblib')
    else:
        import torch
        require(torch.cuda.is_available(), 'the DenseNet arm needs CUDA')
        device = torch.device('cuda')
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = fit_densenet(candidates, args, SEED, device)
        val = all_proposals(dev_rows, False)
        scored = [(boxes, densenet_scores(model, dev_rows[i]['image'], boxes, device) if boxes else [], labels)
                  for i, boxes, labels, _ in val]
        out = make_output_dir(args.output_dir)
        torch.save(model.state_dict(), out/'classifier.pt')
        environment = {'torch': torch.__version__}
    archive = out/'dev_instances'
    archive.mkdir()
    native_labels, native_scores = [], []
    for i, (row, (boxes, scores, labels)) in enumerate(zip(dev_rows, scored)):
        masks = instance_masks(args.arm, row['image'], boxes)
        keep = [k for k, m in enumerate(masks) if m.any()]
        write_line_archive(archive, i, row['sample_id'], [masks[k] for k in keep], [scores[k] for k in keep])
        native_labels += labels
        native_scores += scores
    points, point = read_archive(archive, order, truths)
    predicted = np.asarray(native_scores) >= .5
    labels = np.asarray(native_labels, bool)
    tp, fp, fn = int((predicted & labels).sum()), int((predicted & ~labels).sum()), int((~predicted & labels).sum())
    native = {'proposals': int(len(labels)), 'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
              'f1': 2*tp/max(2*tp+fp+fn, 1), 'threshold': .5, 'label_rule': '>= 3 gold pixels in the box'}
    record = freeze_record(args.arm, args.bundle_sha, archive, point, truths, points, corrected_sha256=args.corrected_sha,
                           smoke_only=args.smoke)
    record['validation']['native_classification'] = native
    write_json(out/'freeze.json', record)
    write_manifest(out, config=to_jsonable(args), inputs=input_hashes([args.bundle, args.corrected]), **{
        'schema': SCHEMA, 'arm': args.arm, 'test_touched': False, 'create_only': True, 'smoke_only': args.smoke,
        'recipe': {'min_ink': MIN_INK, 'steps': STEPS, 'batch': BATCH, 'svm_jobs': SVM_JOBS},
        'data': {'training_lines': len(train_rows), 'proposals': len(candidates), 'corrected': extra_info},
        'declared_inputs': {'bundle': args.bundle_sha, 'corrected': args.corrected_sha, 'densenet121': DENSENET_SHA if not svm else None},
        'outputs': {'freeze.json': digest(out/'freeze.json')}, 'environment': environment,
        'python': platform.python_version()})
    print(json.dumps({'arm': args.arm, 'f1': round(point['f1'], 4), 'tp_fp': [point['true_positive'], point['false_positive']],
                      'native_f1': round(native['f1'], 4)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
