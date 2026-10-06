"""
segment_core.py
================

README — CLASSICAL HUMAN SEGMENTATION (NO TRAINED MODEL)
---------------------------------------------------------

Shared by the HW3 command-line scripts and the web app (Module 4).
Nothing here is a neural network. GrabCut fits color models to the one
image you give it; watershed, background subtraction, and the thermal
pipeline are deterministic image processing.

    cd /Users/home/GSU/Computer_Vision/Computer_Vision
    .venv/bin/python HW3/segment_core.py

WHAT IT PROVIDES
  segment_grabcut(image, box)         graph-cut, seeded by a user box
  segment_watershed(image, box)       markers from that same box
  segment_background(image, empty)    absdiff of a fixed-camera pair
  segment_thermal(image, ...)         CLAHE, Otsu, person-shaped components
  compare_masks(opencv, reference)    IoU, Dice, boundary F, Hausdorff
  comparison_overlay(...)             both / OpenCV only / reference only

BOXES
  [x, y, width, height] in pixels. x is the column, y is the row,
  origin at the top-left. SAM2 wants the two corners instead; the
  SAM2 script converts this box, so both methods see the same region.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

METHODS = ("grabcut", "watershed", "background", "thermal")


def clip_box(box, shape) -> tuple[int, int, int, int]:
    """Clip [x, y, w, h] so it lies inside the image and is at least 2 px."""
    height, width = shape[:2]
    x, y, w, h = (int(round(float(v))) for v in box)
    if w <= 0 or h <= 0:
        raise ValueError("The box must have a positive width and height.")
    x = max(0, min(x, width - 2))
    y = max(0, min(y, height - 2))
    w = max(2, min(w, width - x))
    h = max(2, min(h, height - y))
    return x, y, w, h


def to_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
    return gray


def to_bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _odd(size: int) -> int:
    size = int(size)
    if size <= 1:
        return 1
    return size if size % 2 else size + 1


def _kernel(size: int) -> np.ndarray:
    size = _odd(size)
    if size <= 1:
        return np.ones((1, 1), np.uint8)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def morph_cleanup(mask: np.ndarray, open_k: int = 5, close_k: int = 7) -> np.ndarray:
    """Open (drop speckles), then close (bridge small gaps)."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if open_k > 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _kernel(open_k))
    if close_k > 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _kernel(close_k))
    return binary


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return np.where(labels == keep, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background. The flood starts outside the image, so a
    person touching the border is left alone."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    height, width = binary.shape
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = binary
    flood = padded.copy()
    ff_mask = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 128)
    holes = (flood[1:-1, 1:-1] == 0) & (binary == 0)
    out = binary.copy()
    out[holes] = 255
    return out


def cleanup_person(mask: np.ndarray, open_k: int = 5, close_k: int = 7) -> np.ndarray:
    """Morphology, then the largest blob, then hole filling. Used by the RGB methods."""
    return fill_holes(largest_component(morph_cleanup(mask, open_k, close_k)))


def mask_bbox(mask: np.ndarray, margin: int = 0) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("The mask is empty, so it has no bounding box.")
    height, width = mask.shape[:2]
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(width - 1, int(xs.max()) + margin)
    y1 = min(height - 1, int(ys.max()) + margin)
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def contours_of(mask: np.ndarray) -> list[np.ndarray]:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    found, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return list(found)


def draw_outline(image: np.ndarray, mask: np.ndarray, color=(0, 220, 80), thickness: int = 2) -> np.ndarray:
    canvas = to_bgr(image).copy()
    cv2.drawContours(canvas, contours_of(mask), -1, color, thickness)
    return canvas


def _binary(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8))
    return (binary > 0) & (eroded == 0)


def compare_masks(prediction, reference, tolerance: int = 2) -> dict[str, Any]:
    """Region overlap plus boundary agreement.

    Boundary F counts a boundary pixel as correct when it lies within
    `tolerance` pixels of the other boundary (default 2). Hausdorff and the
    mean boundary distance are in pixels. Empty masks return None for the
    distances rather than a made-up number.
    """
    pred = _binary(prediction)
    ref = _binary(reference)
    if pred.shape != ref.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} vs {ref.shape}.")
    inter = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    pred_n = int(pred.sum())
    ref_n = int(ref.sum())
    iou = 1.0 if union == 0 else inter / union
    dice = 1.0 if pred_n + ref_n == 0 else (2 * inter) / (pred_n + ref_n)
    precision = 1.0 if pred_n == 0 and ref_n == 0 else (0.0 if pred_n == 0 else inter / pred_n)
    recall = 1.0 if pred_n == 0 and ref_n == 0 else (0.0 if ref_n == 0 else inter / ref_n)

    pred_b = boundary_pixels(pred)
    ref_b = boundary_pixels(ref)
    radius = max(1, int(tolerance))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    pred_b_u8 = pred_b.astype(np.uint8) * 255
    ref_b_u8 = ref_b.astype(np.uint8) * 255
    pred_near = cv2.dilate(pred_b_u8, kernel) > 0
    ref_near = cv2.dilate(ref_b_u8, kernel) > 0
    b_pred = int(pred_b.sum())
    b_ref = int(ref_b.sum())
    b_precision = 0.0 if b_pred == 0 else float((pred_b & ref_near).sum()) / b_pred
    b_recall = 0.0 if b_ref == 0 else float((ref_b & pred_near).sum()) / b_ref
    if b_precision + b_recall == 0:
        boundary_f = 0.0
    else:
        boundary_f = 2 * b_precision * b_recall / (b_precision + b_recall)

    hausdorff = None
    mean_boundary = None
    if b_pred and b_ref:
        # distanceTransform measures distance to the nearest ZERO pixel,
        # so invert the boundary: zeros sit on the contour.
        dist_ref = cv2.distanceTransform(cv2.bitwise_not(ref_b_u8), cv2.DIST_L2, 3)
        dist_pred = cv2.distanceTransform(cv2.bitwise_not(pred_b_u8), cv2.DIST_L2, 3)
        pred_to_ref = dist_ref[pred_b]
        ref_to_pred = dist_pred[ref_b]
        hausdorff = float(max(pred_to_ref.max(), ref_to_pred.max()))
        mean_boundary = float(0.5 * (pred_to_ref.mean() + ref_to_pred.mean()))

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "boundary_precision": float(b_precision),
        "boundary_recall": float(b_recall),
        "boundary_f": float(boundary_f),
        "boundary_tolerance_px": radius,
        "hausdorff_px": hausdorff,
        "mean_boundary_px": mean_boundary,
    }


def comparison_overlay(image: np.ndarray, opencv_mask, reference_mask) -> np.ndarray:
    """Green = both, blue = OpenCV only, red = reference (SAM2) only."""
    canvas = to_bgr(image).copy()
    pred = _binary(opencv_mask)
    ref = _binary(reference_mask)
    if pred.shape[:2] != canvas.shape[:2] or ref.shape[:2] != canvas.shape[:2]:
        raise ValueError("Overlay masks must match the image size.")
    both = pred & ref
    only_cv = pred & ~ref
    only_ref = ref & ~pred
    tint = np.zeros_like(canvas)
    tint[both] = (0, 200, 0)
    tint[only_cv] = (220, 140, 0)
    tint[only_ref] = (40, 40, 230)
    painted = canvas.copy()
    region = both | only_cv | only_ref
    painted[region] = cv2.addWeighted(canvas, 0.45, tint, 0.55, 0)[region]
    return painted


def segment_grabcut(image: np.ndarray, box, iterations: int = 5, open_k: int = 5, close_k: int = 7) -> np.ndarray:
    """GrabCut seeded by a rectangle. The GMM color models are fit to this
    image only; there is no training set and no learned weight file."""
    bgr = to_bgr(image)
    x, y, w, h = clip_box(box, bgr.shape)
    mask = np.zeros(bgr.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (x, y, w, h), bgd, fgd, int(iterations), cv2.GC_INIT_WITH_RECT)
    raw = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return cleanup_person(raw, open_k, close_k)


def segment_watershed(image: np.ndarray, box, open_k: int = 3, close_k: int = 3, shrink: float = 0.40) -> np.ndarray:
    """Marker-controlled watershed on the gradient.

    Outside the box is definite background. A rectangle shrunk toward the
    middle of the box is definite foreground. The band between them is unknown
    and the gradient decides where the boundary falls.
    """
    bgr = to_bgr(image)
    x, y, w, h = clip_box(box, bgr.shape)
    shrink = float(shrink)
    if not 0.05 <= shrink <= 0.45:
        raise ValueError("shrink must be between 0.05 and 0.45.")
    ix = int(round(x + w * shrink))
    iy = int(round(y + h * shrink))
    iw = max(2, int(round(w * (1 - 2 * shrink))))
    ih = max(2, int(round(h * (1 - 2 * shrink))))
    ix, iy, iw, ih = clip_box((ix, iy, iw, ih), bgr.shape)

    gray = to_gray(bgr)
    # A 3x3 blur keeps sensor speckle from becoming extra ridges, without
    # thickening the person's outline into a band the flood cannot cross.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elevation = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

    markers = np.zeros(gray.shape, np.int32)
    markers[:] = 1
    markers[y:y + h, x:x + w] = 0
    markers[iy:iy + ih, ix:ix + iw] = 2
    labels = cv2.watershed(elevation, markers)
    raw = np.where(labels == 2, 255, 0).astype(np.uint8)
    return cleanup_person(raw, open_k, close_k)


def segment_background(
    image: np.ndarray,
    background: np.ndarray,
    threshold: int = 25,
    open_k: int = 5,
    close_k: int = 9,
) -> np.ndarray:
    """Fixed camera: the empty scene and the scene with the person."""
    current = to_bgr(image)
    empty = to_bgr(background)
    if current.shape != empty.shape:
        raise ValueError(
            f"The photo and the empty frame must be the same size ({current.shape[:2]} vs {empty.shape[:2]})."
        )
    diff = cv2.absdiff(current, empty)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, raw = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    return cleanup_person(raw, open_k, close_k)


def _person_shaped(x, y, w, h, area: int, min_area: int, max_aspect: float) -> bool:
    if area < min_area or w < 2 or h < 2:
        return False
    aspect = h / float(w)
    limit = max(float(max_aspect), 1.01)
    return (1.0 / limit) <= aspect <= limit


def filter_components(mask: np.ndarray, min_area: int, max_aspect: float, box=None) -> np.ndarray:
    """Keep blobs that are large enough and roughly person-shaped.

    When a box is given, keep only the blobs that overlap that box, so the
    thermal result uses the same prompt as SAM2.
    """
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    chosen = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if _person_shaped(x, y, w, h, area, min_area, max_aspect):
            chosen.append(label)
    if box is not None and chosen:
        bx, by, bw, bh = clip_box(box, binary.shape)
        overlapping = []
        for label in chosen:
            x, y, w, h, _area = (int(v) for v in stats[label])
            ix1, iy1 = max(x, bx), max(y, by)
            ix2, iy2 = min(x + w, bx + bw), min(y + h, by + bh)
            if ix2 > ix1 and iy2 > iy1:
                overlapping.append(label)
        if overlapping:
            chosen = overlapping
    out = np.zeros_like(binary)
    for label in chosen:
        out[labels == label] = 255
    return out


def split_touching(mask: np.ndarray) -> np.ndarray:
    """Split blobs that touch, using peaks of the distance transform as markers."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return binary
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    peak = float(dist.max())
    if peak <= 0:
        return binary
    _, sure = cv2.threshold(dist, 0.45 * peak, 255, cv2.THRESH_BINARY)
    sure = sure.astype(np.uint8)
    sure = cv2.morphologyEx(sure, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, markers = cv2.connectedComponents(sure)
    if count <= 2:
        return binary
    unknown = cv2.subtract(binary, sure)
    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(color, markers.astype(np.int32))
    return np.where(labels > 1, 255, 0).astype(np.uint8)


def segment_thermal(
    image: np.ndarray,
    box=None,
    invert: bool = False,
    min_area: int | None = None,
    max_aspect: float = 4.5,
    clahe_clip: float = 2.0,
    blur: int = 5,
    open_k: int = 3,
    close_k: int = 7,
    split: bool = False,
) -> np.ndarray:
    """People are warmer than the room, so they are the bright blobs.

    CLAHE stretches local contrast, a small Gaussian blur keeps Otsu from
    breaking on sensor noise, and connected components drop blobs that are
    too small or too wide to be a person. `invert` is for palettes that draw
    hot objects as dark.
    """
    gray = to_gray(image)
    height, width = gray.shape
    if min_area is None:
        min_area = max(80, int(0.002 * height * width))
    clip = float(clahe_clip)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, clip), tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    ksize = _odd(blur)
    if ksize >= 3:
        enhanced = cv2.GaussianBlur(enhanced, (ksize, ksize), 0)
    if invert:
        enhanced = cv2.bitwise_not(enhanced)
    _, raw = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = morph_cleanup(raw, open_k, close_k)
    kept = filter_components(cleaned, int(min_area), float(max_aspect), box)
    if split:
        kept = split_touching(kept)
        kept = filter_components(kept, int(min_area), float(max_aspect), box)
    return fill_holes(kept)


def segment(
    image: np.ndarray,
    method: str,
    box=None,
    background: np.ndarray | None = None,
    **params,
) -> dict[str, Any]:
    """Run one classical method and return the mask plus a contour overlay."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Use one of {METHODS}.")
    if method == "grabcut":
        if box is None:
            raise ValueError("GrabCut needs a box around the person.")
        mask = segment_grabcut(
            image, box,
            iterations=int(params.get("iterations", 5)),
            open_k=int(params.get("open_k", 5)),
            close_k=int(params.get("close_k", 7)),
        )
    elif method == "watershed":
        if box is None:
            raise ValueError("Watershed needs a box around the person.")
        mask = segment_watershed(
            image, box,
            open_k=int(params.get("open_k", 3)),
            close_k=int(params.get("close_k", 5)),
        )
    elif method == "background":
        if background is None:
            raise ValueError("Background differencing needs the empty frame.")
        mask = segment_background(
            image, background,
            threshold=int(params.get("threshold", 25)),
            open_k=int(params.get("open_k", 5)),
            close_k=int(params.get("close_k", 9)),
        )
    else:
        mask = segment_thermal(
            image,
            box=box,
            invert=bool(params.get("invert", False)),
            min_area=params.get("min_area"),
            max_aspect=float(params.get("max_aspect", 4.5)),
            clahe_clip=float(params.get("clahe_clip", 2.0)),
            blur=int(params.get("blur", 5)),
            open_k=int(params.get("open_k", 3)),
            close_k=int(params.get("close_k", 7)),
            split=bool(params.get("split", False)),
        )
    return {
        "method": method,
        "mask": mask,
        "overlay": draw_outline(image, mask),
        "box": None if box is None else list(clip_box(box, image.shape)),
    }


def fit_max_side(image: np.ndarray, max_side: int = 1280) -> tuple[np.ndarray, float]:
    """Shrink so the long side is at most max_side. Returns the image and the scale applied to coordinates."""
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return image, 1.0
    scale = max_side / float(long_side)
    resized = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def scale_box(box, scale: float):
    if box is None:
        return None
    return [float(v) * float(scale) for v in box]


def _paint_person(shape, center, scale: float) -> np.ndarray:
    """A standing figure: head, torso, arms, legs. One connected blob.

    Each part overlaps the torso, so the largest-component cleanup keeps
    the whole person instead of discarding a detached head or arm.
    """
    mask = np.zeros(shape, np.uint8)
    cx, cy = center
    s = float(scale)

    def pt(dx, dy):
        return int(round(cx + dx * s)), int(round(cy + dy * s))

    cv2.ellipse(mask, pt(0, -4), (max(4, int(28 * s)), max(4, int(46 * s))), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, pt(-14, 70), (max(4, int(13 * s)), max(4, int(40 * s))), 6, 0, 360, 255, -1)
    cv2.ellipse(mask, pt(14, 70), (max(4, int(13 * s)), max(4, int(40 * s))), -6, 0, 360, 255, -1)
    cv2.ellipse(mask, pt(-32, 6), (max(4, int(14 * s)), max(4, int(34 * s))), 24, 0, 360, 255, -1)
    cv2.ellipse(mask, pt(32, 6), (max(4, int(14 * s)), max(4, int(34 * s))), -24, 0, 360, 255, -1)
    cv2.circle(mask, pt(0, -52), max(4, int(22 * s)), 255, -1)
    return mask


def synthetic_rgb(shape=(360, 480), clutter: bool = False, seed: int = 0, margin: int = 16):
    """Color photo of a person, the ground-truth mask, a box, and the empty frame.

    `shape` is (height, width). The empty frame is the same picture without
    the person, for background differencing.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    base = np.array([170, 150, 110], np.float64)
    empty = np.broadcast_to(base, (height, width, 3)).copy()
    empty += rng.normal(0, 6, empty.shape)
    if clutter:
        for _ in range(18):
            x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
            w, h = int(rng.integers(20, 90)), int(rng.integers(20, 90))
            color = rng.integers(0, 255, size=3).tolist()
            cv2.rectangle(empty, (x, y), (x + w, y + h), color, -1)
        empty = cv2.GaussianBlur(empty.astype(np.uint8), (5, 5), 0).astype(np.float64)
    empty = np.clip(empty, 0, 255).astype(np.uint8)
    mask = _paint_person((height, width), (width // 2, int(height * 0.48)), scale=1.15)
    image = empty.copy()
    person_color = np.array([70, 90, 200], np.uint8)
    noise = rng.integers(-8, 9, size=image.shape, dtype=np.int16)
    colored = np.clip(person_color.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image[mask > 0] = colored[mask > 0]
    box = mask_bbox(mask, margin=margin)
    return image, mask, box, empty


def synthetic_thermal(shape=(360, 480), seed: int = 1, invert: bool = False, distractor: bool = False, margin: int = 14):
    """Bright (warm) person on a cool background. Optional wide hot bar."""
    rng = np.random.default_rng(seed)
    height, width = shape
    cool = rng.normal(28, 6, size=(height, width))
    mask = _paint_person((height, width), (int(width * 0.55), int(height * 0.50)), scale=1.2)
    warm = rng.normal(210, 8, size=(height, width))
    gray = cool.copy()
    gray[mask > 0] = warm[mask > 0]
    if distractor:
        gray[18:48, 20:200] = rng.normal(190, 5, size=(30, 180))
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    if invert:
        gray = cv2.bitwise_not(gray)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    box = mask_bbox(mask, margin=margin)
    return image, mask, box


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_prompts(path: Path | str) -> list[dict]:
    """Read prompts.json. Each item is one image and the box both methods use.

    The file is either ``{"prompts": [ ... ]}`` or a bare list. A box is
    ``[x, y, width, height]`` in pixels.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = data["prompts"] if isinstance(data, dict) else data
    if not isinstance(prompts, list):
        raise ValueError("prompts.json must contain a list of prompts.")
    for item in prompts:
        if "image" not in item:
            raise ValueError("Each prompt needs an 'image' path.")
        item.setdefault("kind", "rgb")
        if item.get("box") is not None and len(item["box"]) != 4:
            raise ValueError(f"Box for {item['image']} must be [x, y, width, height].")
    return prompts


def sam2_mask_path(image_path: Path, root: Path | None = None) -> Path | None:
    """Where sam2_masks.py writes, then the committed sample stand-in."""
    root = repo_root() if root is None else Path(root)
    stem = Path(image_path).stem
    candidates = [
        root / "HW3" / "sam2_masks" / f"{stem}.png",
        root / "HW3" / "samples" / "sam2" / f"{stem}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {path}")
    return image


def run_prompt(prompt: dict, root: Path | None = None, max_side: int = 1280) -> dict[str, Any]:
    """Segment one prompts.json entry and compare with SAM2 when a mask exists.

    The box in prompts.json is in the original photo's pixels. The photo is
    then reduced so the long side is at most `max_side`, and the box is scaled
    by the same factor. A full-resolution SAM2 mask is resized to match.
    """
    root = repo_root() if root is None else Path(root)
    image_path = Path(prompt["image"])
    if not image_path.is_absolute():
        image_path = root / image_path
    image, scale = fit_max_side(read_bgr(image_path), max_side)
    kind = prompt.get("kind", "rgb")
    method = prompt.get("method") or ("thermal" if kind == "thermal" else "grabcut")
    background = None
    if prompt.get("background"):
        bg_path = Path(prompt["background"])
        if not bg_path.is_absolute():
            bg_path = root / bg_path
        background, _bg_scale = fit_max_side(read_bgr(bg_path), max_side)
        if background.shape[:2] != image.shape[:2]:
            background = cv2.resize(background, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    params = {key: prompt[key] for key in ("iterations", "invert", "min_area", "max_aspect", "threshold", "split", "open_k", "close_k") if key in prompt}
    result = segment(image, method, box=scale_box(prompt.get("box"), scale), background=background, **params)
    reference_path = sam2_mask_path(image_path, root)
    metrics = None
    comparison = None
    shown = None
    if reference_path is not None:
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise FileNotFoundError(f"Could not read {reference_path}")
        if reference.shape != result["mask"].shape:
            reference = cv2.resize(reference, (result["mask"].shape[1], result["mask"].shape[0]), interpolation=cv2.INTER_NEAREST)
        metrics = compare_masks(result["mask"], reference)
        comparison = comparison_overlay(image, result["mask"], reference)
        try:
            shown = str(reference_path.resolve().relative_to(root))
        except ValueError:
            shown = str(reference_path)
    return {
        "image": str(image_path),
        "kind": kind,
        "method": method,
        "box": result["box"],
        "mask": result["mask"],
        "overlay": result["overlay"],
        "comparison": comparison,
        "metrics": metrics,
        "sam2_mask": shown,
    }


def _standin_mask(mask: np.ndarray) -> np.ndarray:
    """Ground truth eroded by one pixel.

    These files are not SAM2 output. They only let the web app show an
    overlay before you run sam2_masks.py. Real SAM2 masks replace them.
    One pixel of erosion leaves a thin rim (OpenCV only) while IoU stays high.
    Hausdorff distance is still the single worst point, so a small dent
    can print as tens of pixels even when the mean gap is about one pixel.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(np.where(mask > 0, 255, 0).astype(np.uint8), kernel)


def write_demo_samples(folder: Path | None = None) -> list[dict]:
    """Synthetic photos the hosted demo can run without a camera upload."""
    folder = Path(folder) if folder else repo_root() / "HW3" / "samples"
    sam_dir = folder / "sam2"
    sam_dir.mkdir(parents=True, exist_ok=True)
    prompts: list[dict] = []

    def add(name, image, truth, box, kind, method, background=None, note=""):
        if not cv2.imwrite(str(folder / name), image):
            raise RuntimeError(f"Could not write {name}")
        if not cv2.imwrite(str(sam_dir / name), _standin_mask(truth)):
            raise RuntimeError(f"Could not write sam2 stand-in for {name}")
        item = {
            "image": f"HW3/samples/{name}",
            "kind": kind,
            "method": method,
            "box": [int(v) for v in box],
            "note": note,
        }
        if background:
            item["background"] = f"HW3/samples/{background}"
        prompts.append(item)

    image, truth, box, empty = synthetic_rgb(seed=0)
    cv2.imwrite(str(folder / "rgb_wall_empty.png"), empty)
    add("rgb_wall.png", image, truth, box, "rgb", "grabcut", "rgb_wall_empty.png",
        "Plain background. The empty frame is rgb_wall_empty.png.")

    image, truth, box, _empty = synthetic_rgb(clutter=True, seed=2)
    add("rgb_clutter.png", image, truth, box, "rgb", "grabcut",
        note="Cluttered background. GrabCut has to separate similar colors.")

    image, truth, box = synthetic_thermal(seed=1)
    add("thermal_person.png", image, truth, box, "thermal", "thermal",
        note="One warm person on a cool background.")

    image, truth, box = synthetic_thermal(seed=3, distractor=True)
    add("thermal_room.png", image, truth, box, "thermal", "thermal",
        note="A wide warm bar (a window or a lamp) sits beside the person. Aspect ratio drops it.")

    cool = np.clip(np.random.default_rng(5).normal(30, 5, size=(360, 640)), 0, 255).astype(np.uint8)
    right = _paint_person((360, 640), (480, 175), 1.1)
    left = _paint_person((360, 640), (140, 190), 1.0)
    cool[left > 0] = 200
    cool[right > 0] = 225
    both = cv2.cvtColor(cool, cv2.COLOR_GRAY2BGR)
    add("thermal_two.png", both, right, mask_bbox(right, margin=12), "thermal", "thermal",
        note="Two people. The box selects the person on the right, which is the prompt SAM2 would see.")
    return prompts


def _self_check() -> None:
    image, truth, box, empty = synthetic_rgb()
    grab = segment(image, "grabcut", box)
    shed = segment(image, "watershed", box)
    diff = segment(image, "background", background=empty)
    thermal_img, thermal_truth, thermal_box = synthetic_thermal()
    thermal = segment(thermal_img, "thermal", thermal_box)
    rows = [
        ("grabcut", compare_masks(grab["mask"], truth)),
        ("watershed", compare_masks(shed["mask"], truth)),
        ("background", compare_masks(diff["mask"], truth)),
        ("thermal", compare_masks(thermal["mask"], thermal_truth)),
    ]
    print("Synthetic person, IoU / boundary F")
    for name, metrics in rows:
        print(f"  {name:12}  IoU {metrics['iou']:.3f}   boundary F {metrics['boundary_f']:.3f}")
        if metrics["iou"] < 0.95:
            raise SystemExit(f"{name} IoU {metrics['iou']:.3f} is below 0.95")
    identical = compare_masks(truth, truth)
    if identical["iou"] != 1 or identical["hausdorff_px"] != 0:
        raise SystemExit(f"identical-mask metrics look wrong: {identical}")
    print("self-check ok")


if __name__ == "__main__":
    _self_check()
