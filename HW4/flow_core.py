"""
Optical flow and two-frame Lucas–Kanade tracking.

Dense flow uses Farnebäck's polynomial expansion (OpenCV). Sparse tracks use
pyramidal Lucas–Kanade (OpenCV). The from-scratch tracker below is the same
brightness-constancy least squares, with bilinear interpolation written out
so the worked example can print every number.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# Farnebäck defaults: a 3-level pyramid, 15 px window, 5-tap polynomial.
FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def gray_f64(image: np.ndarray) -> np.ndarray:
    """Single-channel float64 image, intensity roughly 0–255."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = np.asarray(image, dtype=np.float64)
    if out.size and out.max() <= 1.0:
        out = out * 255.0
    return out


def bilinear(image: np.ndarray, x, y):
    """
    Sample ``image`` at floating-point columns ``x`` and rows ``y``.

    Two linear interpolations. With ``x0 = floor(x)``, ``a = x - x0`` and the
    same for ``y`` and ``b``:

        (1-a)(1-b) I00 + a(1-b) I10 + (1-a)b I01 + a b I11

    Coordinates must lie in ``[0, width-1) × [0, height-1)``. Samples that fall
    outside that open range are NaN. ``image`` may be gray or multichannel;
    ``x`` and ``y`` are broadcast together.
    """
    img = np.asarray(image)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h, w = img.shape[:2]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    a = x - x0
    b = y - y0
    valid = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)

    def take(xx, yy):
        # Invalid locations are clipped only so the gather is legal; they are
        # replaced with NaN afterwards.
        return img[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]

    i00 = take(x0, y0).astype(np.float64)
    i10 = take(x0 + 1, y0).astype(np.float64)
    i01 = take(x0, y0 + 1).astype(np.float64)
    i11 = take(x0 + 1, y0 + 1).astype(np.float64)
    # a, b need a trailing axis when the image has channels.
    if img.ndim == 3:
        a = a[..., None]
        b = b[..., None]
        valid_b = valid[..., None]
    else:
        valid_b = valid
    value = (1.0 - a) * (1.0 - b) * i00 + a * (1.0 - b) * i10 + (1.0 - a) * b * i01 + a * b * i11
    return np.where(valid_b, value, np.nan)


def _gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central differences. The border is left at 0 and is not sampled by LK."""
    ix = np.zeros_like(gray)
    iy = np.zeros_like(gray)
    ix[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    iy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    return ix, iy


def _lk_level(gray1: np.ndarray, gray2: np.ndarray, points: np.ndarray, initial: np.ndarray,
              window: int, max_iter: int, eps: float) -> dict:
    """One-level iterative Lucas–Kanade. ``initial`` is the (u, v) guess per point."""
    ix_img, iy_img = _gradients(gray1)
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    dx = dx.ravel()
    dy = dy.ravel()

    points = np.asarray(points, dtype=np.float64)
    disp = np.asarray(initial, dtype=np.float64).copy()
    status = np.ones(len(points), dtype=bool)
    eigenvalues = np.zeros((len(points), 2))
    iterations = np.zeros(len(points), dtype=int)

    for i, (x, y) in enumerate(points):
        xs = x + dx
        ys = y + dy
        gx = bilinear(ix_img, xs, ys)
        gy = bilinear(iy_img, xs, ys)
        template = bilinear(gray1, xs, ys)
        ok = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(template)
        if int(ok.sum()) < 4:
            status[i] = False
            continue
        gx, gy, template = gx[ok], gy[ok], template[ok]
        dx_i, dy_i = dx[ok], dy[ok]
        ata = np.array([[np.dot(gx, gx), np.dot(gx, gy)],
                        [np.dot(gx, gy), np.dot(gy, gy)]], dtype=np.float64)
        eigenvalues[i] = np.linalg.eigvalsh(ata)
        # One tiny eigenvalue: the window only sees an edge (aperture problem).
        if eigenvalues[i, 0] < 1e-4:
            status[i] = False

        u, v = float(disp[i, 0]), float(disp[i, 1])
        for it in range(max_iter):
            sampled = bilinear(gray2, x + u + dx_i, y + v + dy_i)
            good = np.isfinite(sampled)
            if int(good.sum()) < 4:
                status[i] = False
                break
            residual = sampled[good] - template[good]
            gxm, gym = gx[good], gy[good]
            ata_m = np.array([[np.dot(gxm, gxm), np.dot(gxm, gym)],
                              [np.dot(gxm, gym), np.dot(gym, gym)]])
            atb = -np.array([np.dot(gxm, residual), np.dot(gym, residual)])
            try:
                step = np.linalg.solve(ata_m, atb)
            except np.linalg.LinAlgError:
                status[i] = False
                break
            u += float(step[0])
            v += float(step[1])
            iterations[i] = it + 1
            if float(step @ step) < eps * eps:
                break
        disp[i] = (u, v)

    return {
        "points": points + disp,
        "displacement": disp,
        "status": status,
        "eigenvalues": eigenvalues,
        "iterations": iterations,
    }


def lucas_kanade(image1: np.ndarray, image2: np.ndarray, points, window: int = 15,
                 max_iter: int = 20, eps: float = 1e-4, levels: int = 1) -> dict:
    """
    From-scratch Lucas–Kanade.

    Brightness constancy, linearized, gives ``Ix u + Iy v + It = 0``. Over the
    window that is least squares ``(AᵀA) [u, v]ᵀ = Aᵀb``, and the step is repeated
    (Gauss–Newton) until the update is smaller than ``eps``. With ``levels > 1``
    the same solver runs coarse-to-fine.
    """
    g1 = gray_f64(image1)
    g2 = gray_f64(image2)
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if levels < 1:
        raise ValueError("levels must be >= 1")

    pyramids = [(g1, g2)]
    for _ in range(levels - 1):
        pyramids.append((cv2.pyrDown(pyramids[-1][0]), cv2.pyrDown(pyramids[-1][1])))

    # Displacement accumulated in full-resolution pixels. Dividing by ``scale``
    # turns it into the initial guess at that pyramid level.
    guess_full = np.zeros_like(pts)
    result = None
    for level in range(levels - 1, -1, -1):
        scale = float(2 ** level)
        result = _lk_level(
            pyramids[level][0], pyramids[level][1], pts / scale, guess_full / scale,
            window, max_iter, eps,
        )
        guess_full = result["displacement"] * scale
    assert result is not None
    result["displacement"] = guess_full
    result["points"] = pts + guess_full
    return result


def lk_worked_example(image1: np.ndarray, image2: np.ndarray, x: float, y: float,
                      window: int = 5) -> dict:
    """
    One Gauss–Newton step at ``(x, y)``, with the window values kept so they
    can be printed. This is the "show the workout" companion to ``lucas_kanade``.
    """
    g1 = gray_f64(image1)
    g2 = gray_f64(image2)
    ix_img, iy_img = _gradients(g1)
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    yy, xx = np.meshgrid(y + offsets, x + offsets, indexing="ij")
    template = bilinear(g1, xx, yy)
    ix = bilinear(ix_img, xx, yy)
    iy = bilinear(iy_img, xx, yy)
    warped = bilinear(g2, xx, yy)
    flat = np.isfinite(template) & np.isfinite(ix) & np.isfinite(iy) & np.isfinite(warped)
    gx = ix[flat].ravel()
    gy = iy[flat].ravel()
    residual = (warped - template)[flat].ravel()
    if gx.size < 4:
        raise ValueError("That point is too close to the image edge for a 5×5 window.")
    ata = np.array([[np.dot(gx, gx), np.dot(gx, gy)],
                    [np.dot(gx, gy), np.dot(gy, gy)]], dtype=np.float64)
    atb = -np.array([np.dot(gx, residual), np.dot(gy, residual)])
    try:
        step = np.linalg.solve(ata, atb)
    except np.linalg.LinAlgError:
        step = np.array([np.nan, np.nan])
    # One bilinear lookup, written as the four neighbor intensities.
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    a = float(x - x0)
    b = float(y - y0)
    i00 = float(g1[y0, x0])
    i10 = float(g1[y0, x0 + 1])
    i01 = float(g1[y0 + 1, x0])
    i11 = float(g1[y0 + 1, x0 + 1])
    reconstructed = (1 - a) * (1 - b) * i00 + a * (1 - b) * i10 + (1 - a) * b * i01 + a * b * i11
    eigenvalues = np.linalg.eigvalsh(ata)
    return {
        "point": [float(x), float(y)],
        "window": int(window),
        "template": template,
        "Ix": ix,
        "Iy": iy,
        "warped_I2": warped,
        "ATA": ata,
        "ATb": atb,
        "step_uv": step,
        "eigenvalues": eigenvalues,
        "bilinear": {
            "x0": x0, "y0": y0, "a": a, "b": b,
            "I00": i00, "I10": i10, "I01": i01, "I11": i11,
            "value": reconstructed,
        },
    }


def format_worked_example(example: dict) -> str:
    """Plain-text workout for one tracked point."""
    b = example["bilinear"]
    lines = [
        f"Point (x, y) = ({example['point'][0]:.4f}, {example['point'][1]:.4f}), window {example['window']}×{example['window']}",
        "",
        "Template I1 (bilinear samples of the window):",
        np.array2string(example["template"], precision=4, suppress_small=True),
        "",
        "Ix:",
        np.array2string(example["Ix"], precision=4, suppress_small=True),
        "",
        "Iy:",
        np.array2string(example["Iy"], precision=4, suppress_small=True),
        "",
        "I2 at the same window (first iteration, zero displacement):",
        np.array2string(example["warped_I2"], precision=4, suppress_small=True),
        "",
        "AᵀA =",
        np.array2string(example["ATA"], precision=6, suppress_small=True),
        f"Aᵀb = {np.array2string(example['ATb'], precision=6)}",
        f"eigenvalues of AᵀA (small, large) = {example['eigenvalues'][0]:.6f}, {example['eigenvalues'][1]:.6f}",
        (
            "AᵀA is singular, so this window does not determine (u, v)."
            if not np.all(np.isfinite(example["step_uv"]))
            else f"solve (AᵀA) [u, v]ᵀ = Aᵀb  →  (u, v) = ({example['step_uv'][0]:.6f}, {example['step_uv'][1]:.6f})"
        ),
        "",
        "Bilinear lookup of I1 at the point itself:",
        f"  floor = ({b['x0']}, {b['y0']}), a = {b['a']:.6f}, b = {b['b']:.6f}",
        f"  I00 = {b['I00']:.4f}  I10 = {b['I10']:.4f}  I01 = {b['I01']:.4f}  I11 = {b['I11']:.4f}",
        f"  (1-a)(1-b)I00 + a(1-b)I10 + (1-a)b I01 + ab I11 = {b['value']:.6f}",
    ]
    if example["eigenvalues"][0] < 1e-2 * max(example["eigenvalues"][1], 1e-12):
        lines.append("")
        lines.append("The smaller eigenvalue is much less than the larger one: this window is an edge, not a corner.")
    return "\n".join(lines)


def opencv_lk(image1: np.ndarray, image2: np.ndarray, points, window: int = 21, levels: int = 3) -> dict:
    """Pyramidal Lucas–Kanade from OpenCV, used as the second predictor."""
    g1 = gray_f64(image1).astype(np.uint8)
    g2 = gray_f64(image2).astype(np.uint8)
    pts = np.atleast_2d(np.asarray(points, dtype=np.float32)).reshape(-1, 1, 2)
    nxt, status, err = cv2.calcOpticalFlowPyrLK(
        g1, g2, pts, None,
        winSize=(window, window), maxLevel=levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    return {
        "points": nxt.reshape(-1, 2).astype(np.float64),
        "status": status.reshape(-1).astype(bool),
        "error": err.reshape(-1).astype(np.float64) if err is not None else None,
    }


def template_match_point(image1: np.ndarray, image2: np.ndarray, x: float, y: float,
                         window: int = 15, search: int = 48) -> list[float] | None:
    """
    Sub-pixel cross-check: normalized correlation of a patch, then a parabola
    through the peak and its two neighbors.
    """
    g1 = gray_f64(image1).astype(np.float32)
    g2 = gray_f64(image2).astype(np.float32)
    half = window // 2
    xi, yi = int(round(x)), int(round(y))
    if yi - half < 0 or xi - half < 0 or yi + half + 1 > g1.shape[0] or xi + half + 1 > g1.shape[1]:
        return None
    patch = g1[yi - half:yi + half + 1, xi - half:xi + half + 1]
    y0, y1 = max(0, yi - search), min(g2.shape[0], yi + search)
    x0, x1 = max(0, xi - search), min(g2.shape[1], xi + search)
    region = g2[y0:y1, x0:x1]
    if region.shape[0] < patch.shape[0] or region.shape[1] < patch.shape[1]:
        return None
    score = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
    _, _, _, peak = cv2.minMaxLoc(score)
    px, py = peak
    dx = dy = 0.0
    if 0 < px < score.shape[1] - 1:
        left, mid, right = score[py, px - 1], score[py, px], score[py, px + 1]
        denom = left - 2 * mid + right
        if abs(denom) > 1e-8:
            dx = 0.5 * (left - right) / denom
    if 0 < py < score.shape[0] - 1:
        up, mid, down = score[py - 1, px], score[py, px], score[py + 1, px]
        denom = up - 2 * mid + down
        if abs(denom) > 1e-8:
            dy = 0.5 * (up - down) / denom
    # matchTemplate reports the top-left of the patch; add the center offset.
    return [float(x0 + px + dx + half), float(y0 + py + dy + half)]


def dense_flow(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Farnebäck dense flow, shape (H, W, 2), in pixels of the given frames."""
    g1 = gray_f64(prev).astype(np.uint8)
    g2 = gray_f64(nxt).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(g1, g2, None, **FARNEBACK)


def flow_to_hsv(flow: np.ndarray, clip_mag: float | None = None) -> np.ndarray:
    """Hue is direction, brightness is speed. Returns BGR uint8."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    if clip_mag is None:
        clip_mag = float(np.percentile(mag, 99)) or 1.0
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / clip_mag * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_arrows(image: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
    """Yellow arrows on a copy of ``image``. ``image`` is BGR or gray."""
    if image.ndim == 2:
        canvas = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
        if canvas.dtype != np.uint8:
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = float(flow[y, x, 0]), float(flow[y, x, 1])
            if dx * dx + dy * dy < 0.25:
                continue
            end = (int(round(x + dx)), int(round(y + dy)))
            cv2.arrowedLine(canvas, (x, y), end, (0, 220, 255), 1, tipLength=0.35, line_type=cv2.LINE_AA)
    return canvas


def moving_mask(flow: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, float]:
    """Pixels whose speed exceeds ``threshold`` (default: twice the median, at least 0.8 px)."""
    mag = np.linalg.norm(flow, axis=2)
    if threshold is None:
        threshold = max(0.8, float(np.median(mag)) * 2.0)
    mask = (mag > threshold).astype(np.uint8) * 255
    return mask, float(threshold)


def direction_histogram(flow: np.ndarray, bins: int = 16, min_mag: float = 0.5) -> dict:
    mag = np.linalg.norm(flow, axis=2)
    ang = np.arctan2(flow[..., 1], flow[..., 0])
    sel = mag > min_mag
    counts, edges = np.histogram(ang[sel], bins=bins, range=(-np.pi, np.pi))
    return {
        "counts": counts.astype(int).tolist(),
        "bin_edges_deg": np.degrees(edges).tolist(),
        "min_mag": min_mag,
    }


def focus_of_expansion(flow: np.ndarray, min_mag: float = 0.8, stride: int = 8) -> dict:
    """
    Least squares focus of expansion.

    A flow vector (u, v) at (x, y) points away from (fx, fy):
    (x - fx) v - (y - fy) u = 0, so [v, -u] [fx, fy]ᵀ = x v - y u.
    """
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    sample = flow[::stride, ::stride]
    u, v = sample[..., 0], sample[..., 1]
    mag = np.hypot(u, v)
    sel = mag > min_mag
    if int(sel.sum()) < 20:
        return {"x": None, "y": None, "residual": None, "n": int(sel.sum())}
    A = np.column_stack([v[sel], -u[sel]])
    b = xs[sel] * v[sel] - ys[sel] * u[sel]
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    residual = float(np.mean(np.abs(A @ sol - b)))
    return {"x": float(sol[0]), "y": float(sol[1]), "residual": residual, "n": int(sel.sum())}


def ego_motion(prev: np.ndarray, flow: np.ndarray) -> dict:
    """Dominant translation (median flow) and a homography fit on a sparse grid."""
    mag = np.linalg.norm(flow, axis=2)
    median = np.median(flow.reshape(-1, 2), axis=0)
    h, w = flow.shape[:2]
    step = max(16, min(h, w) // 20)
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    src = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    dst = src + flow[ys, xs].reshape(-1, 2).astype(np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    inlier_ratio = float(mask.mean()) if mask is not None else 0.0
    return {
        "median_flow": [float(median[0]), float(median[1])],
        "median_speed": float(np.hypot(*median)),
        "mean_speed": float(mag.mean()),
        "homography_inlier_ratio": inlier_ratio,
        "homography": None if H is None else H.tolist(),
    }


def parallax_cue(flow: np.ndarray) -> dict:
    """
    Near objects move faster than far ones under sideways or forward camera motion.

    Compare the median speed of the fastest quartile with the slowest quartile.
    """
    mag = np.linalg.norm(flow, axis=2).ravel()
    slow = float(np.percentile(mag, 25))
    fast = float(np.percentile(mag, 75))
    return {
        "slow_quartile_median_region": slow,
        "fast_quartile": fast,
        "ratio": float(fast / slow) if slow > 1e-6 else None,
    }


def evaluate_pair(image1: np.ndarray, image2: np.ndarray, points, ground_truth=None,
                  example_index: int = 0, window: int = 21) -> dict:
    """
    Predict where ``points`` go, with the from-scratch tracker, OpenCV's
    pyramidal tracker, and a template-match cross-check. ``ground_truth`` is
    the clicked location in the second frame, when the user has provided it.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if pts.shape[0] < 1:
        raise ValueError("Pick at least one point.")
    scratch = lucas_kanade(image1, image2, pts, window=window, levels=3)
    classic = opencv_lk(image1, image2, pts)
    matched = [template_match_point(image1, image2, float(p[0]), float(p[1])) for p in pts]
    index = int(np.clip(example_index, 0, len(pts) - 1))
    example = lk_worked_example(image1, image2, float(pts[index, 0]), float(pts[index, 1]), window=5)
    gt = None if ground_truth is None else np.atleast_2d(np.asarray(ground_truth, dtype=np.float64))

    def _err(pred):
        if gt is None or len(gt) != len(pts):
            return None
        return np.linalg.norm(np.asarray(pred, dtype=np.float64) - gt, axis=1)

    scratch_err = _err(scratch["points"])
    opencv_err = _err(classic["points"])
    match_arr = np.array([p if p is not None else [np.nan, np.nan] for p in matched], dtype=np.float64)
    match_err = _err(match_arr)
    return {
        "points": pts,
        "scratch": scratch["points"],
        "scratch_status": scratch["status"],
        "eigenvalues": scratch["eigenvalues"],
        "opencv": classic["points"],
        "opencv_status": classic["status"],
        "template": match_arr,
        "ground_truth": gt,
        "scratch_error": scratch_err,
        "opencv_error": opencv_err,
        "template_error": match_err,
        "example_index": index,
        "example": example,
        "example_text": format_worked_example(example),
    }


def interpret_motion(summary: dict) -> str:
    """A short reading of what the flow field is showing."""
    ego = summary["ego_motion"]
    foe = summary["focus_of_expansion"]
    parallax = summary["parallax"]
    moving = summary["moving_fraction_mean"]
    parts = []
    if ego["median_speed"] < 0.4 and moving < 0.35:
        parts.append(
            "The camera is nearly still: most pixels have almost no flow, and the bright "
            "regions in the mask are the things that actually moved."
        )
    elif ego["homography_inlier_ratio"] > 0.6 and (moving > 0.5 or ego["median_speed"] > 1.0):
        parts.append(
            "Flow covers most of the frame and a single homography explains a large share of it, "
            "which is what a moving camera does to a distant or flat scene."
        )
    else:
        parts.append(
            "The field mixes camera motion and independent object motion: a homography does not "
            "explain every pixel, and the mask keeps the faster regions."
        )
    if foe["x"] is not None:
        inside = 0 <= foe["x"] < summary["width"] and 0 <= foe["y"] < summary["height"]
        if inside and ego["median_speed"] > 0.3:
            parts.append(
                f"The focus of expansion sits inside the frame near ({foe['x']:.0f}, {foe['y']:.0f}), "
                "so the camera is moving roughly toward that image point."
            )
    if parallax["ratio"] is not None and parallax["ratio"] > 1.8 and ego["median_speed"] > 0.4:
        parts.append(
            f"Speeds differ across the frame (fast/slow quartile ratio {parallax['ratio']:.1f}). "
            "Under the same camera motion, closer surfaces move faster, so that ratio is a depth cue."
        )
    return " ".join(parts)


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= width:
        out = image
    else:
        scale = width / w
        out = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    # H.264 wants even dimensions.
    h2, w2 = out.shape[:2]
    return out[: h2 - h2 % 2, : w2 - w2 % 2]


def _video_writer(path: Path, fps: float):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=7, macro_block_size=1,
    )


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def good_features(gray: np.ndarray, max_corners: int = 200) -> np.ndarray:
    pts = cv2.goodFeaturesToTrack(gray.astype(np.uint8), max_corners, 0.01, 12)
    if pts is None:
        return np.zeros((0, 2), np.float64)
    return pts.reshape(-1, 2).astype(np.float64)


def process_video(video_path: str | Path, out_dir: str | Path, start_sec: float = 0.0,
                  duration_sec: float = 30.0, max_width: int = 960, progress=None) -> dict:
    """
    Dense flow, arrow overlay, sparse trails, and motion statistics for a clip.

    Writes H.264 MP4s (browser-playable) and ``stats.json`` into ``out_dir``.
    ``progress(fraction, message)`` is optional.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps < 1:
        fps = 30.0
    start_frame = int(round(start_sec * fps))
    n_frames = max(2, int(round(duration_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ValueError("The video has no frame at the requested start time.")
    first = resize_width(first, max_width)
    height, width = first.shape[:2]

    writers = {
        "side_by_side": _video_writer(out_dir / "side_by_side.mp4", fps),
        "hsv": _video_writer(out_dir / "flow_hsv.mp4", fps),
        "arrows": _video_writer(out_dir / "flow_arrows.mp4", fps),
        "tracks": _video_writer(out_dir / "tracks.mp4", fps),
        "original": _video_writer(out_dir / "original.mp4", fps),
    }
    mean_speed = []
    moving_fraction = []
    histogram = None
    ego = None
    foe = None
    parallax = None
    best_speed = -1.0
    clip_mag = None
    prev = first
    prev_gray = gray_f64(prev).astype(np.uint8)
    features = good_features(prev_gray)
    trails: list[list[tuple[int, int]]] = [[(int(p[0]), int(p[1]))] for p in features]
    alive = np.ones(len(features), dtype=bool)
    mask_written = False

    def report(i, message):
        if progress:
            progress(min(0.99, i / n_frames), message)

    report(0, "reading frames")
    for index in range(1, n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_width(frame, max_width)
        flow = dense_flow(prev, frame)
        mag = np.linalg.norm(flow, axis=2)
        speed = float(mag.mean())
        mean_speed.append(speed)
        if clip_mag is None:
            clip_mag = float(np.percentile(mag, 99)) or 1.0
        hsv = flow_to_hsv(flow, clip_mag)
        arrows = draw_arrows(frame, flow)
        mask, thresh = moving_mask(flow)
        moving_fraction.append(float((mask > 0).mean()))
        if speed > best_speed:
            best_speed = speed
            histogram = direction_histogram(flow)
            ego = ego_motion(prev, flow)
            foe = focus_of_expansion(flow)
            parallax = parallax_cue(flow)
            cv2.imwrite(str(out_dir / "moving_mask.png"), mask)
            mask_written = True
            ego["mask_threshold_px"] = thresh

        gray = gray_f64(frame).astype(np.uint8)
        if len(features):
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, features.reshape(-1, 1, 2).astype(np.float32), None,
                winSize=(21, 21), maxLevel=3,
            )
            nxt = nxt.reshape(-1, 2)
            status = status.reshape(-1).astype(bool)
            for i, ok_pt in enumerate(status):
                if not alive[i]:
                    continue
                if not ok_pt:
                    alive[i] = False
                    continue
                trails[i].append((int(round(nxt[i, 0])), int(round(nxt[i, 1]))))
            features = nxt
        track_canvas = frame.copy()
        for trail, on in zip(trails, alive):
            if len(trail) < 2:
                continue
            color = (40, 220, 80) if on else (160, 160, 160)
            pts = np.array(trail[-40:], np.int32).reshape(-1, 1, 2)
            cv2.polylines(track_canvas, [pts], False, color, 2, cv2.LINE_AA)

        side = np.hstack([frame, hsv, arrows])
        # Keep the side-by-side from getting enormous.
        if side.shape[1] > 1600:
            scale = 1600 / side.shape[1]
            side = cv2.resize(side, (1600, int(side.shape[0] * scale)))
            sh, sw = side.shape[:2]
            side = side[: sh - sh % 2, : sw - sw % 2]
        writers["original"].append_data(_bgr_to_rgb(frame))
        writers["hsv"].append_data(_bgr_to_rgb(hsv))
        writers["arrows"].append_data(_bgr_to_rgb(arrows))
        writers["tracks"].append_data(_bgr_to_rgb(track_canvas))
        writers["side_by_side"].append_data(_bgr_to_rgb(side))
        prev, prev_gray = frame, gray
        report(index, f"frame {index}")

    cap.release()
    for writer in writers.values():
        writer.close()
    if not mean_speed:
        raise ValueError("Need at least two frames in the selected clip.")

    summary = {
        "video": str(video_path),
        "fps": fps,
        "start_sec": start_sec,
        "frames": len(mean_speed) + 1,
        "width": width,
        "height": height,
        "mean_speed": mean_speed,
        "mean_speed_average": float(np.mean(mean_speed)),
        "moving_fraction": moving_fraction,
        "moving_fraction_mean": float(np.mean(moving_fraction)),
        "direction_histogram": histogram,
        "ego_motion": ego,
        "focus_of_expansion": foe,
        "parallax": parallax,
        "mask_written": mask_written,
        "files": {
            "original": "original.mp4",
            "hsv": "flow_hsv.mp4",
            "arrows": "flow_arrows.mp4",
            "tracks": "tracks.mp4",
            "side_by_side": "side_by_side.mp4",
            "mask": "moving_mask.png" if mask_written else None,
        },
    }
    summary["interpretation"] = interpret_motion(summary)
    (out_dir / "stats.json").write_text(json.dumps(summary), encoding="utf-8")
    if progress:
        progress(1.0, "done")
    return summary
