"""Calibrated stereo measurement for Assignment 7.

One phone slides sideways along a ruler. The slide length is the baseline B.
A point that shifts by d pixels between the two views is at depth

    Z = f B / d

and its metric position is the pinhole back-projection of that depth. Sizes
are distances between those 3D points. A rectangle reports width and height,
a circle reports its diameter, and any other polygon reports every edge.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HW5 = Path(__file__).resolve().parent
SAMPLE_DIR = HW5 / "samples"
DEMO_LEFT = SAMPLE_DIR / "left.png"
DEMO_RIGHT = SAMPLE_DIR / "right.png"
DEMO_META = SAMPLE_DIR / "demo.json"

# A half-pixel of matching error, used only to show how depth error grows with Z^2.
DEFAULT_DISPARITY_ERROR_PX = 0.5


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_intrinsics(k: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """Scale K when an image is resized. src_wh and dst_wh are (width, height)."""
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    out = np.array(k, dtype=np.float64, copy=True)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def load_monocular(path: Path | str | None = None) -> dict:
    """Read the Module 2 phone calibration. image_size is stored as [height, width]."""
    path = Path(path) if path else ROOT / "HW1" / "camera_calibration.json"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    height, width = data["image_size"]
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).ravel()
    return {
        "k": np.asarray(data["camera_matrix"], dtype=np.float64),
        "dist": dist,
        "image_size": (int(width), int(height)),
    }


def projection(k: np.ndarray, r: np.ndarray | None = None, t: np.ndarray | None = None) -> np.ndarray:
    """P = K [R | t] with X_cam = R X + t."""
    r = np.eye(3) if r is None else np.asarray(r, dtype=np.float64)
    t = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64).ravel()
    return k @ np.hstack([r, t.reshape(3, 1)])


def parallel_pair(k: np.ndarray, baseline: float) -> tuple[np.ndarray, np.ndarray]:
    """Left camera at the origin, right camera slid +baseline along X.

    X_right = X_left - (B, 0, 0), so the right projection uses t = (-B, 0, 0).
    """
    p1 = projection(k)
    p2 = projection(k, t=np.array([-baseline, 0.0, 0.0]))
    return p1, p2


def project_points(points: np.ndarray, k: np.ndarray, r: np.ndarray | None = None, c: np.ndarray | None = None) -> np.ndarray:
    """Project world points. c is the camera center; X_cam = R (X - c)."""
    pts = np.asarray(points, dtype=np.float64)
    r = np.eye(3) if r is None else np.asarray(r, dtype=np.float64)
    c = np.zeros(3) if c is None else np.asarray(c, dtype=np.float64).ravel()
    cam = (r @ (pts - c).T).T
    pix = (k @ cam.T).T
    return pix[:, :2] / pix[:, 2:3]


def disparity_xyz(left_xy: np.ndarray, disparity: np.ndarray, k: np.ndarray, baseline: float) -> np.ndarray:
    """Z = f B / d, then X and Y from the left-image pinhole equations."""
    left_xy = np.asarray(left_xy, dtype=np.float64)
    disparity = np.asarray(disparity, dtype=np.float64).ravel()
    if np.any(disparity <= 1e-9):
        raise ValueError("Disparity must be positive. Click the same point in the right view, to the left of the left-view click.")
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    z = fx * baseline / disparity
    x = (left_xy[:, 0] - cx) * z / fx
    y = (left_xy[:, 1] - cy) * z / fy
    return np.column_stack([x, y, z])


def triangulate_dlt(left_xy: np.ndarray, right_xy: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Linear triangulation (DLT), one point at a time, in float64."""
    left_xy = np.asarray(left_xy, dtype=np.float64)
    right_xy = np.asarray(right_xy, dtype=np.float64)
    out = np.empty((len(left_xy), 3), dtype=np.float64)
    for i, ((u, v), (up, vp)) in enumerate(zip(left_xy, right_xy)):
        a = np.vstack([
            u * p1[2] - p1[0],
            v * p1[2] - p1[1],
            up * p2[2] - p2[0],
            vp * p2[2] - p2[1],
        ])
        _, _, vt = np.linalg.svd(a)
        x = vt[-1]
        out[i] = x[:3] / x[3]
    return out


def depth_uncertainty(z: np.ndarray, focal: float, baseline: float, disparity_error_px: float = DEFAULT_DISPARITY_ERROR_PX) -> np.ndarray:
    """dZ ≈ Z^2 dd / (f B). Farther points and a shorter slide both hurt."""
    z = np.asarray(z, dtype=np.float64)
    return z**2 * disparity_error_px / (focal * baseline)


def edge_lengths(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    nxt = np.roll(pts, -1, axis=0)
    return np.linalg.norm(nxt - pts, axis=1)


def rectangle_size(points: np.ndarray) -> dict:
    """Width is the mean of edges 0 and 2, height the mean of edges 1 and 3."""
    edges = edge_lengths(points)
    if len(edges) != 4:
        raise ValueError("A rectangle needs 4 corners, in order around the outline.")
    return {
        "edges_m": edges,
        "width_m": float((edges[0] + edges[2]) / 2),
        "height_m": float((edges[1] + edges[3]) / 2),
    }


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a unit normal and the centroid. The normal is the smallest singular vector."""
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    return normal, centroid


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis_u = np.cross(normal, helper)
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def fit_circle_2d(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic circle fit (Kasa). Exact when the points lie on a circle."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 3:
        raise ValueError("A circle needs at least 3 rim points.")
    x, y = xy[:, 0], xy[:, 1]
    design = np.column_stack([x, y, np.ones(len(xy))])
    target = -(x * x + y * y)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    center = np.array([-coef[0] / 2, -coef[1] / 2])
    radius_sq = center @ center - coef[2]
    if radius_sq <= 0:
        raise ValueError("Those points do not determine a circle.")
    return center, float(np.sqrt(radius_sq))


def circle_diameter(points: np.ndarray) -> dict:
    """Fit a plane, then a circle in that plane. Diameter is twice the radius."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 5:
        raise ValueError("Click at least 5 points around the rim.")
    normal, centroid = fit_plane(pts)
    axis_u, axis_v = plane_basis(normal)
    local = np.column_stack([(pts - centroid) @ axis_u, (pts - centroid) @ axis_v])
    _center, radius = fit_circle_2d(local)
    return {"diameter_m": float(2 * radius), "radius_m": float(radius), "normal": normal}


def z_first_edges(left_xy: np.ndarray, xyz: np.ndarray, focal: float) -> np.ndarray:
    """Assignment-1 style: edge length ≈ pixel length × Z / f, using the mean Z of the two ends."""
    left_xy = np.asarray(left_xy, dtype=np.float64)
    z = np.asarray(xyz, dtype=np.float64)[:, 2]
    n = len(left_xy)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        pixels = float(np.linalg.norm(left_xy[j] - left_xy[i]))
        out[i] = pixels * float((z[i] + z[j]) / 2) / focal
    return out


def measure_points(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    k: np.ndarray,
    baseline: float,
    shape: str,
    disparity_error_px: float = DEFAULT_DISPARITY_ERROR_PX,
) -> dict:
    """Measure one object from clicked correspondences. Both views are already the same size."""
    shape = shape.lower()
    if shape not in ("rectangle", "circle", "polygon"):
        raise ValueError("shape must be rectangle, circle, or polygon.")
    left_xy = np.atleast_2d(np.asarray(left_xy, dtype=np.float64))
    right_xy = np.atleast_2d(np.asarray(right_xy, dtype=np.float64))
    if left_xy.shape != right_xy.shape or left_xy.shape[1] != 2:
        raise ValueError("Left and right clicks must be the same number of [u, v] points.")
    needed = {"rectangle": 4, "circle": 5, "polygon": 3}[shape]
    if len(left_xy) < needed:
        raise ValueError(f"A {shape} needs at least {needed} points.")
    if shape == "rectangle" and len(left_xy) != 4:
        raise ValueError("A rectangle needs exactly 4 corners.")

    disparity = left_xy[:, 0] - right_xy[:, 0]
    xyz_d = disparity_xyz(left_xy, disparity, k, baseline)
    p1, p2 = parallel_pair(k, baseline)
    xyz_t = triangulate_dlt(left_xy, right_xy, p1, p2)
    focal = float(k[0, 0])
    rows = []
    dz = depth_uncertainty(xyz_d[:, 2], focal, baseline, disparity_error_px)
    for i in range(len(left_xy)):
        rows.append({
            "left": left_xy[i].tolist(),
            "right": right_xy[i].tolist(),
            "disparity_px": float(disparity[i]),
            "row_gap_px": float(left_xy[i, 1] - right_xy[i, 1]),
            "xyz_disparity": xyz_d[i].tolist(),
            "xyz_triangulated": xyz_t[i].tolist(),
            "z_uncertainty_m": float(dz[i]),
        })

    result: dict = {
        "shape": shape,
        "baseline_m": float(baseline),
        "focal_px": focal,
        "points": rows,
        "mean_z_m": float(np.mean(xyz_t[:, 2])),
        "triangulation_gap_m": float(np.max(np.abs(xyz_d - xyz_t))),
        "edges_m": edge_lengths(xyz_t).tolist(),
        "z_first_edges_m": z_first_edges(left_xy, xyz_d, focal).tolist(),
    }
    if shape == "rectangle":
        result.update(rectangle_size(xyz_t))
        result["edges_m"] = result["edges_m"].tolist()
    elif shape == "circle":
        result.update(circle_diameter(xyz_t))
        result["normal"] = result["normal"].tolist()
    return result


def match_along_row(
    left: np.ndarray,
    right: np.ndarray,
    u: float,
    v: float,
    window: int = 7,
    max_disparity: int | None = None,
    row_slack: int = 2,
) -> dict:
    """Find the same patch in the right image, searching along the row (and a few rows around it)."""
    left_g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    h, w = left_g.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if window < 2:
        window = 2
    if ui - window < 0 or vi - window < 0 or ui + window >= w or vi + window >= h:
        raise ValueError("That click is too close to the edge to match.")
    if max_disparity is None:
        max_disparity = max(16, w // 3)
    template = left_g[vi - window: vi + window + 1, ui - window: ui + window + 1].astype(np.float32)
    if float(template.std()) < 1e-3:
        raise ValueError("That patch has no texture, so it cannot be matched. Click a corner or a mark.")

    best = (-1.0, ui, vi)
    scores_at_best_row: list[tuple[int, float]] = []
    v0 = max(window, vi - row_slack)
    v1 = min(h - window - 1, vi + row_slack)
    u_lo = max(window, ui - max_disparity)
    u_hi = min(w - window - 1, ui + window)
    for vv in range(v0, v1 + 1):
        strip = right_g[vv - window: vv + window + 1, u_lo - window: u_hi + window + 1]
        if strip.shape[1] < template.shape[1]:
            continue
        score = cv2.matchTemplate(strip, template.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
        flat = score.ravel()
        peak = int(np.argmax(flat))
        value = float(flat[peak])
        uu = u_lo + peak
        if value > best[0]:
            best = (value, uu, vv)
            scores_at_best_row = [(u_lo + i, float(flat[i])) for i in range(len(flat))]
    if best[0] < 0.4:
        raise ValueError("No match along that row. The views may not be a sideways slide, or the point is hidden in the right view.")
    uu = float(best[1])
    # A perfect score means the integer pixel already matches. The parabola
    # then just fits noise in the neighbors and should be left alone.
    nearby = {col: val for col, val in scores_at_best_row}
    if best[0] < 0.999 and (uu - 1) in nearby and (uu + 1) in nearby:
        s_prev, s_mid, s_next = nearby[uu - 1], nearby[uu], nearby[uu + 1]
        denom = s_prev - 2 * s_mid + s_next
        if abs(denom) > 1e-9:
            uu = uu + 0.5 * (s_prev - s_next) / denom
    # The template was centered on the rounded click. Keep the fraction of the
    # original click, so a click at 100.4 whose patch matches at 68 reports 68.4.
    uu += u - ui
    vv = float(best[2]) + (v - vi)
    return {"u": float(uu), "v": vv, "score": float(best[0]), "disparity_px": float(u - uu)}


def match_points(left: np.ndarray, right: np.ndarray, points: np.ndarray, **kwargs) -> np.ndarray:
    found = [match_along_row(left, right, u, v, **kwargs) for u, v in np.asarray(points, dtype=np.float64)]
    return np.array([[m["u"], m["v"]] for m in found], dtype=np.float64), found


def checkerboard_object(pattern: tuple[int, int], square: float) -> np.ndarray:
    cols, rows = pattern
    obj = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj[:, :2] = grid * square
    return obj


def calibrate_stereo(
    object_points: list[np.ndarray],
    image_points1: list[np.ndarray],
    image_points2: list[np.ndarray],
    k: np.ndarray,
    dist: np.ndarray,
    image_size: tuple[int, int],
) -> dict:
    """Stereo calibration with the phone intrinsics held fixed. image_size is (width, height)."""
    if len(object_points) < 8:
        raise ValueError("Need at least 8 left/right checkerboard pairs.")
    dist = np.asarray(dist, dtype=np.float64).ravel()
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 150, 1e-8)
    rms, k1, d1, k2, d2, r, t, e, f = cv2.stereoCalibrate(
        object_points,
        image_points1,
        image_points2,
        np.array(k, dtype=np.float64, copy=True),
        dist.copy(),
        np.array(k, dtype=np.float64, copy=True),
        dist.copy(),
        image_size,
        flags=flags,
        criteria=criteria,
    )
    t = np.asarray(t, dtype=np.float64).ravel()
    return {
        "rms_px": float(rms),
        "k1": np.asarray(k1, dtype=np.float64),
        "d1": np.asarray(d1, dtype=np.float64).ravel(),
        "k2": np.asarray(k2, dtype=np.float64),
        "d2": np.asarray(d2, dtype=np.float64).ravel(),
        "r": np.asarray(r, dtype=np.float64),
        "t": t,
        "baseline_m": float(np.linalg.norm(t)),
        "e": np.asarray(e, dtype=np.float64),
        "f": np.asarray(f, dtype=np.float64),
        "image_size": [int(image_size[0]), int(image_size[1])],
    }


def rectify_maps(calibration: dict) -> dict:
    """Rectifying maps. After remap, corresponding points lie on the same row."""
    size = (int(calibration["image_size"][0]), int(calibration["image_size"][1]))
    r1, r2, p1, p2, q, _roi1, _roi2 = cv2.stereoRectify(
        calibration["k1"], calibration["d1"], calibration["k2"], calibration["d2"],
        size, calibration["r"], calibration["t"],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1 = cv2.initUndistortRectifyMap(calibration["k1"], calibration["d1"], r1, p1, size, cv2.CV_32FC1)
    map2 = cv2.initUndistortRectifyMap(calibration["k2"], calibration["d2"], r2, p2, size, cv2.CV_32FC1)
    return {"r1": r1, "r2": r2, "p1": p1, "p2": p2, "q": q, "map1": map1, "map2": map2}


def draw_clicks(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], closed: bool = True) -> np.ndarray:
    canvas = image.copy()
    pts = np.round(np.asarray(points, dtype=np.float64)).astype(int)
    if closed and len(pts) >= 2:
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    for i, (u, v) in enumerate(pts):
        cv2.circle(canvas, (int(u), int(v)), 5, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(i + 1), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def make_demo(seed: int = 3) -> dict:
    """A textured fronto-parallel plane, shifted by a known disparity.

    The right image is the left image slid left by `disparity` pixels, which is
    what a pure sideways baseline does to a plane facing the camera.
    """
    height, width = 480, 640
    rng = np.random.default_rng(seed)
    texture = rng.integers(0, 256, (height, width), dtype=np.uint8)
    fx = fy = 700.0
    cx, cy = (width - 1) / 2, (height - 1) / 2
    baseline = 0.20
    disparity = 32.0
    k = camera_matrix(fx, fy, cx, cy)
    left = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    shapes = {
        "rectangle": np.array([[90, 70], [290, 70], [290, 230], [90, 230]], dtype=np.float64),
        "circle": _circle_pixels(np.array([470.0, 150.0]), 60.0, 8),
        "polygon": _regular_polygon_pixels(np.array([230.0, 360.0]), 55.0, 5),
    }
    # Guides are drawn in the browser. Baking them into the texture moves the match off the true shift.
    right = np.zeros_like(left)
    shift = int(disparity)
    right[:, : width - shift] = left[:, shift:]
    z = fx * baseline / disparity
    truth = {}
    for name, pts in shapes.items():
        world = _pixels_to_plane(pts, z, k)
        if name == "rectangle":
            truth[name] = rectangle_size(world)
            truth[name]["edges_m"] = truth[name]["edges_m"].tolist()
        elif name == "circle":
            truth[name] = {"diameter_m": float(2 * 60.0 * z / fx)}
        else:
            truth[name] = {"edges_m": edge_lengths(world).tolist()}
        truth[name] = _jsonable(truth[name])
    meta = {
        "width": width,
        "height": height,
        "k": k.tolist(),
        "baseline_m": baseline,
        "disparity_px": disparity,
        "z_m": z,
        "shapes": {name: pts.tolist() for name, pts in shapes.items()},
        "truth": truth,
        "note": "Synthetic plane. The right photo is the left photo shifted by the disparity. Outlines are guides, not part of the texture.",
    }
    return {"left": left, "right": right, "meta": meta, "shapes": shapes}


def _circle_pixels(center: np.ndarray, radius: float, count: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)])


def _regular_polygon_pixels(center: np.ndarray, radius: float, count: int) -> np.ndarray:
    angles = -np.pi / 2 + np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)])


def _pixels_to_plane(pixels: np.ndarray, z: float, k: np.ndarray) -> np.ndarray:
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    x = (pixels[:, 0] - cx) * z / fx
    y = (pixels[:, 1] - cy) * z / fy
    return np.column_stack([x, y, np.full(len(pixels), z)])


def write_demo(directory: Path | None = None) -> dict:
    directory = directory or SAMPLE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    demo = make_demo()
    if not cv2.imwrite(str(directory / "left.png"), demo["left"]):
        raise RuntimeError("Could not write the left sample.")
    if not cv2.imwrite(str(directory / "right.png"), demo["right"]):
        raise RuntimeError("Could not write the right sample.")
    (directory / "demo.json").write_text(json.dumps(demo["meta"], indent=2), encoding="utf-8")
    (directory / "README.txt").write_text(
        "Synthetic stereo pair used by the website before real photos are added.\n"
        "left.png and right.png differ by a 32 px sideways shift. demo.json holds K, the baseline, and the true sizes.\n",
        encoding="utf-8",
    )
    return demo["meta"]


def load_demo() -> tuple[np.ndarray, np.ndarray, dict]:
    if not DEMO_META.is_file():
        write_demo()
    left = cv2.imread(str(DEMO_LEFT), cv2.IMREAD_COLOR)
    right = cv2.imread(str(DEMO_RIGHT), cv2.IMREAD_COLOR)
    meta = json.loads(DEMO_META.read_text(encoding="utf-8"))
    if left is None or right is None:
        raise FileNotFoundError("The sample stereo pair is missing.")
    return left, right, meta


def result_to_json(result: dict) -> dict:
    return _jsonable(result)
