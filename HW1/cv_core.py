"""
cv_core.py
==========

README — SHARED COMPUTER-VISION CORE FOR MODULE 2
-------------------------------------------------

This module is imported by the command-line scripts in this folder and by the
web application (../webapp). It is not meant to be run directly, but it has a
self-check:

    cd /Users/home/GSU/Computer_Vision/Computer_Vision
    .venv/bin/python HW1/cv_core.py          # prints a synthetic self-test

WHAT IT PROVIDES
  Step 1  calibrate()                      checkerboard calibration (OpenCV)
  Step 2  reference_from_checkerboard()    plane homography from a board in the scene
          reference_from_rectangle()       plane homography from 4 clicked corners
          measure_homography()             pixels -> plane (X, Y) via H^-1
          measure_pinhole()                pixels -> plane (X, Y) via X = Z (u - cx) / fx
  Step 3  error_statistics()               MAE, RMSE, bias, std, 95% CI, % error

PERSPECTIVE PROJECTION MODEL
  A world point X = [X, Y, Z]^T projects to pixel u = [u, v]^T by

      lambda [u, v, 1]^T = K [R | t] [X, Y, Z, 1]^T

  after lens distortion has been removed. For points on the plane Z = 0 this
  reduces to the homography

      lambda [u, v, 1]^T = H [X, Y, 1]^T,     H = K [r1  r2  t]

  and for a plane perpendicular to the optical axis at known depth Z it
  reduces to  X = Z (u - cx) / fx,  Y = Z (v - cy) / fy.

COORDINATE CONVENTIONS
  - Pixel coordinates are (u, v) = (column, row) in the full-resolution image,
    origin at the top-left pixel.
  - Images are always read in the camera's native sensor orientation (EXIF
    rotation ignored) so calibration and measurement pixels share one frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from measure_planar_object_2d import homography_dlt

DEFAULT_PATTERN = (9, 6)  # inner corners (columns, rows) of the course checkerboard

UNIT_TO_METERS = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "in": 0.0254,
    "ft": 0.3048,
}

# Two-sided 95% Student-t critical values for df = 1..30.
_T95 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------

def _is_heif(data: bytes) -> bool:
    return len(data) > 12 and data[4:8] == b"ftyp" and data[8:12] in {
        b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis",
    }


def load_image(source: bytes | str | Path) -> np.ndarray:
    """Decode JPEG/PNG/HEIC bytes or a file path into a BGR image (sensor orientation)."""
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    if _is_heif(data):
        import io

        import pillow_heif
        from PIL import Image

        pillow_heif.register_heif_opener()
        rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError("Unsupported or corrupt image file.")
    return img


# --------------------------------------------------------------------------
# Calibration container
# --------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """Pinhole intrinsics + Brown distortion (k1, k2, p1, p2, k3) at a given resolution."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_width: int
    image_height: int
    rms_px: float | None = None
    num_images: int | None = None
    pattern: tuple[int, int] | None = None
    square_size: float | None = None
    square_units: str | None = None
    source: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        out = {
            "camera_matrix": np.asarray(self.camera_matrix).tolist(),
            "dist_coeffs": np.asarray(self.dist_coeffs).ravel().tolist(),
            "image_size": [self.image_height, self.image_width],
            "rms_px": self.rms_px,
            "num_images": self.num_images,
            "checkerboard_inner_corners": list(self.pattern) if self.pattern else None,
            "square_size": self.square_size,
            "world_units": self.square_units,
            "source": self.source,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_json(cls, data: dict) -> "CameraCalibration":
        dist = np.zeros(5)
        raw = np.asarray(data["dist_coeffs"], dtype=np.float64).ravel()
        dist[: min(5, raw.size)] = raw[:5]
        rows, cols = data["image_size"]
        pattern = data.get("checkerboard_inner_corners")
        known = {
            "camera_matrix", "dist_coeffs", "image_size", "rms_px", "num_images",
            "checkerboard_inner_corners", "square_size", "world_units", "source",
        }
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=dist,
            image_width=int(cols),
            image_height=int(rows),
            rms_px=data.get("rms_px"),
            num_images=data.get("num_images"),
            pattern=tuple(pattern) if pattern else None,
            square_size=data.get("square_size"),
            square_units=data.get("world_units"),
            source=data.get("source", ""),
            extra={k: v for k, v in data.items() if k not in known},
        )

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        import json

        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def intrinsics_for(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Camera matrix for an image of a different resolution with the same aspect ratio.

        Resizing an image by s scales fx, fy, cx, cy by s; distortion coefficients
        act on normalized coordinates and are unchanged.
        """
        if (width, height) == (self.image_width, self.image_height):
            return self.camera_matrix.copy(), self.dist_coeffs.copy()
        sx = width / self.image_width
        sy = height / self.image_height
        if abs(sx - sy) > 0.01:
            raise ValueError(
                f"Image is {width}x{height} but calibration is "
                f"{self.image_width}x{self.image_height}. Use the same camera "
                "orientation (landscape vs portrait) as the calibration photos."
            )
        k = self.camera_matrix.copy()
        k[0, :] *= sx
        k[1, :] *= sy
        return k, self.dist_coeffs.copy()


# --------------------------------------------------------------------------
# Checkerboard detection + Step 1 calibration
# --------------------------------------------------------------------------

def checkerboard_object_points(pattern: tuple[int, int], square_size: float) -> np.ndarray:
    """(N, 3) board corners on Z = 0, row-major with x varying fastest (OpenCV order)."""
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return grid * float(square_size)


def detect_checkerboard(
    image: np.ndarray,
    pattern: tuple[int, int] = DEFAULT_PATTERN,
    detect_sizes: Sequence[int] = (1600, 3200),
) -> np.ndarray | None:
    """
    Find inner checkerboard corners, returning (N, 2) full-resolution pixels or None.

    Detection runs on a downscaled copy (fast on 12 MP phone photos), then every
    corner is refined with sub-pixel accuracy on the full-resolution image.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

    corners = None
    for max_dim in detect_sizes:
        scale = min(1.0, max_dim / max(h, w))
        small = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        found, pts = cv2.findChessboardCornersSB(small, pattern, flags=flags)
        if found:
            corners = pts.reshape(-1, 2).astype(np.float32) / scale
            break
        if scale == 1.0:
            break
    if corners is None:
        return None

    cols, _rows = pattern
    grid = corners.reshape(-1, cols, 2)
    square_px = float(np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2)))
    half_win = int(np.clip(round(square_px * 0.3), 3, 25))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-3)
    refined = cv2.cornerSubPix(
        gray, corners.reshape(-1, 1, 2), (half_win, half_win), (-1, -1), criteria
    )
    return refined.reshape(-1, 2)


@dataclass
class CalibrationView:
    name: str
    detected: bool
    corners: np.ndarray | None = None
    reproj_rms_px: float | None = None
    note: str = ""
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None


def calibrate(
    images: Iterable[tuple[str, np.ndarray]],
    pattern: tuple[int, int] = DEFAULT_PATTERN,
    square_size: float = 1.0,
    square_units: str = "in",
    min_images: int = 20,
) -> tuple[CameraCalibration, list[CalibrationView]]:
    """
    Zhang's checkerboard calibration via cv2.calibrateCamera.

    Model: fx, fy, cx, cy, k1, k2 (tangential distortion and k3 fixed at 0, the
    same model as MATLAB's default). A richer model over-fits phone photos in
    which the board rarely reaches the image corners.
    """
    objp = checkerboard_object_points(pattern, square_size)
    views: list[CalibrationView] = []
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    size: tuple[int, int] | None = None

    for name, img in images:
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        if (w, h) != size:
            views.append(CalibrationView(
                name, False,
                note=f"skipped: {w}x{h} differs from {size[0]}x{size[1]} (rotate/orientation)",
            ))
            continue
        corners = detect_checkerboard(img, pattern)
        if corners is None:
            views.append(CalibrationView(name, False, note="checkerboard not found"))
            continue
        views.append(CalibrationView(name, True, corners=corners))
        obj_points.append(objp)
        img_points.append(corners.reshape(-1, 1, 2).astype(np.float32))

    used = [v for v in views if v.detected]
    if len(used) < min_images:
        raise ValueError(
            f"Need at least {min_images} images with a detected {pattern[0]}x{pattern[1]} "
            f"board; found {len(used)} of {len(views)}."
        )
    assert size is not None

    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, size, None, None, flags=flags
    )
    for view, op, ip, rvec, tvec in zip(used, obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rvec, tvec, k, dist)
        diff = proj.reshape(-1, 2) - ip.reshape(-1, 2)
        view.reproj_rms_px = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        view.rvec, view.tvec = rvec.ravel(), tvec.ravel()

    dist5 = np.zeros(5)
    dist5[: min(5, dist.size)] = dist.ravel()[:5]
    calib = CameraCalibration(
        camera_matrix=k,
        dist_coeffs=dist5,
        image_width=size[0],
        image_height=size[1],
        rms_px=float(rms),
        num_images=len(used),
        pattern=tuple(pattern),
        square_size=float(square_size),
        square_units=square_units,
        source="OpenCV calibrateCamera (fx, fy, cx, cy, k1, k2)",
    )
    return calib, views


# --------------------------------------------------------------------------
# Lens undistortion
# --------------------------------------------------------------------------

def _distort_normalized(xy: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply Brown distortion to normalized points; also return the 2x2 Jacobians."""
    k1, k2, p1, p2, k3 = d
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    d_radial = k1 + 2 * k2 * r2 + 3 * k3 * r2**2  # d(radial)/d(r2)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    jac = np.empty((xy.shape[0], 2, 2))
    jac[:, 0, 0] = radial + 2 * x * x * d_radial + 2 * p1 * y + 6 * p2 * x
    jac[:, 0, 1] = 2 * x * y * d_radial + 2 * p1 * x + 2 * p2 * y
    jac[:, 1, 0] = 2 * x * y * d_radial + 2 * p1 * x + 2 * p2 * y
    jac[:, 1, 1] = radial + 2 * y * y * d_radial + 6 * p1 * y + 2 * p2 * x
    return np.stack([xd, yd], axis=1), jac


def distortion_valid_radius(dist: np.ndarray) -> float:
    """
    Largest normalized radius where the radial model r_d = r (1 + k1 r^2 + k2 r^4 + k3 r^6)
    is still increasing. Beyond it the model folds back and cannot be inverted uniquely.
    """
    k1, k2, _p1, _p2, k3 = dist
    r = np.linspace(0.0, 3.0, 30001)
    deriv = 1 + 3 * k1 * r**2 + 5 * k2 * r**4 + 7 * k3 * r**6
    bad = np.nonzero(deriv <= 0)[0]
    return float(r[bad[0]]) if bad.size else float("inf")


class OutsideLensModelError(ValueError):
    """Raised when pixels fall where the fitted distortion model cannot be inverted."""


def undistort_points(
    uv: np.ndarray, k: np.ndarray, dist: np.ndarray, *, return_valid: bool = False
):
    """
    Remove lens distortion from pixel points, returning undistorted pixels.

    Newton's method on the forward distortion model; agrees with MATLAB's
    undistortPoints to < 0.05 px wherever the model is invertible. Near the image
    corners a fitted k2 < 0 makes the radial curve fold back, so no undistorted
    point exists there: with return_valid=False such points raise
    OutsideLensModelError, with return_valid=True they come back as NaN.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    k = np.asarray(k, dtype=np.float64)
    d = np.zeros(5)
    raw = np.asarray(dist, dtype=np.float64).ravel()
    d[: min(5, raw.size)] = raw[:5]

    hom = np.hstack([uv, np.ones((uv.shape[0], 1))])
    target = (np.linalg.inv(k) @ hom.T).T[:, :2]
    xy = target.copy()
    for _ in range(50):
        fwd, jac = _distort_normalized(xy, d)
        step = np.linalg.solve(jac, (fwd - target)[:, :, None])[:, :, 0]
        xy -= step
        if np.max(np.abs(step)) < 1e-12:
            break

    fwd, _ = _distort_normalized(xy, d)
    residual = np.linalg.norm(fwd - target, axis=1)
    valid = (residual < 1e-9) & (np.linalg.norm(xy, axis=1) < distortion_valid_radius(d))

    out = (k @ np.hstack([xy, np.ones((xy.shape[0], 1))]).T).T
    out = out[:, :2] / out[:, 2:3]
    out[~valid] = np.nan
    if return_valid:
        return out, valid
    if not valid.all():
        bad = ", ".join(f"({u:.0f}, {v:.0f})" for u, v in uv[~valid][:5])
        raise OutsideLensModelError(
            f"{int((~valid).sum())} point(s) lie outside the region where the lens "
            f"distortion model is valid: {bad}. Keep the object nearer the image center, "
            "or recalibrate with photos that place the checkerboard in the image corners."
        )
    return out


def valid_pixel_radius(calib: "CameraCalibration") -> float:
    """Approximate radius (pixels from the principal point) inside which undistortion is valid."""
    k1, k2, _p1, _p2, k3 = calib.dist_coeffs
    r = min(distortion_valid_radius(calib.dist_coeffs), 3.0)
    r_d = r * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    return float(r_d * calib.camera_matrix[0, 0])


# --------------------------------------------------------------------------
# Step 2: reference plane + measurement
# --------------------------------------------------------------------------

@dataclass
class PlaneReference:
    """Homography from plane (X, Y) to undistorted pixels, plus the plane pose if known."""

    homography: np.ndarray
    unit: str
    kind: str
    image_points: np.ndarray
    world_points: np.ndarray
    reproj_rms_px: float
    distance_to_plane: float | None = None
    distance_to_origin: float | None = None
    tilt_deg: float | None = None
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None

    def to_json(self) -> dict:
        def conv(v):
            return None if v is None else float(v)

        return {
            "kind": self.kind,
            "unit": self.unit,
            "homography": self.homography.tolist(),
            "image_points": self.image_points.tolist(),
            "world_points": self.world_points.tolist(),
            "reproj_rms_px": float(self.reproj_rms_px),
            "distance_to_plane": conv(self.distance_to_plane),
            "distance_to_origin": conv(self.distance_to_origin),
            "distance_to_plane_m": None if self.distance_to_plane is None
            else float(self.distance_to_plane * UNIT_TO_METERS[self.unit]),
            "tilt_deg": conv(self.tilt_deg),
        }


def _plane_reference(
    world_xy: np.ndarray,
    image_uv: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    unit: str,
    kind: str,
) -> PlaneReference:
    undist = undistort_points(image_uv, k, dist)
    h = homography_dlt(world_xy, undist)
    reproj = world_to_image(h, world_xy)
    rms = float(np.sqrt(np.mean(np.sum((reproj - undist) ** 2, axis=1))))

    obj3 = np.hstack([world_xy, np.zeros((world_xy.shape[0], 1))]).astype(np.float64)
    method = cv2.SOLVEPNP_IPPE if world_xy.shape[0] >= 4 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj3, image_uv.astype(np.float64), k, dist, flags=method)
    ref = PlaneReference(h, unit, kind, image_uv, world_xy, rms)
    if ok:
        rot, _ = cv2.Rodrigues(rvec)
        normal = rot[:, 2]
        tvec = tvec.ravel()
        ref.rvec, ref.tvec = rvec.ravel(), tvec
        ref.distance_to_plane = float(abs(normal @ tvec))
        ref.distance_to_origin = float(np.linalg.norm(tvec))
        ref.tilt_deg = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
    return ref


def reference_from_checkerboard(
    image: np.ndarray,
    calib: CameraCalibration,
    pattern: tuple[int, int] = DEFAULT_PATTERN,
    square_size: float = 1.0,
    unit: str = "in",
) -> PlaneReference:
    """Detect a checkerboard lying on the measurement plane and use it as the reference."""
    corners = detect_checkerboard(image, pattern)
    if corners is None:
        raise ValueError(f"No {pattern[0]}x{pattern[1]} checkerboard found in the image.")
    h, w = image.shape[:2]
    k, dist = calib.intrinsics_for(w, h)
    world = checkerboard_object_points(pattern, square_size)[:, :2].astype(np.float64)
    return _plane_reference(world, corners.astype(np.float64), k, dist, unit, "checkerboard")


def reference_from_rectangle(
    corners_uv: np.ndarray,
    width: float,
    height: float,
    calib: CameraCalibration,
    image_size: tuple[int, int],
    unit: str = "in",
) -> PlaneReference:
    """Reference from 4 clicked corners of a rectangle of known size (TL, TR, BR, BL)."""
    corners_uv = np.asarray(corners_uv, dtype=np.float64)
    if corners_uv.shape != (4, 2):
        raise ValueError("Click exactly 4 rectangle corners: top-left, top-right, bottom-right, bottom-left.")
    world = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    k, dist = calib.intrinsics_for(*image_size)
    return _plane_reference(world, corners_uv, k, dist, unit, "rectangle")


def world_to_image(h: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    pts = np.hstack([world_xy, np.ones((world_xy.shape[0], 1))])
    proj = (h @ pts.T).T
    return proj[:, :2] / proj[:, 2:3]


def measure_homography(
    h: np.ndarray, uv: np.ndarray, k: np.ndarray, dist: np.ndarray
) -> np.ndarray:
    """Distorted pixels -> plane coordinates: undistort, then [X, Y, 1]^T ~ H^-1 [u, v, 1]^T."""
    undist = undistort_points(uv, k, dist)
    pts = np.hstack([undist, np.ones((undist.shape[0], 1))])
    world = (np.linalg.inv(h) @ pts.T).T
    return world[:, :2] / world[:, 2:3]


def measure_pinhole(uv: np.ndarray, k: np.ndarray, dist: np.ndarray, depth: float) -> np.ndarray:
    """
    Distorted pixels -> plane coordinates for a fronto-parallel plane at depth Z.

        X = Z (u - cx) / fx,   Y = Z (v - cy) / fy

    Output is in the same unit as `depth`.
    """
    undist = undistort_points(uv, k, dist)
    pts = np.hstack([undist, np.ones((undist.shape[0], 1))])
    norm = (np.linalg.inv(k) @ pts.T).T
    return norm[:, :2] * float(depth)


def polygon_metrics(world_xy: np.ndarray, closed: bool) -> dict:
    """Edge lengths, perimeter and (for closed shapes) shoelace area."""
    world_xy = np.asarray(world_xy, dtype=np.float64)
    n = world_xy.shape[0]
    edges = [float(np.linalg.norm(world_xy[i + 1] - world_xy[i])) for i in range(n - 1)]
    if closed and n > 2:
        edges.append(float(np.linalg.norm(world_xy[0] - world_xy[-1])))
    area = None
    if closed and n > 2:
        x, y = world_xy[:, 0], world_xy[:, 1]
        area = float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return {
        "world_xy": world_xy.tolist(),
        "edge_lengths": edges,
        "perimeter": float(sum(edges)),
        "area": area,
    }


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    return value * UNIT_TO_METERS[from_unit] / UNIT_TO_METERS[to_unit]


# --------------------------------------------------------------------------
# Step 3: error statistics
# --------------------------------------------------------------------------

def error_statistics(ground_truth: Sequence[float], estimated: Sequence[float]) -> dict:
    """Accuracy statistics for paired (ground truth, estimate) lengths."""
    gt = np.asarray(ground_truth, dtype=np.float64)
    est = np.asarray(estimated, dtype=np.float64)
    mask = np.isfinite(gt) & np.isfinite(est) & (gt > 0)
    gt, est = gt[mask], est[mask]
    n = int(gt.size)
    if n == 0:
        return {"n": 0}

    err = est - gt
    abs_err = np.abs(err)
    pct = 100.0 * abs_err / gt
    std = float(np.std(err, ddof=1)) if n > 1 else 0.0
    t_crit = _T95[n - 2] if 2 <= n <= 31 else 1.96
    half = t_crit * std / math.sqrt(n) if n > 1 else float("nan")
    corr = float(np.corrcoef(gt, est)[0, 1]) if n > 2 and np.std(gt) > 0 else None
    slope = float(np.polyfit(gt, est, 1)[0]) if n > 1 and np.std(gt) > 0 else None

    return {
        "n": n,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(err.mean()),
        "std": std,
        "median_abs_error": float(np.median(abs_err)),
        "max_abs_error": float(abs_err.max()),
        "mape_percent": float(pct.mean()),
        "max_percent_error": float(pct.max()),
        "bias_ci95": [float(err.mean() - half), float(err.mean() + half)] if n > 1 else None,
        "pearson_r": corr,
        "fit_slope": slope,
    }


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _self_test() -> None:
    k = np.array([[3700.0, 0, 2856], [0, 3700.0, 2142], [0, 0, 1]])
    dist = np.array([0.09, -0.4, 0, 0, 0])
    rvec = np.array([0.25, -0.35, 0.05])
    tvec = np.array([-10.0, -6.0, 98.0])  # inches, ~2.5 m away

    world = np.array([[0, 0], [10, 0], [10, 7], [0, 7]], dtype=np.float64)
    obj = np.array([[2, 1.5], [8, 1.5], [8, 5.5], [2, 5.5]], dtype=np.float64)

    def project(xy):
        p3 = np.hstack([xy, np.zeros((len(xy), 1))])
        return cv2.projectPoints(p3, rvec, tvec, k, dist)[0].reshape(-1, 2)

    calib = CameraCalibration(k, dist, 5712, 4284)
    ref = reference_from_rectangle(project(world), 10, 7, calib, (5712, 4284))
    measured = measure_homography(ref.homography, project(obj), k, dist)
    edges = polygon_metrics(measured, closed=True)["edge_lengths"]
    print("Homography edges (true 6, 4, 6, 4):", [round(e, 4) for e in edges])
    print(f"PnP distance to plane: {ref.distance_to_plane:.2f} in, tilt {ref.tilt_deg:.1f} deg")

    pix = np.array([[400.0, 300.0], [2856, 2142], [5300, 4000]])
    rays = (np.linalg.inv(k) @ np.hstack([pix, np.ones((3, 1))]).T).T
    distorted = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, dist)[0].reshape(-1, 2)
    back, valid = undistort_points(distorted, k, dist, return_valid=True)
    print("Undistort round-trip error (px):", np.round(np.linalg.norm(back - pix, axis=1), 9), valid)

    print("Stats:", error_statistics([10, 20, 30], [10.1, 19.8, 30.3]))


if __name__ == "__main__":
    _self_test()
