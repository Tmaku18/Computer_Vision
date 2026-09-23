#!/usr/bin/env python3
"""
measure_planar_object_2d.py
===========================

README — PLANAR HOMOGRAPHY LIBRARY + DEMO
-----------------------------------------

WHAT THIS DOES
  Implements perspective projection on a plane Z = 0:

      λ [u, v, 1]^T = H [X, Y, 1]^T

  Estimates H from world ↔ image correspondences (DLT), maps object corners to
  world (X, Y), and returns edge lengths in physical units.

WHEN TO USE THIS FILE
  - Library: imported by measure_object_calibrated.py (recommended entry point).
  - Demo / JSON without calibration undistortion:

      cd HW1
      python3 measure_planar_object_2d.py --demo
      python3 measure_planar_object_2d.py --json example_measurements.json

FOR CALIBRATED MEASUREMENT (undistort + your Calibration1.m intrinsics):

      python3 measure_object_calibrated.py

Requires: pip install numpy opencv-python
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def normalize_points_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Similarity transform so centroid is origin and mean distance to origin is sqrt(2)."""
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_dist = np.sqrt((shifted**2).sum(axis=1)).mean()
    if mean_dist < 1e-12:
        raise ValueError("Degenerate point set for normalization.")
    scale = np.sqrt(2.0) / mean_dist
    t = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    hom = np.hstack([points, ones])
    normalized = (t @ hom.T).T[:, :2]
    return normalized, t


def homography_dlt(
    world_xy: np.ndarray, image_uv: np.ndarray, *, normalize: bool = True
) -> np.ndarray:
    """
    Direct Linear Transform for homography: λ [u,v,1]^T = H [X,Y,1]^T.

    world_xy: (N, 2), image_uv: (N, 2), N >= 4.
    """
    world_xy = np.asarray(world_xy, dtype=np.float64)
    image_uv = np.asarray(image_uv, dtype=np.float64)
    if world_xy.shape != image_uv.shape or world_xy.shape[0] < 4:
        raise ValueError("Need at least 4 matching 2D point pairs.")

    src, t_src = normalize_points_2d(world_xy) if normalize else (world_xy, np.eye(3))
    dst, t_dst = normalize_points_2d(image_uv) if normalize else (image_uv, np.eye(3))

    n = src.shape[0]
    a = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y = src[i]
        u, v = dst[i]
        a[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        a[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]

    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    # Denormalize: image ~ H world  =>  t_dst^{-1} H t_src maps original coords.
    h = np.linalg.inv(t_dst) @ h @ t_src
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


def homography_from_correspondences(
    world_xy: np.ndarray, image_uv: np.ndarray, method: str = "dlt"
) -> np.ndarray:
    """Estimate H from point pairs. Uses OpenCV RANSAC if available and method is 'opencv'."""
    world_xy = np.asarray(world_xy, dtype=np.float64)
    image_uv = np.asarray(image_uv, dtype=np.float64)
    if method == "opencv" and cv2 is not None:
        h, _ = cv2.findHomography(world_xy, image_uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if h is None:
            raise RuntimeError("cv2.findHomography failed.")
        return h.astype(np.float64)
    return homography_dlt(world_xy, image_uv)


def image_to_world_plane(h: np.ndarray, image_uv: np.ndarray) -> np.ndarray:
    """Map image points to world (X, Y) on Z=0 via inverse homography."""
    image_uv = np.atleast_2d(np.asarray(image_uv, dtype=np.float64))
    h_inv = np.linalg.inv(h)
    ones = np.ones((image_uv.shape[0], 1), dtype=np.float64)
    hom = np.hstack([image_uv, ones])
    mapped = (h_inv @ hom.T).T
    w = mapped[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return mapped[:, :2] / w


def world_to_image(h: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    """Forward map world plane points to pixels."""
    world_xy = np.atleast_2d(np.asarray(world_xy, dtype=np.float64))
    ones = np.ones((world_xy.shape[0], 1), dtype=np.float64)
    hom = np.hstack([world_xy, ones])
    mapped = (h @ hom.T).T
    w = mapped[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return mapped[:, :2] / w


def euclidean_distance(p: Sequence[float], q: Sequence[float]) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.linalg.norm(p - q))


def polygon_edge_lengths(world_xy: np.ndarray, closed: bool = True) -> list[float]:
    """Lengths of consecutive edges; if closed, includes edge from last to first."""
    world_xy = np.asarray(world_xy, dtype=np.float64)
    n = world_xy.shape[0]
    if n < 2:
        return []
    lengths = []
    for i in range(n - 1):
        lengths.append(euclidean_distance(world_xy[i], world_xy[i + 1]))
    if closed and n > 2:
        lengths.append(euclidean_distance(world_xy[-1], world_xy[0]))
    return lengths


def measure_from_homography(
    h: np.ndarray,
    object_image_points: np.ndarray,
    *,
    closed: bool = True,
) -> dict:
    """Map object corners in the image to the world plane and report edge lengths."""
    world = image_to_world_plane(h, object_image_points)
    lengths = polygon_edge_lengths(world, closed=closed)
    return {
        "world_xy": world.tolist(),
        "edge_lengths": lengths,
        "perimeter": float(sum(lengths)) if lengths else 0.0,
    }


def measure_with_reference_correspondences(
    reference_world: np.ndarray,
    reference_image: np.ndarray,
    object_image: np.ndarray,
    *,
    world_units: str = "mm",
    closed: bool = True,
) -> dict:
    """
    Calibrate the ground plane from known reference geometry, then measure the object.

    reference_world: (N, 2) corners of a reference pattern or known distances on the plane.
    reference_image: (N, 2) matching pixel coordinates.
    object_image: (M, 2) pixels tracing the object to measure.
    """
    h = homography_from_correspondences(reference_world, reference_image)
    result = measure_from_homography(h, object_image, closed=closed)
    result["world_units"] = world_units
    result["homography"] = h.tolist()
    return result


def decompose_homography_to_pose(
    h: np.ndarray, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover camera extrinsics up to scale from H = K [r1 r2 t] (planar Z=0).

    Returns R (3×3), t (3,), n (plane normal in camera frame), plane_distance scale.
    For metric depth you need known baseline or object size on the plane.
    """
    if cv2 is None:
        raise ImportError("OpenCV required for decomposeHomographyMat.")
    k = np.asarray(k, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    n_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(h, k)
    if n_solutions < 1:
        raise RuntimeError("Could not decompose homography.")
    return rotations[0], translations[0], normals[0], k


def project_world_points_on_plane(
    world_xyz: np.ndarray,
    k: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Perspective projection: λ [u,v,1]^T = K [R|t] [X,Y,Z,1]^T.

    world_xyz: (N, 3), k: (3,3), r: (3,3), t: (3,) or (3,1).
    """
    world_xyz = np.atleast_2d(np.asarray(world_xyz, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    r = np.asarray(r, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    pts = world_xyz.T  # 3 x N
    cam = r @ pts + t  # 3 x N
    z = cam[2, :]
    z[np.abs(z) < 1e-12] = 1e-12
    x = cam[0, :] / z
    y = cam[1, :] / z
    hom = np.vstack([x, y, np.ones_like(x)])
    uv = (k @ hom).T[:, :2]
    return uv


def run_demo() -> None:
    """Synthetic plane: recover a rectangle's side lengths from projected corners."""
    # World rectangle on Z=0 (millimeters).
    true_w, true_h = 120.0, 75.0
    world_ref = np.array(
        [[0, 0], [200, 0], [200, 150], [0, 150], [0, 0]], dtype=np.float64
    )
    world_object = np.array(
        [[40, 30], [40 + true_w, 30], [40 + true_w, 30 + true_h], [40, 30 + true_h]],
        dtype=np.float64,
    )

    # Simulated camera (focal length in pixels, principal point).
    fx, fy, cx, cy = 900.0, 900.0, 640.0, 360.0
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    # Camera looks at the plane from an oblique angle.
    r = np.array(
        [
            [0.95, 0.05, -0.31],
            [-0.10, 0.98, -0.16],
            [0.29, 0.19, 0.94],
        ],
        dtype=np.float64,
    )
    t = np.array([[-80.0], [-60.0], [450.0]], dtype=np.float64)

    def lift_xy(xy: np.ndarray) -> np.ndarray:
        return np.hstack([xy, np.zeros((xy.shape[0], 1))])

    image_ref = project_world_points_on_plane(lift_xy(world_ref), k, r, t)
    image_object = project_world_points_on_plane(lift_xy(world_object), k, r, t)

    # Use first 4 reference corners (full quad) for H.
    h = homography_from_correspondences(world_ref[:4], image_ref[:4])
    measured = measure_from_homography(h, image_object, closed=True)
    edges = measured["edge_lengths"]

    print("=== Demo: synthetic perspective projection ===")
    print(f"True object size: {true_w:.2f} x {true_h:.2f} mm")
    print(f"Recovered edge lengths (mm): {[f'{e:.3f}' for e in edges]}")
    print(
        f"Recovered width x height (mm): {edges[0]:.3f} x {edges[1]:.3f} "
        f"(errors: {abs(edges[0]-true_w):.4f}, {abs(edges[1]-true_h):.4f})"
    )


def load_json_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def run_from_json(path: Path) -> None:
    cfg = load_json_config(path)
    ref_w = np.array(cfg["reference_world_xy"], dtype=np.float64)
    ref_i = np.array(cfg["reference_image_uv"], dtype=np.float64)
    obj_i = np.array(cfg["object_image_uv"], dtype=np.float64)
    units = cfg.get("world_units", "mm")
    closed = cfg.get("closed_polygon", True)
    out = measure_with_reference_correspondences(
        ref_w, ref_i, obj_i, world_units=units, closed=closed
    )
    print(json.dumps(out, indent=2))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Real-world 2D dimensions from perspective projection (planar homography)."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run synthetic demo (no image file required).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="FILE",
        help="JSON with reference_world_xy, reference_image_uv, object_image_uv.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.demo:
        run_demo()
        return 0
    if args.json is not None:
        run_from_json(args.json)
        return 0

    parser.print_help()
    print(
        "\nExample JSON keys: reference_world_xy, reference_image_uv, "
        "object_image_uv, world_units (optional), closed_polygon (optional)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
