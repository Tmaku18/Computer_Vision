#!/usr/bin/env python3
"""
uncalibrated_numbers.py
=======================

README — NUMBERS FOR THE HANDWRITTEN UNCALIBRATED-STEREO DERIVATION
-------------------------------------------------------------------

WHAT THIS DOES
  Part 2 of the assignment is a derivation you write by hand. Typed work is
  not accepted for that part. This script does not replace the derivation.
  It prints the numbers for one stereo pair so you can carry the same
  arithmetic onto paper: the fundamental matrix, the epipoles, the epipolar
  residuals, the rectifying homographies, the Bougnoux focal length, and the
  metric pose once the baseline (or one known length) fixes the scale.

  A pure sideways slide puts the epipoles at infinity, and then the focal
  length cannot be recovered from F. The script says so and tells you to use
  the Module 2 focal length for that step.

HOW TO RUN (from the repository root)
  .venv/bin/python HW5/uncalibrated_numbers.py
  .venv/bin/python HW5/uncalibrated_numbers.py --left L.jpg --right R.jpg --baseline 0.20 --unit m

OUTPUT
  Prints the worksheet. With --json, also writes HW5/results/uncalibrated.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stereo_core import load_demo, triangulate_dlt  # noqa: E402


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).ravel()
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: centroid at the origin, mean distance sqrt(2)."""
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    mean_dist = float(np.mean(np.linalg.norm(pts - center, axis=1)))
    if mean_dist < 1e-12:
        raise ValueError("The points are degenerate.")
    scale = np.sqrt(2.0) / mean_dist
    transform = np.array(
        [[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    hom = np.column_stack([pts, np.ones(len(pts))])
    normed = (transform @ hom.T).T
    return normed[:, :2], transform


def algebraic_residuals(f_mat: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([pts1, np.ones(len(pts1))])
    x2 = np.column_stack([pts2, np.ones(len(pts2))])
    return np.einsum("ij,jk,ik->i", x2, f_mat, x1)


def eight_point(pts1: np.ndarray, pts2: np.ndarray) -> dict:
    """Normalized eight-point algorithm, then the rank-2 constraint."""
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if len(pts1) < 8 or len(pts1) != len(pts2):
        raise ValueError("Need at least 8 paired points.")
    n1, t1 = normalize_points(pts1)
    n2, t2 = normalize_points(pts2)
    x, y = n1[:, 0], n1[:, 1]
    xp, yp = n2[:, 0], n2[:, 1]
    design = np.column_stack([xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, np.ones(len(x))])
    _, _, vt = np.linalg.svd(design)
    f_norm = vt[-1].reshape(3, 3)
    u, s, vt = np.linalg.svd(f_norm)
    s = s.copy()
    s[-1] = 0.0
    f_norm = u @ np.diag(s) @ vt
    residual = algebraic_residuals(f_norm, n1, n2)
    f_pix = t2.T @ f_norm @ t1
    f_pix = f_pix / np.linalg.norm(f_pix)
    singular = np.linalg.svd(f_pix, compute_uv=False)
    return {
        "F": f_pix,
        "normalized_residual_max": float(np.max(np.abs(residual))),
        "singular_values": singular,
    }


def epipoles(f_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """e1 satisfies F e1 = 0. e2 satisfies e2^T F = 0."""
    u, _s, vt = np.linalg.svd(f_mat)
    e1 = vt[-1]
    e2 = u[:, -1]
    return e1, e2


def bougnoux_focal(f_mat: np.ndarray, principal_point: tuple[float, float]) -> float:
    """Shared focal length. Square pixels, principal point given (use the image center).

    The formula is singular when the epipole is at infinity, which is exactly a
    fronto-parallel slide. Callers should catch ValueError and use a calibrated f.
    """
    cx, cy = principal_point
    p = np.array([cx, cy, 1.0], dtype=np.float64)
    _u, _s, _vt = np.linalg.svd(f_mat)
    e2 = _u[:, -1]
    if abs(e2[2]) < 1e-8 * np.linalg.norm(e2):
        raise ValueError(
            "The epipole is at infinity, so this pair is a pure slide and f cannot be recovered from F. "
            "Use the Module 2 focal length on this step."
        )
    eye = np.diag([1.0, 1.0, 0.0])
    ex = skew(e2)
    num = float((p @ ex @ eye @ f_mat @ p) * (p @ f_mat.T @ p))
    den = float(p @ ex @ eye @ f_mat @ eye @ f_mat.T @ p)
    if abs(den) < 1e-18 or -num / den <= 0:
        raise ValueError(
            "Bougnoux's formula did not return a positive focal length. "
            "The cameras are too close to parallel. Use the Module 2 focal length."
        )
    return float(np.sqrt(-num / den))


def rectifying_homographies(pts1: np.ndarray, pts2: np.ndarray, f_mat: np.ndarray, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Hartley uncalibrated rectification. image_size is (width, height)."""
    p1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 1, 2)
    p2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 1, 2)
    ok, h1, h2 = cv2.stereoRectifyUncalibrated(p1, p2, f_mat, image_size, threshold=0)
    if not ok:
        raise ValueError("Uncalibrated rectification failed for these points.")
    return np.asarray(h1, dtype=np.float64), np.asarray(h2, dtype=np.float64)


def apply_homography(h_mat: np.ndarray, points: np.ndarray) -> np.ndarray:
    hom = np.column_stack([points, np.ones(len(points))])
    mapped = (h_mat @ hom.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def enforce_essential(e_mat: np.ndarray) -> np.ndarray:
    u, s, vt = np.linalg.svd(e_mat)
    scale = (s[0] + s[1]) / 2
    return u @ np.diag([scale, scale, 0.0]) @ vt


def metric_pose(f_mat: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, k: np.ndarray, baseline: float) -> dict:
    """E = K^T F K, decompose it, and scale the unit translation so its length is the baseline."""
    essential = enforce_essential(k.T @ f_mat @ k)
    _count, r_mat, t_vec, _mask = cv2.recoverPose(
        essential, np.asarray(pts1, dtype=np.float64), np.asarray(pts2, dtype=np.float64), k
    )
    t_vec = np.asarray(t_vec, dtype=np.float64).ravel()
    t_vec = t_vec / np.linalg.norm(t_vec) * float(baseline)
    p1 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = k @ np.hstack([np.asarray(r_mat, dtype=np.float64), t_vec.reshape(3, 1)])
    points = triangulate_dlt(pts1, pts2, p1, p2)
    return {"R": np.asarray(r_mat, dtype=np.float64), "t": t_vec, "points": points, "E": essential}


def worksheet(pts1: np.ndarray, pts2: np.ndarray, image_size: tuple[int, int], baseline: float, k: np.ndarray | None = None) -> dict:
    """Every number the handwritten derivation needs for one pair."""
    solved = eight_point(pts1, pts2)
    f_mat = solved["F"]
    e1, e2 = epipoles(f_mat)
    width, height = image_size
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    focal_note = None
    try:
        focal = bougnoux_focal(f_mat, center)
    except ValueError as exc:
        focal = None
        focal_note = str(exc)
    h1, h2 = rectifying_homographies(pts1, pts2, f_mat, image_size)
    y1 = apply_homography(h1, pts1)[:, 1]
    y2 = apply_homography(h2, pts2)[:, 1]
    body = {
        "F": f_mat,
        "singular_values": solved["singular_values"],
        "normalized_residual_max": solved["normalized_residual_max"],
        "epipole_1": e1,
        "epipole_2": e2,
        "principal_point": center,
        "bougnoux_focal_px": focal,
        "focal_note": focal_note,
        "H1": h1,
        "H2": h2,
        "rectified_row_gap_px": float(np.max(np.abs(y1 - y2))),
        "baseline_m": float(baseline),
    }
    disparity = np.asarray(pts1, dtype=np.float64)[:, 0] - np.asarray(pts2, dtype=np.float64)[:, 0]
    body["disparity_std_px"] = float(np.std(disparity))
    planar = body["disparity_std_px"] < 0.5
    if k is not None and not planar:
        pose_k = k if focal is None else k
        if focal is not None:
            # Square pixels, principal point at the image center, focal length from F.
            pose_k = k.copy()
            pose_k[0, 0] = pose_k[1, 1] = focal
            pose_k[0, 2], pose_k[1, 2] = center
        else:
            body["fallback_focal_px"] = float(k[0, 0])
            body["focal_note"] = (focal_note or "") + " The metric step below uses the supplied K."
        pose = metric_pose(f_mat, pts1, pts2, pose_k, baseline)
        body["R"] = pose["R"]
        body["t"] = pose["t"]
        body["reconstructed_points"] = pose["points"]
    elif planar:
        body["metric_note"] = (
            "Disparity is almost constant, so these points lie on one plane. "
            "F, the epipoles, and the rectification are still valid. "
            "The translation from E is not unique on a single plane; use a pair with depth variation before copying t."
        )
    return body


def _fmt(value, digits=6) -> str:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return f"{float(arr):.{digits}g}"
    return np.array2string(arr, precision=digits, suppress_small=True)


def format_worksheet(body: dict) -> str:
    lines = [
        "Uncalibrated stereo — numbers to copy into the handwritten derivation",
        "Do not turn this printout in as the derivation. Write the steps yourself.",
        "",
        "Fundamental matrix F (||F|| = 1, rank 2):",
        _fmt(body["F"]),
        f"singular values: {_fmt(body['singular_values'])}",
        f"max |x'^T F x| after Hartley normalization: {body['normalized_residual_max']:.3e}",
        "",
        "Epipoles (homogeneous; divide by the last entry when it is not ~0):",
        f"e1 = {_fmt(body['epipole_1'])}",
        f"e2 = {_fmt(body['epipole_2'])}",
        "",
        f"Principal point used for Bougnoux (image center): {_fmt(body['principal_point'], 4)}",
    ]
    if body.get("bougnoux_focal_px") is None:
        lines.append("Bougnoux focal length: not recoverable.")
        lines.append(body.get("focal_note") or "")
    else:
        lines.append(f"Bougnoux focal length: {body['bougnoux_focal_px']:.4f} px")
    lines += [
        "",
        "Rectifying homographies H1, H2:",
        _fmt(body["H1"]),
        _fmt(body["H2"]),
        f"max row gap after rectification: {body['rectified_row_gap_px']:.4f} px",
        "",
        f"Baseline used to fix scale: {body['baseline_m']:.4f} m",
    ]
    if body.get("metric_note"):
        lines.append(body["metric_note"])
    if "t" in body:
        lines.append(f"R =\n{_fmt(body['R'])}")
        lines.append(f"t (scaled so ||t|| = baseline) = {_fmt(body['t'])}")
    return "\n".join(lines)


def demo_correspondences(count: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Correspondences from the built-in pair, which is a pure slide (epipoles at infinity)."""
    _left, _right, meta = load_demo()
    k = np.asarray(meta["k"], dtype=np.float64)
    width, height = int(meta["width"]), int(meta["height"])
    rng = np.random.default_rng(1)
    # Stay inside the overlap of the shifted pair.
    shift = int(meta["disparity_px"])
    u = rng.uniform(40, width - shift - 40, count)
    v = rng.uniform(40, height - 40, count)
    pts1 = np.column_stack([u, v])
    pts2 = np.column_stack([u - meta["disparity_px"], v])
    return pts1, pts2, k, (width, height)


def rotated_example(seed: int = 2) -> dict:
    """A pair with a small rotation, so Bougnoux's focal length is observable. Used by the tests."""
    rng = np.random.default_rng(seed)
    width, height = 640, 480
    f = 800.0
    cx, cy = (width - 1) / 2, (height - 1) / 2
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    r_mat, _ = cv2.Rodrigues(np.array([0.04, -0.18, 0.03], dtype=np.float64))
    baseline = 0.20
    center = np.array([baseline, 0.01, 0.02])
    world = np.column_stack([
        rng.uniform(-0.4, 0.4, 30),
        rng.uniform(-0.3, 0.3, 30),
        rng.uniform(1.6, 2.4, 30),
    ])
    from stereo_core import project_points

    pts1 = project_points(world, k)
    pts2 = project_points(world, k, r_mat, center)
    return {
        "pts1": pts1,
        "pts2": pts2,
        "k": k,
        "image_size": (width, height),
        "baseline": baseline,
        "R": r_mat,
        "center": center,
        "world": world,
        "focal": f,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print uncalibrated-stereo numbers for a handwritten derivation.")
    parser.add_argument("--left", help="Left photo. Omit both photos to use the built-in sample.")
    parser.add_argument("--right")
    parser.add_argument("--baseline", type=float, default=0.20, help="Slide length, in --unit.")
    parser.add_argument("--unit", default="m", choices=["m", "cm", "mm", "in", "ft"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = SCRIPT_DIR.parent
    if str(root / "HW1") not in sys.path:
        sys.path.insert(0, str(root / "HW1"))
    from cv_core import UNIT_TO_METERS, load_image
    from room_core import sift_matches
    from stereo_core import load_monocular, scale_intrinsics

    baseline = args.baseline * UNIT_TO_METERS[args.unit]
    if args.left and args.right:
        try:
            left = load_image(args.left)
            right = load_image(args.right)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if left.shape[:2] != right.shape[:2]:
            raise SystemExit("The two photos must be the same size.")
        pts1, pts2 = sift_matches(left, right)
        mono = load_monocular()
        height, width = left.shape[:2]
        k = scale_intrinsics(mono["k"], mono["image_size"], (width, height))
        size = (width, height)
    elif args.left or args.right:
        raise SystemExit("Pass both --left and --right, or neither.")
    else:
        pts1, pts2, k, size = demo_correspondences()
    body = worksheet(pts1, pts2, size, baseline, k)
    print(format_worksheet(body))
    if args.json:
        out = SCRIPT_DIR / "results" / "uncalibrated.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        out.write_text(json.dumps(convert(body), indent=2), encoding="utf-8")
        print(f"\nWrote {out.relative_to(SCRIPT_DIR.parent)}")


if __name__ == "__main__":
    main()
