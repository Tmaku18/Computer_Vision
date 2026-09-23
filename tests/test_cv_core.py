"""
Tests for HW1/cv_core.py. Run from the repository root:

    .venv/bin/python -m pytest -q
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "HW1"))

import cv_core  # noqa: E402

K = np.array([[3700.0, 0, 2856], [0, 3700.0, 2142], [0, 0, 1]])
DIST = np.array([0.1, -0.3, 0, 0, 0])
SIZE = (5712, 4284)


def project(xy, rvec, tvec, dist=DIST):
    pts = np.hstack([xy, np.zeros((len(xy), 1))])
    return cv2.projectPoints(pts, rvec, tvec, K, dist)[0].reshape(-1, 2)


def test_undistort_inverts_opencv_distortion():
    rays = np.array([[0.0, 0.0, 1], [-0.5, -0.4, 1], [0.45, 0.35, 1], [0.6, -0.2, 1]])
    distorted = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, DIST)[0].reshape(-1, 2)
    expected = (K @ rays.T).T[:, :2]
    np.testing.assert_allclose(cv_core.undistort_points(distorted, K, DIST), expected, atol=1e-6)


def test_points_outside_lens_model_raise():
    strong = np.array([0.1, -0.5, 0, 0, 0])
    with pytest.raises(cv_core.OutsideLensModelError):
        cv_core.undistort_points(np.array([[5700.0, 4280.0]]), K, strong)


def test_homography_recovers_lengths_on_tilted_plane():
    rvec, tvec = np.array([0.3, -0.4, 0.1]), np.array([-5.0, -3.0, 100.0])
    calib = cv_core.CameraCalibration(K, DIST, *SIZE)
    rect = np.array([[0, 0], [11, 0], [11, 8.5], [0, 8.5]], dtype=float)
    ref = cv_core.reference_from_rectangle(project(rect, rvec, tvec), 11, 8.5, calib, SIZE)
    obj = np.array([[2, 2], [9, 2], [9, 6], [2, 6]], dtype=float)
    world = cv_core.measure_homography(ref.homography, project(obj, rvec, tvec), K, DIST)
    edges = cv_core.polygon_metrics(world, closed=True)["edge_lengths"]
    np.testing.assert_allclose(edges, [7, 4, 7, 4], atol=1e-6)
    assert ref.distance_to_plane == pytest.approx(abs(cv2.Rodrigues(rvec)[0][:, 2] @ tvec), rel=1e-6)


def test_pinhole_exact_for_fronto_parallel_plane():
    depth = 98.4  # inches (2.5 m)
    obj = np.array([[-6.0, -4.0], [6.0, -4.0]])
    uv = project(obj, np.zeros(3), np.array([0, 0, depth]))
    world = cv_core.measure_pinhole(uv, K, DIST, depth)
    assert np.linalg.norm(world[1] - world[0]) == pytest.approx(12.0, abs=1e-6)


def test_intrinsics_scale_with_resolution():
    calib = cv_core.CameraCalibration(K, DIST, *SIZE)
    k_half, _ = calib.intrinsics_for(SIZE[0] // 2, SIZE[1] // 2)
    np.testing.assert_allclose(k_half[:2], K[:2] / 2)
    with pytest.raises(ValueError):
        calib.intrinsics_for(SIZE[1], SIZE[0])


def test_error_statistics():
    s = cv_core.error_statistics([10, 20, 30, 40], [10.5, 19.5, 30.5, 39.5])
    assert s["n"] == 4
    assert s["mae"] == pytest.approx(0.5)
    assert s["rmse"] == pytest.approx(0.5)
    assert s["bias"] == pytest.approx(0.0)
    assert s["mape_percent"] == pytest.approx(100 * np.mean([0.05, 0.025, 0.5 / 30, 0.0125]))
    lo, hi = s["bias_ci95"]
    assert lo < 0 < hi


def test_checkerboard_detection_and_calibration_on_synthetic_views():
    pattern, square = (9, 6), 40
    board = np.zeros(((pattern[1] + 1) * square + 80, (pattern[0] + 1) * square + 80), np.uint8) + 255
    for r in range(pattern[1] + 1):
        for c in range(pattern[0] + 1):
            if (r + c) % 2 == 0:
                board[40 + r * square: 40 + (r + 1) * square, 40 + c * square: 40 + (c + 1) * square] = 0
    k = np.array([[900.0, 0, 640], [0, 900.0, 480], [0, 0, 1]])
    views = []
    rng = np.random.default_rng(0)
    for i in range(12):
        ang = rng.uniform(-0.35, 0.35, 3)
        rot, _ = cv2.Rodrigues(ang)
        # Map board pixels (plane Z = 0, 1 px = 1 unit) into the synthetic camera.
        t = np.array([-230.0, -170.0, 700.0]) + rng.uniform(-60, 60, 3)
        h = k @ np.column_stack([rot[:, 0], rot[:, 1], t])
        img = cv2.warpPerspective(board, h, (1280, 960), borderValue=255)
        views.append((f"v{i}", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    calib, used = cv_core.calibrate(views, pattern, square_size=square, min_images=10)
    assert sum(v.detected for v in used) >= 10
    assert calib.camera_matrix[0, 0] == pytest.approx(900, rel=0.02)
    assert calib.rms_px < 0.5
