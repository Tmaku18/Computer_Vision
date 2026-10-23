"""Calibrated stereo sizes, the uncalibrated worksheet, and a synthetic room."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from webapp.main import app
from room_core import dimensions_from_planes, disparity_sgbm, relative_pose, stereo_pair_from_clip, synthetic_room
from stereo_core import (
    calibrate_stereo,
    camera_matrix,
    checkerboard_object,
    circle_diameter,
    depth_uncertainty,
    load_demo,
    make_demo,
    match_along_row,
    measure_points,
    project_points,
    scale_intrinsics,
)
from uncalibrated_numbers import rotated_example, worksheet

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent


def test_module7_pages_render():
    for path in ("/module7", "/module7/stereo", "/module7/room", "/module7/theory", "/module7/report"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert "CSc 8830" in res.text


def _pair(world, k, baseline):
    left = project_points(world, k)
    right = project_points(world, k, c=np.array([baseline, 0.0, 0.0]))
    return left, right


def test_shapes_and_disparity_match_triangulation():
    k = camera_matrix(800, 800, 320, 240)
    baseline = 0.2
    rectangle = np.array([[-0.4, -0.2, 2.0], [0.4, -0.2, 2.0], [0.4, 0.3, 2.0], [-0.4, 0.3, 2.0]])
    measured = measure_points(*_pair(rectangle, k, baseline), k, baseline, "rectangle")
    assert measured["width_m"] == pytest.approx(0.8, abs=1e-6)
    assert measured["height_m"] == pytest.approx(0.5, abs=1e-6)
    assert measured["triangulation_gap_m"] < 1e-6

    radius = 0.3
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    normal = np.array([0.2, 0.1, 1.0])
    normal /= np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0])
    axis_u = np.cross(normal, helper)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    center = np.array([0.0, 0.0, 2.2])
    circle = center + radius * np.cos(angles)[:, None] * axis_u + radius * np.sin(angles)[:, None] * axis_v
    measured = measure_points(*_pair(circle, k, baseline), k, baseline, "circle")
    assert measured["diameter_m"] == pytest.approx(2 * radius, abs=1e-6)
    assert measured["triangulation_gap_m"] < 1e-6

    pang = -np.pi / 2 + np.linspace(0, 2 * np.pi, 5, endpoint=False)
    pentagon = np.stack([0.25 * np.cos(pang), 0.25 * np.sin(pang), np.full(5, 1.8)], axis=1)
    measured = measure_points(*_pair(pentagon, k, baseline), k, baseline, "polygon")
    assert measured["edges_m"] == pytest.approx([float(np.linalg.norm(pentagon[1] - pentagon[0]))] * 5, abs=1e-6)


def test_noisy_circle_diameter():
    rng = np.random.default_rng(0)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    clean = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles), np.zeros(16)])
    noisy = clean + rng.normal(0, 0.002, clean.shape)
    assert circle_diameter(noisy)["diameter_m"] == pytest.approx(1.0, abs=0.02)


def test_depth_error_grows_with_z_squared():
    near = depth_uncertainty(np.array([1.0]), 700, 0.2, 0.5)[0]
    far = depth_uncertainty(np.array([2.0]), 700, 0.2, 0.5)[0]
    assert far == pytest.approx(4 * near, rel=1e-9)


def test_intrinsics_scale_with_the_image():
    k = camera_matrix(1000, 1000, 500, 400)
    scaled = scale_intrinsics(k, (1000, 800), (500, 400))
    assert scaled[0, 0] == pytest.approx(500)
    assert scaled[0, 2] == pytest.approx(250)
    assert scaled[1, 2] == pytest.approx(200)


def test_stereo_calibration_recovers_the_baseline():
    k = camera_matrix(800, 800, 320, 240)
    baseline = 0.2
    obj = checkerboard_object((9, 6), 0.03)
    objects, left_pts, right_pts = [], [], []
    for i in range(10):
        rvec = np.array([0.15 + 0.02 * i, -0.25 + 0.01 * i, 0.05])
        tvec = np.array([-0.05 + 0.01 * i, -0.04, 0.85 + 0.02 * i])
        first, _ = cv2.projectPoints(obj, rvec, tvec, k, np.zeros(5))
        second, _ = cv2.projectPoints(obj, rvec, tvec - np.array([baseline, 0, 0]), k, np.zeros(5))
        objects.append(obj)
        left_pts.append(first.astype(np.float32))
        right_pts.append(second.astype(np.float32))
    result = calibrate_stereo(objects, left_pts, right_pts, k, np.zeros(5), (640, 480))
    assert result["baseline_m"] == pytest.approx(baseline, abs=1e-4)
    assert np.linalg.norm(result["r"] - np.eye(3)) < 1e-3


def test_row_match_on_the_sample():
    demo = make_demo()
    corner = np.asarray(demo["meta"]["shapes"]["rectangle"][0])
    found = match_along_row(demo["left"], demo["right"], corner[0], corner[1])
    assert found["u"] == pytest.approx(corner[0] - demo["meta"]["disparity_px"], abs=0.25)


def test_fundamental_rank_bougnoux_and_pose():
    example = rotated_example()
    baseline = float(np.linalg.norm(example["center"]))
    body = worksheet(example["pts1"], example["pts2"], example["image_size"], baseline, example["k"])
    singular = body["singular_values"]
    assert singular[-1] / singular[0] < 1e-8
    assert body["normalized_residual_max"] < 1e-8
    assert body["bougnoux_focal_px"] == pytest.approx(example["focal"], rel=0.01)
    assert body["rectified_row_gap_px"] < 1.0
    center = -body["R"].T @ body["t"]
    assert np.linalg.norm(center - example["center"]) < 1e-6


def test_room_planes_and_still_split():
    room = synthetic_room()
    dims = dimensions_from_planes(room["floor"], room["ceiling"], room["long_wall"], room["short_wall"])
    assert dims["length_m"] == pytest.approx(room["truth"]["length_m"], abs=1e-6)
    assert dims["width_m"] == pytest.approx(room["truth"]["width_m"], abs=1e-6)
    assert dims["height_m"] == pytest.approx(room["truth"]["height_m"], abs=1e-6)

    dark = np.zeros((8, 8), np.uint8)
    bright = np.full((8, 8), 200, np.uint8)
    frames = [dark] * 5 + [np.full((8, 8), 80, np.uint8), np.full((8, 8), 140, np.uint8)] + [bright] * 5
    pair = stereo_pair_from_clip(frames)
    assert pair["left"].mean() == 0
    assert pair["right"].mean() == 200


def test_pose_scale_and_sgbm():
    k = camera_matrix(800, 800, 320, 240)
    baseline = 0.2
    rng = np.random.default_rng(1)
    world = np.column_stack([
        rng.uniform(-0.3, 0.3, 50),
        rng.uniform(-0.3, 0.3, 50),
        rng.uniform(1.5, 2.5, 50),
    ])
    rotation, _ = cv2.Rodrigues(np.array([0.0, 0.05, 0.0]))
    center = np.array([baseline, 0.0, 0.0])
    pose = relative_pose(project_points(world, k), project_points(world, k, rotation, center), k, baseline)
    recovered = -pose["R"].T @ pose["t"]
    assert np.linalg.norm(recovered - center) < 1e-6

    demo = make_demo()
    disparity = disparity_sgbm(demo["left"], demo["right"], 64)
    patch = disparity[140:200, 160:240]
    assert np.median(patch) == pytest.approx(demo["meta"]["disparity_px"], abs=1.0)


def test_measure_api_one_request():
    _left, _right, meta = load_demo()
    points = meta["shapes"]["rectangle"]
    matches = [[u - meta["disparity_px"], v] for u, v in points]
    res = client.post("/api/m7/measure", data={
        "sample": "1",
        "shape": "rectangle",
        "points": json.dumps(points),
        "matches": json.dumps(matches),
        "baseline": "0.2",
        "unit": "m",
        "tape_width": "1.25",
        "tape_height": "1",
    })
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["width_m"] == pytest.approx(meta["truth"]["rectangle"]["width_m"], abs=1e-6)
    assert body["height_m"] == pytest.approx(meta["truth"]["rectangle"]["height_m"], abs=1e-6)
    assert body["left_url"].startswith("data:image/")
    assert body["right_url"].startswith("data:image/")
    assert abs(body["tape"]["width_error_m"]) < 1e-6


def test_room_api_reports_the_box():
    res = client.post("/api/m7/room", data={"sample": "box"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["length_m"] == pytest.approx(4.0, abs=1e-6)
    assert body["width_m"] == pytest.approx(3.0, abs=1e-6)
    assert body["height_m"] == pytest.approx(2.5, abs=1e-6)
    assert body["disparity_url"].startswith("data:image/")
