"""
Lucas–Kanade, bilinear interpolation, and a synthetic four-view planar scene.

    .venv/bin/python -m pytest tests/test_hw4.py -q
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "HW4"))

import flow_core  # noqa: E402
import sfm_core  # noqa: E402


def test_bilinear_is_exact_on_a_linear_ramp():
    # I = 2 + 3x + 5y is linear in both coordinates, so bilinear interpolation
    # reproduces it exactly (no higher-order remainder).
    xx, yy = np.meshgrid(np.arange(20), np.arange(16))
    image = 2.0 + 3.0 * xx + 5.0 * yy
    x, y = 3.3, 4.7
    got = flow_core.bilinear(image, x, y)
    assert got == pytest.approx(2.0 + 3.0 * x + 5.0 * y, abs=1e-9)


def test_lucas_kanade_recovers_a_known_shift():
    # Product of sinusoids, tracked at windows whose structure tensor has two
    # large eigenvalues (corners). An edge window cannot pin down the shift.
    cols = np.arange(180)[None, :]
    rows = np.arange(160)[:, None]
    image = 100.0 + 50.0 * np.sin(cols / 7.0) * np.cos(rows / 6.0) + 40.0 * np.sin(cols / 5.0) + 40.0 * np.cos(rows / 5.5)
    shift = np.array([0.85, -0.4])
    matrix = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float64)
    moved = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT101)
    points = np.array([[70.0, 120.0], [50.0, 120.0], [50.0, 90.0], [130.0, 90.0]])
    result = flow_core.lucas_kanade(image, moved, points, window=21, levels=1, max_iter=30)
    assert result["status"].all()
    assert np.all(result["eigenvalues"][:, 0] > 1000)
    error = np.linalg.norm(result["points"] - (points + shift), axis=1)
    assert np.max(error) < 0.01


def _camera_looking_at(center, target):
    center = np.asarray(center, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - center
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(np.array([1.0, 0.0, 0.0]), forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)
    rotation = np.stack([right, down, forward])
    translation = -rotation @ center
    return rotation, translation


def _project(k, rotation, translation, points):
    camera = points @ rotation.T + translation
    assert np.all(camera[:, 2] > 0.2)
    pix = camera @ k.T
    return pix[:, :2] / pix[:, 2:3]


def test_planar_four_view_sfm_recovers_the_boundary():
    cv2.setRNGSeed(0)
    k = np.array([[520.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]])
    xs = np.linspace(-1.0, 1.0, 6)
    ys = np.linspace(-0.45, 0.45, 5)
    world = np.array([(x, y, 5.0) for y in ys for x in xs], dtype=np.float64)
    corners = np.array([
        [-1.0, -0.45, 5.0],
        [1.0, -0.45, 5.0],
        [1.0, 0.45, 5.0],
        [-1.0, 0.45, 5.0],
    ])
    boundary = [int(np.argmin(np.linalg.norm(world - c, axis=1))) for c in corners]
    known_length = float(np.linalg.norm(world[boundary[0]] - world[boundary[1]]))

    eyes = [
        (0.0, 0.0, 0.0),
        (0.85, 0.15, 0.15),
        (-0.7, 0.35, 0.25),
        (0.15, -0.55, 0.35),
    ]
    target = np.array([0.0, 0.0, 5.0])
    uv = []
    cam0 = None
    for i, eye in enumerate(eyes):
        rotation, translation = _camera_looking_at(eye, target)
        pix = _project(k, rotation, translation, world)
        assert np.all(pix[:, 0] > 20) and np.all(pix[:, 0] < 620)
        assert np.all(pix[:, 1] > 20) and np.all(pix[:, 1] < 460)
        uv.append(pix)
        if i == 0:
            cam0 = (rotation, translation)
    uv = np.stack(uv)

    result = sfm_core.reconstruct_tracks(
        uv, k, boundary, (boundary[0], boundary[1]), known_length,
    )
    rotation0, translation0 = cam0
    expected = world @ rotation0.T + translation0
    # Same edge length, camera 1 fixed at the origin: the boundary should match.
    got = result["boundary_points"]
    exp = expected[boundary]
    err = np.linalg.norm(got - exp, axis=1)
    assert np.max(err) < 1e-6, err
    assert result["polygon"]["edge_lengths"][0] == pytest.approx(known_length, abs=1e-8)
    text = sfm_core.format_worked_example(result)
    assert "Essential matrix" in text
    assert "DLT" in text
    sfm_core.result_to_json(result)


def _client():
    from fastapi.testclient import TestClient

    from webapp.main import app

    return TestClient(app)


def _jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_module6_pages_and_stateless_tracking():
    import base64

    client = _client()
    for path in ("/module6", "/module6/flow", "/module6/tracking", "/module6/sfm", "/module6/theory", "/module6/report"):
        page = client.get(path)
        assert page.status_code == 200, path
        assert "CSc 8830" in page.text

    image = np.zeros((96, 120), np.uint8)
    cv2.circle(image, (60, 48), 10, 255, -1)
    cv2.rectangle(image, (20, 20), (40, 45), 180, -1)
    data_url = "data:image/jpeg;base64," + base64.b64encode(_jpeg(image)).decode("ascii")
    tracked = client.post("/api/m6/track", json={
        "frame1_data": data_url,
        "frame2_data": data_url,
        "points": [[22, 22]],
        "ground_truth": [[22, 22]],
    })
    assert tracked.status_code == 200, tracked.text
    payload = tracked.json()
    assert "example_text" in payload
    assert payload["scratch_error"][0] == pytest.approx(0.0, abs=0.05)

    missing = client.post("/api/m6/track", json={"points": [[22, 22]]})
    assert missing.status_code == 400
    escaped = client.get("/api/m6/results/file/../../HW1/camera_calibration.json")
    assert escaped.status_code in (400, 404)


def test_results_lists_committed_files():
    client = _client()
    listed = client.get("/api/m6/results")
    assert listed.status_code == 200, listed.text
    body = listed.json()
    assert body["clips"], "expected a committed flow clip under HW4/results"
    assert any(clip["id"] == "flow_static_camera" for clip in body["clips"])
    assert body["tracking"], "expected a committed tracking pair"
    assert body["sfm"] and body["sfm"]["result"]["cameras"]
    assert len(body["sfm"]["views"]) == 4
    urls = []
    for clip in body["clips"]:
        urls.extend(src for src in clip["files"].values() if src)
    for item in body["tracking"]:
        urls.append(item["frame1"])
        urls.append(item["frame2"])
    urls.extend(body["sfm"]["views"])
    assert urls
    for url in urls:
        got = client.get(url)
        assert got.status_code == 200, url
