"""
End-to-end tests of the web API. Tests that need the real smartphone photos are
skipped when HW1/calibration_images/ is empty (e.g. in a fresh clone).

    .venv/bin/python -m pytest -q
"""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from webapp.main import app

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = sorted(p for p in (ROOT / "HW1" / "calibration_images").glob("IMG_*") if p.suffix.lower() in {".jpg", ".heic"})
client = TestClient(app)
needs_photos = pytest.mark.skipif(len(PHOTOS) < 20, reason="real calibration photos not present")


@pytest.mark.parametrize("path", ["/", "/module2", "/module2/calibration", "/module2/measure",
                                  "/module2/validation", "/module2/theory", "/module2/report"])
def test_pages_render(path):
    res = client.get(path)
    assert res.status_code == 200
    assert "CSc 8830" in res.text


def test_default_calibration():
    rec = client.get("/api/calibrations/default").json()
    k = rec["calibration"]["camera_matrix"]
    assert rec["calibration"]["num_images"] >= 20
    assert 3000 < k[0][0] < 4500
    assert rec["valid_radius_px"] > 1000


def test_stats_endpoint():
    res = client.post("/api/stats", json={"rows": [
        {"ground_truth": 10, "homography": 10.2, "pinhole": 9.7},
        {"ground_truth": 20, "homography": 19.9, "pinhole": None},
    ]}).json()
    assert res["homography"]["n"] == 2
    assert res["pinhole"]["n"] == 1
    assert res["homography"]["mae"] == pytest.approx(0.15)


def test_measure_requires_reference_or_distance():
    res = client.post("/api/measure", json={"image_id": "abc", "points": [[0, 0], [1, 1]]})
    assert res.status_code == 400


@needs_photos
def test_checkerboard_reference_and_measurement_on_real_photo():
    path = next(p for p in PHOTOS if p.name == "IMG_5270.jpg")
    meta = client.post("/api/images", files={"file": (path.name, path.read_bytes(), "image/jpeg")}).json()
    assert (meta["width"], meta["height"]) == (5712, 4284)

    ref = client.post("/api/reference/checkerboard", json={
        "image_id": meta["id"], "pattern": [9, 6], "square_size": 1.0, "unit": "in",
    })
    assert ref.status_code == 200, ref.text
    ref = ref.json()
    assert ref["reproj_rms_px"] < 3
    corners = np.array(ref["image_points"])

    # First and last corner of the top row are 8 squares = 8 in apart.
    seg = [corners[0].tolist(), corners[8].tolist()]
    res = client.post("/api/measure", json={
        "image_id": meta["id"], "points": seg, "unit": "in", "homography": ref["homography"],
        "distance": {"value": ref["distance_to_plane_m"], "unit": "m"},
    }).json()
    assert res["homography"]["edge_lengths"][0] == pytest.approx(8.0, abs=0.05)
    assert res["pinhole"]["edge_lengths"][0] > 0

    # Same board used as a known 8 x 5 in rectangle from its 4 outer corners.
    outer = [corners[0], corners[8], corners[53], corners[45]]
    rect = client.post("/api/reference/rectangle", json={
        "image_id": meta["id"], "corners": [c.tolist() for c in outer], "width": 8, "height": 5, "unit": "in",
    }).json()
    mid = [corners[2].tolist(), corners[47].tolist()]  # vertical-ish segment across the board
    res = client.post("/api/measure", json={
        "image_id": meta["id"], "points": mid, "unit": "in", "homography": rect["homography"],
    }).json()
    assert res["homography"]["edge_lengths"][0] == pytest.approx(5.0, abs=0.1)


@needs_photos
def test_calibration_upload_with_20_photos():
    files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in PHOTOS[:20]]
    res = client.post("/api/calibrations", files=files, data={
        "pattern_cols": "9", "pattern_rows": "6", "square_size": "1", "units": "in", "min_images": "20",
    })
    assert res.status_code == 200, res.text
    rec = res.json()
    assert rec["calibration"]["num_images"] == 20
    assert rec["calibration"]["rms_px"] < 4
    thumb = client.get(rec["views"][0]["thumb_url"])
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"


def test_too_few_calibration_photos_rejected():
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((100, 100, 3), 255, np.uint8))
    res = client.post("/api/calibrations", files=[("files", ("blank.jpg", buf.tobytes(), "image/jpeg"))])
    assert res.status_code == 400
    assert "at least 20" in res.json()["detail"]
