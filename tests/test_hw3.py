"""
Tests for HW3 classical segmentation and Fourier-domain edges.

    .venv/bin/python -m pytest tests/test_hw3.py -q
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "HW3"))

import fourier_segmentation as fourier  # noqa: E402
import segment_core as seg  # noqa: E402
from webapp.main import app  # noqa: E402

client = TestClient(app)

MODULE4_PAGES = [
    "/module4",
    "/module4/rgb",
    "/module4/thermal",
    "/module4/fourier",
    "/module4/theory",
    "/module4/report",
]


def test_rgb_pipelines_recover_synthetic_person():
    image, truth, box, empty = seg.synthetic_rgb(seed=0)
    for method, kwargs in (
        ("grabcut", {"box": box}),
        ("watershed", {"box": box}),
        ("background", {"background": empty}),
    ):
        result = seg.segment(image, method, **kwargs)
        metrics = seg.compare_masks(result["mask"], truth)
        assert metrics["iou"] > 0.95, f"{method} IoU {metrics['iou']:.3f}"


def test_thermal_pipeline_recovers_synthetic_person():
    image, truth, box = seg.synthetic_thermal(seed=1)
    result = seg.segment(image, "thermal", box=box)
    assert seg.compare_masks(result["mask"], truth)["iou"] > 0.95

    inverted, truth_i, box_i = seg.synthetic_thermal(seed=1, invert=True)
    result_i = seg.segment(inverted, "thermal", box=box_i, invert=True)
    assert seg.compare_masks(result_i["mask"], truth_i)["iou"] > 0.95


def test_thermal_rejects_wide_hot_bar():
    image, truth, box = seg.synthetic_thermal(seed=3, distractor=True)
    result = seg.segment(image, "thermal", box=box)
    metrics = seg.compare_masks(result["mask"], truth)
    assert metrics["iou"] > 0.95
    # The bar is the bright strip on the top left. It must not be in the mask.
    assert int(result["mask"][18:48, 20:200].sum()) == 0


def test_identical_masks_score_perfectly():
    _image, truth, _box, _empty = seg.synthetic_rgb(seed=0)
    metrics = seg.compare_masks(truth, truth)
    assert metrics["iou"] == 1
    assert metrics["dice"] == 1
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["boundary_f"] == pytest.approx(1)
    assert metrics["hausdorff_px"] == 0
    assert metrics["mean_boundary_px"] == 0
    assert metrics["boundary_tolerance_px"] == 2


def test_precision_is_one_when_prediction_is_inside_reference():
    _image, truth, _box, _empty = seg.synthetic_rgb(seed=0)
    pred = cv2.erode(truth, np.ones((5, 5), np.uint8))
    metrics = seg.compare_masks(pred, truth)
    assert metrics["precision"] == pytest.approx(1)
    assert metrics["recall"] < 1
    assert 0 < metrics["hausdorff_px"] < 20


def test_frequency_laplacian_matches_spatial():
    rng = np.random.default_rng(0)
    gray = rng.normal(size=(48, 64))
    gap = fourier.laplacian_error(gray)
    assert gap < 1e-8
    image, _truth, _box, _empty = seg.synthetic_rgb(seed=1)
    assert fourier.laplacian_error(fourier._gray(image)) < 1e-6


def test_highpass_dc_is_blocked():
    shape = (32, 40)
    for transfer in (
        fourier.ideal_highpass(shape, 6),
        fourier.butterworth_highpass(shape, 6, order=2),
        fourier.gaussian_highpass(shape, 6),
    ):
        assert transfer[shape[0] // 2, shape[1] // 2] == pytest.approx(0, abs=1e-8)
        # Ideal reaches 1. Butterworth and Gaussian approach 1 and are already past 0.99
        # at the corner of this small grid.
        assert float(transfer.max()) > 0.99


def test_prompts_round_trip(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        '{"prompts": [{"image": "HW3/samples/rgb_wall.png", "kind": "rgb", "box": [1, 2, 3, 4]}]}',
        encoding="utf-8",
    )
    prompts = seg.load_prompts(path)
    assert prompts[0]["box"] == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.json"
        bad.write_text('{"prompts": [{"image": "a.png", "box": [1, 2]}]}', encoding="utf-8")
        seg.load_prompts(bad)


@pytest.mark.parametrize("path", MODULE4_PAGES)
def test_module4_pages_render(path):
    res = client.get(path)
    assert res.status_code == 200
    assert "CSc 8830" in res.text


def test_sample_segmentation_api():
    samples = client.get("/api/m4/samples").json()
    wall = next(item for item in samples["samples"] if item["name"] == "rgb_wall.png")
    res = client.post("/api/m4/segment", json={
        "sample": "rgb_wall.png",
        "method": "grabcut",
        "box": wall["box"],
        "sam2_sample": "rgb_wall.png",
    })
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["metrics"]["iou"] > 0.9
    assert body["mask_url"].endswith("/mask.png")
    assert client.get(body["mask_url"]).status_code == 200
    assert client.get(body["comparison_url"]).status_code == 200


def test_fourier_api_on_sample():
    res = client.post("/api/m4/fourier", json={"sample": "rgb_wall.png", "kind": "laplacian"})
    assert res.status_code == 200, res.text
    body = res.json()
    for key in ("filter_url", "spectrum_url", "edges_url", "regions_url"):
        assert client.get(body[key]).status_code == 200


def test_comparison_report_endpoint():
    res = client.get("/api/m4/comparison")
    assert res.status_code == 200
    summary = res.json()["summary"]
    assert summary["rgb"]["n"] >= 1
    assert summary["thermal"]["n"] >= 1
