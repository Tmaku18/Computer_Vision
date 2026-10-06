"""Spatial convolution equals FFT convolution, to numerical precision."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from webapp.main import app
from filtering_core import (
    box_kernel,
    compare,
    convolve_fft,
    convolve_opencv,
    convolve_spatial,
    demo_image,
    gaussian_kernel,
    make_kernel,
    motion_kernel,
    worked_example_1d,
)

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent


def test_module3_pages_render():
    for path in ("/module3", "/module3/blur", "/module3/theory", "/module3/report"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert "CSc 8830" in res.text


@pytest.mark.parametrize("mode,boundary", [("linear", "constant"), ("circular", "wrap")])
def test_spatial_matches_fft(mode, boundary):
    rng = np.random.default_rng(1)
    image = rng.normal(size=(37, 29, 3))
    kernel = rng.random((7, 5))
    kernel /= kernel.sum()
    spatial = convolve_spatial(image, kernel, boundary)
    frequency = convolve_fft(image, kernel, mode)
    assert compare(spatial, frequency)["max_abs"] < 1e-9


def test_opencv_correlation_needs_the_flip():
    rng = np.random.default_rng(2)
    image = rng.random((24, 22))
    kernel = motion_kernel(9, 20)
    ours = convolve_spatial(image, kernel, "reflect")
    theirs = convolve_opencv(image, kernel, "reflect")
    assert compare(ours, theirs)["max_abs"] < 1e-6
    # The same sum with the kernel reversed is a different blur: the streak
    # points the other way. A centered two-way streak would not show this.
    reversed_kernel = convolve_spatial(image, kernel[::-1, ::-1], "reflect")
    assert compare(ours, reversed_kernel)["max_abs"] > 1e-3


def test_kernels_are_normalized_and_odd():
    for kernel in (box_kernel(5), gaussian_kernel(15, 2), make_kernel("disk", 11), motion_kernel(13, 40)):
        assert kernel.shape[0] % 2 == 1
        assert kernel.sum() == pytest.approx(1.0)


def test_gaussian_is_lowpass():
    """A wide Gaussian keeps the DC bin and shrinks a high-frequency bin."""
    kernel = gaussian_kernel(21, 3)
    placed = np.zeros((64, 64))
    placed[:21, :21] = kernel
    spectrum = np.abs(np.fft.fft2(placed))
    assert spectrum[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert spectrum[0, 16] < 0.05


def test_worked_example_matches():
    example = worked_example_1d()
    assert example["full"] == pytest.approx([1, 4, 8, 12, 11, 4])
    assert example["max_abs"] < 1e-9


def test_blur_api_on_demo_image():
    res = client.post("/api/m3/blur", json={"kernel": "gaussian", "size": 9, "sigma": 2, "mode": "linear"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["metrics"]["max_abs"] < 1e-8
    assert body["width"] > 10
    for key in ("spatial_url", "fft_url", "difference_url"):
        assert body[key].startswith("data:image/")


def test_verify_api_passes():
    res = client.post("/api/m3/verify", json={})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["passed"]
    assert body["worst_matching_max_abs"] < 1e-8


def test_example_api():
    res = client.post("/api/m3/example", json={"signal": [1, 2, 3, 4], "kernel": [1, 2, 1]})
    assert res.status_code == 200
    assert res.json()["max_abs"] < 1e-9


def test_demo_image_shape():
    assert demo_image().shape == (360, 480, 3)
    assert (ROOT / "HW2" / "filtering_core.py").is_file()
