"""Module 3 API: spatial blur, FFT blur, the convolution-theorem experiment, the 1-D example."""

from __future__ import annotations

import json
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from webapp.storage import RESULT_DIR, ROOT, read_image, safe_id, write_png
from filtering_core import (
    blur_pair,
    demo_image,
    difference_map,
    downscale,
    make_kernel,
    run_experiment,
    spectra,
    timing_curve,
    to_gray,
    to_uint8,
    worked_example_1d,
)
router = APIRouter()
RESULTS = ROOT / "HW2" / "results"
_verify_cache: dict[str, dict] = {}


class BlurRequest(BaseModel):
    image_id: str | None = None
    kernel: str = "gaussian"
    size: int = Field(15, ge=1, le=51)
    sigma: float = Field(3.0, gt=0, le=30)
    angle: float = Field(0.0, ge=-180, le=180)
    mode: str = "linear"
    max_side: int = Field(720, ge=64, le=1200)


class VerifyRequest(BaseModel):
    image_id: str | None = None


class ExampleRequest(BaseModel):
    signal: list[float] = Field(default_factory=lambda: [1, 2, 3, 4])
    kernel: list[float] = Field(default_factory=lambda: [1, 2, 1])


def _load(image_id: str | None, max_side: int) -> tuple[np.ndarray, str, float]:
    if image_id:
        image, scale = downscale(read_image(image_id), max_side)
        return image.astype(np.float64), image_id, scale
    image, scale = downscale(demo_image(), max_side)
    return image, "demo", scale


def _kernel_preview(kernel: np.ndarray) -> np.ndarray:
    peak = float(kernel.max()) or 1.0
    small = (np.clip(kernel / peak, 0, 1) * 255).astype(np.uint8)
    return cv2.resize(small, (140, 140), interpolation=cv2.INTER_NEAREST)


@router.post("/api/m3/blur")
async def blur(req: BlurRequest):
    if req.mode not in ("linear", "circular"):
        raise HTTPException(400, "mode must be 'linear' or 'circular'.")

    def work():
        image, source, scale = _load(req.image_id, req.max_side)
        try:
            kernel = make_kernel(req.kernel, req.size, req.sigma, req.angle)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        result = blur_pair(image, kernel, mode=req.mode)
        spec = spectra(image, kernel, mode=req.mode)
        diff, diff_info = difference_map(result["spatial"], result["frequency"])
        result_id = uuid.uuid4().hex[:12]
        folder = RESULT_DIR / result_id
        files = {
            "original": to_uint8(image),
            "spatial": to_uint8(result["spatial"]),
            "fft": to_uint8(result["frequency"]),
            "difference": diff,
            "spectrum_image": spec["image"],
            "spectrum_kernel": spec["kernel"],
            "spectrum_product": spec["product"],
            "kernel": _kernel_preview(kernel),
        }
        for name, picture in files.items():
            write_png(folder / f"{name}.png", picture)
        return {
            "id": result_id,
            "source": source,
            "scale": scale,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "kernel_name": req.kernel,
            "kernel_shape": list(kernel.shape),
            "mode": result["mode"],
            "boundary": result["boundary"],
            "metrics": result["metrics"],
            "timing_s": result["timing_s"],
            "difference": diff_info,
            **{f"{name}_url": f"/api/m3/results/{result_id}/{name}.png" for name in files},
        }

    return await run_in_threadpool(work)


@router.post("/api/m3/timing")
async def timing(req: VerifyRequest):
    def work():
        image, source, _scale = _load(req.image_id, 128)
        curve = timing_curve(to_gray(image))
        return {"source": source, "timing": curve}

    return await run_in_threadpool(work)


@router.post("/api/m3/verify")
async def verify(req: VerifyRequest):
    key = req.image_id or "demo"
    if key in _verify_cache:
        return _verify_cache[key]

    def work():
        image, source, _scale = _load(req.image_id, 160)
        report = run_experiment(image, source)
        matching = [row for row in report["rows"] if row["expected_match"]]
        worst = max(matching, key=lambda row: row["max_abs"])
        report["worst_matching_max_abs"] = worst["max_abs"]
        report["passed"] = bool(worst["max_abs"] < 1e-8)
        return report

    report = await run_in_threadpool(work)
    _verify_cache[key] = report
    return report


@router.get("/api/m3/report")
def saved_report():
    path = RESULTS / "report.json"
    if not path.is_file():
        raise HTTPException(404, "No saved experiment yet. Run HW2/verify_convolution_theorem.py.")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/api/m3/report/{name}.png", include_in_schema=False)
def saved_figure(name: str):
    if name not in {"differences", "timing"}:
        raise HTTPException(404)
    path = RESULTS / f"{name}.png"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@router.post("/api/m3/example")
def example(req: ExampleRequest):
    if not (1 <= len(req.signal) <= 12 and 1 <= len(req.kernel) <= 9):
        raise HTTPException(400, "Keep the signal to at most 12 samples and the kernel to at most 9.")
    try:
        return worked_example_1d(tuple(req.signal), tuple(req.kernel))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/api/m3/results/{result_id}/{name}.png", include_in_schema=False)
def result_png(result_id: str, name: str):
    allowed = {"original", "spatial", "fft", "difference", "spectrum_image", "spectrum_kernel", "spectrum_product", "kernel"}
    if name not in allowed:
        raise HTTPException(404)
    path = RESULT_DIR / safe_id(result_id) / f"{name}.png"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")
