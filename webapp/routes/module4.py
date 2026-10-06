"""Module 4 API: classical person outlines, SAM2 comparison, Fourier edges."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from webapp.storage import DATA_DIR, IMAGE_DIR, MAX_FILE_BYTES, RESULT_DIR, ROOT, read_image, safe_id, write_png

import cv_core  # noqa: E402
import fourier_segmentation  # noqa: E402
import segment_core  # noqa: E402
from segment_core import METHODS  # noqa: E402

router = APIRouter()

HW3 = ROOT / "HW3"
SAMPLES = HW3 / "samples"
PROMPTS_PATH = HW3 / "prompts.json"
COMPARISON_PATH = HW3 / "results" / "comparison.json"
MASK_DIR = DATA_DIR / "m4masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

MAX_SIDE = 1280
SEGMENT_FILES = ("mask", "overlay", "comparison")
FOURIER_FILES = ("filter", "spectrum", "edges", "regions")


class SegmentRequest(BaseModel):
    image_id: str | None = None
    sample: str | None = None
    method: str = "grabcut"
    box: list[float] | None = None
    background_id: str | None = None
    background_sample: str | None = None
    sam2_sample: str | None = None
    mask_id: str | None = None
    iterations: int = Field(5, ge=1, le=12)
    open_k: int = Field(5, ge=1, le=21)
    close_k: int = Field(7, ge=1, le=31)
    threshold: int = Field(25, ge=1, le=255)
    invert: bool = False
    min_area: int | None = Field(None, ge=1, le=500000)
    max_aspect: float = Field(4.5, ge=1.1, le=12)
    clahe_clip: float = Field(2.0, gt=0, le=8)
    blur: int = Field(5, ge=1, le=21)
    split: bool = False


class FourierRequest(BaseModel):
    image_id: str | None = None
    sample: str | None = None
    kind: str = "gaussian"
    cutoff: float = Field(24, ge=1, le=400)
    order: int = Field(2, ge=1, le=8)
    sigma1: float = Field(1.2, gt=0, le=20)
    sigma2: float = Field(2.4, gt=0, le=30)


def _fit(image: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1:
        return image
    return cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _sample_path(name: str, folder: Path = SAMPLES) -> Path:
    if not name or Path(name).name != name or ".." in name:
        raise HTTPException(400, "Invalid sample name.")
    path = folder / name
    if not path.is_file():
        raise HTTPException(404, f"Sample '{name}' was not found.")
    return path


def _load_scene(image_id: str | None, sample: str | None) -> tuple[np.ndarray, str]:
    if sample:
        return segment_core.read_bgr(_sample_path(sample)), sample
    if image_id:
        return read_image(image_id), image_id
    raise HTTPException(400, "Choose a sample or upload a photo.")


def _load_mask(mask_id: str) -> np.ndarray:
    path = MASK_DIR / f"{safe_id(mask_id)}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(404, "Mask not found. Upload it again.")
    return mask


def _reference_mask(req: SegmentRequest, image_shape) -> np.ndarray | None:
    if req.mask_id:
        mask = _load_mask(req.mask_id)
    elif req.sam2_sample:
        mask = cv2.imread(str(_sample_path(req.sam2_sample, SAMPLES / "sam2")), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise HTTPException(404, "Could not read that SAM2 mask.")
    else:
        return None
    if mask.shape[:2] != image_shape[:2]:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


@router.get("/api/m4/samples")
def list_samples():
    if not PROMPTS_PATH.is_file():
        return {"samples": []}
    items = []
    for prompt in segment_core.load_prompts(PROMPTS_PATH):
        name = Path(prompt["image"]).name
        sam = segment_core.sam2_mask_path(ROOT / prompt["image"])
        items.append({
            "name": name,
            "kind": prompt.get("kind", "rgb"),
            "label": prompt.get("note") or name,
            "box": prompt.get("box"),
            "background": None if not prompt.get("background") else Path(prompt["background"]).name,
            "sam2": None if sam is None else sam.name,
            "url": f"/api/m4/samples/{name}",
        })
    return {"samples": items}


@router.get("/api/m4/samples/{name}", include_in_schema=False)
def sample_file(name: str):
    return FileResponse(_sample_path(name))


@router.get("/api/m4/comparison")
def comparison():
    if not COMPARISON_PATH.is_file():
        raise HTTPException(404, "No comparison yet. Run HW3/compare_sam2.py.")
    return json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))


@router.post("/api/m4/images")
async def upload_image(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(400, "Photo is larger than 40 MB.")

    def work():
        image = _fit(cv_core.load_image(data))
        image_id = uuid.uuid4().hex[:12]
        cv2.imwrite(str(IMAGE_DIR / f"{image_id}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        meta = {
            "id": image_id,
            "name": file.filename or "photo",
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }
        (IMAGE_DIR / f"{image_id}.json").write_text(json.dumps(meta), encoding="utf-8")
        return meta

    try:
        meta = await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    meta["url"] = f"/api/images/{meta['id']}.jpg"
    return meta


@router.post("/api/m4/masks")
async def upload_mask(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(400, "Mask is larger than 40 MB.")

    def work():
        arr = np.frombuffer(data, np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("Could not read that mask image.")
        if decoded.ndim == 3:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        binary = np.where(decoded > 127, 255, 0).astype(np.uint8)
        mask_id = uuid.uuid4().hex[:12]
        write_png(MASK_DIR / f"{mask_id}.png", binary)
        return {"id": mask_id, "width": int(binary.shape[1]), "height": int(binary.shape[0])}

    try:
        return await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/api/m4/segment")
async def segment(req: SegmentRequest):
    if req.method not in METHODS:
        raise HTTPException(400, f"method must be one of {', '.join(METHODS)}.")

    def work():
        image, source = _load_scene(req.image_id, req.sample)
        background = None
        if req.background_sample:
            background = segment_core.read_bgr(_sample_path(req.background_sample))
        elif req.background_id:
            background = read_image(req.background_id)
        if background is not None and background.shape[:2] != image.shape[:2]:
            background = cv2.resize(background, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        result = segment_core.segment(
            image,
            req.method,
            box=req.box,
            background=background,
            iterations=req.iterations,
            open_k=req.open_k,
            close_k=req.close_k,
            threshold=req.threshold,
            invert=req.invert,
            min_area=req.min_area,
            max_aspect=req.max_aspect,
            clahe_clip=req.clahe_clip,
            blur=req.blur,
            split=req.split,
        )
        reference = _reference_mask(req, image.shape)
        metrics = None
        comparison = None
        if reference is not None:
            metrics = segment_core.compare_masks(result["mask"], reference)
            comparison = segment_core.comparison_overlay(image, result["mask"], reference)
        result_id = uuid.uuid4().hex[:12]
        folder = RESULT_DIR / result_id
        write_png(folder / "mask.png", result["mask"])
        write_png(folder / "overlay.png", result["overlay"])
        if comparison is not None:
            write_png(folder / "comparison.png", comparison)
        payload = {
            "id": result_id,
            "source": source,
            "method": result["method"],
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "box": result["box"],
            "metrics": metrics,
            "mask_url": f"/api/m4/results/{result_id}/mask.png",
            "overlay_url": f"/api/m4/results/{result_id}/overlay.png",
            "comparison_url": None if comparison is None else f"/api/m4/results/{result_id}/comparison.png",
        }
        return payload

    try:
        return await run_in_threadpool(work)
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/api/m4/fourier")
async def fourier(req: FourierRequest):
    if req.kind not in fourier_segmentation.FILTERS:
        raise HTTPException(400, f"kind must be one of {', '.join(fourier_segmentation.FILTERS)}.")

    def work():
        if req.sample or req.image_id:
            image, source = _load_scene(req.image_id, req.sample)
        else:
            image, source = segment_core.read_bgr(_sample_path("rgb_wall.png")), "rgb_wall.png"
        image = _fit(image, 720)
        result = fourier_segmentation.apply_filter(
            image, req.kind, cutoff=req.cutoff, order=req.order, sigma1=req.sigma1, sigma2=req.sigma2,
        )
        result_id = uuid.uuid4().hex[:12]
        folder = RESULT_DIR / result_id
        urls = {}
        for name in FOURIER_FILES:
            write_png(folder / f"{name}.png", result[name])
            urls[f"{name}_url"] = f"/api/m4/results/{result_id}/{name}.png"
        urls.update({
            "id": result_id,
            "source": source,
            "kind": req.kind,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        })
        return urls

    try:
        return await run_in_threadpool(work)
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/api/m4/results/{result_id}/{name}.png", include_in_schema=False)
def result_png(result_id: str, name: str):
    if name not in SEGMENT_FILES and name not in FOURIER_FILES:
        raise HTTPException(404)
    path = RESULT_DIR / safe_id(result_id) / f"{name}.png"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")
