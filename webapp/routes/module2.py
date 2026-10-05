"""Module 2 API: calibration, plane references, measurement, validation statistics."""

from __future__ import annotations

import json
import time
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from webapp.storage import (
    CALIB_DIR,
    DEFAULT_CALIBRATION_ID,
    IMAGE_DIR,
    MAX_CALIBRATION_FILES,
    MAX_FILE_BYTES,
    ROOT,
    THUMB_WIDTH,
    as_array,
    image_meta,
    read_image,
    safe_id,
)

import cv_core
from cv_core import CameraCalibration

MODULE2_DIR = ROOT / "HW1"

router = APIRouter()


def _default_calibration_record() -> dict:
    calib = json.loads((MODULE2_DIR / "camera_calibration.json").read_text(encoding="utf-8"))
    report_path = MODULE2_DIR / "calibration_report.json"
    views = []
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        views = [
            {"name": v["image"], "used": v.get("used", True),
             "reproj_rms_px": v.get("reproj_rms_px"), "note": v.get("note", ""), "thumb_url": None}
            for v in report.get("per_view_errors", [])
        ]
    return {
        "id": DEFAULT_CALIBRATION_ID,
        "label": f"Course smartphone calibration ({calib.get('num_images', len(views))} photos)",
        "created": None,
        "calibration": calib,
        "views": views,
    }


def _calibration_record(calib_id: str) -> dict:
    if calib_id == DEFAULT_CALIBRATION_ID:
        return _default_calibration_record()
    path = CALIB_DIR / f"{safe_id(calib_id)}.json"
    if not path.is_file():
        raise HTTPException(404, f"Calibration '{calib_id}' not found (it may have expired on server restart).")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calibration(calib_id: str) -> CameraCalibration:
    return CameraCalibration.from_json(_calibration_record(calib_id)["calibration"])


def _thumbnail(img: np.ndarray, corners: np.ndarray | None, pattern, found: bool) -> np.ndarray:
    scale = THUMB_WIDTH / img.shape[1]
    thumb = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if corners is not None:
        cv2.drawChessboardCorners(thumb, pattern, (corners * scale).reshape(-1, 1, 2).astype(np.float32), found)
    return thumb


def _check_unit(unit: str) -> None:
    if unit not in cv_core.UNIT_TO_METERS:
        raise HTTPException(400, f"Unknown unit '{unit}'. Use one of {sorted(cv_core.UNIT_TO_METERS)}.")


@router.get("/api/health")
def health():
    return {"status": "ok"}


@router.get("/api/calibrations")
def list_calibrations():
    records = [_default_calibration_record()]
    for path in sorted(CALIB_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return [
        {
            "id": r["id"],
            "label": r["label"],
            "created": r.get("created"),
            "num_images": r["calibration"].get("num_images"),
            "rms_px": r["calibration"].get("rms_px"),
            "image_size": r["calibration"].get("image_size"),
        }
        for r in records
    ]


@router.get("/api/calibrations/{calib_id}")
def get_calibration(calib_id: str):
    record = _calibration_record(calib_id)
    calib = CameraCalibration.from_json(record["calibration"])
    record["valid_radius_px"] = cv_core.valid_pixel_radius(calib)
    return record


@router.get("/api/calibrations/{calib_id}/thumbs/{index}.jpg", include_in_schema=False)
def calibration_thumb(calib_id: str, index: int):
    path = CALIB_DIR / safe_id(calib_id) / f"{int(index)}.jpg"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@router.post("/api/calibrations")
async def create_calibration(
    files: list[UploadFile] = File(...),
    pattern_cols: int = Form(9),
    pattern_rows: int = Form(6),
    square_size: float = Form(1.0),
    units: str = Form("in"),
    min_images: int = Form(20),
    label: str = Form(""),
):
    if len(files) > MAX_CALIBRATION_FILES:
        raise HTTPException(400, f"Upload at most {MAX_CALIBRATION_FILES} photos.")
    if units not in cv_core.UNIT_TO_METERS:
        raise HTTPException(400, f"Unknown unit '{units}'.")
    blobs = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(400, f"{f.filename} is larger than 40 MB.")
        blobs.append((f.filename or "photo", data))

    pattern = (int(pattern_cols), int(pattern_rows))
    calib_id = uuid.uuid4().hex[:12]
    thumb_dir = CALIB_DIR / calib_id
    thumb_dir.mkdir(parents=True, exist_ok=True)

    def work():
        decoded = []
        for name, data in blobs:
            try:
                decoded.append((name, cv_core.load_image(data)))
            except ValueError:
                decoded.append((name, None))
        valid = [(n, img) for n, img in decoded if img is not None]
        calib, views = cv_core.calibrate(valid, pattern, square_size, units, min_images=min_images)
        view_records = []
        for i, (view, (_name, img)) in enumerate(zip(views, valid)):
            thumb = _thumbnail(img, view.corners, pattern, view.detected)
            cv2.imwrite(str(thumb_dir / f"{i}.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            view_records.append({
                "name": view.name, "used": view.detected, "reproj_rms_px": view.reproj_rms_px,
                "note": view.note, "thumb_url": f"/api/calibrations/{calib_id}/thumbs/{i}.jpg",
                "distance": None if view.tvec is None else float(np.linalg.norm(view.tvec)),
            })
        for name, img in decoded:
            if img is None:
                view_records.append({"name": name, "used": False, "reproj_rms_px": None,
                                     "note": "could not decode image", "thumb_url": None})
        return calib, view_records

    try:
        calib, view_records = await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    record = {
        "id": calib_id,
        "label": label.strip() or f"Uploaded calibration ({calib.num_images} photos)",
        "created": time.time(),
        "calibration": calib.to_json(),
        "views": view_records,
    }
    (CALIB_DIR / f"{calib_id}.json").write_text(json.dumps(record), encoding="utf-8")
    record["valid_radius_px"] = cv_core.valid_pixel_radius(calib)
    return record


@router.post("/api/images")
async def upload_image(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(400, "Photo is larger than 40 MB.")

    def work():
        img = cv_core.load_image(data)
        image_id = uuid.uuid4().hex[:12]
        cv2.imwrite(str(IMAGE_DIR / f"{image_id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        meta = {"id": image_id, "name": file.filename or "photo",
                "width": int(img.shape[1]), "height": int(img.shape[0])}
        (IMAGE_DIR / f"{image_id}.json").write_text(json.dumps(meta), encoding="utf-8")
        return meta

    try:
        meta = await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    meta["url"] = f"/api/images/{meta['id']}.jpg"
    return meta


@router.get("/api/images/{image_id}.jpg", include_in_schema=False)
def get_image(image_id: str):
    image_meta(image_id)
    return FileResponse(IMAGE_DIR / f"{image_id}.jpg", media_type="image/jpeg")


class CheckerboardReferenceRequest(BaseModel):
    image_id: str
    calibration_id: str = DEFAULT_CALIBRATION_ID
    pattern: tuple[int, int] = (9, 6)
    square_size: float = Field(1.0, gt=0)
    unit: str = "in"


class RectangleReferenceRequest(BaseModel):
    image_id: str
    calibration_id: str = DEFAULT_CALIBRATION_ID
    corners: list[list[float]]
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    unit: str = "in"


@router.post("/api/reference/checkerboard")
async def reference_checkerboard(req: CheckerboardReferenceRequest):
    _check_unit(req.unit)
    calib = _load_calibration(req.calibration_id)
    img = read_image(req.image_id)
    try:
        ref = await run_in_threadpool(
            cv_core.reference_from_checkerboard, img, calib, tuple(req.pattern), req.square_size, req.unit
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return ref.to_json()


@router.post("/api/reference/rectangle")
def reference_rectangle(req: RectangleReferenceRequest):
    _check_unit(req.unit)
    calib = _load_calibration(req.calibration_id)
    meta = image_meta(req.image_id)
    corners = as_array(req.corners, "corners must be a list of [u, v] pairs.")
    try:
        ref = cv_core.reference_from_rectangle(
            corners, req.width, req.height, calib, (meta["width"], meta["height"]), req.unit
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return ref.to_json()


class Distance(BaseModel):
    value: float = Field(gt=0)
    unit: str = "m"


class MeasureRequest(BaseModel):
    image_id: str
    calibration_id: str = DEFAULT_CALIBRATION_ID
    points: list[list[float]]
    closed: bool = False
    unit: str = "in"
    homography: list[list[float]] | None = None
    distance: Distance | None = None


@router.post("/api/measure")
def measure(req: MeasureRequest):
    _check_unit(req.unit)
    if req.homography is None and req.distance is None:
        raise HTTPException(400, "Set a reference plane or a camera distance first.")
    calib = _load_calibration(req.calibration_id)
    meta = image_meta(req.image_id)
    pts = as_array(req.points, "points must be a list of [u, v] pairs.")
    if pts.shape[0] < 2:
        raise HTTPException(400, "Click at least 2 points.")
    try:
        k, dist = calib.intrinsics_for(meta["width"], meta["height"])
        undist = cv_core.undistort_points(pts, k, dist)
        result = {"unit": req.unit, "undistorted": undist.tolist(), "homography": None, "pinhole": None}
        if req.homography is not None:
            h = np.asarray(req.homography, dtype=np.float64)
            world = cv_core.measure_homography(h, pts, k, dist)
            result["homography"] = cv_core.polygon_metrics(world, req.closed)
        if req.distance is not None:
            _check_unit(req.distance.unit)
            depth = cv_core.convert_length(req.distance.value, req.distance.unit, req.unit)
            world = cv_core.measure_pinhole(pts, k, dist, depth)
            result["pinhole"] = cv_core.polygon_metrics(world, req.closed)
            result["pinhole"]["depth"] = depth
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return result


class StatsRow(BaseModel):
    ground_truth: float
    homography: float | None = None
    pinhole: float | None = None


class StatsRequest(BaseModel):
    rows: list[StatsRow]


@router.post("/api/stats")
def stats(req: StatsRequest):
    out: dict[str, dict] = {}
    for method in ("homography", "pinhole"):
        pairs = [(r.ground_truth, getattr(r, method)) for r in req.rows if getattr(r, method) is not None]
        out[method] = cv_core.error_statistics(*zip(*pairs)) if pairs else {"n": 0}
    return out


@router.get("/api/units")
def units():
    return {"units": list(cv_core.UNIT_TO_METERS)}
