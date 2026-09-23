"""
webapp/main.py
==============

README — CSc 8830 COMPUTER VISION WEB APPLICATION
-------------------------------------------------

WHAT THIS IS
  One website that hosts every course assignment. Module 2 (camera calibration,
  2D measurement with perspective projection, validation, two-camera theory)
  is served under /module2 and reuses the Python code in ../HW1 (cv_core.py).

HOW TO RUN LOCALLY (from the repository root)
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  .venv/bin/uvicorn webapp.main:app --reload --port 8000
  open http://localhost:8000

HOW IT IS DEPLOYED
  The Dockerfile at the repository root runs the same command on Railway.
  Uploaded photos and calibrations live in DATA_DIR (default /tmp/cv-course);
  they are temporary and disappear when the server restarts.

API (JSON, all pixel coordinates are full-resolution (u, v) = (column, row))
  GET  /api/calibrations                 list calibrations
  GET  /api/calibrations/{id}            one calibration + per-photo results
  POST /api/calibrations                 multipart photos -> new calibration   (Step 1)
  POST /api/images                       upload a photo for measurement
  POST /api/reference/checkerboard       plane reference from a board in the photo
  POST /api/reference/rectangle          plane reference from 4 clicked corners
  POST /api/measure                      clicked points -> lengths              (Step 2)
  POST /api/stats                        ground truth vs estimates -> errors    (Step 3)
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODULE2_DIR = ROOT / "HW1"
sys.path.insert(0, str(MODULE2_DIR))

import cv_core  # noqa: E402
from cv_core import CameraCalibration  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/cv-course"))
IMAGE_DIR = DATA_DIR / "images"
CALIB_DIR = DATA_DIR / "calibrations"
for folder in (IMAGE_DIR, CALIB_DIR):
    folder.mkdir(parents=True, exist_ok=True)

MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_CALIBRATION_FILES = 80
THUMB_WIDTH = 480
DEFAULT_CALIBRATION_ID = "default"

app = FastAPI(title="CSc 8830 Computer Vision")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


# --------------------------------------------------------------------------
# Storage helpers
# --------------------------------------------------------------------------

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
    path = CALIB_DIR / f"{_safe_id(calib_id)}.json"
    if not path.is_file():
        raise HTTPException(404, f"Calibration '{calib_id}' not found (it may have expired on server restart).")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calibration(calib_id: str) -> CameraCalibration:
    return CameraCalibration.from_json(_calibration_record(calib_id)["calibration"])


def _safe_id(value: str) -> str:
    if not value or not all(c.isalnum() or c in "-_" for c in value):
        raise HTTPException(400, "Invalid id.")
    return value


def _image_meta(image_id: str) -> dict:
    path = IMAGE_DIR / f"{_safe_id(image_id)}.json"
    if not path.is_file():
        raise HTTPException(404, "Image not found (it may have expired on server restart). Upload it again.")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_image(image_id: str) -> np.ndarray:
    _image_meta(image_id)
    img = cv2.imread(str(IMAGE_DIR / f"{image_id}.jpg"), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise HTTPException(404, "Image file missing.")
    return img


def _thumbnail(img: np.ndarray, corners: np.ndarray | None, pattern, found: bool) -> np.ndarray:
    scale = THUMB_WIDTH / img.shape[1]
    thumb = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if corners is not None:
        cv2.drawChessboardCorners(thumb, pattern, (corners * scale).reshape(-1, 1, 2).astype(np.float32), found)
    return thumb


def _as_array(points, shape_msg: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise HTTPException(400, shape_msg)
    return arr


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

PAGES = {
    "": ("index.html", "Home"),
    "module2": ("module2/overview.html", "Module 2"),
    "module2/calibration": ("module2/calibration.html", "Step 1 · Calibration"),
    "module2/measure": ("module2/measure.html", "Step 2 · Measurement"),
    "module2/validation": ("module2/validation.html", "Step 3 · Validation"),
    "module2/theory": ("module2/theory.html", "Theory · Two cameras"),
    "module2/report": ("module2/report.html", "Report"),
}


def _page(path: str):
    template, title = PAGES[path]

    def handler(request: Request):
        return templates.TemplateResponse(
            request, template, {"title": title, "path": "/" + path}
        )

    return handler


for _path in PAGES:
    app.add_api_route("/" + _path, _page(_path), methods=["GET"], response_class=HTMLResponse,
                      include_in_schema=False)


# --------------------------------------------------------------------------
# API: health + calibration (Step 1)
# --------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/calibrations")
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


@app.get("/api/calibrations/{calib_id}")
def get_calibration(calib_id: str):
    record = _calibration_record(calib_id)
    calib = CameraCalibration.from_json(record["calibration"])
    record["valid_radius_px"] = cv_core.valid_pixel_radius(calib)
    return record


@app.get("/api/calibrations/{calib_id}/thumbs/{index}.jpg", include_in_schema=False)
def calibration_thumb(calib_id: str, index: int):
    path = CALIB_DIR / _safe_id(calib_id) / f"{int(index)}.jpg"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/calibrations")
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


# --------------------------------------------------------------------------
# API: photos + reference plane + measurement (Step 2)
# --------------------------------------------------------------------------

@app.post("/api/images")
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


@app.get("/api/images/{image_id}.jpg", include_in_schema=False)
def get_image(image_id: str):
    _image_meta(image_id)
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


def _check_unit(unit: str) -> None:
    if unit not in cv_core.UNIT_TO_METERS:
        raise HTTPException(400, f"Unknown unit '{unit}'. Use one of {sorted(cv_core.UNIT_TO_METERS)}.")


@app.post("/api/reference/checkerboard")
async def reference_checkerboard(req: CheckerboardReferenceRequest):
    _check_unit(req.unit)
    calib = _load_calibration(req.calibration_id)
    img = _read_image(req.image_id)
    try:
        ref = await run_in_threadpool(
            cv_core.reference_from_checkerboard, img, calib, tuple(req.pattern), req.square_size, req.unit
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return ref.to_json()


@app.post("/api/reference/rectangle")
def reference_rectangle(req: RectangleReferenceRequest):
    _check_unit(req.unit)
    calib = _load_calibration(req.calibration_id)
    meta = _image_meta(req.image_id)
    corners = _as_array(req.corners, "corners must be a list of [u, v] pairs.")
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


@app.post("/api/measure")
def measure(req: MeasureRequest):
    _check_unit(req.unit)
    if req.homography is None and req.distance is None:
        raise HTTPException(400, "Set a reference plane or a camera distance first.")
    calib = _load_calibration(req.calibration_id)
    meta = _image_meta(req.image_id)
    pts = _as_array(req.points, "points must be a list of [u, v] pairs.")
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


# --------------------------------------------------------------------------
# API: validation statistics (Step 3)
# --------------------------------------------------------------------------

class StatsRow(BaseModel):
    ground_truth: float
    homography: float | None = None
    pinhole: float | None = None


class StatsRequest(BaseModel):
    rows: list[StatsRow]


@app.post("/api/stats")
def stats(req: StatsRequest):
    out: dict[str, dict] = {}
    for method in ("homography", "pinhole"):
        pairs = [(r.ground_truth, getattr(r, method)) for r in req.rows if getattr(r, method) is not None]
        out[method] = cv_core.error_statistics(*zip(*pairs)) if pairs else {"n": 0}
    return out


@app.get("/api/units")
def units():
    return {"units": list(cv_core.UNIT_TO_METERS)}
