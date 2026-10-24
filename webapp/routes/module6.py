"""Module 6 API: optical-flow jobs, two-frame tracking, four-view planar SfM."""

from __future__ import annotations

import base64
import json
import math
import sys
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from webapp.routes.module2 import _load_calibration
from webapp.storage import DATA_DIR, IMAGE_DIR, ROOT, image_meta, read_image, safe_id

for _folder in ("HW1", "HW4"):
    _path = str(ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import flow_core  # noqa: E402
import sfm_core  # noqa: E402

router = APIRouter()

RESULTS = ROOT / "HW4" / "results"
VIDEO_DIR = DATA_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
MAX_VIDEO_BYTES = 250 * 1024 * 1024
FLOW_FILES = {
    "original.mp4", "flow_hsv.mp4", "flow_arrows.mp4", "tracks.mp4",
    "side_by_side.mp4", "moving_mask.png", "stats.json",
}

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _store_bgr(image: np.ndarray, name: str) -> dict:
    image_id = uuid.uuid4().hex[:12]
    cv2.imwrite(str(IMAGE_DIR / f"{image_id}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    meta = {
        "id": image_id,
        "name": name,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "url": f"/api/images/{image_id}.jpg",
    }
    (IMAGE_DIR / f"{image_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _finite(value):
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _pair_payload(result: dict) -> dict:
    def column(value):
        if value is None:
            return None
        return np.asarray(value, dtype=np.float64).tolist()

    example = result["example"]
    return _finite({
        "points": column(result["points"]),
        "scratch": column(result["scratch"]),
        "scratch_status": result["scratch_status"].astype(bool).tolist(),
        "opencv": column(result["opencv"]),
        "opencv_status": result["opencv_status"].astype(bool).tolist(),
        "template": column(result["template"]),
        "ground_truth": column(result["ground_truth"]),
        "scratch_error": column(result["scratch_error"]),
        "opencv_error": column(result["opencv_error"]),
        "template_error": column(result["template_error"]),
        "eigenvalues": column(result["eigenvalues"]),
        "example_index": result["example_index"],
        "example_text": result["example_text"],
        "ATA": column(example["ATA"]),
        "ATb": column(example["ATb"]),
        "step_uv": column(example["step_uv"]),
        "eigenvalues_window": column(example["eigenvalues"]),
        "bilinear": example["bilinear"],
    })


def _resize_long_edge(image: np.ndarray, clicks: np.ndarray, max_edge: int = 1600):
    height, width = image.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale == 1.0:
        return image, clicks
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, clicks * scale


@router.post("/api/m6/flow")
async def start_flow(
    file: UploadFile = File(...),
    start: float = Form(0.0),
    duration: float = Form(30.0),
):
    data = await file.read()
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(400, "Video is larger than 250 MB. Trim it to about 30 seconds.")
    if duration <= 0 or duration > 45:
        raise HTTPException(400, "Duration must be between 0 and 45 seconds.")
    if start < 0:
        raise HTTPException(400, "Start time cannot be negative.")
    job_id = uuid.uuid4().hex[:12]
    folder = VIDEO_DIR / job_id
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}:
        suffix = ".mp4"
    video_path = folder / f"source{suffix}"
    video_path.write_bytes(data)
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "status": "running", "progress": 0.0, "message": "starting", "result": None, "error": None}

    def run():
        def progress(fraction, message):
            with _jobs_lock:
                _jobs[job_id]["progress"] = float(fraction)
                _jobs[job_id]["message"] = message

        try:
            summary = flow_core.process_video(video_path, folder, start, duration, max_width=960, progress=progress)
            base = f"/api/m6/jobs/{job_id}/files"
            files = {}
            for key, name in summary["files"].items():
                if name:
                    files[key] = f"{base}/{name}"
            summary["files"] = files
            summary.pop("video", None)
            with _jobs_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["progress"] = 1.0
                _jobs[job_id]["result"] = summary
        except Exception as exc:  # noqa: BLE001 — surfaced to the job status
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(exc)

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


@router.get("/api/m6/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(safe_id(job_id))
    if job is None:
        raise HTTPException(404, "Job not found. The server may have restarted.")
    return job


@router.get("/api/m6/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    if name not in FLOW_FILES or "/" in name:
        raise HTTPException(404)
    path = VIDEO_DIR / safe_id(job_id) / name
    if not path.is_file():
        raise HTTPException(404)
    media = "video/mp4" if name.endswith(".mp4") else "application/json" if name.endswith(".json") else "image/png"
    return FileResponse(path, media_type=media)


@router.post("/api/m6/frames")
async def extract_frames(
    file: UploadFile | None = File(None),
    frame_a: UploadFile | None = File(None),
    frame_b: UploadFile | None = File(None),
    time_sec: float = Form(0.0),
):
    if frame_a is not None and frame_b is not None:
        blobs = [await frame_a.read(), await frame_b.read()]
        names = [frame_a.filename or "frame1.jpg", frame_b.filename or "frame2.jpg"]

        def decode_pair():
            metas = []
            for blob, name in zip(blobs, names):
                image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not decode {name}.")
                metas.append(_store_bgr(image, name))
            return {"frame1": metas[0], "frame2": metas[1]}

        try:
            return await run_in_threadpool(decode_pair)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    if file is None:
        raise HTTPException(400, "Upload a video, or upload both frames.")
    blob = await file.read()
    if len(blob) > MAX_VIDEO_BYTES:
        raise HTTPException(400, "Video is larger than 250 MB.")
    job_folder = VIDEO_DIR / "frames"
    job_folder.mkdir(parents=True, exist_ok=True)
    source = job_folder / f"{uuid.uuid4().hex[:8]}.mp4"
    source.write_bytes(blob)

    def grab():
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError("Could not open that video.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(max(0.0, time_sec) * fps)))
        ok1, first = cap.read()
        ok2, second = cap.read()
        cap.release()
        if not (ok1 and ok2):
            raise ValueError("Need two frames at that time. Try an earlier start.")
        return {
            "frame1": _store_bgr(first, "frame1.jpg"),
            "frame2": _store_bgr(second, "frame2.jpg"),
            "fps": fps,
        }

    try:
        return await run_in_threadpool(grab)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _decode_data_url(text: str) -> np.ndarray:
    if len(text) > 8_000_000:
        raise ValueError("Frame is too large. Shrink it to about 1280 px before uploading.")
    payload = text.split(",", 1)[1] if text.strip().startswith("data:") and "," in text else text
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:  # noqa: BLE001 — any bad payload is a 400
        raise ValueError("Could not decode a frame.") from exc
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode a frame.")
    return image


def _result_path(rel: str) -> Path:
    rel = rel.replace("\\", "/").lstrip("/")
    parts = [part for part in rel.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise HTTPException(400, "Bad path.")
    path = (RESULTS / Path(*parts)).resolve()
    root = RESULTS.resolve()
    if path != root and root not in path.parents:
        raise HTTPException(404)
    if not path.is_file():
        raise HTTPException(404)
    return path


def _result_url(rel: str) -> str:
    return "/api/m6/results/file/" + quote(rel, safe="/")


def _media_type(name: str) -> str:
    if name.endswith(".mp4"):
        return "video/mp4"
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".txt"):
        return "text/plain"
    return "image/jpeg"


@router.get("/api/m6/results")
def list_results():
    """Committed HW4/results. The hosted server has no upload disk between requests."""
    clips = []
    tracking = []
    sfm = None
    if RESULTS.is_dir():
        for folder in sorted(path for path in RESULTS.iterdir() if path.is_dir() and path.name.startswith("flow_")):
            stats_path = folder / "stats.json"
            if not stats_path.is_file():
                continue
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            files = {}
            for key, name in (stats.get("files") or {}).items():
                if name and (folder / name).is_file():
                    files[key] = _result_url(f"{folder.name}/{name}")
            stats["files"] = files
            stats["id"] = folder.name
            stats.pop("video", None)
            clips.append(stats)
        for path in sorted(RESULTS.glob("tracking_*.json")):
            name = path.stem[len("tracking_"):]
            data = json.loads(path.read_text(encoding="utf-8"))
            item = {
                "id": name,
                "label": data.get("label") or name,
                "note": data.get("note"),
                "summary": data.get("summary"),
                "example_text": data.get("example_text"),
                "scratch_error": data.get("scratch_error"),
                "opencv_error": data.get("opencv_error"),
                "template_error": data.get("template_error"),
            }
            for which in ("frame1", "frame2"):
                jpg = f"tracking_{name}_{which}.jpg"
                if (RESULTS / jpg).is_file():
                    item[which] = _result_url(jpg)
            tracking.append(item)
        sfm_path = RESULTS / "sfm" / "sfm.json"
        if sfm_path.is_file():
            body = json.loads(sfm_path.read_text(encoding="utf-8"))
            body.pop("points", None)
            views = []
            for index in range(1, 5):
                view = f"sfm/view{index}.jpg"
                if (RESULTS / view).is_file():
                    views.append(_result_url(view))
            sfm = {"result": body, "views": views}
    return {"clips": clips, "tracking": tracking, "sfm": sfm}


@router.get("/api/m6/results/file/{path:path}")
def result_file(path: str):
    file_path = _result_path(path)
    return FileResponse(file_path, media_type=_media_type(file_path.name))


class TrackRequest(BaseModel):
    frame1_id: str | None = None
    frame2_id: str | None = None
    frame1_data: str | None = None
    frame2_data: str | None = None
    pair: str | None = None
    points: list[list[float]]
    ground_truth: list[list[float]] | None = None
    example_index: int = 0


def _track_images(req: TrackRequest) -> tuple[np.ndarray, np.ndarray]:
    if req.pair:
        name = safe_id(req.pair)
        first = RESULTS / f"tracking_{name}_frame1.jpg"
        second = RESULTS / f"tracking_{name}_frame2.jpg"
        if not first.is_file() or not second.is_file():
            raise HTTPException(404, "That saved pair has no frames.")
        image1 = cv2.imread(str(first), cv2.IMREAD_COLOR)
        image2 = cv2.imread(str(second), cv2.IMREAD_COLOR)
        if image1 is None or image2 is None:
            raise HTTPException(400, "Could not read the saved frames.")
        return image1, image2
    if req.frame1_data and req.frame2_data:
        try:
            return _decode_data_url(req.frame1_data), _decode_data_url(req.frame2_data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    if req.frame1_id and req.frame2_id:
        return read_image(req.frame1_id), read_image(req.frame2_id)
    raise HTTPException(400, "Send two frames as data URLs, two saved image ids, or a saved pair name.")


@router.post("/api/m6/track")
async def track(req: TrackRequest):
    image1, image2 = _track_images(req)
    if len(req.points) < 1:
        raise HTTPException(400, "Click at least one point in frame 1.")
    if req.ground_truth is not None and len(req.ground_truth) not in (0, len(req.points)):
        raise HTTPException(400, "Click the same number of ground-truth points as predictions, in the same order.")
    gt = req.ground_truth or None

    def work():
        return flow_core.evaluate_pair(image1, image2, req.points, gt, req.example_index)

    try:
        result = await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _pair_payload(result)


@router.post("/api/m6/sfm")
async def structure_from_motion(
    files: list[UploadFile] = File(...),
    clicks: str = Form(...),
    known_length: float = Form(...),
    unit: str = Form("in"),
    edge_i: int = Form(0),
    edge_j: int = Form(1),
    check_length: float = Form(0.0),
    check_i: int = Form(1),
    check_j: int = Form(2),
    calibration_id: str = Form("default"),
):
    if len(files) != 4:
        raise HTTPException(400, "Upload exactly 4 views, in order.")
    if known_length <= 0:
        raise HTTPException(400, "The taped length has to be positive.")
    try:
        payload = json.loads(clicks)
        raw = payload["boundary"] if isinstance(payload, dict) else payload
        click_sets = [np.asarray(view, dtype=np.float64) for view in raw]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "Clicks must be four lists of [u, v] points, in the same order.") from exc
    blobs = []
    for upload in files:
        data = await upload.read()
        blobs.append((upload.filename or "view.jpg", data))
    calibration = _load_calibration(calibration_id)

    def work():
        images = []
        scaled_clicks = []
        for (name, data), click in zip(blobs, click_sets):
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode {name}.")
            image, click = _resize_long_edge(image, click, 1600)
            images.append(image)
            scaled_clicks.append(click)
        measured = None
        if check_length > 0:
            measured = {"length": check_length, "unit": unit, "endpoints": [check_i, check_j]}
        result = sfm_core.reconstruct(
            images, scaled_clicks, calibration, known_length, unit,
            edge=(edge_i, edge_j), measured_edges=measured,
        )
        body = sfm_core.result_to_json(result)
        body["workout"] = sfm_core.format_worked_example(result)
        if isinstance(payload, dict):
            if payload.get("camera_positions"):
                body["camera_positions"] = payload["camera_positions"]
            if payload.get("note"):
                body["note"] = payload["note"]
        if check_length > 0:
            edges = body["polygon_unit"]["edge_lengths"]
            n = len(edges)
            if (check_i + 1) % n == check_j:
                index = check_i
            elif (check_j + 1) % n == check_i:
                index = check_j
            else:
                index = None
            if index is not None:
                body["check"] = {
                    "tape": check_length,
                    "reconstructed": edges[index],
                    "error": edges[index] - check_length,
                    "unit": unit,
                }
        # The report does not need every SIFT point.
        body["points"] = body["boundary_points"]
        return body

    try:
        return await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/api/m6/images/{image_id}")
def image_info(image_id: str):
    """Exists so the tracking page can confirm a frame is still on disk."""
    meta = image_meta(image_id)
    meta["url"] = f"/api/images/{meta['id']}.jpg"
    return meta
