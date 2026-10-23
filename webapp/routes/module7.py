"""Module 7 API: calibrated stereo sizes, uncalibrated numbers, room dimensions."""

from __future__ import annotations

import json

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from webapp.storage import ROOT, data_url
from room_core import colorize_disparity, dimensions_from_planes, disparity_sgbm, measure_room_frames, synthetic_room
from stereo_core import (
    load_demo,
    load_monocular,
    match_points,
    measure_points,
    result_to_json,
    scale_intrinsics,
)
from uncalibrated_numbers import demo_correspondences, format_worksheet, worksheet

router = APIRouter()
SAMPLE_DIR = ROOT / "HW5" / "samples"

REFERENCES = [
    "Hartley, R. and Zisserman, A. Multiple View Geometry in Computer Vision, 2nd ed. Cambridge University Press, 2004.",
    "Szeliski, R. Computer Vision: Algorithms and Applications, 2nd ed. Springer, 2022.",
    "Zhang, Z. A flexible new technique for camera calibration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2000.",
    "Hartley, R. In defense of the eight-point algorithm. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1997.",
    "Hartley, R. Theory and practice of projective rectification. International Journal of Computer Vision, 1999.",
    "Bougnoux, S. From projective to Euclidean space under any practical situation, a criticism of self-calibration. ICCV, 1998.",
    "Hirschmüller, H. Stereo processing by semiglobal matching and mutual information. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2008.",
    "Fischler, M. and Bolles, R. Random sample consensus. Communications of the ACM, 1981.",
    "Lowe, D. Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision, 2004.",
    "OpenCV documentation, Camera Calibration and 3D Reconstruction (calib3d).",
]


def _decode(blob: bytes, name: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"Could not read {name}. Use a JPG or PNG.")
    return image


def _fit(image: np.ndarray, points: np.ndarray | None, max_side: int = 1600):
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 0.999:
        image = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
        if points is not None:
            points = points * scale
    return image, points


def _meters(value: str | None, unit: str) -> float | None:
    if value is None or value == "":
        return None
    from cv_core import UNIT_TO_METERS

    try:
        return float(value) * UNIT_TO_METERS[unit]
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, "Baseline and tape values must be numbers, and the unit must be m, cm, mm, in, or ft.") from exc


def _points(raw: str, what: str) -> np.ndarray:
    try:
        pts = np.asarray(json.loads(raw), dtype=np.float64)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"{what} must be a JSON list of [u, v] points.") from exc
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 1:
        raise HTTPException(400, f"{what} must be a JSON list of [u, v] points.")
    return pts


@router.get("/api/m7/demo")
def demo_meta():
    _left, _right, meta = load_demo()
    meta = dict(meta)
    meta["left_url"] = "/api/m7/samples/left.png"
    meta["right_url"] = "/api/m7/samples/right.png"
    return meta


@router.get("/api/m7/samples/{name}")
def sample_file(name: str):
    if name not in {"left.png", "right.png"}:
        raise HTTPException(404)
    path = SAMPLE_DIR / name
    if not path.is_file():
        load_demo()
    return FileResponse(path, media_type="image/png")


@router.post("/api/m7/measure")
async def measure(
    sample: str = Form(""),
    shape: str = Form("rectangle"),
    points: str = Form(...),
    matches: str = Form(""),
    baseline: str = Form("0.2"),
    unit: str = Form("m"),
    tape_width: str = Form(""),
    tape_height: str = Form(""),
    tape_diameter: str = Form(""),
    left: UploadFile | None = File(None),
    right: UploadFile | None = File(None),
):
    left_xy = _points(points, "Points")
    baseline_m = _meters(baseline, unit)
    if baseline_m is None or baseline_m <= 0:
        raise HTTPException(400, "The slide length has to be positive.")

    if sample != "1":
        if left is None or right is None:
            raise HTTPException(400, "Upload the left and right photos, or use the sample pair.")
        left_bytes = await left.read()
        right_bytes = await right.read()
    else:
        left_bytes = right_bytes = b""

    def run():
        pts = _points(points, "Points")
        if sample == "1":
            image_l, image_r, meta = load_demo()
            k = np.asarray(meta["k"], dtype=np.float64)
            if matches:
                right_xy = _points(matches, "Matches")
            else:
                right_xy, _found = match_points(image_l, image_r, pts)
        else:
            image_l = _decode(left_bytes, "the left photo")
            image_r = _decode(right_bytes, "the right photo")
            before = image_l.shape[1]
            image_l, pts = _fit(image_l, pts)
            image_r, _unused = _fit(image_r, None)
            if image_l.shape[:2] != image_r.shape[:2]:
                raise HTTPException(400, "The two photos must be the same size.")
            scale = image_l.shape[1] / before
            mono = load_monocular()
            k = scale_intrinsics(mono["k"], mono["image_size"], (image_l.shape[1], image_l.shape[0]))
            if matches:
                right_xy = _points(matches, "Matches") * scale
            else:
                right_xy, _found = match_points(image_l, image_r, pts)
        try:
            measured = measure_points(pts, right_xy, k, baseline_m, shape)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        from stereo_core import draw_clicks

        body = result_to_json(measured)
        body["left_url"] = data_url(draw_clicks(image_l, pts, (40, 180, 60)), "jpg")
        body["right_url"] = data_url(draw_clicks(image_r, right_xy, (40, 90, 220), closed=False), "jpg")
        tape = {}
        provided = {"width": tape_width, "height": tape_height, "diameter": tape_diameter}
        for key, field_name in (("width_m", "width"), ("height_m", "height"), ("diameter_m", "diameter")):
            truth = _meters(provided[field_name], unit)
            if truth is not None and key in measured:
                tape[f"{field_name}_m"] = truth
                tape[f"{field_name}_error_m"] = float(measured[key] - truth)
        body["tape"] = tape
        body["unit"] = unit
        return body

    return await run_in_threadpool(run)


@router.post("/api/m7/uncalibrated")
async def uncalibrated(sample: str = Form("1"), baseline: str = Form("0.2"), unit: str = Form("m")):
    baseline_m = _meters(baseline, unit) or 0.2

    def work():
        pts1, pts2, k, size = demo_correspondences()
        body = worksheet(pts1, pts2, size, baseline_m, k)

        def convert(value):
            if isinstance(value, np.ndarray):
                return np.asarray(value).tolist()
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        out = convert(body)
        out.pop("reconstructed_points", None)
        out["text"] = format_worksheet(body)
        return out

    if sample != "1":
        raise HTTPException(400, "The numbers worksheet uses the built-in pair. Run HW5/uncalibrated_numbers.py on your own photos.")
    return await run_in_threadpool(work)


@router.post("/api/m7/room")
async def room(
    sample: str = Form(""),
    baseline: str = Form("0.2"),
    unit: str = Form("m"),
    points: str = Form(""),
    matches: str = Form(""),
    left: UploadFile | None = File(None),
    right: UploadFile | None = File(None),
):
    if sample == "box":
        built = synthetic_room()
        dims = dimensions_from_planes(built["floor"], built["ceiling"], built["long_wall"], built["short_wall"])
        image_l, image_r, _meta = load_demo()
        disparity = disparity_sgbm(image_l, image_r, 64)
        dims = dict(dims)
        dims["source"] = "synthetic box, 4 m by 3 m by 2.5 m"
        dims["disparity_url"] = data_url(colorize_disparity(disparity), "jpg")
        dims["note"] = (
            "The length, width, and height come from plane fits on a synthetic room. "
            "The picture is the disparity of the sample stereo pair, so you can see what a sideways slide looks like."
        )
        return dims

    if left is None or right is None:
        raise HTTPException(400, "Upload the two still frames, or ask for the synthetic box.")
    left_bytes = await left.read()
    right_bytes = await right.read()
    baseline_m = _meters(baseline, unit)
    if baseline_m is None or baseline_m <= 0:
        raise HTTPException(400, "The slide length has to be positive.")

    def work():
        image_l = _decode(left_bytes, "the left frame")
        image_r = _decode(right_bytes, "the right frame")
        pts = _points(points, "Corners") if points else None
        raw_width = image_l.shape[1]
        image_l, pts = _fit(image_l, pts, 960)
        image_r, _unused = _fit(image_r, None, 960)
        scale = image_l.shape[1] / raw_width
        if image_l.shape[:2] != image_r.shape[:2]:
            raise HTTPException(400, "The two frames must be the same size.")
        from room_core import default_video_k

        k = default_video_k(image_l)
        right_xy = None
        if pts is not None:
            if len(pts) != 4:
                raise HTTPException(400, "Click 4 wall corners: bottom-left, bottom-right, top-right, top-left.")
            if matches:
                right_xy = _points(matches, "Matches") * scale
            else:
                right_xy, _found = match_points(image_l, image_r, pts)
        try:
            measured = measure_room_frames(image_l, image_r, k, baseline_m, pts, right_xy, num_disparities=64)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        body = {
            "baseline_m": baseline_m,
            "inliers": measured["inliers"],
            "matches": measured["matches"],
            "median_depth_m": measured["median_depth_m"],
            "t": measured["t"].tolist(),
            "disparity_url": data_url(colorize_disparity(measured["disparity"]), "jpg"),
            "k_note": "No video calibration was uploaded, so K assumes a 65° horizontal field of view. Run calibrate_video.py for a real K.",
        }
        wall = measured.get("wall")
        if wall is not None:
            body["width_m"] = wall["width_m"]
            body["height_m"] = wall["height_m"]
            body["edges_m"] = wall["edges_m"].tolist()
        return body

    return await run_in_threadpool(work)


@router.get("/api/m7/report")
def report():
    _image_l, _image_r, meta = load_demo()
    k = np.asarray(meta["k"], dtype=np.float64)
    rows = []
    for shape, raw in meta["shapes"].items():
        left_xy = np.asarray(raw, dtype=np.float64)
        right_xy = left_xy.copy()
        right_xy[:, 0] -= meta["disparity_px"]
        measured = measure_points(left_xy, right_xy, k, meta["baseline_m"], shape)
        rows.append({
            "shape": shape,
            "estimated": result_to_json(measured),
            "truth": meta["truth"][shape],
        })
    room = synthetic_room()
    dims = dimensions_from_planes(room["floor"], room["ceiling"], room["long_wall"], room["short_wall"])
    return {
        "baseline_m": meta["baseline_m"],
        "focal_px": meta["k"][0][0],
        "disparity_px": meta["disparity_px"],
        "rows": rows,
        "room": dims,
        "room_truth": room["truth"],
        "references": REFERENCES,
        "sample_note": meta["note"],
    }
