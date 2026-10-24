#!/usr/bin/env python3
"""
make_demo_results.py
====================

README — SYNTHETIC STAND-IN FOR THE HOSTED MODULE 6 PAGES
----------------------------------------------------------

WHAT THIS DOES
  Builds two short clips and four views of a flat poster, then runs the same
  scripts the assignment uses: optical flow, Lucas–Kanade with known pixel
  locations, and four-view structure from motion. Nothing here is a phone
  recording. Every written JSON carries a note that says so. Replace the
  outputs by recording the clips and photos described in HW4/videos/README.txt
  and HW4/sfm_images/README.txt.

HOW TO RUN (from the repository root)
  .venv/bin/python HW4/make_demo_results.py

OUTPUT
  HW4/results/flow_static_camera/   HW4/results/flow_moving_camera/
  HW4/results/tracking_*.json and the two frame JPEGs
  HW4/results/sfm/
  Raw clips and poster photos go to HW4/videos and HW4/sfm_images (gitignored).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PY = sys.executable
NOTE = (
    "Synthetic stand-in, not a phone recording. "
    "Replace it with the clips and photos described in HW4/videos/README.txt "
    "and HW4/sfm_images/README.txt before submitting."
)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=7, macro_block_size=1,
    )
    for frame in frames:
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()


def _texture(width: int, height: int, seed: int, grid: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(30, 220, (height, width, 3), dtype=np.uint8)
    if grid:
        for i in range(0, width, 36):
            cv2.line(image, (i, 0), (i, height), (20, 20, 20), 1)
        for j in range(0, height, 36):
            cv2.line(image, (0, j), (width, j), (20, 20, 20), 1)
    for k in range(18):
        x, y = int(rng.integers(0, width - 40)), int(rng.integers(0, height - 30))
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.rectangle(image, (x, y), (x + 28, y + 18), color, -1)
    return image


def _static_clip() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    width, height, count, step = 640, 360, 48, 4.0
    background = _texture(width, height, 2)
    frames = []
    points = None
    truth = None
    for index in range(count):
        frame = background.copy()
        x0, y0 = 80 + index * step, 130.0
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x0) + 128, int(y0) + 96), (250, 250, 250), -1)
        for col in range(8):
            for row in range(6):
                if (col + row) % 2 == 0:
                    continue
                xa, ya = int(x0) + col * 16, int(y0) + row * 16
                cv2.rectangle(frame, (xa, ya), (xa + 16, ya + 16), (20, 40, 180), -1)
        frames.append(frame)
        if index == 0:
            pts = []
            for col in range(1, 6):
                for row in range(1, 3):
                    pts.append([x0 + col * 16, y0 + row * 16])
            points = np.asarray(pts, dtype=np.float64)
            truth = points + np.array([step, 0.0])
    return frames, points, truth


def _blocks(width: int, height: int, seed: int) -> np.ndarray:
    """
    A checkerboard fills the frame, so a camera slide moves every pixel.

    Pixel noise will not compress, and a blank wall has no flow. The checker
    does both jobs and stays small on disk.
    """
    del seed
    ys, xs = np.mgrid[0:height, 0:width]
    cell = ((xs // 16 + ys // 16) % 2).astype(np.uint8)
    image = np.zeros((height, width, 3), np.uint8)
    image[..., 0] = np.where(cell, 40, 220)
    image[..., 1] = np.where(cell, 70, 200)
    image[..., 2] = np.where(cell, 160, 60)
    return image


def _moving_clip() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    width, height, count = 640, 360, 48
    shift = np.array([4.0, -1.5])
    scene = _blocks(width + 500, height + 220, 5)
    x0, y0 = 40, 40
    frames = []
    for index in range(count):
        matrix = np.array([[1.0, 0.0, -shift[0] * index], [0.0, 1.0, -shift[1] * index]], dtype=np.float64)
        warped = cv2.warpAffine(scene, matrix, (scene.shape[1], scene.shape[0]), flags=cv2.INTER_LINEAR)
        frames.append(warped[y0:y0 + height, x0:x0 + width].copy())
    gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    found = cv2.goodFeaturesToTrack(gray, 12, 0.01, 28)
    points = found.reshape(-1, 2).astype(np.float64)
    margin = 12.0
    keep = (
        (points[:, 0] > margin) & (points[:, 0] < width - margin)
        & (points[:, 1] > margin) & (points[:, 1] < height - margin)
    )
    points = points[keep][:10]
    truth = points - shift
    return frames, points, truth


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def _annotate_flow(folder: Path, label: str) -> None:
    path = folder / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    stats["label"] = label
    stats["note"] = NOTE
    path.write_text(json.dumps(stats), encoding="utf-8")


def _flow_and_track(name: str, label: str, frames, points, truth, fps: float) -> None:
    video = SCRIPT_DIR / "videos" / f"{name}.mp4"
    _write_video(video, frames, fps)
    clicks = SCRIPT_DIR / "videos" / f"{name}_clicks.json"
    clicks.write_text(json.dumps({
        "frame1": points.tolist(),
        "frame2": truth.tolist(),
    }), encoding="utf-8")
    _run([
        PY, "HW4/optical_flow_video.py",
        "--video", str(video), "--duration", str(len(frames) / fps), "--width", "640",
    ])
    _annotate_flow(SCRIPT_DIR / "results" / f"flow_{name}", label)
    _run([
        PY, "HW4/validate_tracking.py",
        "--video", str(video), "--time", "0", "--clicks", str(clicks),
        "--save", "--width", "640", "--label", label, "--note", NOTE,
    ])


def _look_at(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = target - center
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    translation = -rotation @ center
    return rotation, translation


def _poster() -> None:
    sys.path.insert(0, str(ROOT / "HW1"))
    sys.path.insert(0, str(SCRIPT_DIR))
    import cv_core

    calibration = cv_core.CameraCalibration.load(ROOT / "HW1" / "camera_calibration.json")
    width, height = 960, 720
    k, dist = calibration.intrinsics_for(width, height)
    poster_w = 11 * 0.0254
    poster_h = 8.5 * 0.0254
    # A large unique plane, so SIFT has correspondences across the whole frame.
    # The taped rectangle is drawn in the middle; only its corners are the boundary.
    plane_w, plane_h = 1.6, 1.2
    texture = _texture(1400, 1050, 11, grid=False)
    tex_h, tex_w = texture.shape[:2]

    def _to_tex(x_m, y_m):
        u = int(round((x_m + plane_w / 2) / plane_w * (tex_w - 1)))
        v = int(round((plane_h / 2 - y_m) / plane_h * (tex_h - 1)))
        return u, v

    corners_tex = np.array([
        _to_tex(-poster_w / 2, poster_h / 2),
        _to_tex(poster_w / 2, poster_h / 2),
        _to_tex(poster_w / 2, -poster_h / 2),
        _to_tex(-poster_w / 2, -poster_h / 2),
    ], dtype=np.int32)
    cv2.polylines(texture, [corners_tex], True, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(texture, "MODULE 6", (corners_tex[0, 0] + 8, corners_tex[0, 1] + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    world = np.array([
        [-poster_w / 2, poster_h / 2, 0.0],
        [poster_w / 2, poster_h / 2, 0.0],
        [poster_w / 2, -poster_h / 2, 0.0],
        [-poster_w / 2, -poster_h / 2, 0.0],
    ], dtype=np.float64)
    centers = [
        np.array([0.00, 0.00, 0.62]),
        np.array([0.28, 0.04, 0.58]),
        np.array([-0.30, 0.06, 0.55]),
        np.array([0.10, -0.18, 0.60]),
    ]
    out_dir = SCRIPT_DIR / "sfm_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    click_sets = []
    for index, center in enumerate(centers, start=1):
        rotation, translation = _look_at(center, np.zeros(3))
        plane = np.array([
            [plane_w / tex_w, 0.0, -plane_w / 2],
            [0.0, -plane_h / tex_h, plane_h / 2],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        projection = k @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])
        ideal = cv2.warpPerspective(texture, projection @ plane, (width, height))
        xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        grid = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(grid, k, dist, P=k).reshape(height, width, 2)
        image = cv2.remap(ideal, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        path = out_dir / f"view{index}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(path)
        rvec, _ = cv2.Rodrigues(rotation)
        pixels, _ = cv2.projectPoints(world, rvec, translation.reshape(3, 1), k, dist)
        pixels = pixels.reshape(-1, 2)
        if np.any(pixels < 20) or np.any(pixels[:, 0] > width - 20) or np.any(pixels[:, 1] > height - 20):
            raise SystemExit(f"View {index} clicks fell outside the frame: {pixels}")
        click_sets.append(pixels.tolist())
    clicks_path = out_dir / "sfm_points.json"
    clicks_path.write_text(json.dumps({
        "boundary": click_sets,
        "camera_positions": [
            {"view": i + 1, "x_m": float(c[0]), "y_m": float(c[1]), "z_m": float(c[2]),
             "note": "true center used to render this view, poster at the origin"}
            for i, c in enumerate(centers)
        ],
        "measured": {"length": 8.5, "unit": "in", "endpoints": [1, 2]},
        "note": NOTE,
    }, indent=2), encoding="utf-8")
    _run([
        PY, "HW4/structure_from_motion.py",
        "--images", *[str(path) for path in paths],
        "--clicks", str(clicks_path),
        "--known-length", "11", "--unit", "in",
        "--edge", "0", "1",
    ])
    _run([PY, "HW4/sfm_worked_example.py"])


def main() -> None:
    sfm_only = "--sfm-only" in sys.argv
    if not sfm_only:
        fps = 8.0
        static, static_pts, static_truth = _static_clip()
        moving, moving_pts, moving_truth = _moving_clip()
        _flow_and_track("static_camera", "Static camera, moving object (synthetic)", static, static_pts, static_truth, fps)
        _flow_and_track("moving_camera", "Moving camera (synthetic)", moving, moving_pts, moving_truth, fps)
    _poster()
    print("Wrote HW4/results")


if __name__ == "__main__":
    main()
