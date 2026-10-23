"""Room dimensions from a short stereo slide.

A clip pauses at the left stop, slides along the rail, and pauses at the right
stop. The two still stretches are the stereo pair. The rail length fixes the
scale. Floor and ceiling planes give the height. The wall you are facing gives
the horizontal span (room length or room width, depending on which wall).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from stereo_core import camera_matrix, edge_lengths, fit_plane, scale_intrinsics, triangulate_dlt


def motion_scores(frames: list[np.ndarray]) -> np.ndarray:
    scores = np.empty(len(frames) - 1, dtype=np.float64)
    for i, (a, b) in enumerate(zip(frames, frames[1:])):
        scores[i] = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    return scores


def still_runs(scores: np.ndarray, fraction: float = 0.35) -> tuple[list[tuple[int, int]], float]:
    """Longest quiet stretches. A run (start, end) includes both end frames."""
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = float(scores.min()), float(scores.max())
    thresh = lo + fraction * (hi - lo) if hi > lo else lo
    low = scores <= thresh + 1e-9
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(low):
        if not low[i]:
            i += 1
            continue
        j = i
        while j < len(low) and low[j]:
            j += 1
        runs.append((i, j))
        i = j
    runs.sort(key=lambda run: run[1] - run[0], reverse=True)
    return runs, thresh


def average_run(frames: list[np.ndarray], start: int, end: int) -> np.ndarray:
    chunk = np.stack([frame.astype(np.float32) for frame in frames[start: end + 1]], axis=0)
    averaged = chunk.mean(axis=0)
    if np.issubdtype(frames[0].dtype, np.integer):
        return np.clip(np.rint(averaged), 0, 255).astype(frames[0].dtype)
    return averaged


def stereo_pair_from_clip(frames: list[np.ndarray], fraction: float = 0.35) -> dict:
    """Average the first and last quiet stretches. The slide sits between them."""
    if len(frames) < 4:
        raise ValueError("Need a longer clip: pause, slide, pause.")
    scores = motion_scores(frames)
    runs, thresh = still_runs(scores, fraction)
    if len(runs) < 2:
        raise ValueError("Could not find two still stretches. Pause for about a second at each stop.")
    ordered = sorted(runs, key=lambda run: run[0])
    first, last = ordered[0], ordered[-1]
    if last[0] <= first[1]:
        raise ValueError("The two still stretches overlap. Slide farther between the pauses.")
    return {
        "left": average_run(frames, first[0], first[1]),
        "right": average_run(frames, last[0], last[1]),
        "left_span": first,
        "right_span": last,
        "motion": scores,
        "threshold": thresh,
    }


def plane_distance(normal_a: np.ndarray, point_a: np.ndarray, normal_b: np.ndarray, point_b: np.ndarray) -> float:
    """Distance between two nearly parallel planes."""
    normal_b = np.asarray(normal_b, dtype=np.float64).copy()
    if float(normal_a @ normal_b) < 0:
        normal_b = -normal_b
    return abs(float((point_b - point_a) @ normal_a))


def _oriented_extents(corners: np.ndarray, up: np.ndarray) -> tuple[float, float]:
    corners = np.asarray(corners, dtype=np.float64)
    edges = [corners[(i + 1) % len(corners)] - corners[i] for i in range(len(corners))]
    horizontal, vertical = [], []
    for edge in edges:
        length = float(np.linalg.norm(edge))
        if length < 1e-12:
            continue
        if abs(float(edge @ up)) / length < 0.5:
            horizontal.append(length)
        else:
            vertical.append(length)
    if not horizontal or not vertical:
        raise ValueError("The wall corners do not form a horizontal and a vertical edge.")
    return float(np.mean(horizontal)), float(np.mean(vertical))


def dimensions_from_planes(
    floor: np.ndarray,
    ceiling: np.ndarray,
    long_wall: np.ndarray,
    short_wall: np.ndarray,
) -> dict:
    """Height from the floor/ceiling planes. Length and width from the two facing walls.

    long_wall corners span the room length. short_wall corners span the room width.
    """
    floor_n, floor_c = fit_plane(np.asarray(floor, dtype=np.float64))
    ceil_n, ceil_c = fit_plane(np.asarray(ceiling, dtype=np.float64))
    length, height_long = _oriented_extents(long_wall, floor_n)
    width, height_short = _oriented_extents(short_wall, floor_n)
    return {
        "length_m": length,
        "width_m": width,
        "height_m": plane_distance(floor_n, floor_c, ceil_n, ceil_c),
        "height_from_long_wall_m": height_long,
        "height_from_short_wall_m": height_short,
    }


def synthetic_room(length: float = 4.0, width: float = 3.0, height: float = 2.5, seed: int = 0) -> dict:
    """A box. Floor y=0, ceiling y=height, long wall z=0 (spans x), short wall x=0 (spans z)."""
    rng = np.random.default_rng(seed)

    def sprinkle(origin, axis_a, axis_b, count):
        s = rng.random((count, 1))
        t = rng.random((count, 1))
        return origin + s * axis_a + t * axis_b

    floor = sprinkle(np.zeros(3), np.array([length, 0, 0]), np.array([0, 0, width]), 40)
    ceiling = sprinkle(np.array([0, height, 0]), np.array([length, 0, 0]), np.array([0, 0, width]), 40)
    long_corners = np.array([
        [0, 0, 0],
        [length, 0, 0],
        [length, height, 0],
        [0, height, 0],
    ], dtype=np.float64)
    short_corners = np.array([
        [0, 0, 0],
        [0, 0, width],
        [0, height, width],
        [0, height, 0],
    ], dtype=np.float64)
    return {
        "floor": floor,
        "ceiling": ceiling,
        "long_wall": long_corners,
        "short_wall": short_corners,
        "truth": {"length_m": length, "width_m": width, "height_m": height},
    }


def sift_matches(left: np.ndarray, right: np.ndarray, max_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    gray1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gray2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    sift = cv2.SIFT_create()
    keys1, desc1 = sift.detectAndCompute(gray1, None)
    keys2, desc2 = sift.detectAndCompute(gray2, None)
    if desc1 is None or desc2 is None or len(keys1) < 8 or len(keys2) < 8:
        raise ValueError("Not enough texture to match the two views.")
    pairs = cv2.BFMatcher().knnMatch(desc1, desc2, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        raise ValueError("Only a few features matched. Point the camera at a textured wall.")
    good = sorted(good, key=lambda m: m.distance)[:max_points]
    pts1 = np.float64([keys1[m.queryIdx].pt for m in good])
    pts2 = np.float64([keys2[m.trainIdx].pt for m in good])
    return pts1, pts2


def relative_pose(pts1: np.ndarray, pts2: np.ndarray, k: np.ndarray, baseline: float) -> dict:
    """Essential matrix, cheirality check, then scale the unit translation to the rail length."""
    essential, mask = cv2.findEssentialMat(pts1, pts2, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if essential is None:
        raise ValueError("Could not fit an essential matrix.")
    _count, r_mat, t_vec, pose_mask = cv2.recoverPose(essential, pts1, pts2, k, mask=mask)
    t_vec = np.asarray(t_vec, dtype=np.float64).ravel()
    t_vec = t_vec / np.linalg.norm(t_vec) * float(baseline)
    p1 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = k @ np.hstack([np.asarray(r_mat, dtype=np.float64), t_vec.reshape(3, 1)])
    return {"R": np.asarray(r_mat, dtype=np.float64), "t": t_vec, "P1": p1, "P2": p2, "inliers": int(_count)}


def disparity_sgbm(left: np.ndarray, right: np.ndarray, num_disparities: int = 64, block: int = 5) -> np.ndarray:
    if num_disparities % 16:
        num_disparities += 16 - num_disparities % 16
    gray1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gray2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=int(num_disparities),
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(gray1, gray2).astype(np.float32) / 16.0


def depth_from_disparity(disparity: np.ndarray, focal: float, baseline: float) -> np.ndarray:
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = disparity > 0.5
    depth[valid] = focal * baseline / disparity[valid]
    return depth


def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
    valid = disparity > 0.5
    shown = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(disparity[valid], [2, 98])
        span = max(float(hi - lo), 1e-3)
        shown[valid] = np.clip((disparity[valid] - lo) / span * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(shown, cv2.COLORMAP_TURBO)


def wall_from_clicks(left_xy: np.ndarray, right_xy: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Four wall corners in order: bottom-left, bottom-right, top-right, top-left."""
    xyz = triangulate_dlt(left_xy, right_xy, p1, p2)
    edges = edge_lengths(xyz)
    return {
        "xyz": xyz,
        "edges_m": edges,
        "width_m": float((edges[0] + edges[2]) / 2),
        "height_m": float((edges[1] + edges[3]) / 2),
    }


def downscale(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 0.999:
        return image, 1.0
    size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def intrinsics_for_frame(k: np.ndarray, calib_wh: tuple[int, int], frame_wh: tuple[int, int]) -> np.ndarray:
    return scale_intrinsics(k, calib_wh, frame_wh)


def read_frames(path: Path | str, max_side: int = 960, stride: int = 2) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open {path}.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            small, _ = downscale(frame, max_side)
            frames.append(small)
        index += 1
    cap.release()
    if len(frames) < 4:
        raise ValueError("The video has too few frames.")
    return frames, fps


def k_for_frames(k: np.ndarray, calib_wh: tuple[int, int], frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return intrinsics_for_frame(k, calib_wh, (width, height))


def default_video_k(frame: np.ndarray, hfov_deg: float = 65.0) -> np.ndarray:
    """A stand-in K when the video has not been calibrated yet. Horizontal field of view ~65°."""
    height, width = frame.shape[:2]
    fx = (width / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    return camera_matrix(fx, fx, (width - 1) / 2, (height - 1) / 2)


def measure_room_frames(
    left: np.ndarray,
    right: np.ndarray,
    k: np.ndarray,
    baseline: float,
    clicks_left: np.ndarray | None = None,
    clicks_right: np.ndarray | None = None,
    num_disparities: int = 64,
) -> dict:
    pts1, pts2 = sift_matches(left, right)
    pose = relative_pose(pts1, pts2, k, baseline)
    disp = disparity_sgbm(left, right, num_disparities=num_disparities)
    depth = depth_from_disparity(disp, float(k[0, 0]), baseline)
    valid = np.isfinite(depth)
    body = {
        "R": pose["R"],
        "t": pose["t"],
        "baseline_m": float(baseline),
        "inliers": pose["inliers"],
        "matches": len(pts1),
        "disparity": disp,
        "depth": depth,
        "median_depth_m": float(np.median(depth[valid])) if np.any(valid) else None,
        "P1": pose["P1"],
        "P2": pose["P2"],
    }
    if clicks_left is not None and clicks_right is not None:
        wall = wall_from_clicks(clicks_left, clicks_right, pose["P1"], pose["P2"])
        body["wall"] = wall
    return body
