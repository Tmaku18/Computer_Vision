"""
Four-view structure from motion for a planar object.

The pipeline undistorts with the Module 2 calibration, estimates the essential
matrix between the first two views, triangulates, attaches the other two
cameras with PnP, then bundle-adjusts. A tape-measured length fixes the scale,
which the images alone cannot provide. Boundary clicks are triangulated with
the same cameras and fitted to a plane.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

_HW1 = Path(__file__).resolve().parent.parent / "HW1"
if str(_HW1) not in sys.path:
    sys.path.insert(0, str(_HW1))

import cv_core  # noqa: E402
from cv_core import CameraCalibration  # noqa: E402


def load_intrinsics(calibration: CameraCalibration | str | Path, width: int, height: int):
    """K and distortion for an image of the given size, from the Module 2 file or object."""
    if not isinstance(calibration, CameraCalibration):
        calibration = CameraCalibration.load(calibration)
    return calibration.intrinsics_for(width, height)


def undistort_image(image: np.ndarray, k: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort into the same pixel frame, so the original K still applies."""
    return cv2.undistort(image, k, dist, None, k)


def projection_matrix(k: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """3×4 camera matrix P = K [R | t]."""
    rt = np.hstack([rotation, np.asarray(translation, dtype=np.float64).reshape(3, 1)])
    return k @ rt


def triangulate_dlt(p1: np.ndarray, p2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Linear triangulation (DLT) of N correspondences. Returns (N, 3)."""
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    out = np.empty((len(pts1), 3), dtype=np.float64)
    for i, (a, b) in enumerate(zip(pts1, pts2)):
        out[i] = triangulate_one(p1, p2, a, b)
    return out


def dlt_matrix(p1: np.ndarray, p2: np.ndarray, uv1, uv2) -> np.ndarray:
    """
    4×4 design matrix for one point. Each image contributes the two independent
    rows of ``[u]× P``.
    """
    u, v = float(uv1[0]), float(uv1[1])
    up, vp = float(uv2[0]), float(uv2[1])
    return np.vstack([
        u * p1[2] - p1[0],
        v * p1[2] - p1[1],
        up * p2[2] - p2[0],
        vp * p2[2] - p2[1],
    ])


def triangulate_one(p1: np.ndarray, p2: np.ndarray, uv1, uv2) -> np.ndarray:
    design = dlt_matrix(p1, p2, uv1, uv2)
    _, _, vt = np.linalg.svd(design)
    x = vt[-1]
    return x[:3] / x[3]


def decompose_essential(essential: np.ndarray) -> dict:
    """
    Hartley & Zisserman: E = U diag(1, 1, 0) Vᵀ, then the four (R, t) pairs.

    The singular values are forced to (1, 1, 0) only when building the
    candidates' translation; the raw SVD is returned unchanged for the workout.
    """
    u, s, vt = np.linalg.svd(essential)
    if np.linalg.det(u) < 0:
        u = u.copy()
        u[:, -1] *= -1
    if np.linalg.det(vt) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r1 = u @ w @ vt
    r2 = u @ w.T @ vt
    if np.linalg.det(r1) < 0:
        r1 = -r1
    if np.linalg.det(r2) < 0:
        r2 = -r2
    t = u[:, 2]
    candidates = [(r1, t), (r1, -t), (r2, t), (r2, -t)]
    return {"U": u, "S": s, "Vt": vt, "candidates": candidates}


def count_in_front(rotation, translation, pts1, pts2, k) -> tuple[int, np.ndarray]:
    """How many triangulated points have positive depth in both cameras."""
    p1 = projection_matrix(k, np.eye(3), np.zeros(3))
    p2 = projection_matrix(k, rotation, translation)
    points = triangulate_dlt(p1, p2, pts1, pts2)
    depth1 = points[:, 2]
    depth2 = (points @ rotation.T + np.asarray(translation).reshape(3))[:, 2]
    return int(np.sum((depth1 > 0) & (depth2 > 0))), points


def _skew(translation: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(translation, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def pose_from_essential(pts1: np.ndarray, pts2: np.ndarray, k: np.ndarray) -> dict:
    """Essential matrix (RANSAC) and the (R, t) that recoverPose selects."""
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    essential, mask = cv2.findEssentialMat(
        pts1, pts2, k, method=cv2.RANSAC, prob=0.999, threshold=1.0,
    )
    if essential is None:
        raise ValueError("findEssentialMat failed. Need at least 5 well-spread correspondences.")
    if essential.shape != (3, 3):
        essential = essential[:3, :3]
    inliers, rotation, translation, pose_mask = cv2.recoverPose(essential, pts1, pts2, k, mask=mask)
    translation = translation.reshape(3)
    return _pack_pose(essential, rotation, translation, pts1, pts2, k, mask, pose_mask, int(inliers))


def pose_from_homography(pts1: np.ndarray, pts2: np.ndarray, k: np.ndarray) -> dict:
    """
    Pose of a planar scene from the homography.

    Correspondences that all lie on one plane do not determine a unique essential
    matrix, so ``findEssentialMat`` can return a pose that satisfies the epipolar
    constraint and is still the wrong camera motion. The homography is the right
    constraint here. ``decomposeHomographyMat`` yields four (R, t, n) solutions;
    the one with every point in front of both cameras is kept. Its essential
    matrix E = [t]× R is what the workout decomposes.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 1.0)
    if homography is None:
        raise ValueError("findHomography failed. The four views need a shared planar object.")
    try:
        n_sol, rotations, translations, normals = cv2.decomposeHomographyMat(homography, k)
    except cv2.error as exc:
        raise ValueError("Could not decompose the homography.") from exc
    best = None
    for index in range(n_sol):
        rotation = rotations[index]
        translation = translations[index].reshape(3)
        n_front, _ = count_in_front(rotation, translation, pts1, pts2, k)
        if best is None or n_front > best[0]:
            best = (n_front, rotation, translation, normals[index].reshape(3))
    _, rotation, translation, normal = best
    essential = _skew(translation) @ rotation
    norm = np.linalg.norm(essential)
    if norm > 0:
        essential = essential / norm
    packed = _pack_pose(essential, rotation, translation, pts1, pts2, k, mask, None, int(np.count_nonzero(mask)))
    packed["H"] = homography
    packed["normal"] = normal
    packed["source"] = "homography"
    return packed


def _pack_pose(essential, rotation, translation, pts1, pts2, k, mask, pose_mask, inliers) -> dict:
    decomposition = decompose_essential(essential)
    scored = []
    for rotation_i, translation_i in decomposition["candidates"]:
        n_front, _ = count_in_front(rotation_i, translation_i, pts1, pts2, k)
        scored.append({"R": rotation_i, "t": np.asarray(translation_i).reshape(3), "in_front": n_front})
    return {
        "E": essential,
        "inlier_mask": None if mask is None else np.asarray(mask).reshape(-1).astype(bool),
        "inliers": inliers,
        "R": rotation,
        "t": np.asarray(translation).reshape(3),
        "decomposition": decomposition,
        "candidates": scored,
        "pose_mask": None if pose_mask is None else np.asarray(pose_mask).reshape(-1).astype(bool),
        "source": "essential",
    }


def solve_pnp(points_3d: np.ndarray, image_points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera pose from 3D–2D correspondences. Prefers the planar IPPE solver."""
    obj = np.asarray(points_3d, dtype=np.float64)
    img = np.asarray(image_points, dtype=np.float64)
    if len(obj) < 4:
        raise ValueError("PnP needs at least 4 correspondences.")
    try:
        n_sol, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            obj, img, k, None, flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        n_sol = 0
        rvecs, tvecs, errors = [], [], []
    if n_sol:
        best = int(np.argmin(np.asarray(errors).reshape(-1)))
        rotation, _ = cv2.Rodrigues(rvecs[best])
        return rotation, tvecs[best].reshape(3)
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("solvePnP failed for this view.")
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, tvec.reshape(3)


def project_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, k: np.ndarray) -> np.ndarray:
    camera = points @ rotation.T + np.asarray(translation).reshape(3)
    pix = camera @ k.T
    return pix[:, :2] / pix[:, 2:3]


def reprojection_rms(points, rotations, translations, uv, k) -> list[float]:
    errors = []
    for view, (rotation, translation) in enumerate(zip(rotations, translations)):
        pred = project_points(points, rotation, translation, k)
        errors.append(float(np.sqrt(np.mean((pred - uv[view]) ** 2))))
    return errors


def bundle_adjust(rotations, translations, points, uv, k) -> dict:
    """
    Least squares over cameras 1..N and every 3D point. Camera 0 stays at the origin.

    The residual is the reprojection error in pixels. Scale is still free (a
    similarity gauge); the caller applies the tape measurement afterwards.
    """
    n_cams = len(rotations)
    n_pts = len(points)
    uv = np.asarray(uv, dtype=np.float64)
    r0 = rotations[0]
    t0 = np.asarray(translations[0], dtype=np.float64).reshape(3)

    def unpack(theta):
        rots = [r0]
        trans = [t0]
        cursor = 0
        for _ in range(1, n_cams):
            rvec = theta[cursor:cursor + 3]
            cursor += 3
            trans.append(theta[cursor:cursor + 3])
            cursor += 3
            rotation, _ = cv2.Rodrigues(rvec)
            rots.append(rotation)
        pts = theta[cursor:].reshape(n_pts, 3)
        return rots, trans, pts

    seed = []
    for index in range(1, n_cams):
        rvec, _ = cv2.Rodrigues(rotations[index])
        seed.append(rvec.reshape(3))
        seed.append(np.asarray(translations[index], dtype=np.float64).reshape(3))
    seed.append(np.asarray(points, dtype=np.float64).reshape(-1))
    theta0 = np.concatenate(seed)

    def residual(theta):
        rots, trans, pts = unpack(theta)
        blocks = []
        for view in range(n_cams):
            pred = project_points(pts, rots[view], trans[view], k)
            blocks.append((pred - uv[view]).ravel())
        return np.concatenate(blocks)

    before = float(np.sqrt(np.mean(residual(theta0) ** 2)))
    solved = least_squares(residual, theta0, method="lm", ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=80)
    rots, trans, pts = unpack(solved.x)
    after = float(np.sqrt(np.mean(solved.fun ** 2)))
    return {
        "rotations": rots,
        "translations": trans,
        "points": pts,
        "rms_before": before,
        "rms_after": after,
        "success": bool(solved.success),
    }


def apply_scale(translations, points, index_a: int, index_b: int, known_length: float) -> tuple[list, np.ndarray, float]:
    """Scale translations and points so the segment (a, b) has the taped length."""
    current = float(np.linalg.norm(points[index_a] - points[index_b]))
    if current < 1e-12:
        raise ValueError("The two scale points triangulated to the same location.")
    scale = float(known_length) / current
    scaled_points = points * scale
    scaled_t = [np.asarray(t, dtype=np.float64).reshape(3) * scale for t in translations]
    return scaled_t, scaled_points, scale


def fit_plane(points: np.ndarray) -> dict:
    """Best-fit plane by SVD, plus 2D coordinates of each point on that plane."""
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    axis_u, axis_v = vt[0], vt[1]
    coords = np.column_stack([(points - centroid) @ axis_u, (points - centroid) @ axis_v])
    return {
        "centroid": centroid,
        "normal": normal,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "uv": coords,
    }


def polygon_report(points_3d: np.ndarray, plane_uv: np.ndarray) -> dict:
    """Edge lengths in 3D and the shoelace area of the plane polygon."""
    n = len(points_3d)
    edges = [float(np.linalg.norm(points_3d[(i + 1) % n] - points_3d[i])) for i in range(n)]
    x, y = plane_uv[:, 0], plane_uv[:, 1]
    area = float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return {"edge_lengths": edges, "perimeter": float(sum(edges)), "area": area, "uv": plane_uv}


def camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Center in the world frame: C = -Rᵀ t."""
    return -rotation.T @ np.asarray(translation, dtype=np.float64).reshape(3)


def _as_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return value


def reconstruct_tracks(uv: np.ndarray, k: np.ndarray, boundary: list[int], edge: tuple[int, int],
                       known_length: float) -> dict:
    """
    Run SfM on known correspondences.

    ``uv`` is (4, N, 2): the same N points in four undistorted views.
    ``boundary`` lists the point indices of the polygon, in order.
    ``edge`` gives the two point indices whose 3D distance is ``known_length``.
    """
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 3 or uv.shape[0] != 4 or uv.shape[2] != 2:
        raise ValueError("uv must have shape (4, N, 2).")
    if len(boundary) < 3:
        raise ValueError("Click at least 3 boundary points, in the same order in every view.")
    k = np.asarray(k, dtype=np.float64)
    # The assignment's object is planar, so the two-view pose comes from the
    # homography. findEssentialMat is still run and kept for the workout: on a
    # plane its (R, t) is not unique, and the homography candidate count shows that.
    pose = pose_from_homography(uv[0], uv[1], k)
    try:
        pose["essential_ransac"] = pose_from_essential(uv[0], uv[1], k)
    except (cv2.error, ValueError):
        pose["essential_ransac"] = None
    p1 = projection_matrix(k, np.eye(3), np.zeros(3))
    p2 = projection_matrix(k, pose["R"], pose["t"])
    points = triangulate_dlt(p1, p2, uv[0], uv[1])

    rotations = [np.eye(3), pose["R"]]
    translations = [np.zeros(3), pose["t"]]
    pnp_records = []
    for view in (2, 3):
        rotation, translation = solve_pnp(points, uv[view], k)
        rotations.append(rotation)
        translations.append(translation)
        pnp_records.append({"view": view, "R": rotation, "t": translation})

    rms_before = reprojection_rms(points, rotations, translations, uv, k)
    adjusted = bundle_adjust(rotations, translations, points, uv, k)
    translations, points, scale = apply_scale(
        adjusted["translations"], adjusted["points"], edge[0], edge[1], known_length,
    )
    rotations = adjusted["rotations"]
    rms_after = reprojection_rms(points, rotations, translations, uv, k)

    boundary_pts = points[list(boundary)]
    plane = fit_plane(boundary_pts)
    polygon = polygon_report(boundary_pts, plane["uv"])

    # Worked example: one boundary point, using the initial (pre-BA) cameras.
    example_index = int(boundary[0])
    design = dlt_matrix(p1, p2, uv[0, example_index], uv[1, example_index])
    _, _, vt = np.linalg.svd(design)
    cameras = []
    for rotation, translation in zip(rotations, translations):
        cameras.append({
            "R": rotation,
            "t": translation,
            "center": camera_center(rotation, translation),
        })
    return {
        "K": k,
        "E": pose["E"],
        "essential_svd": {"U": pose["decomposition"]["U"], "S": pose["decomposition"]["S"], "Vt": pose["decomposition"]["Vt"]},
        "pose_candidates": pose["candidates"],
        "cameras": cameras,
        "points": points,
        "boundary_indices": list(boundary),
        "boundary_points": boundary_pts,
        "plane": {key: plane[key] for key in ("centroid", "normal", "axis_u", "axis_v")},
        "polygon": polygon,
        "scale": scale,
        "known_length": float(known_length),
        "reproj_rms_before": rms_before,
        "reproj_rms_after": rms_after,
        "bundle": {"rms_before": adjusted["rms_before"], "rms_after": adjusted["rms_after"], "success": adjusted["success"]},
        "worked_example": {
            "point_index": example_index,
            "uv": uv[:, example_index, :],
            "dlt_matrix": design,
            "dlt_singular_vector": vt[-1],
            "dlt_point_unit_baseline": triangulate_one(p1, p2, uv[0, example_index], uv[1, example_index]),
            "pnp_view3": pnp_records[0],
            "candidates": pose["candidates"],
        },
        "pose_source": pose.get("source", "homography"),
        "essential_ransac_inliers": None if pose.get("essential_ransac") is None else int(pose["essential_ransac"]["inliers"]),
    }


def sift_matches(image_a: np.ndarray, image_b: np.ndarray, ratio: float = 0.75):
    """Lowe ratio test. Returns keypoints and a list of (index_a, index_b)."""
    sift = cv2.SIFT_create()
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY) if image_a.ndim == 3 else image_a
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY) if image_b.ndim == 3 else image_b
    kp_a, desc_a = sift.detectAndCompute(gray_a, None)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return kp_a or [], kp_b or [], []
    pairs = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
    kept = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            kept.append((best.queryIdx, best.trainIdx))
    return kp_a, kp_b, kept


def _tracks_from_images(images: list[np.ndarray], clicks: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """
    SIFT tracks anchored on view 0, plus one track per clicked boundary point.

    A SIFT track is kept when it is seen in all four views, so every column of
    the returned (4, N, 2) array is a complete correspondence.
    """
    keypoints = []
    match_to = []
    for view in range(1, 4):
        kp0, kp_v, pairs = sift_matches(images[0], images[view])
        # A repeated texture can pass the ratio test and still be the wrong point.
        # On this planar object the right matches agree with one homography.
        if len(pairs) >= 8:
            src = np.float64([kp0[i].pt for i, _ in pairs])
            dst = np.float64([kp_v[j].pt for _, j in pairs])
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
            if mask is not None:
                pairs = [pair for pair, keep in zip(pairs, mask.ravel()) if keep]
        keypoints.append(kp_v)
        match_to.append({src: dst for src, dst in pairs})
        if view == 1:
            base_kp = kp0
    if not match_to[0]:
        base_kp = []
    common = set(match_to[0]) & set(match_to[1]) & set(match_to[2])
    rows = [[], [], [], []]
    for src in sorted(common):
        rows[0].append(base_kp[src].pt)
        for view in range(1, 4):
            rows[view].append(keypoints[view - 1][match_to[view - 1][src]].pt)
    n_sift = len(rows[0])
    for view in range(4):
        for point in np.asarray(clicks[view], dtype=np.float64):
            rows[view].append(point)
    if len(rows[0]) < 5:
        raise ValueError(
            "Need at least 5 correspondences between view 1 and view 2. "
            "Click more boundary points, or use a more textured object so SIFT can match."
        )
    uv = np.asarray(rows, dtype=np.float64)
    boundary = list(range(n_sift, n_sift + len(clicks[0])))
    return uv, boundary


def reconstruct(images: list[np.ndarray], clicks: list[np.ndarray], calibration, known_length: float,
                unit: str = "in", edge: tuple[int, int] = (0, 1), measured_edges: dict | None = None) -> dict:
    """
    Full pipeline from four photographs and clicked boundary polygons.

    ``clicks[view]`` is (M, 2) in the same point order in every view.
    ``edge`` indexes that polygon (not the SIFT points). ``known_length`` is
    in ``unit`` and is converted to meters for the reconstruction.
    """
    if len(images) != 4 or len(clicks) != 4:
        raise ValueError("Structure from motion here uses exactly 4 views.")
    counts = [len(c) for c in clicks]
    if len(set(counts)) != 1 or counts[0] < 3:
        raise ValueError("Click the same number of boundary points (at least 3) in every view, in the same order.")
    if unit not in cv_core.UNIT_TO_METERS:
        raise ValueError(f"Unknown unit '{unit}'.")

    height, width = images[0].shape[:2]
    k, dist = load_intrinsics(calibration, width, height)
    undistorted = []
    click_ud = []
    for image, click in zip(images, clicks):
        if image.shape[1] != width or image.shape[0] != height:
            raise ValueError("All four photos must have the same resolution.")
        undistorted.append(undistort_image(image, k, dist))
        click_ud.append(cv_core.undistort_points(np.asarray(click, dtype=np.float64), k, dist))

    uv, boundary = _tracks_from_images(undistorted, click_ud)
    # ``edge`` is an index into the clicked polygon; boundary[i] is the column.
    i0, i1 = boundary[edge[0]], boundary[edge[1]]
    length_m = cv_core.convert_length(known_length, unit, "m")
    result = reconstruct_tracks(uv, k, boundary, (i0, i1), length_m)
    result["dist_coeffs"] = np.asarray(dist, dtype=np.float64)
    result["unit"] = unit
    result["known_length_input"] = float(known_length)
    result["image_size"] = [int(height), int(width)]
    # Also report lengths in the unit the tape was marked in.
    scale_to_unit = cv_core.convert_length(1.0, "m", unit)
    polygon = result["polygon"]
    result["polygon_unit"] = {
        "unit": unit,
        "edge_lengths": [e * scale_to_unit for e in polygon["edge_lengths"]],
        "perimeter": polygon["perimeter"] * scale_to_unit,
        "area": polygon["area"] * scale_to_unit ** 2,
    }
    if measured_edges:
        result["tape"] = measured_edges
    result["n_sift_tracks"] = int(boundary[0])  # sift columns come first
    result["n_boundary"] = len(boundary)
    return result


def result_to_json(result: dict) -> dict:
    """JSON-ready copy. Drops nothing the report needs; arrays become lists."""
    return _as_json(result)


def format_worked_example(result: dict) -> str:
    """The mathematical workout, printed with the numbers from this reconstruction."""
    e = np.asarray(result["E"])
    svd = result["essential_svd"]
    lines = [
        "Essential matrix E (views 1–2):",
        np.array2string(e, precision=6, suppress_small=True),
        "",
        "SVD E = U diag(S) Vᵀ",
        "S = " + np.array2string(np.asarray(svd["S"]), precision=6),
        "U =",
        np.array2string(np.asarray(svd["U"]), precision=6, suppress_small=True),
        "Vᵀ =",
        np.array2string(np.asarray(svd["Vt"]), precision=6, suppress_small=True),
        "",
        "Four (R, t) candidates and how many points land in front of both cameras:",
    ]
    for i, cand in enumerate(result["pose_candidates"]):
        lines.append(
            f"  candidate {i + 1}: in front = {cand['in_front']}, "
            f"t = {np.array2string(np.asarray(cand['t']), precision=4)}"
        )
        lines.append("  R =")
        lines.append(np.array2string(np.asarray(cand["R"]), precision=4, suppress_small=True))
    example = result["worked_example"]
    lines += [
        "",
        f"DLT triangulation of boundary point index {example['point_index']}.",
        "Design matrix A (4×4), from the two images and P, P':",
        np.array2string(np.asarray(example["dlt_matrix"]), precision=4, suppress_small=True),
        "Smallest right singular vector (homogeneous X) = "
        + np.array2string(np.asarray(example["dlt_singular_vector"]), precision=6),
        "Cartesian point before the tape-measure scale = "
        + np.array2string(np.asarray(example["dlt_point_unit_baseline"]), precision=6),
        "",
        "PnP pose of view 3 (before bundle adjustment), R and t:",
        np.array2string(np.asarray(example["pnp_view3"]["R"]), precision=6, suppress_small=True),
        "t = " + np.array2string(np.asarray(example["pnp_view3"]["t"]), precision=6),
        "",
        f"Bundle adjustment reprojection RMS: {result['bundle']['rms_before']:.6e} px before, "
        f"{result['bundle']['rms_after']:.6e} px after.",
        f"Per-view RMS after scaling (px): {np.array2string(np.asarray(result['reproj_rms_after']), precision=4)}",
        f"Scale applied so the taped edge equals {result['known_length']:.6f} m: s = {result['scale']:.6f}",
        "",
        "The two-view pose is the homography decomposition (the object is planar, so the",
        "essential matrix is not unique). E above is [t]× R for that pose.",
        f"findEssentialMat inliers on the same correspondences: {result.get('essential_ransac_inliers')}.",
        "",
        "Camera centers in meters, relative to view 1 (C = -Rᵀ t):",
    ]
    for i, cam in enumerate(result["cameras"]):
        center = np.asarray(cam["center"])
        lines.append(f"  view {i + 1}: {np.array2string(center, precision=4)} m")
    poly = result["polygon"]
    lines.append("")
    lines.append("Boundary edge lengths (m): " + ", ".join(f"{e:.4f}" for e in poly["edge_lengths"]))
    lines.append(f"Boundary area (m²): {poly['area']:.6f}")
    return "\n".join(lines)
