"""
filtering_core.py
=================

Spatial and Fourier-domain blurring, and the comparison that shows they match.

Convolution in space is multiplication in frequency (the convolution theorem).
Two details decide whether the numbers agree to machine precision:

  linear    zero-pad the image and the kernel, multiply the FFTs, crop back.
            This equals a spatial convolution that also sees zeros outside the image.
  circular  no padding. The DFT treats the image as tiling itself, which equals
            a spatial convolution with wrap-around borders.

A reflected border (the usual choice for a nice-looking blur) is a third,
different boundary condition, so it does NOT match either FFT. The blur is
still a low-pass filter; only the border pixels differ.

Kernel origin is the center pixel, so every kernel size must be odd.
cv2.filter2D computes correlation, not convolution, so matching it requires
flipping the kernel first. The flip is invisible for a symmetric Gaussian and
obvious for a diagonal motion blur.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

BOUNDARIES = ("reflect", "constant", "wrap", "replicate")
MODES = ("linear", "circular")


def require_odd(size: int, name: str) -> int:
    size = int(size)
    if size < 1 or size % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {size}.")
    return size


def box_kernel(size: int) -> np.ndarray:
    """Moving average: every tap equal, sums to 1."""
    size = require_odd(size, "size")
    kernel = np.ones((size, size), dtype=np.float64)
    return kernel / kernel.sum()


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Sampled Gaussian. Its Fourier transform is another Gaussian, so this is a low-pass filter."""
    size = require_odd(size, "size")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def disk_kernel(size: int) -> np.ndarray:
    """Pillbox / defocus blur: equal weight inside a disk."""
    size = require_odd(size, "size")
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = ((xx ** 2 + yy ** 2) <= (size / 2.0) ** 2).astype(np.float64)
    if kernel.sum() == 0:
        kernel[size // 2, size // 2] = 1.0
    return kernel / kernel.sum()


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Straight-line blur, like camera motion. Asymmetric unless the line is horizontal or vertical."""
    length = require_odd(length, "length")
    kernel = np.zeros((length, length), dtype=np.float64)
    # One direction only (center pixel to the right). A full line through the
    # center is unchanged by a 180 degree flip, which would hide the difference
    # between convolution and correlation.
    kernel[length // 2, length // 2:] = 1.0
    center = ((length - 1) / 2.0, (length - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length), flags=cv2.INTER_LINEAR, borderValue=0)
    total = kernel.sum()
    if total <= 0:
        kernel[length // 2, length // 2] = 1.0
        total = 1.0
    return kernel / total


def make_kernel(name: str, size: int = 15, sigma: float = 3.0, angle: float = 0.0) -> np.ndarray:
    name = name.lower()
    if name == "box":
        return box_kernel(size)
    if name == "gaussian":
        return gaussian_kernel(size, sigma)
    if name == "disk":
        return disk_kernel(size)
    if name == "motion":
        return motion_kernel(size, angle)
    raise ValueError(f"Unknown kernel '{name}'. Use box, gaussian, disk, or motion.")


def _as_float(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.astype(np.float64)
    return image.astype(np.float64, copy=False)


def convolve_spatial(image: np.ndarray, kernel: np.ndarray, boundary: str = "reflect") -> np.ndarray:
    """Convolution as an explicit sum of shifted copies of the image.

    output[y, x] = sum_{i,j} kernel[i, j] * image[y + cy - i, x + cx - j]
    with cy, cx the center pixel of the kernel. `boundary` says what image
    values to use when that index falls outside the picture.
    """
    image = _as_float(image)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError("kernel must be 2-D.")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel size must be odd so it has a center pixel.")
    if image.ndim == 3:
        channels = [convolve_spatial(image[:, :, c], kernel, boundary) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)
    if image.ndim != 2:
        raise ValueError("image must be grayscale or color.")
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary must be one of {BOUNDARIES}.")
    pad_mode = {"reflect": "reflect", "constant": "constant", "wrap": "wrap", "replicate": "edge"}[boundary]
    cy, cx = kh // 2, kw // 2
    padded = np.pad(image, ((cy, cy), (cx, cx)), mode=pad_mode)
    # Correlation with the flipped kernel is convolution. The flip is the
    # difference between sliding the kernel and the convolution sum above.
    flipped = kernel[::-1, ::-1]
    out = np.zeros_like(image)
    height, width = image.shape
    for i in range(kh):
        for j in range(kw):
            weight = flipped[i, j]
            if weight == 0.0:
                continue
            out += weight * padded[i:i + height, j:j + width]
    return out


def _centered_kernel(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Place a kernel so its center lies on index (0, 0), wrapping around. This is the circular-convolution alignment."""
    kh, kw = kernel.shape
    cy, cx = kh // 2, kw // 2
    placed = np.zeros(shape, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            placed[(i - cy) % shape[0], (j - cx) % shape[1]] += kernel[i, j]
    return placed


def convolve_fft(image: np.ndarray, kernel: np.ndarray, mode: str = "linear") -> np.ndarray:
    """Convolution by multiplying Fourier transforms.

    linear:   zero-pad to (H + kh - 1, W + kw - 1), multiply, inverse FFT, crop
              back to the original size (the crop keeps the kernel's center).
    circular: no padding; the kernel center is wrapped to index (0, 0).
    """
    image = _as_float(image)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel must be 2-D with odd size.")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}.")
    if image.ndim == 3:
        channels = [convolve_fft(image[:, :, c], kernel, mode) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)
    height, width = image.shape
    kh, kw = kernel.shape
    if mode == "linear":
        shape = (height + kh - 1, width + kw - 1)
        full = np.fft.ifft2(np.fft.fft2(image, shape) * np.fft.fft2(kernel, shape)).real
        cy, cx = kh // 2, kw // 2
        return full[cy:cy + height, cx:cx + width]
    placed = _centered_kernel(kernel, (height, width))
    return np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(placed)).real


def convolve_opencv(image: np.ndarray, kernel: np.ndarray, boundary: str = "reflect") -> np.ndarray:
    """cv2.filter2D is correlation, so the kernel is flipped to make it convolution."""
    border = {
        "reflect": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
        "constant": cv2.BORDER_CONSTANT,
    }[boundary]
    flipped = np.asarray(kernel, dtype=np.float64)[::-1, ::-1]
    return cv2.filter2D(_as_float(image), cv2.CV_64F, flipped, borderType=border)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """How far apart two images are. max_abs near 1e-12 means they are the same up to rounding."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    mse = float(np.mean(diff ** 2))
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    peak = 255.0 if max(float(np.max(a)), float(np.max(b))) > 1.5 else 1.0
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10((peak ** 2) / mse))
    return {"max_abs": max_abs, "rmse": float(np.sqrt(mse)), "mse": mse, "psnr_db": psnr}


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(image, dtype=np.float64)), 0, 255).astype(np.uint8)


def difference_map(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, dict]:
    """Absolute per-pixel difference, averaged over color, drawn with a colormap.

    When the two images match, the map is black and `gain` stays 1 (there is
    nothing to amplify). Otherwise the map is stretched so the largest
    difference is bright.
    """
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    max_abs = float(diff.max()) if diff.size else 0.0
    gain = 1.0 if max_abs < 1e-6 else float(min(255.0 / max_abs, 1e6))
    shown = np.clip(diff * gain, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(shown, cv2.COLORMAP_INFERNO)
    return color, {"max_abs": max_abs, "gain": gain}


def to_gray(image: np.ndarray) -> np.ndarray:
    image = _as_float(image)
    if image.ndim == 2:
        return image
    # OpenCV images are BGR.
    return 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]


def _spectrum_uint8(transformed: np.ndarray) -> np.ndarray:
    magnitude = np.log1p(np.abs(np.fft.fftshift(transformed)))
    magnitude -= magnitude.min()
    peak = magnitude.max()
    if peak > 0:
        magnitude /= peak
    return (magnitude * 255).astype(np.uint8)


def spectra(image: np.ndarray, kernel: np.ndarray, mode: str = "linear") -> dict[str, np.ndarray]:
    """Log-magnitude pictures of F, H, and F·H. Low frequencies sit in the middle."""
    gray = to_gray(image)
    height, width = gray.shape
    kh, kw = kernel.shape
    if mode == "linear":
        shape = (height + kh - 1, width + kw - 1)
        transformed_image = np.fft.fft2(gray, shape)
        transformed_kernel = np.fft.fft2(np.asarray(kernel, dtype=np.float64), shape)
    elif mode == "circular":
        transformed_image = np.fft.fft2(gray)
        transformed_kernel = np.fft.fft2(_centered_kernel(kernel, (height, width)))
    else:
        raise ValueError(f"mode must be one of {MODES}.")
    product = transformed_image * transformed_kernel
    return {
        "image": _spectrum_uint8(transformed_image),
        "kernel": _spectrum_uint8(transformed_kernel),
        "product": _spectrum_uint8(product),
    }


def downscale(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    image = np.asarray(image)
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale >= 1.0:
        return image, 1.0
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def blur_pair(image: np.ndarray, kernel: np.ndarray, mode: str = "linear") -> dict:
    """Run both methods. `mode` selects the boundary condition that makes them equal."""
    boundary = "constant" if mode == "linear" else "wrap"
    started = time.perf_counter()
    spatial = convolve_spatial(image, kernel, boundary=boundary)
    spatial_s = time.perf_counter() - started
    started = time.perf_counter()
    frequency = convolve_fft(image, kernel, mode=mode)
    fft_s = time.perf_counter() - started
    return {
        "spatial": spatial,
        "frequency": frequency,
        "metrics": compare(spatial, frequency),
        "timing_s": {"spatial": spatial_s, "fft": fft_s},
        "boundary": boundary,
        "mode": mode,
    }


def timing_curve(image: np.ndarray, sizes: tuple[int, ...] = (3, 9, 21, 41)) -> list[dict]:
    """Spatial vs FFT time as the Gaussian grows. FFT wins once the kernel is large."""
    rows = []
    for size in sizes:
        kernel = gaussian_kernel(size, max(0.6, 0.18 * size))
        result = blur_pair(image, kernel, mode="linear")
        rows.append({
            "size": int(size),
            "sigma": round(max(0.6, 0.18 * size), 3),
            "spatial_s": result["timing_s"]["spatial"],
            "fft_s": result["timing_s"]["fft"],
            "max_abs": result["metrics"]["max_abs"],
        })
    return rows


def _kernel_cases() -> list[tuple[str, np.ndarray]]:
    return [
        ("box 5x5", box_kernel(5)),
        ("box 15x15", box_kernel(15)),
        ("gaussian 15 sigma 2.5", gaussian_kernel(15, 2.5)),
        ("gaussian 31 sigma 4", gaussian_kernel(31, 4.0)),
        ("disk 11", disk_kernel(11)),
        ("motion 15 at 25 deg", motion_kernel(15, 25.0)),
    ]


def run_experiment(image: np.ndarray, name: str = "image") -> dict:
    """One image, every kernel, both boundary conditions that the theorem predicts, plus reflect (which should NOT match)."""
    gray_or_color = downscale(image, 160)[0]
    rows = []
    for label, kernel in _kernel_cases():
        for mode in MODES:
            result = blur_pair(gray_or_color, kernel, mode=mode)
            rows.append({
                "image": name,
                "kernel": label,
                "mode": mode,
                "boundary": result["boundary"],
                "expected_match": True,
                **result["metrics"],
                "spatial_s": result["timing_s"]["spatial"],
                "fft_s": result["timing_s"]["fft"],
            })
        reflected = convolve_spatial(gray_or_color, kernel, boundary="reflect")
        linear = convolve_fft(gray_or_color, kernel, mode="linear")
        mismatch = compare(reflected, linear)
        rows.append({
            "image": name,
            "kernel": label,
            "mode": "linear",
            "boundary": "reflect",
            "expected_match": False,
            **mismatch,
            "spatial_s": None,
            "fft_s": None,
        })
    curve_image = downscale(image, 128)[0]
    if curve_image.ndim == 3:
        curve_image = to_gray(curve_image)
    return {"image": name, "rows": rows, "timing": timing_curve(curve_image)}


def demo_image(height: int = 360, width: int = 480) -> np.ndarray:
    """A sharp synthetic picture, so blur and its spectrum are obvious without a photo."""
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, 3), dtype=np.float64)
    image[:] = (48, 56, 72)
    image[30:110, 30:190] = (230, 230, 236)
    radius = np.hypot(xx - 330, yy - 190)
    image[radius < 55] = (40, 170, 230)
    image[(radius > 68) & (radius < 84)] = (245, 245, 245)
    image[:, 250:254] = (255, 255, 255)
    image[220:224, :] = (80, 80, 240)
    checker = ((xx // 12 + yy // 12) % 2) == 0
    image[250:340, 40:200][checker[250:340, 40:200]] = (20, 20, 20)
    image[250:340, 40:200][~checker[250:340, 40:200]] = (230, 210, 40)
    return image


def worked_example_1d(signal: tuple[float, ...] = (1, 2, 3, 4), kernel: tuple[float, ...] = (1, 2, 1)) -> dict:
    """Every step of a length-4 signal convolved with a length-3 kernel, in space and by DFT.

    Printed by worked_example.py. The theory page recomputes the same steps.
    """
    f = np.asarray(signal, dtype=np.float64)
    h = np.asarray(kernel, dtype=np.float64)
    if f.ndim != 1 or h.ndim != 1:
        raise ValueError("signal and kernel must be 1-D.")
    if len(h) % 2 == 0:
        raise ValueError("kernel length must be odd.")
    full = np.convolve(f, h)
    center = len(h) // 2
    same = full[center:center + len(f)]
    n = len(f) + len(h) - 1
    transformed_f = np.fft.fft(f, n)
    transformed_h = np.fft.fft(h, n)
    product = transformed_f * transformed_h
    recovered = np.fft.ifft(product).real
    spatial_steps = []
    for n_out in range(n):
        terms = []
        total = 0.0
        for k in range(len(h)):
            i = n_out - k
            if 0 <= i < len(f):
                terms.append({"i": i, "k": k, "f": float(f[i]), "h": float(h[k]), "product": float(f[i] * h[k])})
                total += f[i] * h[k]
        spatial_steps.append({"n": n_out, "terms": terms, "sum": float(total)})

    def _complex(values: np.ndarray) -> list[dict]:
        return [{"re": float(v.real), "im": float(v.imag)} for v in values]

    return {
        "signal": f.tolist(),
        "kernel": h.tolist(),
        "full": full.tolist(),
        "same": same.tolist(),
        "length": n,
        "spatial_steps": spatial_steps,
        "F": _complex(transformed_f),
        "H": _complex(transformed_h),
        "product": _complex(product),
        "inverse": recovered.tolist(),
        "max_abs": float(np.max(np.abs(recovered - full))),
    }


def _self_test() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((30, 28))
    kernel = rng.random((5, 5))
    kernel /= kernel.sum()
    linear = compare(convolve_spatial(image, kernel, "constant"), convolve_fft(image, kernel, "linear"))
    circular = compare(convolve_spatial(image, kernel, "wrap"), convolve_fft(image, kernel, "circular"))
    if linear["max_abs"] > 1e-9 or circular["max_abs"] > 1e-9:
        raise AssertionError(f"convolution theorem failed: {linear} {circular}")
    example = worked_example_1d()
    if example["max_abs"] > 1e-9:
        raise AssertionError("1D example did not match")
    print(f"self-test ok  linear max|{linear['max_abs']:.2e}|  circular max|{circular['max_abs']:.2e}|")


if __name__ == "__main__":
    _self_test()
