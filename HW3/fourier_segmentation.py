#!/usr/bin/env python3
"""
fourier_segmentation.py
=======================

README — EDGES AND REGIONS IN THE FOURIER DOMAIN
-------------------------------------------------

Edges are high spatial frequencies. This script builds those edges by
multiplying the image's DFT by a high-pass filter, and it builds regions
by smoothing (a low-pass) and thresholding, or by Gabor energy.

The Laplacian check is exact, not visual: the frequency-domain multiplier
-4 + 2 cos(2πu/N) + 2 cos(2πv/M) is the DFT of the 3x3 discrete Laplacian,
so it matches a spatial convolution with wrap-around borders. At low
frequency that multiplier is the sampled form of -4π²(u² + v²).

HOW TO RUN (from the repository root)
  .venv/bin/python HW3/fourier_segmentation.py --image photo.jpg --filter gaussian --cutoff 24
  .venv/bin/python HW3/fourier_segmentation.py --check-laplacian
  .venv/bin/python HW3/fourier_segmentation.py --image HW3/samples/rgb_wall.png --filter dog --out HW3/results/fourier

FILTERS
  ideal         hard cutoff. Sharp, and it rings (Gibbs).
  butterworth   smooth cutoff of a chosen order.
  gaussian      smooth cutoff, no ringing.
  gradient      derivative theorem: F{df/dx} = j 2π u F.
  laplacian     discrete Laplacian, equal to the spatial kernel.
  dog           difference of Gaussians, then zero crossings.
  gabor         filter-bank energy, then Otsu, for textured regions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

FILTERS = ("ideal", "butterworth", "gaussian", "gradient", "laplacian", "dog", "gabor")

_LAPLACIAN_KERNEL = np.array(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def _gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float64)


def frequency_grid(shape: tuple[int, int], shift: bool = False):
    """u, v in cycles per pixel. Unshifted grids line up with np.fft.fft2."""
    rows, cols = shape
    if shift:
        u = np.fft.fftshift(np.fft.fftfreq(cols))
        v = np.fft.fftshift(np.fft.fftfreq(rows))
    else:
        u = np.fft.fftfreq(cols)
        v = np.fft.fftfreq(rows)
    uu, vv = np.meshgrid(u, v)
    return uu, vv


def distance_from_dc(shape: tuple[int, int]) -> np.ndarray:
    """Pixel distance from DC on an fftshifted grid."""
    rows, cols = shape
    vv, uu = np.ogrid[:rows, :cols]
    cy, cx = rows // 2, cols // 2
    return np.sqrt((uu - cx) ** 2 + (vv - cy) ** 2)


def ideal_highpass(shape, cutoff: float) -> np.ndarray:
    d = distance_from_dc(shape)
    return (d > float(cutoff)).astype(np.float64)


def butterworth_highpass(shape, cutoff: float, order: int = 2) -> np.ndarray:
    d = distance_from_dc(shape)
    cutoff = max(float(cutoff), 1e-6)
    order = max(1, int(order))
    # H = 1 / (1 + (D0/D)^(2n)). DC is 0 (a pure high-pass).
    with np.errstate(divide="ignore"):
        ratio = np.where(d == 0, np.inf, cutoff / d)
    return 1.0 / (1.0 + ratio ** (2 * order))


def gaussian_highpass(shape, cutoff: float) -> np.ndarray:
    d = distance_from_dc(shape)
    cutoff = max(float(cutoff), 1e-6)
    return 1.0 - np.exp(-(d ** 2) / (2.0 * cutoff ** 2))


def discrete_laplacian_transfer(shape: tuple[int, int]) -> np.ndarray:
    """DFT multiplier of the 3x3 Laplacian [[0,1,0],[1,-4,1],[0,1,0]].

    H(u, v) = -4 + 2 cos(2πu) + 2 cos(2πv), with u, v in cycles per pixel.
    The Taylor expansion cos θ ≈ 1 - θ²/2 turns this into -4π²(u² + v²)
    when the frequency is low.
    """
    uu, vv = frequency_grid(shape, shift=False)
    return -4.0 + 2.0 * np.cos(2 * np.pi * uu) + 2.0 * np.cos(2 * np.pi * vv)


def continuous_laplacian_transfer(shape: tuple[int, int]) -> np.ndarray:
    """The continuous formula -4π²(u² + v²), fftshifted for display."""
    uu, vv = frequency_grid(shape, shift=True)
    return -4.0 * np.pi ** 2 * (uu ** 2 + vv ** 2)


def spatial_laplacian(gray: np.ndarray) -> np.ndarray:
    """Circular convolution with the 3x3 discrete Laplacian.

    The kernel is centered, and the image wraps at the border, which is the
    boundary condition the DFT uses. OpenCV's filter2D in this build rejects
    wrap borders, so the sum is the kernel weights times rolled copies.
    """
    values = np.asarray(gray, np.float64)
    kernel = _LAPLACIAN_KERNEL
    center_y, center_x = 1, 1
    out = np.zeros_like(values)
    for dy in range(kernel.shape[0]):
        for dx in range(kernel.shape[1]):
            weight = kernel[dy, dx]
            if weight == 0:
                continue
            out += weight * np.roll(np.roll(values, center_y - dy, axis=0), center_x - dx, axis=1)
    return out


def frequency_laplacian(gray: np.ndarray) -> np.ndarray:
    """Same Laplacian, applied as a multiply in the Fourier domain."""
    values = np.asarray(gray, np.float64)
    transfer = discrete_laplacian_transfer(values.shape)
    return np.fft.ifft2(np.fft.fft2(values) * transfer).real


def frequency_gradient(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derivative theorem: d/dx ↔ multiply by j 2π u."""
    values = np.asarray(gray, np.float64)
    spectrum = np.fft.fft2(values)
    uu, vv = frequency_grid(values.shape, shift=False)
    dx = np.fft.ifft2(1j * 2 * np.pi * uu * spectrum).real
    dy = np.fft.ifft2(1j * 2 * np.pi * vv * spectrum).real
    magnitude = np.hypot(dx, dy)
    return dx, dy, magnitude


def zero_crossings(field: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """Pixels where the field changes sign against a neighbor.

    `strength` drops crossings where the jump is smaller than that value,
    which is how a LoG ignores flat noise.
    """
    values = np.asarray(field, np.float64)
    crosses = np.zeros(values.shape, np.uint8)
    horizontal = values[:, :-1] * values[:, 1:] < 0
    vertical = values[:-1, :] * values[1:, :] < 0
    if strength > 0:
        horizontal &= np.abs(values[:, :-1] - values[:, 1:]) >= strength
        vertical &= np.abs(values[:-1, :] - values[1:, :]) >= strength
    crosses[:, :-1][horizontal] = 255
    crosses[:-1, :][vertical] = 255
    return crosses


def _dog_field(gray: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    narrow = max(float(sigma1), 0.4)
    wide = max(float(sigma2), narrow + 0.2)
    fine = cv2.GaussianBlur(gray, (0, 0), narrow)
    coarse = cv2.GaussianBlur(gray, (0, 0), wide)
    return fine - coarse


def gabor_energy(
    gray: np.ndarray,
    orientations: int = 8,
    sigmas: tuple[float, ...] = (2.0, 4.0),
    wavelengths: tuple[float, ...] = (4.0, 8.0),
) -> np.ndarray:
    """Sum of squared Gabor responses. High energy marks textured regions."""
    values = np.asarray(gray, np.float64)
    energy = np.zeros(values.shape, np.float64)
    ksize = 31
    for theta in np.linspace(0, np.pi, int(orientations), endpoint=False):
        for sigma in sigmas:
            for wavelength in wavelengths:
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), float(sigma), float(theta), float(wavelength), 0.5, 0, ktype=cv2.CV_64F,
                )
                response = cv2.filter2D(values, cv2.CV_64F, kernel)
                energy += response ** 2
    return np.sqrt(energy)


def _as_uint8(field: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(field, np.float64))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, np.uint8)
    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((values - lo) / (hi - lo), 0, 1)
    return (scaled * 255).round().astype(np.uint8)


def _otsu(field: np.ndarray) -> np.ndarray:
    image = _as_uint8(field)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _spectrum_image(gray: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(np.asarray(gray, np.float64)))
    return _as_uint8(np.log1p(np.abs(spectrum)))


def apply_filter(image: np.ndarray, kind: str, cutoff: float = 24, order: int = 2,
                 sigma1: float = 1.2, sigma2: float = 2.4) -> dict:
    """Run one frequency-domain filter.

    Returns uint8 images: the filter H(u, v), the log spectrum, the edge map,
    and a region map (low-pass + Otsu, DoG sign, or Gabor energy + Otsu).
    """
    if kind not in FILTERS:
        raise ValueError(f"Unknown filter '{kind}'. Use one of {FILTERS}.")
    gray = _gray(image)
    rows, cols = gray.shape
    spectrum = np.fft.fft2(gray)
    shifted = np.fft.fftshift(spectrum)

    if kind == "ideal":
        transfer = ideal_highpass((rows, cols), cutoff)
        filtered = np.fft.ifft2(np.fft.ifftshift(shifted * transfer)).real
        edges = _as_uint8(filtered)
        lowpass = np.fft.ifft2(np.fft.ifftshift(shifted * (1.0 - transfer))).real
        regions = _otsu(lowpass)
    elif kind == "butterworth":
        transfer = butterworth_highpass((rows, cols), cutoff, order)
        filtered = np.fft.ifft2(np.fft.ifftshift(shifted * transfer)).real
        edges = _as_uint8(filtered)
        lowpass = np.fft.ifft2(np.fft.ifftshift(shifted * (1.0 - transfer))).real
        regions = _otsu(lowpass)
    elif kind == "gaussian":
        transfer = gaussian_highpass((rows, cols), cutoff)
        filtered = np.fft.ifft2(np.fft.ifftshift(shifted * transfer)).real
        edges = _as_uint8(filtered)
        lowpass = np.fft.ifft2(np.fft.ifftshift(shifted * (1.0 - transfer))).real
        regions = _otsu(lowpass)
    elif kind == "gradient":
        _dx, _dy, magnitude = frequency_gradient(gray)
        transfer = np.fft.fftshift(np.hypot(*frequency_grid((rows, cols), shift=False)))
        edges = _as_uint8(magnitude)
        regions = _otsu(cv2.GaussianBlur(gray, (0, 0), 2.0))
        filtered = magnitude
    elif kind == "laplacian":
        transfer = np.fft.fftshift(discrete_laplacian_transfer((rows, cols)))
        filtered = frequency_laplacian(gray)
        jump = float(np.std(filtered)) * 0.5
        edges = zero_crossings(filtered, strength=jump)
        regions = _otsu(cv2.GaussianBlur(gray, (0, 0), 2.0))
    elif kind == "dog":
        filtered = _dog_field(gray, sigma1, sigma2)
        # DoG is a band-pass. Show the two Gaussian transfers' difference.
        d = distance_from_dc((rows, cols))
        g1 = np.exp(-(d ** 2) / (2.0 * max(sigma1, 0.4) ** 2))
        g2 = np.exp(-(d ** 2) / (2.0 * max(sigma2, sigma1 + 0.2) ** 2))
        transfer = np.abs(g1 - g2)
        jump = float(np.std(filtered)) * 0.4
        edges = zero_crossings(filtered, strength=jump)
        regions = np.where(filtered > 0, 255, 0).astype(np.uint8)
    else:
        filtered = gabor_energy(gray)
        transfer = _gabor_preview((rows, cols))
        edges = _as_uint8(filtered)
        regions = _otsu(filtered)

    return {
        "kind": kind,
        "filter": _as_uint8(transfer),
        "spectrum": _spectrum_image(gray),
        "edges": edges,
        "regions": regions,
        "response": filtered,
    }


def _gabor_preview(shape: tuple[int, int]) -> np.ndarray:
    """One oriented Gabor kernel, padded and fftshifted, as a picture of H."""
    kernel = cv2.getGaborKernel((31, 31), 4.0, np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_64F)
    canvas = np.zeros(shape, np.float64)
    kh, kw = kernel.shape
    canvas[:kh, :kw] = kernel
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(canvas)))
    return spectrum


def worked_derivative(signal: tuple[float, ...] = (1.0, 2.0, 1.0, 0.0)) -> dict:
    """1-D derivative theorem, every product printed by the caller.

    Multiplying the DFT by j 2π u and transforming back is the derivative of
    the trigonometric interpolant of the samples. A one-step finite difference
    is a different operator, so the two columns are not expected to match.
    """
    values = np.asarray(signal, np.float64).reshape(-1)
    if values.size < 2 or values.size > 12:
        raise ValueError("Use 2 to 12 samples.")
    spectrum = np.fft.fft(values)
    freq = np.fft.fftfreq(values.size)
    multiplier = 1j * 2 * np.pi * freq
    product = multiplier * spectrum
    derivative = np.fft.ifft(product).real
    finite = np.roll(values, -1) - values
    return {
        "signal": values.tolist(),
        "frequency": freq.tolist(),
        "spectrum_real": spectrum.real.tolist(),
        "spectrum_imag": spectrum.imag.tolist(),
        "multiplier_real": multiplier.real.tolist(),
        "multiplier_imag": multiplier.imag.tolist(),
        "derivative": derivative.tolist(),
        "finite_difference": finite.tolist(),
    }


def laplacian_error(gray: np.ndarray) -> float:
    spatial = spatial_laplacian(gray)
    freq = frequency_laplacian(gray)
    return float(np.max(np.abs(spatial - freq)))


def _read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read {path}")
    return image


def _write_set(out: Path, stem: str, result: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for key in ("filter", "spectrum", "edges", "regions"):
        path = out / f"{stem}_{key}.png"
        if not cv2.imwrite(str(path), result[key]):
            raise SystemExit(f"Could not write {path}")
        print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fourier-domain edges and regions.")
    parser.add_argument("--image", type=Path, help="Input image.")
    parser.add_argument("--filter", default="gaussian", choices=FILTERS)
    parser.add_argument("--cutoff", type=float, default=24, help="High-pass cutoff in pixels from DC.")
    parser.add_argument("--order", type=int, default=2, help="Butterworth order.")
    parser.add_argument("--sigma1", type=float, default=1.2, help="Narrow Gaussian of the DoG.")
    parser.add_argument("--sigma2", type=float, default=2.4, help="Wide Gaussian of the DoG.")
    parser.add_argument("--out", type=Path, default=Path("HW3/results/fourier"))
    parser.add_argument("--check-laplacian", action="store_true",
                        help="Print the max gap between the spatial and Fourier Laplacians.")
    args = parser.parse_args(argv)

    if args.check_laplacian:
        source = _gray(_read(args.image)) if args.image else np.random.default_rng(0).normal(size=(48, 64))
        gap = laplacian_error(source)
        print(f"max |spatial - frequency| Laplacian = {gap:.3e}")
        if gap > 1e-6:
            raise SystemExit("Laplacian mismatch is larger than 1e-6.")
        if args.image is None:
            return

    if args.image is None:
        raise SystemExit("Pass --image, or --check-laplacian to run the numeric check only.")
    image = _read(args.image)
    result = apply_filter(
        image, args.filter, cutoff=args.cutoff, order=args.order, sigma1=args.sigma1, sigma2=args.sigma2,
    )
    _write_set(args.out, args.image.stem + "_" + args.filter, result)


if __name__ == "__main__":
    main(sys.argv[1:])
