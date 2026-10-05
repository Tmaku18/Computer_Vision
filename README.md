# CSc 8830 Computer Vision — Tanaka Makuvaza

Course assignments, with a web application that demonstrates each one.

- **Web app:** _deployment link added here after hosting is set up_ (run it locally with the commands below).
- **Module 2** (folder `HW1`): smartphone camera calibration, real-world 2D measurement, validation at > 2 m, two-camera theory. Guide: [`HW1/RUN_ASSIGNMENT.md`](HW1/RUN_ASSIGNMENT.md).
- **Module 3** (folder `HW2`): image blurring with spatial filters and the matching Fourier-domain multiply. Guide: [`HW2/RUN_ASSIGNMENT.md`](HW2/RUN_ASSIGNMENT.md).

## Run locally

```bash
git clone https://github.com/Tmaku18/Computer_Vision.git
cd Computer_Vision
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn webapp.main:app --reload --port 8000
# open http://localhost:8000
```

Python 3.11 or newer is required. HEIC photos from an iPhone are supported through `pillow-heif`.

## Layout

| Path | Contents |
|---|---|
| `HW1/cv_core.py` | Shared computer-vision core: calibration, undistortion, homography/pinhole measurement, error statistics |
| `HW2/filtering_core.py` | Spatial convolution, FFT convolution, kernels, comparison metrics |
| `HW1/*.py`, `HW1/*.m` | Command-line scripts for each step, in Python (OpenCV) and MATLAB. Each has a README header explaining how to run it. |
| `HW1/camera_calibration.json` | Calibration of the phone camera (29 photos, 2.41 px RMS), used by default in the web app |
| `webapp/` | FastAPI backend. Pages and nav come from `pages.py`; each module has a router in `webapp/routes/` |
| `tests/` | `pytest` tests for the core and the API |
| `Dockerfile` | Container for deployment (listens on `$PORT`) |

## Tests

```bash
.venv/bin/python HW1/cv_core.py
.venv/bin/python -m pytest tests -q
```
