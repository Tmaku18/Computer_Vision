# CSc 8830 — Module 2 run guide

Every step can be done in the **web application** (recommended for the demo) or with the **scripts** in this folder. Both use the same code in `cv_core.py`.

Run all commands from the repository root:

```bash
cd /Users/home/GSU/Computer_Vision/Computer_Vision
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Web application

```bash
.venv/bin/uvicorn webapp.main:app --reload --port 8000
```

Open <http://localhost:8000/module2>. The public deployment is linked in the root `README.md`.

| Page | What it does |
|---|---|
| 1 · Calibrate | Shows the course calibration (29 photos). Upload 20 or more checkerboard photos (JPG/PNG/HEIC) to run a new calibration, which then becomes the active calibration. |
| 2 · Measure | Upload a photo and set a reference on the object's plane (detected checkerboard, or 4 clicked corners of a known rectangle). Click the object's corners to get edge lengths, perimeter, and area. |
| 3 · Validate | Record 20 segments (2 clicks plus the tape-measured length) and view the error statistics and charts. Export the rows as CSV/JSON. |
| Theory | The two-camera derivation, with an interactive calculator. |
| Report | A printable summary of all steps. Use the browser's *Print → Save as PDF*. |

Uploaded photos and new calibrations are stored temporarily on the server. Validation rows are stored in the browser (localStorage), so export them before clearing browser data.

## Step 1 — Camera calibration (smartphone, ≥ 20 photos)

1. Print a checkerboard with 9 × 6 inner corners (10 × 7 squares) and 1 in squares, and tape it flat.
2. Take 20 or more **landscape** photos with the same phone camera (1× lens). Vary the tilt and distance, and put the board in every part of the frame, including the corners.
3. Copy them into `HW1/calibration_images/` (HEIC is accepted), then run:

```bash
.venv/bin/python HW1/calibrate_camera.py
# options: --folder DIR --pattern-cols 9 --pattern-rows 6 --square-size 1 --units in --min-images 20
```

This writes `HW1/camera_calibration.json` (K, distortion, RMS) and `HW1/calibration_report.json` (per-photo errors).

MATLAB alternative: `cd HW1; calibrate_from_folder`, which writes `cameraParams.mat`. It uses the same distortion model (k1, k2, no tangential terms).

## Step 2 — Real-world 2D dimensions

The object and a reference (the checkerboard, or a rectangle of known size such as a Letter sheet) must lie on the **same flat plane**. Pixels are undistorted, and the homography from the reference is inverted to map them to plane coordinates:
$\lambda[u, v, 1]^\top = K[r_1\ r_2\ t][X, Y, 1]^\top$.

```bash
.venv/bin/python HW1/pick_measurement_points.py --image path/to/photo.jpg   # click reference, then object points
.venv/bin/python HW1/measure_object_calibrated.py --json measurement_points.json
.venv/bin/python HW1/measure_object_calibrated.py                            # built-in example
```

MATLAB alternative: `pick_measurement_points`, then `measure_object_calibrated`.

## Step 3 — Validation at more than 2 m (20 measurements)

Shooting protocol:

1. Tape the checkerboard (or a Letter sheet) flat onto a wall or table, next to the objects to be measured.
2. Put the phone on a tripod or support **more than 2 m** away (for example 2.5 m), measured with a tape from the lens to the wall. Point it roughly straight at the wall.
3. Measure 20 different straight lengths with a tape or ruler: box edges, book sides, door-frame parts, tiles, and so on. Write each one down.
4. Take one or more photos (landscape, same camera as in Step 1). Every measured object must be in the plane of the reference.
5. In the web app's **3 · Validate** page, upload each photo, set the reference and camera distance, click the two ends of each segment, type the tape length, and add the row. Then export the CSV.

Scripts:

```bash
# statistics from the web-app export
.venv/bin/python HW1/validate_measurements.py --csv validation_measurements_export.csv

# or fully offline: set image_path / camera_distance_m / reference in HW1/validation_config.json
mkdir -p HW1/validation_images   # put the photo here
.venv/bin/python HW1/pick_validation_points.py --count 20
.venv/bin/python HW1/validate_measurements.py
```

The output, `HW1/validation_report.json`, contains per-measurement errors plus MAE, RMSE, bias, standard deviation, the 95% CI of the bias, MAPE, max error, Pearson r, and the fitted slope. These are computed for both the homography method and the pinhole method $X = Z(u - c_x)/f_x$.

MATLAB alternative: `validate_measurements` (raw-pixel mode) or `validate_measurements("validation_measurements_export.csv")`.

## Theory

`THEORY_two_camera_geometry.md` has the derivation. The web app's Theory page shows the same content, with an interactive check.

## Tests

```bash
.venv/bin/python HW1/cv_core.py       # synthetic self-test
.venv/bin/python -m pytest tests -q   # core + API tests (real-photo tests skip without HW1/calibration_images)
```
