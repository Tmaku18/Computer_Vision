Put 20 or more smartphone photos of the checkerboard here for Step 1 calibration.
(The raw photos are not committed to git because they total about 100 MB.
The calibration result they produced is HW1/camera_calibration.json.)

Supported formats: .jpg, .jpeg, .png, .heic, .heif

Tips:
  - Same phone camera and lens for every photo, always in landscape.
  - Vary the tilt and distance, and cover every part of the frame, including the corners.
  - Keep the whole checkerboard in frame (default: 9 x 6 inner corners, 1 inch squares).

Then run, from the repository root:

  Python (OpenCV):
    .venv/bin/python HW1/calibrate_camera.py

  MATLAB (from HW1/):
    calibrate_from_folder

Or upload the photos on the web app's "1 · Calibrate" page.
