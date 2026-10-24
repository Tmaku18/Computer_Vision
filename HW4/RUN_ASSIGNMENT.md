# Module 6 (HW4) — how to run it

Optical flow on two clips, a check of the Lucas–Kanade equations against points you click, and a four-view reconstruction of one flat object. The same code is on the website under `/module6`.

## Install

From the repository root, with the virtualenv already created for the web app:

```bash
.venv/bin/pip install -r requirements.txt
```

`scipy` is used for bundle adjustment. `imageio-ffmpeg` writes the H.264 flow videos that browsers can play.

## Optical flow

```bash
.venv/bin/python HW4/optical_flow_video.py --video clip.mov --start 0 --duration 30
```

Output lands in `HW4/results/flow_<name>/`: the original, the color-wheel flow, the arrows, the sparse trails, a side-by-side video, a moving-object mask, and `stats.json`. Pass `--width 720` for clips you commit; the hosted page plays those files and does not recompute them.

Use two different clips if you can. One with the phone still, so only the moving objects light up. One while you walk or pan, so the whole frame flows and nearer things move faster. Put the raw files in `HW4/videos/` (gitignored).

## Tracking check

```bash
.venv/bin/python HW4/validate_tracking.py --video clip.mov --time 3.5 --clicks clicks.json --save --width 720
.venv/bin/python HW4/validate_tracking.py --frame1 a.jpg --frame2 b.jpg --clicks clicks.json --save
```

`clicks.json` is `{"frame1": [[x, y], ...], "frame2": [[x, y], ...]}`. The frame-2 clicks are the true locations. `--save` also writes the two frames as JPEGs, scaled the same way as the points, so the hosted tracking page can load the pair without an upload. The web page at `/module6/tracking` is the easier way to click them: it has the zoom loupe. On the hosted site the frames are shrunk in the browser and sent with the request.

The script prints the pixel errors and one fully written Gauss–Newton step (5×5 window, gradients, \(A^\top A\), \(A^\top b\), the step, and the bilinear weights).

## Structure from motion

Photograph one flat textured object from four viewpoints. Same phone, 1× lens, landscape, so `HW1/camera_calibration.json` applies. Tape-measure the width (this sets the scale) and the height (this checks it).

Click the same corners and edge points, in the same order, in all four photos. Save them as `sfm_points.json`:

```json
{
  "boundary": [
    [[100, 200], [400, 210], [390, 500], [110, 490]],
    [[120, 180], [430, 200], [410, 520], [100, 500]],
    [[90, 220], [380, 190], [400, 510], [130, 530]],
    [[140, 210], [450, 230], [420, 540], [150, 500]]
  ]
}
```

Then:

```bash
.venv/bin/python HW4/structure_from_motion.py \
    --images v1.jpg v2.jpg v3.jpg v4.jpg \
    --clicks sfm_points.json \
    --known-length 11 --unit in
.venv/bin/python HW4/sfm_worked_example.py
```

That writes `HW4/results/sfm/` (`sfm.json`, the four views, `sfm_points.json`, the workout, and two figures). Add `"camera_positions"` to the clicks file (one `{view, x_m, y_m, z_m}` per shot, in the object's frame) and the report prints them beside the recovered centers. The web page `/module6/sfm` does the clicking and shows the cameras in 3D. Photos are shrunk in the browser; "Load committed example" shows this folder.

Until the phone clips and photos exist, this builds a labeled synthetic stand-in:

```bash
.venv/bin/python HW4/make_demo_results.py
```

## Tests

```bash
.venv/bin/python -m pytest tests/test_hw4.py -q
```

## Report

Open `/module6/report`, fill in the GitHub, app, and video links, then Print → Save as PDF. The references are on that page and on `/module6/theory`.
