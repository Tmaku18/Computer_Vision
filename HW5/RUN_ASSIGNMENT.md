# Assignment 7 — run notes

Code is in `HW5/`. The web pages are under `/module7`.

The built-in sample is a synthetic sideways slide, so the site runs before you shoot anything. Replace it with your own pairs before you cite sizes in the report.

## 1. The rail

Tape a meter stick to a table edge or a shelf. Mark stops at 0 and 20 cm. Use the same phone, 1× lens, landscape, as Module 2. Press the phone against the stick, shoot at the left stop, slide to the right stop without turning it.

## 2. Stereo calibration

Prop up the same 9×6 checkerboard. Shoot at the left stop, slide, shoot at the right stop. Move the board and repeat, 15 to 20 times. Put matching names in `HW5/calibration/left/` and `HW5/calibration/right/`.

```bash
.venv/bin/python HW5/stereo_calibrate.py --baseline 20 --unit cm
```

`|T|` in `HW5/stereo_calibration.json` should be close to 20 cm.

## 3. Object pairs

At 1 to 2 m, shoot a left/right pair of a rectangle (a book or a box face), a circle (a plate or a clock), and one other polygon. Tape-measure each size and the camera-to-object distance.

On `/module7/stereo`, upload the pair, click the corners in order, type the tape values, and measure. Or:

```bash
.venv/bin/python HW5/measure_stereo.py --left L.jpg --right R.jpg --clicks clicks.json --shape rectangle --baseline 20 --unit cm
.venv/bin/python HW5/validate_stereo.py --json HW5/results/validation.json
```

`clicks.json` is `{"left": [[u, v], ...], "right": [[u, v], ...]}`. Omit `right` to let the script match along the row.

## 4. Handwrite the uncalibrated derivation

Typed work is not accepted for this part. Photos of the paper are not accepted either.

```bash
.venv/bin/python HW5/uncalibrated_numbers.py --left L.jpg --right R.jpg --baseline 20 --unit cm
```

Copy those numbers into a handwritten derivation: the epipolar constraint and the eight-point matrix, the projective cameras, why length is not fixed yet, then \(E = K^{\top} F K\), the scale from the baseline, and the triangulation. Scan it with a scanner app (Notes, or Adobe Scan).

A pure slide often makes the focal length unrecoverable from \(F\). When the script says so, use the Module 2 focal length for that step and say why.

## 5. Video calibration

Settings → Camera → Record Video → 1080p at 30 fps. Turn off Enhanced Stabilization or Action mode if it is shown. Film the checkerboard for about 40 seconds.

```bash
.venv/bin/python HW5/calibrate_video.py --video checkerboard.mov --square-size 1 --unit in
```

## 6. The room

With the same video settings, pause the phone on the rail facing the long wall, slide 20 cm, and pause again. Repeat facing the short wall. Tape or laser-measure the room's length, width, and height.

```bash
.venv/bin/python HW5/room_dimensions.py --video long_wall.mov --baseline 20 --unit cm --name length --clicks corners.json
.venv/bin/python HW5/room_dimensions.py --video short_wall.mov --baseline 20 --unit cm --name width
```

The two stills can also be uploaded on `/module7/room`. The synthetic box on that page is only a check that the plane fit recovers 4 m × 3 m × 2.5 m.

## 7. Video and PDF

Record the screen (Cmd+Shift+5): each shape on the stereo page, the room page against your tape measurements, and the theory page.

Open `/module7/report`, fill in the links, Print → Save as PDF. In Preview, drag the scanned derivation into the thumbnail sidebar so it is the last part of the file. Upload the PDF and the video. The repository is https://github.com/Tmaku18/Computer_Vision.

Commit a photo only if you are willing to have it in a public repository. The pages also work from uploads that are never committed.
