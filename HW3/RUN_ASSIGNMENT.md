# Module 4 — run notes

Code is in `HW3/`. The web pages are under `/module4`.

## What is already in the repo

`HW3/samples/` holds synthetic photos and a box for each one (`HW3/prompts.json`), so the site runs before you add real pictures. The PNGs in `HW3/samples/sam2/` are **not** SAM2. They exist so the overlay has something to draw. Replace them by running SAM2, then `compare_sam2.py`, before you cite numbers.

## 1. Color photos

Take 4–6 photos of a person who agrees to be in them. Vary the background: a plain wall, a cluttered room, outdoors. For at least one pair, do not move the camera: shoot the empty scene, then the same scene with the person. That pair is what background differencing uses.

Put them in `HW3/images/rgb/`. Commit a photo only if you are willing to have it in a public GitHub repo. The site also works from uploads that are never committed.

## 2. Thermal images

Use a public set. Search Roboflow Universe for “thermal person”, or use LLVIP (paired visible / infrared pedestrians). Note the license and cite the set in the report. Put the files in `HW3/images/thermal/`.

## 3. Draw the boxes

Open `/module4/rgb` and `/module4/thermal`. Drag a box around the person and click **Download prompts.json**. Save that file as `HW3/prompts.json` (it can list every photo). The classical method and SAM2 must see the same box. A box is `[x, y, width, height]` in pixels.

Uploads are reduced so the long side is at most 1280 pixels, and the downloaded box is in that smaller image. Point `image` in `prompts.json` at a copy of the photo resized the same way (or at a file that was already that size). `sam2_masks.py` reads the file named there.

## 4. SAM2, once, on your Mac

SAM2 is not part of the hosted app. Use Python 3.12 if PyTorch has no wheel for your default Python.

```bash
python3.12 -m venv .venv-sam2
.venv-sam2/bin/pip install -r HW3/requirements-sam2.txt
mkdir -p HW3/checkpoints
curl -L -o HW3/checkpoints/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
.venv-sam2/bin/python HW3/sam2_masks.py
```

Masks land in `HW3/sam2_masks/<image>.png`. The script uses Apple MPS when PyTorch can see it, otherwise CPU. The checkpoint is about 180 MB.

If the install fails: open Meta's SAM2 demo, segment the person, save the mask, and upload it on the RGB or Thermal page.

## 5. Classical outlines and the comparison

```bash
.venv/bin/python HW3/segment_rgb.py --prompts HW3/prompts.json
.venv/bin/python HW3/segment_thermal.py --prompts HW3/prompts.json
.venv/bin/python HW3/compare_sam2.py
.venv/bin/python HW3/fourier_segmentation.py --check-laplacian
```

`compare_sam2.py` writes `HW3/results/comparison.json`. Reload `/module4/report` to see the table.

## 6. Video and PDF

- RGB page: upload a photo, draw the box, show the OpenCV outline, then the SAM2 overlay and the scores.
- Do the same on the Thermal page.
- Show the Fourier page and the Theory page briefly.
- On `/module4/report`, fill in the links, then Print → Save as PDF.

Screen recording on macOS is Cmd+Shift+5, Record Entire Screen.
