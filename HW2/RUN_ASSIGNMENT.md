# CSc 8830 — Module 3 run guide (folder HW2)

Blur an image with a spatial filter and with the Fourier-domain equivalent, and show that the results are the same. The web app and the scripts share `HW2/filtering_core.py`.

Run from the repository root:

```bash
cd /Users/home/GSU/Computer_Vision/Computer_Vision
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn webapp.main:app --reload --port 8000
```

Open <http://localhost:8000/module3>.

| Page | What it does |
|---|---|
| Blur | Upload a photo (or use the built-in test image). Pick a kernel. See the spatial blur, the FFT blur, the difference, and the spectra. |
| Theory | The convolution theorem, including why zero-padding matters, plus an editable 1-D example. |
| Report | Printable summary. Use the browser's Print → Save as PDF. |

## Scripts

```bash
# one image, both methods (omit --image to use the built-in picture)
.venv/bin/python HW2/blur_image.py --image photo.jpg --kernel gaussian --size 21 --sigma 4
.venv/bin/python HW2/blur_image.py --kernel motion --size 15 --angle 25 --mode circular

# the experiment the report quotes: every kernel, both boundary conditions
.venv/bin/python HW2/verify_convolution_theorem.py
# photos dropped in HW2/images/ are included automatically

# the typed 1-D example, every product printed
.venv/bin/python HW2/worked_example.py

.venv/bin/python HW2/filtering_core.py    # quick self-test
.venv/bin/python -m pytest tests/test_hw2.py -q
```

`linear` mode zero-pads and matches a spatial blur that sees black outside the image. `circular` mode wraps around and matches a spatial blur with wrap-around borders. Reflected borders look nicer and are still a blur, but they are not what the DFT computes, so those rows in the experiment are the ones that do not match. That difference is the evidence, not a bug.

## What you do by hand

Nothing on paper. The assignment allows typed work. `worked_example.py` and the Theory page write out every product for \(f = [1, 2, 3, 4]\) and \(h = [1, 2, 1]\). Read that page before recording, so you can say what each step is.

## Demo video and PDF

1. Open the Blur page. Upload a sharp photo (text or a fence works well). Switch kernels. Point at the difference map and the max-difference number, which should be about \(10^{-12}\).
2. Show the spectra: the product keeps the bright center and loses the edges.
3. Open Theory and change the 1-D signal so the table recomputes.
4. Open Report, fill in the web-app and video links, and Print → Save as PDF.
5. Upload the PDF and the video to Google Classroom. The repository link is <https://github.com/Tmaku18/Computer_Vision>.
