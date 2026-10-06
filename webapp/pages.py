"""Page registry. Each course module lists the pages that appear in its sub-nav."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

MODULES = [
    {
        "id": "module2",
        "prefix": "/module2",
        "nav_label": "Module 2",
        "card_title": "Camera calibration & real-world measurement",
        "blurb": (
            "Calibrate a smartphone camera, measure 2D object dimensions with perspective "
            "projection, validate 20 measurements from more than 2 m, and derive the "
            "two-camera image relationship."
        ),
        "status": "Live",
        "folder": "HW1",
        "steps": "Step 1 · Step 2 · Step 3 · Theory · Report",
        "show_calibration_pill": True,
        "links": [
            ("/module2", "Module 2"),
            ("/module2/calibration", "1 · Calibrate"),
            ("/module2/measure", "2 · Measure"),
            ("/module2/validation", "3 · Validate"),
            ("/module2/theory", "Theory"),
            ("/module2/report", "Report"),
        ],
    },
    {
        "id": "module3",
        "prefix": "/module3",
        "nav_label": "Module 3",
        "card_title": "Blurring: spatial filters and the Fourier domain",
        "blurb": (
            "Blur an image with a spatial filter, repeat the blur by multiplying in the "
            "Fourier domain, and see that the two results match — convolution in space is "
            "multiplication in frequency."
        ),
        "status": "Live",
        "folder": "HW2",
        "steps": "Blur · Theory · Report",
        "show_calibration_pill": False,
        "links": [
            ("/module3", "Module 3"),
            ("/module3/blur", "Blur"),
            ("/module3/theory", "Theory"),
            ("/module3/report", "Report"),
        ],
    },
    {
        "id": "module4",
        "prefix": "/module4",
        "nav_label": "Module 4",
        "card_title": "Human boundaries in RGB and thermal images",
        "blurb": (
            "Outline a person with classical OpenCV in a color photo and in a thermal image, "
            "compare both outlines with SAM2 using the same box, and derive Fourier-domain edges."
        ),
        "status": "Live",
        "folder": "HW3",
        "steps": "RGB · Thermal · Fourier · Theory · Report",
        "show_calibration_pill": False,
        "links": [
            ("/module4", "Module 4"),
            ("/module4/rgb", "RGB"),
            ("/module4/thermal", "Thermal"),
            ("/module4/fourier", "Fourier"),
            ("/module4/theory", "Theory"),
            ("/module4/report", "Report"),
        ],
    },
]

# path without the leading slash -> (template, title)
PAGES = {
    "": ("index.html", "Home"),
    "module2": ("module2/overview.html", "Module 2"),
    "module2/calibration": ("module2/calibration.html", "Step 1 · Calibration"),
    "module2/measure": ("module2/measure.html", "Step 2 · Measurement"),
    "module2/validation": ("module2/validation.html", "Step 3 · Validation"),
    "module2/theory": ("module2/theory.html", "Theory · Two cameras"),
    "module2/report": ("module2/report.html", "Report"),
    "module3": ("module3/overview.html", "Module 3"),
    "module3/blur": ("module3/blur.html", "Blurring"),
    "module3/theory": ("module3/theory.html", "Theory · Convolution theorem"),
    "module3/report": ("module3/report.html", "Report"),
    "module4": ("module4/overview.html", "Module 4"),
    "module4/rgb": ("module4/rgb.html", "RGB boundaries"),
    "module4/thermal": ("module4/thermal.html", "Thermal boundaries"),
    "module4/fourier": ("module4/fourier.html", "Fourier edges"),
    "module4/theory": ("module4/theory.html", "Theory · Fourier segmentation"),
    "module4/report": ("module4/report.html", "Report"),
}


def active_module(path: str) -> dict | None:
    for module in MODULES:
        if path == module["prefix"] or path.startswith(module["prefix"] + "/"):
            return module
    return None


def register_pages(app: FastAPI, templates: Jinja2Templates) -> None:
    def make_handler(path: str):
        template, title = PAGES[path]

        def handler(request: Request):
            url_path = "/" + path if path else "/"
            return templates.TemplateResponse(
                request,
                template,
                {
                    "title": title,
                    "path": url_path,
                    "modules": MODULES,
                    "module": active_module(url_path),
                },
            )

        return handler

    for path in PAGES:
        app.add_api_route(
            "/" + path,
            make_handler(path),
            methods=["GET"],
            response_class=HTMLResponse,
            include_in_schema=False,
        )
