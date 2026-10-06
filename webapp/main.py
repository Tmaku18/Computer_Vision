"""
webapp/main.py
==============

README — CSc 8830 COMPUTER VISION WEB APPLICATION
-------------------------------------------------

WHAT THIS IS
  One website that hosts every course assignment.
    Module 2 (HW1)  camera calibration, 2D measurement, validation, two-camera theory
    Module 3 (HW2)  image blurring in space and in the Fourier domain
    Module 4 (HW3)  classical human outlines in RGB and thermal images, compared with SAM2

HOW TO RUN LOCALLY (from the repository root)
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  .venv/bin/uvicorn webapp.main:app --reload --port 8000
  open http://localhost:8000

HOW IT IS DEPLOYED
  The Dockerfile at the repository root runs the same command.
  Uploaded photos live in DATA_DIR (default /tmp/cv-course) and disappear
  when the server restarts.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.pages import register_pages
from webapp.routes.module2 import router as module2_router
from webapp.routes.module3 import router as module3_router
from webapp.routes.module4 import router as module4_router
from webapp.storage import WEB_DIR

app = FastAPI(title="CSc 8830 Computer Vision")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")
register_pages(app, templates)
app.include_router(module2_router)
app.include_router(module3_router)
app.include_router(module4_router)
