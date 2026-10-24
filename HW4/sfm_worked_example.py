#!/usr/bin/env python3
"""
sfm_worked_example.py
=====================

README — PRINT THE STRUCTURE-FROM-MOTION WORKOUT
------------------------------------------------

WHAT THIS DOES
  Prints the algebra for a reconstruction that structure_from_motion.py already
  saved: the essential matrix and its SVD, the four (R, t) candidates with the
  cheirality count, the DLT matrix of one boundary point, one PnP pose, and the
  bundle-adjustment error before and after.

HOW TO RUN (from the repository root)
  .venv/bin/python HW4/sfm_worked_example.py
  .venv/bin/python HW4/sfm_worked_example.py --result HW4/results/sfm/sfm.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sfm_core import format_worked_example  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the SfM workout from sfm.json.")
    parser.add_argument("--result", default=str(SCRIPT_DIR / "results" / "sfm" / "sfm.json"))
    args = parser.parse_args()
    path = Path(args.result)
    if not path.is_file():
        raise SystemExit(f"No reconstruction at {path}. Run structure_from_motion.py first.")
    result = json.loads(path.read_text(encoding="utf-8"))
    # format_worked_example accepts lists; make the one place it indexes like an array safe.
    result["reproj_rms_after"] = np.asarray(result["reproj_rms_after"])
    print(format_worked_example(result))


if __name__ == "__main__":
    main()
