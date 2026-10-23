#!/usr/bin/env python3
"""
validate_stereo.py
==================

README — ERRORS AGAINST TAPE MEASUREMENTS
-----------------------------------------

WHAT THIS DOES
  Compares estimated lengths (widths, heights, diameters, edges, depths)
  with the numbers you measured with a tape. Prints MAE, RMSE, and the
  mean absolute percent error.

HOW TO RUN (from the repository root)
  .venv/bin/python HW5/validate_stereo.py --json HW5/results/validation.json

JSON
  {"rows": [{"name": "book width", "estimated_m": 0.21, "tape_m": 0.20}, ...]}

  Lengths must already be in meters. The web page writes this file when you
  type the tape values and click Measure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "HW5") not in sys.path:
    sys.path.insert(0, str(ROOT / "HW5"))


def error_stats(estimated: np.ndarray, tape: np.ndarray) -> dict:
    estimated = np.asarray(estimated, dtype=np.float64)
    tape = np.asarray(tape, dtype=np.float64)
    if len(estimated) == 0:
        raise ValueError("No rows to compare.")
    err = estimated - tape
    return {
        "n": int(len(err)),
        "mae_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "mape_percent": float(np.mean(np.abs(err) / np.maximum(np.abs(tape), 1e-9)) * 100),
        "max_abs_m": float(np.max(np.abs(err))),
        "bias_m": float(np.mean(err)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stereo measurement errors against a tape.")
    parser.add_argument("--json", type=Path, default=Path("HW5/results/validation.json"))
    args = parser.parse_args()
    data = json.loads(args.json.read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) else data
    estimated = np.array([row["estimated_m"] for row in rows], dtype=np.float64)
    tape = np.array([row["tape_m"] for row in rows], dtype=np.float64)
    stats = error_stats(estimated, tape)
    print(f"{stats['n']} comparisons")
    print(f"MAE  {stats['mae_m'] * 100:.2f} cm")
    print(f"RMSE {stats['rmse_m'] * 100:.2f} cm")
    print(f"MAPE {stats['mape_percent']:.2f} %")
    print(f"bias {stats['bias_m'] * 100:.2f} cm (estimate minus tape)")
    for row, err in zip(rows, estimated - tape):
        print(f"  {row.get('name', '?'):20s}  est {row['estimated_m']:.4f}  tape {row['tape_m']:.4f}  err {err * 100:+.2f} cm")


if __name__ == "__main__":
    main()
