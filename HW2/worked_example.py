#!/usr/bin/env python3
"""
worked_example.py
=================

README — SHOW THE CONVOLUTION THEOREM ON A 4-SAMPLE SIGNAL
----------------------------------------------------------

WHAT THIS DOES
  Convolves f = [1, 2, 3, 4] with h = [1, 2, 1] by hand (every product written
  out), then repeats it with the DFT: zero-pad both to length 6, multiply the
  transforms, inverse-transform. The two full results are identical, which is
  the "show by example" evidence for the assignment. Nothing here is handwritten;
  the course accepts typed work.

HOW TO RUN (from the repository root)
  .venv/bin/python HW2/worked_example.py
  .venv/bin/python HW2/worked_example.py --signal 1 4 2 0 --kernel 1 3 1

OUTPUT
  Printed steps, and HW2/results/worked_example.json (the report page reads this).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from filtering_core import worked_example_1d  # noqa: E402

OUT_PATH = SCRIPT_DIR / "results" / "worked_example.json"


def _fmt(value: dict) -> str:
    re, im = value["re"], value["im"]
    if abs(im) < 1e-9:
        return f"{re:9.4f}"
    sign = "+" if im >= 0 else "-"
    return f"{re:9.4f} {sign} {abs(im):.4f}j"


def main() -> None:
    parser = argparse.ArgumentParser(description="1-D convolution theorem, every step printed.")
    parser.add_argument("--signal", type=float, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--kernel", type=float, nargs="+", default=[1, 2, 1])
    args = parser.parse_args()

    example = worked_example_1d(tuple(args.signal), tuple(args.kernel))
    print(f"signal f = {example['signal']}")
    print(f"kernel h = {example['kernel']}   (center tap is index {len(example['kernel']) // 2})")
    print("\nSpatial convolution, zero outside the signal:")
    print("  (f * h)[n] = sum_k f[n - k] h[k]")
    for step in example["spatial_steps"]:
        parts = " + ".join(f"{t['f']:g}*{t['h']:g}" for t in step["terms"]) or "0"
        print(f"  n={step['n']}: {parts} = {step['sum']:.4f}")
    print(f"\nfull result ({example['length']} samples): {['%.4f' % v for v in example['full']]}")
    print(f"same-size crop, aligned on the center tap: {['%.4f' % v for v in example['same']]}")

    print(f"\nDFT length N = {example['length']} (len f + len h - 1), both sequences zero-padded:")
    print(f"  {'k':>3} {'F[k]':>22} {'H[k]':>22} {'F[k] H[k]':>22}")
    for k, (f_k, h_k, p_k) in enumerate(zip(example["F"], example["H"], example["product"])):
        print(f"  {k:3d} {_fmt(f_k):>22} {_fmt(h_k):>22} {_fmt(p_k):>22}")
    print(f"\ninverse DFT: {['%.4f' % v for v in example['inverse']]}")
    print(f"max |inverse DFT - spatial sum| = {example['max_abs']:.3e}")
    print("MATCH" if example["max_abs"] < 1e-8 else "MISMATCH")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(example, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
