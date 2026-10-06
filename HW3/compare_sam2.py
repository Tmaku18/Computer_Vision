#!/usr/bin/env python3
"""
compare_sam2.py
===============

README — COMPARE THE CLASSICAL MASK WITH THE SAM2 MASK
-------------------------------------------------------

Runs every entry in prompts.json and, where a SAM2 mask exists, records
IoU, Dice, precision, recall, boundary F (2 px), Hausdorff, and the mean
boundary distance. Writes HW3/results/comparison.json and prints mean IoU
and mean boundary F for RGB and for thermal.

SAM2 masks are read from HW3/sam2_masks/<name>.png if you have run
sam2_masks.py, otherwise from HW3/samples/sam2/<name>.png. The sample
files shipped with the repo are synthetic stand-ins, not SAM2 output.
Replace them by running sam2_masks.py before you quote numbers in the report.

HOW TO RUN (from the repository root)
  .venv/bin/python HW3/compare_sam2.py
  .venv/bin/python HW3/compare_sam2.py --prompts HW3/prompts.json --out HW3/results/comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import segment_core


def _mean(rows: list[dict], key: str):
    values = [row["metrics"][key] for row in rows if row.get("metrics") and row["metrics"].get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize(records: list[dict]) -> dict:
    summary = {}
    for kind in ("rgb", "thermal"):
        rows = [row for row in records if row["kind"] == kind and row.get("metrics")]
        summary[kind] = {
            "n": len(rows),
            "mean_iou": _mean(rows, "iou"),
            "mean_dice": _mean(rows, "dice"),
            "mean_boundary_f": _mean(rows, "boundary_f"),
            "mean_hausdorff_px": _mean(rows, "hausdorff_px"),
        }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch classical-vs-SAM2 comparison.")
    parser.add_argument("--prompts", type=Path, default=Path("HW3/prompts.json"))
    parser.add_argument("--out", type=Path, default=Path("HW3/results/comparison.json"))
    args = parser.parse_args(argv)
    root = segment_core.repo_root()
    prompts = segment_core.load_prompts(args.prompts if args.prompts.is_absolute() else root / args.prompts)
    records = []
    for prompt in prompts:
        result = segment_core.run_prompt(prompt, root)
        row = {
            "image": prompt["image"],
            "kind": result["kind"],
            "method": result["method"],
            "box": result["box"],
            "sam2_mask": result["sam2_mask"],
            "metrics": result["metrics"],
            "note": prompt.get("note", ""),
        }
        records.append(row)
        if result["metrics"] is None:
            print(f"{prompt['image']}: no SAM2 mask yet")
        else:
            metrics = result["metrics"]
            print(
                f"{prompt['image']}: IoU {metrics['iou']:.3f}  "
                f"boundary F {metrics['boundary_f']:.3f}  "
                f"Hausdorff {metrics['hausdorff_px']:.2f} px"
            )
    report = {
        "note": (
            "Masks in HW3/samples/sam2/ are eroded ground truth so the demo has "
            "an overlay before SAM2 is installed. Masks in HW3/sam2_masks/ replace "
            "them after you run sam2_masks.py, and those are the numbers to cite."
        ),
        "images": records,
        "summary": summarize(records),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print()
    print(f"{'set':<10} {'n':>3} {'mean IoU':>10} {'mean boundary F':>16}")
    for kind, stats in report["summary"].items():
        iou = "—" if stats["mean_iou"] is None else f"{stats['mean_iou']:.3f}"
        bf = "—" if stats["mean_boundary_f"] is None else f"{stats['mean_boundary_f']:.3f}"
        print(f"{kind:<10} {stats['n']:>3} {iou:>10} {bf:>16}")


if __name__ == "__main__":
    main(sys.argv[1:])
