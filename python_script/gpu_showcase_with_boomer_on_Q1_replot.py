#!/usr/bin/env python3
"""Re-render the Q1 BoomerAMG GPU showcase plots without rerunning simulations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from gpu_showcase_replot import load_rows, write_comparison_plot, write_phase_plot


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
    }
)


def parse_args() -> argparse.Namespace:
    default_dir = (
        Path(__file__).resolve().parent
        / "gpu_showcase_with_boomer_on_Q1_results"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_dir / "gpu_showcase_with_boomer_on_Q1.csv",
        help="input CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=None,
        help="comparison PNG (default: <input dir>/gpu_showcase_with_boomer_on_Q1.png)",
    )
    parser.add_argument(
        "--phases-out",
        type=Path,
        default=None,
        help="phase PNG (default: <input dir>/gpu_showcase_with_boomer_on_Q1_phases.png)",
    )
    args = parser.parse_args()
    args.input = args.input.resolve()
    if not args.input.is_file():
        parser.error(f"input CSV not found: {args.input}")
    parent = args.input.parent
    args.comparison_out = (
        args.comparison_out
        or parent / "gpu_showcase_with_boomer_on_Q1.png"
    ).resolve()
    args.phases_out = (
        args.phases_out
        or parent / "gpu_showcase_with_boomer_on_Q1_phases.png"
    ).resolve()
    return args


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"no rows in {args.input}")
    write_comparison_plot(rows, args.comparison_out)
    write_phase_plot(rows, args.phases_out)
    print(f"Comparison: {args.comparison_out}")
    print(f"Phases:     {args.phases_out}")


if __name__ == "__main__":
    main()
