#!/usr/bin/env python3
"""Re-render the GPU showcase PNGs from an existing gpu_showcase_<platform>.csv.

Reads the CSV produced by gpu_showcase.py and overwrites the comparison and
phases PNGs carrying the same platform tag in the same directory, with a layout
that does not overlap the tallest bars with the legend or the per-bar value
labels. Does not re-run any simulation.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from profiling_and_error import default_tagged_input, split_platform_tag

DEVICE_LABEL = {"cpu": "CPU", "cuda": "GPU (CUDA)"}
DEVICE_COLOR = {"cpu": "#315b7d", "cuda": "#e36b3d"}
PHASES = (
    ("cold_start_seconds", "Early initialization", "#61788a"),
    (
        "assembly_seconds",
        "Operator construction (incl. GPU first use)",
        "#94a89a",
    ),
    ("setup_seconds", "Preconditioner setup", "#d6a84b"),
    ("warmup_solve_seconds", "Initial solve (all CG iterations)", "#dc7653"),
)


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as stream:
        return list(csv.DictReader(stream))


def write_comparison_plot(rows: list[dict], path: Path) -> None:
    cases = list(dict.fromkeys(row["case"] for row in rows))
    metrics = (
        ("steady_solve_seconds", "Median repeated solve (all CG iterations)"),
        ("first_run", "Profiled setup + initial solve (all CG iterations)"),
    )
    by_key = {(row["case"], row["device"]): row for row in rows}
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=100)
    x = np.arange(len(cases))
    width = 0.36

    for axis, (field, title) in zip(axes, metrics):
        all_values: list[float] = []
        for index, device in enumerate(("cpu", "cuda")):
            values = []
            for case in cases:
                row = by_key[(case, device)]
                value = (
                    sum(
                        float(row[name])
                        for name in (
                            "cold_start_seconds",
                            "assembly_seconds",
                            "setup_seconds",
                            "warmup_solve_seconds",
                        )
                    )
                    if field == "first_run"
                    else float(row[field])
                )
                values.append(value)
            all_values.extend(values)
            offset = (index - 0.5) * width
            bars = axis.bar(
                x + offset, values, width,
                label=DEVICE_LABEL[device], color=DEVICE_COLOR[device],
            )
            for bar, value in zip(bars, values):
                axis.annotate(
                    f"{value:.3g}s",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, rotation=90,
                )
        for case_index, case in enumerate(cases):
            values = []
            for device in ("cpu", "cuda"):
                row = by_key[(case, device)]
                value = (
                    sum(
                        float(row[name])
                        for name in (
                            "cold_start_seconds",
                            "assembly_seconds",
                            "setup_seconds",
                            "warmup_solve_seconds",
                        )
                    )
                    if field == "first_run"
                    else float(row[field])
                )
                values.append(value)
            cpu, cuda = values
            speedup = cpu / cuda
            comparison = (
                f"{speedup:.1f}x faster"
                if speedup >= 1.0
                else f"{1.0 / speedup:.1f}x slower"
            )
            axis.text(
                case_index, max(cpu, cuda) * 3.0,
                comparison,
                ha="center", va="bottom", fontweight="bold", fontsize=9,
            )
        axis.set_yscale("log")
        # Leave headroom above the comparison annotations and shared legend.
        axis.set_ylim(top=max(all_values) * 20.0)
        axis.set_title(title)
        axis.set_xticks(x, cases, rotation=18, ha="right")
        axis.set_ylabel("Time (seconds, log scale)")
        axis.grid(axis="y", which="both", linestyle=":", alpha=0.5)

    figure.suptitle(
        "MFEM GPU Showcase (annotations compare GPU with CPU)", fontsize=15, y=0.995,
    )
    # One shared legend, above the axes and below the suptitle, so it never
    # overlaps the tallest CPU bars in either subplot.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DEVICE_COLOR[d]) for d in ("cpu", "cuda")
    ]
    figure.legend(
        handles, [DEVICE_LABEL[d] for d in ("cpu", "cuda")],
        loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.955),
        frameon=False, fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_phase_plot(rows: list[dict], path: Path) -> None:
    labels = [f"{row['case']}\n{DEVICE_LABEL[row['device']]}" for row in rows]
    figure, axis = plt.subplots(figsize=(14, 6.4), dpi=100)
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    totals = np.zeros(len(rows))
    for field, label, color in PHASES:
        values = np.array([float(row[field]) for row in rows])
        axis.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
        totals += values

    for xi, total in zip(x, totals):
        axis.annotate(
            f"{total:.3g}s",
            (xi, total), xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )
    axis.set_ylim(top=totals.max() * 1.15)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("Time (seconds)")
    axis.set_title("Profiled Setup + Initial Solve Breakdown")
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    # Legend below the plot so it can never occlude the tallest stack
    # (p1 PA Jacobi CPU is ~100s tall on this axis).
    axis.legend(
        ncols=len(PHASES), loc="upper center",
        bbox_to_anchor=(0.5, -0.22), frameon=False,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent / "gpu_showcase_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=default_tagged_input(default_dir, "gpu_showcase"),
        help="path to gpu_showcase_<platform>.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--comparison-out", type=Path, default=None,
        help="output PNG for the comparison plot "
             "(default: <input dir>/gpu_showcase_<platform>.png)",
    )
    parser.add_argument(
        "--phases-out", type=Path, default=None,
        help="output PNG for the phases plot "
             "(default: <input dir>/gpu_showcase_phases_<platform>.png)",
    )
    args = parser.parse_args()
    args.input = args.input.resolve()
    if not args.input.is_file():
        parser.error(f"input CSV not found: {args.input}")
    parent = args.input.parent
    # Follow the tag the CSV already carries, so replotting a Linux run on a Mac
    # does not relabel its PNGs.
    base, tag = split_platform_tag(args.input.stem)
    args.comparison_out = (
        args.comparison_out or parent / f"{base}{tag}.png"
    ).resolve()
    args.phases_out = (
        args.phases_out or parent / f"{base}_phases{tag}.png"
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
