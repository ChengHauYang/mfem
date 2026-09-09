#!/usr/bin/env python3
"""Compare GPU preconditioners for the 2D p=6, n=512 Poisson problem."""

import argparse
import csv
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gpu_showcase import convert_result
from profiling_and_error import profile_case, tagged

ORDER = 6
SIZE = 512
RELATIVE_TOLERANCE = 1e-12
MAX_ITERATIONS = 20_000


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--np", dest="mpi_ranks", type=int, default=1)
    parser.add_argument(
        "--executable", type=Path, default=repo / "examples" / "ex1p"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "gpu_p6_n512_preconditioner_comparison_results",
    )
    args = parser.parse_args()
    if args.repeats <= 0 or args.mpi_ranks <= 0:
        parser.error("--repeats and --np must be positive")
    args.executable = args.executable.resolve()
    args.output = args.output.resolve()
    if not args.executable.is_file():
        parser.error(f"executable not found: {args.executable}")
    return args


def write_csv(results: list, path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def write_comparison_plot(results: list, path: Path) -> None:
    metrics = (
        ("steady_solve_seconds", "Median repeated solve"),
        ("assembly_setup_solve_total_seconds", "Setup + initial solve"),
    )
    colors = ("#315b7d", "#3f8f72", "#d6a84b", "#d85f45")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    x = np.arange(len(results))
    labels = [result.case for result in results]
    for axis, (field, title) in zip(axes, metrics):
        values = [getattr(result, field) for result in results]
        bars = axis.bar(x, values, color=colors)
        for bar, result, value in zip(bars, results, values):
            status = "converged" if result.steady_converged else "NOT CONVERGED"
            axis.annotate(
                f"{value:.3g}s\n{result.steady_iterations} it.\n{status}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel("Time (seconds, log scale)")
        axis.grid(axis="y", which="both", linestyle=":", alpha=0.5)
        axis.set_ylim(top=max(values) * 8)
    figure.suptitle(
        "2D GPU Poisson: p=6, n=512, rtol=1e-12, max iterations=20,000"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_phase_plot(results: list, path: Path) -> None:
    phases = (
        ("cold_start_seconds", "Early initialization", "#61788a"),
        ("assembly_seconds", "Operator construction", "#94a89a"),
        ("setup_seconds", "Preconditioner setup", "#d6a84b"),
        ("warmup_solve_seconds", "Initial solve", "#dc7653"),
    )
    x = np.arange(len(results))
    bottom = np.zeros(len(results))
    figure, axis = plt.subplots(figsize=(11, 6), dpi=100)
    for field, label, color in phases:
        values = np.array([getattr(result, field) for result in results])
        axis.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    for xi, total in zip(x, bottom):
        axis.annotate(
            f"{total:.3g}s", (xi, total), xytext=(0, 4),
            textcoords="offset points", ha="center", fontsize=8,
        )
    axis.set_xticks(x, [result.case for result in results], rotation=18, ha="right")
    axis.set_ylabel("Time (seconds)")
    axis.set_title("2D GPU p=6, n=512: Setup + Initial Solve Breakdown")
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    axis.legend(ncols=2, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = (
        ("PA Jacobi", "pa-jacobi", (), "n/a"),
        ("CEED AMG", "ceed-amg", (), "Q6 -> Q3 -> Q1 -> Q3 -> Q6"),
        (
            "CEED AMG + Q1 BoomerAMG", "ceed-amg", ("-ac",),
            "Q6 -> Q3 -> Q1 (BoomerAMG) -> Q3 -> Q6",
        ),
        ("Hypre BoomerAMG", "hypre-amg", (), "n/a"),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="mfem-gpu-p6-n512-") as temporary_dir:
        work_dir = Path(temporary_dir)
        for label, solver, flags, hierarchy in cases:
            result, _ = profile_case(
                args.executable,
                work_dir / label.replace(" ", "-"),
                args.mpi_ranks,
                "cuda",
                solver,
                "bubble-exp",
                ORDER,
                SIZE,
                args.repeats,
                flags,
                cg_relative_tolerance=RELATIVE_TOLERANCE,
                cg_max_iterations=MAX_ITERATIONS,
            )
            results.append(
                replace(convert_result(result), case=label, hierarchy=hierarchy)
            )

    base = "gpu_p6_n512_preconditioner_comparison"
    csv_path = args.output / tagged(base, ".csv")
    comparison_path = args.output / tagged(base, ".png")
    phase_path = args.output / tagged(f"{base}_phases", ".png")
    write_csv(results, csv_path)
    write_comparison_plot(results, comparison_path)
    write_phase_plot(results, phase_path)

    print("\n2D GPU p=6, n=512 preconditioner comparison:")
    for result in results:
        print(
            f"{result.case:28s} converged={result.steady_converged!s:5s} "
            f"iterations={result.steady_iterations:5d} "
            f"residual={result.steady_final_residual_norm:.6e} "
            f"solve={result.steady_solve_seconds:.6g}s "
            f"setup+first={result.assembly_setup_solve_total_seconds:.6g}s "
            f"GPU peak={result.gpu_memory_peak_mib:.0f} MiB "
            f"delta={result.gpu_memory_peak_delta_mib:.0f} MiB"
        )

    print(f"\nCSV:        {csv_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Phases:     {phase_path}")


if __name__ == "__main__":
    main()
