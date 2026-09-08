#!/usr/bin/env python3
"""Compare CUDA CEED AMG with Chebyshev and BoomerAMG coarse solvers."""

import argparse
import csv
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from profiling_and_error import ProfileResult, profile_case


@dataclass(frozen=True)
class ComparisonResult:
    coarse_solver: str
    hierarchy: str
    order: int
    n: int
    dofs: int
    l2_error: float
    warmup_iterations: int
    steady_iterations: int
    cold_start_seconds: float
    assembly_seconds: float
    setup_seconds: float
    warmup_solve_seconds: float
    steady_solve_seconds: float
    full_pipeline_seconds: float


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64, help="elements per side")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--np", dest="mpi_ranks", type=int, default=1)
    parser.add_argument(
        "--executable", type=Path, default=repo / "examples" / "ex1p"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "ceed_coarse_solver_results",
    )
    args = parser.parse_args()
    if args.size <= 0 or args.repeats <= 0 or args.mpi_ranks <= 0:
        parser.error("--size, --repeats, and --np must be positive")
    args.executable = args.executable.resolve()
    args.output = args.output.resolve()
    if not args.executable.is_file():
        parser.error(f"executable not found: {args.executable}")
    return args


def convert(result: ProfileResult, coarse_solver: str) -> ComparisonResult:
    return ComparisonResult(
        coarse_solver=coarse_solver,
        hierarchy="Q6 -> Q3 -> Q1 -> Q3 -> Q6",
        order=result.order,
        n=result.n,
        dofs=result.dofs,
        l2_error=result.l2_error,
        warmup_iterations=result.warmup_iterations,
        steady_iterations=result.steady_iterations,
        cold_start_seconds=result.cold_start_seconds,
        assembly_seconds=result.assembly_seconds,
        setup_seconds=result.setup_seconds,
        warmup_solve_seconds=result.warmup_solve_seconds,
        steady_solve_seconds=result.steady_solve_seconds,
        full_pipeline_seconds=result.assembly_setup_solve_total_seconds,
    )


def write_csv(results: list[ComparisonResult], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def write_plot(results: list[ComparisonResult], path: Path) -> None:
    labels = [result.coarse_solver for result in results]
    metrics = (
        ("setup_seconds", "Setup"),
        ("steady_solve_seconds", "Steady solve"),
        ("full_pipeline_seconds", "Full pipeline"),
    )
    x = np.arange(len(labels))
    width = 0.24
    colors = ("#d6a84b", "#7a5195", "#315b7d")
    figure, axis = plt.subplots(figsize=(9, 5.5), dpi=100)
    for index, ((field, name), color) in enumerate(zip(metrics, colors)):
        values = [getattr(result, field) for result in results]
        bars = axis.bar(x + (index - 1) * width, values, width, label=name, color=color)
        axis.bar_label(bars, fmt="%.3gs", padding=3, fontsize=9, rotation=90)
    for index, result in enumerate(results):
        axis.text(
            index,
            max(getattr(result, field) for field, _ in metrics) * 1.65,
            f"{result.steady_iterations} CG iterations",
            ha="center",
            fontweight="bold",
        )
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Time (seconds, log scale)")
    axis.set_title("CUDA CEED AMG: Q1 Coarse Solver Comparison\nQ6 -> Q3 -> Q1 -> Q3 -> Q6")
    axis.grid(axis="y", which="both", linestyle=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = (
        ("Chebyshev on Q1", ()),
        ("BoomerAMG on Q1", ("-ac",)),
    )
    results: list[ComparisonResult] = []
    with tempfile.TemporaryDirectory(prefix="mfem-ceed-coarse-") as temporary_dir:
        work_dir = Path(temporary_dir)
        for name, flags in variants:
            result, _ = profile_case(
                args.executable,
                work_dir / name.replace(" ", "-"),
                args.mpi_ranks,
                "cuda",
                "ceed-amg",
                "bubble-exp",
                6,
                args.size,
                args.repeats,
                flags,
            )
            results.append(convert(result, name))

    csv_path = args.output / "ceed_coarse_solver_comparison.csv"
    plot_path = args.output / "ceed_coarse_solver_comparison.png"
    write_csv(results, csv_path)
    write_plot(results, plot_path)

    print("\nCUDA CEED AMG coarse-solver comparison:")
    for result in results:
        print(
            f"{result.coarse_solver:20s} DOFs={result.dofs:,} "
            f"iterations={result.steady_iterations} "
            f"setup={result.setup_seconds:.6g}s "
            f"steady={result.steady_solve_seconds:.6g}s "
            f"full={result.full_pipeline_seconds:.6g}s"
        )
    original, hybrid = results
    print(
        f"\nHybrid/original iteration ratio: "
        f"{hybrid.steady_iterations / original.steady_iterations:.3f}\n"
        f"Original/hybrid steady speedup: "
        f"{original.steady_solve_seconds / hybrid.steady_solve_seconds:.3f}x"
    )
    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
