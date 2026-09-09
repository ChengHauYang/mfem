#!/usr/bin/env python3
"""Compare profiled setup-and-solve time to MMS accuracy."""

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from profiling_and_error import DEVICES, MMS_CHOICES, ProfileResult, profile_case

P6_SIZES = (4, 8, 16, 32, 64)
P1_SIZES = (32, 64, 128, 256, 512)


@dataclass(frozen=True)
class TimeToAccuracyResult:
    configuration: str
    device: str
    solver: str
    mms: str
    order: int
    n: int
    dofs: int
    l2_error: float
    warmup_iterations: int
    steady_iterations: int
    cold_start_seconds: float
    assembly_seconds: float
    setup_seconds: float
    initial_solve_seconds: float
    median_repeated_solve_seconds: float
    initial_minus_median_repeated_seconds: float
    profiled_setup_initial_solve_seconds: float


def convert_result(result: ProfileResult) -> TimeToAccuracyResult:
    configuration = (
        "P6 CEED AMG + Q1 BoomerAMG"
        if result.order == 6
        else "P1 Hypre BoomerAMG"
    )
    return TimeToAccuracyResult(
        configuration=configuration,
        device=result.device,
        solver=result.solver,
        mms=result.mms,
        order=result.order,
        n=result.n,
        dofs=result.dofs,
        l2_error=result.l2_error,
        warmup_iterations=result.warmup_iterations,
        steady_iterations=result.steady_iterations,
        cold_start_seconds=result.cold_start_seconds,
        assembly_seconds=result.assembly_seconds,
        setup_seconds=result.setup_seconds,
        initial_solve_seconds=result.warmup_solve_seconds,
        median_repeated_solve_seconds=result.steady_solve_seconds,
        initial_minus_median_repeated_seconds=(
            result.warmup_solve_seconds - result.steady_solve_seconds
        ),
        profiled_setup_initial_solve_seconds=(
            result.cold_start_seconds
            + result.assembly_seconds
            + result.setup_seconds
            + result.warmup_solve_seconds
        ),
    )


def write_csv(results: list[TimeToAccuracyResult], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=DEVICES, default="cuda")
    parser.add_argument("--mms", choices=MMS_CHOICES, default="bubble-exp")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--np", dest="mpi_ranks", type=int, default=1)
    parser.add_argument(
        "--executable", type=Path, default=repo / "examples" / "ex1p"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "boomer_q1_time_to_accuracy_results",
    )
    args = parser.parse_args()
    if not 3 <= args.repeats <= 5:
        parser.error("--repeats must be between 3 and 5")
    if args.mpi_ranks <= 0:
        parser.error("--np must be positive")
    args.executable = args.executable.resolve()
    args.output = args.output.resolve()
    if not args.executable.is_file():
        parser.error(f"executable not found: {args.executable}")
    return args


def write_plot(
    results: list[TimeToAccuracyResult], path: Path, mms: str, device: str
) -> None:
    series_definitions = (
        (
            "P6 CEED AMG + Q1 BoomerAMG",
            "ceed-amg",
            6,
            "#d85f45",
            "o",
        ),
        (
            "P1 Hypre BoomerAMG",
            "hypre-amg",
            1,
            "#315b7d",
            "s",
        ),
    )
    figure, axis = plt.subplots(figsize=(9, 6.5), dpi=100)
    for label, solver, order, color, marker in series_definitions:
        series = sorted(
            (
                result
                for result in results
                if result.solver == solver and result.order == order
            ),
            key=lambda result: result.n,
        )
        x_values = [result.profiled_setup_initial_solve_seconds for result in series]
        y_values = [result.l2_error for result in series]
        axis.loglog(
            x_values,
            y_values,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=label,
        )
        for result, x_value, y_value in zip(series, x_values, y_values):
            axis.annotate(
                f"n={result.n}",
                (x_value, y_value),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )

    axis.set_xlabel("Profiled setup + initial solve (all CG iterations) (s)")
    axis.set_ylabel(r"$L^2$ error with respect to MMS")
    axis.set_title(
        f"Profiled Setup-and-Solve Time to Accuracy "
        f"({DEVICES[device]['label']}, MMS: {mms})"
    )
    axis.grid(which="major", color="0.75", linewidth=0.8)
    axis.grid(which="minor", color="0.88", linewidth=0.5, linestyle=":")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = (
        ("ceed-amg", 6, P6_SIZES, ("-ac",)),
        ("hypre-amg", 1, P1_SIZES, ()),
    )
    results: list[TimeToAccuracyResult] = []
    solution_dir = args.output / "solutions"
    for solver, order, sizes, extra_flags in cases:
        for size in sizes:
            result, _ = profile_case(
                args.executable,
                solution_dir,
                args.mpi_ranks,
                args.device,
                solver,
                args.mms,
                order,
                size,
                args.repeats,
                extra_flags,
                save_solution=True,
            )
            results.append(convert_result(result))

    csv_path = args.output / "boomer_q1_time_to_accuracy.csv"
    plot_path = args.output / "boomer_q1_time_to_accuracy.png"
    write_csv(results, csv_path)
    write_plot(results, plot_path, args.mms, args.device)

    print("\nProfiled setup-and-solve time-to-accuracy results:")
    for result in results:
        print(
            f"{result.configuration:30s} n={result.n:3d} "
            f"DOFs={result.dofs:9d} iterations={result.steady_iterations:4d} "
            f"profiled-setup+initial-solve="
            f"{result.profiled_setup_initial_solve_seconds:.6g}s "
            f"L2={result.l2_error:.6e}"
        )
    print(f"\nCSV:       {csv_path}")
    print(f"Plot:      {plot_path}")
    print(f"Solutions: {solution_dir}")


if __name__ == "__main__":
    main()
