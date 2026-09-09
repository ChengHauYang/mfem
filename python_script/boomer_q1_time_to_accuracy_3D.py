#!/usr/bin/env python3
"""Compare profiled setup-and-solve time to MMS accuracy on the unit cube."""

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from profiling_and_error import (
    DEVICES,
    MMS_CHOICES,
    ProfileResult,
    build_has_cuda,
    default_device,
    platform_suffix,
    profile_case,
    tagged,
)

DIM = 3
# Roughly a quarter of the 2D sizes in each direction: the DOF counts start
# where the 2D study starts and climb far higher.
#
# The sweep is capped by memory, not by time, because ex1p builds the *serial*
# Mesh on every rank before partitioning it. The laptop stops at 128^3 = 2.1M
# hexes; the Linux node has enough memory for one more refinement of each
# sweep, so it also runs p6 n=32 (7.2M DOFs) and p1 n=256 (16.7M hexes of
# replicated serial mesh, 17.0M DOFs). Unknown platforms get the cautious set.
SIZES_BY_PLATFORM = {
    "linux": {"p6": (2, 4, 8, 16, 32), "p1": (16, 32, 64, 128, 256)},
    "mac": {"p6": (2, 4, 8, 16), "p1": (16, 32, 64, 128)},
}
DEFAULT_SIZES = SIZES_BY_PLATFORM.get(platform_suffix(), SIZES_BY_PLATFORM["mac"])
P6_SIZES = DEFAULT_SIZES["p6"]
P1_SIZES = DEFAULT_SIZES["p1"]


@dataclass(frozen=True)
class TimeToAccuracyResult:
    configuration: str
    dim: int
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
        dim=result.dim,
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
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default=default_device(repo),
        help="default: %(default)s (cuda only when this MFEM build has "
             "MFEM_USE_CUDA=YES and a GPU is visible)",
    )
    parser.add_argument("--mms", choices=MMS_CHOICES, default="bubble-exp")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--np", dest="mpi_ranks", type=int, default=1)
    parser.add_argument(
        "--p6-sizes",
        nargs="+",
        type=int,
        default=P6_SIZES,
        help="elements per side for the p6 sweep "
             f"(default on {platform_suffix()}: {list(P6_SIZES)})",
    )
    parser.add_argument(
        "--p1-sizes",
        nargs="+",
        type=int,
        default=P1_SIZES,
        help="elements per side for the p1 sweep "
             f"(default on {platform_suffix()}: {list(P1_SIZES)})",
    )
    parser.add_argument(
        "--executable", type=Path, default=repo / "examples" / "ex1p"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "boomer_q1_time_to_accuracy_3D_results",
    )
    args = parser.parse_args()
    if not 3 <= args.repeats <= 5:
        parser.error("--repeats must be between 3 and 5")
    if args.mpi_ranks <= 0:
        parser.error("--np must be positive")
    if args.device == "cuda" and build_has_cuda(repo) is False:
        parser.error(
            "this MFEM build has MFEM_USE_CUDA=NO, so --device cuda aborts in "
            "Device::Setup; rebuild with CUDA or use --device cpu"
        )
    args.p6_sizes = tuple(sorted(set(args.p6_sizes)))
    args.p1_sizes = tuple(sorted(set(args.p1_sizes)))
    if any(size <= 0 for size in args.p6_sizes + args.p1_sizes):
        parser.error("mesh sizes must be positive")
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
        f"3D Profiled Setup-and-Solve Time to Accuracy "
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
        ("ceed-amg", 6, args.p6_sizes, ("-ac",)),
        ("hypre-amg", 1, args.p1_sizes, ()),
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
                dim=DIM,
            )
            results.append(convert_result(result))

    csv_path = args.output / tagged("boomer_q1_time_to_accuracy_3D", ".csv")
    plot_path = args.output / tagged("boomer_q1_time_to_accuracy_3D", ".png")
    write_csv(results, csv_path)
    write_plot(results, plot_path, args.mms, args.device)

    print("\n3D profiled setup-and-solve time-to-accuracy results:")
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
