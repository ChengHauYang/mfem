#!/usr/bin/env python3
"""Run focused MFEM cases that expose GPU throughput and startup costs."""

import argparse
import csv
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from profiling_and_error import DEVICES, SOLVERS, ProfileResult, profile_case


@dataclass(frozen=True)
class ShowcaseResult:
    case: str
    hierarchy: str
    device: str
    solver: str
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
    solve_total_seconds: float
    assembly_setup_solve_total_seconds: float


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p1-size",
        type=int,
        default=512,
        help="elements per side for the p1 PA Jacobi case (default: 512)",
    )
    parser.add_argument(
        "--p6-size",
        type=int,
        default=64,
        help="elements per side for all p6 solver cases (default: 64)",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--np", dest="mpi_ranks", type=int, default=1)
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=DEVICES,
        default=("cpu", "cuda"),
    )
    parser.add_argument(
        "--executable", type=Path, default=repo / "examples" / "ex1p"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "gpu_showcase_results",
    )
    args = parser.parse_args()
    if args.p1_size <= 0 or args.p6_size <= 0:
        parser.error("--p1-size and --p6-size must be positive")
    if args.repeats <= 0 or args.mpi_ranks <= 0:
        parser.error("--repeats and --np must be positive")
    if len(args.devices) != 2 or set(args.devices) != {"cpu", "cuda"}:
        parser.error("showcase requires exactly --devices cpu cuda")
    args.executable = args.executable.resolve()
    args.output = args.output.resolve()
    if not args.executable.is_file():
        parser.error(f"executable not found: {args.executable}")
    return args


def case_name(order: int, size: int, solver: str) -> str:
    return f"p{order} {SOLVERS[solver]['label']} (n={size})"


def convert_result(result: ProfileResult) -> ShowcaseResult:
    hierarchy = "Q6 -> Q3 -> Q1 -> Q3 -> Q6" if result.solver == "ceed-amg" else "n/a"
    return ShowcaseResult(
        case=case_name(result.order, result.n, result.solver),
        hierarchy=hierarchy,
        device=result.device,
        solver=result.solver,
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
        solve_total_seconds=result.solve_total_seconds,
        assembly_setup_solve_total_seconds=result.assembly_setup_solve_total_seconds,
    )


def write_csv(results: list[ShowcaseResult], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def write_comparison_plot(results: list[ShowcaseResult], path: Path) -> None:
    cases = list(dict.fromkeys(result.case for result in results))
    metrics = (
        ("steady_solve_seconds", "Steady-state solve"),
        ("assembly_setup_solve_total_seconds", "Cold + assembly + setup + two solves"),
    )
    colors = {"cpu": "#315b7d", "cuda": "#e36b3d"}
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=100)
    x = np.arange(len(cases))
    width = 0.36

    for axis, (field, title) in zip(axes, metrics):
        by_key = {(result.case, result.device): result for result in results}
        for index, device in enumerate(("cpu", "cuda")):
            values = [getattr(by_key[(case, device)], field) for case in cases]
            offset = (index - 0.5) * width
            bars = axis.bar(
                x + offset,
                values,
                width,
                label=DEVICES[device]["label"],
                color=colors[device],
            )
            for bar, value in zip(bars, values):
                axis.annotate(
                    f"{value:.3g}s",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
        for case_index, case in enumerate(cases):
            cpu = getattr(by_key[(case, "cpu")], field)
            cuda = getattr(by_key[(case, "cuda")], field)
            axis.text(
                case_index,
                max(cpu, cuda) * 1.8,
                f"{cpu / cuda:.1f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xticks(x, cases, rotation=18, ha="right")
        axis.set_ylabel("Time (seconds, log scale)")
        axis.grid(axis="y", which="both", linestyle=":", alpha=0.5)

    axes[0].legend(loc="upper left")
    figure.suptitle("MFEM GPU Showcase (labels show CPU / GPU speedup)", fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_phase_plot(results: list[ShowcaseResult], path: Path) -> None:
    labels = [f"{result.case}\n{DEVICES[result.device]['label']}" for result in results]
    phases = (
        ("cold_start_seconds", "Cold start", "#61788a"),
        ("assembly_seconds", "Assembly", "#94a89a"),
        ("setup_seconds", "Setup", "#d6a84b"),
        ("warmup_solve_seconds", "Warm-up solve", "#dc7653"),
        ("steady_solve_seconds", "Steady solve", "#7a5195"),
    )
    figure, axis = plt.subplots(figsize=(14, 6), dpi=100)
    x = np.arange(len(results))
    bottom = np.zeros(len(results))
    for field, label, color in phases:
        values = np.array([getattr(result, field) for result in results])
        axis.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("Time (seconds)")
    axis.set_title("Independent Profiling Phases")
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    axis.legend(ncols=3, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = (
        (1, args.p1_size, "pa-jacobi"),
        (6, args.p6_size, "pa-jacobi"),
        (6, args.p6_size, "ceed-amg"),
        (6, args.p6_size, "hypre-amg"),
    )
    results: list[ShowcaseResult] = []
    with tempfile.TemporaryDirectory(prefix="mfem-gpu-showcase-") as temporary_dir:
        work_dir = Path(temporary_dir)
        for order, size, solver in cases:
            for device in args.devices:
                result, _ = profile_case(
                    args.executable,
                    work_dir,
                    args.mpi_ranks,
                    device,
                    solver,
                    "bubble-exp",
                    order,
                    size,
                    args.repeats,
                )
                results.append(convert_result(result))

    csv_path = args.output / "gpu_showcase.csv"
    comparison_path = args.output / "gpu_showcase.png"
    phase_path = args.output / "gpu_showcase_phases.png"
    write_csv(results, csv_path)
    write_comparison_plot(results, comparison_path)
    write_phase_plot(results, phase_path)

    by_key = {(result.case, result.device): result for result in results}
    print("\nCPU/GPU comparison:")
    for case in dict.fromkeys(result.case for result in results):
        cpu = by_key[(case, "cpu")]
        cuda = by_key[(case, "cuda")]
        hierarchy = f", hierarchy={cuda.hierarchy}" if cuda.hierarchy != "n/a" else ""
        print(
            f"{case:20s} DOFs={cuda.dofs:9d}{hierarchy}\n"
            f"  steady: CPU={cpu.steady_solve_seconds:.6g}s "
            f"GPU={cuda.steady_solve_seconds:.6g}s "
            f"speedup={cpu.steady_solve_seconds / cuda.steady_solve_seconds:.2f}x\n"
            f"  full:   CPU={cpu.assembly_setup_solve_total_seconds:.6g}s "
            f"GPU={cuda.assembly_setup_solve_total_seconds:.6g}s "
            f"speedup={cpu.assembly_setup_solve_total_seconds / cuda.assembly_setup_solve_total_seconds:.2f}x"
        )
    print(f"\nCSV:        {csv_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Phases:     {phase_path}")


if __name__ == "__main__":
    main()
