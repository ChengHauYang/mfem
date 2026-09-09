#!/usr/bin/env python3
"""Run the GPU showcase with an additional CEED AMG Q1 BoomerAMG case."""

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

from gpu_showcase import convert_result, write_comparison_plot, write_csv, write_phase_plot
from profiling_and_error import profile_case, tagged


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
        "--executable", type=Path, default=repo / "examples" / "ex1p"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "gpu_showcase_with_boomer_on_Q1_results",
    )
    args = parser.parse_args()
    if args.p1_size <= 0 or args.p6_size <= 0:
        parser.error("--p1-size and --p6-size must be positive")
    if args.repeats <= 0 or args.mpi_ranks <= 0:
        parser.error("--repeats and --np must be positive")
    args.executable = args.executable.resolve()
    args.output = args.output.resolve()
    if not args.executable.is_file():
        parser.error(f"executable not found: {args.executable}")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = (
        ("p1 PA Jacobi", 1, args.p1_size, "pa-jacobi", ()),
        ("p6 PA Jacobi", 6, args.p6_size, "pa-jacobi", ()),
        ("p6 CEED AMG", 6, args.p6_size, "ceed-amg", ()),
        (
            "p6 CEED AMG\n+ Q1 BoomerAMG",
            6,
            args.p6_size,
            "ceed-amg",
            ("-ac",),
        ),
        ("p6 Hypre BoomerAMG", 6, args.p6_size, "hypre-amg", ()),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="mfem-gpu-showcase-q1-") as temporary_dir:
        work_dir = Path(temporary_dir)
        for label, order, size, solver, flags in cases:
            for device in ("cpu", "cuda"):
                result, _ = profile_case(
                    args.executable,
                    work_dir / label.replace(" ", "-").replace("\n", "-"),
                    args.mpi_ranks,
                    device,
                    solver,
                    "bubble-exp",
                    order,
                    size,
                    args.repeats,
                    flags,
                )
                converted = convert_result(result)
                hierarchy = (
                    "Q6 -> Q3 -> Q1 (BoomerAMG) -> Q3 -> Q6"
                    if flags
                    else converted.hierarchy
                )
                results.append(replace(converted, case=label, hierarchy=hierarchy))

    csv_path = args.output / tagged("gpu_showcase_with_boomer_on_Q1", ".csv")
    comparison_path = args.output / tagged("gpu_showcase_with_boomer_on_Q1", ".png")
    phase_path = args.output / tagged(
        "gpu_showcase_with_boomer_on_Q1_phases", ".png"
    )
    write_csv(results, csv_path)
    write_comparison_plot(results, comparison_path)
    write_phase_plot(results, phase_path)

    by_key = {(result.case, result.device): result for result in results}
    print("\nCPU/GPU comparison:")
    for label, _, _, _, _ in cases:
        cpu = by_key[(label, "cpu")]
        cuda = by_key[(label, "cuda")]
        print(
            f"{label.replace(chr(10), ' '):32s} DOFs={cuda.dofs:9d}\n"
            f"  CPU:  repeated iterations={cpu.steady_iterations:4d} "
            f"median repeated solve={cpu.steady_solve_seconds:.6g}s "
            f"profiled setup + initial solve="
            f"{cpu.assembly_setup_solve_total_seconds:.6g}s\n"
            f"  CUDA: repeated iterations={cuda.steady_iterations:4d} "
            f"median repeated solve={cuda.steady_solve_seconds:.6g}s "
            f"profiled setup + initial solve="
            f"{cuda.assembly_setup_solve_total_seconds:.6g}s\n"
            f"  speedup: median repeated solve="
            f"{cpu.steady_solve_seconds / cuda.steady_solve_seconds:.2f}x "
            f"profiled setup + initial solve="
            f"{cpu.assembly_setup_solve_total_seconds / cuda.assembly_setup_solve_total_seconds:.2f}x"
        )

    print(f"\nCSV:        {csv_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Phases:     {phase_path}")


if __name__ == "__main__":
    main()
