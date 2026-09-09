#!/usr/bin/env python3
"""Run and plot a mesh-convergence study for MFEM ex1p on the unit square."""

import argparse
import csv
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from profiling_and_error import tagged

DEFAULT_ORDERS = tuple(range(1, 8))
MMS_CHOICES = ("sine", "multimode", "bubble-exp")
DEFAULT_SIZES_BY_ORDER = {
    1: (4, 8, 16, 32, 64),
    2: (2, 4, 8, 16, 32),
    3: (2, 4, 8, 16, 32),
    4: (1, 2, 4, 8, 16),
    5: (1, 2, 4, 8, 16),
    6: (1, 2, 4, 8),
    7: (1, 2, 4, 8),
}
SOLUTION_PATTERN = "sol.[0-9][0-9][0-9][0-9][0-9][0-9]"
MESH_PATTERN = "mesh.[0-9][0-9][0-9][0-9][0-9][0-9]"


@dataclass(frozen=True)
class StudyConfig:
    executable: Path
    output_dir: Path
    sizes: tuple[int, ...] | None
    orders: tuple[int, ...]
    mpi_ranks: int
    mms: str
    keep_fields: bool


@dataclass(frozen=True)
class CaseResult:
    mms: str
    order: int
    n: int
    h: float
    elements: int
    dofs: int
    l2_error: float


def write_inline_quad(path: Path, size: int) -> None:
    path.write_text(
        "MFEM INLINE mesh v1.0\n\n"
        f"type = quad\nnx = {size}\nny = {size}\nsx = 1.0\nsy = 1.0\n"
    )


def clean_case_outputs(case_dir: Path) -> None:
    for pattern in (SOLUTION_PATTERN, MESH_PATTERN):
        for path in case_dir.glob(pattern):
            path.unlink()
    shutil.rmtree(case_dir / "ParaView", ignore_errors=True)


def run_case(config: StudyConfig, order: int, size: int) -> CaseResult:
    case_dir = config.output_dir / f"order{order}" / f"n{size:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    clean_case_outputs(case_dir)

    mesh_path = case_dir / "inline-quad.mesh"
    log_path = case_dir / "run.log"
    write_inline_quad(mesh_path, size)

    command = [
        "mpirun",
        "-np",
        str(config.mpi_ranks),
        str(config.executable),
        "-m",
        str(mesh_path),
        "-o",
        str(order),
        "-rs",
        "0",
        "-pa",
        "-no-fa",
        "-d",
        "ceed-cpu",
        "-a",
        "-no-vis",
        "-l2",
        "-mms",
        config.mms,
        "-pv" if config.keep_fields else "-no-pv",
    ]
    print(f"Running order {order}, {size} x {size} mesh ...", flush=True)
    process = subprocess.run(command, cwd=case_dir, text=True, capture_output=True)
    log_path.write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"order={order}, n={size} failed; see {log_path}")

    dofs_match = re.search(r"Number of finite element unknowns: (\d+)", process.stdout)
    error_match = re.search(r"L2 norm of error: ([0-9.eE+-]+)", process.stdout)
    if not dofs_match or not error_match:
        raise RuntimeError(f"could not parse DOFs or L2 error; see {log_path}")

    result = CaseResult(
        mms=config.mms,
        order=order,
        n=size,
        h=1.0 / size,
        elements=size * size,
        dofs=int(dofs_match.group(1)),
        l2_error=float(error_match.group(1)),
    )
    if not config.keep_fields:
        clean_case_outputs(case_dir)
    return result


def convergence_rate(results: list[CaseResult]) -> float:
    asymptotic_results = results[-3:]
    log_h = [math.log(result.h) for result in asymptotic_results]
    log_error = [math.log(result.l2_error) for result in asymptotic_results]
    mean_h = sum(log_h) / len(log_h)
    mean_error = sum(log_error) / len(log_error)
    return sum(
        (h - mean_h) * (error - mean_error)
        for h, error in zip(log_h, log_error)
    ) / sum((h - mean_h) ** 2 for h in log_h)


def write_csv(results: list[CaseResult], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def write_png(
    results: list[CaseResult], rates: dict[int, float], mms: str, path: Path
) -> None:
    plt.rc("text", usetex=True)
    plt.rc("font", family="Times New Roman")
    figure = plt.figure(figsize=(9, 7), dpi=80, facecolor="w")
    colors = plt.cm.tab10(np.linspace(0, 1, len(rates)))
    markers = ("o", "s", "^", "D", "v", "P", "X")

    for color, marker, order in zip(colors, markers, rates):
        order_results = [result for result in results if result.order == order]
        mesh_sizes = np.array([result.h for result in order_results])
        errors = np.array([result.l2_error for result in order_results])
        plt.loglog(
            mesh_sizes,
            errors,
            marker=marker,
            color=color,
            linewidth=1.8,
            markersize=6,
            markerfacecolor="none",
            markeredgewidth=1.5,
            label=rf"$p={order}$, slope ${rates[order]:.2f}$",
        )

        # Draw a 1:(p+1) slope triangle beside the finest-grid segment.
        x0 = mesh_sizes[-1] * 1.12
        x1 = x0 * 1.35
        y1 = errors[-1] * (x1 / mesh_sizes[-1]) ** (order + 1) / 3.0
        y0 = y1 / (x1 / x0) ** (order + 1)
        plt.loglog(
            [x0, x1, x1, x0],
            [y0, y0, y1, y0],
            color="black",
            linewidth=1.0,
        )
        plt.text(
            math.sqrt(x0 * x1),
            y0 / 1.35,
            r"$1$",
            fontsize=9,
            horizontalalignment="center",
            verticalalignment="top",
        )
        plt.text(
            x1 * 1.035,
            math.sqrt(y0 * y1),
            rf"${order + 1}$",
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="center",
        )

    plt.xlabel(r"$\textrm{Mesh size }h$", fontsize=15)
    plt.ylabel(r"$L^2\textrm{ error}$", fontsize=15)
    plt.title(
        rf"$\textrm{{Convergence by Polynomial Order ({mms} MMS)}}$",
        fontsize=17,
    )
    plt.legend(fontsize=11, loc="best", frameon=True, framealpha=1.0)
    plt.grid(which="major", color="0.75", linewidth=0.8)
    plt.grid(which="minor", color="0.88", linewidth=0.5, linestyle=":")
    plt.tick_params(
        axis="both", which="both", direction="in", top=True, right=True,
        labelsize=12, width=1.2
    )
    figure.savefig(path, format="png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_config() -> StudyConfig:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        help="mesh sizes for every order (default: order-specific sizes)",
    )
    parser.add_argument("--orders", nargs="+", type=int, default=DEFAULT_ORDERS)
    parser.add_argument(
        "--mms",
        choices=MMS_CHOICES,
        default="bubble-exp",
        help="manufactured solution (default: bubble-exp)",
    )
    parser.add_argument("--np", dest="mpi_ranks", type=int, default=1)
    parser.add_argument("--executable", type=Path, default=repo / "examples" / "ex1p")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "mesh_refinement_results",
    )
    parser.add_argument(
        "--keep-fields",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep per-rank fields and write a separate ParaView dataset for each n",
    )
    args = parser.parse_args()

    sizes = tuple(sorted(set(args.sizes))) if args.sizes else None
    orders = tuple(sorted(set(args.orders)))
    if sizes is not None and len(sizes) < 2:
        parser.error("provide at least two mesh sizes")
    if sizes is not None and any(size <= 0 for size in sizes):
        parser.error("mesh sizes must be positive")
    if not orders or any(order not in DEFAULT_SIZES_BY_ORDER for order in orders):
        parser.error("orders must be between 1 and 7")
    if args.mpi_ranks <= 0:
        parser.error("--np must be positive")

    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable not found: {executable}; run 'make -C examples ex1p'")
    return StudyConfig(
        executable=executable,
        output_dir=args.output.resolve(),
        sizes=sizes,
        orders=orders,
        mpi_ranks=args.mpi_ranks,
        mms=args.mms,
        keep_fields=args.keep_fields,
    )


def main() -> None:
    config = parse_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for order in config.orders:
        sizes = config.sizes or DEFAULT_SIZES_BY_ORDER[order]
        results.extend(run_case(config, order, size) for size in sizes)
    rates = {
        order: convergence_rate([result for result in results if result.order == order])
        for order in config.orders
    }

    csv_path = config.output_dir / tagged("convergence", ".csv")
    plot_path = config.output_dir / tagged("mesh_convergence", ".png")
    write_csv(results, csv_path)
    write_png(results, rates, config.mms, plot_path)

    print("\n p   n       DOFs       L2 error")
    for result in results:
        print(
            f"{result.order:2d} {result.n:3d} {result.dofs:10d} "
            f"{result.l2_error:.6e}"
        )
    print("\nObserved convergence rates:")
    for order, rate in rates.items():
        print(f"  p={order}: {rate:.3f} (expected L2 rate {order + 1})")
    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
