#!/usr/bin/env python3
"""Compare MFEM solver configurations by setup-and-solve time versus L2 error."""

import argparse
import csv
import re
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_ORDERS = tuple(range(1, 8))
DEFAULT_SIZES_BY_ORDER = {
    1: (8, 16, 32, 64, 128, 256),
    2: (4, 8, 16, 32, 64, 128),
    3: (4, 8, 16, 32, 64, 128),
    4: (2, 4, 8, 16, 32),
    5: (2, 4, 8, 16),
    6: (1, 2, 4, 8),
    7: (1, 2, 4, 8),
}
MMS_CHOICES = ("sine", "multimode", "bubble-exp")
DEVICES = {
    "cpu": {"label": "CPU", "ceed": "ceed-cpu", "hypre": "cpu"},
    "cuda": {"label": "GPU (CUDA)", "ceed": "ceed-cuda", "hypre": "cuda"},
}
SOLVERS = {
    "ceed-amg": {
        "label": "CEED AMG",
        "family": "ceed",
        "flags": ("-pa", "-no-fa", "-a"),
    },
    "pa-jacobi": {
        "label": "PA Jacobi",
        "family": "ceed",
        "flags": ("-pa", "-no-fa", "-no-a"),
    },
    "hypre-amg": {
        "label": "Hypre BoomerAMG",
        "family": "hypre",
        "flags": ("-no-pa", "-no-fa", "-no-a"),
    },
}


@dataclass(frozen=True)
class ProfileResult:
    device: str
    solver: str
    mms: str
    order: int
    n: int
    h: float
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


def write_inline_quad(path: Path, size: int) -> None:
    path.write_text(
        "MFEM INLINE mesh v1.0\n\n"
        f"type = quad\nnx = {size}\nny = {size}\nsx = 1.0\nsy = 1.0\n"
    )


def run_once(
    executable: Path,
    case_dir: Path,
    mpi_ranks: int,
    device: str,
    solver: str,
    mms: str,
    order: int,
    size: int,
    repeats: int,
    extra_flags: tuple[str, ...] = (),
    save_solution: bool = False,
) -> tuple[int, float, int, list[int], dict[str, float], list[float]]:
    mesh_path = case_dir / "inline-quad.mesh"
    write_inline_quad(mesh_path, size)
    device_backend = DEVICES[device][SOLVERS[solver]["family"]]
    command = [
        "mpirun",
        "-np",
        str(mpi_ranks),
        str(executable),
        "-m",
        str(mesh_path),
        "-o",
        str(order),
        "-rs",
        "0",
        *SOLVERS[solver]["flags"],
        "-d",
        device_backend,
        "-no-vis",
        "-no-pv",
        "-out" if save_solution else "-no-out",
        "-l2",
        "-mms",
        mms,
        "-pr",
        str(repeats),
        *extra_flags,
    ]
    process = subprocess.run(command, cwd=case_dir, text=True, capture_output=True)
    output = process.stdout + process.stderr
    if process.returncode:
        raise RuntimeError(
            f"{device}/{solver}, order={order}, n={size} failed:\n{output}"
        )

    dofs = re.search(r"Number of finite element unknowns: (\d+)", process.stdout)
    error = re.search(r"L2 norm of error: ([0-9.eE+-]+)", process.stdout)
    warmup_iterations = re.search(r"^CG iterations: (\d+)$", process.stdout, re.MULTILINE)
    steady_iterations = [
        int(value)
        for value in re.findall(r"Steady-state CG iterations: (\d+)", process.stdout)
    ]
    labels = {
        "cold_start": "Cold start time",
        "assembly": "Operator assembly time",
        "setup": "Preconditioner setup time",
        "warmup": "Warm-up solve time",
    }
    timings = {}
    for key, label in labels.items():
        match = re.search(rf"{re.escape(label)}: ([0-9.eE+-]+)", process.stdout)
        if match:
            timings[key] = float(match.group(1))
    steady_times = [
        float(value)
        for value in re.findall(r"Steady-state solve time: ([0-9.eE+-]+)", process.stdout)
    ]
    if (
        not dofs
        or not error
        or not warmup_iterations
        or len(timings) != len(labels)
        or len(steady_times) != repeats
        or len(steady_iterations) != repeats
    ):
        raise RuntimeError(
            f"could not parse {device}/{solver}, order={order}, n={size}:\n{output}"
        )
    return (
        int(dofs.group(1)),
        float(error.group(1)),
        int(warmup_iterations.group(1)),
        steady_iterations,
        timings,
        steady_times,
    )


def profile_case(
    executable: Path,
    work_dir: Path,
    mpi_ranks: int,
    device: str,
    solver: str,
    mms: str,
    order: int,
    size: int,
    repeats: int,
    extra_flags: tuple[str, ...] = (),
    save_solution: bool = False,
) -> tuple[ProfileResult, list[tuple[int, float]]]:
    case_dir = work_dir / device / solver / f"order{order}" / f"n{size:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Profiling {DEVICES[device]['label']} / {SOLVERS[solver]['label']}, "
        f"p={order}, n={size} ...",
        flush=True,
    )

    dofs, l2_error, warmup_iterations, steady_iterations, timings, steady_times = run_once(
        executable, case_dir, mpi_ranks, device, solver, mms, order, size, repeats,
        extra_flags, save_solution,
    )
    steady_solve = statistics.median(steady_times)
    solve_total = timings["warmup"] + steady_solve
    full_total = (
        timings["cold_start"]
        + timings["assembly"]
        + timings["setup"]
        + timings["warmup"]
    )

    return (
        ProfileResult(
            device=device,
            solver=solver,
            mms=mms,
            order=order,
            n=size,
            h=1.0 / size,
            dofs=dofs,
            l2_error=l2_error,
            warmup_iterations=warmup_iterations,
            steady_iterations=round(statistics.median(steady_iterations)),
            cold_start_seconds=timings["cold_start"],
            assembly_seconds=timings["assembly"],
            setup_seconds=timings["setup"],
            warmup_solve_seconds=timings["warmup"],
            steady_solve_seconds=steady_solve,
            solve_total_seconds=solve_total,
            assembly_setup_solve_total_seconds=full_total,
        ),
        list(zip(steady_iterations, steady_times)),
    )


def write_csv(rows: list[object], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_plot(
    results: list[ProfileResult], path: Path, time_field: str, x_label: str
) -> None:
    # Fall back to matplotlib's mathtext when no system LaTeX is installed
    # (system where this runs may not have texlive).
    plt.rc("text", usetex=shutil.which("latex") is not None)
    plt.rc("font", family="Times New Roman")
    figure, axis = plt.subplots(figsize=(10, 7), dpi=80, facecolor="w")
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    solver_markers = {"ceed-amg": "o", "pa-jacobi": "s", "hypre-amg": "^"}
    device_linestyles = {"cpu": "-", "cuda": "--"}
    devices_present = [d for d in DEVICES if any(r.device == d for r in results)]

    for order, color in zip(DEFAULT_ORDERS, colors):
        for solver, marker in solver_markers.items():
            for device in devices_present:
                series = sorted(
                    (
                        result
                        for result in results
                        if result.order == order
                        and result.solver == solver
                        and result.device == device
                    ),
                    key=lambda result: result.l2_error,
                    reverse=True,
                )
                if not series:
                    continue
                axis.loglog(
                    [getattr(result, time_field) for result in series],
                    [result.l2_error for result in series],
                    linestyle=device_linestyles[device],
                    marker=marker,
                    color=color,
                    linewidth=1.5,
                    markersize=5,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                )

    order_handles = [
        Line2D([0], [0], color=color, linewidth=2, label=rf"$p={order}$")
        for order, color in zip(DEFAULT_ORDERS, colors)
    ]
    solver_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            marker=marker,
            markerfacecolor="none",
            label=SOLVERS[solver]["label"],
        )
        for solver, marker in solver_markers.items()
    ]
    device_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=device_linestyles[device],
            label=DEVICES[device]["label"],
        )
        for device in devices_present
    ]
    order_legend = axis.legend(
        handles=order_handles,
        title="Polynomial order",
        fontsize=10,
        title_fontsize=11,
        loc="lower left",
    )
    solver_legend = axis.legend(
        handles=solver_handles,
        title="Solver configuration",
        fontsize=10,
        title_fontsize=11,
        loc="upper right",
    )
    axis.add_artist(order_legend)
    if len(device_handles) > 1:
        axis.add_artist(solver_legend)
        axis.legend(
            handles=device_handles,
            title="Device",
            fontsize=10,
            title_fontsize=11,
            loc="lower right",
        )
    axis.set_xlabel(x_label, fontsize=14)
    axis.set_ylabel(r"$L^2\mathrm{ error}$", fontsize=14)
    axis.set_title(r"$\mathrm{Time-to-Accuracy Comparison}$", fontsize=17)
    axis.grid(which="major", color="0.75", linewidth=0.8)
    axis.grid(which="minor", color="0.88", linewidth=0.5, linestyle=":")
    axis.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=12,
        width=1.2,
    )
    figure.savefig(path, format="png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def detect_cuda() -> bool:
    """Return True when an NVIDIA GPU is visible via nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return False
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return probe.returncode == 0 and bool(probe.stdout.strip())


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", type=int, default=DEFAULT_ORDERS)
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=tuple(SOLVERS))
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=DEVICES,
        default=None,
        help="device backends to run (default: cpu+cuda when a CUDA GPU is "
             "detected via nvidia-smi, else cpu only)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--mms",
        choices=MMS_CHOICES,
        default="bubble-exp",
        help="manufactured solution (default: hardest non-eigenmode case)",
    )
    parser.add_argument("--np", dest="mpi_ranks", type=int, default=1)
    parser.add_argument("--executable", type=Path, default=repo / "examples" / "ex1p")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "profiling_and_error_results",
    )
    args = parser.parse_args()
    args.orders = tuple(sorted(set(args.orders)))
    args.sizes = tuple(sorted(set(args.sizes))) if args.sizes else None
    if args.devices is None:
        args.devices = ("cpu", "cuda") if detect_cuda() else ("cpu",)
    # Preserve user-specified device order so CPU-then-GPU prints intuitively.
    seen: set[str] = set()
    args.devices = tuple(d for d in args.devices if not (d in seen or seen.add(d)))
    if not args.orders or any(order not in DEFAULT_ORDERS for order in args.orders):
        parser.error("orders must be between 1 and 7")
    if args.sizes is not None and (not args.sizes or any(size <= 0 for size in args.sizes)):
        parser.error("mesh sizes must be positive")
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
    results = []
    raw_rows = []
    skipped: list[tuple[str, str, int, int, str]] = []
    with tempfile.TemporaryDirectory(prefix="mfem-profile-") as temporary_dir:
        work_dir = Path(temporary_dir)
        for device in args.devices:
            for solver in args.solvers:
                for order in args.orders:
                    sizes = args.sizes or DEFAULT_SIZES_BY_ORDER[order]
                    for size in sizes:
                        try:
                            result, samples = profile_case(
                                args.executable,
                                work_dir,
                                args.mpi_ranks,
                                device,
                                solver,
                                args.mms,
                                order,
                                size,
                                args.repeats,
                            )
                        except RuntimeError as failure:
                            reason = str(failure).splitlines()[0]
                            print(f"  SKIP {device}/{solver} p={order} n={size}: {reason}", flush=True)
                            skipped.append((device, solver, order, size, reason))
                            continue
                        results.append(result)
                        raw_rows.extend(
                            RawSample(
                                device, solver, args.mms, order, size,
                                run, iterations, elapsed,
                            )
                            for run, (iterations, elapsed) in enumerate(samples, 1)
                        )

    summary_path = args.output / "profiling_summary.csv"
    raw_path = args.output / "profiling_samples.csv"
    solve_plot_path = args.output / "solve_time_vs_error.png"
    full_plot_path = args.output / "assembly_setup_solve_time_vs_error.png"
    write_csv(results, summary_path)
    write_csv(raw_rows, raw_path)
    write_plot(
        results,
        solve_plot_path,
        "solve_total_seconds",
        r"$\mathrm{Warm\!-\!up\ solve + steady\!-\!state\ solve\ time\ (s)}$",
    )
    write_plot(
        results,
        full_plot_path,
        "assembly_setup_solve_total_seconds",
        r"$\mathrm{Cold\ start + assembly + setup + first\ solve\ (s)}$",
    )

    print("\nIndependent phase and aggregate times (seconds):")
    for result in results:
        print(
            f"{DEVICES[result.device]['label']:11s} {result.solver:11s} "
            f"p={result.order} n={result.n:3d} "
            f"cold={result.cold_start_seconds:.3e} "
            f"assembly={result.assembly_seconds:.3e} setup={result.setup_seconds:.3e} "
            f"warmup={result.warmup_solve_seconds:.3e} "
            f"steady={result.steady_solve_seconds:.3e} "
            f"error={result.l2_error:.6e}"
        )
    print(f"Summary:    {summary_path}")
    print(f"Samples:    {raw_path}")
    print(f"Solve plot: {solve_plot_path}")
    print(f"Full plot:  {full_plot_path}")
    if skipped:
        print(f"\nSkipped {len(skipped)} configuration(s):")
        for device, solver, order, size, reason in skipped:
            print(f"  {device}/{solver} p={order} n={size}: {reason}")


@dataclass(frozen=True)
class RawSample:
    device: str
    solver: str
    mms: str
    order: int
    n: int
    run: int
    iterations: int
    seconds: float


if __name__ == "__main__":
    main()
