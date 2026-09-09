# GPU status — mom-01

Snapshot of the CUDA build + profiling workflow on the remote Linux box, as of
2026-09-08 (phased-profiler sweep). Companion to `build_mfem.md` (build
details) and the header of `setup_cuda_env.sh` (shim rationale).

## Machine

- Host: `mom-01`, Ubuntu 22.04
- GPUs: 2x NVIDIA RTX A5000 (`sm_86`)
- Driver: 550.54.15 (matches CUDA 12.4 runtime)
- MPI: OpenMPI 4.1 at `/usr/bin/mpicxx`
- Compile-time nvcc: 12.9 from miniforge `fenicsx` env
- Runtime libnvrtc: 12.4 from `/usr/local/cuda-12.4/lib64` (driver-matched;
  see `setup_cuda_env.sh` for why we split compile vs runtime CUDA)

## Build

Verified **2026-08-22** with hypre v2.32.0, libCEED v0.12.0, METIS 5.1.0.

    cd mfem/scripts
    source ./setup_cuda_env.sh      # env + shim dirs on /tmp/pae_build
    ./build_mfem.sh cuda            # ~15 min from cold

`build_mfem.md` walks through the four failure modes the shim script papers
over (nvcc helper resolution, bare `g++`, METIS `-L` isolation, PTX version
mismatch). Do not merge those shims into `build_mfem.sh`; the CPU path on the
Mac would break.

## Running ex1p on GPU

Re-source the env once per shell before any `-d ceed-cuda` run:

    source scripts/setup_cuda_env.sh
    cd examples
    mpirun -np 1 ./ex1p -m ../data/inline-hex.mesh -o 4 -pa -a -d ceed-cuda -no-vis

`mpirun -np 2` also works (one rank per A5000), but the profiling sweep below
was collected with `-np 1` to keep the two-GPU box available for other users.

## Profiling

`python_script/profiling_and_error.py` now takes a `--devices` flag and
auto-selects `cpu+cuda` when `nvidia-smi` reports a GPU:

    source scripts/setup_cuda_env.sh
    python python_script/profiling_and_error.py                # cpu + cuda
    python python_script/profiling_and_error.py --devices cuda # gpu only

CPU/GPU rows are plotted on the same time-vs-error charts, distinguished by
line style (solid = CPU, dashed = CUDA) with solver marker and order colour
carried over from the CPU-only plots. Each case now runs once and profiles
non-overlapping cold-start, operator-assembly, preconditioner-setup, first-solve,
and repeated-solve phases in the same process. The outputs are:

- `solve_time_vs_error.png`: first-solve time plus median repeated-solve time.
- `assembly_setup_solve_time_vs_error.png`: cold start, assembly, setup,
  first solve, and median repeated solve added together.

The summary CSV retains every independent phase as well as both aggregate
values, so the components can be inspected without inferring them from a plot.

## Latest sweep

Output tree under `python_script/profiling_and_error_results/` (gitignored):

    profiling_summary_linux.csv           independent phases and aggregate times
    profiling_samples_linux.csv           same-process steady-solve samples
    solve_time_vs_error_linux.png         first + median repeated solve
    assembly_setup_solve_time_vs_error_linux.png all profiled phases added together

Coverage:

| device | ceed-amg | pa-jacobi | hypre-amg |
|--------|----------|-----------|-----------|
| cpu    | 35       | 35        | 35        |
| cuda   | 30       | 35        | 35        |

Five `cuda / ceed-amg` cases were skipped by the runner: `p=4 n=2`,
`p=4 n=4`, `p=5 n=4`, `p=6 n=4`, `p=7 n=4` — a cluster at `n=4` for
`p >= 4`, not the "higher orders" pattern I first assumed. Not yet diagnosed.

## Timing snapshot (warm-up + steady solve, seconds)

Values are `solve_total_seconds = warmup_solve_seconds + steady_solve_seconds`
from the phased profiler: one first solve plus the median of five same-process
repeats. That is exactly what `solve_time_vs_error.png` charts, so a row here
matches the point plotted for that (device, solver, order, n). `n` is elements
per side of the 2D inline-quad mesh, DOFs is `p*n+1` squared. L2 error is
manufactured-solution error and is device-independent (CPU/GPU rows share it).

Read across the row for a size to compare preconditioners; read down a
column for weak/strong scaling on that (solver, device) pair.

Rules of thumb visible in the numbers below (solve only; assembly and setup
are separate columns in `profiling_summary_<platform>.csv`):
- CUDA carries a per-case overhead of ~0.03-0.1s from context init and
  kernel launch, so CPU still wins the smallest cases outright.
- `pa-jacobi cuda` is the decisive crossover once DOFs are moderate:
  `p=1 n=256` is 12.4s CPU vs 0.116s CUDA (~107x), `p=3 n=128` is 21.6s
  vs 0.338s (~64x).
- `ceed-amg cuda` no longer plateaus at ~1.3-2s the way the pre-phased
  profiler suggested; the solve alone is 0.09-0.32s at the tested sizes.
  Most of what the old snapshot attributed to "solve" was AMG setup, which
  the new profiler charges to `setup_seconds`.
- `hypre-amg cuda` bottoms out around 0.02-0.05s, so CPU wins on the small
  cases and the two devices converge on the largest ones.

### Order p=1

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 8   | 81    | 2.62e-03 | 0.003 | 0.003 | 0.002 | 0.003 | 0.002 | 0.050 |
| 16  | 289   | 6.57e-04 | 0.007 | 0.004 | 0.010 | 0.005 | 0.004 | 0.060 |
| 32  | 1089  | 1.64e-04 | 0.015 | 0.008 | 0.036 | 0.009 | 0.009 | 0.050 |
| 64  | 4225  | 4.11e-05 | 0.037 | 0.016 | 0.197 | 0.018 | 0.017 | 0.048 |
| 128 | 16641 | 1.03e-05 | 0.146 | 0.037 | 1.551 | 0.040 | 0.056 | 0.057 |
| 256 | 66049 | 2.57e-06 | 0.581 | 0.112 | 12.444 | 0.116 | 0.236 | 0.080 |

### Order p=2

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 4   | 81    | 4.86e-04 | 0.003 | 0.066 | 0.001 | 0.003 | 0.002 | 0.060 |
| 8   | 289   | 6.15e-05 | 0.007 | 0.043 | 0.006 | 0.006 | 0.005 | 0.038 |
| 16  | 1089  | 7.71e-06 | 0.021 | 0.047 | 0.027 | 0.012 | 0.012 | 0.058 |
| 32  | 4225  | 9.64e-07 | 0.063 | 0.058 | 0.123 | 0.024 | 0.025 | 0.050 |
| 64  | 16641 | 1.21e-07 | 0.251 | 0.099 | 0.915 | 0.061 | 0.081 | 0.069 |
| 128 | 66049 | 1.51e-08 | 1.071 | 0.297 | 7.303 | 0.214 | 0.332 | 0.106 |

### Order p=3

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 4   | 169    | 1.33e-05 | 0.005 | 0.044 | 0.003 | 0.005 | 0.004 | 0.035 |
| 8   | 625    | 8.44e-07 | 0.012 | 0.046 | 0.014 | 0.010 | 0.012 | 0.048 |
| 16  | 2401   | 5.29e-08 | 0.030 | 0.049 | 0.057 | 0.020 | 0.026 | 0.060 |
| 32  | 9409   | 3.31e-09 | 0.105 | 0.061 | 0.341 | 0.040 | 0.084 | 0.076 |
| 64  | 37249  | 2.07e-10 | 0.442 | 0.100 | 2.681 | 0.096 | 0.337 | 0.098 |
| 128 | 148225 | 1.29e-11 | 1.888 | 0.315 | 21.646 | 0.338 | 1.710 | 0.176 |

### Order p=4

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 2  | 81    | 8.09e-06 | 0.004 | —     | 0.002 | 0.003 | 0.003 | 0.024 |
| 4  | 289   | 2.62e-07 | 0.008 | —     | 0.006 | 0.007 | 0.008 | 0.035 |
| 8  | 1089  | 8.27e-09 | 0.018 | 0.107 | 0.028 | 0.015 | 0.018 | 0.050 |
| 16 | 4225  | 2.59e-10 | 0.051 | 0.110 | 0.121 | 0.029 | 0.041 | 0.059 |
| 32 | 16641 | 8.10e-12 | 0.238 | 0.127 | 0.879 | 0.065 | 0.159 | 0.083 |

### Order p=5

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 2  | 121  | 2.46e-07 | 0.005 | 0.390 | 0.003 | 0.004 | 0.005 | 0.037 |
| 4  | 441  | 3.96e-09 | 0.011 | —     | 0.011 | 0.010 | 0.013 | 0.047 |
| 8  | 1681 | 6.23e-11 | 0.024 | 0.107 | 0.042 | 0.020 | 0.029 | 0.052 |
| 16 | 6561 | 9.76e-13 | 0.073 | 0.103 | 0.225 | 0.041 | 0.104 | 0.063 |

### Order p=6

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 1 | 49   | 6.96e-07 | 0.006 | 0.102 | 0.001 | 0.002 | 0.003 | 0.020 |
| 2 | 169  | 6.02e-09 | 0.006 | 0.133 | 0.004 | 0.006 | 0.007 | 0.040 |
| 4 | 625  | 4.84e-11 | 0.013 | —     | 0.015 | 0.012 | 0.019 | 0.046 |
| 8 | 2401 | 3.80e-13 | 0.029 | 0.093 | 0.062 | 0.025 | 0.045 | 0.063 |

### Order p=7

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 1 | 64   | 2.88e-08 | 0.006 | 0.128 | 0.002 | 0.003 | 0.003 | 0.027 |
| 2 | 225  | 1.24e-10 | 0.008 | 0.481 | 0.006 | 0.007 | 0.010 | 0.036 |
| 4 | 841  | 4.96e-13 | 0.016 | —     | 0.020 | 0.015 | 0.025 | 0.049 |
| 8 | 3249 | 2.58e-15 | 0.038 | 0.098 | 0.090 | 0.031 | 0.085 | 0.067 |

## Known caveats

- `data/periodic-cube.mesh` is still unusable with `ex1p` on either device
  (empty `ess_tdof_list`, PCG diverges) — see CLAUDE.md.
- `usetex` in the plot now falls back to matplotlib's mathtext when no system
  LaTeX is on `PATH`; mom-01 does not have `texlive` installed, so plots use
  mathtext there.
- `LD_LIBRARY_PATH` order matters: the setup script prepends CUDA 12.4 so
  the driver-matching `libnvrtc.so.12` wins over the fenicsx env's 12.9
  copy. If a run dies inside `CeedCompile_Cuda` with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, re-source the setup script.

## Not done yet

- Multi-rank GPU sweep (`-np 2`, one per A5000) for the profiling script.
- Investigation of the skipped `cuda / ceed-amg` cases.
- Cross-machine comparison (Mac CPU vs mom-01 GPU) on the same figure.
