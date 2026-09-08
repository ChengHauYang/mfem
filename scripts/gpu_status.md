# GPU status — mom-01

Snapshot of the CUDA build + profiling workflow on the remote Linux box, as of
2026-09-08. Companion to `build_mfem.md` (build details) and the header of
`setup_cuda_env.sh` (shim rationale).

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

    profiling_summary.csv                 independent phases and aggregate times
    profiling_samples.csv                 same-process steady-solve samples
    solve_time_vs_error.png               first + median repeated solve
    assembly_setup_solve_time_vs_error.png all profiled phases added together

Coverage:

| device | ceed-amg | pa-jacobi | hypre-amg |
|--------|----------|-----------|-----------|
| cpu    | 35       | 35        | 35        |
| cuda   | 30       | 35        | 35        |

Five `cuda / ceed-amg` cases were skipped by the runner: `p=4 n=2`,
`p=4 n=4`, `p=5 n=4`, `p=6 n=4`, `p=7 n=4` — a cluster at `n=4` for
`p >= 4`, not the "higher orders" pattern I first assumed. Not yet diagnosed.

## Legacy timing snapshot (setup + first solve, seconds)

These values predate the independent-phase profiler above and should not be
compared directly with its new aggregate columns. Three separate process runs
were used for each median. `n` is elements per side of the 2D
inline-quad mesh, DOFs is `p*n+1` squared. L2 error is the manufactured
solution error and is device-independent (rows share it across CPU/GPU).

Read across the row for a size to compare preconditioners; read down a
column for weak/strong scaling on that (solver, device) pair.

Rules of thumb visible in the numbers below:
- CUDA carries a fixed per-case overhead of ~0.1-0.7s (kernel JIT,
  transfers), so CPU wins at small `n`.
- `pa-jacobi cuda` crosses over decisively at moderate DOFs — at `p=3
  n=128` it is 10.7s CPU vs 0.30s CUDA (~36x).
- `ceed-amg cuda` sits on a flat plateau (~1.3-2.2s) at high orders
  regardless of `n`, i.e. AMG setup dominates and the solve is free.

### Order p=1

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 8   | 81    | 2.62e-03 | 0.002 | 0.122 | 0.001 | 0.120 | 0.002 | 0.030 |
| 16  | 289   | 6.57e-04 | 0.006 | 0.123 | 0.006 | 0.123 | 0.003 | 0.061 |
| 32  | 1089  | 1.64e-04 | 0.015 | 0.124 | 0.026 | 0.124 | 0.007 | 0.066 |
| 64  | 4225  | 4.11e-05 | 0.028 | 0.131 | 0.102 | 0.129 | 0.011 | 0.081 |
| 128 | 16641 | 1.03e-05 | 0.097 | 0.141 | 0.778 | 0.140 | 0.033 | 0.107 |
| 256 | 66049 | 2.57e-06 | 0.387 | 0.187 | 6.215 | 0.184 | 0.134 | 0.124 |

### Order p=2

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 4   | 81    | 4.86e-04 | 0.003 | 0.668 | 0.001 | 0.126 | 0.002 | 0.043 |
| 8   | 289   | 6.15e-05 | 0.008 | 0.674 | 0.004 | 0.128 | 0.004 | 0.062 |
| 16  | 1089  | 7.71e-06 | 0.021 | 0.672 | 0.020 | 0.130 | 0.009 | 0.068 |
| 32  | 4225  | 9.64e-07 | 0.046 | 0.679 | 0.068 | 0.135 | 0.015 | 0.086 |
| 64  | 16641 | 1.21e-07 | 0.151 | 0.703 | 0.456 | 0.156 | 0.045 | 0.106 |
| 128 | 66049 | 1.51e-08 | 0.634 | 0.817 | 3.650 | 0.236 | 0.184 | 0.156 |

### Order p=3

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 4   | 169    | 1.33e-05 | 0.005 | 0.684 | 0.002  | 0.132 | 0.003 | 0.046 |
| 8   | 625    | 8.44e-07 | 0.011 | 0.691 | 0.010  | 0.134 | 0.009 | 0.071 |
| 16  | 2401   | 5.29e-08 | 0.028 | 0.688 | 0.037  | 0.138 | 0.018 | 0.090 |
| 32  | 9409   | 3.31e-09 | 0.070 | 0.698 | 0.176  | 0.149 | 0.045 | 0.112 |
| 64  | 37249  | 2.07e-10 | 0.260 | 0.719 | 1.344  | 0.177 | 0.181 | 0.147 |
| 128 | 148225 | 1.29e-11 | 1.125 | 0.856 | 10.747 | 0.301 | 0.918 | 0.220 |

### Order p=4

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 2  | 81    | 8.09e-06 | 0.004 | —     | 0.001 | 0.147 | 0.002 | 0.044 |
| 4  | 289   | 2.62e-07 | 0.008 | —     | 0.005 | 0.150 | 0.006 | 0.067 |
| 8  | 1089  | 8.27e-09 | 0.019 | 1.319 | 0.020 | 0.152 | 0.014 | 0.082 |
| 16 | 4225  | 2.59e-10 | 0.044 | 1.320 | 0.069 | 0.160 | 0.023 | 0.105 |
| 32 | 16641 | 8.10e-12 | 0.148 | 1.335 | 0.443 | 0.179 | 0.086 | 0.134 |

### Order p=5

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 2  | 121  | 2.46e-07 | 0.006 | 2.209 | 0.002 | 0.349 | 0.003 | 0.064 |
| 4  | 441  | 3.96e-09 | 0.011 | —     | 0.007 | 0.351 | 0.009 | 0.081 |
| 8  | 1681 | 6.23e-11 | 0.025 | 1.930 | 0.030 | 0.357 | 0.018 | 0.086 |
| 16 | 6561 | 9.76e-13 | 0.057 | 1.929 | 0.120 | 0.367 | 0.056 | 0.107 |

### Order p=6

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 1 | 49   | 6.96e-07 | 0.005 | 1.298 | 0.001 | 0.135 | 0.002 | 0.028 |
| 2 | 169  | 6.02e-09 | 0.007 | 1.336 | 0.003 | 0.137 | 0.005 | 0.067 |
| 4 | 625  | 4.84e-11 | 0.014 | —     | 0.012 | 0.141 | 0.013 | 0.083 |
| 8 | 2401 | 3.80e-13 | 0.030 | 1.293 | 0.041 | 0.147 | 0.025 | 0.106 |

### Order p=7

| n | DOFs | L2 error | ceed-amg cpu | ceed-amg cuda | pa-jacobi cpu | pa-jacobi cuda | hypre-amg cpu | hypre-amg cuda |
|---|---|---|---|---|---|---|---|---|
| 1 | 64   | 2.88e-08 | 0.007 | 1.342 | 0.002 | 0.142 | 0.003 | 0.047 |
| 2 | 225  | 1.24e-10 | 0.009 | 1.704 | 0.004 | 0.142 | 0.007 | 0.063 |
| 4 | 841  | 4.96e-13 | 0.017 | —     | 0.018 | 0.146 | 0.015 | 0.087 |
| 8 | 3249 | 2.58e-15 | 0.036 | 1.316 | 0.055 | 0.153 | 0.046 | 0.106 |

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
