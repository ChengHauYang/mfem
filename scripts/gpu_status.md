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

CPU/GPU rows are plotted on the same time-vs-error chart, distinguished by
line style (solid = CPU, dashed = CUDA) with solver marker and order colour
carried over from the CPU-only plots.

## Latest sweep

Output tree under `python_script/profiling_and_error_results/` (gitignored):

    profiling_summary.csv    205 rows: orders 1-7, sizes per DEFAULT_SIZES_BY_ORDER
    profiling_samples.csv    per-run raw timings (3 repeats each)
    time_vs_error.png        CPU vs GPU time-to-accuracy plot

Coverage:

| device | ceed-amg | pa-jacobi | hypre-amg |
|--------|----------|-----------|-----------|
| cpu    | 35       | 35        | 35        |
| cuda   | 30       | 35        | 35        |

Five `cuda / ceed-amg` cases at higher orders were skipped by the runner
(recorded to the summary block at end of run). Worth revisiting when time
permits — likely a libCEED AMG setup path that needs a bigger problem to be
worth the transfer, but not confirmed.

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
