# Run GPU profiling

This guide runs the MFEM CPU/CUDA time-to-accuracy study on a GPU machine. The
commands below assume the repository is on branch `mfem-learn` and that hypre
and libCEED are sibling directories of `mfem`.

## 1. Update the source

On the GPU machine, update the working tree so it includes the profiling changes
in `examples/ex1p.cpp` and `python_script/profiling_and_error.py`.

Verify the branch and working tree before building:

```bash
git branch --show-current
git status --short
```

The branch should be `mfem-learn`. Do not reuse an older `examples/ex1p`
executable because it will not recognize the new `-pr/--profile-repeats`
option.

## 2. Prepare the CUDA environment

The verified `mom-01` setup uses two RTX A5000 GPUs (`sm_86`), CUDA 12.4 at
runtime, and the `fenicsx` Conda environment for the CUDA compiler:

```bash
source scripts/setup_cuda_env.sh
nvidia-smi
```

For another machine, set the overrides before sourcing the script:

```bash
export MFEM_CUDA_ENV=fenicsx
export MFEM_CUDA_TOOLKIT=/usr/local/cuda-12.4
export MFEM_CUDA_ARCH=sm_86
source scripts/setup_cuda_env.sh
```

Change `MFEM_CUDA_TOOLKIT` and `MFEM_CUDA_ARCH` to match that machine. See
`scripts/build_mfem.md` and `scripts/gpu_status.md` for the verified toolchain
and the reasons for the CUDA/METIS shim directories.

## 3. Rebuild the modified example

If MFEM is already configured for CUDA, rebuilding `ex1p` is sufficient:

```bash
make -C examples ex1p
```

For a new checkout or an incompatible existing configuration, rebuild MFEM and
its dependencies:

```bash
cd scripts
./build_mfem.sh cuda
cd ..
```

## 4. Verify the profiling output

Run one small CUDA case before starting a sweep:

```bash
python3 python_script/profiling_and_error.py \
  --devices cuda \
  --solvers ceed-amg \
  --orders 3 \
  --sizes 16 \
  --repeats 3 \
  --np 1 \
  --output python_script/profiling_gpu_smoke
```

A successful run prints independent values for:

```text
cold
assembly
setup
warmup
steady
```

It also creates:

```text
python_script/profiling_gpu_smoke/profiling_summary_linux.csv
python_script/profiling_gpu_smoke/profiling_samples_linux.csv
python_script/profiling_gpu_smoke/solve_time_vs_error_linux.png
python_script/profiling_gpu_smoke/assembly_setup_solve_time_vs_error_linux.png
```

Every CSV and PNG the python drivers write carries a `_linux` / `_mac` tag just
before the extension, so results from the cluster and from a laptop can sit in
the same directory without overwriting each other.

The first solve and all steady-state repeats execute in the same `ex1p`
process. They reuse the same operator and preconditioner. This is required for
the steady-state samples to represent a warmed CUDA execution rather than a
sequence of process cold starts.

## 5. Run a CPU/CUDA comparison

Use one MPI rank for the initial comparison so one GPU is used and MPI scaling
does not obscure the device comparison:

```bash
python3 python_script/profiling_and_error.py \
  --devices cpu cuda \
  --solvers ceed-amg pa-jacobi hypre-amg \
  --orders 1 2 3 4 \
  --sizes 8 16 32 \
  --repeats 5 \
  --np 1 \
  --output python_script/profiling_cpu_cuda_check
```

This explicit `--sizes` list applies every listed size to every listed order.
Omit `--sizes` for the order-specific defaults used by the full study.

## 6. Run the full sweep

```bash
python3 python_script/profiling_and_error.py \
  --devices cpu cuda \
  --repeats 5 \
  --np 1
```

By default, results are written to:

```text
python_script/profiling_and_error_results/
```

Failed configurations are reported as `SKIP` and listed again at the end. A
skipped case should be investigated; it is not included in the CSV or plots.

## Timing definitions

The summary CSV stores non-overlapping measured phases:

| Column | Definition |
| --- | --- |
| `cold_start_seconds` | Hypre and MFEM device initialization after `MPI_Init` |
| `assembly_seconds` | Bilinear-form assembly and `FormLinearSystem` |
| `setup_seconds` | Preconditioner and CG setup |
| `warmup_solve_seconds` | First solve in the process, including lazy/JIT work triggered by it |
| `steady_solve_seconds` | Median of the additional same-process solves |

The plots add these independent phases as follows:

```text
solve_total_seconds
  = warmup_solve_seconds
  + steady_solve_seconds
```

```text
assembly_setup_solve_total_seconds
  = cold_start_seconds
  + assembly_seconds
  + setup_seconds
  + warmup_solve_seconds
  + steady_solve_seconds
```

Therefore, `solve_time_vs_error.png` represents one first solve plus one
representative repeated solve. `assembly_setup_solve_time_vs_error.png` adds the
cold-start, assembly, and setup phases to that value. The individual columns
remain available in `profiling_summary_<platform>.csv` so their costs can be compared
without relying only on the aggregate plots.

`profiling_samples.csv` contains every steady-state repeat and its CG iteration
count. The warm-up and steady-state iteration counts are also summarized in
`profiling_summary_<platform>.csv`.

## Multi-GPU runs

On the verified two-GPU host, a two-rank run can be launched with:

```bash
python3 python_script/profiling_and_error.py \
  --devices cuda \
  --repeats 5 \
  --np 2 \
  --output python_script/profiling_cuda_np2
```

Confirm that the MPI launcher assigns one rank per GPU. If the site does not do
that automatically, use its documented rank-to-GPU binding mechanism. Do not
compare `--np 1` and `--np 2` as a pure GPU-kernel benchmark: the latter also
includes inter-rank communication and CG global reductions.

## Troubleshooting

If `ex1p` reports an unknown `-pr` option, rebuild `examples/ex1p` from the
updated source.

If libCEED fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, source
`scripts/setup_cuda_env.sh` again and verify that `LD_LIBRARY_PATH` selects a
`libnvrtc.so` compatible with the installed NVIDIA driver.

If CUDA is slower for small cases, inspect the independent phase columns. CUDA
context initialization, libCEED JIT compilation, allocations, kernel launches,
and synchronization can dominate small problems. Compare sufficiently large
DOF counts before drawing a throughput conclusion.

The CUDA CEED AMG path does not use the CPU-only assembled Hypre coarse solver.
Compare `warmup_iterations` and `steady_iterations` between CPU and CUDA when
interpreting CEED AMG results, because a weaker CUDA coarse-level treatment can
increase solve time even when individual GPU kernels are fast.
