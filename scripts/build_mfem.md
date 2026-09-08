# MFEM + hypre + libCEED build notes

Companion to `build_mfem.sh`. Layout assumed:

    Package/                 <- $ROOT
      hypre/                 cloned by the script
      libCEED/               cloned by the script
      mfem/                  upstream clone; work on branch `mfem-learn`
        CLAUDE.md            tracked on mfem-learn only
        .claude/skills/      tracked on mfem-learn only  (build_mfem skill)
        scripts/             tracked on mfem-learn only
          build_mfem.sh
          build_mfem.md      <- this file

These three paths are committed on the personal branch `mfem-learn` and exist
nowhere else. `mfem-learn` is a learning branch: it is never merged into
`master` and never pushed upstream. Switching to `master` makes the three paths
disappear from the working tree (expected -- git removes files the target branch
does not track); `git switch mfem-learn` brings them back.

Layout follows MOOSE's `scripts/` convention. Note it is deliberately not
`build-tooling/` or any `build-`-prefixed name: MFEM's own `.gitignore:483` has
`build-*/*` for VPATH builds and would silently swallow the whole directory.

The script resolves `$ROOT` two levels up from its own directory. Override with
`MFEM_ROOT=/path/to/root ./build_mfem.sh cpu` if the layout differs (e.g. on a
cluster where hypre/ and libCEED/ are not siblings of mfem/).

## Verified

CPU build verified 2026-08-21, Apple M4 Pro (8P+4E), macOS 25.6, Apple clang 21,
MPICH 4.3.2, hypre v2.32.0, libCEED v0.12.0, METIS 5.1.0 (brew).

    cd mfem/scripts && ./build_mfem.sh cpu

CUDA build verified 2026-08-22, `mom-01` (Ubuntu 22.04, 2x RTX A5000 sm_86,
driver 550.54.15 = CUDA 12.4), OpenMPI 4.1, hypre v2.32.0, libCEED v0.12.0,
METIS 5.1.0 + nvcc 12.9 from the miniforge `fenicsx` conda env, driver-matching
libnvrtc.so.12.4.127 from `/usr/local/cuda-12.4/lib64`.

    cd mfem/scripts
    source ./setup_cuda_env.sh                # env vars + shim dirs
    ./build_mfem.sh cuda                      # ~15 min from cold

Then, for anything that invokes `ex1p` on the GPU (including the profiling
script), re-source `setup_cuda_env.sh` once per shell:

    source ./setup_cuda_env.sh
    cd ../examples
    mpirun -np 1 ./ex1p -m ../data/inline-hex.mesh -o 4 -pa -a -d ceed-cuda -no-vis

## CUDA build on this machine: what the setup script is fixing

The `mom-01` box has an unusual toolchain layout that hit four distinct
failures during the first bring-up. `setup_cuda_env.sh` codifies the fixes;
this section explains why each one is there so the recipe can be adapted if
another cluster looks different.

1. **`nvcc` cannot find its helpers.** The fenicsx env keeps `nvcc` at
   `bin/nvcc` and the helpers (`cudafe++`, `cicc`, `ptxas`, `fatbinary`,
   `nvcc.profile`, `crt/link.stub`) also under `bin/`. Its
   `targets/x86_64-linux/bin/nvcc` is a relative symlink up to `bin/nvcc`;
   invoking that path leaves `_HERE_ = targets/x86_64-linux/bin/`, and
   nvcc.profile's `TOP` expansion never reaches the helpers. Symptom:
   `sh: 1: cudafe++: not found` mid-hypre-build. Fix: build a shim
   `CUDA_HOME=/tmp/pae_build/cuda_home/` whose `bin/` contains absolute
   symlinks to nvcc **and** every helper it launches, plus a `crt/` symlink.

2. **OpenMPI's `mpicxx` needs a bare `g++`.** Ubuntu 22.04 installs only
   `/usr/bin/g++-11`; there is no unversioned `g++`. OpenMPI's wrapper hard-
   codes the underlying compiler at build time and errors out with
   "Open MPI wrapper compiler was unable to find the specified compiler g++
   in your PATH." Fix: `OMPI_CXX=g++-11 OMPI_CC=gcc-11`, plus a `g++ ->
   g++-11` alias in a private `hostbin/` on PATH so anything that shells out
   to bare `g++` (nvcc's own device-link stage) also resolves.

3. **METIS_DIR must not expose the conda env's MPI libs.** Setting
   `METIS_DIR=$FE` puts `-L$FE/lib` on the link line, and `$FE/lib` also
   contains fenicsx's MPICH (`libmpi.so -> libmpi.so.12`). The linker resolves
   `-lmpi` from that first, shadowing the OpenMPI 4.x hypre was configured
   against. Symptom: `libmpi.so.40, needed by libmpi_cxx.so, may conflict
   with libmpi.so.12` followed by hundreds of `undefined reference to
   ompi_mpi_int/ompi_mpi_op_max/...`. Fix: build a shim
   `METIS_DIR=/tmp/pae_build/metis/` with **only** `libmetis.so` + `metis.h`
   symlinked, so `-L` cannot reach any MPI library.

4. **Driver-matching libnvrtc for runtime JIT.** The driver on this box is
   550.54.15 (CUDA 12.4); the fenicsx nvcc is 12.9, so the libnvrtc bundled
   with it emits PTX 8.5 which the driver rejects at runtime with
   `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` the first time libCEED JITs a kernel.
   Compiling MFEM/hypre works either way (they emit SASS for sm_86), but any
   `-d ceed-cuda` run dies inside `CeedCompile_Cuda`. Fix: prepend
   `/usr/local/cuda-12.4/lib64` to `LD_LIBRARY_PATH` so `dlopen("libnvrtc
   .so.12")` picks the 12.4 runtime the driver knows how to load. The
   compile-time nvcc stays 12.9; only the runtime JIT is downgraded.

The `config/config.hpp` that `make config` "loses" is not lost -- it is a
checked-in stub. Never delete it during a manual clean.

**Do not merge these fixes into `build_mfem.sh`.** That script is verified on
the Mac; changing it risks breaking the CPU path. Machine-specific shims live
in `setup_cuda_env.sh`, which is only sourced on Linux boxes that need it.

## Working run

    cd mfem/examples
    mpirun -np 8 ./ex1p -m ../data/inline-hex.mesh -o 4 -pa -a -d ceed-cpu -no-vis
    # 16,974,593 DOFs, reduction factor 0.216, ~2 min on 8 ranks

Exercises MPI + partial assembly + libCEED + algebraic p-multigrid + CG together.

## Gotchas found the hard way

**periodic-cube.mesh does not work with ex1p.** Periodicity makes every face
internal, so `MarkExternalBoundaries()` marks nothing and `ess_tdof_list` stays
empty (ex1p.cpp:207-217). The Poisson operator is then singular (constant
nullspace) while the RHS `(1,phi)` is not orthogonal to it -> inconsistent
system. PCG prints "The operator is not positive definite" and diverges to
~1e28. Reproduces at order 1 with only 110k DOFs, so it is not a size problem.
Use `inline-hex.mesh` or `fichera.mesh` instead.

**`-a` requires MFEM_USE_CEED.** The flag is only registered inside
`#ifdef MFEM_USE_CEED` (ex1p.cpp:108-112). Without libCEED it is not "ignored",
it is an unknown option.

**ex1p has no refinement CLI flag.** Serial refinement fills to <=10,000
elements, then `par_ref_levels = 2` hardcoded (ex1p.cpp:150,164). For
`inline-*.mesh` the resulting DOF count is `(nx * 2^4 * order + 1)^3`. To change
problem size, copy an `inline-*.mesh` out of the repo and edit `nx/ny/nz` --
those files are generator specs (`MFEM INLINE mesh v1.0`), not real meshes.

**METIS 5 needs `MFEM_USE_METIS_5=YES`.** MFEM defaults to METIS 4 at
`../metis-4.0`; brew ships METIS 5. Wrong flag = API signature mismatch.

**hypre install path lines up by luck.** `make install` from `hypre/src` writes
to `hypre/src/hypre/`, which is already MFEM's default `HYPRE_DIR`
(config/defaults.mk:257). Do not "fix" it to a prefix outside the tree.

**Two MPIs installed.** brew has both mpich 4.3.2 and open-mpi 5.0.7; mpich is
first on PATH. hypre and MFEM must use the same one or linking breaks. Check
with `mpicxx -show`.

**No `timeout(1)` on macOS.** Use `gtimeout` (coreutils) or just background it.

## No GPU on this Mac

MFEM has no Metal/MPS backend. Backend list is CUDA/HIP/OMP/CPU plus
OCCA/RAJA/CEED variants (general/device.cpp:58-64). `-d mps` fails with
"Invalid backend name: 'mps'" (device.cpp:241). libCEED has no Metal backend,
and MFEM's `OccaDeviceSetup()` only wires CUDA/OpenMP/Serial (device.cpp:479-509)
even though upstream OCCA has a Metal mode. There is no route to the M4 GPU.

OpenMP was left OFF: Apple clang needs explicit libomp paths that MFEM's
`OPENMP_OPT = -fopenmp` default does not supply. Add it only if `-d omp` is
wanted.

## CUDA unknowns (to resolve on first cluster build)

- `CUDA_ARCH` must be set (sm_80 A100 / sm_90 H100 / sm_70 V100 / sm_89 L40S).
- hypre `--enable-unified-memory` is on; fine on A100/H100, may cost
  performance on older cards. Drop it and re-measure if needed.
- Confirm libCEED actually built `/gpu/cuda/*` backends, not just CPU ones.
- NVIDIA MPS (multi-rank sharing one GPU) is orthogonal to all of the above and
  needs no MFEM changes:
      nvidia-cuda-mps-control -d
      mpirun -np 8 ./ex1p ... -d ceed-cuda
      echo quit | nvidia-cuda-mps-control
  Only worth it if per-rank GPU utilisation is low; `-pa` already saturates the
  device, so measure `-np 1` against `-np 8 + MPS` before committing.
