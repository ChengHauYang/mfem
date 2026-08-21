---
name: build_mfem
description: >
  Build MFEM with MPI + libCEED (+ CUDA) and debug until it compiles and ex1p
  runs. Use when the user wants to build/rebuild MFEM, configure it for GPU or
  libCEED, port the working CPU build to a CUDA cluster, or when an MFEM build
  or an ex1p/example run is failing. Wraps a verified build script and a list
  of failure modes already hit once.
user-invocable: true
---

# Build MFEM

Do not re-derive the build. It is captured in two files in this repo:

- `scripts/build_mfem.sh` — the build
- `scripts/build_mfem.md` — gotchas and unknowns

**Read `build_mfem.md` before doing anything.** Most failures in this stack are
already listed there with the file:line that causes them.

## Procedure

1. `cd <mfem repo>/scripts`
2. `./build_mfem.sh cpu` (verified) or `CUDA_ARCH=sm_80 ./build_mfem.sh cuda`
   (unverified — expect to debug). The script operates on the directory
   *containing* this repo (where hypre/ and libCEED/ live), not on the repo
   itself; override with `MFEM_ROOT=`.
3. The script skips any component already built. To force one, delete its
   artifact (`hypre/src/hypre/lib/libHYPRE.a`, `libCEED/lib/libceed.*`).
4. Verify with a run that exercises the whole stack:
   ```
   cd ../examples
   mpirun -np 8 ./ex1p -m ../data/inline-hex.mesh -o 4 -pa -a -d ceed-cpu -no-vis
   ```
   Expect ~17M DOFs and `Average reduction factor` around 0.216. A reduction
   factor near 1.0, or `PCG: The operator is not positive definite`, means the
   *problem* is wrong, not the build — check the mesh (see below).

## Debug loop

Iterate until it compiles and the verification run converges. When something
fails, check in this order:

1. Is it in `scripts/build_mfem.md`? Apply that fix.
2. `mpicxx -show` — is the same MPI used for hypre and MFEM?
3. `cd mfem && make status` — are `MFEM_USE_MPI/METIS_5/CEED/CUDA` what you
   expect? A stale `config/config.mk` silently keeps old settings; `make config`
   again with the full flag list, never a partial one.
4. Only then read the actual compiler error.

Record any new failure mode in `scripts/build_mfem.md`. That file is the deliverable of
debugging, not just the working binary.

## Hard constraints

- **No GPU on the local Mac.** MFEM has no Metal/MPS backend at all. Never
  suggest `-d mps`, `-d ceed-cuda`, or `-d gpu` on this machine;
  `scripts/build_mfem.md` has the file:line proof. GPU work needs the remote CUDA cluster.
- **Work on branch `mfem-learn`, never on `master`.** This is the user's personal
  learning branch: `scripts/`, `.claude/` and `CLAUDE.md` are tracked there and
  nowhere else. It is never merged into `master` and never pushed upstream. If
  `git branch --show-current` says `master`, switch before changing anything.
  Keep new tooling inside those three paths.
- **Never name a directory `build-*`.** MFEM's `.gitignore:483` (`build-*/*`,
  for VPATH builds) silently swallows it. New scripts go in `scripts/`, MOOSE
  style.
- **`../data/periodic-cube.mesh` does not work with ex1p** — singular system.
  Use `inline-hex.mesh` (edit `nx/ny/nz` in a copy to change size) or
  `fichera.mesh`.
