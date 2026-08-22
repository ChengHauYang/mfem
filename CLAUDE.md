# MFEM — local working notes

Upstream clone, built **in-source** with MPI + METIS 5 + libCEED (CPU).

**Branch `mfem-learn` is the working branch.** It is a personal learning branch:
`scripts/`, `.claude/` and `CLAUDE.md` are tracked there and nowhere else, it is
never merged into `master`, and it is never pushed upstream. Check
`git branch --show-current` before committing.

## Building

Use the `build_mfem` skill, or directly:

    cd scripts && ./build_mfem.sh cpu               # verified 2026-08-21
    CUDA_ARCH=sm_80 ./build_mfem.sh cuda           # written, NEVER RUN

The script is idempotent and operates on the *parent* of this repo, where
`hypre/` and `libCEED/` sit alongside it. Read `scripts/build_mfem.md`
before debugging anything — the failure modes already hit are listed there with
the file:line that causes them.

## Two things that will bite

**No GPU on this Mac.** MFEM has no Metal/MPS backend (`general/device.cpp:58-64`);
`-d mps` fails with "Invalid backend name". The M4 Pro GPU is unreachable from
MFEM. GPU runs need the remote CUDA cluster, and that path is unverified.

**`data/periodic-cube.mesh` does not work with ex1p.** Periodicity leaves
`ess_tdof_list` empty (`examples/ex1p.cpp:207-217`), making the Poisson operator
singular; PCG diverges to ~1e28 even at order 1. Use `data/inline-hex.mesh`
(a generator spec — edit `nx/ny/nz` in a copy to resize) or `data/fichera.mesh`.

## Known-good run

    cd examples
    mpirun -np 8 ./ex1p -m ../data/inline-hex.mesh -o 4 -pa -a -d ceed-cpu -no-vis
    # 16,974,593 DOFs, Average reduction factor 0.216, ~2 min on 8 ranks

## Visualization

Use the `visualize_glvis` skill, or install GLVis directly:

    cd scripts && ./install_glvis.sh
    ../../bin/glvis -np 8 -m ../examples/mesh -g ../examples/sol

The installer builds against this MFEM configuration and installs the executable
at the package root's `bin/glvis`.
