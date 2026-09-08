# GPU showcase — interpretation

Four cases, CPU vs CUDA on `mom-01` (2× RTX A5000, `sm_86`, one rank). Data
in `gpu_showcase.csv`; see `python_script/gpu_showcase.py` for the runner
and the phased profiler it wraps. Times are seconds; "speedup" is CPU / CUDA.

## Cases at a glance

| Case                    |     DOFs |   iters | steady solve (CPU→CUDA) | full pipeline (CPU→CUDA) |
|-------------------------|---------:|--------:|-------------------------|--------------------------|
| p1 PA Jacobi (n=512)    |  263,169 |   1,374 | 50.42s → 0.26s (**194×**) | 101.18s → 1.25s (**81×**) |
| p6 PA Jacobi (n=64)     |  148,225 |   1,907 | 12.42s → 0.27s (**47×**)  | 24.93s → 1.25s (**20×**)  |
| p6 CEED AMG (n=64)      |  148,225 | 18 / 72 | 1.12s → 0.11s (**10×**)   | 2.49s → 2.14s (**1.2×**)  |
| p6 Hypre BoomerAMG (n=64)| 148,225 |   23–25 | 1.82s → 0.060s (**30×**)  | 4.88s → 1.55s (**3.1×**)  |

"Full pipeline" is `cold_start + assembly + setup + warmup_solve + steady_solve`.
CEED AMG iterations differ by device — see the CEED AMG note below.

## What the numbers say

**PA Jacobi is where the GPU dominates.** With no preconditioner setup and
thousands of cheap CG iterations, per-iteration kernel throughput is all that
matters. Steady-state speedups of 194× (p=1) and 47× (p=6) fall out directly.
The full-pipeline speedups drop to 81× and 20× because CUDA cold start
(~0.1s) and first-touch assembly (~0.4–0.5s) are non-negligible next to a
solve that has been sped up so aggressively.

**Hypre BoomerAMG wins in solve, but assembly caps the full-pipeline gain.**
Iteration counts match across devices (23 CUDA vs 25 CPU), the steady solve
is 30× faster on the GPU (0.060s vs 1.82s), but the ~1s assembly is the same
on both devices, so the full pipeline settles at 3.1×.

**CEED AMG barely wins on the full pipeline.** Two effects work against
CUDA here:

- The CUDA coarse-level treatment is weaker. Steady iteration count is
  72 on CUDA vs 18 on CPU — 4× more work per solve. The GPU still finishes
  the solve 10× faster in wall clock, but not by as much as its raw kernel
  throughput would suggest.
- AMG setup is 6× more expensive on CUDA (1.27s vs 0.23s). Setup dominates
  the full pipeline at this problem size, so the 10× solve speedup collapses
  to 1.2× when everything is added.

**Cold start is the same ~0.1s tax on every CUDA run.** It's the fixed cost
of `Device::Configure("ceed-cuda")` (context init, libCEED JIT primer).
Amortized across many solves it disappears; in a one-shot process it shows
up as the difference between the "steady solve" and "full pipeline"
columns.

## When the GPU is the right choice

- **Iterative solvers with many kernel-bound iterations** (unpreconditioned
  or diagonal-preconditioned CG on smooth problems). PA Jacobi at 1900
  iterations is the archetype.
- **Cases where you'll do many solves in one process** (nonlinear iterations,
  time stepping, parameter sweeps) so the ~0.1s cold-start and the first-solve
  JIT are amortized.
- **Problems large enough** that the GPU's DOFs/s is not fighting the
  fixed launch overhead. The p1 PA Jacobi case (263K DOFs) makes this
  clean because the CPU is 101s and the GPU is 1.3s end-to-end.

## When the CPU is at least as good

- **Small AMG-preconditioned solves** (few iterations, expensive setup).
  The `p6 CEED AMG` row shows the pathological version: the GPU wins the
  solve outright but loses the full pipeline to setup.
- **One-shot single solves at small `n`.** The 0.1s CUDA cold start is a
  larger fraction of the whole run than any solve speedup can recover.

## Correctness spot check

L2 error agrees to floating precision on PA Jacobi and Hypre BoomerAMG
(the small-order digits differ for CEED AMG because the two implementations
converge along different paths, but both cases end 4-5 orders of magnitude
below the discretization error at these `p, n` — see `l2_error` column).
