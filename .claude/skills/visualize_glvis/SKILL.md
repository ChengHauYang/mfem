---
name: visualize_glvis
description: Install GLVis and open MFEM mesh, GridFunction, or quadrature-function output. Use when the user asks to install/build GLVis, visualize MFEM results, open mesh.* and sol.* files, or troubleshoot GLVis dependencies and file loading.
---

# Visualize MFEM output with GLVis

## Install

Run the repository installer; do not reconstruct the dependency list manually:

```bash
cd scripts
./install_glvis.sh
```

The script installs the macOS dependencies with Homebrew, clones GLVis beside
MFEM, builds it against MFEM's existing `config/config.mk`, and installs the
executable at `../bin/glvis` relative to the package root. It is idempotent and
uses an existing GLVis checkout without changing its branch or local changes.

Default layout:

```text
Package/
  bin/glvis
  glvis/
  mfem/
```

Overrides: `MFEM_ROOT`, `GLVIS_DIR`, `PREFIX`, `GLVIS_TAG`, and `NPROC`.

## Open output

Use the installed executable directly so the user's `PATH` is irrelevant:

```bash
/path/to/Package/bin/glvis -m refined.mesh -g sol.gf
/path/to/Package/bin/glvis -np 8 -m mesh -g sol
```

For parallel output, determine the rank count from the contiguous
`mesh.000000`, `mesh.000001`, ... files. Confirm the same ranks exist for the
solution prefix, then pass that count to `-np`. The arguments are prefixes, not
the rank-zero filenames: use `-m mesh -g sol`, not `-m mesh.000000`.

Other native forms:

```bash
glvis -m mesh_file
glvis -m mesh_file -q quadrature_function
glvis -np N -m mesh_prefix -q quadrature_prefix
```

Use `-gc COMPONENT` or `-qc COMPONENT` to select a component. Run `glvis -h`
for all options.

## Troubleshoot

- `glm/mat4x4.hpp file not found`: run `brew install glm`, then rerun the installer.
- `glvis: command not found`: invoke `Package/bin/glvis` directly or add `Package/bin` to `PATH`.
- Missing rank file: do not guess a smaller `-np`; verify the simulation completed and produced matching mesh and field files.
- ParaView cannot directly open MFEM native `mesh.*` and `sol.*` output. Use GLVis, or rerun/export through an MFEM ParaView data collection.
