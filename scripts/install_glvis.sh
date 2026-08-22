#!/usr/bin/env bash
#
# Install GLVis for the sibling MFEM build and expose it as <root>/bin/glvis.
#
#   ./install_glvis.sh
#   GLVIS_TAG=master ./install_glvis.sh
#
# This script lives in <root>/mfem/scripts/. Override the inferred layout with
# MFEM_ROOT=/path/to/root, GLVIS_DIR=/path/to/glvis, or PREFIX=/install/path.
#
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MFEM_ROOT:-$(cd "$SELF/../.." && pwd)}"
MFEM_DIR="$ROOT/mfem"
GLVIS_DIR="${GLVIS_DIR:-$ROOT/glvis}"
PREFIX="${PREFIX:-$ROOT/bin}"
GLVIS_TAG="${GLVIS_TAG:-v4.5}"
NPROC="${NPROC:-$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.logicalcpu)}"

[ "$(uname -s)" = Darwin ] || {
  echo "ERROR: this installer currently supports macOS with Homebrew only"
  echo "See https://github.com/GLVis/glvis/blob/master/INSTALL for Linux dependencies."
  exit 1
}
command -v brew >/dev/null || {
  echo "ERROR: Homebrew is required: https://brew.sh"
  exit 1
}
[ -f "$MFEM_DIR/config/config.mk" ] || {
  echo "ERROR: MFEM is not configured at $MFEM_DIR"
  echo "Build MFEM first with $SELF/build_mfem.sh cpu"
  exit 1
}

DEPS=(fontconfig freetype sdl2 glew glm libpng)
MISSING=()
for dep in "${DEPS[@]}"; do
  brew list --versions "$dep" >/dev/null 2>&1 || MISSING+=("$dep")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "==> installing missing GLVis dependencies: ${MISSING[*]}"
  HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK=1 brew install "${MISSING[@]}"
else
  echo "==> GLVis dependencies are already installed"
fi

if [ ! -d "$GLVIS_DIR" ]; then
  echo "==> cloning GLVis $GLVIS_TAG"
  git clone --depth 1 --branch "$GLVIS_TAG" \
    https://github.com/GLVis/glvis.git "$GLVIS_DIR"
elif [ ! -d "$GLVIS_DIR/.git" ]; then
  echo "ERROR: $GLVIS_DIR exists but is not a Git checkout"
  exit 1
else
  echo "==> using existing checkout: $GLVIS_DIR"
fi

echo "==> building GLVis with $NPROC jobs"
make -C "$GLVIS_DIR" -j"$NPROC" MFEM_DIR="$MFEM_DIR"
make -C "$GLVIS_DIR" install MFEM_DIR="$MFEM_DIR" PREFIX="$PREFIX"

cat <<EOF

==> installed: $PREFIX/glvis

Add it to this shell if needed:
  export PATH="$PREFIX:\$PATH"

Visualize an 8-rank MFEM result:
  cd "$MFEM_DIR/examples"
  "$PREFIX/glvis" -np 8 -m mesh -g sol
EOF
