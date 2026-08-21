#!/usr/bin/env bash
#
# Build MFEM + hypre + libCEED for:  ex1p -pa -a -d ceed-{cpu,cuda}
#
#   ./build_mfem.sh cpu           # verified 2026-08-21 on Apple M4 Pro / macOS 25.6
#   ./build_mfem.sh cuda          # NOT YET VERIFIED -- no CUDA machine available
#
# This script lives in <root>/mfem/scripts/ and operates on <root>, the directory
# holding mfem/ alongside hypre/ and libCEED/ (the latter two are cloned if
# missing). Override with MFEM_ROOT=/path/to/root.
#
set -euo pipefail

MODE="${1:-cpu}"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MFEM_ROOT:-$(cd "$SELF/../.." && pwd)}"   # scripts/ is 2 levels below root
cd "$ROOT"

HYPRE_TAG=v2.32.0        # MFEM detects this as HYPRE version 23200
CEED_TAG=v0.12.0         # MFEM INSTALL requires libCEED >= 0.12.0
NPROC="${NPROC:-$( (nproc 2>/dev/null) || sysctl -n hw.perflevel0.logicalcpu )}"

case "$MODE" in
  cpu)  ;;
  cuda) : "${CUDA_ARCH:?cuda mode needs CUDA_ARCH, e.g. sm_80 (A100) sm_90 (H100) sm_70 (V100) sm_89 (L40S)}" ;;
  *)    echo "ERROR: mode must be 'cpu' or 'cuda', got '$MODE'"; exit 1 ;;
esac
[ -d mfem ] || { echo "ERROR: no mfem/ in $ROOT (set MFEM_ROOT to override)"; exit 1; }
echo "==> root=$ROOT mode=$MODE jobs=$NPROC${CUDA_ARCH:+ arch=$CUDA_ARCH}"

# --------------------------------------------------------------- METIS 5
# MFEM defaults to ../metis-4.0; we use METIS 5 and must say so explicitly
# (MFEM_USE_METIS_5=YES), otherwise the METIS API signatures won't match.
if [ "$MODE" = cpu ] && [ -d /opt/homebrew/opt/metis ]; then
  METIS_DIR=/opt/homebrew/opt/metis          # brew install metis
else
  : "${METIS_DIR:=/usr}"                     # cluster: module load metis
fi

# --------------------------------------------------------------- hypre
# `make install` lands in src/hypre/, which is exactly MFEM's default HYPRE_DIR
# (config/defaults.mk:257), so no override is needed.
if [ ! -d hypre ]; then
  git clone --depth 1 -b "$HYPRE_TAG" https://github.com/hypre-space/hypre.git
fi
if [ ! -f hypre/src/hypre/lib/libHYPRE.a ]; then
  pushd hypre/src >/dev/null
  if [ "$MODE" = cuda ]; then
    ./configure --disable-fortran --with-MPI \
                --with-cuda --with-gpu-arch="${CUDA_ARCH#sm_}" \
                --enable-unified-memory
  else
    # --disable-fortran: avoids needing a Fortran ABI match; MFEM never calls
    # hypre's Fortran interface.
    ./configure --disable-fortran --with-MPI CC=mpicc CXX=mpicxx
  fi
  make -j"$NPROC" install
  popd >/dev/null
fi

# --------------------------------------------------------------- libCEED
if [ ! -d libCEED ]; then
  git clone --depth 1 -b "$CEED_TAG" https://github.com/CEED/libCEED.git
fi
if [ ! -e libCEED/lib/libceed.so ] && [ ! -e libCEED/lib/libceed.dylib ]; then
  pushd libCEED >/dev/null
  if [ "$MODE" = cuda ]; then
    make -j"$NPROC" CUDA_DIR="${CUDA_HOME:-/usr/local/cuda}"
  else
    make -j"$NPROC"
  fi
  popd >/dev/null
fi

# --------------------------------------------------------------- MFEM
pushd mfem >/dev/null
CFG=( MFEM_USE_MPI=YES
      MFEM_USE_METIS=YES MFEM_USE_METIS_5=YES
      METIS_DIR="$METIS_DIR"
      METIS_OPT="-I$METIS_DIR/include"
      METIS_LIB="-L$METIS_DIR/lib -lmetis"
      MFEM_USE_CEED=YES CEED_DIR="$ROOT/libCEED"
      MPICXX=mpicxx )
[ "$MODE" = cuda ] && CFG+=( MFEM_USE_CUDA=YES CUDA_ARCH="${CUDA_ARCH}" )

make config "${CFG[@]}"
make status | grep -E "MFEM_USE_(MPI|CUDA|CEED|METIS)|CUDA_ARCH"
make -j"$NPROC"
make -C examples ex1p
popd >/dev/null

cat <<EOF

==> built: $ROOT/mfem/examples/ex1p

  cd $ROOT/mfem/examples
  mpirun -np 8 ./ex1p -m ../data/inline-hex.mesh -o 4 -pa -a -d ceed-$MODE -no-vis

  Do NOT use ../data/periodic-cube.mesh -- see $SELF/build_mfem.md
EOF
