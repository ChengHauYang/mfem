#!/usr/bin/env bash
#
# Source this file before running `./build_mfem.sh cuda`, invoking `ex1p`
# on the GPU, or running python_script/profiling_and_error.py with a
# `cuda` device on this Linux box.
#
#     source ./setup_cuda_env.sh          # prepares env + shims
#
# What it does:
#   1. Locates a conda env carrying nvcc + CUDA headers/libs (default: fenicsx).
#   2. Locates a driver-matching libnvrtc.so (default: /usr/local/cuda-12.4).
#      The NVIDIA driver here (550.x, CUDA 12.4) rejects PTX emitted by nvcc
#      12.9 with UNSUPPORTED_PTX_VERSION, so we preload the 12.4 nvrtc at
#      runtime while still compiling with the conda nvcc.
#   3. Rebuilds two shim directories on /tmp:
#        cuda_home/  -- CUDA_HOME with a standard bin/include/lib64 layout
#                       whose bin/ contains absolute symlinks to nvcc AND its
#                       helpers (cudafe++, cicc, ptxas, fatbinary, nvlink, crt).
#                       Needed because the conda env's targets/x86_64-linux/
#                       nvcc symlink cannot find its sibling helpers.
#        metis/      -- METIS_DIR with just libmetis.so + metis.h symlinked in.
#                       Isolates METIS from the fenicsx env's -L path so the
#                       linker cannot pick up fenicsx's MPICH libmpi.so.12,
#                       which would shadow the OpenMPI hypre was built with.
#        hostbin/    -- provides `g++` -> `g++-11` (Ubuntu ships only g++-11);
#                       required by OpenMPI's mpicxx wrapper.
#   4. Exports CUDA_HOME, CUDA_DIR, PATH, LD_LIBRARY_PATH, CUDA_ARCH,
#      METIS_DIR, OMPI_CXX, OMPI_CC.
#
# Overrides (before sourcing):
#   MFEM_CUDA_ENV=<conda env name>       default: fenicsx
#   MFEM_CUDA_TOOLKIT=<path>             default: /usr/local/cuda-12.4
#   MFEM_CUDA_ARCH=<sm_XX>               default: sm_86 (RTX A5000)
#   MFEM_CUDA_SHIM_ROOT=<dir>            default: /tmp/pae_build
#
# Verified 2026-08-22 on Ubuntu 22.04, 2x RTX A5000 (sm_86), driver 550.54.15
# (CUDA 12.4), OpenMPI 4.1 (system /usr/bin/mpicxx), miniforge fenicsx env
# providing nvcc 12.9 + cudart/cusparse/cublas/cusolver 12.9 + METIS 5.1.

: "${MFEM_CUDA_ENV:=fenicsx}"
: "${MFEM_CUDA_TOOLKIT:=/usr/local/cuda-12.4}"
: "${MFEM_CUDA_ARCH:=sm_86}"
: "${MFEM_CUDA_SHIM_ROOT:=/tmp/pae_build}"

_MFEM_CUDA_ENV_DIR="${HOME}/miniforge/envs/${MFEM_CUDA_ENV}"
_MFEM_CUDA_TARGETS="${_MFEM_CUDA_ENV_DIR}/targets/x86_64-linux"

# --- sanity checks -----------------------------------------------------------
_die() { echo "setup_cuda_env.sh: $*" >&2; return 1; }
[ -x "${_MFEM_CUDA_ENV_DIR}/bin/nvcc" ]              || _die "no nvcc in ${_MFEM_CUDA_ENV_DIR}/bin (set MFEM_CUDA_ENV)" || return 1
[ -x "${_MFEM_CUDA_ENV_DIR}/bin/cudafe++" ]          || _die "no cudafe++ in ${_MFEM_CUDA_ENV_DIR}/bin" || return 1
[ -f "${_MFEM_CUDA_TARGETS}/include/cuda_runtime.h" ] || _die "no CUDA headers in ${_MFEM_CUDA_TARGETS}/include" || return 1
[ -f "${_MFEM_CUDA_ENV_DIR}/lib/libmetis.so" ]       || _die "no libmetis.so in ${_MFEM_CUDA_ENV_DIR}/lib" || return 1
[ -f "${_MFEM_CUDA_ENV_DIR}/include/metis.h" ]       || _die "no metis.h in ${_MFEM_CUDA_ENV_DIR}/include" || return 1
[ -f "${MFEM_CUDA_TOOLKIT}/lib64/libnvrtc.so.12" ]   || _die "no libnvrtc.so.12 in ${MFEM_CUDA_TOOLKIT}/lib64 (set MFEM_CUDA_TOOLKIT to a driver-matching CUDA)" || return 1
[ -x /usr/bin/g++-11 ]                                || _die "/usr/bin/g++-11 missing" || return 1
[ -x /usr/bin/mpicxx ]                                || _die "/usr/bin/mpicxx missing" || return 1

# --- shim: CUDA_HOME (bin has nvcc + all helpers together) -------------------
_SHIM_CUDA="${MFEM_CUDA_SHIM_ROOT}/cuda_home"
mkdir -p "${_SHIM_CUDA}/bin"
for tool in nvcc cudafe++ ptxas fatbinary nvlink cuobjdump cu++filt bin2c nvprune nvcc.profile; do
  if [ -e "${_MFEM_CUDA_ENV_DIR}/bin/${tool}" ]; then
    ln -sfn "${_MFEM_CUDA_ENV_DIR}/bin/${tool}" "${_SHIM_CUDA}/bin/${tool}"
  fi
done
[ -e "${_MFEM_CUDA_ENV_DIR}/nvvm/bin/cicc" ] && ln -sfn "${_MFEM_CUDA_ENV_DIR}/nvvm/bin/cicc" "${_SHIM_CUDA}/bin/cicc"
[ -d "${_MFEM_CUDA_ENV_DIR}/bin/crt" ]       && ln -sfn "${_MFEM_CUDA_ENV_DIR}/bin/crt"       "${_SHIM_CUDA}/bin/crt"
ln -sfn "${_MFEM_CUDA_TARGETS}/include" "${_SHIM_CUDA}/include"
ln -sfn "${_MFEM_CUDA_TARGETS}/lib"     "${_SHIM_CUDA}/lib64"
ln -sfn "${_MFEM_CUDA_TARGETS}/lib"     "${_SHIM_CUDA}/lib"
ln -sfn "${_MFEM_CUDA_ENV_DIR}/nvvm"    "${_SHIM_CUDA}/nvvm"

# --- shim: METIS_DIR (only libmetis.so + metis.h, no other conda libs) -------
_SHIM_METIS="${MFEM_CUDA_SHIM_ROOT}/metis"
mkdir -p "${_SHIM_METIS}/lib" "${_SHIM_METIS}/include"
ln -sfn "${_MFEM_CUDA_ENV_DIR}/lib/libmetis.so" "${_SHIM_METIS}/lib/libmetis.so"
ln -sfn "${_MFEM_CUDA_ENV_DIR}/include/metis.h" "${_SHIM_METIS}/include/metis.h"

# --- shim: hostbin (bare g++, c++) ------------------------------------------
_SHIM_HOSTBIN="${MFEM_CUDA_SHIM_ROOT}/hostbin"
mkdir -p "${_SHIM_HOSTBIN}"
ln -sfn /usr/bin/g++-11 "${_SHIM_HOSTBIN}/g++"
ln -sfn /usr/bin/g++-11 "${_SHIM_HOSTBIN}/c++"

# --- exports -----------------------------------------------------------------
export CUDA_HOME="${_SHIM_CUDA}"
export CUDA_DIR="${_SHIM_CUDA}"
export CUDA_ARCH="${MFEM_CUDA_ARCH}"
export METIS_DIR="${_SHIM_METIS}"
export OMPI_CXX=g++-11
export OMPI_CC=gcc-11

case ":${PATH}:" in
  *":${_SHIM_CUDA}/bin:${_SHIM_HOSTBIN}:"*) ;;
  *) export PATH="${_SHIM_CUDA}/bin:${_SHIM_HOSTBIN}:${PATH}" ;;
esac

_LD_ADD="${MFEM_CUDA_TOOLKIT}/lib64:${_SHIM_METIS}/lib:${_MFEM_CUDA_TARGETS}/lib:${_MFEM_CUDA_ENV_DIR}/nvvm/lib64"
case ":${LD_LIBRARY_PATH:-}:" in
  *":${_LD_ADD}:"*) ;;
  *) export LD_LIBRARY_PATH="${_LD_ADD}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

echo "setup_cuda_env: CUDA_HOME=${CUDA_HOME}"
echo "setup_cuda_env: METIS_DIR=${METIS_DIR}"
echo "setup_cuda_env: CUDA_ARCH=${CUDA_ARCH}  nvcc=$(nvcc --version | tail -1)"
echo "setup_cuda_env: nvrtc runtime=$(readlink -f ${MFEM_CUDA_TOOLKIT}/lib64/libnvrtc.so.12)"

unset _MFEM_CUDA_ENV_DIR _MFEM_CUDA_TARGETS _SHIM_CUDA _SHIM_METIS _SHIM_HOSTBIN _LD_ADD
unset -f _die
