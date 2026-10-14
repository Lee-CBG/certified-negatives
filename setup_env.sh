#!/usr/bin/env bash
# Installs the CUDA build of PyTorch into the active conda env, then verifies it.
#
#   ./setup_env.sh          # defaults to cu124
#   ./setup_env.sh cu121    # older driver
#   ./setup_env.sh cpu      # no GPU
#
# Pick the tag by looking at the "CUDA Version" field in `nvidia-smi`. That
# field reports the HIGHEST CUDA version your driver supports, not what is
# installed -- any wheel tag at or below it will work. You do NOT need a system
# CUDA toolkit; the wheel ships its own runtime. You only need the driver.
set -euo pipefail

TAG="${1:-cu124}"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "No conda environment is active."
  echo "Run:  conda env create -f environment.yml && conda activate tcr-ablation"
  exit 1
fi

echo "Active environment: $CONDA_PREFIX"
echo "Installing torch build: $TAG"
echo

if [ "$TAG" = "cpu" ]; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch --index-url "https://download.pytorch.org/whl/$TAG"
fi

echo
echo "Verifying..."
python3 src/check_gpu.py