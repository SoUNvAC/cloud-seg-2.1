#!/usr/bin/env bash

# Reproducible Conda bootstrap for Cloud-Adapter / experiment 01.
#
# Usage:
#   bash setup.sh                    # creates/updates "cloud-adapter"
#   CLOUD_ADAPTER_ENV=myenv bash setup.sh

set -Eeuo pipefail

ENV_NAME="${CLOUD_ADAPTER_ENV:-cloud-adapter}"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

log() {
    printf '[setup] %s\n' "$*"
}

die() {
    printf '[setup] ERROR: %s\n' "$*" >&2
    exit 1
}

command -v conda >/dev/null 2>&1 || die \
    "conda was not found. Install Miniconda/Anaconda and run this script again."

# `conda run` makes the script work from both an activated base environment and
# a plain shell; users do not need to source conda.sh first.
if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
    log "reusing Conda environment: $ENV_NAME"
else
    log "creating Conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create --yes --name "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

ACTUAL_PYTHON="$(conda run --name "$ENV_NAME" python -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' | tr -d '\r')"
[[ "$ACTUAL_PYTHON" == "$PYTHON_VERSION" ]] || die \
    "environment '$ENV_NAME' uses Python $ACTUAL_PYTHON; Python $PYTHON_VERSION is required."

cd "$SCRIPT_DIR"
log "upgrading packaging tools"
conda run --no-capture-output --name "$ENV_NAME" \
    python -m pip install --upgrade pip setuptools wheel

log "installing pinned CUDA 12.1 / PyTorch 2.1 dependencies"
conda run --no-capture-output --name "$ENV_NAME" \
    python -m pip install --prefer-binary -r requirements.txt

log "checking dependency consistency"
conda run --no-capture-output --name "$ENV_NAME" python -m pip check

log "running import and CUDA-operator smoke test"
conda run --no-capture-output --name "$ENV_NAME" python -c \
    "import torch, mmcv, mmengine, mmseg, mmdet, xformers; from mmcv.ops import MultiScaleDeformableAttention; import cloud_adapter; expected={'torch':'2.1.2','mmcv':'2.1.0','mmengine':'0.10.4','mmseg':'1.2.2','mmdet':'3.3.0','xformers':'0.0.23.post1'}; actual={'torch':torch.__version__.split('+')[0],'mmcv':mmcv.__version__,'mmengine':mmengine.__version__,'mmseg':mmseg.__version__,'mmdet':mmdet.__version__,'xformers':xformers.__version__}; mismatched={k:(actual[k],v) for k,v in expected.items() if actual[k] != v}; assert not mismatched, f'version mismatch: {mismatched}'; print('versions:', actual); print('CUDA available:', torch.cuda.is_available()); print('Cloud-Adapter import: OK')"

log "environment is ready"
printf '\nRun:\n  conda activate %s\n\n' "$ENV_NAME"
printf 'Then start with:\n  python tools/experiment_01/run_matrix.py --dry-run all\n'
