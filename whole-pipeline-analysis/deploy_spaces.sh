#!/bin/bash
# Deploy all 3 DataSage environments to HuggingFace Spaces
# Usage: bash deploy_spaces.sh

set -e

ENVS=("cleaning" "enrichment" "answering")
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

for env in "${ENVS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Deploying: datasage-${env}"
    echo "=============================================="

    ENV_DIR="${BASE_DIR}/environments/${env}"

    # Copy shared modules into each environment so they're self-contained
    echo "  Bundling shared modules..."
    rm -rf "${ENV_DIR}/environments"
    mkdir -p "${ENV_DIR}/environments/shared"
    cp "${BASE_DIR}/environments/__init__.py" "${ENV_DIR}/environments/"
    cp "${BASE_DIR}/environments/shared/"*.py "${ENV_DIR}/environments/shared/"

    # Generate uv.lock if missing
    if [ ! -f "${ENV_DIR}/uv.lock" ]; then
        echo "  Generating uv.lock..."
        (cd "${ENV_DIR}" && uv lock 2>/dev/null || echo "  uv lock skipped (non-critical)")
    fi

    # Push to HF Spaces
    echo "  Pushing to HF Space..."
    openenv push "${ENV_DIR}" --repo-id "ricalanis/datasage-${env}"

    echo "  Done: https://huggingface.co/spaces/ricalanis/datasage-${env}"
done

echo ""
echo "=============================================="
echo "All 3 spaces deployed!"
echo "=============================================="
echo "  - https://ricalanis-datasage-cleaning.hf.space"
echo "  - https://ricalanis-datasage-enrichment.hf.space"
echo "  - https://ricalanis-datasage-answering.hf.space"
