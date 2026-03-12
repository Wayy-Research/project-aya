#!/bin/bash
# Quick one-liner to bootstrap and start training on RunPod.
# Paste this into the RunPod web terminal.
#
# Pod ID: i60pto5lzs54ci
# URL: https://www.runpod.io/console/pods/i60pto5lzs54ci

set -euo pipefail

# Clone repos
mkdir -p /workspace/aya && cd /workspace/aya
git clone -b dev https://github.com/Wayy-Research/project-aya.git 2>/dev/null || (cd project-aya && git pull origin dev && cd ..)
git clone https://github.com/Wayy-Research/aetheris.git 2>/dev/null || (cd aetheris && git pull origin main && cd ..)

# Run full setup
bash /workspace/aya/project-aya/scripts/runpod_setup.sh
