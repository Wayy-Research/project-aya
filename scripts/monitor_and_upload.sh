#!/bin/bash
# Monitor training progress and auto-upload checkpoints to HuggingFace.
#
# Runs in a loop, checking for new checkpoints every 5 minutes.
# Uploads new checkpoints and logs quality metrics.
#
# Usage:
#   nohup bash scripts/monitor_and_upload.sh > /workspace/monitor.log 2>&1 &
#
# Wayy Research, 2024-2026

set -uo pipefail

cd /workspace/aya/project-aya
source .venv/bin/activate

REPO_ID="wayyresearch/aetheris"
CHECK_INTERVAL=300  # 5 minutes
LAST_UPLOADED_S1=""
LAST_UPLOADED_S2=""
LAST_UPLOADED_S3=""

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

upload_if_new() {
    local stage=$1
    local stage_dir=$2
    local last_var=$3

    if [ ! -d "$stage_dir" ]; then
        return
    fi

    # Find the latest checkpoint
    local latest=$(ls -t "$stage_dir"/step_*.pt "$stage_dir"/best.pt 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        return
    fi

    # Check if we already uploaded this one
    local current_ref="${!last_var}"
    local latest_mod=$(stat -c %Y "$latest" 2>/dev/null || echo "0")

    if [ "$latest_mod" = "$current_ref" ]; then
        return
    fi

    log "New checkpoint detected for Stage $stage: $latest"

    # Extract quick metrics from training log
    local step_info=$(grep -oP '\[Step \d+/\d+\]' /workspace/training.log 2>/dev/null | tail -1)
    local loss_info=$(grep -oP 'loss=[\d.]+' /workspace/training.log 2>/dev/null | tail -1)
    local cka_info=$(grep -oP 'cka_mean=[\d.]+' /workspace/training.log 2>/dev/null | tail -1)

    log "  Progress: ${step_info:-unknown} ${loss_info:-} ${cka_info:-}"

    # Upload
    local msg="Stage ${stage} checkpoint: ${step_info:-auto} ${loss_info:-} ${cka_info:-}"
    python scripts/upload_checkpoint.py --stage "$stage" --checkpoint "$latest" --message "$msg" 2>&1

    if [ $? -eq 0 ]; then
        log "  Upload successful!"
        eval "${last_var}=${latest_mod}"
    else
        log "  Upload FAILED"
    fi
}

quality_report() {
    log "=== QUALITY REPORT ==="

    # GPU status
    local gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null)
    log "  GPU: $gpu_info"

    # Training process
    local train_pid=$(pgrep -f "run_full_distill" 2>/dev/null)
    if [ -n "$train_pid" ]; then
        log "  Training PID: $train_pid (running)"
    else
        log "  WARNING: Training process not found!"
    fi

    # Latest metrics from log
    local last_lines=$(tail -20 /workspace/training.log 2>/dev/null)
    log "  Recent log:"
    echo "$last_lines" | while read line; do
        echo "    $line"
    done

    # Checkpoint sizes
    for stage_dir in checkpoints/stage1_alignment checkpoints/stage2_kl checkpoints/stage3_sft; do
        if [ -d "$stage_dir" ]; then
            local count=$(ls "$stage_dir"/*.pt 2>/dev/null | wc -l)
            local size=$(du -sh "$stage_dir" 2>/dev/null | cut -f1)
            log "  $stage_dir: $count checkpoints, $size"
        fi
    done

    # Disk space
    local disk=$(df -h /workspace | tail -1 | awk '{print $4 " free of " $2}')
    log "  Disk: $disk"

    log "=== END REPORT ==="
}

# Initial report
log "Starting checkpoint monitor for $REPO_ID"
log "Check interval: ${CHECK_INTERVAL}s"
quality_report

# Main loop
iteration=0
while true; do
    iteration=$((iteration + 1))

    # Check for new checkpoints in each stage
    upload_if_new 1 "checkpoints/stage1_alignment" "LAST_UPLOADED_S1"
    upload_if_new 2 "checkpoints/stage2_kl" "LAST_UPLOADED_S2"
    upload_if_new 3 "checkpoints/stage3_sft" "LAST_UPLOADED_S3"

    # Quality report every 6 iterations (30 min)
    if [ $((iteration % 6)) -eq 0 ]; then
        quality_report
    fi

    # Check if training is still running
    if ! pgrep -f "run_full_distill" > /dev/null 2>&1; then
        log "Training process completed or crashed!"
        quality_report

        # Upload any final checkpoints
        upload_if_new 1 "checkpoints/stage1_alignment" "LAST_UPLOADED_S1"
        upload_if_new 2 "checkpoints/stage2_kl" "LAST_UPLOADED_S2"
        upload_if_new 3 "checkpoints/stage3_sft" "LAST_UPLOADED_S3"

        log "Final upload complete. Exiting monitor."
        break
    fi

    sleep $CHECK_INTERVAL
done
