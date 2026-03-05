#!/bin/bash
# ==============================================================================
# PROJECT: BeUlta VLM Audit Orchestrator (Professional Edition) 🚀
# AUTHOR:  Matha Goram 🛠️
# LOCATION: /diagnostics/launch_audit.sh
# ==============================================================================

# --- CONFIGURATION & HELP ---

show_help() {
cat << EOF
================================================================================
🚀 BEULTA VLM ORCHESTRATOR: MANUAL & HOUSEKEEPING
================================================================================
USAGE:
  bash diagnostics/launch_audit.sh [OPTION]

OPTIONS:
  --help, -h  Display this manual and housekeeping strategies.
  --force     [NUCLEAR OPTION] Forcefully terminates all processes currently
              holding handles on the GPU. Use this after a crash or OOM event.

SAFE MODE & GPU HYGIENE:
  * OCCUPANCY CHECK: The script verifies pre-existing VRAM usage is < 500MB.
    This prevents memory fragmentation, which is critical on 6GB hardware.
  * THE 90% RULE: Monitor the top tmux pane. If VRAM usage exceeds 5.4GB (90%),
    the auditor is likely to crash during the next 'Attention' cycle.
  * ZOMBIE PROCESSES: PyTorch sometimes fails to release GPU handles on exit.
    If 'nvidia-smi' shows usage but no active apps, use the --force flag.

SYSTEM RAM TIPS (8GB LIMIT):
  * Before launching, clear the system pagecache for weight-loading headroom:
    'sync && sudo sysctl -w vm.drop_caches=3'
  * Avoid keeping web browsers (Chrome/Firefox) open during model load.

USAGE SUMMARY 📋:
  * View your manual:
    bash diagnostics/launch_audit.sh --help
  * Start a clean session:
    bash diagnostics/launch_audit.sh
  * Clear the GPU after a crash and restart:
    bash diagnostics/launch_audit.sh --force
================================================================================
EOF
}

# --- INITIALIZATION & FLAGS ---

SESSION="vlm_audit"
VRAM_THRESHOLD=500  # MB
FORCE_CLEAN=false

# Flag Parsing
case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    --force)
        FORCE_CLEAN=true
        ;;
esac

# Resolve paths relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- PRE-FLIGHT CHECKS ---

# 1. Dependency Check
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed. Run: sudo apt install tmux"
    exit 1
fi

# 2. 🧨 NUCLEAR OPTION: Force Clear GPU
if [ "$FORCE_CLEAN" = true ]; then
    echo "☢️  NUCLEAR OPTION ENABLED: Terminating all GPU processes..."
    # fuser finds PIDs using the device, xargs kills them
    fuser -v /dev/nvidia* 2>/dev/null | awk '{print $NF}' | xargs -r kill -9
    sleep 2
    echo "✅ GPU handles cleared."
fi

# 3. 🛡️ SAFE MODE: GPU Occupancy Check
CURRENT_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')

if [ "$CURRENT_VRAM" -gt "$VRAM_THRESHOLD" ]; then
    echo "🤨 You gotta be kidding! GPU is already occupied."
    echo "📊 Current Usage: ${CURRENT_VRAM}MB (Threshold: ${VRAM_THRESHOLD}MB)"
    echo "--------------------------------------------------------"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo "--------------------------------------------------------"
    read -p "⚠️  Memory is tight. Launch anyway? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "🛑 Launch aborted. Use --force to clear the GPU."
        exit 1
    fi
fi

# --- TMUX ORCHESTRATION ---

# Cleanup existing tmux session
tmux kill-session -t $SESSION 2>/dev/null

# Create session and set working directory to root
tmux new-session -d -s $SESSION -c "$PROJECT_ROOT"

# PANE 0 (TOP): VRAM Monitor
# We explicitly set the PYTHONPATH to the root for the diagnostics module
tmux send-keys -t $SESSION "PYTHONPATH=$PROJECT_ROOT python3 diagnostics/vram_monitor.py" C-m

# PANE 1 (BOTTOM): VLM Auditor
tmux split-window -v -t $SESSION -c "$PROJECT_ROOT"
tmux send-keys -t $SESSION "PYTHONPATH=$PROJECT_ROOT python3 clip_exercise/clip_work/video_auditor.py" C-m

# UI Layout (Top pane gets 10 lines of height)
tmux resize-pane -t $SESSION:0.0 -y 10

# Attach to the session
tmux attach-session -t $SESSION
