#!/usr/bin/env bash
# ==============================================================================
# 🧠 VLM INTEGRITY & GPU AUDIT WRAPPER
# ==============================================================================
PROJ_ROOT=$(git rev-parse --show-toplevel)
VENV_PY="$PROJ_ROOT/.venv/bin/python3"
AUDIT_SCRIPT="$PROJ_ROOT/utilities/check_vlm_health.py"

# --- STYLING ---
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}🚀 Starting VLM Environment Audit...${NC}"

# 1. Run the Python Integrity Check
if [[ -f "$AUDIT_SCRIPT" ]]; then
    AUDIT_RAW=$($VENV_PY "$AUDIT_SCRIPT")

    # Extract values for display
    STATUS=$(echo "$AUDIT_RAW" | grep "VLM_STATUS:" | cut -d':' -f2)
    VRAM=$(echo "$AUDIT_RAW" | grep "VRAM_GB:" | cut -d':' -f2)
    GPU=$(echo "$AUDIT_RAW" | grep "GPU_NAME:" | cut -d':' -f2)

    # Set Icon (Fixed the logic here)
    if [[ "$STATUS" == "OK" ]]; then
        ICON="🟢"
    else
        ICON="🔴"
    fi
else
    echo "❌ Audit script not found at $AUDIT_SCRIPT"
    exit 1
fi

# 2. Terminal Dashboard
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BOLD}VLM SYSTEM STATUS :${NC} $ICON $STATUS"
echo -e "${BOLD}ACTIVE GPU        :${NC} $GPU"
echo -e "${BOLD}AVAILABLE VRAM    :${NC} $VRAM GB"
echo -e "${BLUE}════════════════════════════════════════${NC}"
