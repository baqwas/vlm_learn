#!/usr/bin/env bash
# ================================================================================
# PROJECT: VLM Learn / ParkCircus Productions 🚀
# AUTHOR: Matha Goram
# LICENSE: MIT
# PURPOSE: Automatically apply headers and print a verification summary.
# ================================================================================

# --- Configuration ---
VERSION_FILE=".version"
HEADER_FILE="license_header.tmp"
DRY_RUN=false # Set to false to apply changes

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit

# --- Metadata Logic ---
[ ! -f "$VERSION_FILE" ] && echo "1.0.0" > "$VERSION_FILE"
CURRENT_VER=$(cat "$VERSION_FILE")

# Increment Patch (e.g., 1.0.0 -> 1.0.1)
IFS='.' read -r major minor patch <<< "$CURRENT_VER"
NEW_VER="$major.$minor.$((patch + 1))"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

if [ "$DRY_RUN" = false ]; then
    echo "$NEW_VER" > "$VERSION_FILE"
fi

# Define the dynamic header
cat << HEADER_EOF > "$HEADER_FILE"
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
VERSION: $NEW_VER
UPDATED: $TIMESTAMP
COPYRIGHT: (c) 2026 ParkCircus Productions; All Rights Reserved.
AUTHOR: Matha Goram
LICENSE: MIT
PURPOSE: [REPLACE WITH FILE DESCRIPTION]
================================================================================
"""
HEADER_EOF

# --- Counters for Summary ---
COUNT_APPLIED=0
COUNT_UPDATED=0
COUNT_SKIPPED=0

# Color coding
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Scanning for files...${NC}"

mapfile -t TARGET_FILES < <(find . -name "*.py" -not -path "./projects/archive/*" -not -path "./.venv/*" -not -path "./.*")

for file in "${TARGET_FILES[@]}"; do
    if ! grep -q "PROJECT: VLM Learn" "$file"; then
        # Case 1: No Header Found
        ((COUNT_APPLIED++))
        if [ "$DRY_RUN" = false ]; then
            tmp_out="tmp_out.py"
            if head -n 1 "$file" | grep -q "^#!"; then
                head -n 1 "$file" > "$tmp_out"
                cat "$HEADER_FILE" >> "$tmp_out"
                tail -n +2 "$file" >> "$tmp_out"
            else
                cat "$HEADER_FILE" "$file" > "$tmp_out"
            fi
            mv "$tmp_out" "$file"
        fi
    else
        # Case 2: Header Exists, Update Metadata
        ((COUNT_UPDATED++))
        if [ "$DRY_RUN" = false ]; then
            sed -i "s/VERSION: .*/VERSION: $NEW_VER/" "$file"
            sed -i "s/UPDATED: .*/UPDATED: $TIMESTAMP/" "$file"
            # Ensure Copyright is present if missing
            if ! grep -q "COPYRIGHT:" "$file"; then
                sed -i "/UPDATED:/a COPYRIGHT: (c) 2026 ParkCircus Productions; All Rights Reserved." "$file"
            fi
        fi
    fi
done

[ -f "$HEADER_FILE" ] && rm "$HEADER_FILE"

# --- Verification Summary Table ---
echo -e "\n${YELLOW}====================================================${NC}"
echo -e "${YELLOW}           VERIFICATION SUMMARY (v$NEW_VER)         ${NC}"
echo -e "${YELLOW}====================================================${NC}"
printf "%-30s | %-10s\n" "CATEGORY" "COUNT"
echo -e "----------------------------------------------------"
printf "%-30s | %-10s\n" "New Headers Applied" "$COUNT_APPLIED"
printf "%-30s | %-10s\n" "Existing Headers Updated" "$COUNT_UPDATED"
printf "%-30s | %-10s\n" "Total Files Processed" "${#TARGET_FILES[@]}"
echo -e "${YELLOW}====================================================${NC}"

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}STATUS: DRY RUN (No files modified)${NC}\n"
else
    echo -e "${GREEN}STATUS: SUCCESS (Files successfully updated)${NC}\n"
fi
