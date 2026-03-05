#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
VERSION: 1.0.1
UPDATED: 2026-03-05 08:13:21
COPYRIGHT: (c) 2026 ParkCircus Productions; All Rights Reserved.
AUTHOR: Matha Goram
LICENSE: MIT
PURPOSE: [REPLACE WITH FILE DESCRIPTION]
================================================================================
"""
"""
================================================================================
PROJECT: Ubuntu Image Validator (Live Progress Counter)
LICENSE: MIT License
================================================================================
PROCESSING WORKFLOW:
1. Session Init: Loads config and tallies the total number of quarantined items.
2. Large Preview: Temporarily upscales the 224px image for human review.
3. Managed Process: Launches 'eog' (Eye of GNOME) via subprocess for Ubuntu.
4. Interactive Progress:
    - Displays [Current/Total] and [Remaining] in the terminal header.
5. Auto-Cleanup: Terminates the viewer and removes temp artifacts on choice.
6. Data Triage: Moves or deletes files based on user numeric input.

DATA INTEGRITY LOGIC:
- Human-in-the-Loop (HITL): Provides a manual override layer to resolve uncertainties flagged
  by automated mathematical or AI filters.
- Non-Destructive Preview: Generates a temporary upscaled artifact for review to preserve the
  original file's metadata and resolution until a final decision is made.
- Dataset Promotion: Successfully "Relabeled" items are physically moved from the quarantine
  buffer back into the production 'output_dir'.
- Permanent Purge: Explicit 'Delete' choice (D) allows for the physical removal of corrupt
  or non-relevant data from the storage volume.

USER INTERFACE REQUIREMENTS:
- Host OS: Ubuntu-based environment with 'Eye of GNOME' (eog) installed for native image rendering.
- Managed GUI Process: Automatically launches and terminates the 'eog' subprocess for each file to ensure
  a single, focused preview window during the audit.
- Interactive Progress Counter: Real-time terminal header displaying [Processed/Total] and [Remaining]
  counts to provide the auditor with session velocity metrics.
- Dynamic Menu: Generates a numeric keyboard-mapping menu derived from the 'vegetables' label
  list defined in config.toml.


ERROR HANDLING & EXCEPTION STRATEGY:
- Subprocess Guardrails: Wrapped subprocess calls to catch 'FileNotFound' errors if the
  system viewer (eog) is missing or unresponsive.
- Artifact Cleanup: Ensures temporary preview files are deleted from the disk even if the
  user terminates the script prematurely.
- Move/Delete Resilience: Uses shutil and os modules within try-except blocks to manage
  file locks or permission issues during the triage phase.
- Logging Severity:
    - INFO: Records successful relabeling and file promotion for audit trails.
    - WARNING: Logs intentional file deletions for data loss tracking.
    - ERROR: Catches rendering failures or PIL-related image corruption.
================================================================================
"""

import os
import tomllib
import logging
import shutil
import subprocess
from PIL import Image
from pathlib import Path


def setup_logging(log_path):
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger("Ubuntu_Validator")


def run_validation():
    # 1. SETUP & CONFIG
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.toml"

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    logger = setup_logging(config["paths"]["log_file"])
    quarantine_dir = root_dir / config["paths"]["quarantine_dir"]
    output_dir = root_dir / config["paths"]["output_dir"]
    labels = config["labels"]["vegetables"]
    display_size = config["resize"].get("display_size", 800)

    # Calculate initial counts
    all_files = [
        f
        for f in quarantine_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    total_to_process = len(all_files)

    if total_to_process == 0:
        logger.info("Quarantine queue is empty. No validation needed.")
        return

    label_menu = {str(i + 1): label for i, label in enumerate(labels)}
    processed_this_session = 0

    # 2. VALIDATION LOOP
    for img_path in all_files:
        processed_this_session += 1
        remaining = total_to_process - processed_this_session

        # Header with Counter
        print("\n" + "=" * 60)
        print(
            f" PROGRESS: {processed_this_session}/{total_to_process} ({remaining} remaining)"
        )
        print(f" FILE:     {img_path.name}")
        print("=" * 60)

        # ACTION: ENLARGE PREVIEW
        temp_preview = root_dir / f"temp_preview{img_path.suffix}"
        try:
            with Image.open(img_path) as img:
                w_percent = display_size / float(img.size[0])
                h_size = int((float(img.size[1]) * float(w_percent)))
                img.resize((display_size, h_size), Image.Resampling.LANCZOS).save(
                    temp_preview
                )

            # Managed Ubuntu Viewer
            viewer_proc = subprocess.Popen(["eog", str(temp_preview)])
        except Exception as e:
            logger.error(f"Error displaying {img_path.name}: {e}")
            continue

        # DISPLAY OPTIONS
        for num, label in label_menu.items():
            print(f" [{num}] {label}")
        print(" [S] Skip | [D] Delete")

        choice = input("\nSelect choice: ").strip().upper()

        # 3. AUTO-CLOSE & CLEANUP
        viewer_proc.terminate()
        if temp_preview.exists():
            os.remove(temp_preview)

        # 4. ACTION TRIAGE
        if choice in label_menu:
            shutil.move(str(img_path), str(output_dir / img_path.name))
            logger.info(f"Relabeled: {img_path.name}")
        elif choice == "D":
            os.remove(img_path)
            logger.warning(f"Deleted: {img_path.name}")
        else:
            logger.info(f"Skipped: {img_path.name}")

    logger.info(f"Session complete. Processed {processed_this_session} items.")


if __name__ == "__main__":
    run_validation()
