#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
VERSION: 1.0.5
UPDATED: 2026-03-05 15:05:00
COPYRIGHT: (c) 2026 ParkCircus Productions; All Rights Reserved.
AUTHOR: Matha Goram
LICENSE: MIT
PURPOSE: Integrity check for VLM weights and config.toml using native tomllib.
================================================================================
"""

import os
import sys
import tomllib
import torch


def print_status(component, status, detail=""):
    """Prints a formatted status message with Unicode icons."""
    icon = "✅" if status else "❌"
    print(f"{icon} {component:.<30} {detail}")


def check_vlm_health():
    """
    Standardized health check utilizing tomllib for clean path resolution.
    Validates hardware, configuration sections, and local file assets.
    """
    print("🩺 Starting VLM Integrity Check...\n")
    all_passed = True

    # 1. Load config.toml using native tomllib (requires binary mode)
    config_path = "config.toml"
    if not os.path.exists(config_path):
        # Adjust path if called from utilities/ subfolder
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")

    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        print_status("Configuration File", True, f"Found: {config_path}")
    except Exception as e:
        print_status("Configuration File", False, f"Error: {str(e)}")
        return False

    # 2. Check Hardware Capability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_status("Hardware detected", True, f"Using {device.upper()}")

    # 3. Check [hello_world] Assets with tomllib-aware pathing
    if "hello_world" in config:
        # tomllib automatically handles the quotes used in config.toml
        test_image = config["hello_world"].get("test_image", "").strip()

        if os.path.exists(test_image):
            print_status("Test Image Asset", True, os.path.basename(test_image))
        else:
            # Enhanced diagnostic: check for traverse permission issues
            parent_dir = os.path.dirname(test_image)
            if os.path.exists(parent_dir) and not os.access(parent_dir, os.X_OK):
                detail = f"PERMISSION DENIED (check chmod o+x {parent_dir})"
            else:
                detail = f"Missing: {test_image}"

            print_status("Test Image Asset", False, detail)
            all_passed = False
    else:
        print_status("Config Section [hello_world]", False, "Missing in TOML")
        all_passed = False

    # 4. Check VLM Model Target
    # Falls back to hello_world if video_auditor section is absent
    model_id = config.get("video_auditor", {}).get(
        "model_id", config.get("hello_world", {}).get("model_id", "Unknown")
    )

    print_status("VLM Model ID Target", True, model_id)

    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL SYSTEMS GO: Ready for Qwen3-VL Benchmarks.")
        return True
    else:
        print("❌ CRITICAL: Fix missing assets or permissions before proceeding.")
        return False


if __name__ == "__main__":
    # Ensure environment is clean for sub-process execution
    if not check_vlm_health():
        sys.exit(1)
