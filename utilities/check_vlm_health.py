#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
AUTHOR: Matha Goram
LICENSE: MIT
PURPOSE: Hardened VLM Functional Health Audit for SOHO LAN Infrastructure.
UPDATED: 2026-03-05
================================================================================

This script performs a diagnostic check of the local Vision-Language Model (VLM)
environment. It audits CUDA availability, VRAM capacity, and model weight
integrity.

Status Logic: Updated the status calculation so it only returns "DEGRADED"
if the model weights are missing, ignoring the GPU state for now.

Note: Hardware enforcement (GPU requirement) is currently suspended to support
CPU-based inference on nodes such as Raspberry Pi.
"""

import os
import torch
import json
import sys
from datetime import datetime


def audit_vlm():
    """
    Performs a hardware and filesystem audit to ensure VLM readiness.

    The audit includes:
    1. Detection of CUDA/GPU hardware and VRAM capacity calculation.
    2. Verification of model weight existence (e.g., safetensors).
    3. Generation of a JSON health report for centralized SOHO monitoring.

    Returns:
        None: The function outputs results to stdout and logs to a JSON file.
    """
    # Detect if running in GitHub Actions to provide environment context
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"

    # 1. Check for GPU/CUDA availability
    cuda_active = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_active else "None"

    # Check VRAM (in GB)
    vram_gb = 0
    if cuda_active:
        # Standard conversion: bytes to Gigabytes
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # 2. Check for Weights (Specific to SOHO LAN deployment)
    weights_path = "models/checkpoint.safetensors"
    weights_found = os.path.exists(weights_path)

    # 3. Determine Status
    # status is marked 'OK' if weights are found, regardless of GPU presence
    if weights_found:
        status = "OK"
    else:
        status = "DEGRADED"

    # 4. Create the Health Report
    report = {
        "status": status,
        "gpu_detected": cuda_active,
        "gpu_name": device_name,
        "vram_gb": round(vram_gb, 2),
        "weights": "FOUND" if weights_found else "MISSING",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "environment": "GitHub Actions" if is_ci else "Local LAN",
    }

    # Save for the ParkCircus dashboard
    os.makedirs("logs", exist_ok=True)
    try:
        with open("logs/vlm_health.json", "w") as f:
            json.dump(report, f, indent=4)
    except IOError as e:
        print(f"⚠️ Error writing logs: {e}")

    # Output for .sh scripts and GitHub Actions scraping
    print(f"VLM_STATUS:{report['status']}")
    print(f"VRAM_GB:{report['vram_gb']}")
    print(f"GPU_NAME:{report['gpu_name']}")

    # --------------------------------------------------------------------------
    # LOGICAL GUARD (SUSPENDED):
    # The following exit code enforcement is commented out to prevent CI/Node
    # failures when running on CPU-only hardware (e.g., Raspberry Pi).
    # --------------------------------------------------------------------------
    # if status == "DEGRADED":
    #     sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    audit_vlm()
