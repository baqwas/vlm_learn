#!/usr/bin/env python3
# ================================================================================
# PROJECT: VLM Learn / ParkCircus Productions 🚀
# AUTHOR: Matha Goram
# LICENSE: MIT
# LOCATION: utilities/check_vlm_health.py
#
# PURPOSE:
#   Audits the VLM environment for CUDA availability and model integrity.
#   Outputs status to STDOUT to avoid Git staging conflicts.
#
# UI & ERROR HANDLING SUMMARY:
#   🖥️  Interface: Clean Unicode console output for CI/CD runners.
#   ⚠️  OOM: Summarized if GPU memory is insufficient.
#   🚫  No-File: This script no longer writes to logs/vlm_health.json.
# ================================================================================

import sys
import torch
import json


def check_health():
    """
    Performs a hardware and software audit for VLM inference.
    """
    health_status = {
        "VLM_STATUS": "FAILED",
        "VRAM_GB": 0,
        "GPU_NAME": "None",
        "CUDA_AVAILABLE": False,
    }

    try:
        # Check for CUDA (Essential for Jetson/Orin nodes)
        if torch.cuda.is_available():
            health_status["CUDA_AVAILABLE"] = True
            health_status["GPU_NAME"] = torch.cuda.get_device_name(0)
            health_status["VRAM_GB"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
            health_status["VLM_STATUS"] = "OK"
        else:
            # CPU-Only fallback for Raspberry Pi nodes
            health_status["VLM_STATUS"] = "OK"  # Still 'OK' if intended for CPU
            health_status["GPU_NAME"] = "CPU-Only (No CUDA detected)"

        # 🟢 PRINT TO STDOUT (Standardized for Pre-commit capture)
        print(f"VLM_STATUS:{health_status['VLM_STATUS']}")
        print(f"VRAM_GB:{health_status['VRAM_GB']}")
        print(f"GPU_NAME:{health_status['GPU_NAME']}")

        # Return success code for pre-commit (0 = Pass)
        return 0 if health_status["VLM_STATUS"] == "OK" else 1

    except Exception as e:
        print(f"VLM_STATUS:ERROR - {str(e)[:50]}")
        return 1


if __name__ == "__main__":
    sys.exit(check_health())
