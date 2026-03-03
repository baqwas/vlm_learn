#!/usr/bin/env python3

import os
import torch
import json

from datetime import datetime


def audit_vlm():
    # 1. Check for GPU/CUDA availability
    cuda_active = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_active else "None"

    # Check VRAM (in GB)
    vram_gb = 0
    if cuda_active:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # 2. Check for Weights (using the path you'll likely use later)
    weights_path = "models/checkpoint.safetensors"
    weights_found = os.path.exists(weights_path)

    # 3. Create the Health Report
    report = {
        "status": "OK" if cuda_active else "DEGRADED",
        "gpu_name": device_name,
        "vram_gb": round(vram_gb, 2),
        "weights": "FOUND" if weights_found else "MISSING",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save for the dashboard
    os.makedirs("logs", exist_ok=True)
    with open("logs/vlm_health.json", "w") as f:
        json.dump(report, f, indent=4)

    # Output for the .sh script and GitHub Actions to scrape
    print(f"VLM_STATUS:{report['status']}")
    print(f"VRAM_GB:{report['vram_gb']}")
    print(f"GPU_NAME:{report['gpu_name']}")


if __name__ == "__main__":
    audit_vlm()
