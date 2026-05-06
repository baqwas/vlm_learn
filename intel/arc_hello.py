#!/usr/bin/env python3
"""
==============================================================================
PROJECT: Intel Arc XPU Availability & Environment Audit
AUTHOR:  Matha Goram
LICENSE: MIT License
COPYRIGHT: (c) ParkCircus Productions
REFERENCE: https://pytorch.org/get-started/locally/
VERSION: 1.2.0

UPDATE TRACKING:
    2026-05-05: v1.0.0 - Initial deployment.
    2026-05-05: v1.1.0 - Added environment mismatch detection (+cpu vs +xpu).
    2026-05-05: v1.2.0 - Integrated native XPU support (Post-IPEX EOL).

PROCESSING WORKFLOW:
    1. Retrieve and log the currently loaded PyTorch build version.
    2. Interrogate the hardware abstraction layer for XPU (Intel GPU) visibility.
    3. If XPU is unavailable, perform a string-check on the version to
       diagnose if a CPU-only build is the root cause.
    4. Enumerate hardware properties for the primary compute device.

USER INTERFACE:
    Standard console output with diagnostic remedies for build mismatches.

EXCEPTION HANDLING:
    - ModuleNotFoundError: Raised if torch is missing from the active .venv.
    - SystemExit: Triggered upon critical backend failure.
==============================================================================
"""

import torch
import sys


def audit_xpu_environment():
    """
    Diagnoses the PyTorch build and verifies Intel Arc 140T hardware access.
    """
    version = torch.__version__
    xpu_ready = torch.xpu.is_available() if hasattr(torch, 'xpu') else False

    print(f"--- Environment Audit: {version} ---")

    if xpu_ready:
        print(f"STATUS: XPU Backend successfully initialized.")
        print(f"DEVICE: {torch.xpu.get_device_name(0)}")
    else:
        print("STATUS: XPU Backend NOT available.")

        # Diagnostic Check for +cpu builds
        if "+cpu" in version:
            print("\nDIAGNOSTIC: You are running a CPU-only build of PyTorch.")
            print("REMEDY: Run 'pip install torch --index-url https://download.pytorch.org/whl/xpu'")
        else:
            print("\nDIAGNOSTIC: Build supports XPU, but hardware was not detected.")
            print("REMEDY: Verify Intel Graphics Driver 31.0.101.5333+ and BIOS Re-Size BAR.")


if __name__ == "__main__":
    try:
        audit_xpu_environment()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)