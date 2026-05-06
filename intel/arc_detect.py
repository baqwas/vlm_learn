#!/usr/bin/env python3
"""
==============================================================================
PROJECT: Intel Arc GPU Capability Audit
AUTHOR:  Matha Goram
LICENSE: MIT License
COPYRIGHT: (c) 2026 ParkCircus Productions
REFERENCE: https://www.intel.com/content/www/us/en/developer/tools/oneapi/pytorch.html
VERSION: 1.0.1

UPDATE TRACKING:
    2026-05-05: v1.0.0 - Initial deployment for Intel Arc 140T (Arrow Lake).
    2026-05-05: v1.0.1 - Added granular error handling for missing XPU drivers.

PROCESSING WORKFLOW:
    1. Import PyTorch with XPU (Intel) backend support.
    2. Query hardware abstraction layer for Intel-specific device counts.
    3. Extract and format device metadata (VRAM, Compute Units).
    4. Output hardware profile to console.

USER INTERFACE:
    Console-based output displaying hardware identification and VRAM capacity.

EXCEPTION HANDLING:
    - AttributeError: Caught if torch version lacks XPU attribute.
    - RuntimeError: Caught if drivers are present but hardware is unresponsive.
==============================================================================
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to
the following conditions: The above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
==============================================================================
"""

import torch
import sys


def test_intel_gpu():
    """
    Identifies and audits Intel Arc 140T GPU hardware capabilities.
    """
    print("--- Intel Arc 140T Capability Audit ---")

    try:
        # Check for XPU (Intel GPU) availability in the PyTorch environment
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            print(f"Status: Intel GPU hardware acceleration is ACTIVE.")

            for i in range(device_count):
                name = torch.xpu.get_device_name(i)
                props = torch.xpu.get_device_properties(i)

                # Format output for professional technical logs
                vram_gb = props.total_memory / (1024 ** 3)
                print(f"\n[Device ID {i}]: {name}")
                print(f"  - Total Dedicated VRAM: {vram_gb:.2f} GB")
                print(f"  - Execution Units (EU): {props.max_compute_units}")
        else:
            print("ERROR: Intel XPU backend not found.")
            print("REMEDY: Ensure 'intel-extension-for-pytorch' or 'torch-xpu' is installed.")

    except AttributeError:
        print("EXCEPTION: Current PyTorch installation does not support XPU attributes.")
    except Exception as e:
        print(f"CRITICAL SYSTEM ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    test_intel_gpu()