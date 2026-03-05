#!/usr/bin/env python3
"""
🚀 VLM-Learn Smoke Test Suite
================================================================================
Project:      vlm_learn (Vision Language Model Benchmarking)
Maintainer:   Reza (ParkCircus Productions)
Location:     tests/smoke_test.py
Created:      2026-03-03
Description:  Validates the integrity of the Python environment, patched
              cloud-AI libraries, and hardware acceleration (CUDA).
              Integrated with Pytest markers for selective execution.

Execution:    pytest -m smoke
================================================================================
"""

import sys
import os
import torch
import pytest
from google.cloud import aiplatform


@pytest.mark.smoke
def test_vlm_environment_integrity():
    """
    Validates the basic environment and library versions.
    This test is lightweight and can be run frequently.
    """
    print("\n--- Starting VLM-Learn Smoke Test ---")

    # 1. Verify Patched Libraries
    # Ensures google-cloud-aiplatform is patched per SECURITY.md
    print(f"[🔍] AI Platform Version: {aiplatform.version.__version__}")
    assert aiplatform.version.__version__ >= "1.139.0"

    # 2. Verify Compute Fabric (GPU/NPU)
    cuda_available = torch.cuda.is_available()
    print(
        f"[🔥] CUDA Hardware Acceleration: {'Enabled' if cuda_available else 'Disabled'}"
    )

    # 3. Path Validation
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"[📂] Project Root: {root_path}")
    assert os.path.exists(root_path)

    print("--- ✨ Result: Environment Stable ---")


@pytest.mark.hardware
def test_cuda_device_memory():
    """
    More intensive test requiring physical GPU access.
    Excluded during 'smoke' runs to save time/power.
    """
    if not torch.cuda.is_available():
        pytest.skip("Hardware test skipped: No CUDA device detected.")

    device_name = torch.cuda.get_device_name(0)
    print(f"[📟] Testing Memory on: {device_name}")
    # Simple allocation test
    x = torch.randn(1000, 1000).cuda()
    assert x.is_cuda
