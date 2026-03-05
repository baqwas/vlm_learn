#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
AUTHOR: Matha Goram
LICENSE: MIT
LOCATION: tests/smoke_test.py
PURPOSE: Validates environment integrity, cloud-AI patches, and hardware.
UPDATED: 2026-03-05
================================================================================
"""

import sys
import os
import torch
import pytest
from google.cloud import aiplatform


@pytest.mark.smoke
def test_vlm_environment_integrity():
    """Validates the basic environment and library versions."""
    print("\n--- Starting VLM-Learn Smoke Test ---")
    print(f"[🔍] AI Platform Version: {aiplatform.version.__version__}")
    assert aiplatform.version.__version__ >= "1.139.0"

    cuda_available = torch.cuda.is_available()
    print(
        f"[🔥] CUDA Hardware Acceleration: {'Enabled' if cuda_available else 'Disabled'}"
    )

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.exists(root_path)
    print("--- ✨ Result: Environment Stable ---")


@pytest.mark.vlm_hardware
@pytest.mark.hardware
def test_cuda_device_memory():
    """
    Intensive test requiring physical GPU access.
    Automatically skipped on GitHub Runners via vlm_hardware marker.
    """
    if not torch.cuda.is_available():
        pytest.skip("Hardware test skipped: No CUDA device detected.")

    device_name = torch.cuda.get_device_name(0)
    print(f"[📟] Testing Memory on: {device_name}")
    x = torch.randn(1000, 1000).cuda()
    assert x.is_cuda
