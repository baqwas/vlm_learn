#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
AUTHOR: Matha Goram
LICENSE: MIT
LOCATION: tests/test_qwen_inference.py
PURPOSE: Verification of Qwen model loading and inference capabilities.
UPDATED: 2026-03-05
================================================================================
"""

import pytest
import torch
from transformers import AutoModelForCausalLM


@pytest.mark.vlm_hardware
@pytest.mark.timeout(300)
def test_model_loading():
    """
    Verifies model instantiation.
    Skipped in CI to prevent timeout and OOM issues on non-GPU runners.
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device, torch_dtype=torch.float32
    )
    assert model is not None
