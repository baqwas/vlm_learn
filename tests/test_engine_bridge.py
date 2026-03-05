#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
AUTHOR: Matha Goram
LICENSE: MIT
LOCATION: tests/test_engine_bridge.py
PURPOSE: Interface validation for C++ Native Extensions (vlm_engine).
UPDATED: 2026-03-05
================================================================================
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

# Force Project Root into Path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))


@pytest.mark.vlm_hardware
@pytest.mark.engine
def test_compute_mean_accuracy():
    """Verify C++ mean calculation against NumPy reference."""
    import vlm_engine  # Deferred import to allow skipping before failure

    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    expected = np.mean(data)
    result = vlm_engine.compute_mean(data)
    assert result == pytest.approx(expected, rel=1e-5)


@pytest.mark.vlm_hardware
@pytest.mark.engine
def test_normalize_in_place():
    """Verify C++ normalization modifies buffer in-place."""
    import vlm_engine

    img = np.array([0.0, 127.5, 255.0, 100.0], dtype=np.float32)
    vlm_engine.normalize_image(img, 0.5, 0.5)
    assert img[0] == pytest.approx(-1.0)
    assert img[2] == pytest.approx(1.0)


@pytest.mark.vlm_hardware
@pytest.mark.benchmark
def test_normalization_performance():
    """Verify C++ speedup vs Python loops (Expect > 20x)."""
    import vlm_engine

    size = (1000, 1000, 3)
    img_cpp = np.random.randint(0, 256, size).astype(np.float32)
    img_py = img_cpp.copy()
    mean, std = 0.481, 0.268

    # Python Benchmark
    start_py = time.perf_counter()
    _ = [((x / 255.0) - mean) / std for x in img_py.flatten()]
    py_duration = time.perf_counter() - start_py

    # C++ Benchmark
    start_cpp = time.perf_counter()
    vlm_engine.normalize_image(img_cpp, mean, std)
    cpp_duration = time.perf_counter() - start_cpp

    speedup = py_duration / cpp_duration
    assert speedup > 20.0
