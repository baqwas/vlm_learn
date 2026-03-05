#!/usr/bin/env python3
"""
test_engine_bridge.py
"""
import pytest
import numpy as np
import time
import sys
from pathlib import Path

# ==============================================================================
# NATIVE EXTENSION ADAPTATION
# ==============================================================================
# Force Python to look in the project root for the compiled .so file.
# This bypasses issues where the .venv might shadow local imports.
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

try:
    import vlm_engine
except ImportError as e:
    pytest.fail(
        f"❌ Could not load vlm_engine.\n"
        f"Target Path: {root_path}\n"
        f"Error: {e}\n"
        f"Hint: Run 'make' in the build directory and ensure vlm_engine.so exists."
    )


# ==============================================================================
# FUNCTIONAL TESTS
# ==============================================================================


@pytest.mark.engine
def test_compute_mean_accuracy():
    """Verify C++ mean calculation matches NumPy's reference implementation."""
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    expected = np.mean(data)

    result = vlm_engine.compute_mean(data)

    assert result == pytest.approx(expected, rel=1e-5)


@pytest.mark.engine
def test_normalize_in_place():
    """Verify C++ normalization correctly modifies the buffer in-place."""
    # Create a simple 2x2 image
    img = np.array([0.0, 127.5, 255.0, 100.0], dtype=np.float32)
    mean, std = 0.5, 0.5

    # Expected: ((x/255) - 0.5) / 0.5
    # 0.0   -> (0.0 - 0.5) / 0.5 = -1.0
    # 255.0 -> (1.0 - 0.5) / 0.5 =  1.0

    vlm_engine.normalize_image(img, mean, std)

    assert img[0] == pytest.approx(-1.0)
    assert img[2] == pytest.approx(1.0)


# ==============================================================================
# PERFORMANCE BENCHMARKS
# ==============================================================================


@pytest.mark.benchmark
def test_normalization_performance():
    """Verify C++ normalization is significantly faster than Python loops."""
    # 1. Setup: 1MP Image (1000x1000 pixels, 3 channels)
    size = (1000, 1000, 3)
    img_cpp = np.random.randint(0, 256, size).astype(np.float32)
    img_py = img_cpp.copy()

    mean, std = 0.481, 0.268

    # 2. Benchmark Pure Python (List Comprehension)
    # We simulate the overhead of iterating over pixels in Python
    start_py = time.perf_counter()
    flat_data = img_py.flatten()
    _ = [((x / 255.0) - mean) / std for x in flat_data]
    end_py = time.perf_counter()
    py_duration = end_py - start_py

    # 3. Benchmark C++ Native Extension
    start_cpp = time.perf_counter()
    vlm_engine.normalize_image(img_cpp, mean, std)
    end_cpp = time.perf_counter()
    cpp_duration = end_cpp - start_cpp

    # 4. Results & Assertion
    speedup = py_duration / cpp_duration

    print(f"\n[📊] BENCHMARK: 1,000,000 Pixels (RGB)")
    print(f"    - Python Loop: {py_duration:.4f}s")
    print(f"    - C++ Native:  {cpp_duration:.4f}s")
    print(f"    - Speedup:     {speedup:.1f}x faster")

    # On BeUlta, we expect at least a 20x improvement over standard loops
    assert speedup > 20.0
