#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM-Learn Engine Heartbeat 💓
AUTHOR:  Matha Goram 🛠️
VERSION: 1.0.3
DESCRIPTION: Verifies C++/Python bridge, OpenMP threading, and math accuracy.
LOCATION: /diagnostics/
================================================================================
NOTES:
Run:
    PYTHONPATH=. python3 diagnostics/heartbeat_engine.py
Execution Time:
    On your BeUlta workstation, a 1080p frame (~6.2 million floats)
    should normalize in significantly less than 5ms.
    If it's taking much longer, OpenMP might not be utilizing all cores.
Normalization Accuracy:
    If this fails, there is a mismatch in how the pointers are being handled
    between NumPy and C++.
Import Errors:
    If this fails, your .so file is either named incorrectly or
    isn't in the directory specified by your PYTHONPATH.
PASS:
    Once you see the "PASS" labels, you can confidently
    launch the full video_auditor.py!
"""

import numpy as np
import time
import os
import sys

# Automatically include project root in path to find the .so file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

try:
    import vlm_engine

    print("✅ SUCCESS: vlm_engine library found and imported.")
except ImportError:
    print(f"❌ ERROR: vlm_engine not found in {project_root}.")
    print("Ensure the .so file exists in the project root.")
    exit(1)


def check_engine():
    print("\n" + "---" * 10)
    print("🛠️  ENGINE HEARTBEAT CHECK")
    print("---" * 10)

    # 1. Test compute_mean (Basic Bridge Check) 📊
    test_data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    mean_res = vlm_engine.compute_mean(test_data)
    status = "PASS ✅" if np.isclose(mean_res, 20.0) else "FAIL ❌"
    print(f"📊 Mean Calculation: {status} (Result: {mean_res})")

    # 2. Test normalize_image (Logic & OpenMP Check) ⚡
    # Create a dummy image frame (1080p RGB)
    size = (1080, 1920, 3)
    frame = np.full(size, 255.0, dtype=np.float32)
    mean, std = 0.5, 0.5

    print(f"⚡ Normalizing {frame.size:,} pixels via OpenMP...")

    # We create a flat version for the C++ bridge
    frame_flat = frame.flatten()

    start_time = time.perf_counter()
    vlm_engine.normalize_image(frame_flat, mean, std)
    end_time = time.perf_counter()

    duration = (end_time - start_time) * 1000

    # Check if the flattened array was modified correctly
    success = np.allclose(frame_flat[0:10], 1.0)
    print(f"📐 Normalization Accuracy: {'PASS ✅' if success else 'FAIL ❌'}")
    print(f"⏱️  Execution Time: {duration:.4f} ms")

    # 3. Hardware Threading Hint 🧵
    threads = os.cpu_count()
    print(f"🧵 Available CPU Threads: {threads}")

    print("\n--- 🏁 HEARTBEAT COMPLETE ---\n")


if __name__ == "__main__":
    check_engine()
