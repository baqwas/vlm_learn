#!/usr/bin/env python3
"""
==============================================================================
PROJECT: GPU Compute Stress Test (GEMM Benchmark)
AUTHOR:  Matha Goram
LICENSE: MIT License
COPYRIGHT: (c) 2026 ParkCircus Productions
VERSION: 2.1.0

UPDATE TRACKING:
    2026-05-05: v2.1.0 - Optimized for FP16 precision to match Arc 140T architecture.
                       - Integrated torch.xpu.synchronize() for accurate timing.

PROCESSING WORKFLOW:
    1. Validate XPU device presence.
    2. Initialize high-dimensional tensors (4096^2) directly on GPU memory.
    3. Execute 'Warm-up' pass to initialize GPU kernels.
    4. Run 50 iterations of matrix multiplication.
    5. Synchronize device threads and calculate execution latency.

USER INTERFACE:
    Terminal progress output with final performance metrics in milliseconds.

EXCEPTION HANDLING:
    - torch.OutOfMemoryError: Triggered if matrix size exceeds available iGPU VRAM.
    - KeyboardInterrupt: Allows graceful exit during long stress tests.
==============================================================================
"""

import torch
import time
import sys


def benchmark_arc_compute():
    """
    Performs high-load matrix multiplication to stress-test Arc 140T throughput.
    """
    print("--- GPU Stress Test: GEMM Performance ---")

    try:
        if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
            raise RuntimeError("Hardware acceleration (XPU) is unavailable.")

        device = torch.device("xpu")
        print(f"Targeting: {torch.xpu.get_device_name(0)}")

        # Configure workload (4096 x 4096 Matrix)
        matrix_dim = 4096
        print(f"Initializing {matrix_dim}x{matrix_dim} tensors in FP16...")

        # Allocate tensors directly to XPU
        alpha = torch.randn(matrix_dim, matrix_dim, device=device, dtype=torch.float16)
        beta = torch.randn(matrix_dim, matrix_dim, device=device, dtype=torch.float16)

        # Warm-up: Essential for JIT kernel compilation
        _ = torch.matmul(alpha, beta)
        torch.xpu.synchronize()

        print("Executing 50 compute iterations...")
        iterations = 50
        start_mark = time.perf_counter()

        for _ in range(iterations):
            _ = torch.matmul(alpha, beta)

        # Synchronize ensures all GPU tasks are finished before stopping the clock
        torch.xpu.synchronize()
        end_mark = time.perf_counter()

        # Calculate Results
        total_time = end_mark - start_mark
        avg_latency = (total_time / iterations) * 1000

        print("\nBENCHMARK RESULTS:")
        print(f"  - Total Elapsed: {total_time:.4f}s")
        print(f"  - Average Latency: {avg_latency:.2f} ms per pass")

    except torch.OutOfMemoryError:
        print("ERROR: Matrix size too large for available GPU memory.")
    except RuntimeError as e:
        print(f"HARDWARE ERROR: {e}")
    except KeyboardInterrupt:
        print("\nUser aborted benchmark.")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {type(e).__name__} - {e}")


if __name__ == "__main__":
    benchmark_arc_compute()