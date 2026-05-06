#!/usr/bin/env python3
"""
==============================================================================
PROJECT: Qwen2-VL Hardware Comparative Benchmark (Secure Version)
AUTHOR:  Abu Reza M Wajih
LICENSE: MIT License (c) 2026
VERSION: 1.1.0

UPDATE TRACKING:
    2026-05-05: v1.1.0 - Integrated dotenv for secure HF_TOKEN handling.
                       - Resolved Arc 140T memory query limitation.
==============================================================================
"""

import os
import torch
import time
import sys
from dotenv import load_dotenv
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def run_comparative_benchmark():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    try:
        print("--- Initializing Multimodal Processor ---")
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type": "text", "text": "What is in this image?"}
            ]
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        # --- PHASE 1: INTEL ARC 140T (GPU) ---
        print("\n[PHASE 1] Testing: Intel Arc 140T (XPU)")

        # FIX: We use device="xpu" directly instead of device_map="auto"
        # to prevent 'accelerate' from trying to query VRAM stats.
        model_gpu = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            token=hf_token
        ).to("xpu")

        inputs_gpu = processor(text=[text], images=image_inputs, return_tensors="pt").to("xpu")

        start_gpu = time.perf_counter()
        # Generate 20 tokens to measure speed
        out_gpu = model_gpu.generate(**inputs_gpu, max_new_tokens=20)
        torch.xpu.synchronize()
        end_gpu = time.perf_counter()

        gpu_time = end_gpu - start_gpu
        print(f"GPU Latency: {gpu_time:.4f}s")

        # Explicitly clear GPU memory
        del model_gpu
        torch.xpu.empty_cache()

        # --- PHASE 2: CORE ULTRA (CPU) ---
        print("\n[PHASE 2] Testing: Core Ultra (CPU)")
        model_cpu = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            token=hf_token
        ).to("cpu")

        inputs_cpu = processor(text=[text], images=image_inputs, return_tensors="pt").to("cpu")

        start_cpu = time.perf_counter()
        out_cpu = model_cpu.generate(**inputs_cpu, max_new_tokens=20)
        end_cpu = time.perf_counter()

        cpu_time = end_cpu - start_cpu
        print(f"CPU Latency: {cpu_time:.4f}s")

        print("\n--- FINAL COMPARISON ---")
        print(f"Arc 140T is {cpu_time / gpu_time:.2f}x faster than CPU.")

    except Exception as e:
        print(f"CRITICAL BENCHMARK ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_comparative_benchmark()