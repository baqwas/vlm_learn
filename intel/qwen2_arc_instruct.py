#!/usr/bin/env python3
"""
==============================================================================
PROJECT: Qwen-VL Inference (GPU/XPU)
AUTHOR:  Abu Reza M Wajih
LICENSE: MIT License (c) 2026
==============================================================================
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time

def run_qwen_gpu():
    print("--- Starting Qwen2-VL-2B (Arc GPU/XPU) ---")
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    device = torch.device("xpu")

    # 1. Load Model & Processor
    # We use float16 as it is the native high-speed format for Arc GPUs
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # 2. Prepare Multimodal Input
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {"type": "text", "text": "Describe this image in one sentence."}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)

    # 3. Inference & Timing
    start = time.perf_counter()
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    torch.xpu.synchronize() # Wait for GPU to finish
    end = time.perf_counter()

    # 4. Output Results
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(f"\nGPU Response: {output_text[0]}")
    print(f"GPU Latency: {end - start:.4f}s")

if __name__ == "__main__":
    run_qwen_gpu()