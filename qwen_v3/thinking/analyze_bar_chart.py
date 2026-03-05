#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_bar_chart.py

================================================================================
PROJECT: Multimodal Reasoning for Chart Analysis (CPU-Only)
AUTHOR: Matha Goram
DATE: 2025-12-01
VERSION: 1.0.0
================================================================================

MIT License

Copyright (c) 2025 ParkCircus Productions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os
from typing import Union

# --- Configuration for CPU-Only and Thinking Model ---
# The smallest reasoning-enhanced model for complex multimodal tasks.
MODEL_ID: str = 'Qwen/Qwen3-VL-2B-Thinking'

# Forces the model to load and run on the system's CPU/RAM.
DEVICE: str = 'cpu'

# Use float32 for maximum CPU compatibility, which requires more RAM (approx. 8GB+).
DTYPE = torch.float32

# Path to the image file - the Apple Quarterly Revenue chart
CHART_PATH: str = '../images/apple.png'


# --- Core Inference Function ---

def run_chart_analysis(chart_path: str, chart_prompt: str, model_id: str, device: str, dtype: torch.dtype) -> None:
    """
    Loads the Qwen3-VL-Thinking model and performs a multimodal reasoning task
    on a bar chart using a CPU-only configuration.

    Args:
        chart_path (str): The local file path to the chart image.
        chart_prompt (str): The text prompt for the reasoning task.
        model_id (str): HuggingFace ID of the model (e.g., Qwen/Qwen3-VL-2B-Thinking).
        device (str): The device to run on ('cpu' or 'cuda').
        dtype (torch.dtype): The tensor data type for model weights.
    """

    if not os.path.exists(chart_path):
        print(f"🚨 ERROR: Chart file not found at '{chart_path}'. Please generate the chart or adjust the path.")
        return

    # 1. Load Image
    try:
        chart_image = Image.open(chart_path).convert("RGB")
    except Exception as e:
        print(f"🚨 ERROR: Failed to load image from '{chart_path}'. Details: {e}")
        return

    print(f"🖼️ Loaded Chart: {chart_path}")
    print(f"🧠 Loading Model: {model_id} onto {device} with {dtype}...")

    # 2. Load Model and Processor (CPU configuration)
    try:
        # device_map="auto" is recommended for multi-device, but for 'cpu'
        # it forces loading onto the CPU. trust_remote_code=True is essential.
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"🚨 FATAL MODEL LOAD ERROR: {e}")
        print("Ensure you have all dependencies installed and sufficient RAM (16GB+).")
        return

    # 3. Construct Multimodal Prompt (Chat Format)
    # The Qwen models use a specific chat template for multimodal input.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": chart_image},  # Visual Input
                {"type": "text", "text": chart_prompt}  # Textual Query
            ]
        }
    ]

    # Apply the chat template and tokenize the input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Move inputs to the designated device (CPU)
    inputs = inputs.to(model.device)

    print("\n⏳ Starting CPU inference. Expect a very long latency (minutes per token)...")

    # 4. Inference: Generation of the Output
    try:
        # Low temperature and do_sample=False is ideal for logical reasoning
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.01,
        )
    except Exception as e:
        print(f"\n🚨 FATAL INFERENCE ERROR: {e}")
        print("Inference failed, likely due to out-of-memory or a very long timeout on CPU.")
        return

    # Decode the generated IDs, skipping the input prompt tokens
    generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):].tolist()
    response = processor.decode(generated_ids_trimmed, skip_special_tokens=True)

    # 5. Print Output
    print("\n" + "=" * 80)
    print(f"REQUESTED PROBLEM: {chart_prompt}")
    print("\n🤖 Qwen3-VL Reasoning Response (including <think> trace):")
    # Qwen3-Thinking model output may contain a detailed internal <think>...</think> block
    print(response.strip())
    print("=" * 80)


# --- Main Execution ---

if __name__ == "__main__":

    # Example prompt for the Apple Quarterly Revenue Chart
    CHART_ANALYSIS_PROMPT = (
        "Analyze the provided bar chart and explain the quarter-over-quarter "
        "trend in Apple's revenue for Fiscal Year 2024. Which quarter had the "
        "highest revenue, and what is the difference between the highest and "
        "lowest reported revenue quarters?"
    )

    try:
        run_chart_analysis(
            chart_path=CHART_PATH,
            chart_prompt=CHART_ANALYSIS_PROMPT,
            model_id=MODEL_ID,
            device=DEVICE,
            dtype=DTYPE
        )
    except Exception as e:
        print(f"\nAn unhandled error occurred during execution: {e}")
