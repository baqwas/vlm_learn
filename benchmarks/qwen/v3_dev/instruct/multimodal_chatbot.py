#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
VERSION: 1.0.1
UPDATED: 2026-03-05 08:13:21
COPYRIGHT: (c) 2026 ParkCircus Productions; All Rights Reserved.
AUTHOR: Matha Goram
LICENSE: MIT
PURPOSE: [REPLACE WITH FILE DESCRIPTION]
================================================================================
"""
# -*- coding: utf-8 -*-
"""
multimodal_chatbot.py

================================================================================
PROJECT: Qwen-VL Multimodal Chatbot Client (HUGGINGFACE INFERENCE API)
AUTHOR: AI Assistant
DATE: 2025-11-30
VERSION: 1.4.0 (Added explicit proxy handling)
================================================================================
# ... (omitted license and header comments)

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multimodal_chatbot.py

================================================================================
PROJECT: Qwen-VL Multimodal Chatbot Client (LOCAL HUGGINGFACE TRANSFORMERS)
AUTHOR: AI Assistant
DATE: 2025-11-30
VERSION: 2.1.0 (Refactored for local CPU-only inference)
================================================================================
# This script loads the Qwen3-VL-2B-Instruct model locally using the
# HuggingFace transformers library for direct, interactive inference on CPU.
"""

import os
import argparse
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image


# --- Utility Functions ---


def _strip_quotes(value):
    """Utility function to defensively strip leading/trailing quotes."""
    if isinstance(value, str):
        if value.startswith('"') and value.endswith('"'):
            return value.strip('"')
        elif value.startswith("'") and value.endswith("'"):
            return value.strip("'")
    return value


# --- Core Inference Function (Based on user's snippet) ---


def run_local_qwen_query(image_path, text_prompt):
    """
    Loads Qwen3-VL and runs the multimodal query locally using the
    HuggingFace transformers pipeline.
    """

    # print("Loading Qwen3-VL-2B-Instruct model and processor...")
    # print("Loading Qwen3-VL-4B-Instruct model and processor...")
    print("Loading Qwen3-VL-8B-Instruct model and processor...")
    print("--- Running in CPU-ONLY mode ---")

    try:
        # Load the model and explicitly move it to CPU
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            dtype=torch.float32,  # Use float32 for CPU compatibility
            device_map=None,  # Disable automatic device mapping (e.g., CUDA)
        ).to(
            "cpu"
        )  # Explicitly move model to CPU

        # processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    except Exception as e:
        print(
            f"Error loading model or processor. Please ensure you have transformers and torch installed, and check your environment for local model requirements. Error: {e}"
        )
        return

    print(f"Loading image from: {image_path}")
    # Load the local image using PIL
    try:
        local_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image at '{image_path}': {e}")
        return

    # Prepare the chat messages in the required format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    # Pass the PIL Image object directly to the processor
                    "image": local_image,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    # Preparation for inference
    print("Preparing inputs for inference...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move inputs to the model's device (which is now guaranteed to be 'cpu')
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    print("Generating response (Inference)...")
    # Added parameters for generation control
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,  # Increased for a complete response
        do_sample=False,  # For deterministic output
    )

    # Trim the input prompt tokens to get only the new response
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the output
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[
        0
    ]  # Take the first and only result

    print("\n--- Qwen-VL Local Response (via transformers) ---")
    print(output_text.strip())
    print("-------------------------------------------------")


# --- Main Execution ---

if __name__ == "__main__":
    # Hardcoded defaults since config.toml is no longer used for API settings
    DEFAULT_IMAGE_PATH = "/qwen_v3/images/Istanbul Haydari.jpg"
    DEFAULT_TEXT_PROMPT = (
        "What is the main subject of this image, and what cuisine does it convey?"
    )

    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Local multimodal chatbot client for Qwen3-VL Instruct (using transformers)."
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Specify the path to the input image file."
    )
    args = parser.parse_args()

    # 2. Determine the image path (CLI overrides default)
    image_path = args.input if args.input else DEFAULT_IMAGE_PATH
    image_path = _strip_quotes(image_path)

    print(f"DEBUG: Final image_path for existence check: '{image_path}'")
    print(f"Image Path: {image_path}")

    # 3. Check if the determined image file exists
    if not os.path.exists(image_path):
        print(f"🚨 CRITICAL: The image file '{image_path}' was not found.")
        print(
            "Please ensure the file exists or specify a correct path using the -i/--input flag."
        )
    else:
        print(f"\nUser Prompt: {DEFAULT_TEXT_PROMPT}")

        # 4. Run the local query
        run_local_qwen_query(image_path, DEFAULT_TEXT_PROMPT)
