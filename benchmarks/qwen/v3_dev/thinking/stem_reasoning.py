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
"""
# -*- coding: utf-8 -*-
stem_reasoning.py

Qwen3-VL-2B-Thinking (Smallest Reasoning Model) CPU Inference for STEM/Math Problems.
---------------------------------------------------------------------------------

### Description:
This script demonstrates complex, multimodal reasoning using the Qwen3-VL-2B-Thinking
large language model from Hugging Face Transformers. It is specifically configured
for **CPU-only inference** and is optimized for detailed, step-by-step logical reasoning
on STEM and mathematical problems presented in an image.

### Author & Copyright:
* **Author**: Matha Goram
* **Email**: baqwas@yahoo.com
* **Created**: 2025-11-11
* **License**: MIT License
* **Copyright**: (c) 2025 ParkCircus Productions

### Configuration:
* **MODEL_ID**: Qwen/Qwen3-VL-2B-Thinking
* **DEVICE**: "cpu"

This script demonstrates complex, multimodal reasoning using the Qwen3-VL-2B-Thinking
large language model from Hugging Face Transformers. It is specifically configured
for **CPU-only inference** and uses the "Thinking" variant of the model, which is
optimized for detailed, step-by-step logical reasoning, particularly useful for
STEM and mathematical problems presented in an image.

NOTE: CPU-only inference with this model is **extremely slow** and memory-intensive
compared to GPU inference. The script forces `torch.float32` for maximum CPU
compatibility.

### Configuration:
* **MODEL_ID**: Qwen/Qwen3-VL-2B-Thinking (Smallest model with enhanced reasoning capabilities)
* **DEVICE**: "cpu" (Force CPU inference)
* **DTYPE**: torch.float32 (Maximizes CPU compatibility and stability)
* **Input**: Requires a local image file (`LOCAL_IMAGE_PATH`) containing a STEM/Math problem (e.g., a geometry figure, circuit diagram).

### Functions:
1.  `load_local_image(file_path: str) -> Image.Image`:
    * Reads an image from a local file path using PIL.
    * **Raises**: `FileNotFoundError` if the path is invalid.
2.  `download_and_prepare_image(url: str) -> Image.Image`:
    * (Currently commented out/not used in the main block) Downloads an image from a URL and prepares it as a PIL Image object.
3.  `run_stem_reasoning(model_id: str, device: str, image_path: str, math_prompt: str)`:
    * Main function to load the model, process the image and text prompt, and perform inference.
    * Constructs a chat-like input message with the image and the detailed math prompt.
    * Sets generation parameters for logical output (`do_sample=False`, `temperature=0.01`).
    * Prints the model's step-by-step reasoning response.

### Usage:
1.  Ensure you have the necessary libraries installed (`torch`, `transformers`, `Pillow`, `requests`).
2.  Set `LOCAL_IMAGE_PATH` to a valid path for a STEM/Math problem image on your system.
3.  Modify the `STEM_PROMPT` for the specific question you want the model to answer based on the image.
4.  Run the script: `python stem_reasoning.py`

### Example Problem (Placeholder):
The script is configured to use an image at `LOCAL_IMAGE_PATH` and asks the model to:
> "Calculate the total area of the figures shown in the diagram. Show your work."
"""
import os

import torch
import requests
import io
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# --- Configuration for CPU-Only and Thinking Model ---
# Smallest Thinking model variant for enhanced reasoning
MODEL_ID: str = "Qwen/Qwen3-VL-2B-Thinking"
# Force model loading onto system RAM for CPU-only inference
DEVICE: str = "cpu"
# Use float32 for maximum CPU compatibility, even if it uses more memory
DTYPE = torch.float32

# URL for a sample STEM/Math diagram or problem image
# NOTE: Replace this with a URL of a math or science problem image for real testing
# Example uses a generic diagram to demonstrate the input structure
# IMAGE_URL: str = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/qwen_vl_example.png"
# IMAGE_URL: str = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%2Fid%2FOIP.a7l-qmhc0jgbFhMi9j5wAgHaHa%3Fcb%3Ducfimgc2%26pid%3DApi&f=1&ipt=451cf4a2a6f8a19061c238e015c7f6059b2f6775ada691d50ba27b57da297d2e&ipo=images"
IMAGE_URL: str = (
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%2Fid%2FOIP.XH4dq3OGiKkdCkDBiDkAOgHaJl%3Fpid%3DApi&f=1&ipt=2ab3f3b00c630efa29090002069be5241a70765e4a83f41de7530e3cf1f49029&ipo=images"
)

# Placeholder for a local STEM/Math diagram or problem image
# NOTE: Replace 'path/to/your/local/image.png' with the actual file path on your system.
# The image must be accessible from where you run the script.
LOCAL_IMAGE_PATH: str = "../images/composite_area_figure.png"


def load_local_image(file_path: str) -> Image.Image:
    """Reads an image from a local file path and returns a PIL Image object."""
    print(f"Attempting to load image from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: Local image file not found at path: {file_path}"
        )

    try:
        # Use PIL's Image.open() for local file reading
        # .convert('RGB') ensures a consistent image format
        image = Image.open(file_path).convert("RGB")
        print("✅ Image loaded successfully.")
        return image
    except Exception as e:
        print(f"Error loading image file: {e}")
        raise


def download_and_prepare_image(url: str) -> Image.Image:
    """Downloads an image from a URL and returns a PIL Image object."""
    print(f"Downloading image from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        raise


def run_stem_reasoning(model_id: str, device: str, image_path: str, math_prompt: str):
    """Loads the model and performs the multimodal reasoning inference."""

    # --- 1. Load Model and Processor ---
    print(f"\n🧠 Loading model {model_id} to {device} (requires significant RAM)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, device_map=device, dtype=DTYPE, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("✅ Model loaded successfully.")

    # --- 2. Prepare Multimodal Input (Image + Text) ---
    # image = download_and_prepare_image(image_url)
    image = load_local_image(image_path)

    # Define the conversation structure
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # The image object
                # The prompt asks for step-by-step reasoning
                {
                    "type": "text",
                    "text": f"Calculate the total area of the figures shown in the diagram. Show your work.",
                },
            ],
        }
    ]

    # --- 3. Process Input and Generate Response ---
    # Apply the chat template and tokenize the input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move inputs to the designated device (CPU)
    inputs = inputs.to(model.device)

    print("\n⏳ Starting CPU inference. Expect long latency...")

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,  # Increased tokens for detailed reasoning
        do_sample=False,  # Prefer deterministic (reasoned) output
        temperature=0.01,  # Low temperature for logical precision
        # Qwen Thinking models are designed for this type of detailed, logical output.
    )

    # Decode the generated IDs, skipping the input prompt tokens
    generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]) :].tolist()
    response = processor.decode(generated_ids_trimmed, skip_special_tokens=True)

    # --- 4. Print Output ---
    print("\n" + "=" * 50)
    print(f"REQUESTED PROBLEM: {math_prompt}")
    print("\n🤖 Qwen3-VL Reasoning Response:")
    print(response.strip())
    print("=" * 50)


if __name__ == "__main__":
    # Example complex math/STEM question
    STEM_PROMPT = "Given the circuit diagram in the image, determine the equivalent resistance between points A and B, showing all steps."

    try:
        run_stem_reasoning(MODEL_ID, DEVICE, LOCAL_IMAGE_PATH, STEM_PROMPT)
    except Exception as e:
        print(f"An error occurred: {e}")
