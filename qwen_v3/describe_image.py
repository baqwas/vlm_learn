#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
describe_image.py

Multimodal CPU Inference Script using Qwen3-VL-7B-Instruct.

@license: MIT License
@author: ParkCircus Productions (Refactored by Gemini AI)
@version: 1.1.0
@date: 2025-10-28

Dependencies:
- torch (>= 1.13.1)
- transformers (Requires Qwen3-VL support)
- Pillow (PIL)
- requests

Further Reading:
- Qwen3-VL Hugging Face page: https://huggingface.co/models
- Hugging Face Transformers documentation: https://huggingface.co/docs

Run:
- python -m qwen_v3.describe_image
"""
import torch
import os
import requests
import io
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from typing import List, Dict, Any, Tuple

# --- Configuration ---
# Using the smaller 2B model is highly recommended for CPU-only inference.
MODEL_ID: str = "Qwen/Qwen3-VL-2B-Instruct"
# Setting device_map="cpu" forces the entire model to load onto system RAM for CPU-only inference.
DEVICE: str = "cpu"

# FIX APPLIED: Changed to a known-working Qwen-VL example image URL (demo.jpeg)
IMAGE_URL: str = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


def load_model_and_processor(model_id: str, device: str) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Loads the Qwen3-VL model using AutoModelForImageTextToText for CPU inference.

    Args:
        model_id: The Hugging Face model ID.
        device: The device to load the model onto ('cpu').

    Returns:
        A tuple containing the loaded model and processor.
    """
    print(f"Loading model **{model_id}** to **{device}**...")

    # Load the token from the environment variable (if available) for gated models
    hf_token = os.environ.get("HF_TOKEN")

    # device_map="cpu" is critical for CPU-only inference.
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.float32,
        trust_remote_code=True,
        token=hf_token
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    print("✅ Model loaded successfully.")
    return model, processor


def prepare_multimodal_input(image_url: str, prompt_text: str) -> List[Dict[str, Any]]:
    """
    Downloads an image and constructs the chat messages structure.

    Args:
        image_url: URL of the image to download.
        prompt_text: The text prompt for the model.

    Returns:
        The prepared list of messages for the chat template.
    """
    print(f"Downloading image from: {image_url}")
    # Download the image content and open it with PIL
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to download image from {image_url}. Status code: {response.status_code}")

    image = Image.open(io.BytesIO(response.content)).convert('RGB')
    print("✅ Image downloaded and prepared.")

    # Define the user's conversation message with the PIL Image object
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # Pass the PIL Image object
                {"type": "text", "text": prompt_text}
            ],
        }
    ]
    print("✅ Messages prepared.")
    return messages


def generate_response(model: AutoModelForImageTextToText, processor: AutoProcessor,
                      messages: List[Dict[str, Any]]) -> str:
    """
    Processes the input and generates a response from the Qwen3-VL model.

    Args:
        model: The loaded Qwen3-VL model.
        processor: The associated processor.
        messages: The prepared list of chat messages.

    Returns:
        The decoded model's text response.
    """
    # Apply the chat template and tokenize the input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Move inputs to the designated device (in this case, CPU)
    inputs = inputs.to(model.device)

    print("\n🚀 Starting CPU generation (this will take significant time)...")

    # Inference: Generation of the output
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
    )

    # Decode the generated IDs, skipping the input prompt tokens
    response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    return response


def main():
    """
    Main execution function for the Qwen3-VL CPU inference script.
    """
    # --- 1. Load Model and Processor ---
    model, processor = load_model_and_processor(MODEL_ID, DEVICE)

    # --- 2. Prepare Multimodal Input (Image Download + Text) ---
    # REVISED PROMPT for the new demo.jpeg image
    prompt = "Describe the people and animals in this picture."
    messages = prepare_multimodal_input(IMAGE_URL, prompt)

    # --- 3. Process Input and Generate Response ---
    response = generate_response(model, processor, messages)

    # --- 4. Print Output ---
    print("\n" + "=" * 50)
    print("             **Model Response**")
    print("=" * 50)
    print(response.strip())  # Strip whitespace from the response
    print("=" * 50)


if __name__ == "__main__":
    main()

