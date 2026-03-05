#!/usr/bin/env python
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
Generate text descriptions for an image using Qwen2.5-VL-7B-Instruct
via the Hugging Face transformers pipeline, with a local image file.

References:
    - Hugging Face Qwen2.5-VL-7B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
    - Hugging Face pipeline documentation for image-text-to-text:
      (The general structure is similar to visual-question-answering pipeline examples)

Prerequisites:
    - python -m pip install transformers torch pillow accelerate
    - Ensure you have a local image file in the specified path (e.g., "../images/docks.jpg")
"""
import torch
from transformers import pipeline
from PIL import Image
import os


def text_image_pipeline_local():
    # 1. Define the model ID
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    # 2. Create the pipeline
    print(f"Loading pipeline for {model_id}...")
    try:
        # device=0 attempts to use GPU if available, otherwise it will fall back to CPU if device_map="auto" is used
        # torch_dtype=torch.bfloat16 is recommended for performance on compatible hardware
        pipe = pipeline(
            task="image-text-to-text",
            model=model_id,
            device=(
                0 if torch.cuda.is_available() else -1
            ),  # Use GPU (device 0) if available, else CPU (-1)
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float32
            ),
        )
        print("Pipeline loaded successfully!")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print(
            "Ensure you have installed all prerequisites and your environment is correctly set up for GPU if attempting to use it."
        )
        return

    # 3. Prepare an image (read from local project folder)
    image_dataset = (
        "ocrssp.png"  # Make sure this image exists in the ../images/ directory
    )
    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, "../images", image_dataset)
    # Define the queryfor the image
    query = "What are the key differences in algorithms for hyperbolic tangent calculations using single or double precision arithmetic?"

    print(f"Loading image from: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded from local path: {image_path}")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}.")
        print(
            "Please ensure you have an '../images' folder in the same directory as this script,"
        )
        print(f"and '{image_dataset}' is inside it.")
        return
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return

    # 4. Define the question/prompt for the image
    # The pipeline expects a list of messages for multimodal input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # Pass the PIL Image object directly
                {"type": "text", "text": query},
            ],
        }
    ]

    print(f"\nAsking the model to {query} ...")
    # 5. Run the pipeline
    # Pass the messages directly to the 'text' argument of the pipeline call
    # max_new_tokens controls the length of the generated description
    # return_full_text=False ensures only the generated text is returned, not the full prompt + generated text
    try:
        results = pipe(text=messages, max_new_tokens=50, return_full_text=False)

        # 6. Display the results
        print("\n--- Image Description Result ---")
        if results and isinstance(results, list) and len(results) > 0:
            # The result is typically a list of dictionaries, each with a 'generated_text' key
            # Access the first item in the list and then the 'generated_text' key
            generated_description = results[0].get(
                "generated_text", "No description found."
            )
            print(f"Description: {generated_description}")
        else:
            print("No description could be generated.")

    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")


if __name__ == "__main__":
    text_image_pipeline_local()
