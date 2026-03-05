#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate text based on Visual Question Answering on an image with [Pipeline]

References
    https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_5_vl.md

Prerequisites
    python -m pip install transformers torch pillow accelerate
    python -m pip install qwen-vl-utils # This might install torchvision by default for image processing
    python -m pip install decord # Or torchcodec for faster video processing
"""
import torch
from transformers import pipeline
from PIL import Image
import os


def text_image_pipeline_local():
    # 1. Define the model ID
    # model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    # model_id = "Qwen/Qwen3-Embedding-0.6B"

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
        "docks.jpg"  # Make sure this image exists in the ../images/ directory
    )
    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, "../images", image_dataset)
    query = "Describe the image in detail."

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

    print(f"Asking the model to {query}...")
    # 5. Run the pipeline
    # Pass the messages directly to the 'text' argument of the pipeline call
    # max_new_tokens controls the length of the generated description
    # return_full_text=False ensures only the generated text is returned, not the full prompt + generated text
    try:
        results = pipe(text=messages, max_new_tokens=50, return_full_text=False)

        # 6. Print the results
        print(f"\n--- Image {image_path} Result ---")
        if results and isinstance(results, list) and len(results) > 0:
            # The result is typically a list of dictionaries, each with a 'generated_text' key
            print(
                f"Response: {results[0].get('generated_text', 'No description found.')}"
            )
        else:
            print(f"No result could be generated using {model_id}.")

    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")


if __name__ == "__main__":
    text_image_pipeline_local()
