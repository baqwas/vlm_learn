#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
describe_multiple.py

Multimodal CPU Inference Script for processing multiple images from a text file.
It utilizes the Hugging Face recommended AutoModelForImageTextToText and
the Qwen/Qwen3-VL-2B-Instruct model. Responses, along with model and runtime
parameters, are saved to an external JSON file for evaluation purposes, with
a timestamp automatically appended to the output filename.

@license: MIT License
@author: ParkCircus Productions
@version: 1.1.0
@date: 2025-10-28

Dependencies:
- torch (>= 1.13.1)
- transformers (Requires Qwen3-VL support)
- Pillow (PIL)
- requests
- argparse (Standard Python library)
- json (Standard Python library)
- datetime (Standard Python library)

Further Reading:
- Qwen3-VL Hugging Face page: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- Hugging Face Transformers documentation: https://huggingface.co/docs

Usage:
  # Example run:
  # If executed at 2025-10-28 21:50:15, the output file will be:
  # results_20251028_215015.json
  python describe_multiple.py

  # If run with a custom name:
  # output_data_20251028_215015.json
  python describe_multiple.py --output_file output_data.json
"""
import torch
import os
import requests
import io
import argparse
import json
from datetime import datetime
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from typing import List, Dict, Any, Tuple

# --- Configuration Constants ---
MODEL_ID: str = "Qwen/Qwen3-VL-2B-Instruct"
DEVICE: str = "cpu"
# --- Default Values for Command Line Arguments ---
DEFAULT_PROMPT: str = "Describe the people and animals in this picture."
DEFAULT_INPUT_FILE: str = "image_urls.txt"
DEFAULT_OUTPUT_FILE: str = "results.json"
MAX_NEW_TOKENS: int = 40
TEMPERATURE: float = 0.7
TOP_P: float = 0.8


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


def download_image(image_url: str) -> Image.Image:
    """
    Downloads an image from a URL and returns a PIL Image object.
    """
    print(f"Downloading image from: {image_url}")
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to download image from {image_url}. Status code: {response.status_code}")

    try:
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise Image.UnidentifiedImageError(f"Could not identify image from {image_url}: {e}")

    return image


def prepare_multimodal_input(image: Image.Image, prompt_text: str) -> List[Dict[str, Any]]:
    """
    Constructs the chat messages structure using a PIL Image object.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ],
        }
    ]
    return messages


def generate_response(
        model: AutoModelForImageTextToText,
        processor: AutoProcessor,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float
) -> str:
    """
    Processes the input and generates a response from the Qwen3-VL model.
    """
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip()


def parse_args():
    """
    Parses command-line arguments for input and output file paths, now with defaults.
    """
    parser = argparse.ArgumentParser(
        description="Process multiple image URLs from a text file using Qwen3-VL model."
    )
    # --- Updated arguments with default values ---
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_FILE,  # Default value set here
        help=f"Path to the text file containing one image URL per line. Default: '{DEFAULT_INPUT_FILE}'"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,  # Default value set here
        help=f"Base name for the JSON file where results will be written. A timestamp will be appended. Default: '{DEFAULT_OUTPUT_FILE}'"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"The text prompt to use for image descriptions. Default: '{DEFAULT_PROMPT}'"
    )
    # --- Generation arguments (already had defaults) ---
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Maximum new tokens to generate. Default: {MAX_NEW_TOKENS}"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Temperature for text generation. Default: {TEMPERATURE}"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=TOP_P,
        help=f"Top-p value for text generation. Default: {TOP_P}"
    )
    return parser.parse_args()


def main():
    """
    Main execution function for the Qwen3-VL CPU inference script for multiple images.
    """
    args = parse_args()

    # --- 1. Load Model and Processor ---
    model, processor = load_model_and_processor(MODEL_ID, DEVICE)

    results = []

    # --- 2. Read Image URLs from Input File ---
    try:
        with open(args.input_file, 'r') as f:
            image_urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        print(
            f"Please create a file named '{DEFAULT_INPUT_FILE}' containing image URLs, one per line, or specify the file using the --input_file argument.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    if not image_urls:
        print(f"Warning: Input file '{args.input_file}' is empty. No images to process.")
        return

    print(f"\nProcessing {len(image_urls)} images from '{args.input_file}'...")

    # --- 3. Process Each Image Individually ---
    for i, image_url in enumerate(image_urls):
        print(f"\n--- Processing image {i + 1}/{len(image_urls)}: {image_url} ---")
        try:
            # Download image
            image_pil = download_image(image_url)

            # Prepare input messages
            messages = prepare_multimodal_input(image_pil, args.prompt)

            # Generate response
            print("🚀 Starting CPU generation...")
            response = generate_response(
                model,
                processor,
                messages,
                args.max_new_tokens,
                args.temperature,
                args.top_p
            )
            print(f"✅ Generated response for {image_url}: {response[:100]}...")

            # Store successful result
            results.append({
                "image_url": image_url,
                "prompt": args.prompt,
                "response": response,
                "model_id": MODEL_ID,
                "device": DEVICE,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "timestamp": datetime.now().isoformat() + "Z"
            })

        except (ConnectionError, Image.UnidentifiedImageError) as e:
            error_message = f"ERROR: {e.__class__.__name__}: {str(e).splitlines()[0]}..."
            print(f"❌ Error processing {image_url}: {error_message}")
            # Store error result
            results.append({
                "image_url": image_url,
                "prompt": args.prompt,
                "response": error_message,
                "model_id": MODEL_ID,
                "device": DEVICE,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "timestamp": datetime.now().isoformat() + "Z"
            })
        except Exception as e:
            error_message = f"UNEXPECTED ERROR: {e.__class__.__name__}: {str(e).splitlines()[0]}..."
            print(f"❌ An unexpected error occurred while processing {image_url}: {error_message}")
            # Store unexpected error result
            results.append({
                "image_url": image_url,
                "prompt": args.prompt,
                "response": error_message,
                "model_id": MODEL_ID,
                "device": DEVICE,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "timestamp": datetime.now().isoformat() + "Z"
            })

    # --- 4. Write Results to JSON Output File ---
    try:
        # Generate the timestamped filename
        timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(args.output_file)
        final_output_file = f"{base}{timestamp}{ext}"

        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ All results written to '{final_output_file}'.")
    except Exception as e:
        # Note: final_output_file may not be defined in all error paths, but it is defined here.
        # Fallback to args.output_file if necessary, but using the calculated name is better for context.
        print(f"Error writing results to output file '{args.output_file}' with stamp attempt: {e}")


if __name__ == "__main__":
    main()