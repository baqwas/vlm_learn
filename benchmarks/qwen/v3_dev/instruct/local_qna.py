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
================================================================================
PROJECT: Qwen-VL Multimodal General Question Answering (GQA) Script
AUTHOR: ParkCircus Productions
DATE: 2025-11-30
VERSION: 1.1.0
DESCRIPTION: A script demonstrating local (CPU-only) General Question Answering
             using the Qwen3-VL-2B-Instruct model from HuggingFace transformers.
             It reads a sample image from a local file path for inference.
================================================================================

MIT License

Copyright (c) 2024 ParkCircus Productions

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
import os
import io
import base64
import requests  # Required for downloading the sample image
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from typing import List, Dict, Any, Tuple
from datetime import datetime

# --- Configuration Constants ---
MODEL_ID: str = "Qwen/Qwen3-VL-2B-Instruct"
# Default to CPU for maximum compatibility as requested in earlier steps
DEVICE: str = "cpu"
DTYPE = torch.float32  # Use float32 for reliable CPU inference

DEFAULT_MAX_NEW_TOKENS: int = 100
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.8
# Original Internet URL (kept for reference)
SAMPLE_IMAGE_URL: str = (
    "https://aeropuertosglobales.com/wp-content/uploads/Aeropuerto-Internacional-Chicago-OHare-ORD-900x600.jpg"
)
# New constant: Local image path to be used instead of the URL
DEFAULT_LOCAL_IMAGE_PATH: str = "/qwen_v3/images/shrimp.jpg"
# DEFAULT_LOCAL_IMAGE_PATH: str = "/home/reza/PycharmProjects/vlm_learn/qwen_v3/images/submersible.jpg"


class QwenVLDemoHelper:
    """
    A minimal helper class for running Qwen3-VL inference demonstrations.
    """

    def __init__(
        self, model_id: str = MODEL_ID, device: str = DEVICE, dtype: torch.dtype = DTYPE
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.model, self.processor = self._load_model_and_processor(
            model_id, device, dtype
        )
        print(f"Model initialized on: {self.device} with dtype: {self.dtype}")

    def _load_model_and_processor(
        self, model_id: str, device: str, dtype: torch.dtype
    ) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
        """Loads the Qwen3-VL model and its associated processor."""
        print(f"Loading model **{model_id}**...")
        hf_token = os.environ.get("HF_TOKEN")

        # Explicitly load model to the specified device (CPU)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            trust_remote_code=True,
            token=hf_token,
        )
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token
        )
        print("✅ Model loaded successfully.")
        return model, processor

    @staticmethod
    def download_and_encode_image(image_url: str) -> str:
        """
        Original method: Downloads an image from a URL and converts it to a Base64 string.
        (Kept for compatibility, but not used in the modified main function)
        """
        print(f"Downloading sample image from: {image_url}")
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        # 1. Save the image to an in-memory buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")  # Use a standard format for encoding

        # 2. Encode the buffer content to Base64
        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print("✅ Sample image downloaded and encoded to Base64.")
        return base64_string

    @staticmethod
    def read_and_encode_local_image(image_path: str) -> str:
        """
        NEW METHOD: Reads a local image file and converts it to a Base64 string.
        """
        print(f"Reading sample image from local path: {image_path}")
        try:
            img = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            # Re-raise with a more informative message if the file is not found
            raise FileNotFoundError(f"Local image file not found: {image_path}")
        except Exception as e:
            raise Exception(f"Error opening local image file: {e}")

        # 1. Save the image to an in-memory buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")  # Use a standard format for encoding

        # 2. Encode the buffer content to Base64
        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print("✅ Sample image read and encoded to Base64.")
        return base64_string

    def decode_base64_to_image(self, base64_string: str) -> Image.Image:
        """Decodes Base64 string into a PIL Image object."""
        # Simple check for data URI prefix (if present)
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image

    def _prepare_multimodal_input(
        self, image: Image.Image, prompt_text: str
    ) -> List[Dict[str, Any]]:
        """Constructs the chat messages structure."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        return messages

    def _generate_response(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """Processes input and generates a response."""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Ensure inputs are on the same device as the model (CPU)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        start_time = datetime.now()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
        )
        duration = (datetime.now() - start_time).total_seconds()

        response = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

        return response.strip(), duration

    def run_gqa_prompt(self, base64_string: str, prompt: str) -> Tuple[str, float]:
        """Runs a single GQA prompt and returns the response and duration."""
        try:
            image_pil = self.decode_base64_to_image(base64_string)
            messages = self._prepare_multimodal_input(image_pil, prompt)

            print(f"\n[Running Prompt]: {prompt}")
            response, duration = self._generate_response(
                messages, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE
            )
            return response, duration

        except Exception as e:
            return f"Error running prompt: {e}", 0.0


def main():
    try:
        # --- 1. Prepare Environment and Image ---
        # Read the image from a local path and get the Base64 string
        print("\n--- Preparing Image from Local File ---")
        if not os.path.exists(DEFAULT_LOCAL_IMAGE_PATH):
            # Re-raise a FileNotFoundError if the file is not where expected
            raise FileNotFoundError(
                f"Local image file not found at: {DEFAULT_LOCAL_IMAGE_PATH}. Please update the path."
            )

        # Use the new method to read from the local file
        encoded_image = QwenVLDemoHelper.read_and_encode_local_image(
            DEFAULT_LOCAL_IMAGE_PATH
        )

        # Initialize the model (This step takes the most time and memory)
        print("\n--- Initializing Qwen3-VL 2B Instruct ---")
        demo_assistant = QwenVLDemoHelper()
        print(f"Model Ready: {demo_assistant.model_id} on {demo_assistant.device}")

        # --- 2. Define Simple GQA Tasks ---
        gqa_tasks = [
            {
                "type": "Image Description",
                "prompt": "Describe this picture in a single, short sentence.",
                "notes": "Tests basic object recognition and scene summarization.",
            },
            {
                "type": "Visual Question Answering (Attribute)",
                "prompt": "What are the colors of the dominant items in the image?",
                "notes": "Tests attribute identification (color of a specific object).",
            },
            {
                "type": "Visual Question Answering (Object Count)",
                "prompt": "How many types of items are visible in the image?",
                "notes": "Tests simple object counting.",
            },
        ]

        # --- 3. Execute Demonstrations ---
        print("\n" + "=" * 80)
        print(f"GENERAL QUESTION AND ANSWER DEMONSTRATION (Qwen3-VL 2B Instruct)")
        print(f"Sample Image Path: {DEFAULT_LOCAL_IMAGE_PATH}")
        print("=" * 80)

        total_duration = 0
        for task in gqa_tasks:
            print("\n" + "-" * 40)
            print(f"TASK: {task['type']}")
            print(f"Notes: {task['notes']}")

            response, duration = demo_assistant.run_gqa_prompt(
                encoded_image, task["prompt"]
            )

            total_duration += duration

            print("\n🤖 MODEL RESPONSE:")
            print(f"{response}")
            print(f"\n[Inference Time]: {duration:.2f} seconds")

        print("\n" + "=" * 80)
        print(
            f"Demonstration Complete. Total generation time: {total_duration:.2f} seconds."
        )
        print("=" * 80)

    except requests.exceptions.HTTPError as e:
        # This error is less likely now, but kept for robust failure handling
        print(
            f"\nERROR: Failed to download sample image. Please check the URL or your network connection. Status: {e}"
        )
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Please ensure the image file exists at the specified path.")
    except Exception as e:
        print(f"\nAn error occurred during the demonstration: {e}")
        print(
            "Please ensure you have all dependencies (torch, transformers, PIL, requests) installed and sufficient system RAM (16+ GB is recommended for the 2B model on CPU)."
        )


if __name__ == "__main__":
    main()
