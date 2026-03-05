#!/usr/bin/env python3
#
import torch
import os
import io
import base64
import argparse # New import for command-line arguments
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from typing import List, Dict, Any, Tuple

# --- Configuration Constants (Updated for Low-Latency GPU Use) ---
MODEL_ID: str = "Qwen/Qwen3-VL-2B-Instruct"
# Prefer CUDA if available, otherwise fall back to CPU
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
# Use a reduced precision data type for speed on GPU
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16 if torch.cuda.is_available() else torch.float32

DEFAULT_MAX_NEW_TOKENS: int = 50
DEFAULT_TEMPERATURE: float = 0.5
DEFAULT_TOP_P: float = 0.8
# Default placeholder path for demonstration (must be replaced with a real image path)
DEFAULT_IMAGE_PATH: str = "placeholder_image.jpg"


class QwenVLChatbotHelper:
    """
    Real-Time Conversational Helper using Qwen3-VL 2B, optimized for low latency
    via GPU, quantization, and efficient data handling.
    """
    def __init__(self, model_id: str = MODEL_ID, device: str = DEVICE, dtype: torch.dtype = DTYPE):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.model, self.processor = self._load_model_and_processor(model_id, device, dtype)
        print(f"Model initialized on: {self.device} with dtype: {self.dtype}")

    def _load_model_and_processor(self, model_id: str, device: str, dtype: torch.dtype) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
        """Loads the Qwen3-VL model using quantization for efficiency."""
        print(f"Loading model **{model_id}** to **{device}** using {dtype}...")
        hf_token = os.environ.get("HF_TOKEN")

        # 1. Quantization Configuration (Crucial for Low Latency/Memory)
        quantization_config = None
        if device == "cuda":
            print("Applying 4-bit quantization for performance...")
            # Note: Requires 'pip install bitsandbytes accelerate'
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )

        # 2. Load Model
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map=device,
            Adtype=dtype,
            trust_remote_code=True,
            quantization_config=quantization_config,
            token=hf_token
        )

        # 3. Load Processor
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        print("✅ Model loaded successfully.")
        return model, processor

    @staticmethod
    def encode_image_to_base64(image_path: str, format: str = 'JPEG') -> str:
        """
        Loads an image file from a local path and converts it to a Base64 string.

        Args:
            image_path: The local file path to the image (e.g., './photo.jpg').
            format: The format to use when encoding (e.g., 'JPEG', 'PNG').

        Returns:
            The Base64 encoded string of the image.
        """
        print(f"Encoding image from: {image_path}")
        try:
            # 1. Load the image from the file path
            img = Image.open(image_path).convert('RGB')

            # 2. Save the image to an in-memory buffer
            buffer = io.BytesIO()
            # Ensure the format is uppercase as required by PIL
            img.save(buffer, format=format.upper())

            # 3. Encode the buffer content to Base64
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            print("✅ Image file successfully encoded to Base64.")

            return base64_string
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at path: {image_path}")
        except Exception as e:
            raise Exception(f"Failed to encode image to Base64: {e}")

    def decode_base64_to_image(self, base64_string: str) -> Image.Image:
        """Decodes Base64 string into a PIL Image object."""
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        try:
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode Base64 string to image: {e}")

    def _prepare_multimodal_input(self, image: Image.Image, prompt_text: str) -> List[Dict[str, Any]]:
        """Constructs the chat messages structure."""
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

    def _generate_response(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float = DEFAULT_TOP_P
    ) -> str:
        """Processes input and generates a response."""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_beams=1
        )

        response = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response.strip()

    def get_realtime_response(
        self,
        base64_string: str,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """Processes Base64 image and prompt for real-time chat."""
        try:
            # 1. Decode Base64 string to PIL Image
            image_pil = self.decode_base64_to_image(base64_string)

            # 2. Prepare Multimodal Input
            messages = self._prepare_multimodal_input(image_pil, prompt)

            # 3. Generate Response
            print(f"\n🚀 Starting generation with prompt: {prompt[:50]}...")
            response = self._generate_response(
                messages,
                max_new_tokens,
                temperature
            )
            return response

        except ValueError as ve:
            return f"Error: Input data issue: {ve}"
        except Exception as e:
            return f"Error: An unexpected error occurred during inference: {e}"


# --- Example of Integration (The Chatbot Loop) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Real-Time Chatbot with Local Image Encoding."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help=f"Path to the image file to be processed. Default: '{DEFAULT_IMAGE_PATH}' (Placeholder)"
    )
    args = parser.parse_args()

    # --- Step 1: Encode Local Image to Base64 ---
    try:
        # This will fail if a real image is not provided, but demonstrates the function's usage.
        encoded_image = QwenVLChatbotHelper.encode_image_to_base64(args.image_path)
    except FileNotFoundError as fnfe:
        print(f"\nFATAL ERROR: {fnfe}")
        print("Please provide a valid image file path using `--image_path <path>` or create a dummy file.")
        exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR during image encoding: {e}")
        exit(1)


    # --- Step 2: Initialize and Run Chatbot ---
    try:
        assistant = QwenVLChatbotHelper()

        print("\n--- Real-Time Chatbot Simulation ---")
        print(f"Model: {MODEL_ID} | Device: {assistant.device} | Image Source: {args.image_path}")

        while True:
            user_input = input("\nUser (Prompt): ")
            if user_input.lower() in ['quit', 'exit']:
                break

            # The encoded image is passed in every turn.
            response = assistant.get_realtime_response(
                base64_string=encoded_image,
                prompt=user_input,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                temperature=0.4
            )

            print(f"\n🤖 Qwen3-VL: {response}")

    except ImportError as ie:
        print(f"\nSetup Error: {ie}")
        print("For GPU use, please install: 'pip install bitsandbytes accelerate'")
    except Exception as e:
        print(f"Critical error during operation: {e}")
