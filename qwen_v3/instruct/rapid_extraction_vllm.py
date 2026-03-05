import torch
import os
import io
import base64
import requests
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from typing import List, Dict, Any, Tuple
from datetime import datetime

# --- Configuration Constants ---
MODEL_ID: str = "Qwen/Qwen3-VL-2B-Instruct"
# Default to CPU for wide compatibility, but retains the speed optimization logic.
DEVICE: str = "cpu"
DTYPE = torch.float32

# Global variable definition
DEFAULT_MAX_NEW_TOKENS: int = 10  # CRITICAL for speed (Rapid Extraction)
DEFAULT_TEMPERATURE: float = 0.1  # Low temperature for reliable, deterministic classification
DEFAULT_TOP_P: float = 0.8
SAMPLE_IMAGE_URL: str = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


# This image contains a person, a horse, and two dogs in a field.

class QwenVLRapidHelper:
    """
    Helper optimized for rapid, low-latency extraction and classification tasks.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = DEVICE, dtype: torch.dtype = DTYPE):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.model, self.processor = self._load_model_and_processor(model_id, device, dtype)
        print(f"Model initialized on: {self.device} with dtype: {self.dtype}")

    def _load_model_and_processor(self, model_id: str, device: str, dtype: torch.dtype) -> Tuple[
        AutoModelForImageTextToText, AutoProcessor]:
        """Loads the Qwen3-VL model."""
        print(f"Loading model **{model_id}**...")
        hf_token = os.environ.get("HF_TOKEN")

        # Load with minimal memory footprint (DTYPE is float32 for CPU default)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            trust_remote_code=True,
            token=hf_token
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        print("✅ Model loaded successfully.")
        return model, processor

    @staticmethod
    def download_and_encode_image(image_url: str) -> str:
        """Downloads an image and converts it to a Base64 string."""
        print(f"Downloading sample image from: {image_url}")
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content)).convert('RGB')

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        print("✅ Sample image downloaded and encoded to Base64.")
        return base64_string

    def decode_base64_to_image(self, base64_string: str) -> Image.Image:
        """Decodes Base64 string into a PIL Image object."""
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image

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
    ) -> Tuple[str, float]:
        """Processes input and generates a response."""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        start_time = datetime.now()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=DEFAULT_TOP_P,
            num_beams=1
        )
        duration = (datetime.now() - start_time).total_seconds()

        response = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        return response.strip(), duration

    def run_rapid_task(self, base64_string: str, prompt: str) -> Tuple[str, float]:
        """Runs a single task with speed-optimized parameters."""
        # Use the current global values for generation
        global DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE

        try:
            image_pil = self.decode_base64_to_image(base64_string)
            messages = self._prepare_multimodal_input(image_pil, prompt)

            response, duration = self._generate_response(
                messages,
                DEFAULT_MAX_NEW_TOKENS,
                DEFAULT_TEMPERATURE
            )
            return response, duration

        except Exception as e:
            return f"Error running prompt: {e}", 0.0


def main():
    # Declare DEFAULT_MAX_NEW_TOKENS as global here so it can be modified in this function.
    global DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P

    try:
        # --- 1. Prepare Environment and Image ---
        # Download and encode the sample image
        encoded_image = QwenVLRapidHelper.download_and_encode_image(SAMPLE_IMAGE_URL)

        # Initialize the model
        print("\n--- Initializing Qwen3-VL 2B Instruct for Rapid Tasks ---")
        rapid_assistant = QwenVLRapidHelper()
        print(f"Model Ready: {rapid_assistant.model_id} on {rapid_assistant.device}")

        # --- 2. Define Rapid Extraction/Classification Tasks ---
        rapid_tasks = [
            {
                "type": "Object Identification (Extraction)",
                "prompt": "List the three main living things visible in the image, separated by commas. Start with the person.",
                "max_tokens": 15,
                "notes": "Fast token generation to extract key entities."
            },
            {
                "type": "Image-to-Text Transcription",
                "prompt": "Identify the type of setting: a) beach, b) desert, c) park, or d) open field.",
                "max_tokens": 5,
                "notes": "Forces a classification (d) by providing limited choices."
            },
            {
                "type": "Attribute Extraction (Color)",
                "prompt": "The color of the horse is:",
                "max_tokens": 3,
                "notes": "Minimal token output for a single attribute."
            }
        ]

        # --- 3. Execute Demonstrations ---
        print("\n" + "=" * 80)
        print(f"RAPID EXTRACTION AND CLASSIFICATION DEMONSTRATION (Qwen3-VL 2B Instruct)")
        print(f"Optimization: Initial Max tokens: {DEFAULT_MAX_NEW_TOKENS}, Temperature: {DEFAULT_TEMPERATURE}.")
        print("=" * 80)

        total_duration = 0
        for task in rapid_tasks:
            # --- FIX APPLIED HERE: Global declaration for assignment ---
            # Temporarily store the original value
            old_max_tokens = DEFAULT_MAX_NEW_TOKENS

            # Set the new global value for this task
            DEFAULT_MAX_NEW_TOKENS = task.get("max_tokens", old_max_tokens)

            print("\n" + "-" * 40)
            print(f"TASK: {task['type']} (Max Tokens set to {DEFAULT_MAX_NEW_TOKENS})")
            print(f"Prompt: {task['prompt']}")
            print(f"Goal: {task['notes']}")

            response, duration = rapid_assistant.run_rapid_task(encoded_image, task["prompt"])

            total_duration += duration

            print("\n🤖 MODEL RESPONSE (Extracted/Classified):")
            print(f"{response}")
            print(f"\n[Inference Time]: {duration:.2f} seconds")

            # Reset max tokens back to the default/previous value
            DEFAULT_MAX_NEW_TOKENS = old_max_tokens

        print("\n" + "=" * 80)
        print(
            f"Demonstration Complete. Total generation time for {len(rapid_tasks)} tasks: {total_duration:.2f} seconds.")
        print("This speed is achieved by dynamically forcing extremely short, focused outputs.")
        print("=" * 80)

    except requests.exceptions.HTTPError as e:
        print(f"\nERROR: Failed to download sample image. Status: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during main execution: {e}")
        print("Please ensure all dependencies are installed and system resources are sufficient.")


if __name__ == "__main__":
    main()
