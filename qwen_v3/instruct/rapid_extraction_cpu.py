import os
import io
import base64
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# --- Dependencies for CPU-Only Inference ---
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# FastAPI and Pydantic for the web API structure
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# --- Configuration ---
# Set the device explicitly to CPU and use the standard float32 dtype
MODEL_ID: str = "Qwen/Qwen3-VL-2B-Instruct"
DEVICE: str = "cpu"
DTYPE = torch.float32


# --- Pydantic Data Schemas for API ---
class ImageInferenceRequest(BaseModel):
    """Schema for the incoming API request payload."""
    base64_image: str = Field(..., description="Base64 encoded image string.")
    prompt: str = Field(..., description="The textual prompt or question for the model.")
    max_new_tokens: int = Field(50, ge=1, le=512, description="Maximum tokens to generate.")
    temperature: float = Field(0.5, ge=0.01, le=1.0, description="Sampling temperature for generation.")


class InferenceResponse(BaseModel):
    """Schema for the outgoing API response."""
    model_id: str
    response: str
    time_ms: int
    tokens_per_second: float


# --- Model Initialization ---
class ModelDeployment:
    """
    Handles singleton loading of the LLM using standard Hugging Face for CPU.
    NOTE: This requires significant system RAM (16GB+).
    """

    def __init__(self):
        print(f"Loading Qwen3-VL model on CPU: {MODEL_ID}")

        # 1. Load Model
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,  # Load onto CPU RAM
            dtype=DTYPE,  # Use standard float32 for CPU
            trust_remote_code=True
        )
        # 2. Load Processor
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("✅ Hugging Face CPU Model Initialization Complete.")

    def decode_and_prepare_input(self, base64_image: str, prompt: str) -> Tuple[torch.Tensor, int]:
        """Decodes the Base64 image and prepares the multimodal input tensors."""
        try:
            # 1. Decode Base64 image to PIL Image
            if ',' in base64_image:
                base64_image = base64_image.split(',', 1)[1]
            image_data = base64.b64decode(base64_image)
            image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Base64 image data: {e}")

        # 2. Construct the chat message list (as required by Qwen)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        # 3. Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # 4. Move inputs to the CPU
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Return inputs and the length of the input tokens for decoding
        return inputs, inputs["input_ids"].shape[-1]

    def generate(self, inputs: Dict[str, torch.Tensor], input_length: int, params: ImageInferenceRequest) -> str:
        """Synchronously calls the model's generate function."""

        # This function blocks the thread while the CPU calculates the response.
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=params.max_new_tokens,
            do_sample=True,
            temperature=params.temperature,
            top_p=0.8,
            num_beams=1,
        )

        # Decode the generated IDs, skipping the input prompt tokens
        generated_ids_trimmed = outputs[0][input_length:].tolist()
        response = self.processor.decode(generated_ids_trimmed, skip_special_tokens=True)

        return response.strip(), len(generated_ids_trimmed)


# --- FastAPI Setup ---
app = FastAPI(
    title="Qwen3-VL 2B CPU Inference API (Slow)",
    description="Low-throughput deployment using Hugging Face on CPU.",
    version="1.0.0"
)
# Initialize the model globally (this loads the model into RAM)
model_deployment = ModelDeployment()


# --- Endpoint ---
@app.post("/generate", response_model=InferenceResponse, status_code=status.HTTP_200_OK)
async def generate_response(request: ImageInferenceRequest):
    """
    Handles image and text prompts on CPU. Uses asyncio.to_thread to prevent
    the entire server from blocking during the long inference calculation.
    """

    try:
        # Prepare input: decode base64 and format prompt
        inputs, input_length = model_deployment.decode_and_prepare_input(
            request.base64_image,
            request.prompt
        )

        start_time = datetime.now()

        # --- CRITICAL FOR CPU SERVER RESPONSIVENESS ---
        # Run the blocking model.generate() call in a separate thread.
        # This keeps the main FastAPI event loop free to handle other requests.
        generated_text, num_generated_tokens = await asyncio.to_thread(
            model_deployment.generate,
            inputs,
            input_length,
            request
        )

        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Calculate throughput metric
        tokens_per_second = num_generated_tokens / (duration_ms / 1000) if duration_ms > 0 else 0

        return InferenceResponse(
            model_id=MODEL_ID,
            response=generated_text,
            time_ms=duration_ms,
            tokens_per_second=tokens_per_second
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Internal generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during inference: {e}"
        )


# --- How to Run the Server ---
if __name__ == "__main__":
    import uvicorn

    print(f"⚠️ WARNING: Running Qwen3-VL 2B on CPU. Expect long latencies (seconds to minutes) per request.")
    print(f"Server is starting. Access API at http://0.0.0.0:8000/docs for testing.")

    # Use multiple worker processes (e.g., 4) to handle multiple requests concurrently,
    # as each request will block a single worker thread.
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)