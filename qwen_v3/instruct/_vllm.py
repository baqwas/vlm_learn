#!/usr/bin/env python3
"""
_vllm.py
"""

import os
import io
import base64
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime

# FastAPI is the standard for high-performance Python web APIs
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image

# vLLM for high-throughput, cost-efficient inference
# NOTE: Requires a working CUDA environment and 'pip install vllm fastapi uvicorn pydantic'
try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal.image import ImagePixelData, ImageProcessor
except ImportError:
    print("Warning: vLLM, a necessary library for high throughput, is not installed.")
    print("Please run: pip install vllm fastapi uvicorn")
    LLM = None
    SamplingParams = None
    ImagePixelData = None
    ImageProcessor = None

# --- Configuration ---
MODEL_ID: str = "Qwen/Qwen3-VL-2B-Instruct"
# 4-bit quantization for cost-efficiency and max throughput
QUANTIZATION: str = "awq"  # Use 'awq' or 'sgl' for Qwen/Qwen3 optimization
DTYPE: str = "bfloat16"  # Use bfloat16 for speed on modern GPUs


# --- Pydantic Data Schemas for API ---
class ImageInferenceRequest(BaseModel):
    """Schema for the incoming API request payload."""

    # Base64 string of the image (without the data:image/... header)
    base64_image: str = Field(..., description="Base64 encoded image string.")
    prompt: str = Field(
        ..., description="The textual prompt or question for the model."
    )
    max_new_tokens: int = Field(
        50, ge=1, le=512, description="Maximum tokens to generate."
    )
    temperature: float = Field(
        0.5, ge=0.01, le=1.0, description="Sampling temperature for generation."
    )


class InferenceResponse(BaseModel):
    """Schema for the outgoing API response."""

    model_id: str
    response: str
    time_ms: int
    tokens_per_second: float


# --- Model Initialization ---
class ModelDeployment:
    """Handles singleton loading of the LLM for the FastAPI application."""

    def __init__(self):
        if LLM is None:
            raise RuntimeError("vLLM is not available. Cannot initialize model.")

        print(f"Loading Qwen3-VL model with vLLM: {MODEL_ID}")
        # vLLM automatically handles PagedAttention and efficient batching
        self.llm = LLM(
            model=MODEL_ID,
            quantization=QUANTIZATION,
            dtype=DTYPE,
            gpu_memory_utilization=0.9,  # Maximize utilization
            enforce_eager=True,  # Ensures model is ready to serve instantly
            max_model_len=2048,  # Max context window size
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.image_processor = ImageProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        print("✅ vLLM Model Initialization Complete.")

    def get_sampling_params(self, request: ImageInferenceRequest) -> SamplingParams:
        """Creates vLLM sampling parameters from the request."""
        return SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_new_tokens,
            top_p=0.8,
            n=1,
        )

    def decode_and_prepare_input(
        self, base64_image: str, prompt: str
    ) -> Tuple[str, ImagePixelData]:
        """Decodes the Base64 image and prepares the multimodal input text."""
        try:
            # 1. Decode Base64 image to PIL Image
            if "," in base64_image:
                base64_image = base64_image.split(",", 1)[1]
            image_data = base64.b64decode(base64_image)
            image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid Base64 image data: {e}"
            )

        # 2. Process image into vLLM's internal pixel format
        pixel_data = self.image_processor.preprocess(image_pil)

        # 3. Apply the chat template to the text prompt
        # NOTE: A real Qwen-VL vLLM implementation should handle this automatically.
        # For the Qwen-VL instruct model, the standard input is [Image Token] + Prompt.
        final_prompt = f"<img></img>{prompt}"

        return final_prompt, pixel_data


# --- FastAPI Setup ---
app = FastAPI(
    title="Qwen3-VL 2B Production Inference API",
    description="High-throughput, cost-efficient deployment using vLLM.",
    version="1.0.0",
)
model_deployment = ModelDeployment()


# --- Endpoint ---
@app.post("/generate", response_model=InferenceResponse, status_code=status.HTTP_200_OK)
async def generate_response(request: ImageInferenceRequest):
    """
    Handles image and text prompts, leveraging vLLM for high-throughput generation.
    """
    if model_deployment.llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model server is not initialized. Check vLLM and CUDA status.",
        )

    try:
        # Prepare input: decode base64 and format prompt
        final_prompt, pixel_data = model_deployment.decode_and_prepare_input(
            request.base64_image, request.prompt
        )

        sampling_params = model_deployment.get_sampling_params(request)

        start_time = datetime.now()

        # Asynchronously call the vLLM engine's generate method
        # This is where the continuous batching magic happens
        results = await model_deployment.llm.agen_generate(
            prompts=[final_prompt],
            sampling_params=sampling_params,
            image_pixel_data=[pixel_data],
        )

        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Process the single result
        output = results[0].outputs[0]
        generated_text = output.text.strip()
        num_generated_tokens = len(output.token_ids)

        # Calculate tokens per second (a key throughput metric)
        tokens_per_second = (
            num_generated_tokens / (duration_ms / 1000) if duration_ms > 0 else 0
        )

        return InferenceResponse(
            model_id=MODEL_ID,
            response=generated_text,
            time_ms=duration_ms,
            tokens_per_second=tokens_per_second,
        )

    except HTTPException:
        # Re-raise explicit HTTP exceptions
        raise
    except Exception as e:
        print(f"Internal generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during inference: {e}",
        )


# --- How to Run the Server ---
if __name__ == "__main__":
    import uvicorn

    # When deploying, you would wrap this command in a startup script for your container.
    print(f"Server is starting. Access API at http://0.0.0.0:8000/docs for testing.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
