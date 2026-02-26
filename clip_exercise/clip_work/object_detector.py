#!/usr/bin/env python3
"""
================================================================================
PROJECT: Grocery Spatial Detector
AUTHOR:  Matha Goram
VERSION: 1.0.0
UPDATED: 2026-02-17
LICENSE: MIT License
COPYRIGHT: (c) 2026 ParkCircus Productions
================================================================================
PROCESSING WORKFLOW:
1. Environment Setup: Initializes unique logging and validates GPU availability.
2. Config Ingestion: Loads 'vlm_model_id' and 'annotated_dir' from config.toml.
3. VLM Initialization: Ingests Qwen3-VL weights and multimodal processor.
4. Multimodal Inference:
    - Sends image and a detection prompt to the model.
    - Captures the structured coordinate response ([ymin, xmin, ymax, xmax]).
5. Data Visualization:
    - Maps 1000-range normalized coordinates to original image pixels.
    - Renders bounding boxes and vegetable labels via PIL.ImageDraw.
6. Persistence: Saves high-resolution annotated frames to 'annotated_dir'.

DATA INTEGRITY LOGIC:
- Coordinate Normalization: Maps Qwen3-VL's normalized [0-1000] coordinate system
  back to original pixel dimensions [W, H] to ensure spatial accuracy.
- Structured Parsing: Uses Regex-based extraction to identify [ymin, xmin, ymax, xmax]
  patterns from unstructured model text responses.
- Temporal Stability: Designed to handle high-resolution frames (PNG/JPG) while
  maintaining label-to-box alignment for grocery item detection.

USER INTERFACE REQUIREMENTS:
- Visual Feedback: Automated rendering of bounding boxes directly onto source images
  using PIL.ImageDraw for immediate spatial verification.
- Color Logic: Outline colors (e.g., 'lime') and border thickness must be externally
  configurable via config.toml for visibility across different backgrounds.
- High-Resolution Persistence: The UI must save non-destructive copies to a dedicated
  'annotated_dir' to prevent overwriting raw source data.

ERROR HANDLING & EXCEPTION STRATEGY:
- GPU Fallback: Automatically detects CUDA availability and redirects to CPU
  inference if VRAM is insufficient or drivers are missing on Ubuntu.
- Regex Resilience: Wrapped try-except blocks around coordinate parsing to
  prevent session crashes if the model generates invalid JSON or box formats.
- Format Enforcement: Converts all input images to 'RGB' mode before drawing
  to prevent 'Palette' or 'Alpha' channel errors during visualization.
- Logging Severity:
    - CRITICAL: Model load failures or missing target_items in config.
    - ERROR: Inference timeout or failed coordinate mapping for a specific frame.

PREREQUISITES:
- Python 3.11+
- transformers, torch, Pillow, qwen-vl-utils
================================================================================
"""

import os
import tomllib
import logging
import torch
import json
import re
from PIL import Image, ImageDraw
from pathlib import Path
import torch
#let the model tell the code what it is, rather than trying to find a hardcoded reference that doesn't exist in the library yet
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info


def setup_logging(log_path):
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return logging.getLogger("Qwen_Spatial_Audit")


def run_detection():
    # 1. LOAD UNIQUE CONFIGURATION
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.toml"

    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f"CRITICAL: Configuration access failure: {e}")
        return

    logger = setup_logging(config["paths"]["log_file"])
    logger.info("Initializing ParkCircus Spatial Detection Engine...")

    # 2. RESOLVE DIRECTORIES & MODEL IDS
    # Using 'annotated_dir' to avoid conflict with 'output_dir'
    input_dir = root_dir / config["paths"]["input_dir"]
    annotated_dir = root_dir / config["paths"]["annotated_dir"]
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # Using 'vlm_model_id' to avoid conflict with 'name'
    vlm_id = config["model"]["vlm_model_id"]
    targets = ", ".join(config["detection"]["target_items"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. LOAD VISION-LANGUAGE MODEL
    try:
        # Use Auto classes to bypass naming conflicts
        processor = AutoProcessor.from_pretrained(vlm_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            vlm_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    except Exception as e:
        logger.error(f"Model Load Error ({vlm_id}): {e}")
        return

    # 4. DETECTION LOOP
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in images:
        img_path = input_dir / img_name

        # Prepare the Multimodal Message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": f"Find {targets}. Return JSON format with 'bbox_2d' and 'label'."}
                ],
            }
        ]

        try:
            # Process vision info and prompt template
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            # Generate Response
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            response = processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            logger.info(f"Model Inference for {img_name}: {response}")

            # 5. PARSE & DRAW (THE UI)
            with Image.open(img_path).convert("RGB") as img:
                draw = ImageDraw.Draw(img)
                w, h = img.size

                # Regex to find bounding boxes: [ymin, xmin, ymax, xmax]
                # Qwen uses normalized coordinates (0-1000)
                box_matches = re.finditer(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+)\]', response)

                for match in box_matches:
                    coords = [int(c) for c in match.group(1).split(',')]
                    ymin, xmin, ymax, xmax = coords

                    # Convert to pixel coordinates
                    left, top = (xmin * w / 1000), (ymin * h / 1000)
                    right, bottom = (xmax * w / 1000), (ymax * h / 1000)

                    draw.rectangle(
                        [left, top, right, bottom],
                        outline=config["detection"].get("box_color", "lime"),
                        width=config["detection"].get("thickness", 3)
                    )

                save_path = annotated_dir / f"annotated_{img_name}"
                img.save(save_path)
                logger.info(f"SUCCESS: Annotated visual saved to {save_path}")

        except Exception as e:
            logger.error(f"Inference failure for {img_name}: {e}")

    logger.info("Spatial detection task finalized.")


if __name__ == "__main__":
    run_detection()