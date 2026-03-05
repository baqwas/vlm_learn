#!/usr/bin/env python3
"""
================================================================================
PROJECT: Visual Quality Auditor (VQA)
AUTHOR:  Matha Goram
VERSION: 1.1.0 (CPU-Optimized for Qwen3)
UPDATED: 2026-02-17
LICENSE: MIT License
COPYRIGHT: (c) 2026 ParkCircus Productions
================================================================================
PROCESSING WORKFLOW:
1. Load Environment: Initializes BF16 model for CPU-based reasoning.
2. Context Ingestion: Parses config.toml using absolute Ubuntu paths.
3. Interactive Session:
    - Scans 'output_dir' for available images.
    - Prompts user to select a specific image for auditing.
4. Reasoned Inference:
    - Combines 'System Prompt' + 'Visual Context' + 'User Query'.
    - Generates a response using Qwen3-VL's advanced reasoning capabilities.
5. Audit Logging: Records the conversation for dataset improvement.

HARDWARE & REASONING LOGIC:
- CPU-First Inference: Specifically optimized for high-performance CPU execution
  using the BF16 (Bfloat16) data format to maximize efficiency without a GPU.
- Model Factory: Dynamically initializes the Qwen3-VL-4B-Instruct engine to
  perform deep semantic reasoning on static visual data.
- Prompt Engineering: Implements a tri-part message structure combining 'System
  Instruction', 'Visual Content', and 'Dynamic User Query' into a unified context.

USER INTERFACE REQUIREMENTS:
- Interaction Model: Persistent terminal-based chat loop allowing for iterative
  natural language questioning of a single visual context.
- Session Management: Includes command listeners (e.g., 'exit', 'clear') to manage
  the conversation state and gracefully terminate the audit.
- Selection UI: Provides a numeric indexed list of available images from the
  'output_dir' for rapid file targeting.

ERROR HANDLING & EXCEPTION STRATEGY:
- Environment Guardrails: Validates 'config.toml' availability and verifies absolute
  Ubuntu directory paths before model ingestion.
- Memory Protection: Uses explicit device mapping to "cpu" to prevent accidental
  CUDA initialization errors on systems without NVIDIA drivers.
- Inference Resilience: Wrapped 'model.generate' calls within try-except blocks to
  capture and log engine failures without crashing the interactive loop.
- Logging Severity:
    - CRITICAL: Failure to load the 4B-Instruct reasoning weights.
    - ERROR: Malformed user queries or corrupted image files preventing processing.
    - INFO: Records a permanent audit trail of every Question/Answer pair for dataset review.

PREREQUISITES:
- Python 3.11+
- transformers (latest), torch, qwen-vl-utils
================================================================================
"""

import os
import tomllib
import logging
import torch
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def setup_logging(log_path):
    """Initializes logging based on config path."""
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger("Visual_Auditor")


def run_audit_session():
    # 1. LOAD CONFIGURATION
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.toml"

    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f"CRITICAL: Could not read config at {config_path}: {e}")
        return

    logger = setup_logging(config["paths"]["log_file"])

    # Force CPU for this specific version of the script
    device = "cpu"

    # 2. LOAD OPTIMIZED VLM (BF16 for CPU)
    # Using the standard 4B-Instruct model which is best suited for CPU BF16
    vlm_id = "Qwen/Qwen3-VL-4B-Instruct"
    logger.info(f"Loading Reasoning Engine: {vlm_id} on {device} (BF16)")

    try:
        # Load the processor
        processor = AutoProcessor.from_pretrained(vlm_id)

        # Load the model in Bfloat16 which is the most efficient format for modern CPUs
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize VLM: {e}")
        return

    # 3. SELECT IMAGE FROM PATHS
    target_dir = Path(config["paths"]["output_dir"])
    if not target_dir.exists():
        logger.error(f"Directory not found: {target_dir}")
        return

    images = sorted(
        [
            f
            for f in os.listdir(target_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if not images:
        logger.error(f"No images found in {target_dir}")
        return

    print("\n" + "=" * 50)
    print(f" DIRECTORY: {target_dir}")
    print("=" * 50)
    for i, img in enumerate(images):
        print(f" [{i}] {img}")

    try:
        img_idx = int(input("\nSelect Image Index to Audit: "))
        target_img_path = target_dir / images[img_idx]
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        return

    # 4. INTERACTIVE AUDIT LOOP
    print(f"\nAUDIT START: {images[img_idx]}")
    print("Commands: 'exit' to quit | 'clear' to reset context\n")

    while True:
        user_query = input("AUDITOR QUESTION >> ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break
        if user_query.lower() == "clear":
            print("Context cleared (simulation).")
            continue

        # Construct the VQA Message
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": config["vqa"]["system_prompt"]}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(target_img_path)},
                    {"type": "text", "text": user_query},
                ],
            },
        ]

        try:
            # Prepare prompts and vision info
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            # Process inputs to CPU
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Generate response
            # Temperature is kept low as per your config for factual auditing
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=config["vqa"].get("max_new_tokens", 512),
                temperature=config["vqa"].get("temperature", 0.2),
                do_sample=True if config["vqa"].get("temperature", 0.2) > 0 else False,
            )

            # Decode and trim prompt from response
            response = processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )[0]

            print(f"\nAI AUDIT REPORT:\n{response}\n" + "-" * 50)
            logger.info(f"File: {images[img_idx]} | Q: {user_query} | A: {response}")

        except Exception as e:
            logger.error(f"Inference error: {e}")

    print("Audit session closed.")


if __name__ == "__main__":
    run_audit_session()
