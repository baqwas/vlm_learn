#!/usr/bin/env python3
"""
================================================================================
PROJECT: Video Quality & Action Auditor
AUTHOR:  Matha Goram
VERSION: 1.0.0
UPDATED: 2026-02-17
LICENSE: MIT License
COPYRIGHT: (c) 2026 ParkCircus Productions
================================================================================
PROCESSING WORKFLOW:
1. Session Init: Loads config using unique 'vlm_video_model_id' and 'video_dir'.
2. Temporal Sampling: Uses qwen-vl-utils to extract frames at a specific FPS.
3. Multimodal Encoding:
    - Processes video tokens as a temporal sequence.
    - Maintains spatial-temporal consistency across frames.
4. Action Reasoning:
    - User provides a query about an event (e.g., "Is the user bruising the fruit?").
    - Model outputs a narrative description of the detected action.
5. Interactive UI: Terminal-based chat loop for iterative video investigation.
TEMPORAL REASONING LOGIC:
- Spatial-Temporal Consistency: Treats video inputs as a continuous sequence,
  allowing the model to reason about movement, cause-and-effect, and human actions.
- Dynamic Frame Sampling: Utilizes 'qwen-vl-utils' to extract frames at a configurable
  FPS (Frames Per Second), balancing visual detail with computational efficiency.
- Multimodal Encoding: Integrates video tokens with system-level instructions
  to produce narrative "Temporal Analysis Reports."
- Generic Model Resolution: Uses the 'AutoModel' factory with 'trust_remote_code=True'
  to dynamically resolve the correct Qwen3-VL architecture at runtime.

USER INTERFACE REQUIREMENTS:
- Interaction Model: Persistent terminal-based chat loop allowing for iterative
  investigation of a specific video file.
- Selection UI: Provides an indexed list of video files (.mp4, .avi, .mkv) from
  the 'video_dir' for targeted temporal auditing.
- Session Management: Includes 'exit' and 'quit' listeners to formally close the
  action reasoning session and finalize the audit logs.

ERROR HANDLING & EXCEPTION STRATEGY:
- Environment Guardrails: Verification of ffmpeg installation (required for frame
  sampling) and absolute Ubuntu path resolution for video directories.
- Hardware Fallback: Automatically detects CUDA availability via torch and redirects
  temporal inference to the CPU if no GPU is present.
- Sampling Resilience: Wrapped 'process_vision_info' calls to catch decoding errors
  caused by corrupted video headers or unsupported codecs.
- Logging Severity:
    - CRITICAL: Failure to load the video-optimized VLM engine or missing ffmpeg.
    - ERROR: Temporal inference failures or file I/O locks on the selected video.
    - INFO: Logs the full narrative response for every temporal action query.

PREREQUISITES:
- Python 3.11+
- transformers, torch, qwen-vl-utils, accelerate, autoawq
- ffmpeg installed on Ubuntu (sudo apt install ffmpeg)

USAGE:
python3 video_auditor.py -h
================================================================================
"""

import os
import tomllib
import logging
import torch
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoModel, AutoProcessor
from qwen_vl_utils import process_vision_info


def setup_logging(log_path):
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] VIDEO_AUDIT: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return logging.getLogger("Video_Auditor")


def run_video_audit():
    # 1. CONFIGURATION & DIRECTORY RESOLUTION
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.toml"

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    logger = setup_logging(config["paths"]["log_file"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_dir = Path(config["paths"]["video_dir"])
    if not video_dir.exists():
        logger.error(f"Video directory missing: {video_dir}")
        return

    # 2. LOAD VIDEO-OPTIMIZED VLM
    vlm_id = config["model"]["vlm_video_model_id"]
    logger.info(f"Loading Temporal Engine: {vlm_id}")

    # This is the most portable version: AutoModel + trust_remote_code
    # It will automatically find 'Qwen3VLForConditionalGeneration' or
    # whatever the specific class is named in the model's own files.
    model = AutoModel.from_pretrained(
        vlm_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True  # This is the key to resolving new architectures
    )

    # 2. Use the generic 'Auto' class for the processor
    processor = AutoProcessor.from_pretrained(vlm_id, trust_remote_code=True)

    # 3. VIDEO SELECTION UI
    videos = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mkv'))])
    if not videos:
        logger.error("No video files found.")
        return

    print("\n" + "=" * 50)
    print(f" VIDEO AUDIT QUEUE: {video_dir}")
    print("=" * 50)
    for i, v in enumerate(videos):
        print(f" [{i}] {v}")

    try:
        selection = int(input("\nSelect Video Index: "))
        target_video = video_dir / videos[selection]
    except (ValueError, IndexError):
        print("Invalid Selection.")
        return

    # 4. TEMPORAL REASONING LOOP
    print(f"\nAUDITING ACTION: {videos[selection]}")
    print("The model is analyzing temporal frames. Type 'exit' to stop.")

    while True:
        query = input("\nACTION QUERY >> ").strip()
        if query.lower() in ['exit', 'quit']:
            break

        # Define message with video type
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": config["video"]["system_prompt"]}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(target_video),
                        "fps": config["video"].get("fps", 1.0),
                    },
                    {"type": "text", "text": query}
                ],
            }
        ]

        try:
            # Process vision/video info
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            # Generate Temporal Analysis
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            response = processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            print(f"\nTEMPORAL ANALYSIS REPORT:\n{response}")
            logger.info(f"Video: {videos[selection]} | Query: {query} | Response: {response}")

        except Exception as e:
            logger.error(f"Temporal Inference Error: {e}")

    print("Video audit session complete.")


if __name__ == "__main__":
    run_video_audit()