#!/usr/bin/env python3
"""
================================================================================
PROJECT: Video Quality & Action Auditor 🚀
AUTHOR:  Matha Goram 🛠️
VERSION: 1.8.1 (Fixed Torchcodec Indexing Syntax)
UPDATED: 2026-03-04
================================================================================
OVERVIEW:
    This module provides a temporal auditing interface for analyzing high-resolution
    aviation footage using the popular FOSS Vision-Language Models. It is specifically
    architected for the BeUlta workstation to balance aggressive VRAM savings
    with high-speed CPU-bound preprocessing.

PIPELINE STAGES:
    1.  **Temporal Sampling**: Dynamic calculation of FPS to target a 32-frame
        visual context window, preventing VRAM overflow during attention cycles.
    2.  **Native Acceleration**: Intercepts raw decoded frames and offloads
        normalization math to the 'vlm_engine' (C++/OpenMP) to maximize
        multicore CPU throughput.
    3.  **Quantized Inference**: Utilizes BitsAndBytes NF4 quantization to fit
        the 3B parameter model and video tensors within a <6GB VRAM footprint.
    4.  **Disk-Buffer Loading**: Implements SSD-based weight offloading to bypass
        the 8GB System RAM bottleneck during model initialization.
    5.  **Explicit Mapping**: Bypasses AutoModel registries to load the Qwen2.5-VL
        class directly from remote code, ensuring .generate() availability.

HARDWARE REQUIREMENTS:
    - GPU: (optional) NVIDIA with >= 6GB VRAM (Supports NF4) 🙃🤌
    - CPU: Multi-core (x86_64) for OpenMP acceleration
    - Storage: High-speed SSD for offload_cache partition
OPTIMIZATIONS:
1. Disk Offloading: Uses SSD buffer to bypass 8GB System RAM limits 🛡️.
2. Static Device Mapping: Direct GPU target to avoid OOM-killer 💎.
3. Reduced Context: 32-frame sampling target for VRAM safety 📽️.
4. Torchcodec Slicing: Implements native frame extraction to suppress warnings 🚀.
================================================================================
NOTES:
Recommendations:
    Aim for ~32 frames (Slightly leaner for VRAM safety)
Tests:
    3B Model (Efficiency):
    PYTHONPATH=. python3 video_auditor.py --model Qwen/Qwen2.5-VL-3B-Instruct --fps 1.0
Action Queries:
    "Identify the exact moment of rotation (when the nose wheel leaves the ground) in seconds."
    "Is the landing gear fully retracted before the aircraft leaves the frame?"
    "Describe the livery on the tail of the aircraft to identify the airline."
================================================================================
"""

import os
import logging
import torch
import numpy as np
import sys
import argparse

try:
    import tomllib
except ImportError:
    import toml as tomllib

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration
)
# Note: process_vision_info is still used for text/image, but we override video loading
from qwen_vl_utils import process_vision_info

# Setup paths for C++ Engine
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import vlm_engine

    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False
    print("⚠️ WARNING: vlm_engine.so not found. Falling back to slow Python math. 🙄")

try:
    import torchcodec

    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False
    print("⚠️ WARNING: torchcodec not found. Defaulting to legacy loading. 🐢")


def load_video_with_torchcodec(video_path, target_fps=2.0, max_frames=32):
    """
    Native torchcodec implementation for high-speed frame slicing.
    Bypasses torchvision to suppress deprecated feature warnings.
    Updated v1.8.1: Uses standard square bracket indexing for VideoDecoder frames.
    """
    if not HAS_TORCHCODEC:
        return None

    # Instantiate decoder as the single source of truth for the file
    decoder = torchcodec.decoders.VideoDecoder(video_path)
    metadata = decoder.metadata

    duration = metadata.duration_seconds
    total_video_frames = metadata.num_frames

    total_frames_needed = min(int(duration * target_fps), max_frames)
    if total_frames_needed <= 0:
        total_frames_needed = 1

    # Generate uniform integer indices for sampling frames
    indices = torch.linspace(0, total_video_frames - 1, steps=total_frames_needed).to(torch.int64)

    # Standard loop using square bracket indexing supported by torchcodec
    frames = []
    try:
        for idx in indices:
            # VideoDecoder objects support [index] syntax to retrieve Frame objects
            frame_obj = decoder[int(idx.item())]
            # Extract the raw tensor data from the frame object
            frames.append(frame_obj.data)

        # Stack frames into [T, C, H, W] tensor required by the VLM processor
        return torch.stack(frames)
    except Exception as e:
        print(f"❌ Critical failure in torchcodec retrieval: {e}")
        return None


def load_config(config_path="config.toml"):
    """Loads settings from the [video_auditor] section of config.toml."""
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        return config.get("video", {}) or config.get("video_auditor", {})
    except Exception as e:
        print(f"⚠️ Could not load config: {e}. Using script defaults.")
        return {}


def setup_logging(log_path):
    """Initializes logging to the specified file and console."""
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] VIDEO_AUDIT: %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
    )


def run_video_audit(args, config):
    """Orchestrates the video audit process including model loading and inference."""
    model_id = args.model or config.get("vlm_video_model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
    video_dir = args.video_dir or config.get("video_dir", "/home/reza/Videos/opencv/vlm/qwen2.5-vl/videos")
    video_dir = os.path.expanduser(video_dir)
    sys_prompt = config.get("system_prompt", "You are a professional aviation auditor.")
    target_fps = args.fps or config.get("fps", 2.0)
    target_device = "cuda:0" if args.device == "gpu" else "cpu"

    offload_dir = os.path.join(project_root, "offload_cache")
    os.makedirs(offload_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info(f"🚀 INITIALIZING: {model_id} on {target_device.upper()}")

    bnb_config = None
    if "cuda" in target_device:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=target_device,
            offload_folder=offload_dir,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=torch.float32 if target_device == "cpu" else "auto"
        )
    except Exception as e:
        logger.error(f"🤦‍♂️ Model Load Failed: {e}")
        return

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if not os.path.exists(video_dir):
        logger.error(f"Directory not found: {video_dir}")
        return

    videos = sorted([f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mkv', '.mov'))])
    if not videos: return

    print("\n" + "🛫" * 30)
    print(f" 🎬 VIDEO AUDIT QUEUE: {video_dir}")
    print("🛫" * 30)
    for idx, v in enumerate(videos):
        print(f" [{idx}] 🎥 {v}")

    try:
        selection = int(input("\n🔢 Select Video Index: "))
        video_path = os.path.join(video_dir, videos[selection])
    except (ValueError, IndexError):
        return

    while True:
        user_query = input("\n🔎 ACTION QUERY >> ")
        if user_query.lower() in ['exit', 'quit']: break

        try:
            # 1. Manually Load Video using torchcodec
            video_inputs = load_video_with_torchcodec(video_path, target_fps=target_fps)

            # 2. Setup message structure
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [
                    {"type": "video", "video": video_path, "fps": target_fps},
                    {"type": "text", "text": user_query}
                ]}
            ]

            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Use process_vision_info only for image/text metadata
            image_inputs, _ = process_vision_info(messages)

            # C++ Engine Acceleration for Normalization
            if HAS_CPP_ENGINE and video_inputs is not None:
                mean_val, std_val = 0.48145466, 0.26862954
                for i in range(len(video_inputs)):
                    frame = video_inputs[i]
                    orig_shape = frame.shape
                    frame_flat = np.ascontiguousarray(frame.cpu().numpy().flatten().astype(np.float32))
                    vlm_engine.normalize_image(frame_flat, mean_val, std_val)
                    video_inputs[i] = torch.from_numpy(frame_flat.reshape(orig_shape)).to(frame.device)

            # Wrapping in list as processor expects batch format
            video_list = [video_inputs] if video_inputs is not None else None

            inputs = processor(text=[text_prompt], images=image_inputs, videos=video_list,
                               padding=True, return_tensors="pt").to(target_device)

            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

            print(f"\n📑 REPORT ({model_id}):\n{response}")

        except Exception as e:
            logger.error(f"❌ Error during inference: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BeUlta Parametric Video Auditor")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--model", type=str, help="Override model_id")
    parser.add_argument("--video-dir", type=str, help="Override source video directory")
    parser.add_argument("--fps", type=float, help="Override sampling FPS")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config file")

    args = parser.parse_args()
    config = load_config(args.config)

    setup_logging(config.get("log_file", "video_audit.log"))
    run_video_audit(args, config)
