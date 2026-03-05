#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
VERSION: 1.0.3
UPDATED: 2026-03-05 08:50:00
COPYRIGHT: (c) 2026 ParkCircus Productions; All Rights Reserved.
AUTHOR: Matha Goram
LICENSE: MIT
PURPOSE: Configuration-driven Hello World for Qwen3-VL using config.toml.
================================================================================
"""

import torch
import argparse
import configparser
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_config(config_path="config.toml"):
    """Reads the centralized configuration file."""
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        # Fallback if script is run from a subdirectory
        config_path = os.path.join(os.path.dirname(__file__), "../../../", config_path)

    config.read(config_path)
    return config["hello_world"]


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Hello World")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    args = parser.parse_args()

    # Load assets from config.toml
    cfg = load_config(args.config)
    image_path = cfg.get("test_image")
    prompt_text = cfg.get("default_prompt")

    # 1. Load Model & Processor
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="bfloat16", device_map="cpu", trust_remote_code=True
    ).eval()

    # 2. Prepare Input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # 3. Inference Pipeline
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cpu")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    print(f"\n🚀 Qwen3-VL Response (Config-Driven):\n{response[0]}")


if __name__ == "__main__":
    main()
