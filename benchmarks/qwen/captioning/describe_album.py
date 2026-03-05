#!/usr/bin/env python3
# ================================================================================
# PROJECT: VLM Learn / ParkCircus Productions 🚀
# AUTHOR: Matha Goram
# LICENSE: MIT
# LOCATION: benchmarks/qwen/captioning/describe_album.py
#
# PURPOSE:
#   Batch benchmarking for Object Counting and Visual Grounding.
#   Uses Qwen3-VL to detect entities and return absolute [x1, y1, x2, y2] coordinates.
#
# UI & ERROR HANDLING SUMMARY:
#   🖥️  Interface: Console-based with Unicode status tracking.
#   ⚠️  OSError: Raised if 'album_path' is inaccessible; summarized as "Path Missing".
#   🔥  RuntimeError: Occurs if VRAM is insufficient for 7B models; summarized as "OOM".
#   🧩  ImportError: Triggered if 'qwen_vl_utils' is missing; run 'pip install'.
# ================================================================================

import os
import json
from pathlib import Path
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def benchmark_counting_and_grounding(album_path, target_object="birds"):
    """
    🎯 Task: Detect every instance of 'target_object' and count them.
    Output: JSON with coordinates and total count.
    """
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"  # Using 3B for SOHO edge performance

    print(f"🔄 Initializing {model_id}...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)[:50]}...")
        return

    results = {}
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}

    # 📂 Processing Loop
    for img_path in Path(album_path).glob("*"):
        if img_path.suffix.lower() not in img_exts:
            continue

        print(f"🔍 [SCANNING] {img_path.name} for '{target_object}'...")

        # 💡 PROMPT: Optimized for Qwen3-VL Grounding & Counting
        prompt = (
            f"Detect every {target_object} in this image. "
            f"Return the count and their locations as a JSON list of "
            f"{{'bbox_2d': [x1, y1, x2, y2], 'label': '{target_object}'}}."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # ⚙️ Inference Execution
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=512)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        results[img_path.name] = {"target": target_object, "raw_response": response}

    # 💾 Save Results
    output_file = Path("logs/benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✨ [COMPLETE] Results cached in {output_file}")


if __name__ == "__main__":
    # Ensure this folder exists or is refactored via your new bash script
    target_folder = "projects/sample_images"
    if os.path.exists(target_folder):
        benchmark_counting_and_grounding(target_folder, target_object="person")
    else:
        print(
            f"🚫 [ERROR] Directory '{target_folder}' not found. Check refactor status."
        )
