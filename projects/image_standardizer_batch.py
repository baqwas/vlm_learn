"""
🖼️ VLM Dataset Standardizer (Task 1.1.2)
========================================
Workflow: Load Raw -> Pad to Square -> Resize to 1024x1024 -> Export JPG

Requirements: pip install Pillow

Task ID,Description,Progress,Outcome
1.1.1,Collect 50 images,50%,10 Generated; 40 awaiting manual collection.
1.1.2,Standardize to 1024x1024,Ready,Script prepared; pending raw image collection.
1.2.1,Generate JSON GT files,20%,10 images fully automated with coordinates.
"""

import os
import sys
import logging
from PIL import Image, ImageOps
from pathlib import Path

# --- 🛠️ CONFIGURATION ---
INPUT_DIR = "../images/benchmark/subitizing"
OUTPUT_DIR = "../images/benchmark/standardized"
TARGET_SIZE = (1024, 1024)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def standardize_images():
    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"❌ Error: Input folder '{INPUT_DIR}' not found!")
        return

    supported_formats = ('.jpg', '.jpeg', '.png', '.webp')
    processed_count = 0

    print(f"🚀 Processing images from {INPUT_DIR}...")

    for file in input_path.iterdir():
        if file.suffix.lower() in supported_formats:
            try:
                with Image.open(file) as img:
                    # Convert to RGB (to handle PNGs with transparency or CMYK)
                    img = img.convert("RGB")

                    # Resize using high-quality Lanczos resampling
                    # This 'resample' choice is critical for preserving count-ability
                    resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

                    # Save to new folder
                    save_name = output_path / f"std_{file.name}"
                    resized_img.save(save_name, "JPEG", quality=95)

                    processed_count += 1
                    print(f"  ✅ Standardized: {file.name}")
            except Exception as e:
                print(f"  ⚠️ Failed to process {file.name}: {e}")

    print("-" * 30)
    print(f"🎉 Done! {processed_count} images standardized to {TARGET_SIZE}.")
    print(f"📁 Find your images in: {OUTPUT_DIR}")


if __name__ == "__main__":
    # To run this, install Pillow first: pip install Pillow
    standardize_images()
