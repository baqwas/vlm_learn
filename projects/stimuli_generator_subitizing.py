"""
🎯 VLM Subitizing Stimuli Generator
====================================
A utility for procedurally generating vision-language model
counting benchmarks with high-fidelity ground truth.

Processing Workflow:
1. Environment Setup: Validate/create output directories.
2. Geometry Engine: Generate non-overlapping coordinates using Euclidean distance.
3. Rendering: Rasterize shapes onto a standardized canvas.
4. Data Export: Serialize bounding boxes in [ymin, xmin, ymax, xmax] format.

MIT License
Copyright (c) 2026 ParkCircus Productions
Full license text available in project root.

🛠️ Starting point
* Zero-Cost Data: It generates the first 10%–20% of your dataset instantly without requiring manual searching or labeling.
* Calibration: It establishes the "Ground Truth" format ([ymin, xmin, ymax, xmax]) that you will need to match when you manually label the other 40 images.
* Environment Check: It confirms your Python environment and Pillow library are correctly configured before you move into more complex batch processing.
"""

import os
import json
import random
import sys
import logging
from math import sqrt
from pathlib import Path
from PIL import Image, ImageDraw

# --- 🛠️ CONFIGURATION ---
LOG_FORMAT = "%(levelname)s ⮕ %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# --- 🧩 CORE LOGIC ---

def generate_subitizing_set(
        output_dir: str = "../images/benchmark/subitizing",
        num_images: int = 10,
        img_size: int = 1024,
        radius: int = 40
) -> None:
    """
    Generates a subitizing dataset with JSON ground truth.

    Requirements:
        - Pillow (pip install Pillow)

    User Interface Instructions:
        1. Set the 'num_images' for your count range (e.g., 10 or 50).
        2. Execute script via terminal: `python stimuli_gen.py`.
        3. Access images and 'ground_truth.json' in your specified output_dir.
    """
    logger.info("🚀 Initializing Generation Workflow...")

    # Exception Handling: Directory Management
    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"❌ Failed to create directory {output_dir}: {e}")
        sys.exit(1)

    metadata = {}
    min_dist = radius * 2.5  # Buffer to prevent visual crowding

    for n in range(1, num_images + 1):
        try:
            # Create a high-fidelity white canvas
            img = Image.new('RGB', (img_size, img_size), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            bboxes = []
            centers = []

            attempts = 0
            placed = 0

            # Non-overlapping Geometry Logic
            while placed < n and attempts < 2000:
                x = random.randint(radius, img_size - radius)
                y = random.randint(radius, img_size - radius)

                # Check for collisions with existing centers
                is_overlapping = any(
                    sqrt((x - cx) ** 2 + (y - cy) ** 2) < min_dist for cx, cy in centers
                )

                if not is_overlapping:
                    # Draw visual element
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill='green', outline='black')

                    # Store normalized [ymin, xmin, ymax, xmax] for VLM benchmarking
                    bboxes.append([y - radius, x - radius, y + radius, x + radius])
                    centers.append((x, y))
                    placed += 1
                attempts += 1

            if placed < n:
                logger.warning(f"⚠️ Image {n}: Resource limit reached. Placed {placed}/{n} objects.")

            # File IO Workflow
            img_filename = f"subitizing_{n:02d}.jpg"
            img_path = out_path / img_filename
            img.save(img_path, "JPEG", quality=95)

            metadata[img_filename] = {
                "count": placed,
                "bboxes": bboxes,
                "resolution": [img_size, img_size]
            }
            logger.info(f"  🖼️ Created: {img_filename} (Objects: {placed})")

        except Exception as e:
            logger.error(f"❌ Error during image {n} generation: {e}")

    # Metadata persistence
    try:
        json_path = out_path / "ground_truth.json"
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ SUCCESS: Dataset stored in '{output_dir}'.")
    except IOError as e:
        logger.error(f"❌ Failed to write Ground Truth JSON: {e}")


if __name__ == "__main__":
    generate_subitizing_set()