#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Counting Benchmark
MODULE:  labeler.py
AUTHOR:  Matha Goram
VERSION: 1.3.0
DATE:    2026-02-12
COPYRIGHT (c) ParkCircus Productions; All Rights Reserved
LICENSE: MIT
================================================================================
DESCRIPTION:
    A high-precision bounding box annotation tool designed for VLM Ground Truth
    generation. Features dynamic category-aware window titles to guide the
    user through diverse labeling tasks (Subitizing, Patterns, etc.).

    This script outputs coordinates in the [ymin, xmin, ymax, xmax] format,
    normalized to a 1024x1024 coordinate space, specifically for VLM models
    like Qwen2.5-VL.

KEYBOARD CONTROLS:
    [MOUSE DRAG]   Define bounding box area.
    [SPACE/ENTER]  Confirm/Register the current box.
    [C]            Clear all boxes for the current image.
    [N]            Save data and proceed to the NEXT image.
    [Q]            Save and QUIT the session immediately.

REQUIREMENTS:
    - opencv-python (Full version required for GUI/ROI support)
    - pathlib
================================================================================
"""

import cv2
import json
import os
import sys
import logging
from pathlib import Path

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BBoxLabeler:
    """Main application class for category-aware image annotation."""

    def __init__(self, base_dir="../images/benchmark", output_file="ground_truth.json"):
        self.base_dir = Path(base_dir)
        self.output_file = Path(output_file)
        self.dataset = self._load_json()
        # Track already labeled images to allow resumption of sessions
        self.processed_files = {item['image_id'] for item in self.dataset.get('dataset', [])}

    def _load_json(self):
        """Loads existing ground truth data or initializes new structure."""
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logging.error(f"Failed to load {self.output_file}: {e}")
                sys.exit(1)
        return {"project": "VLM Counting Benchmark", "version": "1.3.0", "dataset": []}

    def _save_json(self):
        """Persists dataset to disk in a human-readable JSON format."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, indent=4)
        except IOError as e:
            logging.error(f"Critical Error: Could not save data to disk! {e}")

    def _display_instructions(self, idx, total, cat, name):
        """Renders CLI instructions for the user's reference."""
        print("\n" + "=" * 75)
        print(f" 🚀 PROGRESS:  [{idx}/{total}]")
        print(f" 🎯 CATEGORY:  {cat.upper()}")
        print(f" 📂 FILE:      {name}")
        print("-" * 75)
        print(" [ACTION]        [INPUT]")
        print(" Draw Box:       Left-Click & Drag Mouse")
        print(" Confirm Box:    Press SPACE or ENTER")
        print(" Reset Image:    Press 'C'")
        print(" Next Image:     Press 'N'")
        print(" Quit Session:   Press 'Q'")
        print("=" * 75)

    def start_session(self):
        """Main loop for processing images and managing the OpenCV UI."""
        # Find all JPGs in the base directory and subdirectories
        image_list = list(self.base_dir.rglob("*.jpg"))
        total_count = len(image_list)

        if total_count == 0:
            logging.warning(f"No images found in {self.base_dir}. Verify folder path.")
            return

        logging.info(f"Session initialized. Scanning {total_count} total images.")

        for idx, img_path in enumerate(image_list, start=1):
            # Skip if image has been processed in a previous session
            if img_path.name in self.processed_files:
                continue

            category = img_path.parent.name
            # Set the window title to include the current Category for user clarity
            window_title = f"[{idx}/{total_count}] TARGET: {category.upper()} | File: {img_path.name}"

            self._display_instructions(idx, total_count, category, img_path.name)

            img = cv2.imread(str(img_path))
            if img is None:
                logging.error(f"Skipping unreadable file: {img_path.name}")
                continue

            # Fallback resize to ensure 1:1 mapping with the 1024px coordinate system
            if img.shape[0] != 1024 or img.shape[1] != 1024:
                img = cv2.resize(img, (1024, 1024))

            canvas = img.copy()
            current_bboxes = []

            # Create an explicit window for each image to update the Title Bar
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

            while True:
                # selectROI returns [xmin, ymin, width, height]
                rect = cv2.selectROI(window_title, canvas, fromCenter=False, showCrosshair=True)

                x, y, w, h = rect
                if w > 0 and h > 0:
                    # Convert to VLM standard format: [ymin, xmin, ymax, xmax]
                    bbox = [int(y), int(x), int(y + h), int(x + w)]
                    current_bboxes.append(bbox)

                    # Draw a persistent green box on the canvas for user feedback
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    logging.info(f"Object Registered: {bbox}")

                # Wait for user keypress
                key = cv2.waitKey(0) & 0xFF
                if key == ord('n'):  # Next image
                    break
                elif key == ord('c'):  # Clear image
                    canvas = img.copy()
                    current_bboxes = []
                    logging.info("Current image cleared.")
                elif key == ord('q'):  # Quit session
                    logging.info("Saving progress and exiting...")
                    cv2.destroyAllWindows()
                    return

            # Store the data for this image
            self.dataset['dataset'].append({
                "image_id": img_path.name,
                "category": category,
                "gt_count": len(current_bboxes),
                "bboxes": [{"label": "object", "box_2d": b} for b in current_bboxes]
            })
            self.processed_files.add(img_path.name)
            self._save_json()

            # Close window for current image to prepare for the next title bar
            cv2.destroyWindow(window_title)

        logging.info("🏁 All images in the directory have been labeled.")


if __name__ == "__main__":
    # Parameters can be adjusted if your folder structure changes
    labeler = BBoxLabeler(
        base_dir="../images/benchmark",
        output_file="ground_truth.json"
    )
    labeler.start_session()
