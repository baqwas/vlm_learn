"""
================================================================================
PROJECT: VLM Counting Benchmark
MODULE:  validate_gt.py
VERSION: 1.1.0
LICENSE: MIT
================================================================================
DESCRIPTION:
    A robust validation utility for 'ground_truth.json'. It verifies structural
    integrity, coordinate normalization, and box geometry to ensure high-quality
    input for VLM benchmarking (Phase 2).

VALIDATION CHECKS:
    1. Schema validation (Dataset existence, correct keys).
    2. Zero-area boxes (Width or height = 0).
    3. Coordinate bounds (0-1024 range).
    4. Categorical distribution and progress reporting.

AUTHOR: Gemini / Reza
DATE: 2026
================================================================================
"""

import json
import logging
from pathlib import Path

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


class GTValidator:
    """Validator for Ground Truth JSON data."""

    def __init__(self, filename="ground_truth.json"):
        self.filename = Path(filename)
        self.target_size = 1024

    def validate(self):
        """Executes the validation suite and prints a summary report."""
        if not self.filename.exists():
            logging.error(f"File not found: {self.filename}")
            return

        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format in {self.filename}: {e}")
            return

        dataset = data.get('dataset', [])
        total_images = len(dataset)
        stats = {}
        issues = []

        print("\n" + "=" * 60)
        print(f"🔍 VALIDATING: {self.filename}")
        print("=" * 60)

        for entry in dataset:
            img_id = entry.get('image_id', 'Unknown')
            cat = entry.get('category', 'Uncategorized')
            bboxes = entry.get('bboxes', [])

            # Track counts per category
            stats[cat] = stats.get(cat, 0) + 1

            # Check 1: Empty labels
            if not bboxes:
                issues.append(f"⚠️  [{img_id}] Category '{cat}' has 0 labels.")

            # Check 2: Box Integrity [ymin, xmin, ymax, xmax]
            for i, box_wrapper in enumerate(bboxes):
                box = box_wrapper.get('box_2d', [])

                if len(box) != 4:
                    issues.append(f"❌ [{img_id}] Box {i} has incorrect dimensions.")
                    continue

                ymin, xmin, ymax, xmax = box

                # Check for zero area
                if ymax <= ymin or xmax <= xmin:
                    issues.append(f"❌ [{img_id}] Box {i} has zero or negative area.")

                # Check for out of bounds
                if any(val < 0 or val > self.target_size for val in box):
                    issues.append(f"❌ [{img_id}] Box {i} exceeds image boundaries (0-1024).")

        self._print_report(total_images, stats, issues)

    def _print_report(self, total, stats, issues):
        """Displays formatted summary of findings."""
        print(f"\n📈 COMPLETION SUMMARY:")
        for cat, count in stats.items():
            print(f"   - {cat:12}: {count} images")
        print(f"   - TOTAL       : {total} images")

        print(f"\n⚠️  QUALITY AUDIT:")
        if not issues:
            print("   ✅ All labels passed validation checks.")
        else:
            for issue in issues:
                print(f"   {issue}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    validator = GTValidator()
    validator.validate()