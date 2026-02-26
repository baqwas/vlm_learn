#!/usr/bin/env python3
"""
================================================================================
AI IMAGE DATA INTEGRITY VALIDATOR (PARALLELIZED)
================================================================================

DESCRIPTION:
    A high-performance utility designed to validate the quality of image datasets
    used in Vision Language Models (VLM) and CLIP training pipelines. It uses
    Laplacian Variance to detect image blur and simulates "quarantine" rates
    across a parametric range of thresholds.

PROCESSING WORKFLOW:
    1.  Initialization: Load hardware-agnostic settings from `config.toml`.
    2.  Discovery: Scan target directories for valid image formats (.jpg, .png).
    3.  Parallel Math: Distribute images across all available CPU cores using
        `concurrent.futures.ProcessPoolExecutor` for Laplacian computation.
    4.  Parametric Sweep: Iterate through threshold values to calculate
        quarantine counts and pass rates.
    5.  Reporting: Tabulate results to stdout and export a CSV to the source dir.

USER INTERFACE (CLI):
    Usage: python threshold_validator.py [--dir PATH] [--step INT]
    Arguments:
        --dir:  Override the input directory defined in config.toml.
        --step: Override the increment size for the threshold sweep.

EXCEPTION HANDLING & ERROR MESSAGES:
    - FileNotFoundError: Raised if 'config.toml' is missing from the parent dir.
    - OSError/cv2.error: Caught within worker processes to handle corrupted
      headers or locked files without crashing the main sweep.
    - "No valid images found": Logged if the target path is empty or invalid.

PREREQUISITES:
    - Python 3.11+ (for native tomllib support)
    - OpenCV (cv2), Pandas

LICENSE:
    MIT License
    Copyright (c) 2026 ParkCircus Productions; All Rights Reserved
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
================================================================================
"""
import cv2
import os
import pandas as pd
import logging
import argparse
import tomllib
from concurrent.futures import ProcessPoolExecutor

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageValidator:
    def __init__(self, config_path="../config.toml"):
        self.config = self._load_config(config_path)

    @staticmethod
    def _load_config(path):
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {path} not found.")
            raise

    @staticmethod
    def get_blur_score(image_path):
        """Static helper for parallel processing with targeted exceptions."""
        try:
            # cv2.imread is generally 'safe' but returns None on failure
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                # Log that the file exists but couldn't be decoded as an image
                return None

            return cv2.Laplacian(img, cv2.CV_64F).var()

        except (OSError, cv2.error) as e:
            # Only catch file system errors or OpenCV specific issues
            # We don't log here because it's running in a sub-process
            return None

    def run_sweep(self, directory=None, step_override=None):
        dir_path = directory or self.config['paths']['input_dir']
        step = step_override or self.config['threshold']['step']

        # Discover files
        valid_exts = ('.jpg', '.jpeg', '.png')
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                 if f.lower().endswith(valid_exts)]

        if not files:
            logger.error(f"No valid images found in {dir_path}")
            return None

        logger.info(f"Processing {len(files)} images using all CPU cores...")

        # --- Parallel Execution ---
        # Using ProcessPoolExecutor to bypass the GIL
        with ProcessPoolExecutor() as executor:
            # map returns results in the same order as the input list
            raw_scores = list(executor.map(self.get_blur_score, files))

        # Filter out failed reads
        scores = [s for s in raw_scores if s is not None]
        total = len(scores)

        # --- Threshold Sweep ---
        results = []
        start = self.config['threshold']['start_threshold']
        end = self.config['threshold']['end_threshold']

        for t in range(start, end + step, step):
            quarantined = sum(1 for s in scores if s < t)
            pass_rate = ((total - quarantined) / total) * 100
            results.append({
                "Threshold": t,
                "Quarantined": quarantined,
                "Pass Rate %": round(pass_rate, 2)
            })

        df = pd.DataFrame(results)

        # Save results
        output_csv = os.path.join(dir_path, "validation_report.csv")
        df.to_csv(output_csv, index=False)

        print("\n--- PARALLEL THRESHOLD REPORT ---")
        print(df.to_string(index=False))
        logger.info(f"Results saved to {output_csv}")

        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel AI Image Validator")
    parser.add_argument("--dir", type=str, help="Directory of images")
    parser.add_argument("--step", type=int, help="Sweep step size")
    args = parser.parse_args()

    validator = ImageValidator()
    validator.run_sweep(directory=args.dir, step_override=args.step)