#!/usr/bin/env python3
"""
================================================================================
PROJECT: Google SigLIP Vegetable Evaluator (with Enhanced Summary & Triage)
AUTHOR:  Matha Goram
VERSION: 1.3.0
DATE:    2021-07-09
Copyright (c) 2026 ParkCircus Productions
LICENSE: MIT License
================================================================================
PROCESSING WORKFLOW:
1. Setup Logging: Persistent logging for audit trails.
2. Load Configuration: Retrieves model, paths, and thresholds.
3. Model Initialization: Ingests Google SigLIP weights.
4. Inference & Triage:
    - Calculates Sigmoid similarity for each image-label pair.
    - If top score >= threshold: Tally the result and keep in output_dir.
    - If top score < threshold: Move file to quarantine_dir for review.
5. Enhanced Reporting:
    - Outputs a total count of scanned, successful, and quarantined images.
    - Provides a categorical breakdown of all successfully identified vegetables.

DATA INTEGRITY LOGIC:
- Sigmoid Similarity: Utilizes Google SigLIP's sigmoid-based activation to calculate
  probability scores for each image against a dynamic list of vegetable labels.
- Automated Triage: Implements a "Confidence Guardrail" where images failing to meet
   the 'threshold' parameter are physically isolated from the main dataset.
- Quarantine Protocol: Moves low-confidence data to a dedicated 'quarantine_dir'
  to preserve the integrity of the 'output_dir' for downstream AI training.

USER INTERFACE REQUIREMENTS:
- Batch Observability: Real-time terminal streaming of confidence scores (e.g., MATCH vs. UNCERTAIN)
  to provide live visibility into model performance.
- Summary Analytics: Generates a post-process "Executive Summary" including total scan counts,
  quarantine rates, and a percentage-based breakdown of successful categorizations.
- Persistence: All decision logic must be mirrored in a local audit trail (log file) for
  post-hoc verification of automated triage decisions.

ERROR HANDLING & EXCEPTION STRATEGY:
- Path Validation: Verification of 'root_dir' and 'config.toml' resolution using
  absolute Ubuntu paths before model ingestion.
- File System Resilience: Uses 'shutil.move' within try-except blocks to handle
  potential permission errors or file locks during the quarantine move.
- Inference Protection: Wrapped torch.no_grad() blocks to prevent memory leaks
  and ensure session stability during massive batch iterations.
- Logging Severity:
    - CRITICAL: Model weight initialization failure or missing label configuration.
    - WARNING: Specific image moved to quarantine due to low confidence score.
    - ERROR: Corruption in image binary preventing PIL open or SigLIP processing.

PREREQUISITES:
- Python 3.11+
- transformers, torch, Pillow, shutil
================================================================================
"""

import os
import tomllib
import logging
import torch
import shutil
from PIL import Image
from pathlib import Path
from collections import Counter
from transformers import AutoProcessor, AutoModel


def setup_logging(log_path):
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger("SigLIP_Quarantine")


def run_evaluation():
    # 1. LOAD CONFIGURATION
    # Resolves path to find config.toml in the parent directory
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.toml"

    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        print(f"CRITICAL: Config not found at {config_path}")
        return

    logger = setup_logging(config["paths"]["log_file"])

    # 2. DIRECTORY PREPARATION
    image_dir = root_dir / config["paths"]["output_dir"]
    quarantine_dir = root_dir / config["paths"]["quarantine_dir"]
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # 3. INITIALIZE MODEL
    model_name = config["model"]["name"]
    threshold = config["model"].get("threshold", 0.30)

    logger.info(f"Loading Model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    labels = config["labels"]["vegetables"]

    results_tally = Counter()
    quarantine_count = 0
    images = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    total_scanned = len(images)

    logger.info(
        f"Starting eval on {total_scanned} images. Confidence Threshold: {threshold}"
    )

    # 4. INFERENCE & TRIAGE LOOP
    for img_name in images:
        img_path = image_dir / img_name

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(
                text=labels,
                images=image,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.sigmoid(outputs.logits_per_image)
            conf, idx = torch.max(probs, dim=1)
            conf_val = conf.item()

            if conf_val < threshold:
                # ACTION: QUARANTINE
                logger.warning(
                    f"UNCERTAIN ({conf_val:.2f}): Moving {img_name} to quarantine."
                )
                shutil.move(str(img_path), str(quarantine_dir / img_name))
                quarantine_count += 1
            else:
                # ACTION: TALLY SUCCESS
                label = labels[idx.item()]
                results_tally[label] += 1
                logger.info(f"MATCH ({conf_val:.2f}): {img_name} -> {label}")

        except Exception as e:
            logger.error(f"Failed to process {img_name}: {e}")

    # 5. ENHANCED SUMMARY
    total_success = sum(results_tally.values())

    logger.info("=" * 50)
    logger.info("FINAL BATCH EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"{'TOTAL IMAGES SCANNED':<35} : {total_scanned}")
    logger.info(f"{'SUCCESSFULLY PROCESSED':<35} : {total_success}")
    logger.info(f"{'QUARANTINED (LOW CONFID)':<35} : {quarantine_count}")
    logger.info("-" * 50)
    logger.info("CATEGORY BREAKDOWN (SUCCESSES ONLY):")

    if total_success > 0:
        for veg, count in results_tally.most_common():
            # Calculate percentage for additional context
            percentage = (count / total_success) * 100
            logger.info(f" - {veg.upper():<32} : {count} ({percentage:.1f}%)")
    else:
        logger.info(" - No images were successfully categorized.")

    logger.info("=" * 50)


if __name__ == "__main__":
    run_evaluation()
