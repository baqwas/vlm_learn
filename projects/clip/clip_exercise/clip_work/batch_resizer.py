"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
VERSION: 1.0.1
UPDATED: 2026-03-05 08:13:21
COPYRIGHT: (c) 2026 ParkCircus Productions; All Rights Reserved.
AUTHOR: Matha Goram
LICENSE: MIT
PURPOSE: [REPLACE WITH FILE DESCRIPTION]
================================================================================
"""

"""
================================================================================
PROJECT: Professional Image Batch Resizer with Logging
AUTHOR:  Matha Goram
VERSION: 1.0.1
DATE:    2021-01-21
LICENSE: MIT License
COPYRIGHT: (c) 2026 ParkCircus Productions; All Rights Reserved
================================================================================
PROCESSING WORKFLOW:
1. Setup Logging: Initializes a persistent log file and console stream.
2. Load Config: Parses config.toml for directory and scaling parameters.
3. Batch Execution: Processes supported images using high-quality resampling.
4. Error Tracking: Records failures and successes to the log file.

DATA INTEGRITY LOGIC:
- Aspect Ratio Preservation: Mathematically calculates target height based on the source image's original dimensions to prevent distortion.
- Resampling Quality: Employs the LANCZOS filter for high-quality downsampling, preserving visual data for downstream VLM auditing.
- Storage Optimization: Implements standard JPEG/WebP optimization and quality weighting (85%) to balance file size with visual fidelity.

USER INTERFACE REQUIREMENTS:
- Progress Monitoring: Simultaneous StreamHandler output to terminal for live
  operational status (Success vs. Failure counters).
- Observability: Persistent local logging to facilitate headless auditing.
- Configuration: Externalized UI parameters (e.g., target_width) managed via
  config.toml to allow non-developer adjustments.
  - Headless Operation: Designed for non-interactive execution via terminal or cron job.
- Real-time Observability: Simultaneous output to console and local log files for live progress tracking.
- Batch Metrics: Final summary report indicating total success vs. failure counts for the entire session.

ERROR HANDLING & EXCEPTION STRATEGY:
- Directory Guardrails: Validates the existence of the input path and automatically creates the output directory if missing.
- Fault Tolerance: Uses isolated try-except blocks per file to ensure a single corrupt image does not terminate the batch process.
- Format Validation: Filters files by extension (.png, .jpg, .jpeg, .webp) before processing to avoid binary read errors.
- Logging Severity:
    - CRITICAL: Missing config.toml or inaccessible root directories.
    - ERROR: Failed to open or resample a specific image (logs reason and continues).

  PREREQUISITES:
- Python 3.11+ (for tomllib)
- Pillow (pip install pillow)
================================================================================
"""

import os
import tomllib
import logging
from PIL import Image
from pathlib import Path


def setup_logging(log_path):
    """Configures the logging utility to write to both file and console."""
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger("BatchResizer")


def batch_resize():
    # 1. PATH RESOLUTION & CONFIG LOAD
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.toml"

    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        print(f"CRITICAL: Config not found at {config_path}")
        return

    # 2. EXTRACT CONFIG & SETUP LOGGER
    logger = setup_logging(config["paths"]["log_file"])

    input_dir = root_dir / config["paths"]["input_dir"]
    output_dir = root_dir / config["paths"]["output_dir"]
    target_width = config["resize"]["new_width"]

    # 3. DIRECTORY VALIDATION
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initialized processing. Input: {input_dir} | Width: {target_width}px")

    # 4. PROCESSING LOOP
    processed_count = 0
    error_count = 0

    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            img_path = input_dir / filename
            save_path = output_dir / filename

            try:
                with Image.open(img_path) as img:
                    # Maintain aspect ratio
                    w_percent = target_width / float(img.size[0])
                    target_height = int((float(img.size[1]) * float(w_percent)))

                    resized_img = img.resize(
                        (target_width, target_height), Image.Resampling.LANCZOS
                    )
                    resized_img.save(save_path, optimize=True, quality=85)

                    logger.info(f"Successfully resized: {filename}")
                    processed_count += 1

            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
                error_count += 1

    logger.info(
        f"Batch Complete. Total Success: {processed_count}, Total Failures: {error_count}"
    )


if __name__ == "__main__":
    batch_resize()
