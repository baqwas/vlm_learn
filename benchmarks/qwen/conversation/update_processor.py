#!/usr/bin/env python3
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
# -*- coding: utf-8 -*-
"""
Update the AutoProcessor for Qwen2.5-VL-7B-Instruct
to the latest version to eliminate the warning about using the slow processor.
This script loads the processor from the Hugging Face Hub and saves it locally,
so you can use it in your main script without the warning.

This script is useful if you encounter a warning like:
    UserWarning: The processor for Qwen/Qwen2.5-VL-7B-Instruct is using the slow processor.
    Please use `AutoProcessor.from_pretrained(model_id, use_fast=False)` to load the slow processor
    or `AutoProcessor.from_pretrained(model_id, use_fast=True)` to load the new, faster one.

    Update your Python script to use the latest processor version as follows:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("./qwen-local-processor", trust_remote_code=True)
References:

"""
import os
from transformers import AutoProcessor

# Define the model ID
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# Define the local folder where you want to save the updated processor
output_folder = "./qwen-local-processor"

print(f"Loading and saving the processor to {output_folder}...")

try:
    # Load the processor from the Hugging Face Hub
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Save the processor to the local folder.
    # This automatically updates the file format.
    processor.save_pretrained(output_folder)

    print("Processor saved successfully! The warning should now be gone.")
    print(f"You can now load the processor from '{output_folder}' in your main script.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you have `transformers` and `qwen-vl-utils` installed.")
