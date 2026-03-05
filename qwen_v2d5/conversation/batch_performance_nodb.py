#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate text based on Visual Question Answering on an image
using the Qwen2.5-VL-7B-Instruct model with the correct AutoProcessor.

References
    https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/README.md
"""
import argparse, configparser
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image, UnidentifiedImageError
import json  # to work with JSON data
import os, time  # Add this import at the top

APP_NAME = "batch_performance"


def text_image_generation_local(image_folder_path, output_log_file):
    """
    image_folder_path: Path to the folder containing images
    output_log_file: Path to the log file where performance metrics will be saved
    """
    # 1. Define the model ID
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    # 2. Load the model and processor
    print(f"Loading model and processor for {model_id}...")
    try:
        # Load the model using its specific class
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

        # FIX: Use AutoProcessor and set `use_fast=False` to explicitly use the slow processor
        # that the model was trained with, which is what the warning recommends.
        # Alternatively, you could try setting `use_fast=True` to use the new, faster one.
        # This will eliminate the first warning.
        processor = AutoProcessor.from_pretrained(
            "qwen-local-processor", trust_remote_code=True, use_fast=False
        )  # Use the slow processor as recommended for BeUlta machine
        print("Model and processor loaded successfully!")

    except Exception as e:
        print(f"Error loading model or processor: {e}")
        print(
            "Please ensure you have installed `transformers`, `torch`, `pillow`, `accelerate` and `qwen-vl-utils`."
        )
        return None

    # 3. Prepare the image (read from the local project folder)
    query = "Describe the image in detail."
    try:
        # Create a list of image files to process
        image_files = [
            f
            for f in os.listdir(image_folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not image_files:
            print(
                f"No images (filetypes: JPG, JPEG or PNG) found in the folder: {image_folder_path}"
            )
            return None
    except FileNotFoundError:
        print(f"Error: The image folder '{image_folder_path}' was not found.")
        return None

    # Open the log file in appended mode
    with open(output_log_file, "a") as log_file:
        log_file.write("--- Image Captioning Performance Log ---\n")
        log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 5. Loop through each image and process it individually
        for image_file in image_files:
            image_path = os.path.join(image_folder_path, image_file)
            query = "Describe the image in detail."

            try:
                print(f"Processing image: {image_path}")
                image = Image.open(image_path).convert("RGB")

                # Prepare the input for the model
                messages = [
                    {"role": "user", "content": [{"image": image, "text": query}]}
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = processor(text=[text], images=[image], return_tensors="pt").to(
                    model.device
                )

                print(f"Asking the model to {query} ...")

                # Generate the response and measure performance
                start_time = time.time()
                generated_ids = model.generate(
                    **inputs, max_new_tokens=512, do_sample=True, temperature=0.7
                )
                end_time = time.time()
                generation_time = end_time - start_time
                num_generated_tokens = (
                    generated_ids.shape[1] - inputs.input_ids.shape[1]
                )

                response = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                # Write results to the log file (now inside the loop)
                json_record = {
                    "model_id": model_id,
                    "image_file": image_file,
                    "query": query,
                    "response": response,
                    "generated_tokens": num_generated_tokens,
                    "generation_time_seconds": round(generation_time, 2),
                }
                if generation_time > 0:
                    json_record["tokens_per_second"] = round(
                        num_generated_tokens / generation_time, 2
                    )

                log_file.write(json.dumps(json_record) + "\n")

                # Print the result to the console for this specific image
                print(f"\n--- Image {image_path} Result ---")
                print(f"Response: {response}")

            except UnidentifiedImageError:
                print(
                    f"Error: The file '{image_file}' is not a valid image file. Skipping."
                )
                continue
            except Exception as e:
                print(
                    f"An unexpected error occurred while processing '{image_file}': {e}. Skipping."
                )
                continue


if __name__ == "__main__":
    # Define the folder containing your images and the log file name
    script_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        description="Describe all images in a folder individually."
    )
    # Parse arguments specifically for the config file first
    # This allows us to use the specified config file to load other defaults
    parser.add_argument(
        "--config_file",
        type=str,
        default=f"{APP_NAME}.ini",
        help="Optional path to a configuration file to load default settings from "
        f"(default: '{APP_NAME}.ini').",
    )
    config = configparser.ConfigParser()
    defaults = dict(
        folder="../images/album",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        kpi_log=f"{APP_NAME}.log",
    )
    config_args, unknown = parser.parse_known_args()
    config_file_name = config_args.config_file
    print()
    # Add the folder and kpi_log arguments to the parser
    parser.add_argument(
        "--folder",
        type=str,
        default=defaults["folder"],
        help="Path to the folder containing images (default: ../images/album)",
    )
    parser.add_argument(
        "--kpi_log",
        type=str,
        default=defaults["kpi_log"],
        help=")Path to the log file for performance metrics (default: performance_log.txt)",
    )
    if os.path.exists(config_file_name):
        config.read(config_file_name)
        print(f"Reading defaults from configuration file: {config_file_name}")
        if "Settings" in config:
            if "folder" in config["Settings"]:
                defaults["folder"] = config["Settings"]["folder"]
            if "kpi_log" in config["Settings"]:
                defaults["kpi_log"] = config["Settings"]["kpi_log"]
        else:
            print(
                f"Warning: No '[Settings]' section found in '{config_file_name}'. Using hardcoded defaults."
            )
    else:
        print(
            f"Warning: Configuration file '{config_file_name}' not found. Using hardcoded defaults."
        )
    album = os.path.join(script_dir, defaults["folder"])
    print(f"Using folder: {album} & kpi_log file: {defaults['kpi_log']}")

    # Call the main function with the new parameters
    text_image_generation_local(album, defaults["kpi_log"])
