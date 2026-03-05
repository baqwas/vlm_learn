#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate text based on Visual Question Answering on an image using the Qwen2.5-VL-7B-Instruct model
with the correct AutoProcessor.

CREATE TABLE VLM_performance_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_id VARCHAR(255) NOT NULL, -- New column for the model ID
    image_file VARCHAR(255) NOT NULL,
    query VARCHAR(255) NOT NULL,
    response TEXT NOT NULL,
    generated_tokens INT,
    generation_time_seconds FLOAT,
    tokens_per_second FLOAT,
    log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
import argparse, configparser
import torch
from sympy.solvers.ode import constantsimp
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image, UnidentifiedImageError
import json
import os, time
import mariadb
from mariadb import mariadb.constants

APP_NAME = "batch_performance"

def text_image_generation_local(image_folder_path, output_log_file, db_config):
    """
    image_folder_path: Path to the folder containing images
    output_log_file: Path to the log file where performance metrics will be saved
    db_config: Dictionary containing database connection details
"""
# 1. Define the model ID
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# 2. Load the model and processor
print(f"Loading model and processor for {model_id}...")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    print("Model and processor loaded successfully!")

except Exception as e:
    print(f"Error loading model or processor: {e}")
    print("Please ensure you have installed `transformers`, `torch`, `pillow`, `accelerate` and `qwen-vl-utils`.")
    return

# 3. Create a list of image files to process
try:
    image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    if not image_files:
        print(f"No JPG or JPEG images found in the folder: {image_folder_path}")
        return
except FileNotFoundError:
    print(f"Error: The image folder '{image_folder_path}' was not found.")
    return

conn = None
try:
    # 4. Establish database connection once, before the loop
    print("Connecting to the MariaDB database...")
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    print("Connection successful!")

    # 5. SQL INSERT statement
    insert_query = """
INSERT
INTO
image_performance_log(
    image_file,
    query,
    response,
    generated_tokens,
    generation_time_seconds,
    tokens_per_second
)
VALUES( % s, %s, %s, %s, %s, %s)
"""

# 6. Open the log file in append mode before the loop
with open(output_log_file, 'a') as log_file:
    log_file.write("--- Image Captioning Performance Log ---\n")
    log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 7. Loop through each image and process it individually
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        query = "Describe the image in detail."

        try:
            print(f"Processing image: {image_path}")
            image = Image.open(image_path).convert("RGB")

            messages = [{"role": "user", "content": [{"image": image, "text": query}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

            print(f"Asking the model to {query}....")

            start_time = time.time()
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
            end_time = time.time()
            generation_time = end_time - start_time
            num_generated_tokens = generated_ids.shape[1] - inputs.input_ids.shape[1]

            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            json_record = {
                "image_file": image_file,
                "query": query,
                "response": response,
                "generated_tokens": num_generated_tokens,
                "generation_time_seconds": round(generation_time, 2)
            }
            if generation_time > 0:
                json_record["tokens_per_second"] = round(num_generated_tokens / generation_time, 2)

            log_file.write(json.dumps(json_record) + "\n")

            data_to_insert = (
                json_record.get("image_file"),
                json_record.get("query"),
                json_record.get("response"),
                json_record.get("generated_tokens"),
                json_record.get("generation_time_seconds"),
                json_record.get("tokens_per_second", None)
            )
            cursor.execute(insert_query, data_to_insert)

            print(f"\n--- Image {image_path} Result ---")
            print(f"Response: {response}")

        except UnidentifiedImageError:
            print(f"Error: The file '{image_file}' is not a valid image file. Skipping.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing '{image_file}': {e}. Skipping.")
            continue

conn.commit()
print("\nAll records successfully committed to the database.")

except mysql.connector.Error as err:
if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with your user name or password.")
elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist.")
else:
    print(f"Database error: {err}")
except FileNotFoundError:
print(f"Error: The image folder or log file was not found.")
except Exception as e:
print(f"An unexpected error occurred: {e}")

finally:
if conn and conn.is_connected():
    cursor.close()
    conn.close()
    print("Database connection closed.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = configparser.ConfigParser()

    defaults = {
        "config_file": f"{APP_NAME}.ini",
        "folder": "../images/album",
        "kpi_log": f"{APP_NAME}.log",
        "db_host": None,
        "db_user": None,
        "db_password": None,
        "db_name": None
    }

    parser = argparse.ArgumentParser(
        description=f"Run a batch performance test on images with {APP_NAME}.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=defaults["config_file"],
        help="Path to the configuration file"
    )

    config_args, unknown = parser.parse_known_args()
    config_file_name = config_args.config_file

    # Check if the config file exists and read it
    if os.path.exists(config_file_name):
        config.read(config_file_name)
        print(f"Reading defaults from configuration file: {config_file_name}")

        # Load from the [Settings] section
        if "Settings" in config:
            if "folder" in config["Settings"]:
                defaults["folder"] = config["Settings"]["folder"]
            if "kpi_log" in config["Settings"]:
                defaults["kpi_log"] = config["Settings"]["kpi_log"]

        # Load from the new [Database] section
        if "Database" in config:
            if "db_host" in config["Database"]:
                defaults["db_host"] = config["Database"]["db_host"]
            if "db_user" in config["Database"]:
                defaults["db_user"] = config["Database"]["db_user"]
            if "db_password" in config["Database"]:
                defaults["db_password"] = config["Database"]["db_password"]
            if "db_name" in config["Database"]:
                defaults["db_name"] = config["Database"]["db_name"]
        else:
            print(f"Warning: No '[Database]' section found in '{config_file_name}'. Database operations will likely fail.")

    else:
        print(f"Warning: Configuration file '{config_file_name}' not found. Using hardcoded defaults.")

    parser.add_argument("--folder", type=str, default=defaults["folder"], help="Path to the folder containing images")
    parser.add_argument("--kpi_log", type=str, default=defaults["kpi_log"], help="Path to the log file for performance metrics")
    parser.add_argument("--db_host", type=str, default=defaults["db_host"], help="Database host")
    parser.add_argument("--db_user", type=str, default=defaults["db_user"], help="Database user")
    parser.add_argument("--db_password", type=str, default=defaults["db_password"], help="Database password")
    parser.add_argument("--db_name", type=str, default=defaults["db_name"], help="Database name")

    args = parser.parse_args()

    album = os.path.join(script_dir, args.folder)
    print(f"Using folder: {album} & kpi_log file: {args.kpi_log}")

    db_config = {
        'host': args.db_host,
        'user': args.db_user,
        'password': args.db_password,
        'database': args.db_name
    }

    # Pass the database config to the main function
    text_image_generation_local(album, args.kpi_log, db_config)
