#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file vqa_single_save.py
@brief Generate text based on Visual Question Answering on an image using the Qwen2.5-VL-7B-Instruct model with the correct AutoProcessor.
@details
Generate text based on Visual Question Answering on an image using the Qwen2.5-VL-7B-Instruct model with the correct AutoProcessor.

This script processes images in a specified folder, generates captions using the model, and logs performance metrics to a file and a MariaDB database.

This script requires the following Python packages:
- transformers, torch, pillow, mariadb, opencv-python, mutagen

Mutagen for audio file properties, if necessary for use case
MariaDB Connector/C for mariadb database operations.
Sudo apt-get install libmariadb3 libmariadb-dev

It is designed to be run in a local environment with access to the specified model and database.

This script is part of a batch performance testing suite for image captioning tasks.
@author: Matha Goram
@version: 1.0
@date: 2024-01-01
@note: Ensure you have the necessary permissions to access the database and the image folder.
@note: This script assumes the existence of a MariaDB database with a table named 'performance_log_vlm' that has the appropriate schema.
@note: The script uses the Qwen2.5-VL-7B-Instruct model, which should be downloaded and available in the local environment.
@note: The script is designed to handle various file types, including images, videos, and audio files, but focuses on image captioning.
@note: The script logs performance metrics such as generation time, tokens per second, and file properties to a specified log file and database.
@note: The script includes error handling for file reading and database operations to ensure robustness.
@note: The script can be customized via command-line arguments or configuration files for different use cases.
@note: The script is intended for use in processing a single set of images ond corresponding query.
@note: The script can be run from the command line with various options to specify the input folder, log file, and database connection details.
@note: The script is designed to be modular and can be integrated into larger workflows for image processing and analysis.
@note: The script is compatible with Python 3.8 and above, and requires the specified packages to be installed in the Python environment.
@note: The script is part of a larger suite of tools for visual language models and image processing tasks.
@note: The script is designed to be run in a local environment with access to the specified model and database.

"""
import argparse, configparser
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image, UnidentifiedImageError
import json
import os, time
import mariadb
from mariadb.constants import ERR
import cv2
from mutagen.mp3 import MP3

APP_NAME = "vqa_single_save"

def get_file_properties(file_path):
    """
    Determines file type and extracts properties based on file extension.
    """
    file_properties = {}
    file_type = "text"

    file_extension = os.path.splitext(file_path)[1].lower()

    # Image properties
    image_extensions = ['.jpg', '.jpeg', 'JPG', '.png', '.bmp', '.gif']
    if file_extension in image_extensions:
        try:
            with Image.open(file_path) as img:
                file_properties['width'] = img.width
                file_properties['height'] = img.height
            file_type = "image"
        except UnidentifiedImageError:
            print(f"Warning: Could not identify image file: {file_path}")
        except Exception as e:
            print(f"Error reading image properties for {file_path}: {e}")
        return file_type, file_properties

    # Video properties
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    if file_extension in video_extensions:
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                file_properties['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                file_properties['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                file_properties['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                file_properties['fps'] = cap.get(cv2.CAP_PROP_FPS)
                file_type = "video"
            cap.release()
        except Exception as e:
            print(f"Error reading video properties for {file_path}: {e}")
        return file_type, file_properties

    # Audio properties
    audio_extensions = ['.mp3', '.wav', '.flac']
    if file_extension in audio_extensions:
        try:
            if file_extension == '.mp3':
                audio = MP3(file_path)
                file_properties['duration_seconds'] = int(audio.info.length)
                file_properties['bitrate_kbps'] = audio.info.bitrate // 1000
            file_type = "audio"
        except Exception as e:
            print(f"Error reading audio properties for {file_path}: {e}")
        return file_type, file_properties

    return file_type, file_properties


def text_image_generation_local(image_folder_path, output_log_file, db_config, platform, user_query):
    """
    image_folder_path: Path to the folder containing images
    output_log_file: Path to the log file where performance metrics will be saved
    db_config: Dictionary containing database connection details
    platform: The platform string from the config file
    """
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    print(f"Loading model and processor for {model_id}...")
    try:
        """
        Use the update_processor.py script to update the processor for the Qwen2.5-VL-7B-Instruct model.
        """
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, # deprecated: use "./qwen-local-processor", instead
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False
        )
        print("Model and processor loaded successfully!")

    except Exception as e:
        print(f"Error loading model or processor: {e}")
        print("Please ensure you have installed `transformers`, `torch`, `pillow`, `accelerate` and `qwen-vl-utils`.")
        return

    try:
        all_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]
        if not all_files:
            print(f"No files found in the folder: {image_folder_path}")
            return
    except FileNotFoundError:
        print(f"Error: The folder '{image_folder_path}' was not found.")
        return

    conn = None
    try:
        print("Connecting to the MariaDB database...")
        # Establishing connection to the database using appropriate charset
        conn = mariadb.connect(**db_config) #, charset='utf8mb4')
        cursor = conn.cursor()
        print("Connection successful!")

        # Updated SQL INSERT statement to include new columns
        insert_query = """
        INSERT INTO performance_log_vlm (
            model_id,
            platform,
            file_name,
            file_type,
            query,
            response,
            generated_tokens,
            generation_time,
            tokens_per_second,
            file_properties
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with (open(output_log_file, 'a') as log_file):
            log_file.write("--- Image Captioning Performance Log ---\n")
            log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for file_name in all_files:
                file_path = os.path.join(image_folder_path, file_name)
                break

            file_type, file_properties = get_file_properties(file_path)

            file_types_valid = ["image", "video", "audio", "text"]
            if file_type in file_types_valid:

                try:
                    print(f"Processing image: {file_path}")
                    image = Image.open(file_path).convert("RGB")

                    messages = [
                        {"role": "user",
                         "content": [
                             {"image": image,
                              "text": user_query}
                         ]
                         }
                    ]
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    inputs = processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt"
                    ).to(model.device)

                    print(f"Asking the model to {user_query} ...")

                    start_time = time.time()
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7
                    )
                    end_time = time.time()
                    generation_time = end_time - start_time
                    print(f"")
                    num_generated_tokens = generated_ids.shape[1] - inputs.input_ids.shape[1]

                    response = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]

                    json_record = {
                        "model_id": model_id,
                        "platform": platform,
                        "file_name": file_name,
                        "file_type": file_type,
                        "query": user_query,
                        "response": response,
                        "generated_tokens": num_generated_tokens,
                        "generation_time": round(generation_time, 2),
                        "tokens_per_second": None,
                        "file_properties": file_properties
                    }
                    if generation_time > 0:
                        tokens_per_second = round(num_generated_tokens / generation_time, 2)
                        json_record["tokens_per_second"] = tokens_per_second

                    log_file.write(json.dumps(json_record) + "\n")

                    data_to_insert = (
                        json_record.get("model_id"),
                        json_record.get("platform"),
                        json_record.get("file_name"),
                        json_record.get("file_type"),
                        json_record.get("query"),
                        json_record.get("response"),
                        json_record.get("generated_tokens"),
                        json_record.get("generation_time"),
                        json_record.get("tokens_per_second"),
                        json.dumps(json_record.get("file_properties", {}))
                    )
                    cursor.execute(insert_query, data_to_insert)
                    # 4. Explicitly commit the transaction for this record
                    conn.commit()
                    print(f"Transaction committed for '{file_name}'.")

                    print(f"\n--- Image {file_path} Result ---\nResponse: {response}")

                except UnidentifiedImageError:
                    print(f"Error: The file '{file_name}' is not a valid image file. Skipping.")
                except Exception as e:
                    print(f"An unexpected error occurred while processing '{file_name}': {e}. Skipping.")

            else:
                print(f"Skipping non-image file: {file_name} (type: {file_type})")

        conn.commit()
        print("\nAll records successfully committed to the database.")

    except mariadb.Error as err:
        if err.errno == mariadb.constants.ERR.ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password.")
        elif err.errno == mariadb.constants.ERR.BAD_DB_ERROR:
            print("Database does not exist.")
        else:
            print(f"Database error: {err}")
    except FileNotFoundError:
        print(f"Error: The image folder or log file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        if conn: # and conn.is_connected():
            cursor.close()
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_app = configparser.ConfigParser()
    config_db = configparser.ConfigParser()

    defaults = {
        "config_file": f"{APP_NAME}.ini",
        "db_config_file": "db_config.ini",
        "folder": "../images/album",
        "kpi_log": f"{APP_NAME}.log",
        "platform": "Ubuntu 25.04 x86_64",
        "chat_query": "Describe the image in detail.",
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
        help="Path to the application configuration file"
    )
    parser.add_argument(
        "--db_config_file",
        type=str,
        default=defaults["db_config_file"],
        help="Path to the database configuration file"
    )

    config_args, unknown = parser.parse_known_args()
    app_config_file_name = config_args.config_file
    db_config_file_name = config_args.db_config_file

    # Read application settings
    if os.path.exists(app_config_file_name):
        config_app.read(app_config_file_name)
        print(f"Reading application defaults from: {app_config_file_name}")

        if "Settings" in config_app:
            defaults["folder"] = config_app["Settings"].get("folder", defaults["folder"])
            defaults["kpi_log"] = config_app["Settings"].get("kpi_log", defaults["kpi_log"])
            defaults["chat_query"] = config_app["Settings"].get("chat_query", defaults["chat_query"])
        if "App" in config_app:
            defaults["platform"] = config_app["App"].get("platform", defaults["platform"])
    else:
        print(f"Warning: Application config file '{app_config_file_name}' not found. Using hardcoded defaults.")

    # Read database settings
    if os.path.exists(db_config_file_name):
        config_db.read(db_config_file_name)
        print(f"Reading database credentials from: {db_config_file_name}")

        if "Database" in config_db:
            defaults["db_host"] = config_db["Database"].get("db_host", defaults["db_host"])
            defaults["db_user"] = config_db["Database"].get("db_user", defaults["db_user"])
            defaults["db_password"] = config_db["Database"].get("db_password", defaults["db_password"])
            defaults["db_name"] = config_db["Database"].get("db_name", defaults["db_name"])
    else:
        print(f"Warning: Database config file '{db_config_file_name}' not found. Database operations may fail.")

    parser.add_argument("--folder", type=str, default=defaults["folder"], help="Path to the folder containing images")
    parser.add_argument("--kpi_log", type=str, default=defaults["kpi_log"],
                        help="Path to the log file for performance metrics")
    parser.add_argument("--platform", type=str, default=defaults["platform"], help="Platform string from config")
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

    text_image_generation_local(album, args.kpi_log, db_config, args.platform, defaults["chat_query"])
