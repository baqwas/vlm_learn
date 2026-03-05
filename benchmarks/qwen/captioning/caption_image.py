#!/usr/bin/env python
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
caption_image.py
This script generates a description for a given image using a pre-trained model.
It uses the Hugging Face Transformers library to load a model and tokenizer,
and generates a caption based on the image input.

It processes two command-line parameters:
1. --image_path: Path to the local image file to be captioned.
2. --model_name: The name of the Hugging Face model to use for captioning.

It uses an external file named caption_image.ini in the same folder. If the file
Usage:
    python caption_image.py
    python caption_image.py --config_file my_settings.ini
    python caption_image.py --image_path ./photos/example.jpg --model_name my/new-model
    python caption_image.py --config_file specific_config.ini --image_path ./photos/example.jpg --model_name my/new-model

Example `config.ini` (or any custom config file) file:
    [Settings]
    image_path = ../images/sample.jpg
    model_name = Qwen/Qwen2.5-VL-3B-Instruct

Dependencies:
    python -m pip install torch transformers Pillow argparse accelerate torchvision torchaudio
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    python -m pip install transformers Pillow argparse
Notes:
    This script demonstrates the key steps:
        Loading the Model and Processor:
            Using AutoModelForCausalLM and AutoProcessor from the transformers library.
        Preparing the Image:
            Loading an image using the Pillow library (PIL).
            The script uses a sample URL, but you can change this to a local file path.
        Constructing the Multimodal Prompt:
            Building the messages list with the correct role and content structure
            for the Qwen-VL model, including both the text query and the image reference.
        Tokenizing the Inputs:
            Using processor.apply_chat_template to convert the messages list
            into the tokenized format the model expects.
        Generating the Response:
            Calling model.generate() to produce the output token IDs.
        Decoding the Output:
            Slicing the output to remove the input prompt and decoding the remaining tokens
            into a readable string.

    Trust_remote_code=True:
        This parameter is necessary to load the custom code from the Qwen models on Hugging Face.
        It's a security risk to be aware of, but it's standard practice for these models.
    RAM Requirements:
        The Qwen2.5-VL-3B-Instruct model is a 3-billion-parameter model.
        The script loads the model with torch_dtype="auto". While this is great for a GPU, on a CPU,
        it will likely default to the model's native float32 precision unless a quantized version (like 8-bit or 4-bit)
        is explicitly loaded.
        A 3B parameter model in float32 requires approximately 3B * 4 bytes/parameter = 12 GB of memory
        just for the model weights. You also need RAM for the Python interpreter, the operating system, and
        the intermediate tensors and activations generated during inference.
        Therefore, a computer would need at least 16GB of RAM to have a chance of running this script
        without severe memory issues, and 32GB or more would be highly recommended for stable and faster operation.
    Performance:
    Inference Speed:
        Running this model on a CPU will be orders of magnitude slower than on a GPU.
        The difference can be seconds on a high-end GPU versus minutes or even longer on a CPU
        for a single image description, especially because the model has to process both the image and the text.
        CPU Utilization:
        The script will be computationally intensive, likely maxing out one or more CPU cores for a prolonged period.
        The speed will depend heavily on the number of cores and the clock speed of your CPU.

    caption_image.ini example
    [Settings]
    image_path = ../images/flower.jpg
    model_name = Qwen/Qwen2.5-VL-7B-Instruct
    display_input=0
    display_output=0
    print_ids=0
    print_caption=0

The original file by Dr. Satya Mallick resides at https://colab.research.google.com/drive/1eSmOZTyF2P_U2xcCl2gQJZfu6kaHFPCX
"""
# !pip install qwen-vl-utils  # Upgrade Qwen-VL utilities during Jupyter/Colab session
# 2  Imports
# ── Standard library ────────────────────────────────────────────
import os  # File‑system helpers (paths, env vars, etc.)
import random  # Lightweight randomness (e.g., sample prompts)
import textwrap  # Nicely format long strings for display
import io  # In‑memory byte streams (e.g., image buffers)
import requests  # Simple HTTP requests for downloading assets
import argparse
import configparser
import time

# ── Numerical computing ─────────────────────────────────────────
import numpy as np  # Core array maths (fast, vectorized operations)

# ── Deep‑learning stack ─────────────────────────────────────────
import torch  # Tensor library + GPU acceleration
from transformers import (
    Qwen2_5_VLForConditionalGeneration,  # Multimodal LLM (image+text)
    AutoProcessor,  # Paired tokenizer/feature‑extractor
)

# ── Imaging & visualisation ─────────────────────────────────────
from PIL import Image  # Pillow: load/save/manipulate images
import matplotlib.pyplot as plt  # Quick plots in notebooks
import matplotlib.patches as patches  # Bounding‑box overlays, etc.

# ── Project‑specific helpers ────────────────────────────────────
from qwen_vl_utils import process_vision_info  # Post‑process Qwen outputs

# ── Notebook conveniences ──────────────────────────────────────
# import IPython.display as ipd         # Inline display (images, audio, HTML)

APP_NAME = "caption_image"


# helper function to process command line and configuration file
def read_config_defaults(config_file_name):
    """
    Reads default values for command-line arguments from a configuration file.

    Args:
        config_file_name (str): The name/path of the configuration file to read.

    Returns:
        dict: A dictionary containing default values for 'image_path' and 'model_name'.
              If the file or keys are not found, it returns None or a hardcoded default.
    """
    config = configparser.ConfigParser()
    defaults = {
        "image_path": None,  # No default, so it's required if not in config or CLI
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",  # Hardcoded default
        "display_input": False,
        "display_output": False,
        "print_ids": False,
        "print_caption": False,
        "print_prompt": False,
        "measure_performance": False,
        "message_prompt": "Describe this image.",
    }

    if os.path.exists(config_file_name):
        config.read(config_file_name)
        print(f"Reading defaults from configuration file: {config_file_name}")
        if "Settings" in config:
            if "image_path" in config["Settings"]:
                defaults["image_path"] = config["Settings"]["image_path"]
            if "model_name" in config["Settings"]:
                defaults["model_name"] = config["Settings"]["model_name"]

            # Read boolean values safely
            for key in [
                "display_input",
                "display_output",
                "print_ids",
                "print_caption",
                "print_prompt",
                "measure_performance",
            ]:
                try:
                    if key in config["Settings"]:
                        defaults[key] = config.getboolean("Settings", key)
                except ValueError as e:
                    print(
                        f"Warning: Invalid boolean value for '{key}' in config file. Using hardcoded default. Error: {e}"
                    )

        else:
            print(
                f"Warning: No '[Settings]' section found in '{config_file_name}'. Using hardcoded defaults."
            )
    else:
        print(
            f"Warning: Configuration file '{config_file_name}' not found. Using hardcoded defaults."
        )

    return defaults


def caption_image():
    """
    1. Create the parser to handle command-line arguments
        Main function to parse arguments and run the image captioning process.
    """
    parser = argparse.ArgumentParser(
        description="Generate a caption for an image using a specified model."
    )

    """
    2. Add the command-line arguments
        Define the argument for the config file, as its value affects other defaults
    """
    parser.add_argument(
        "--config_file",
        type=str,
        default=f"{APP_NAME}.ini",
        help="Optional path to a configuration file to load default settings from "
        f"(default: '{APP_NAME}.ini').",
    )

    # Parse arguments specifically for the config file first
    # This allows us to use the specified config file to load other defaults
    config_args, unknown = parser.parse_known_args()

    """
    3. Read default values from the specified config file
    """
    config_defaults = read_config_defaults(config_args.config_file)
    """
    Now add the other arguments, using config values as defaults.
    We re-create the parser or add to it, ensuring that previously unknown args are parsed.
    For simplicity, and to ensure proper default overriding, we can rebuild the parser
    or ensure all arguments are added before the final parse_args().
    A common pattern is to parse known args first to get the config file, then re-parse all.

    Re-initialize parser to correctly handle all defaults and overrides
    """
    parser = argparse.ArgumentParser(
        description="Generate a caption for an image using a specified model."
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=f"{APP_NAME}.ini",
        help="Optional path to a configuration file to load default settings from "
        f"(default: '{APP_NAME}.ini').",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default=config_defaults["image_path"],
        help="Path to the input image file (e.g., '../images/sample.jpg')."
        f" Defaults to value in {APP_NAME}.ini or is required if not present.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=config_defaults["model_name"],
        help="Name of the Hugging Face model to use. "
        "Defaults to value in {APP_NAME}.ini or a hardcoded default.",
    )

    # to display or not to display that is the question
    # A boolean type can be used here for explicit true/false values
    parser.add_argument(
        "--display_input",
        type=lambda x: (x.lower() in ["true", "1", "t", "y", "yes"]),
        default=config_defaults["display_input"],
        help="Display the input image using a graphical window (default: True).",
    )
    parser.add_argument(
        "--display_output",
        type=lambda x: (x.lower() in ["true", "1", "t", "y", "yes"]),
        default=config_defaults["display_output"],
        help="Display the output image using a graphical window (default: True).",
    )
    parser.add_argument(
        "--print_ids",
        type=lambda x: (x.lower() in ["true", "1", "t", "y", "yes"]),
        default=config_defaults["print_ids"],
        help="Print the ids in the console window (default: True).",
    )
    parser.add_argument(
        "--print_caption",
        type=lambda x: (x.lower() in ["true", "1", "t", "y", "yes"]),
        default=config_defaults["print_caption"],
        help="Print the caption in the console window (default: True).",
    )
    parser.add_argument(
        "--print_prompt",
        type=lambda x: (x.lower() in ["true", "1", "t", "y", "yes"]),
        default=config_defaults["print_prompt"],
        help="Print the prompt message in the console window (default: True).",
    )

    parser.add_argument(
        "--measure_performance",
        type=lambda x: (x.lower() in ["true", "1", "t", "y", "yes"]),
        default=config_defaults["measure_performance"],
        help="Measure and print the inference speed of the model (default: False).",
    )

    parser.add_argument(
        "--message_prompt",
        type=str,
        default=config_defaults["message_prompt"],
        help="Text in message template to processor. "
        f"Defaults to value in {APP_NAME}.ini or a hardcoded default.",
    )

    # 4. Parse all arguments again, this time including potential overrides
    args = parser.parse_args()

    # Get the final values. Command-line args will override config defaults.
    image_path = args.image_path
    model_name = args.model_name
    display_input = args.display_input
    display_output = args.display_output
    print_ids = args.print_ids
    print_caption = args.print_caption
    print_prompt = args.print_prompt
    measure_performance = args.measure_performance
    message_prompt = args.message_prompt

    # 5. Check if we have a valid image path from either source
    if not image_path:
        print(
            "Error: No image path provided. Please specify --image_path "
            "on the command line or in your configuration file."
        )
        parser.print_help()  # Print help if an image path is missing and required
        return
    if not os.path.exists(image_path):
        print(f"Error: The image file '{image_path}' was not found.")
        return

    print(f"Loading model: {model_name}")
    print(f"Processing image: {image_path}")

    """
    6.  Device & model load
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",  # automatically uses FP16 on GPU, FP32 on CPU
        device_map="auto",  # dispatches layers to the available device(s)
    )
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Model loaded on: {model.device}")

    """
    7. Read an image
    """
    image_specified = Image.open(image_path).convert("RGB")

    # Display the image
    if display_input:
        plt.imshow(image_specified)
        plt.axis("off")
        plt.title(f"Input: {image_path}")
        plt.show()

    """
    8.  Build a single-turn multimodal chat-style prompt
    8.1 Create a message in JSON format
    Qwen VL uses the same multi-turn message format as Qwen-2.5-Chat:
    The processor:
        turns these messages into plain text (with special tokens) and
        extracts the visual tensors
    so the model receives *both* modalities.
    """
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_specified},
                {"type": "text", "text": f"{message_prompt}"},
            ],
        }
    ]
    """
    8.2 Apply Single-Turn Multimodal Prompt Chat Template
    Every chat model ships with a pattern.
    The above JSON format needs to be converted to this template and
    this is done using `apply_chat_template`.
    It performs three tasks.
    1. **Reads the model’s chat template**:
        Qwen 2.5VL has a chat template that looks like `<|im_start|>{role}\n{content}<|im_end|>`.
    2. **Fills in the template with the msgs list**.
        It loops over each message, swaps in the role (user, assistant, etc.) and
        the content (text plus special <image> markers), and concatenates the result into one long string.
    3. **Adds the “assistant starts talking now” marker** (add_generation_prompt=True).
        At the end it appends `<|im_start|>assistant`
    The function will NOT tokenize the text because
    we are going to use the processor to process text and images together, and
    tokenization will be performed in that step.
    """

    # Build the full textual prompt that Qwen-VL expects
    # --------------------------------------------------
    # • msgs : list of message dicts (roles + content, including <image> markers)
    # • tokenize=False : return a plain string—not token IDs—so we can
    #                    combine it with image tensors in the next processor() call
    # • add_generation_prompt=True : appends the “assistant is about to speak” marker
    #                                (e.g. "<|im_start|>assistant\n"), which tells the
    #                                model where its reply should begin.
    text_prompt = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

    # For sanity-checking: print the raw prompt string that will be fed to the model
    if print_prompt:
        print(text_prompt)

    """
    8.3 Extract image & video inputs from `msg`

    In the previous step we converted our message `msg` from JSON format to the chat template the model understands.

    Now, we need to extract images and videos from `msg` using `process_vision_info` utility. It performs the following tasks
    1. Walks through every message,
    2. Finds all `"image"` / `"video"` entries,
    3. Applies the Qwen-VL visual pre-processing to ensure each image is a PIL.Image (or each video is a list / tensor of frames).
    4. Returns two parallel lists/tensors
     - `image_inputs`  → batched image tensors (or [] if none)
     - `video_inputs`  → batched video tensors (or [] if none)
    """

    # Extract vision-modalities from msgs and convert them to model-ready tensors
    # --------------------------------------------------------------------------
    # • msgs: the same chat-style list you fed to apply_chat_template.
    #          Each dict can include items like {"type": "image", "image": img}
    #          or {"type": "video", "video": video_clip}.
    # • process_vision_info : project utility that
    #       1) walks through every message,
    #       2) finds all `"image"` / `"video"` entries,
    #       3) applies the Qwen-VL visual pre-processing to ensure each
    #          image is a PIL.Image (or each video is a list / tensor of frames).
    #       4) returns two parallel lists/tensors:
    #            – `image_inputs`  → batched image tensors (or [] if none)
    #            – `video_inputs`  → batched video tensors (or [] if none)
    #   These outputs plug straight into the `processor(...)` call that follows,
    #   ensuring the vision data is aligned with the text prompt.
    image_inputs, video_inputs = process_vision_info(msgs)

    """
    9  Generate the caption
    9.1 Run inference
    """

    # ── Pack text + vision into model-ready tensors ──────────────────────────────
    inputs = processor(
        text=[text_prompt],  # 1-element batch containing the chat prompt string
        images=image_inputs,  # list of raw PIL images (pre-processed inside processor)
        videos=video_inputs,  # list of raw video clips (if any)
        padding=True,  # pad sequences so text/vision tokens line up in a batch
        return_tensors="pt",  # return a dict of PyTorch tensors (input_ids, pixel_values, …)
    ).to(
        model.device
    )  # move every tensor—text and vision—to the model’s GPU/CPU

    # ── Run inference (no gradients, pure generation) ───────────────────────────
    start_time = time.time()
    with torch.no_grad():  # disable autograd to save memory
        generated_ids = model.generate(  # autoregressive decoding
            **inputs,  # unpack dict into generate(...)
            max_new_tokens=254,  # cap the response at 64 tokens
        )
    end_time = time.time()
    generation_time = end_time - start_time
    num_generated_tokens = generated_ids.shape[1] - inputs.input_ids.shape[1]
    if print_ids:
        print(inputs.input_ids[0])
        print(generated_ids)

    """
    9.2 Decode output"""

    # Extract the newly generated tokens (skip the prompt length)
    caption = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True
    )[0]

    # Display the image
    if display_output:
        plt.imshow(image_specified)
        plt.axis("off")
        plt.show()

    # Print caption
    if print_caption:
        width = 80
        wrapped_caption = textwrap.fill(caption, width)
        print(wrapped_caption)

    # 10. Print Performance Metrics if enabled
    if measure_performance:
        print("\n--- Performance Metrics ---")
        print(f"Generated tokens: {num_generated_tokens}")
        print(f"Generation time: {generation_time:.2f} seconds")
        if generation_time > 0:
            tokens_per_second = num_generated_tokens / generation_time
            print(f"Tokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    caption_image()
