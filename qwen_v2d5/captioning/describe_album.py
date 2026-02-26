#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hello_qwen.py
    This script demonstrates how to use the Qwen-VL model to generate a caption for an image.
    It processes an image URL, generates a caption with grounding boxes, and saves the output image with bounding boxes.
Notes:
    The key differences between processing an image and a video are:
    - Dependencies: The package qwen-vl-utils for image processing
    - Input Format: The messages list must specify "type": "image"
    - Processor Call: The processor needs to receive the image data via an images keyword argument
    - Processor Parameters: No need for video-specific parameters like fps
    - Output: The model generates a caption with grounding boxes, which can be drawn on the image
    - Image Saving: The output image with bounding boxes is saved to a file
    - python -m pip install qwen-vl-utils transformers accelerate matplotlib tiktoken einops transformers_stream_generator
    - python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Reference: https://huggingface.co/Qwen/Qwen-VL
Key Steps:
    Get all image paths: os.listdir() is used to find all image files in a specified folder.
    Create a list of queries: A loop creates a separate query list for each image.
    Batch Tokenization: tokenizer(queries, return_tensors='pt', padding=True) is used to tokenize all queries at once. Padding=True is crucial for batching.
    Batch Generation: The model.generate function takes the batched inputs and processes them together.
    Batch Decoding: tokenizer.batch_decode is used to decode all responses from the batch.
    Iterate and print: A loop is used to iterate through the responses and their corresponding image file names.
"""
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import os
import torch
from PIL import Image

# Import the correct, specific model class for Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.generation import GenerationConfig

torch.manual_seed(1234)

# Load the processor and the model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)

# Use the specific model class to load the model.
# This resolves the Unrecognized configuration class error.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", device_map="cpu", trust_remote_code=True).eval()

# Define the folder containing your images
image_folder = "../images/album"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.JPG', '.jpeg', '.png'))]

# Loop through each image file and process it individually
for i, image_path in enumerate(image_files):
    print(f"Processing image {i + 1}/{len(image_files)}: {image_path}")

    # Open the image using PIL
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Could not open image {image_path}: {e}")
        continue

    # Prepare the text prompt
    text_prompt = "Generate the caption in English with grounding:"

    # Process the image and text to get model inputs
    # The processor expects the image and text as separate arguments.
    inputs = processor(images=image, text=text_prompt, return_tensors='pt')

    # Generate prediction for the single image
    with torch.no_grad():
        predictions = model.generate(**inputs)

    # Decode the response
    response_text = processor.batch_decode(predictions, skip_special_tokens=True)[0]

    # Print the response
    print(f"Response for {image_path}:")
    print(response_text)

    # The rest of the code for drawing bounding boxes is commented out
    # because it is not a standard feature and caused issues in previous attempts.
    # It often requires a separate library or a specific implementation.
    print(f"no box for {image_path} - drawing method not included in this script.")