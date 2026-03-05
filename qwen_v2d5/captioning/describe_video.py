#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
describe_video.py
This script demonstrates how to use the Qwen2.5-VL model to generate a description of a video.
It processes a video file, extracts frames, and generates a textual description based on the content of the video.

Notes:
The key differences between processing an image and a video are:
    Dependencies: The package decord for efficient video loading
    Input Format: The messages list must specify "type": "video"
    Processor Call: The processor needs to receive the video data via a videos keyword argument
    Processor Parameters: video_fps to control how many frames per second the model processes

    python -m pip install decord qwen-vl-utils[decord]
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import requests

# 1. Define the model ID
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# 2. Load the model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 3. Define the path to your video file
# video_file_path = "/home/reza/Videos/yolo/yolo11/RPi5/videos/signaltest.mp4"
video_file_path = "/home/reza/Videos/yolo/yolo11/RPi5/videos/hurricane5.mp4"

# 4. Construct the multimodal prompt with a video placeholder
# The 'content' list now specifies a video file path.
# query = "What is happening in this video."
query = "Describe the main events happening in this video."
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_file_path,  # Use a local file path or URL
                "fps": 1.0,  # Process 1 frame per second
            },
            {"type": "text", "text": query}
        ]
    }
]

# 5. Tokenize the inputs in a two-step process
# Step 1: Apply the chat template to get the formatted text string
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Step 2: Process the vision information from the messages
# This utility function separates text, images, and videos
# The result is a tuple of (images_list, videos_list)
image_inputs, video_inputs = process_vision_info(messages)

# Step 3: Pass the text string and the video object to the processor
inputs = processor(
    text=[text],              # Pass the formatted text string
    images=image_inputs,      # Pass the image list (will be empty for video)
    videos=video_inputs,      # Pass the video list
    return_tensors="pt",
).to(model.device)

# 6. Generate the response
generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
)

# 7. Decode the response
decoded_output = processor.batch_decode(
    generated_ids[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)

# Print the result
print("Generated Description:")
print(decoded_output[0])
