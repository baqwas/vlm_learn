#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hello_qwen.py
# This script demonstrates how to use the Qwen-VL model to generate a caption for an image.
# It processes an image URL, generates a caption with grounding boxes, and saves the output image with bounding boxes.
# # Notes:
# The key differences between processing an image and a video are:
#     - Dependencies: The package qwen-vl-utils for image processing
#     - Input Format: The messages list must specify "type": "image"
#     - Processor Call: The processor needs to receive the image data via an images keyword argument
#     - Processor Parameters: No need for video-specific parameters like fps
#     - Output: The model generates a caption with grounding boxes, which can be drawn on the image
#     - Image Saving: The output image with bounding boxes is saved to a file
#     - python -m pip install qwen-vl-utils transformers accelerate matplotlib tiktoken einops transformers_stream_generator
#     - python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# https://huggingface.co/Qwen/Qwen-VL
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True
).eval()
# use cuda device
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
"""
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'text': 'Generate the caption in English with grounding:'},
])
"""

image_path = "../images/winter_uk.jpg"

query = tokenizer.from_list_format(
    [
        {"image": image_path},
        {"text": "Generate the caption in English with grounding:"},
    ]
)

inputs = tokenizer(query, return_tensors="pt")
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
print(response)
# <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
image = tokenizer.draw_bbox_on_latest_picture(response)
if image:
    image.save("2.jpg")
else:
    print("no box")
