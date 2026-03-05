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
This script demonstrates how to use the Qwen2.5-VL-7B-Instruct model for image and text processing.
It loads the model and processor, prepares input messages with an image and text,
and generates a response based on the input.

Prerequisites:
- Install the required libraries: transformers, torch
- Ensure you have access to the Qwen2.5-VL-7B-Instruct model on Hugging Face.

"""
# 1 import necessary libraries
# Ensure that the specific variant is loaded from the transformers library
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

"""
2 load the model and processor
This is the core step.
Use the from_pretrained method on each of the Auto classes to load the correct components from the Hugging Face Hub.
The trust_remote_code=True parameter is often necessary for custom model architectures like Qwen-VL.
You should also specify the torch_dtype and device_map to manage memory and device usage.
"""
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

"""
3 prepare input messages
"""
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to("cuda")

"""
4 Run Inference
This is where the model processes the input and
generates a response
that is presented in a human-readable format.
"""
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
