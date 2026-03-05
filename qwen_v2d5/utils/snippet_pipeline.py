#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A simple example of using the Hugging Face Transformers pipeline for image-text-to-text tasks.
This script demonstrates how to use the Qwen2.5-VL-7B-Instruct model to process an image and a text prompt,
and generate a text response based on the image content.

Prerequisites:
    transformers library installed
    the qwen-vl-utils library installed
    the torch library installed
"""
# 1 Import the packages
import torch
from transformers import pipeline, ImageTextToTextPipeline

"""
2 Load the pipeline for image-text-to-text task
Create an instance of the pipeline class,
Specifying:
    The task to be performed and
    The model to be used
This automatically loads the model and its associated processor and tokenizer.
"""
pipe: ImageTextToTextPipeline = pipeline(
    task="image-text-to-text",
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    device=0,
    torch_dtype=torch.bfloat16,
)

"""
3 Prepare the Multimodal Input
For Qwen-VL, which handles both text and images, the input needs to be structured as a list of dictionaries.
Each dictionary represents a turn in the conversation and contains the user's text and image(s).
"""
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://bezaman.parkcircus.org/examples/demo.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

"""
4 Run Inference and obtain the output
The `pipe` function processes the input messages and generates a response.
The `max_new_tokens` parameter controls the maximum number of tokens to generate in the response.
The pipeline handles all the processing, and the output will be a dictionary containing the generated text response.
"""
completion_code = pipe(text=messages, max_new_tokens=20, return_full_text=False)
print(completion_code[0]["generated_text"])
