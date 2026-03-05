#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An example of batch inference using Qwen3-30B-A3B with mixed media inputs.

Notes:
    The model can batch inputs composed of mixed samples of various types such as images, videos, and text.
    This example is from GitHub/huggingface/transformers

References:
    https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_vl.md

"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Define the conversations with mixed media inputs
# Conversation for the first image
conversation1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "../images/4sa.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# Conversation with two images
conversation2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "../images/latex.png"},
            {"type": "image", "path": "../images/ocrreceipt.png"},
            {"type": "text", "text": "What is written in the pictures?"}
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {
        "role": "user",
        "content": "who are you?"
    }
]


# Conversation with mixed midia
conversation4 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "../images/eyeover.jpg"},
            {"type": "image", "path": "../images/Big Ben and Transportation.jpg"},
            {"type": "video", "path": "../videos/egll-27r.mp4"},
            {"type": "text", "text": "What are the common elements in these medias?"},
        ],
    }
]

conversations = [conversation1] #, conversation2, conversation3, conversation4] # <==
# Preparation for batch inference
inputs = processor.apply_chat_template(
    conversations,
    video_fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# Batch Inference
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
