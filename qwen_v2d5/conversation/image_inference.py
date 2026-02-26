#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

References:
    https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_vl.md
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load the model in half-precision on the available device(s)
print("Loading...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto",
    torch_dtype = "auto",
    trust_remote_code = True
)
print("Processing...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

print("Preparing conversation...")
conversation = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }
]

print("Applying chat template...")
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Inference: Generation of the output
print("Generating ids...")
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
print("Batch decoding...")
output_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)
print(output_text)
