import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Load the model in half-precision on the available device(s)
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
    )
processor = AutoProcessor.from_pretrained(model_id,
                                          trust_remote_code=True,
                                          use_fast=False
                                          )
# Video https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_vl.md
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/home/reza/Videos/yolo/yolo11/RPi5/videos/hurricane.mp4"},
            {"type": "text", "text": "What happened in the video?"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    video_fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):]
                 for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)
print(output_text)
