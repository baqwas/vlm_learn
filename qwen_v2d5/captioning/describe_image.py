#!/usr/bin/env python3
# -- coding: utf-8 --
"""
describe_image.py
This script generates a description for a given image using a pre-trained model.
It uses the Hugging Face Transformers library to load a model and tokenizer,
and generates a caption based on the image input.
Usage:
    python describe_image.py --image_path <path_to_image> --model_name <model_name>

Example:
    python describe_image.py --image_path ./images/sample.jpg --model_name openai/clip-vit-base-patch32

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
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests

# 1. Define the model ID
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# 2. Load the model and processor with the CORRECT class name
# The Qwen-VL model requires this specific class, not AutoModelForCausalLM.
"""
Use torch_dtype="auto" for faster inference on GPU if available.
This is the key line that makes the script portable. 
Hugging Face's accelerate library (which is used under the hood by transformers when device_map is set) 
intelligently distributes the model's layers across all available devices.
The device_map="auto" setting will automatically place the model on the first available GPU if one is present.
If a GPU is present, it will use that first.
If there is no GPU, device_map="auto" will automatically default to loading the model onto the CPU.
If the model's weights and activations exceed the available system RAM, 
accelerate can even offload parts of the model to the hard drive, 
though this will make the already slow inference even slower.
For practical, usable performance, 
a dedicated GPU with at least 8GB of VRAM (for the 3B model) is the minimum recommended hardware.
"""
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 3. Prepare the image (using a URL or local image)
# image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
# local_image_path = "/home/reza/Videos/opencv/vlm/qwen2.5-vl/bridge.jpg"
# local_image_path = "../images/dhabba.jpg"
# local_image_path = "/home/reza/Videos/opencv/vlm/qwen2.5-vl/images/traffic.png"
# local_image_path = "../images/winter_uk.jpg"
# local_image_path = "../images/multilingual.png"
# local_image_path = "../images/qwen_arc.jpg"
# local_image_path = "../images/balloons.jpg"
local_image_path = ("../images/hpi_annuals.png"
                    "")
image = Image.open(local_image_path).convert("RGB")

# 4. Construct the multimodal prompt with a placeholder for the image and a text query
# query = "Driving from the location shown on the image at the posted speed just after 4 PM, will I reach Richmond, VA before sunset on July 4?"
# query = "Summarize the contents of the image."
# query = "What was the tax rate applied to the receipt?"
# query = "Count and rank the objects within the frame in the image by color?"
# query = "Outline the blue objects in this image and count each of them?"
# query = "Which river is under the bridge?"
# query = "What is the location of the image? Is it a popular place?"
# query = "Convert the image to machine readable text as a set of equations. Use LaTeX format for the equations."
# query = "Identify the languages in the image and translate the text to German."
# query = "Explain the diagram in the image."
# query = "How does data flow in the diagram? What are the key components and their interactions?"
# query = "Based on the first image, which brand of coffee or tea should I purchase from the second image?"
query = "Is the company growing or shrinking during the reporting years in the chart?"


# The messages list should match the expected input format for the Qwen-VL model.
"""
{"type": "image_url", "image_url": {"url": image_url}},
{"type": "text", "text": query}
"""
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": query}
        ]
    }
]

# 5. Tokenize the inputs, passing both the messages and the image object
# The processor matches the "type": "image" placeholder with the image provided
# Step 1: Apply the chat template to get the formatted text string
text = processor.apply_chat_template(
    messages,
    tokenize=False,  # Important: get the string, not the tokens
    add_generation_prompt=True,
)
# Step 2: Pass the text string and the image object to the processor
inputs = processor(
    text=text,            # Pass the formatted text string
    images=image,         # Pass the PIL Image object here
    return_tensors="pt",
).to(model.device)

# Move inputs to the appropriate device (e.g., GPU)
# inputs = input_ids.to(model.device)

# 6. Generate the response
generated_ids = model.generate(
    # inputs=inputs,
    **inputs,
    max_new_tokens=512,
)

# 7. Decode the response
decoded_output = processor.batch_decode(
    # generated_ids[:, inputs.shape[1]:],
    generated_ids[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)

# Print the result
print("Generated Description:")
print(decoded_output[0])