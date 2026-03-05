#!/usr/bin/env python3
"""
ecommerce_agent.py
"""
import torch
import os
import argparse
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# --- Configuration Constants ---
MODEL_ID: str = (
    "Qwen/Qwen3-VL-2B-Instruct"  # Using the publicly available Qwen3-VL-2B-Instruct model
)
# NOTE: device_map="auto" is recommended to utilize GPU if available.
# Set to "cpu" if you must run on CPU (requires substantial RAM).
DEVICE_MAP: str = "auto"

# --- SCENARIO SETUP ---
# For this demonstration, we assume the user has provided an image of a:
# 1. Product Type: Backpack
# 2. Color: Red
# 3. Material: Canvas
# 4. Key Features: Side zipper
DUMMY_IMAGE_PATH: str = "./red_canvas_backpack.jpg"

# The user's request contains constraints that conflict with the image attributes
USER_QUERY: str = (
    "I need to find this bag, but make sure it is **black** and the material should be **leather**."
)

# The instruction set is crafted to force the model to output the required thinking trace
SYSTEM_INSTRUCTIONS: str = (
    "You are an e-commerce search agent. Your task is to analyze the image and the user's request. "
    "You must output a detailed thinking trace in three steps (Perception, Reasoning, Action) before the Final Agent Response. "
    "The thinking process must demonstrate Constraint Fusion by prioritizing the user's explicit textual constraints over the visual attributes when a conflict exists."
)

# The Thinking Trace template guides the model to produce the structured output
THINKING_TRACE_TEMPLATE: str = f"""
***Thinking Trace***

**Step 1: Perception (Visual Grounding)**
- Analyze the image to extract core attributes.
- Product Category (Image): Backpack
- Color (Image): Red
- Material (Image): Canvas
- Key Features (Image): Side zipper

**Step 2: Reasoning (Constraint Integration & Conflict Fusion)**
- Analyze User Constraints:
    - Target Color: "black"
    - Target Material: "leather"
- Integrate and Resolve Conflicts:
    - Color Constraint: The user's request for 'black' overrides the image's color ('Red'). Final Color: **Black**.
    - Material Constraint: The user's request for 'leather' overrides the image's material ('Canvas'). Final Material: **Leather**.
- Fused Final Attributes (from image where no conflict, or from user constraint where conflict):
    - Product Category: Backpack
    - Color: Black
    - Material: Leather
    - Key Features: Side zipper

**Step 3: Action (Structured Query Generation)**
- Generate a precise JSON query for an external search engine tool using the Fused Final Attributes from Step 2.
- The output must be a valid JSON object wrapped in a code block.
```json
{{
  "search_tool": "e_commerce_product_search",
  "query_parameters": {{
    "product_category": "Backpack",
    "color": "Black",
    "material": "Leather",
    "key_features": ["Side zipper"]
  }}
}}
```
"""
