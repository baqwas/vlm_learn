# 🤖 VLM Learn (Vision Language Models)

This repository serves as a centralized laboratory for **Vision-Language Model (VLM)** experimentation, focusing on **Qwen-VL** and **CLIP** architectures. It also integrates a high-fidelity **NOAA Satellite Infrastructure** for real-time geophysical data processing.

## 📂 Project Structure

### 👁️ Vision Tasks

Scripts focused on multimodal understanding, zero-shot classification, and visual reasoning.

* **`clip_exercise/`**: Implementation of OpenAI's Contrastive Language-Image Pretraining for image-text alignment.
* **`qwen_v2d5/`**: Utilities for the Qwen-VL-Chat model, including batch inference and conversation pipelines.
* **`qwen_v3/`**: Next-generation reasoning scripts featuring situational awareness and agentic planning.
* **`projects/labeler.py`**: A custom utility for managing image annotation and ground truth verification.

### 🧠 General AI & Utilities

Core scripts for model management and system auditing.

* **`genai/`**: Streamlit-based interfaces for interacting with Generative AI models.
* **`projects/gantt_chart.py`**: Project management utility for tracking AI development timelines.

---

## 🛠️ Setup & Security

### Environment

This project requires **Python 3.11+**. Core dependencies include:

* `transformers` & `accelerate` (Hugging Face)
* `torch` (PyTorch)
* `tomllib` (Config parsing)

### Git Governance

To maintain a lean repository, the following assets are strictly excluded via `.gitignore`:

* **Heavy Binaries**: `*.mp4`, `*.jpg`, `*.ttf` (kept local-only).
* **Model Weights**: `*.pth`, `*.bin`, `*.safetensors`.
* **Local Configs**: `config.toml` (contains sensitive SMTP/MQTT credentials).

## 🚀 Getting Started

To set up your local environment for Qwen-VL and CLIP development, follow these steps:
1. Prerequisites

Ensure you have Python 3.11+ and a virtual environment activated.
2. Core Installation

Install the primary VLM stack using the following commands:
Bash

# Update pip to the latest version
pip install --upgrade pip

# Install the multimodal processing stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes sentencepiece
pip install pillow requests tqdm

3. Model-Specific Libraries

For Qwen-VL specific dynamic resolution and efficient inference, install these additional dependencies:
Bash

# For image processing and web-based demos
pip install opencv-python matplotlib streamlit

# For Qwen-VL specific requirements
pip install einops transformers_stream_generator

4. Verification

Run the following snippet to verify your installation:
Python

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"CUDA Available: {torch.cuda.is_available()}")
# Qwen-VL models will be loaded here using 'AutoModelForCausalLM'

### 🧹 Maintenance Note

As specified in the .gitignore:

    Do not run pip freeze > requirements.txt if your environment contains local paths.

    Always ensure *.bin and *.safetensors are not tracked before committing new scripts.

---

## ⚖️ License

This project is licensed under the **MIT License** - Copyright (c) 2026 **ParkCircus Productions**.

---
