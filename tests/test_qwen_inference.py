import pytest
import torch
from transformers import AutoModelForCausalLM


@pytest.mark.timeout(300)  # 5-minute limit for model download/loading
def test_model_loading():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    # In CI, we use float32 and CPU because bfloat16/CUDA aren't available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device, torch_dtype=torch.float32
    )
    assert model is not None
