import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it", device_map="cpu")
print("Model loaded. Inspecting text tower layers...")

# Usually text layers are under `model.text_model` or just `model.model` if the vision/audio are separate.
# In Gemma multimodal, text model is usually `model.text_model` or `text_model.model`
for name, module in model.named_modules():
    if "q_proj" in name and ("text" in name or "language" in name or "model.layers" in name):
        print(f"Text Layer: {name} -> {type(module).__name__}")
        break

for name, module in model.named_modules():
    if "q_proj" in name and "model.layers" in name:
        print(f"Fallback Layer: {name} -> {type(module).__name__}")
        break
