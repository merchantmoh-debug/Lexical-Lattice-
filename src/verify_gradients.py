import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import transformers

print("1. Injecting Gemma4ClippableLinear Monkey Patch...")
if hasattr(transformers.models, 'gemma4'):
    transformers.models.gemma4.modeling_gemma4.Gemma4ClippableLinear.__bases__ = (torch.nn.Linear,)

model_name = "google/gemma-4-E4B-it"
print(f"2. Loading base model {model_name} on CPU...")

# Load standard model (no value head needed to test basic PEFT backprop)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.bfloat16)

# Force gradient checkpointing and requires_grad to enable backprop on CPU
model.gradient_checkpointing_enable()

print("3. Applying new PEFT Configuration...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[r"model\.layers\..*q_proj", r"model\.layers\..*v_proj"]
)

model = get_peft_model(model, lora_config)
model.train()

print("\n--- PEFT Trainable Parameters ---")
model.print_trainable_parameters()

print("\n4. Running Forward and Backward Pass on dummy data...")
# Dummy query
inputs = tokenizer("The capital of France is", return_tensors="pt")

# Forward pass
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss

print(f"Loss: {loss.item()}")

# Backward pass
print("Running backward()...")
loss.backward()

print("5. Checking Gradients on Reasoning Engine...")

# Check lora_B gradients
grad_found = False
grad_zero = False
for name, param in model.named_parameters():
    if "lora_B" in name and param.grad is not None:
        grad_found = True
        mean_grad = param.grad.abs().mean().item()
        print(f"{name} -> grad abs mean: {mean_grad:.8f}")
        if mean_grad == 0.0:
            grad_zero = True

if grad_found and not grad_zero:
    print("\n[SUCCESS] GRADIENT VERIFICATION PASSED. The Reasoning Engine is receiving gradients.")
else:
    print("\n[FAILURE] Gradients are missing or zero. The Lexicographic Governor is not connected.")
