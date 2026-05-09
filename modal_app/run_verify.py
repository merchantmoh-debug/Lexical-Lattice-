import modal
from modal_app.app import image, VOL_MOUNT, vol, HF_SECRET

app = modal.App("paper7-verify-grad")

@app.function(image=image, volumes={VOL_MOUNT: vol}, secrets=[HF_SECRET], gpu="A10G", timeout=60*20)
def check_gradients():
    import os
    os.environ["HF_HOME"] = f"{VOL_MOUNT}/hf_cache"
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    import transformers

    print("1. Skipping Monkey Patch, relying on explicit Regex target modules...")

    model_name = "google/gemma-4-E4B-it"
    print(f"2. Loading base model {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    
    print("3. Applying new PEFT Configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*language_model.*(?:q_proj|v_proj).*"
    )

    model = get_peft_model(model, lora_config)
    model.train()
    
    # Required for gradient checkpointing with PEFT
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print("\n--- PEFT Trainable Parameters ---")
    model.print_trainable_parameters()

    print("\n4. Running Forward and Backward Pass on dummy data...")
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)

    # Forward pass
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    print(f"Loss: {loss.item()}")

    print("Running backward()...")
    loss.backward()

    print("5. Checking Gradients on Reasoning Engine...")
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

@app.local_entrypoint()
def run():
    check_gradients.remote()
