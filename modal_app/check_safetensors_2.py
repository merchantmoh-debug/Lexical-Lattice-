import modal
from modal_app.app import image, VOL_MOUNT, vol

app = modal.App("paper7-check-tensors-2")

@app.function(image=image, volumes={VOL_MOUNT: vol})
def check():
    from safetensors.torch import load_file
    import torch
    tensors = load_file(f"{VOL_MOUNT}/checkpoints/lex_intervention_default/adapter_model.safetensors")
    
    # Check if lora_B is all zeros
    b_weights = [t for k, t in tensors.items() if "lora_B" in k]
    if not b_weights:
        print("No lora_B weights found!")
    else:
        for i, w in enumerate(b_weights[:5]):
            print(f"lora_B {i} mean abs: {w.abs().mean().item():.8f}, sum: {w.sum().item():.8f}")
            
    # Check lora_A
    a_weights = [t for k, t in tensors.items() if "lora_A" in k]
    if a_weights:
        for i, w in enumerate(a_weights[:5]):
            print(f"lora_A {i} mean abs: {w.abs().mean().item():.8f}")

@app.local_entrypoint()
def run():
    check.remote()
