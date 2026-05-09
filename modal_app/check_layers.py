import modal
from modal_app.app import image, VOL_MOUNT, vol

app = modal.App("paper7-check-layers-all")

@app.function(image=image, volumes={VOL_MOUNT: vol})
def check():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it", device_map="cpu")
    for name, module in model.named_modules():
        if "q_proj" in name:
            print(f"{name}: {type(module).__name__}")

@app.local_entrypoint()
def run():
    check.remote()
