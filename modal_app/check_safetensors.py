import modal
from modal_app.app import image, VOL_MOUNT, vol

app = modal.App("paper7-check-tensors")

@app.function(image=image, volumes={VOL_MOUNT: vol})
def check():
    from safetensors.torch import load_file
    tensors = load_file(f"{VOL_MOUNT}/checkpoints/lex_intervention_default/adapter_model.safetensors")
    for key in list(tensors.keys())[:10]:
        print(key)

@app.local_entrypoint()
def run():
    check.remote()
