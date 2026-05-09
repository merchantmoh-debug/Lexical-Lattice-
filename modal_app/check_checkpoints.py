import modal
from modal_app.app import image, VOL_MOUNT, vol

app = modal.App("paper7-check")

@app.function(image=image, volumes={VOL_MOUNT: vol})
def check():
    import os
    print("Checking /vol/checkpoints...")
    os.system("ls -la /vol/checkpoints")
    os.system("ls -la /vol/checkpoints/scalar_baseline_default")
    os.system("ls -la /vol/checkpoints/lex_intervention_default")

@app.local_entrypoint()
def run():
    check.remote()
