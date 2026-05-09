from __future__ import annotations
import os
import modal

APP_NAME = "paper7-sandbox"
VOLUME_NAME = "lex-rlhf-data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "trl==0.8.1",
        "peft>=0.12.0",
        "datasets>=2.18.0",
        "accelerate>=0.29.0",
        "bitsandbytes>=0.43.1",
        "numpy>=1.26.0"
    )
    .add_local_python_source("src", "eval", "config", "modal_app")
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

VOL_MOUNT = "/vol"
HF_CACHE = f"{VOL_MOUNT}/hf_cache"
HF_SECRET = modal.Secret.from_name("huggingface-secret")

@app.function(
    image=image,
    gpu="A100",
    volumes={VOL_MOUNT: vol},
    timeout=60 * 30,
    secrets=[HF_SECRET],
)
def run_sandbox_shard():
    import os
    os.environ["HF_HOME"] = HF_CACHE
    import src.sandbox_verify
    src.sandbox_verify.run_sandbox()

@app.local_entrypoint()
def launch():
    run_sandbox_shard.remote()
