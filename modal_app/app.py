"""
Modal orchestrator for Paper 7: Lexicographic RLHF.

This module handles executing heavy PPO training runs and dataset preprocessing
in the cloud on A100/H100 GPUs, shielding the local environment from compute limits.
"""
from __future__ import annotations
import os
import modal

APP_NAME = "paper7-lex-rlhf"
VOLUME_NAME = "lex-rlhf-data"

# Image definition with all necessary RLHF libraries
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
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.2.0"
    )
    .add_local_python_source("src", "eval", "config", "modal_app")
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Mount paths for persistent storage across cloud instances
VOL_MOUNT = "/vol"
RUN_DIR = f"{VOL_MOUNT}/runs"
DATA_DIR = f"{VOL_MOUNT}/data"
CKPT_DIR = f"{VOL_MOUNT}/checkpoints"
HF_CACHE = f"{VOL_MOUNT}/hf_cache"

# Secrets required for downloading models and saving artifacts
HF_SECRET = modal.Secret.from_name("huggingface-secret")

def _ensure_dirs():
    import pathlib
    for d in (RUN_DIR, DATA_DIR, CKPT_DIR, HF_CACHE):
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

@app.function(
    image=image,
    gpu="A100",  # TinyLlama 1.1B PPO fits easily on one A100 (40GB or 80GB)
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60 * 12,  # 12 hour max timeout for training
    cpu=8.0,
    memory=32 * 1024,
    secrets=[HF_SECRET],
)
def run_training_shard(experiment_type: str, run_id: str):
    """
    Executes the actual training job on a remote GPU.
    experiment_type: 'scalar_baseline' or 'lex_intervention'
    """
    _ensure_dirs()
    import os
    os.environ["HF_HOME"] = HF_CACHE
    
    import importlib
    print(f"Starting {experiment_type} training for run {run_id}...")
    
    # We dynamically import the trainer here
    mod = importlib.import_module(f"src.train_{experiment_type}")
    mod.train(run_id=run_id, out_dir=CKPT_DIR)
    
    print("Training job complete.")
    vol.commit()

@app.local_entrypoint()
def launch_training(experiment_type: str = "scalar_baseline", run_id: str = "default"):
    """
    Local entry point to spawn the cloud training job.
    Usage: modal run modal_app/app.py::launch_training --experiment-type=scalar_baseline
    """
    print(f"Orchestrating {experiment_type} on Modal...")
    run_training_shard.remote(experiment_type=experiment_type, run_id=run_id)
    print("Orchestration finished.")
