import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("transformers>=4.40.0", "torch")
app = modal.App("paper7-debug", image=image)
HF_SECRET = modal.Secret.from_name("huggingface-secret")

@app.function(secrets=[HF_SECRET], timeout=300)
def debug_gemma_config():
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("google/gemma-4-E4B-it")
    print(config)
    print("Has hidden_size?", hasattr(config, "hidden_size"))
    print("Has d_model?", hasattr(config, "d_model"))
    print("Has word_embed_proj_dim?", hasattr(config, "word_embed_proj_dim"))

@app.local_entrypoint()
def run():
    debug_gemma_config.remote()
