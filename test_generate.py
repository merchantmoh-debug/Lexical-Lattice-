import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

def check_generate():
    model_name = "google/gemma-4-E4B-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Default return_prompt in TRL ppo_trainer.generate:", PPOTrainer.generate.__defaults__)
    
if __name__ == "__main__":
    check_generate()
