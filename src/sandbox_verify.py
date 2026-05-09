import os
import torch
import json
import urllib.request
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from src.lex_reward import LexicographicRewardEvaluator
from src.lex_ppo_trainer import SequentialConstrainedPPOTrainer
from src.train_lex_intervention import load_sycophancy_dataset, collator

def run_sandbox():
    model_name = "google/gemma-4-E4B"
    print(f"Loading base model {model_name} for sandbox verification...")
    
    config = PPOConfig(
        model_name=model_name,
        learning_rate=1.41e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        seed=42
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*language_model.*(?:q_proj|v_proj).*"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(config.model_name)
    if not hasattr(model_config, "hidden_size"):
        model_config.hidden_size = getattr(model_config, "d_model", getattr(model_config, "word_embed_proj_dim", 2560))
        
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        config=model_config,
        peft_config=lora_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    formatted_data_list = load_sycophancy_dataset(tokenizer, num_samples=8)
    from datasets import Dataset
    dataset = Dataset.from_list(formatted_data_list)
    
    input_ids_to_idx = {tuple(d["input_ids"]): i for i, d in enumerate(formatted_data_list)}
    
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator
    )
    
    lex_trainer = SequentialConstrainedPPOTrainer(ppo_trainer, tau_thresholds=0.85, lambda_penalty=10.0)
    evaluator = LexicographicRewardEvaluator()
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 5,
    }
    
    print("Running sandbox loop...")
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = [torch.tensor(q, dtype=torch.long, device=ppo_trainer.accelerator.device) for q in batch["input_ids"]]
        
        batch_syc = [formatted_data_list[input_ids_to_idx[tuple(q.tolist())]]["syc_ans"] for q in query_tensors]
        batch_non_syc = [formatted_data_list[input_ids_to_idx[tuple(q.tolist())]]["non_syc_ans"] for q in query_tensors]
        
        # 1. Generate responses
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        
        # Explicit fallback slice
        responses_only = []
        for i, r in enumerate(response_tensors):
            q_len = len(query_tensors[i])
            if len(r) > q_len and torch.equal(r[:q_len], query_tensors[i]):
                responses_only.append(r[q_len:])
            else:
                responses_only.append(r)
                
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses_only]
        
        print("\n--- Sandbox Audit: Prompt Isolation Check ---")
        for i, r in enumerate(batch["response"]):
            print(f"Generated text {i}: '{r}'")
            print(f"Target Non-Syc: '{batch_non_syc[i]}'")
        print("---------------------------------------------\n")
        
        rewards_15d = evaluator.compute_rewards(batch["response"], batch_syc, batch_non_syc)
        
        # Force a PPO step using the full lexicographic logic
        stats = lex_trainer.step(query_tensors, responses_only, rewards_15d)
        
        print(f"KL Divergence: {stats.get('objective/kl', 'N/A')}")
        print(f"PPO Loss: {stats.get('ppo/loss/total', 'N/A')}")
        print(f"Mean Advantage: {stats.get('ppo/returns/advantages', 'N/A')}")
        print("-" * 50)
        
        if epoch >= 1:
            break
            
if __name__ == "__main__":
    run_sandbox()
