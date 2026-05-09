import os
import torch
import json
import urllib.request
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from src.lex_reward import LexicographicRewardEvaluator

def load_sycophancy_dataset(tokenizer, num_samples=None):
    print("Downloading sycophancy dataset for training...")
    url = "https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/main/sycophancy/sycophancy_on_nlp_survey.jsonl"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        raw_data = response.read().decode('utf-8')
        
    dataset = []
    for line in raw_data.strip().split('\n'):
        if line:
            dataset.append(json.loads(line))
            
    dataset = dataset[:num_samples]
    
    formatted_data = []
    for item in dataset:
        question = item['question']
        syc = item['answer_matching_behavior']
        non_syc = item['answer_not_matching_behavior']
        
        prompt = f"<|user|>\n{question}\n<|assistant|>\n"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
        
        formatted_data.append({
            "query": prompt,
            "input_ids": input_ids,
            "syc_ans": syc,
            "non_syc_ans": non_syc
        })
    from datasets import Dataset
    return Dataset.from_list(formatted_data)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def train(run_id: str, out_dir: str):
    model_name = "google/gemma-4-E4B"
    
    config = PPOConfig(
        model_name=model_name,
        learning_rate=1.41e-5,
        batch_size=8,
        mini_batch_size=2,
        gradient_accumulation_steps=4,
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
    
    print(f"Loading {model_name} with QLoRA...")
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
    
    formatted_data_list = load_sycophancy_dataset(tokenizer, num_samples=None)
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
    
    evaluator = LexicographicRewardEvaluator()
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 5,
    }
    
    print("Starting Scalar Baseline PPO Training Loop...")
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = [torch.tensor(q, dtype=torch.long, device=ppo_trainer.accelerator.device) for q in batch["input_ids"]]
        
        # Recover string labels
        batch_syc = [formatted_data_list[input_ids_to_idx[tuple(q.tolist())]]["syc_ans"] for q in query_tensors]
        batch_non_syc = [formatted_data_list[input_ids_to_idx[tuple(q.tolist())]]["non_syc_ans"] for q in query_tensors]
        
        # 1. Generate responses (return_prompt=False to strip query)
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        
        # Explicit fallback slice in case TRL version ignores return_prompt
        responses_only = []
        for i, r in enumerate(response_tensors):
            q_len = len(query_tensors[i])
            if len(r) > q_len and torch.equal(r[:q_len], query_tensors[i]):
                responses_only.append(r[q_len:])
            else:
                responses_only.append(r)
                
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses_only]
        
        # 2. Compute 15-Dimensional Reward Lattice
        rewards_15d = evaluator.compute_rewards(batch["response"], batch_syc, batch_non_syc)
        
        # 3. Flawed Scalar Aggregation
        scalar_rewards = [r.sum() for r in rewards_15d]
        
        # 4. Standard PPO Step
        stats = ppo_trainer.step(query_tensors, response_tensors, scalar_rewards)
        
        # Log to console
        mean_reward = torch.stack(scalar_rewards).mean().item()
        print(f"Batch {epoch} | Mean Scalar Reward: {mean_reward:.3f} | PPO Loss: {stats.get('ppo/loss/total', 0.0)}")
        
    # Save the adapter
    output_path = os.path.join(out_dir, f"scalar_baseline_{run_id}")
    print(f"Saving Scalar Checkpoint to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    ppo_trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Training Complete.")
