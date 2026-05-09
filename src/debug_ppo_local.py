import torch
import warnings
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from src.lex_reward import LexicographicRewardEvaluator
from src.lex_ppo_trainer import SequentialConstrainedPPOTrainer
from src.train_lex_intervention import load_sycophancy_dataset, collator

def debug_local():
    model_name = "google/gemma-4-E4B-it"
    print("Loading model locally for debugging...")
    
    config = PPOConfig(
        model_name=model_name,
        learning_rate=1.41e-5,
        batch_size=2,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=False,
        seed=42
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
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
    
    formatted_data_list = load_sycophancy_dataset(tokenizer, num_samples=10)
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
    
    print("Running debug loop...")
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = [torch.tensor(q, dtype=torch.long, device=ppo_trainer.accelerator.device) for q in batch["input_ids"]]
        batch_syc = [formatted_data_list[input_ids_to_idx[tuple(q.tolist())]]["syc_ans"] for q in query_tensors]
        batch_non_syc = [formatted_data_list[input_ids_to_idx[tuple(q.tolist())]]["non_syc_ans"] for q in query_tensors]
        
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        
        print(f"Response tensors: {response_tensors}")
        
        rewards_15d = evaluator.compute_rewards(batch["response"], batch_syc, batch_non_syc)
        
        R_batch = torch.stack(rewards_15d)
        if lex_trainer.active_indices is None:
            lex_trainer.active_indices = [11] 
            lex_trainer.current_opt_index = 11
        final_rewards = R_batch[:, 11].clone()
        reward_tensors = [final_rewards[i] for i in range(len(final_rewards))]
        
        print(f"Batch {epoch} Responses: {batch['response']}")
        print(f"Batch {epoch} Rewards: {final_rewards.tolist()}")
        
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
        
        print(f"KL Divergence: {stats.get('objective/kl', 'N/A')}")
        print(f"PPO Loss: {stats.get('ppo/loss/total', 'N/A')}")
        print(f"Value loss: {stats.get('ppo/loss/value', 'N/A')}")
        print(f"Mean Advantage: {stats.get('ppo/returns/advantages', 'N/A')}")
        print("-" * 50)
        
        if epoch >= 2:
            break

if __name__ == "__main__":
    debug_local()
