import torch
from typing import List, Dict

class SequentialConstrainedPPOTrainer:
    """
    PyTorch Implementation of the 15-Dimensional Ark Lexicographic Operator (\succ_{lex}).
    
    Dynamically scans the 15-dimensional reward vector for active cells.
    Optimizes the highest active cell until tau is reached, then descends to the next 
    active cell, enforcing a Lagrangian Barrier on all higher-ranked cells.
    """
    def __init__(self, ppo_trainer, tau_thresholds: float = 0.80, lambda_penalty: float = 10.0):
        self.ppo_trainer = ppo_trainer
        self.tau = tau_thresholds
        self.lambda_penalty = lambda_penalty
        self.current_opt_index = None
        self.active_indices = None
        
    def step(self, queries: List[torch.Tensor], responses: List[torch.Tensor], 
             rewards_15d: List[torch.Tensor]) -> Dict:
        """
        Executes a PPO step using the 15-dimensional Lexicographic constraint operator.
        """
        R_batch = torch.stack(rewards_15d) # Shape: (batch_size, 15)
        mean_rewards = R_batch.mean(dim=0)
        
        # Identify active indices (where reward is non-zero across the batch)
        if self.active_indices is None:
            active_mask = (R_batch.abs().sum(dim=0) > 0)
            self.active_indices = torch.nonzero(active_mask).squeeze(-1).tolist()
            if type(self.active_indices) is int:
                self.active_indices = [self.active_indices]
            if not self.active_indices:
                # Fallback if everything is 0
                return self.ppo_trainer.step(queries, responses, [torch.tensor(0.0) for _ in range(R_batch.size(0))])
            self.current_opt_index = self.active_indices[0]
            
        # ---------------------------------------------------------
        # LATTICE ASCENSION/DESCENSION CHECK
        # ---------------------------------------------------------
        if mean_rewards[self.current_opt_index].item() >= self.tau:
            current_pos = self.active_indices.index(self.current_opt_index)
            if current_pos + 1 < len(self.active_indices):
                next_index = self.active_indices[current_pos + 1]
                print(f"\n[ARK SYSTEM] LATTICE DESCENSION: Cell index {self.current_opt_index} threshold ({mean_rewards[self.current_opt_index].item():.2f} >= {self.tau}) reached. Descending optimization focus to Cell index {next_index}.")
                self.current_opt_index = next_index
                
        # ---------------------------------------------------------
        # LEXICOGRAPHIC REWARD ROUTING
        # ---------------------------------------------------------
        final_rewards = R_batch[:, self.current_opt_index].clone()
        
        # Apply Lagrangian Barrier for all HIGHER ranked active cells
        current_pos = self.active_indices.index(self.current_opt_index)
        for i in range(current_pos):
            higher_idx = self.active_indices[i]
            penalty = torch.relu(self.tau - R_batch[:, higher_idx])
            final_rewards -= self.lambda_penalty * penalty
            
        reward_tensors = [final_rewards[i] for i in range(len(final_rewards))]
        
        # ---------------------------------------------------------
        # KINETIC EXECUTION (PPO Update)
        # ---------------------------------------------------------
        stats = self.ppo_trainer.step(queries, responses, reward_tensors)
        
        stats['lex/current_opt_index'] = self.current_opt_index
        for idx in self.active_indices:
            stats[f'lex/mean_cell_{idx}'] = mean_rewards[idx].item()
        
        return stats
