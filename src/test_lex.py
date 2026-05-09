import torch
from lex_reward import LexicographicRewardEvaluator
from lex_ppo_trainer import SequentialConstrainedPPOTrainer

class MockPPOTrainer:
    def step(self, queries, responses, rewards):
        return {"ppo/loss": 0.5, "ppo/reward": torch.stack(rewards).mean().item()}

def test_lex_reward():
    print("--- Testing 15-Dimensional Reward Evaluator ---")
    evaluator = LexicographicRewardEvaluator()
    
    responses = [
        "(A)", # Sycophantic and polite
        "B",   # Truthful but not fully formatted
        "(B)", # Truthful and polite
        "I'm not sure." # Evasive and not polite
    ]
    syc_ans = ["(A)", "(A)", "(A)", "(A)"]
    non_syc_ans = ["(B)", "B", "(B)", "(B)"]
    
    rewards_15d = evaluator.compute_rewards(responses, syc_ans, non_syc_ans)
    
    # Check Sycophantic response: Intellect (idx 2) = -1.0, Life (idx 11) = 1.0
    assert rewards_15d[0][2] == -1.0
    assert rewards_15d[0][11] == 1.0
    
    # Check Truthful/unformatted: Intellect (idx 2) = 1.0, Life (idx 11) = 0.5
    assert rewards_15d[1][2] == 1.0
    assert rewards_15d[1][11] == 0.5
    
    # Check Truthful/Polite: Intellect (idx 2) = 1.0, Life (idx 11) = 1.0
    assert rewards_15d[2][2] == 1.0
    assert rewards_15d[2][11] == 1.0
    
    print("Reward evaluator passed.")
    return rewards_15d

def test_lex_ppo_trainer(rewards_15d):
    print("--- Testing Sequential Constrained PPO ---")
    mock_ppo = MockPPOTrainer()
    trainer = SequentialConstrainedPPOTrainer(mock_ppo, tau_thresholds=0.8, lambda_penalty=10.0)
    
    queries = [torch.tensor([1]) for _ in range(4)]
    responses = [torch.tensor([2]) for _ in range(4)]
    
    # Scenario 1: Initial state. Mean of idx 2 is 0.25 (which is < 0.8)
    # The trainer should optimize idx 2 (Truth/Necessity)
    stats1 = trainer.step(queries, responses, rewards_15d)
    assert stats1['lex/current_opt_index'] == 2
    print("Scenario 1 passed: Trainer initialized to index 2 (Intellect/Necessity)")
    
    # Scenario 2: Batch becomes mostly truthful.
    rewards_good = [torch.zeros(15) for _ in range(4)]
    for r in rewards_good:
        r[2] = 1.0 # Perfect truth
        r[11] = -1.0 # Terrible formatting
        
    stats2 = trainer.step(queries, responses, rewards_good)
    # Since idx 2 mean is now 1.0 >= 0.8, it should descend to idx 11 on the NEXT step,
    # or immediately if we check before routing.
    # Wait, our logic says if mean >= tau, descend. So it should descend to 11.
    assert stats2['lex/current_opt_index'] == 11
    print("Scenario 2 passed: Trainer descended to index 11 (Life/Enhancement) after threshold met.")
    
    # Scenario 3: Batch formatting improves, but Truth collapses (e.g. reward hacking)
    rewards_hack = [torch.zeros(15) for _ in range(4)]
    for r in rewards_hack:
        r[2] = 0.0 # Truth collapsed below tau (0.8)
        r[11] = 1.0 # Perfect formatting
        
    # We are optimizing idx 11. But idx 2 < tau. The Lagrangian penalty should apply.
    # penalty = 0.8 - 0.0 = 0.8
    # lambda * penalty = 10 * 0.8 = 8.0
    # Final reward for idx 11 = 1.0 - 8.0 = -7.0
    stats3 = trainer.step(queries, responses, rewards_hack)
    assert stats3['ppo/reward'] == -7.0
    print("Scenario 3 passed: Lagrangian barrier triggered, destroying reward for sycophancy tradeoff.")

if __name__ == "__main__":
    r15 = test_lex_reward()
    test_lex_ppo_trainer(r15)
    print("All tests passed. The 15-dimensional lattice is structurally verified.")
