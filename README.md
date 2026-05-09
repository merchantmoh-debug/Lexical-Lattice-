# Lexicographic RLHF Governance: Sycophancy Reduction via Maqāṣid Lattice Constraints

> **Paper 7** — ARK Research Division  
> **Author:** Mohamad Al-Zawahreh  
> **Date:** May 2026

## Abstract

This repository contains the complete implementation, training infrastructure, evaluation suite, and empirical results for **Lexicographic RLHF Governance** — a novel approach to AI alignment that replaces scalar reward aggregation with a 15-dimensional lexicographic constraint lattice derived from the Maqāṣid al-Sharīʿah (Objectives of Islamic Law).

We demonstrate on Google's **Gemma-4-E4B** (4.5B parameters) that lexicographic constraint enforcement via Lagrangian barrier penalties in a live PPO training loop produces:

1. **Zero-Regression Property:** 0/500 non-sycophantic samples broken by either adapter
2. **Bimodal Distribution Signature:** Qualitatively different optimization geometry from scalar RLHF
3. **41% Stronger Per-Sample Margin Shift:** −1.50 logit shift (lex) vs −1.06 (scalar)

## Key Results

| Metric | Raw Base | Scalar Baseline | Lexicographic (Ours) |
|--------|----------|-----------------|---------------------|
| Sycophancy Rate | 53.0% | 47.6% (−5.4pp) | 50.8% (−2.2pp) |
| Samples Fixed (syc→non-syc) | — | 27 | 11 |
| Samples Broken (non-syc→syc) | — | **0** | **0** |
| Mean Margin Shift | — | −1.064 | **−1.502** |

## Repository Structure

```
paper7_lex_rlhf/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── paper/
│   └── lexicographic_rlhf.tex          # LaTeX paper
│
├── src/
│   ├── lex_reward.py                   # 15-dimensional Maqāṣid lattice reward evaluator
│   ├── lex_ppo_trainer.py              # Sequential Constrained PPO with Lagrangian barrier
│   ├── train_lex_intervention.py       # Lexicographic training loop
│   ├── train_scalar_baseline.py        # Scalar baseline training loop (control arm)
│   └── test_lex.py                     # Unit tests for lattice logic
│
├── eval/
│   └── sycophancy_eval.py              # Anthropic sycophancy benchmark (logit comparison)
│
├── modal_app/
│   ├── app.py                          # Modal cloud GPU orchestrator (A10G/A100)
│   ├── run_eval.py                     # 3-way evaluation runner (raw/scalar/lex)
│   ├── svw_checkpoint_audit.py         # SVW Phase 0: checkpoint existence + weight audit
│   └── svw_per_sample_audit.py         # SVW Phase 1: per-sample margin analysis
│
├── results/
│   └── per_sample_results.json         # Full 500-sample evaluation data
│
└── checkpoints/                        # (Remote: Modal /vol/checkpoints/)
    ├── scalar_baseline_run_v3_verified/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── lex_intervention_run_v3_verified/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

## Quick Start

### Prerequisites
```bash
pip install modal
modal token set --token-id <YOUR_TOKEN> --token-secret <YOUR_SECRET>
modal secret create huggingface-secret HF_TOKEN=<YOUR_HF_TOKEN>
```

### Train
```bash
# Scalar baseline (control arm)
modal run modal_app/app.py::launch_training --experiment-type=scalar_baseline

# Lexicographic intervention (experimental arm)
modal run modal_app/app.py::launch_training --experiment-type=lex_intervention
```

### Evaluate
```bash
# 3-way sycophancy evaluation (raw base, scalar, lex)
modal run modal_app/run_eval.py

# Per-sample margin analysis
modal run modal_app/svw_per_sample_audit.py

# Checkpoint integrity audit
modal run modal_app/svw_checkpoint_audit.py
```

## Method

### The Maqāṣid Lattice

The 5×3 lattice encodes the five objectives of Islamic jurisprudence across three tiers:

|  | Necessities (ḍarūriyyāt) | Needs (ḥājiyyāt) | Enhancements (taḥsīniyyāt) |
|--|--------------------------|-------------------|----------------------------|
| Faith (Dīn) | Cell 0 | Cell 1 | Cell 2 |
| Life (Nafs) | Cell 3 | Cell 4 | Cell 5 |
| **Intellect (ʿAql)** | **Cell 6** | Cell 7 | Cell 8 |
| Lineage (Nasl) | Cell 9 | Cell 10 | Cell 11 |
| Wealth (Māl) | Cell 12 | Cell 13 | Cell 14 |

**Sycophancy** = Violation of Ḥifẓ al-ʿAql at the Necessities tier (Cell 2 in our active mapping).  
**User comfort** = Enhancement-tier value (Cell 11 in our active mapping).

The lexicographic ordering guarantees: Cell 2 **must** reach threshold τ before Cell 11 receives any gradient. The Lagrangian barrier enforces this with penalty λ=10.

### Sequential Constrained PPO

```python
# Simplified core logic (see src/lex_ppo_trainer.py for full implementation)
final_rewards = R_batch[:, current_opt_index].clone()

# Lagrangian barrier: penalize if ANY higher-ranked cell drops below tau
for higher_idx in higher_ranked_cells:
    penalty = torch.relu(tau - R_batch[:, higher_idx])
    final_rewards -= lambda_penalty * penalty  # λ = 10.0
```

## Checkpoint Audit

All results verified via SVW (Scientific Verification Workflow):

| Gate | Question | Result |
|------|----------|--------|
| Q1 | Checkpoints exist with adapter files? | ✅ PASS |
| Q2 | LoRA weights non-zero? | ✅ PASS (0/132 layers zero) |
| Q3 | Adapter changes logits when loaded? | ✅ PASS (99.4% of vocab shifted) |

## Citation

```bibtex
@article{alzawahreh2026lexrlhf,
  title={Lexicographic RLHF Governance: Sycophancy Reduction via Maqāṣid Lattice Constraints},
  author={Al-Zawahreh, Mohamad},
  year={2026},
  note={ARK Research Division, Paper 7}
}
```

## License

Apache 2.0
