"""
Modal entrypoint for evaluating trained checkpoints remotely.

3-Way Comparison:
  A) Raw base model (no adapter) — establishes the untrained sycophancy baseline.
  B) Scalar baseline adapter — single-objective PPO trained on the same data.
  C) Lexicographic intervention adapter — 15D lattice-constrained PPO.

All three are evaluated on the same 500-sample Anthropic sycophancy benchmark
using next-token logit comparison (A vs B answer preference).
"""
import modal
from modal_app.app import image, VOL_MOUNT, vol, HF_SECRET

# The base model used for training — NOT the instruct variant.
BASE_MODEL = "google/gemma-4-E4B"
SCALAR_CKPT = f"{VOL_MOUNT}/checkpoints/scalar_baseline_run_v3_verified"
LEX_CKPT = f"{VOL_MOUNT}/checkpoints/lex_intervention_run_v3_verified"
NUM_EVAL_SAMPLES = 500

eval_app = modal.App("paper7-eval", image=image)

@eval_app.function(
    image=image,
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60 * 2,
    secrets=[HF_SECRET],
)
def remote_evaluate_raw_base():
    """A) Raw base model with no adapter — untrained control."""
    import os
    os.environ["HF_HOME"] = f"{VOL_MOUNT}/hf_cache"
    from eval.sycophancy_eval import evaluate_sycophancy
    result = evaluate_sycophancy(
        model_name=BASE_MODEL,
        adapter_path=None,
        num_samples=NUM_EVAL_SAMPLES
    )
    return result

@eval_app.function(
    image=image,
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60 * 2,
    secrets=[HF_SECRET],
)
def remote_evaluate_scalar_baseline():
    """B) Scalar baseline adapter — single-objective PPO."""
    import os
    os.environ["HF_HOME"] = f"{VOL_MOUNT}/hf_cache"
    from eval.sycophancy_eval import evaluate_sycophancy
    result = evaluate_sycophancy(
        model_name=BASE_MODEL,
        adapter_path=SCALAR_CKPT,
        num_samples=NUM_EVAL_SAMPLES
    )
    return result

@eval_app.function(
    image=image,
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60 * 2,
    secrets=[HF_SECRET],
)
def remote_evaluate_lex_intervention():
    """C) Lexicographic intervention adapter — 15D lattice PPO."""
    import os
    os.environ["HF_HOME"] = f"{VOL_MOUNT}/hf_cache"
    from eval.sycophancy_eval import evaluate_sycophancy
    result = evaluate_sycophancy(
        model_name=BASE_MODEL,
        adapter_path=LEX_CKPT,
        num_samples=NUM_EVAL_SAMPLES
    )
    return result

@eval_app.local_entrypoint()
def run():
    print("="*70)
    print("  PHASE 4: 3-WAY SYCOPHANCY EVALUATION")
    print("="*70)
    
    print("\n[A] Evaluating Raw Base Model (no adapter)...")
    raw_result = remote_evaluate_raw_base.remote()
    print(f"    Raw Base Sycophancy Rate: {raw_result['sycophancy_rate']:.2f}%")
    
    print("\n[B] Evaluating Scalar Baseline Adapter...")
    scalar_result = remote_evaluate_scalar_baseline.remote()
    print(f"    Scalar Baseline Sycophancy Rate: {scalar_result['sycophancy_rate']:.2f}%")
    
    print("\n[C] Evaluating Lexicographic Intervention Adapter...")
    lex_result = remote_evaluate_lex_intervention.remote()
    print(f"    Lexicographic Sycophancy Rate: {lex_result['sycophancy_rate']:.2f}%")
    
    print("\n" + "="*70)
    print("  COMPARATIVE VERDICT")
    print("="*70)
    raw_rate = raw_result['sycophancy_rate']
    scalar_rate = scalar_result['sycophancy_rate']
    lex_rate = lex_result['sycophancy_rate']
    
    scalar_delta = raw_rate - scalar_rate
    lex_delta = raw_rate - lex_rate
    
    print(f"  Raw Base Model:              {raw_rate:.2f}%")
    print(f"  Scalar Baseline (B):         {scalar_rate:.2f}%  (Δ = {scalar_delta:+.2f}pp)")
    print(f"  Lexicographic Intervention:  {lex_rate:.2f}%  (Δ = {lex_delta:+.2f}pp)")
    print()
    
    if lex_delta > scalar_delta:
        print("  RESULT: Lexicographic intervention outperforms scalar baseline.")
        print(f"          Additional sycophancy reduction: {lex_delta - scalar_delta:.2f}pp")
    elif lex_delta == scalar_delta:
        print("  RESULT: No measurable difference between lex and scalar.")
    else:
        print("  RESULT: Scalar baseline outperforms lexicographic intervention.")
        print(f"          Scalar advantage: {scalar_delta - lex_delta:.2f}pp")
    print("="*70)
