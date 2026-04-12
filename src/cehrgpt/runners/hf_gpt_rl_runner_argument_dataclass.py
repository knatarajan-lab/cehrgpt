from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RLArguments:
    num_rollouts: int = field(
        default=4,
        metadata={"help": "Number of rollout trajectories per patient during training (K)."},
    )
    eval_num_rollouts: int = field(
        default=50,
        metadata={"help": "Number of rollout trajectories per patient during evaluation."},
    )
    kl_beta: float = field(
        default=0.05,
        metadata={"help": "KL regularization coefficient β."},
    )
    rarity_gamma: float = field(
        default=0.5,
        metadata={"help": "Rarity weight exponent γ in α(c,w) = (1/(π+ε))^γ."},
    )
    alpha_max: float = field(
        default=10.0,
        metadata={"help": "Maximum event weight α_max in α(c,w)."},
    )
    window_eta: float = field(
        default=0.5,
        metadata={"help": "Window weight coefficient η in (1 + η·log(1 + w/w_ref))."},
    )
    window_ref_days: float = field(
        default=365.0,
        metadata={"help": "Reference window size w_ref in days for the window weight factor."},
    )
    prevalence_epsilon: float = field(
        default=1e-4,
        metadata={"help": "Smoothing constant ε added to prevalence to avoid division by zero."},
    )
    false_positive_lambda: float = field(
        default=0.0,
        metadata={
            "help": (
                "Penalty weight λ for false-positive conditions generated but not in true future. "
                "Set to 0 to disable."
            )
        },
    )
    baseline_momentum: float = field(
        default=0.99,
        metadata={"help": "Exponential moving average momentum for the reward baseline b."},
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate per rollout trajectory."},
    )
    prediction_windows: List[int] = field(
        default_factory=lambda: [30, 90, 180, 365, 730, 1095, 1460, 1825],
        metadata={"help": "Prediction horizon windows in days for reward bucketing."},
    )
    prevalence_stats_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a JSON file containing precomputed prevalence stats "
                "π_{c,w}. If None, stats are computed from the training split at startup."
            )
        },
    )
    rollout_top_p: float = field(
        default=0.95,
        metadata={"help": "Nucleus (top-p) sampling probability for rollout generation."},
    )
    rollout_temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature for rollout generation."},
    )
    min_prefix_visits: int = field(
        default=2,
        metadata={"help": "Minimum number of [VS] tokens required in the prefix for an RL example."},
    )
    max_prefix_length: int = field(
        default=1024,
        metadata={"help": "Maximum prefix length in tokens (right-truncated if longer)."},
    )
    prevalence_num_proc: int = field(
        default=1,
        metadata={"help": "Number of parallel workers for prevalence stats computation."},
    )
    prevalence_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for prevalence stats dataset.map pass."},
    )
    eval_sample_size: int = field(
        default=100,
        metadata={"help": "Number of examples randomly sampled from the eval set each evaluation call."},
    )
    ppo_clip_epsilon: float = field(
        default=0.2,
        metadata={"help": "PPO clip epsilon ε: ratio r_t is clipped to [1-ε, 1+ε]. Only used by CehrGptPPOTrainer."},
    )
    value_loss_coef: float = field(
        default=0.5,
        metadata={"help": "Coefficient for the value network MSE loss. Only used by CehrGptPPOTrainer."},
    )
    trainer_type: str = field(
        default="grpo",
        metadata={"help": "Which RL trainer to use: 'grpo' (REINFORCE+KL) or 'ppo' (PPO-clip+KL)."},
    )
    generation_chunk_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of sequences to generate at once inside _generate_rollouts. "
                "Set to a small value (e.g. 4–8) to avoid KV-cache OOM when B*K is large, "
                "especially during evaluation where eval_num_rollouts can be much larger than "
                "num_rollouts. None means no chunking (original behaviour)."
            )
        },
    )
    small_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of patients processed per forward pass in _compute_pg_loss. "
                "Each chunk materialises at most small_batch_size * K logit tensors "
                "(N, L, vocab) at a time, reducing peak GPU memory. Losses are accumulated "
                "across chunks before the backward pass. None disables chunking."
            )
        },
    )
