from dataclasses import dataclass, field
from typing import Optional


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
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate per rollout trajectory."},
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
    max_future_length: int = field(
        default=0,
        metadata={
            "help": (
                "Maximum number of future tokens to store for embedding reward comparison. "
                "0 means use max_prefix_length."
            )
        },
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
