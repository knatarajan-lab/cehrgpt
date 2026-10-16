"""
The open vocabulary head: the chance and the time of any question about the future of a patient.

It reads the global (linear probing) and the local (last token) state of the patient, like the
combined features of the linear probing extraction, and answers any number of questions at once.
A question enters only through the frozen embedding of its name, so a concept that was never a
MOTOR task, or a free text, can be asked about.

Like the MOTOR head (MotorTaskHead) it models a piecewise exponential hazard, in log2 units per day:

    log2 hazard[piece p, question q] = norm(W_p f) . w(e_q) + a(e_q) + piece_bias[p]

with f the features, e_q the embedding of the question, W_p one linear map per time piece, and
w(.), a(.) small functions of the embedding in place of the table of weights and biases of the
MOTOR tasks. The same loss is used, see open_vocab_nll_loss.

from_motor_head starts the head from a trained MOTOR head: the local half of the features is
ignored at first, and w(.), a(.) are fitted to the weights of the MOTOR tasks from their name
embeddings, so training starts from the MOTOR predictions and learns the rest.
"""

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from cehrgpt.models.activations import RMSNorm


class OpenVocabularyHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        question_dim: int,
        hidden_dim: int,
        num_time_pieces: int,
        residual_dim: int = 256,
        bias_init: float = -10.0,
        eps: float = 1e-6,
    ):
        """
        Args:
            feature_dim: width of the features, 2 * hidden size for the global and local states
            question_dim: width of the question embeddings
            hidden_dim: width the features are projected to, the hidden size of the MOTOR head
            num_time_pieces: number of pieces of the piecewise exponential hazard
            residual_dim: width of the hidden layer that refines the linear map of a question
            bias_init: the log2 hazard per day of a question before any training
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.question_dim = question_dim
        self.hidden_dim = hidden_dim
        self.num_time_pieces = num_time_pieces
        self.final_layer = nn.Linear(feature_dim, hidden_dim * num_time_pieces)
        self.norm = RMSNorm(hidden_dim, eps)
        # a question is a linear map of its embedding (which the MOTOR weights can initialise)
        # plus a residual that starts at zero
        self.question_weight = nn.Linear(question_dim, hidden_dim)
        self.question_bias = nn.Linear(question_dim, 1)
        self.weight_residual = _zero_residual(question_dim, residual_dim, hidden_dim)
        self.bias_residual = _zero_residual(question_dim, residual_dim, 1)
        # one offset per time piece, shared by all questions
        self.piece_bias = nn.Parameter(torch.zeros(num_time_pieces))
        nn.init.zeros_(self.question_weight.weight)
        nn.init.zeros_(self.question_weight.bias)
        nn.init.zeros_(self.question_bias.weight)
        nn.init.constant_(self.question_bias.bias, bias_init)

    def question_parameters(self, question_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The weight (questions, hidden) and the bias (questions,) of each question."""
        weight = self.question_weight(question_embeddings) + self.weight_residual(question_embeddings)
        bias = self.question_bias(question_embeddings) + self.bias_residual(question_embeddings)
        return weight, bias.squeeze(-1)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """(rows, pieces, hidden): the patient, once per time piece."""
        x = self.final_layer(features).reshape(features.shape[0], self.num_time_pieces, self.hidden_dim)
        return self.norm(x)

    def forward(self, features: torch.Tensor, question_embeddings: torch.Tensor) -> torch.Tensor:
        """The log2 hazard per day, (rows, pieces, questions)."""
        weight, bias = self.question_parameters(question_embeddings)
        return self.project(features) @ weight.T + bias + self.piece_bias[None, :, None]

    @classmethod
    def from_motor_head(
        cls,
        motor_head: nn.Module,
        task_embeddings: torch.Tensor,
        feature_dim: Optional[int] = None,
        ridge: float = 1e-3,
        **kwargs,
    ) -> "OpenVocabularyHead":
        """
        Start from a trained MotorTaskHead.

        Args:
            motor_head: the trained head, with final_layer, norm, task_layer and task_time_bias
            task_embeddings: the embeddings of the MOTOR tasks, (tasks, question_dim), in task order
            feature_dim: width of the new features, at least the width of the old ones (default: twice)
            ridge: the regularisation of the fit of the question maps to the MOTOR weights
        """
        hidden = motor_head.input_dim
        feature_dim = feature_dim or 2 * hidden
        if feature_dim < hidden:
            raise ValueError(f"feature_dim {feature_dim} is smaller than the MOTOR head input {hidden}")
        if task_embeddings.shape[0] != motor_head.motor_tte_vocab_size:
            raise ValueError(
                f"{task_embeddings.shape[0]} task embeddings for {motor_head.motor_tte_vocab_size} MOTOR tasks"
            )
        head = cls(
            feature_dim, task_embeddings.shape[1], hidden, motor_head.motor_num_time_pieces, **kwargs
        )
        with torch.no_grad():
            # the old input first, the new (local) half of the features starts at zero
            head.final_layer.weight.zero_()
            head.final_layer.weight[:, :hidden] = motor_head.final_layer.weight
            head.final_layer.bias.copy_(motor_head.final_layer.bias)
            head.norm.load_state_dict(motor_head.norm.state_dict())
            weights = motor_head.task_layer.weight.detach().double()  # (tasks, hidden)
            biases = motor_head.task_layer.bias.detach().double()
            targets = torch.cat([weights, biases[:, None]], dim=1)
            fitted = _ridge(task_embeddings.detach().double(), targets, ridge)  # (q + 1, hidden + 1)
            head.question_weight.weight.copy_(fitted[:-1, :hidden].T)
            head.question_weight.bias.copy_(fitted[-1, :hidden])
            head.question_bias.weight.copy_(fitted[:-1, hidden:].T)
            head.question_bias.bias.copy_(fitted[-1, hidden:])
            # the time bias per task and piece is shared here: its mean over the tasks
            head.piece_bias.copy_(motor_head.task_time_bias.detach().mean(dim=(0, 2)))
        return head


def _zero_residual(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    layers = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
    nn.init.zeros_(layers[-1].weight)
    nn.init.zeros_(layers[-1].bias)
    return layers


def _ridge(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    """Least squares of y on [x, 1], with the penalty on the weights of x only: (dim + 1, outputs)."""
    design = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
    penalty = ridge * torch.eye(design.shape[1], dtype=x.dtype)
    penalty[-1, -1] = 0.0
    return torch.linalg.solve(design.T @ design + penalty, design.T @ y)


def open_vocab_nll_loss(
    hazards: torch.Tensor,
    event_rows: torch.Tensor,
    event_questions: torch.Tensor,
    event_times: torch.Tensor,
    censor_times: torch.Tensor,
    time_bins: Sequence[float],
    chunk_rows: Optional[int] = None,
) -> torch.Tensor:
    """
    The negative log likelihood of the piecewise exponential model, the loss of the MOTOR head.

    Args:
        hazards: log2 hazards per day, (rows, pieces, questions), see OpenVocabularyHead.forward
        event_rows, event_questions, event_times: the questions that happen after each row (a row
            is a prediction point): the row, the question slot and the days until it happens
        censor_times: (rows,) days from each prediction point to the end of the record; a question
            that is not listed for a row is censored at this time
        time_bins: the edges of the pieces in days, pieces + 1 values
        chunk_rows: compute this many rows at a time to bound the memory (same result)

    The loss is averaged over the rows and the questions.
    """
    rows, pieces, questions = hazards.shape
    bins = torch.as_tensor(list(time_bins), dtype=hazards.dtype, device=hazards.device)
    if bins.shape[0] != pieces + 1:
        raise ValueError(f"{bins.shape[0]} time bin edges for {pieces} pieces")
    starts, ends = bins[:-1], bins[1:]
    widths = (ends - starts)[None, :, None]
    starts_3d, ends_3d = starts[None, :, None], ends[None, :, None]
    chunk_rows = chunk_rows or rows
    total = hazards.new_zeros(())
    for first in range(0, rows, chunk_rows):
        last = min(first + chunk_rows, rows)
        times = censor_times[first:last].to(hazards.dtype).view(-1, 1).repeat(1, questions)
        happened = torch.zeros((last - first, questions), dtype=torch.bool, device=hazards.device)
        selected = (event_rows >= first) & (event_rows < last)
        r, q = event_rows[selected] - first, event_questions[selected]
        times[r, q] = event_times[selected].to(hazards.dtype)
        happened[r, q] = True

        times_3d = times[:, None, :]
        in_piece = torch.minimum(torch.clamp(times_3d - starts_3d, min=0), widths)
        log_in_piece = torch.where(
            in_piece > 0, torch.log2(torch.clamp(in_piece, min=1e-8)), torch.full_like(in_piece, -math.inf)
        )
        event_in_piece = happened[:, None, :] & (times_3d >= starts_3d) & (times_3d < ends_3d)
        chunk = hazards[first:last]
        survival = torch.exp2(chunk + log_in_piece).sum()
        event = -math.log(2) * torch.where(event_in_piece, chunk, torch.zeros_like(chunk)).sum()
        total = total + survival + event
    return total / (rows * questions)


@torch.no_grad()
def window_probabilities(
    hazards: torch.Tensor,
    time_bins: Sequence[float],
    windows: Sequence[Tuple[float, float]],
    gap_days: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    The chance that each question happens in each window (start, end] days after the prediction time.

    Args:
        hazards: (rows, pieces, questions) from OpenVocabularyHead
        windows: (start, end) in days
        gap_days: (rows,) days since the last event, counted in the cumulative hazard; zero if None
    Returns:
        (rows, windows, questions)
    """
    # the same formula as the zero-shot tool for the pretrained MOTOR head
    from cehrgpt.tools.zero_shot_motor_predictions import window_probability

    bins = torch.as_tensor(list(time_bins), dtype=hazards.dtype, device=hazards.device)
    return torch.stack([window_probability(hazards, bins, start, end, gap_days) for start, end in windows], dim=1)
