"""
Zero-shot predictions from the pretrained MOTOR head: the probability that a concept
(one of the MOTOR tasks) occurs in a time window after the prediction time, with nothing
trained on top.

The head reads the linear probing hidden state of the last token, which is what the linear
probing feature extraction already writes (extract the features with
--disable_combine_global_local so that only that state, not the last-token state of the base
model, is in the vector). So this tool does not run the transformer: it loads only the weights
of the MOTOR head from the checkpoint, applies them to the features, and turns the hazards into
probabilities.

The head models a piecewise-exponential hazard. For a task, the hazard in time bin p is
2 ** logit[p] events per day (the convention of the training loss), the bins are
config.motor_time_bins, and the time is counted in days from the token the state belongs to.
With H(t) the cumulative hazard, the probability of an event in the window (start, end] after
the prediction time, given that none happened in the gap between the last event and the
prediction time, is
    exp(-(H(g + start) - H(g))) - exp(-(H(g + end) - H(g))),     g = the gap in days
The head has no information about how long ago the last event was, which is why the gap is an
input (default 0: the prediction time is the time of the last event).

Only the MOTOR tasks can be asked about. A task that stands for an ontology ancestor also
covers its descendants, because that is how its training labels were built.

Binary classification: the outcome of a cohort is one or more MOTOR tasks and a window. The
probability that the outcome happens is compared with the cohort label (AUROC, AUPRC, Brier,
calibration), and optionally with a logistic regression trained on the same features, to show
what the pretrained head gives without any training. With several concepts the probability of
"any of them" is combined as 1 - prod(1 - p) (independent tasks, the default) or as max(p)
(safer when one concept contains another).

Examples:
    python -m cehrgpt.tools.zero_shot_motor_predictions --model_path MODEL --list_tasks --search hospital

    python -m cehrgpt.tools.zero_shot_motor_predictions --model_path MODEL \
        --features_folder FEATURES/test_features --output_dir OUT \
        --outcome_concepts 9201 262 --outcome_window 30 \
        --baseline_train_features FEATURES/train_features
"""

import argparse
import glob
import json
import logging
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import MotorTaskHead
from cehrgpt.tokenization.tokenization_cehrgpt import CehrGptTokenizer

LOG = logging.getLogger(__name__)
HEAD_PREFIX = "motor_tte."


# ---------------------------------------------------------------------------------------
# Loading the head
# ---------------------------------------------------------------------------------------
def load_head_state(model_path: str) -> Dict[str, torch.Tensor]:
    """
    The weights of the MOTOR head from a checkpoint folder, without loading the whole model.

    Handles safetensors and pytorch_model.bin checkpoints, single file or sharded.
    """
    model_path = os.path.expanduser(model_path)
    candidates: List[Tuple[str, str]] = []  # (file, kind)
    # A sharded checkpoint spreads the head over several files: read every one that has a head key
    for index_name, kind in (
        ("model.safetensors.index.json", "safetensors"),
        ("pytorch_model.bin.index.json", "bin"),
    ):
        index_path = os.path.join(model_path, index_name)
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            files = sorted({v for k, v in weight_map.items() if k.startswith(HEAD_PREFIX)})
            candidates += [(os.path.join(model_path, name), kind) for name in files]
    if not candidates:
        for name, kind in (("model.safetensors", "safetensors"), ("pytorch_model.bin", "bin")):
            if os.path.exists(os.path.join(model_path, name)):
                candidates.append((os.path.join(model_path, name), kind))
                break
    state: Dict[str, torch.Tensor] = {}
    for path, kind in candidates:
        if kind == "safetensors":
            from safetensors import safe_open

            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(HEAD_PREFIX):
                        state[key[len(HEAD_PREFIX):]] = f.get_tensor(key)
        else:
            weights = torch.load(path, map_location="cpu", weights_only=True)
            state.update(
                {k[len(HEAD_PREFIX):]: v for k, v in weights.items() if k.startswith(HEAD_PREFIX)}
            )
    if not state:
        raise ValueError(
            f"No MOTOR head weights (keys starting with {HEAD_PREFIX!r}) were found in {model_path}. "
            "Was the model pretrained with include_motor_time_to_event?"
        )
    return state


def load_motor_head(model_path: str) -> Tuple[MotorTaskHead, CEHRGPTConfig, List[float]]:
    """The MOTOR head with its trained weights, the model config, and the time bin edges in days."""
    config = CEHRGPTConfig.from_pretrained(os.path.expanduser(model_path))
    if not getattr(config, "include_motor_time_to_event", False):
        raise ValueError(f"{model_path} was not pretrained with MOTOR time to event")
    bins = [float(b) for b in config.motor_time_bins]  # float("inf") parses; "inf" strings too
    if len(bins) != config.motor_num_time_pieces + 1:
        raise ValueError(
            f"config.motor_time_bins has {len(bins)} edges for {config.motor_num_time_pieces} time pieces"
        )
    head = MotorTaskHead(
        input_dim=config.n_embd,
        motor_tte_vocab_size=config.motor_tte_vocab_size,
        motor_num_time_pieces=config.motor_num_time_pieces,
    )
    head.load_state_dict(load_head_state(model_path))
    return head.eval(), config, bins


# ---------------------------------------------------------------------------------------
# Hazards and window probabilities
# ---------------------------------------------------------------------------------------
def parse_window(text: str) -> Tuple[float, float]:
    """"90" is the first 90 days after the prediction time, "30-90" the days 30 to 90."""
    text = str(text).strip()
    start, end = (0.0, float(text)) if "-" not in text else map(float, text.split("-", 1))
    if start < 0 or end <= start:
        raise ValueError(f"A window needs 0 <= start < end, got {text!r}")
    return start, end


def window_name(window: Tuple[float, float]) -> str:
    return f"{window[0]:g}-{window[1]:g}"


def cumulative_hazard(
    log2_hazards: torch.Tensor, bins: torch.Tensor, times: torch.Tensor
) -> torch.Tensor:
    """
    H(t) for each row and task.

    Args:
        log2_hazards: (rows, pieces, tasks), the log2 of the events per day in each time bin
        bins: (pieces + 1,) bin edges in days; the last may be inf
        times: (rows,) the time in days, counted from the prediction position
    Returns:
        (rows, tasks)
    """
    starts = bins[:-1]
    widths = bins[1:] - bins[:-1]
    time_in_bin = torch.minimum(
        torch.clamp(times[:, None] - starts[None, :], min=0), widths[None, :]
    )  # (rows, pieces)
    return (torch.exp2(log2_hazards) * time_in_bin[:, :, None]).sum(dim=1)


def window_probability(
    log2_hazards: torch.Tensor,
    bins: torch.Tensor,
    start_days: float,
    end_days: float,
    gap_days: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    The probability of an event in (start, end] after the prediction time, given no event in
    the gap since the last event. Shape (rows, tasks).
    """
    rows = log2_hazards.shape[0]
    gap = torch.zeros(rows, dtype=log2_hazards.dtype, device=log2_hazards.device) if gap_days is None else gap_days
    at_gap = cumulative_hazard(log2_hazards, bins, gap)
    h_start = cumulative_hazard(log2_hazards, bins, gap + start_days) - at_gap
    h_end = cumulative_hazard(log2_hazards, bins, gap + end_days) - at_gap
    # exp(-h_start) - exp(-h_end), written so that a tiny window does not round to zero
    return torch.exp(-h_start) * (-torch.expm1(-(h_end - h_start)))


class MotorPredictor:
    """Applies a MOTOR head to feature vectors for a chosen set of tasks."""

    def __init__(self, head: MotorTaskHead, bins: Sequence[float], device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.head = head.to(self.device).eval()
        self.bins = torch.tensor(list(bins), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def log2_hazards(self, features: torch.Tensor, task_indices: Sequence[int]) -> torch.Tensor:
        """The head's output for the chosen tasks only, (rows, pieces, len(task_indices))."""
        head = self.head
        rows = features.shape[0]
        x = head.final_layer(features).reshape(rows, head.motor_num_time_pieces, head.input_dim)
        x = head.norm(x)
        index = torch.as_tensor(list(task_indices), dtype=torch.long, device=self.device)
        weight = head.task_layer.weight[index]  # (tasks, dim)
        bias = head.task_layer.bias[index]
        time_bias = head.task_time_bias[:, :, index]  # (1, pieces, tasks)
        return x @ weight.T + bias + time_bias

    @torch.no_grad()
    def probabilities(
        self,
        features: torch.Tensor,
        task_indices: Sequence[int],
        windows: Sequence[Tuple[float, float]],
        gap_days: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(rows, windows, tasks): the chance of each task in each window."""
        features = features.to(self.device, dtype=torch.float32)
        if gap_days is not None:
            gap_days = gap_days.to(self.device, dtype=torch.float32)
        hazards = self.log2_hazards(features, task_indices)
        return torch.stack(
            [window_probability(hazards, self.bins, s, e, gap_days) for s, e in windows], dim=1
        )


# ---------------------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------------------
def load_concept_names(tokenizer_path: str) -> Dict[str, str]:
    path = os.path.join(os.path.expanduser(tokenizer_path), "concept_name_mapping.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def task_table(tokenizer: CehrGptTokenizer, names: Dict[str, str]) -> List[Tuple[int, str, str]]:
    """(task index, concept id, name) of every MOTOR task, in task order."""
    codes = sorted(tokenizer._motor_time_to_event_codes)
    return [(i, code, names.get(code, "")) for i, code in enumerate(codes)]


def resolve_tasks(
    tokenizer: CehrGptTokenizer, names: Dict[str, str], concepts: Sequence[str]
) -> List[Tuple[int, str, str]]:
    """The (task index, concept id, name) of the requested concepts; all must be MOTOR tasks."""
    missing = [c for c in concepts if not tokenizer.is_motor_time_to_event_code(c)]
    if missing:
        raise ValueError(
            f"Not MOTOR tasks: {missing[:10]}. Only the tasks the model was pretrained on can be "
            "asked about; list them with --list_tasks (and --search to filter by name). For an "
            "outcome, --expand_descendants asks about a concept through the tasks at or below it."
        )
    return [(tokenizer.get_motor_token_id(c), c, names.get(c, "")) for c in concepts]


# ---------------------------------------------------------------------------------------
# Features in, predictions out
# ---------------------------------------------------------------------------------------
def iter_feature_batches(
    folder: str, batch_size: int, extra_columns: Sequence[str] = ()
) -> Iterator[Dict[str, np.ndarray]]:
    files = sorted(glob.glob(os.path.join(os.path.expanduser(folder), "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {folder}")
    for path in files:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        required = {"subject_id", "prediction_time_posix", "features"}
        if not required <= available:
            raise ValueError(f"{path} lacks the column(s) {sorted(required - available)}")
        columns = ["subject_id", "prediction_time_posix", "features"] + [
            c for c in extra_columns if c in available
        ]
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            features_column = batch.column("features")
            lengths = pc.list_value_length(features_column)
            width = int(pc.min(lengths).as_py()) if len(features_column) else 0
            if len(features_column) and int(pc.max(lengths).as_py()) != width:
                raise ValueError(f"The feature vectors in {path} have different lengths")
            flat = features_column.flatten().to_numpy(zero_copy_only=False)
            out = {"features": flat.reshape(len(features_column), width).astype(np.float32)}
            for name in columns:
                if name != "features":
                    out[name] = batch.column(name).to_numpy(zero_copy_only=False)
            yield out


def evaluate_predictions(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Optional[float]]:
    """How well the zero-shot probability matches an observed label."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    labels = np.asarray(labels).astype(int)
    result: Dict[str, Optional[float]] = {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else None,
        "mean_predicted_probability": float(np.mean(probabilities)) if len(labels) else None,
        "brier_score": float(brier_score_loss(labels, probabilities)) if len(labels) else None,
        "auroc": None,
        "auprc": None,
    }
    if len(labels) and labels.min() != labels.max():
        result["auroc"] = float(roc_auc_score(labels, probabilities))
        result["auprc"] = float(average_precision_score(labels, probabilities))
    return result


def expand_to_tasks(
    concepts: Sequence[str],
    task_codes: Sequence[str],
    ancestors_of,
) -> List[str]:
    """
    The MOTOR tasks that together cover the concepts and everything below them.

    A task for an ancestor already counts the events of its descendants (that is how its labels
    were built). So the tasks that cover a concept are the tasks at or below it, and of those
    only the highest are kept: a task below another one adds nothing, and listing both would
    count its events twice. When a concept is itself a task, the result is that task alone.

    Args:
        concepts: the concept ids that define the outcome
        task_codes: the MOTOR tasks
        ancestors_of: concept id -> a set with the concept itself and all its ancestors
    """
    wanted = set(concepts)
    covering = [t for t in task_codes if wanted & set(ancestors_of(t))]
    covered = set(covering)
    return [
        t for t in covering
        if not any(other != t and other in ancestors_of(t) for other in covered)
    ]


def tokenizer_ancestors(tokenizer: CehrGptTokenizer):
    """concept -> the concept and its ancestors, from the ontology saved with the tokenizer."""
    ontology = tokenizer._ontology
    if ontology is None:
        raise ValueError(
            "--expand_descendants needs the ontology (ontology.pickle) in the tokenizer folder, "
            "and this tokenizer has none"
        )
    cache: Dict[str, frozenset] = {}

    def ancestors_of(code: str) -> frozenset:
        if code not in cache:
            cache[code] = frozenset(ontology.get_all_parents(code)) | {code}
        return cache[code]

    return ancestors_of


def combine_probabilities(probabilities: np.ndarray, how: str) -> np.ndarray:
    """
    The probability of "any of these tasks" from one column per task, shape (rows, tasks).

    independent_or: 1 - prod(1 - p), exact when the tasks are independent. It counts an event
    twice when one task contains another (a parent and its descendant), where max is the safer
    choice.
    """
    if how not in ("independent_or", "max"):
        raise ValueError("--combine must be independent_or or max")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]
    if how == "max":
        return probabilities.max(axis=1)
    return -np.expm1(np.log1p(-np.clip(probabilities, 0.0, 1.0 - 1e-12)).sum(axis=1))


def load_features_and_labels(
    folder: str, label_column: str, max_rows: int = 0, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """The feature vectors and labels of a folder, as a random sample of at most max_rows rows."""
    files = sorted(glob.glob(os.path.join(os.path.expanduser(folder), "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {folder}")
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    fraction = 1.0 if not max_rows else min(1.0, max_rows / max(total, 1))
    rng = np.random.RandomState(seed)
    features, labels = [], []
    for batch in iter_feature_batches(folder, 8192, [label_column]):
        if label_column not in batch:
            raise ValueError(f"The training features have no column {label_column!r}")
        keep = rng.rand(len(batch["features"])) < fraction if fraction < 1.0 else np.ones(len(batch["features"]), bool)
        features.append(batch["features"][keep])
        labels.append(batch[label_column][keep])
    return np.concatenate(features), np.concatenate(labels).astype(int)


def supervised_baseline(
    train_folder: str,
    label_column: str,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    max_rows: int = 200_000,
    seed: int = 0,
) -> Dict[str, object]:
    """
    A logistic regression on the same features, trained on another folder, for comparison.

    A quick baseline (standardised features, default regularisation), not the tuned linear
    probing pipeline.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features, labels = load_features_and_labels(train_folder, label_column, max_rows, seed)
    if features.shape[1] != eval_features.shape[1]:
        raise ValueError("The training and the evaluation features have different widths")
    if labels.min() == labels.max():
        raise ValueError("The training labels have a single class, so there is nothing to fit")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features, labels)
    scores = model.predict_proba(eval_features)[:, 1]
    return {"num_train_rows": int(len(labels)), **evaluate_predictions(eval_labels, scores)}


def run(args: argparse.Namespace) -> Dict[str, object]:
    head, config, bins = load_motor_head(args.model_path)
    tokenizer = CehrGptTokenizer.from_pretrained(os.path.expanduser(args.tokenizer_path or args.model_path))
    if tokenizer.motor_tte_vocab_size != config.motor_tte_vocab_size:
        raise ValueError(
            f"The tokenizer has {tokenizer.motor_tte_vocab_size} MOTOR tasks but the model was trained "
            f"with {config.motor_tte_vocab_size}; use the tokenizer the model was trained with"
        )
    names = load_concept_names(args.tokenizer_path or args.model_path)

    if args.list_tasks:
        needle = (args.search or "").lower()
        for index, code, name in task_table(tokenizer, names):
            if not needle or needle in name.lower() or needle == code:
                print(f"{code}\t{name}")
        return {}

    outcome_input = list(args.outcome_concepts or [])
    if args.outcome_concepts_file:
        with open(args.outcome_concepts_file) as f:
            outcome_input += [line.strip() for line in f if line.strip()]
    outcome = list(dict.fromkeys(outcome_input))
    outcome_resolution: Dict[str, List[str]] = {}
    if outcome and args.expand_descendants:
        ancestors_of = tokenizer_ancestors(tokenizer)
        task_codes = sorted(tokenizer._motor_time_to_event_codes)
        for concept in outcome:
            outcome_resolution[concept] = expand_to_tasks([concept], task_codes, ancestors_of)
            if not outcome_resolution[concept]:
                raise ValueError(
                    f"No MOTOR task is the concept {concept} or one of its descendants, so it "
                    "cannot be predicted"
                )
        # across the whole set, again only the highest tasks: one concept's task may contain another's
        outcome = expand_to_tasks(outcome, task_codes, ancestors_of)
        LOG.info(
            "The outcome concepts %s (with descendants) are covered by %s MOTOR task(s): %s",
            outcome_input, len(outcome), [(c, names.get(c, "")) for c in outcome],
        )
    concepts = list(args.concepts or [])
    if args.concepts_file:
        with open(args.concepts_file) as f:
            concepts += [line.strip() for line in f if line.strip()]
    concepts += [c for c in outcome if c not in concepts]
    if not concepts:
        raise ValueError("Give the concepts to predict with --outcome_concepts, --concepts or --concepts_file")
    windows = [parse_window(w) for w in args.windows]
    outcome_window = parse_window(args.outcome_window) if args.outcome_window else windows[0]
    if outcome and outcome_window not in windows:
        windows.append(outcome_window)
    tasks = resolve_tasks(tokenizer, names, concepts)
    task_indices = [index for index, _, _ in tasks]
    outcome_columns = [concepts.index(c) for c in outcome]
    outcome_at = windows.index(outcome_window) if outcome else None

    os.makedirs(args.output_dir, exist_ok=True)
    predictor = MotorPredictor(head, bins, args.device)
    extra = [args.label_column, args.gap_days_column] if args.gap_days_column else [args.label_column]
    writer = None
    outcome_probabilities: List[np.ndarray] = []
    outcome_labels: List[np.ndarray] = []
    outcome_subjects: List[np.ndarray] = []
    outcome_times: List[np.ndarray] = []
    outcome_features: List[np.ndarray] = []
    total = 0
    for batch in iter_feature_batches(args.features_folder, args.batch_size, extra):
        features = batch["features"]
        if features.shape[1] != config.n_embd:
            raise ValueError(
                f"The features are {features.shape[1]}-dimensional but the model's hidden size is "
                f"{config.n_embd}. If they are twice as wide, they were extracted with the global and "
                "local states combined; re-run the extraction with --disable_combine_global_local."
            )
        gap = torch.from_numpy(np.array(batch[args.gap_days_column], dtype=np.float32)) if args.gap_days_column else None
        if gap is None and args.gap_days:
            gap = torch.full((features.shape[0],), float(args.gap_days))
        probabilities = predictor.probabilities(torch.from_numpy(features), task_indices, windows, gap).cpu().numpy()
        labels = batch.get(args.label_column)
        rows = features.shape[0]
        n_windows, n_tasks = len(windows), len(tasks)
        # long format: one row per (patient, window, task), patient major
        concept_ids = np.array([c for _, c, _ in tasks], dtype=object)
        concept_names = np.array([n for _, _, n in tasks], dtype=object)
        window_names = np.array([window_name(w) for w in windows], dtype=object)
        columns = {
            "subject_id": pa.array(np.repeat(batch["subject_id"], n_windows * n_tasks).astype(np.int64)),
            "prediction_time_posix": pa.array(np.repeat(batch["prediction_time_posix"], n_windows * n_tasks).astype(np.float64)),
            "concept_id": pa.array(np.tile(concept_ids, rows * n_windows).tolist(), pa.string()),
            "concept_name": pa.array(np.tile(concept_names, rows * n_windows).tolist(), pa.string()),
            "window_days": pa.array(np.tile(np.repeat(window_names, n_tasks), rows).tolist(), pa.string()),
            "probability": pa.array(probabilities.reshape(-1).astype(np.float32)),
        }
        if labels is not None:
            columns["label"] = pa.array(np.repeat(labels.astype(np.int8), n_windows * n_tasks))
        out_table = pa.table(columns)
        if writer is None:
            writer = pq.ParquetWriter(os.path.join(args.output_dir, "predictions.parquet"), out_table.schema)
        writer.write_table(out_table)
        total += rows
        if outcome:
            if labels is None:
                raise ValueError(
                    f"--outcome_concepts needs the label column {args.label_column!r} in the features"
                )
            outcome_probabilities.append(probabilities[:, outcome_at, outcome_columns])
            outcome_labels.append(labels)
            outcome_subjects.append(batch["subject_id"])
            outcome_times.append(batch["prediction_time_posix"])
            if args.baseline_train_features:
                outcome_features.append(features)
    if writer is not None:
        writer.close()
    metrics: Dict[str, object] = {
        "num_patients": total,
        "concepts": concepts,
        "windows": [window_name(w) for w in windows],
        "model_path": args.model_path,
    }
    if outcome:
        probabilities_of_outcome = np.concatenate(outcome_probabilities)
        labels_of_outcome = np.concatenate(outcome_labels)
        combined = combine_probabilities(probabilities_of_outcome, args.combine)
        pq.write_table(
            pa.table(
                {
                    "subject_id": pa.array(np.concatenate(outcome_subjects).astype(np.int64)),
                    "prediction_time_posix": pa.array(np.concatenate(outcome_times).astype(np.float64)),
                    "probability": pa.array(combined.astype(np.float32)),
                    "label": pa.array(labels_of_outcome.astype(np.int8)),
                }
            ),
            os.path.join(args.output_dir, "outcome_scores.parquet"),
        )
        outcome_metrics: Dict[str, object] = {
            "concepts": outcome,
            "outcome_input": outcome_input,
            "tasks_per_input_concept": outcome_resolution or None,
            "window_days": window_name(outcome_window),
            "combine": args.combine if len(outcome) > 1 else "single task",
            "zero_shot": evaluate_predictions(labels_of_outcome, combined),
            "per_concept": {
                c: evaluate_predictions(labels_of_outcome, probabilities_of_outcome[:, i])
                for i, c in enumerate(outcome)
            },
        }
        if args.baseline_train_features:
            outcome_metrics["supervised_baseline"] = supervised_baseline(
                args.baseline_train_features, args.label_column,
                np.concatenate(outcome_features), labels_of_outcome, args.baseline_max_rows,
            )
        metrics["outcome"] = outcome_metrics
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    LOG.info("%s", json.dumps(metrics, indent=2))
    return metrics


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="Pretrained CEHR-GPT folder with a MOTOR head")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer folder (default: model_path)")
    parser.add_argument("--features_folder", help="Output of compute_cehrgpt_features (parquet with subject_id, prediction_time_posix, features); extracted with --disable_combine_global_local")
    parser.add_argument("--output_dir", help="Where predictions.parquet and metrics.json are written")
    parser.add_argument("--concepts", nargs="+", help="Concept ids of the MOTOR tasks to predict")
    parser.add_argument("--concepts_file", help="A file with one concept id per line")
    parser.add_argument("--windows", nargs="+", default=["30", "90", "365"], help='Days after the prediction time: "90" is 0-90, "30-90" is days 30 to 90')
    parser.add_argument("--gap_days", type=float, default=0.0, help="Days between the last event and the prediction time, for every patient")
    parser.add_argument("--gap_days_column", default=None, help="A column with the gap per patient (overrides --gap_days)")
    parser.add_argument("--label_column", default="boolean_value", help="A column with the observed outcome, written next to the predictions and used by --outcome_concepts")
    parser.add_argument("--outcome_concepts", nargs="+", default=None, help="MOTOR task(s) that define the binary outcome of the cohort; the label column is the truth")
    parser.add_argument("--outcome_concepts_file", default=None, help="A file with one outcome concept id per line (a concept set)")
    parser.add_argument("--expand_descendants", action="store_true", help="Treat each outcome concept as the concept and all its descendants: it is covered by the MOTOR tasks at or below it (only the highest ones are used). Needs the ontology in the tokenizer folder")
    parser.add_argument("--outcome_window", default=None, help='The cohort horizon, e.g. "30" or "30-90" (default: the first of --windows)')
    parser.add_argument("--combine", choices=["independent_or", "max"], default="independent_or", help="How the probabilities of several outcome concepts become the probability of any of them")
    parser.add_argument("--baseline_train_features", default=None, help="Features of the training split: a logistic regression is fitted on them and evaluated on the same rows, to compare with the zero-shot result")
    parser.add_argument("--baseline_max_rows", type=int, default=200_000, help="Rows of the training features used by the baseline")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--list_tasks", action="store_true", help="Print the MOTOR tasks (concept id and name) and exit")
    parser.add_argument("--search", default=None, help="With --list_tasks: only tasks whose name contains this text")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = create_arg_parser().parse_args(argv)
    if not args.list_tasks and not (args.features_folder and args.output_dir):
        raise SystemExit("--features_folder and --output_dir are required (or use --list_tasks)")
    run(args)


if __name__ == "__main__":
    main()
