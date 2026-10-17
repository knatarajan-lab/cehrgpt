"""
Zero-shot answers of the patient timeline QA head: the probability that any question about the future of a
patient, given as a concept id or as free text, comes true in a time window after the prediction time.

Where zero_shot_motor_predictions can only ask about the MOTOR tasks, this head takes the question as the
embedding of its name, so a concept that was never a MOTOR task, or a text, can be asked about. Like that tool it
does not run the transformer: it reads the features that compute_cehrgpt_features writes (the global and the local
state, its default; do not use --disable_combine_global_local), loads only the weights of the head from the
checkpoint, and turns the hazards into probabilities of the windows (see zero_shot_motor_predictions for the
formula, and for the gap between the last event and the prediction time).

The questions are embedded by the encoder the model's question embeddings were made with. The model records it
(encoder, revision, template, probes), the tool loads that encoder and refuses to run unless it reproduces the
probes of the model, see cehrgpt.open_vocab.question_encoder. There is no option to use another encoder.

Questions: --questions "text" ..., --questions_file (one per line), or --concepts ids with --embeddings_dir, the
folder of question_embeddings, which holds the name of each concept id (question_texts.parquet).

With --evaluate and a label column in the features, the probability of each question in --outcome_window is
compared with the label: AUROC, AUPRC, Brier score. The questions are scored one by one; they are not combined.

Example:
    python -m cehrgpt.tools.zero_shot_qa_predictions --model_path MODEL \
        --features_folder FEATURES/test_features --output_dir OUT \
        --questions "Type 2 diabetes mellitus" "Heart failure" --windows 30 90 365
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cehrgpt.open_vocab.predictor import OpenVocabPredictor
from cehrgpt.open_vocab.question_encoder import QuestionEncoder
from cehrgpt.tools.zero_shot_motor_predictions import (
    evaluate_predictions,
    iter_feature_batches,
    parse_window,
    window_name,
)

LOG = logging.getLogger(__name__)


def read_questions(args: argparse.Namespace) -> List[str]:
    """The question texts: --questions, --questions_file, and the names of --concepts."""
    questions = list(args.questions or [])
    if args.questions_file:
        with open(os.path.expanduser(args.questions_file)) as f:
            questions += [line.strip() for line in f if line.strip()]
    if args.concepts:
        if not args.embeddings_dir:
            raise ValueError("--concepts needs --embeddings_dir, the folder of question_embeddings, for the names")
        import polars as pl

        table = pl.read_parquet(os.path.join(os.path.expanduser(args.embeddings_dir), "question_texts.parquet"))
        names = dict(zip(table["concept_id"].to_list(), table["text"].to_list()))
        missing = [c for c in args.concepts if int(c) not in names]
        if missing:
            raise ValueError(f"Concepts that are not questions of the embeddings: {missing[:10]}")
        questions += [names[int(c)] for c in args.concepts]
    questions = list(dict.fromkeys(questions))
    if not questions:
        raise ValueError("Give the questions with --questions, --questions_file or --concepts")
    return questions


def run(args: argparse.Namespace, encoder=None) -> Dict[str, object]:
    """encoder: for tests, an object with describe() and encode(); it is verified like the one that is loaded."""
    predictor = OpenVocabPredictor.from_pretrained(args.model_path, args.device)
    # builds the encoder the model was trained with and verifies it; raises EmbedderMismatchError otherwise
    config_embedder = predictor.embedder
    encoder = QuestionEncoder(config_embedder, encoder, device=args.device, fp16=args.fp16, probe_similarity=args.probe_similarity)
    questions = read_questions(args)
    question_vectors = encoder.embed(questions)
    windows = [parse_window(w) for w in args.windows]
    outcome_window = parse_window(args.outcome_window) if args.outcome_window else windows[0]
    if args.evaluate and outcome_window not in windows:
        windows.append(outcome_window)
    outcome_at = windows.index(outcome_window)

    os.makedirs(args.output_dir, exist_ok=True)
    extra = [args.label_column] + ([args.gap_days_column] if args.gap_days_column else [])
    writer = None
    scored: List[np.ndarray] = []
    labels_seen: List[np.ndarray] = []
    total = 0
    n_windows, n_questions = len(windows), len(questions)
    question_names = np.array(questions, dtype=object)
    window_names = np.array([window_name(w) for w in windows], dtype=object)
    for batch in iter_feature_batches(args.features_folder, args.batch_size, extra):
        gap = torch.from_numpy(np.array(batch[args.gap_days_column], dtype=np.float32)) if args.gap_days_column else None
        if gap is None and args.gap_days:
            gap = torch.full((batch["features"].shape[0],), float(args.gap_days))
        probabilities = predictor.probabilities(torch.from_numpy(batch["features"]), question_vectors, windows, gap).cpu().numpy()
        rows = probabilities.shape[0]
        labels = batch.get(args.label_column)
        # long format, patient major: one row per (patient, window, question)
        columns = {
            "subject_id": pa.array(np.repeat(batch["subject_id"], n_windows * n_questions).astype(np.int64)),
            "prediction_time_posix": pa.array(np.repeat(batch["prediction_time_posix"], n_windows * n_questions).astype(np.float64)),
            "question": pa.array(np.tile(question_names, rows * n_windows).tolist(), pa.string()),
            "window_days": pa.array(np.tile(np.repeat(window_names, n_questions), rows).tolist(), pa.string()),
            "probability": pa.array(probabilities.reshape(-1).astype(np.float32)),
        }
        if labels is not None:
            columns["label"] = pa.array(np.repeat(labels.astype(np.int8), n_windows * n_questions))
        table = pa.table(columns)
        if writer is None:
            writer = pq.ParquetWriter(os.path.join(args.output_dir, "predictions.parquet"), table.schema)
        writer.write_table(table)
        total += rows
        if args.evaluate:
            if labels is None:
                raise ValueError(f"--evaluate needs the label column {args.label_column!r} in the features")
            scored.append(probabilities[:, outcome_at, :])
            labels_seen.append(labels)
    if writer is not None:
        writer.close()
    metrics: Dict[str, object] = {
        "num_patients": total,
        "questions": questions,
        "windows": [window_name(w) for w in windows],
        "model_path": args.model_path,
        "question_embedder": {
            "fingerprint": encoder.fingerprint,
            **{k: v for k, v in config_embedder.items() if k != "probes"},
        },
    }
    if args.evaluate:
        probabilities_of_outcome = np.concatenate(scored)
        labels_of_outcome = np.concatenate(labels_seen)
        metrics["outcome"] = {
            "window_days": window_name(outcome_window),
            "per_question": {
                q: evaluate_predictions(labels_of_outcome, probabilities_of_outcome[:, i])
                for i, q in enumerate(questions)
            },
        }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    LOG.info("%s", json.dumps(metrics, indent=2))
    return metrics


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="CEHR-GPT folder trained with the patient timeline QA head")
    parser.add_argument("--features_folder", required=True, help="Output of compute_cehrgpt_features with the global and local states combined (the default)")
    parser.add_argument("--output_dir", required=True, help="Where predictions.parquet and metrics.json are written")
    parser.add_argument("--questions", nargs="+", default=None, help="Questions as text, e.g. the name of a concept")
    parser.add_argument("--questions_file", default=None, help="A file with one question per line")
    parser.add_argument("--concepts", nargs="+", default=None, help="Concept ids of questions of the embeddings; needs --embeddings_dir")
    parser.add_argument("--embeddings_dir", default=None, help="The folder of question_embeddings, for the names of --concepts")
    parser.add_argument("--windows", nargs="+", default=["30", "90", "365"], help='Days after the prediction time: "90" is 0-90, "30-90" is days 30 to 90')
    parser.add_argument("--gap_days", type=float, default=0.0, help="Days between the last event and the prediction time, for every patient")
    parser.add_argument("--gap_days_column", default=None, help="A column with the gap per patient (overrides --gap_days)")
    parser.add_argument("--label_column", default="boolean_value", help="A column with the observed outcome, written next to the predictions")
    parser.add_argument("--evaluate", action="store_true", help="Compare the probability of each question in --outcome_window with the label column")
    parser.add_argument("--outcome_window", default=None, help="The window to evaluate (default: the first of --windows)")
    parser.add_argument("--probe_similarity", type=float, default=0.999, help="The lowest cosine similarity between a probe and its stored vector for the encoder to be accepted")
    parser.add_argument("--fp16", action="store_true", help="Run the question encoder in half precision on a GPU (still has to reproduce the probes)")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(create_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
