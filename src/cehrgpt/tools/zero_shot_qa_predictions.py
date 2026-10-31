"""
Zero-shot answers of the patient timeline QA head: the probability that any question about the future of a
patient, given as a concept id or as free text, comes true in a time window after the prediction time.

Where zero_shot_motor_predictions can only ask about the MOTOR tasks, this head takes the question as the
embedding of its name, so a concept that was never a MOTOR task, or a text, can be asked about. Like that tool it
does not run the transformer: it reads the features that compute_cehrgpt_features writes (the global and the local
state, its default; do not use --disable_combine_global_local), loads only the weights of the head from the
checkpoint, and turns the hazards into the probability of the window (see zero_shot_motor_predictions for the
formula, and for the gap between the last event and the prediction time).

The questions are embedded by the encoder the model's question embeddings were made with. The model records it
(encoder, revision, template, probes), the tool loads that encoder and refuses to run unless it reproduces the
probes of the model, see cehrgpt.open_vocab.question_encoder. There is no option to use another encoder.

Questions: --questions "text" ..., --questions_file (one per line), or --concepts ids with --embeddings_dir, the
folder of question_embeddings, which holds the name of each concept id (question_texts.parquet).

The evaluation is automatic: the probability of each question in the window is compared with the label column of the
features (--label_column, default boolean_value; the questions are usually names for the outcome of that cohort), and
the AUROC, AUPRC and Brier score of every question are in metrics.json and the log. A question with an outcome of its
own takes it from --question_label_columns "question=column". The label is written next to each probability in
predictions.parquet. Without the label column the predictions are still written and a warning names the columns of the
features. The questions are scored one by one; they are not combined.

Example:
    python -m cehrgpt.tools.zero_shot_qa_predictions --model_path MODEL \
        --features_folder FEATURES/test_features --output_dir OUT \
        --questions "Coronary Artery Bypass Graft" "cabg"
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


def question_label_columns(args: argparse.Namespace, questions: List[str]) -> Dict[str, Optional[str]]:
    """
    The feature column that holds the label of each question: --question_label_columns, and for a
    question without one, --label_column (the label of the cohort of the features, shared by the questions).
    """
    columns: Dict[str, Optional[str]] = {q: None for q in questions}
    for entry in getattr(args, "question_label_columns", None) or []:
        question, separator, column = entry.rpartition("=")
        question, column = question.strip(), column.strip()
        if not separator or not question or not column:
            raise ValueError(f'--question_label_columns takes "question=column", got {entry!r}')
        if question not in columns:
            raise ValueError(f"--question_label_columns names {question!r}, which is not one of the questions: {questions}")
        columns[question] = column
    return {q: column or args.label_column for q, column in columns.items()}


def format_table(metrics: Dict[str, Dict[str, Optional[float]]]) -> str:
    """One line per question: n, positives, prevalence, AUROC, AUPRC, Brier score."""

    def cell(value: Optional[float], digits: int = 3) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    lines = [f"{'question':<40} {'n':>9} {'positives':>9} {'prevalence':>10} {'auroc':>7} {'auprc':>7} {'brier':>7}"]
    for question, m in metrics.items():
        lines.append(
            f"{question[:40]:<40} {m['n']:>9} {m['positives']:>9} {cell(m['prevalence'], 4):>10} "
            f"{cell(m['auroc']):>7} {cell(m['auprc']):>7} {cell(m['brier_score']):>7}"
        )
    return "\n".join(lines)


def feature_columns(folder: str) -> List[str]:
    """The columns of the first features file, except the features themselves (to name the label columns)."""
    import glob

    files = sorted(glob.glob(os.path.join(os.path.expanduser(folder), "**", "*.parquet"), recursive=True))
    if not files:
        return []
    return [c for c in pq.ParquetFile(files[0]).schema_arrow.names if c != "features"]


def run(args: argparse.Namespace, encoder=None) -> Dict[str, object]:
    """encoder: for tests, an object with describe() and encode(); it is verified like the one that is loaded."""
    predictor = OpenVocabPredictor.from_pretrained(args.model_path, args.device)
    # builds the encoder the model was trained with and verifies it; raises EmbedderMismatchError otherwise
    config_embedder = predictor.embedder
    encoder = QuestionEncoder(config_embedder, encoder, device=args.device, fp16=args.fp16, probe_similarity=args.probe_similarity)
    questions = read_questions(args)
    question_vectors = encoder.embed(questions)
    window = parse_window(args.window)

    os.makedirs(args.output_dir, exist_ok=True)
    n_questions = len(questions)
    label_columns = question_label_columns(args, questions)
    extra = sorted({c for c in label_columns.values() if c}) + ([args.gap_days_column] if args.gap_days_column else [])
    writer = None
    scored: Dict[str, List[np.ndarray]] = {q: [] for q in questions}
    labels_seen: Dict[str, List[np.ndarray]] = {q: [] for q in questions}
    total = 0
    question_names = np.array(questions, dtype=object)
    for batch in iter_feature_batches(args.features_folder, args.batch_size, extra):
        gap = torch.from_numpy(np.array(batch[args.gap_days_column], dtype=np.float32)) if args.gap_days_column else None
        if gap is None and args.gap_days:
            gap = torch.full((batch["features"].shape[0],), float(args.gap_days))
        probabilities = predictor.probabilities(torch.from_numpy(batch["features"]), question_vectors, [window], gap)[:, 0].cpu().numpy()
        rows = probabilities.shape[0]
        # one label per (patient, question); only when every question has its column in the features
        columns_of_labels = [batch.get(label_columns[q]) if label_columns[q] else None for q in questions]
        labels = None
        if all(c is not None for c in columns_of_labels):
            labels = np.stack([np.asarray(c) for c in columns_of_labels], axis=1)
        # long format, patient major: one row per (patient, question)
        columns = {
            "subject_id": pa.array(np.repeat(batch["subject_id"], n_questions).astype(np.int64)),
            "prediction_time_posix": pa.array(np.repeat(batch["prediction_time_posix"], n_questions).astype(np.float64)),
            "question": pa.array(np.tile(question_names, rows).tolist(), pa.string()),
            "probability": pa.array(probabilities.reshape(-1).astype(np.float32)),
        }
        if labels is not None:
            columns["label"] = pa.array(labels.reshape(-1).astype(np.int8))
        table = pa.table(columns)
        if writer is None:
            writer = pq.ParquetWriter(os.path.join(args.output_dir, "predictions.parquet"), table.schema)
        writer.write_table(table)
        total += rows
        for i, column in enumerate(columns_of_labels):
            if column is not None:
                scored[questions[i]].append(probabilities[:, i])
                labels_seen[questions[i]].append(np.asarray(column))
    if writer is not None:
        writer.close()
    metrics: Dict[str, object] = {
        "num_patients": total,
        "questions": questions,
        "label_columns": label_columns,
        "window_days": window_name(window),
        "model_path": args.model_path,
        "question_embedder": {
            "fingerprint": encoder.fingerprint,
            **{k: v for k, v in config_embedder.items() if k != "probes"},
        },
    }
    per_question = {
        q: evaluate_predictions(np.concatenate(labels_seen[q]), np.concatenate(scored[q])) for q in questions if scored[q]
    }
    if per_question:
        metrics["outcome"] = {"per_question": per_question}
        LOG.info("Per question, against its label column:\n%s", format_table(per_question))
    skipped = [q for q in questions if q not in per_question]
    if skipped:
        LOG.warning(
            "Not evaluated, the label column is not in the features for: %s. The columns of the features are %s; "
            "name the label column with --label_column (all questions) or --question_label_columns (one question)",
            skipped,
            feature_columns(args.features_folder),
        )
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
    parser.add_argument("--window", default="365", help='Days after the prediction time: "365" (default) is the first year, "30-90" is days 30 to 90')
    parser.add_argument("--gap_days", type=float, default=0.0, help="Days between the last event and the prediction time, for every patient")
    parser.add_argument("--gap_days_column", default=None, help="A column with the gap per patient (overrides --gap_days)")
    parser.add_argument("--label_column", default="boolean_value", help="The feature column with the observed outcome, the label of every question that has no --question_label_columns; the predictions are scored per question against it")
    parser.add_argument("--question_label_columns", nargs="+", default=None, metavar="QUESTION=COLUMN", help='The feature column with the observed outcome of one question when it differs from --label_column, e.g. "Heart failure=label_hf"')
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
