"""
Train the patient query model on a dataset built by cehrgpt.qa.build_query_dataset.

The model answers yes/no queries (a concept with its descendants, and a time window after
the index date) from the linear probing hidden state of a patient. The run holds out patients
and whole concepts, and reports AUROC separately for the MOTOR tasks and for everything else.
"""

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

from cehrgpt.models.patient_query_model import PatientQueryModel, PatientQueryModelConfig
from cehrgpt.qa.build_query_dataset import QUERY_VOCABULARY_FILE
from cehrgpt.qa.concept_embeddings import (
    build_concept_embedding_matrix,
    load_cehrgpt_token_vectors,
)
from cehrgpt.qa.concept_hierarchy import ConceptHierarchy
from cehrgpt.qa.query_data import QueryCollator, evaluate_queries, split_queries

LOG = logging.getLogger(__name__)


@dataclass
class QueryRunnerArguments:
    query_dataset_path: str = field(
        metadata={"help": "Output folder of cehrgpt.qa.build_query_dataset"}
    )
    tokenizer_path: str = field(
        metadata={"help": "Tokenizer folder; its ontology gives the concept hierarchy"}
    )
    lm_type: str = field(default="gpt2", metadata={"help": "Trunk architecture: gpt2 or llama"})
    hidden_size: int = field(default=256)
    num_hidden_layers: int = field(default=4)
    num_attention_heads: int = field(default=4)
    num_prefix_tokens: int = field(
        default=1, metadata={"help": "Soft tokens the patient vector is expanded into"}
    )
    cehrgpt_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained CEHR-GPT folder; its token embeddings start the concept "
            "embeddings, so related concepts begin close together"
        },
    )
    freeze_concept_embeddings: bool = field(default=False)
    validation_patient_fraction: float = field(default=0.1)
    held_out_concept_fraction: float = field(
        default=0.1,
        metadata={
            "help": "Share of query concepts never trained on (with their ancestors and "
            "descendants), used to test generalisation to unseen concepts"
        },
    )
    held_out_max_covered_tokens: int = field(
        default=3,
        metadata={"help": "Only concepts covering at most this many tokens are held out"},
    )
    block_ancestors_of_held_out: bool = field(
        default=False,
        metadata={
            "help": "Also remove the ancestors of held-out concepts from training, a stricter "
            "test; by default only their descendants are removed"
        },
    )
    max_train_rows: int = field(default=0, metadata={"help": "0 = all"})


def build_model(
    args: QueryRunnerArguments,
    vocabulary: list,
    feature_dim: int,
    hierarchy: ConceptHierarchy,
) -> PatientQueryModel:
    embedding_matrix = None
    if args.cehrgpt_model_path:
        vectors = load_cehrgpt_token_vectors(args.cehrgpt_model_path, vocabulary)
        embedding_matrix, found = build_concept_embedding_matrix(vocabulary, hierarchy, vectors)
        LOG.info("%.1f%% of the query concepts start from CEHR-GPT vectors", 100 * found)
    config = PatientQueryModelConfig(
        lm_type=args.lm_type,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        patient_feature_dim=feature_dim,
        num_prefix_tokens=args.num_prefix_tokens,
        query_concept_ids=vocabulary,
        concept_embedding_dim=embedding_matrix.shape[1] if embedding_matrix is not None else None,
    )
    model = PatientQueryModel(config)
    if embedding_matrix is not None:
        model.init_concept_embeddings(embedding_matrix, freeze=args.freeze_concept_embeddings)
    return model


def run(args: QueryRunnerArguments, training_args: TrainingArguments) -> dict:
    set_seed(training_args.seed)
    dataset = load_from_disk(args.query_dataset_path)
    with open(os.path.join(args.query_dataset_path, QUERY_VOCABULARY_FILE)) as f:
        vocabulary_file = json.load(f)
    vocabulary = vocabulary_file["concept_ids"]
    hierarchy = ConceptHierarchy.from_tokenizer_folder(args.tokenizer_path)
    parts = split_queries(
        dataset,
        hierarchy,
        vocabulary,
        args.validation_patient_fraction,
        args.held_out_concept_fraction,
        training_args.seed,
        covered_tokens=vocabulary_file.get("covered_tokens"),
        max_held_out_covered_tokens=args.held_out_max_covered_tokens,
        block_ancestors=args.block_ancestors_of_held_out,
    )
    LOG.info("Rows: %s", {k: len(v) for k, v in parts.items()})
    if not len(parts["train"]):
        raise ValueError("The training split is empty: check the split fractions")
    train = parts["train"]
    if args.max_train_rows:
        train = train.select(range(min(args.max_train_rows, len(train))))

    model = build_model(args, vocabulary, len(dataset[0]["patient_features"]), hierarchy)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=parts["validation"] if len(parts["validation"]) else None,
        data_collator=QueryCollator(model),
    )
    trainer.train()
    trainer.save_model()

    report = {}
    for name in ("train", "validation", "test_unseen_concepts"):
        part = parts[name] if name != "train" else train.select(range(min(len(train), 20000)))
        report[name] = evaluate_queries(model, part) if len(part) else None
    with open(os.path.join(training_args.output_dir, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)
    LOG.info("Evaluation: %s", json.dumps(report, indent=2))
    return report


def main():
    parser = HfArgumentParser((QueryRunnerArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    run(args, training_args)


if __name__ == "__main__":
    main()
