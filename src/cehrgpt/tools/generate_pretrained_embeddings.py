import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from cehrgpt.models.pretrained_embeddings import (
    PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
    PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
)
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer


def generate_embeddings_batch(texts, tokenizer, device, model):
    input = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    input = {k: v.to(device) for k, v in input.items()}

    with torch.no_grad():
        outputs = model(**input)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return normalize(embeddings)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        "dunzhang/stella_en_1.5B_v5", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "dunzhang/stella_en_1.5B_v5", trust_remote_code=True
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Load cehrgpt tokenizer")
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_path)
    concept_ids = [_ for _ in cehrgpt_tokenizer.get_vocab().keys() if _.isnumeric()]
    vocab = pd.DataFrame(concept_ids, columns=["concept_id"])
    vocab.drop_duplicates(subset=["concept_id"], inplace=True)
    vocab = vocab.astype(str)

    print("Load concept dataframe")
    concept = pd.read_parquet(args.concept_parquet_file_path)
    concept = concept.astype(str)

    print("Merge concept ids and concept names")
    vocab_with_name = vocab.merge(
        concept, how="inner", left_on="concept_id", right_on="concept_id"
    )

    concept_ids = vocab_with_name["concept_id"].to_list()
    concept_names = vocab_with_name["concept_name"].to_list()
    all_embeddings = []
    concept_dict = []
    for i in tqdm(
        range(0, (len(concept_names) + args.batch_size - 1), args.batch_size)
    ):
        batched_concept_names = concept_names[i : i + args.batch_size]
        batched_concept_ids = concept_ids[i : i + args.batch_size]
        try:
            batch_embeddings = generate_embeddings_batch(
                batched_concept_names, tokenizer, device, model
            )
            all_embeddings.extend(batch_embeddings)
            concept_dict.extend(
                [
                    {"concept_id": concept_id, "concept_name": concept_name}
                    for concept_id, concept_name in zip(
                        batched_concept_ids, batched_concept_names
                    )
                ]
            )
        except Exception as e:
            print(f"Error processing batch: {str(e)}")

    np.save(
        os.path.join(args.output_folder_path, PRETRAINED_EMBEDDING_VECTOR_FILE_NAME),
        all_embeddings,
    )

    with open(
        os.path.join(args.output_folder_path, PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME),
        "wb",
    ) as file:
        pickle.dump(concept_dict, file)


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Create pretrained embeddings")
    parser.add_argument(
        "--tokenizer_path",
        dest="tokenizer_path",
        action="store",
        help="The path for the vocabulary json file",
        required=True,
    )
    parser.add_argument(
        "--concept_parquet_file_path",
        dest="concept_parquet_file_path",
        action="store",
        help="The path for your concept_path",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=16,
        action="store",
        help="Batch size to process the concept_names",
        required=True,
    )
    parser.add_argument(
        "--output_folder_path",
        dest="output_folder_path",
        action="store",
        help="Output folder path for saving the embeddings and concept_names",
        required=True,
    )
    return parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
