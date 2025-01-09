import argparse
from sklearn.preprocessing import normalize
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import tqdm
from typing import List, Generator
import numpy as np
import os
import pickle
from cehrgpt.models.pretrained_embeddings import PRETRAINED_EMBEDDING_VECTOR_FILE_NAME, PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME


def generate_embeddings_batch(texts, tokenizer, device, model):
    input = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input = {k: v.to(device) for k, v in input.items()}

    with torch.no_grad():
        outputs = model(**input)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return normalize(embeddings)


def create_batches(texts: List[str], batch_size: int) -> Generator[List[str], None, None]:
    """Generate batches of texts."""
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_1.5B_v5", trust_remote_code=True)
    model = AutoModel.from_pretrained("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    vocab = pd.read_json(args.vocab_json_file_path, typ='series').reset_index()
    vocab.columns = ['concept_id', 'idx']

    vocab.drop_duplicates(subset=['concept_id'], inplace=True)
    vocab = vocab.astype(str)

    concept = pd.read_parquet(args.concept_parquet_file_path)
    concept = concept.astype(str)

    vocab_with_name = vocab.merge(concept, how='inner', left_on='concept_id', right_on='concept_id')

    concept_names = vocab_with_name['concept_name'].to_list()
    data_with_name = vocab_with_name[['concept_id', 'concept_name']]

    total_batches = (len(concept_names) + args.batch_size - 1) // args.batch_size

    all_embeddings = []

    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for batch in create_batches(concept_names, args.batch_size):
            try:
                batch_embeddings = generate_embeddings_batch(batch, tokenizer, device, model)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error processing batch: {str(e)}")

    concept_dict = data_with_name.to_dict('records')

    np.save(os.path.join(args.output_folder_path, PRETRAINED_EMBEDDING_VECTOR_FILE_NAME), all_embeddings)

    with open(os.path.join(args.output_folder_path, PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME), 'wb') as file:
        pickle.dump(concept_dict, file)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Create pretrained embeddings')
    parser.add_argument(
        "--vocab_json_file_path",
        dest="vocab_json_file_path",
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