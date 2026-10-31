import glob
import os
from typing import Iterable, List, Optional, Union

import pyarrow.parquet as pq
from datasets import Dataset, IterableDataset, load_dataset

# Per-token columns of the patient sequences that the tokenizer statistics, the tokenization mapping
# and the pretraining collator never read. The tokenized dataset drops them after the mapping
# (see create_cehrgpt_pretraining_dataset), but Arrow stores every one of them at a fixed width of
# 4 to 8 bytes per token, so loading them roughly doubles what is written to the datasets cache
# and what the tokenization map reads and writes.
UNUSED_PRETRAINING_COLUMNS = (
    "orders",
    "record_ranks",
    "priorities",
    "concept_orders",
    "visit_concept_orders",
    "visit_rank_orders",
    "visit_concept_ids",
    "visit_segments",
    "event_group_ids",
    "dates",
    "mlm_skip_values",
)


def load_parquet_as_dataset(
    data_folder: str,
    split: str = "train",
    streaming: bool = False,
    exclude_columns: Optional[Iterable[str]] = None,
) -> Union[Dataset, IterableDataset]:
    """
    Load the parquet files of a folder, without the columns in exclude_columns.

    Same as cehrbert's load_parquet_as_dataset, with the columns left out when the parquet files are
    converted, so they are never written to the datasets cache. Columns that are not in the files
    are ignored.
    """
    data_files = glob.glob(os.path.join(os.path.expanduser(data_folder), "*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found in {data_folder}")
    columns: Optional[List[str]] = None
    if exclude_columns:
        excluded = set(exclude_columns)
        names = pq.read_schema(sorted(data_files)[0]).names
        columns = [name for name in names if name not in excluded and name != "__index_level_0__"]
        if len(columns) == len(names):
            columns = None
    return load_dataset(
        "parquet",
        data_files=data_files,
        split=split,
        streaming=streaming,
        **({"columns": columns} if columns is not None else {}),
    )
