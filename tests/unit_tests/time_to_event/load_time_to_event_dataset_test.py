from types import SimpleNamespace
from unittest import mock

from datasets import Dataset, DatasetDict

from cehrgpt.time_to_event import time_to_event_prediction as tte


def make_args():
    return SimpleNamespace(
        tokenized_full_dataset_path="/tokenized",
        cohort_folder="/cohort",
        output_folder="/out",
        observation_window=365,
        allow_missing_tokenized_persons=False,
        tokenizer_folder="/tokenizer",
    )


def test_cohort_sequences_are_extracted_with_the_tokenizer():
    """extract_cohort_sequences takes the tokenizer (not a CacheFileCollector) to re-encode
    the demographics after the observation window slicing."""
    tokenizer = object()
    extracted = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"concept_ids": [["a", "b"], ["c"]], "classifier_label": [0, 1]}
            )
        }
    )
    with mock.patch.object(
        tte.CehrGptTokenizer, "from_pretrained", return_value=tokenizer
    ) as load_tokenizer, mock.patch.object(
        tte, "extract_cohort_sequences", return_value=extracted
    ) as extract:
        dataset = tte.load_time_to_event_dataset(make_args())

    load_tokenizer.assert_called_once_with("/tokenizer")
    data_args, cehrgpt_args, passed_tokenizer = extract.call_args.args
    assert passed_tokenizer is tokenizer
    assert data_args.observation_window == 365
    assert cehrgpt_args.tokenized_full_dataset_path == "/tokenized"
    assert dataset["label"] == [0, 1]
    assert dataset["num_of_concepts"] == [2, 1]
