import inspect
import json
import os
import time
from types import SimpleNamespace
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from transformers.generation.utils import GenerationMixin

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.tools import evaluate_diagnosis_rollouts as evaluation
from cehrgpt.tools.evaluate_diagnosis_rollouts import (
    bootstrap_c_index,
    create_arg_parser,
    eligible_visit_cutoffs,
    find_observed_diagnosis,
    generate_diagnosis_rollouts,
    harrell_c_index,
    load_concept_name,
    load_condition_tokens,
    resolve_device,
    resolve_gpu_ids,
    strip_cli_option,
    TrajectoryParquetWriter,
    truncate_prefix,
)


def test_sampling_signature_matches_pinned_transformers():
    cehrgpt_parameters = list(
        inspect.signature(CEHRGPT2LMHeadModel._sample).parameters
    )
    transformers_parameters = list(
        inspect.signature(GenerationMixin._sample).parameters
    )

    assert cehrgpt_parameters == transformers_parameters


def test_sampling_arguments_accept_aliases():
    args = create_arg_parser().parse_args(
        [
            "--model",
            "model",
            "--sequences",
            "sequences",
            "--concept",
            "concept",
            "--output",
            "output.json",
            "--top-p",
            "0.9",
            "--top-k",
            "50",
            "--temp",
            "0.7",
        ]
    )

    assert args.top_p == 0.9
    assert args.top_k == 50
    assert args.temperature == 0.7
    assert args.min_history_visits == 2
    assert args.diagnosis_concept_id is None


def test_diagnosis_arguments_accept_aliases():
    args = create_arg_parser().parse_args(
        [
            "--model",
            "model",
            "--sequences",
            "sequences",
            "--concept",
            "concept",
            "--output",
            "output.json",
            "--diagnosis_concept_id",
            "100",
            "--concept_ancestor",
            "concept_ancestor",
        ]
    )

    assert args.diagnosis_concept_id == 100
    assert args.concept_ancestor == "concept_ancestor"


def test_load_condition_tokens_expands_diagnosis_descendants(tmp_path):
    concept_path = tmp_path / "concept"
    concept_ancestor_path = tmp_path / "concept_ancestor"
    concept_path.mkdir()
    concept_ancestor_path.mkdir()
    pq.write_table(
        pa.table(
            {
                "concept_id": [100, 101, 102, 103, 200],
                "domain_id": [
                    "Condition",
                    "Condition",
                    "Drug",
                    "Condition",
                    "Condition",
                ],
            }
        ),
        concept_path / "part.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "ancestor_concept_id": [100, 100, 999],
                "descendant_concept_id": [101, 102, 103],
            }
        ),
        concept_ancestor_path / "part.parquet",
    )
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"100": 0, "101": 1, "102": 2, "200": 3}
    )

    tokens = load_condition_tokens(
        str(concept_path),
        tokenizer,
        diagnosis_concept_id=100,
        concept_ancestor_path=str(concept_ancestor_path),
    )

    assert tokens == {"100", "101"}


def test_load_concept_name(tmp_path):
    concept_path = tmp_path / "concept"
    concept_path.mkdir()
    pq.write_table(
        pa.table(
            {
                "concept_id": [100, 101],
                "concept_name": ["Target diagnosis", "Descendant diagnosis"],
            }
        ),
        concept_path / "part.parquet",
    )

    assert load_concept_name(str(concept_path), 100) == "Target diagnosis"


def test_load_concept_name_rejects_unknown_concept(tmp_path):
    concept_path = tmp_path / "concept"
    concept_path.mkdir()
    pq.write_table(
        pa.table({"concept_id": [100], "concept_name": ["Known diagnosis"]}),
        concept_path / "part.parquet",
    )

    with pytest.raises(ValueError, match="Diagnosis concept 999 is absent"):
        load_concept_name(str(concept_path), 999)


def test_load_condition_tokens_requires_ancestor_data_for_diagnosis(tmp_path):
    tokenizer = SimpleNamespace(get_vocab=lambda: {"100": 0})

    with pytest.raises(ValueError, match="--concept-ancestor is required"):
        load_condition_tokens(
            str(tmp_path), tokenizer, diagnosis_concept_id=100
        )


def test_load_condition_tokens_accepts_tokenized_descendant(tmp_path):
    concept_path = tmp_path / "concept"
    concept_ancestor_path = tmp_path / "concept_ancestor"
    concept_path.mkdir()
    concept_ancestor_path.mkdir()
    pq.write_table(
        pa.table({"concept_id": [100, 101], "domain_id": ["Condition"] * 2}),
        concept_path / "part.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "ancestor_concept_id": [100],
                "descendant_concept_id": [101],
            }
        ),
        concept_ancestor_path / "part.parquet",
    )
    tokenizer = SimpleNamespace(get_vocab=lambda: {"101": 0})

    tokens = load_condition_tokens(
        str(concept_path),
        tokenizer,
        diagnosis_concept_id=100,
        concept_ancestor_path=str(concept_ancestor_path),
    )

    assert tokens == {"101"}


def test_load_condition_tokens_rejects_unavailable_diagnosis_family(tmp_path):
    concept_path = tmp_path / "concept"
    concept_ancestor_path = tmp_path / "concept_ancestor"
    concept_path.mkdir()
    concept_ancestor_path.mkdir()
    pq.write_table(
        pa.table({"concept_id": [100, 101], "domain_id": ["Condition"] * 2}),
        concept_path / "part.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "ancestor_concept_id": [100],
                "descendant_concept_id": [101],
            }
        ),
        concept_ancestor_path / "part.parquet",
    )
    tokenizer = SimpleNamespace(get_vocab=lambda: {"999": 0})

    with pytest.raises(
        ValueError,
        match="Diagnosis concept 100 and its Condition-domain descendants",
    ):
        load_condition_tokens(
            str(concept_path),
            tokenizer,
            diagnosis_concept_id=100,
            concept_ancestor_path=str(concept_ancestor_path),
        )


def test_harrell_c_index_is_patient_level_and_handles_censoring():
    rows = [
        {"observed_days": 5.0, "observed_event": True, "predicted_risk": 0.9},
        {"observed_days": 10.0, "observed_event": True, "predicted_risk": 0.7},
        {"observed_days": 20.0, "observed_event": False, "predicted_risk": 0.1},
    ]

    value, comparable = harrell_c_index(rows)

    assert value == 1.0
    assert comparable == 3


def test_harrell_c_index_gives_half_credit_to_risk_ties():
    rows = [
        {"observed_days": 5.0, "observed_event": True, "predicted_risk": 0.5},
        {"observed_days": 10.0, "observed_event": True, "predicted_risk": 0.5},
    ]

    value, comparable = harrell_c_index(rows)

    assert value == 0.5
    assert comparable == 1


def test_harrell_c_index_compares_event_and_censoring_at_same_time():
    rows = [
        {"observed_days": 5.0, "observed_event": True, "predicted_risk": 0.9},
        {"observed_days": 5.0, "observed_event": False, "predicted_risk": 0.1},
    ]

    value, comparable = harrell_c_index(rows)

    assert value == 1.0
    assert comparable == 1


def test_bootstrap_c_index_is_reproducible():
    rows = [
        {
            "observed_days": float(index + 1),
            "observed_event": True,
            "predicted_risk": float(10 - index),
        }
        for index in range(10)
    ]

    first = bootstrap_c_index(rows, n_boot=20, seed=42)
    second = bootstrap_c_index(rows, n_boot=20, seed=42)

    assert first == second
    assert first == (1.0, 1.0, 20)


def test_truncate_prefix_starts_at_next_complete_visit():
    prefix = ["old", "[VS]", "a", "[VE]", "[VS]", "b", "[VE]"]

    assert truncate_prefix(prefix, 4) == ["[VS]", "b", "[VE]"]
    assert truncate_prefix(prefix, 20) == prefix


def test_find_observed_diagnosis_uses_condition_tokens_in_sequence():
    concepts = ["[VS]", "history", "[VE]", "D10", "condition", "other"]
    epoch_times = [0, 0, 0, 10 * 86400, 10 * 86400, 20 * 86400]

    result = find_observed_diagnosis(
        concepts,
        epoch_times,
        cutoff_index=2,
        cutoff_time=0,
        condition_tokens={"condition"},
        horizon_days=365,
        last_observed_time=20 * 86400,
    )

    assert result == (10.0, True, "condition", 20.0)


def test_find_observed_diagnosis_censors_when_no_condition_is_in_horizon():
    result = find_observed_diagnosis(
        ["[VE]", "D400", "condition"],
        [0, 400 * 86400, 400 * 86400],
        cutoff_index=0,
        cutoff_time=0,
        condition_tokens={"condition"},
        horizon_days=365,
        last_observed_time=400 * 86400,
    )

    assert result == (365.0, False, None, 365.0)


def test_find_observed_diagnosis_censors_at_last_observed_event():
    result = find_observed_diagnosis(
        ["[VE]", "D100", "other"],
        [0, 100 * 86400, 100 * 86400],
        cutoff_index=0,
        cutoff_time=0,
        condition_tokens={"condition"},
        horizon_days=365,
        last_observed_time=100 * 86400,
    )

    assert result == (100.0, False, None, 100.0)


def test_eligible_cutoffs_require_two_completed_history_visits():
    day = 86400
    concepts = ["[VS]", "a", "[VE]", "[VS]", "b", "[VE]", "D800"]
    epoch_times = [0, 0, 0, day, day, day, 800 * day]

    cutoffs = eligible_visit_cutoffs(
        concepts,
        epoch_times,
        min_history_visits=2,
    )

    assert cutoffs == [(5, float(day), 2)]


def test_eligible_cutoffs_require_positive_followup():
    concepts = ["[VS]", "a", "[VE]", "[VS]", "b", "[VE]"]
    epoch_times = [0, 0, 0, 86400, 86400, 86400]

    assert eligible_visit_cutoffs(concepts, epoch_times, 2) == []


def test_generate_diagnosis_rollouts_parses_event_and_end_token():
    prefix = ["[VS]", "history", "[VE]"]

    class Predictor:
        generation_config = SimpleNamespace(num_return_sequences=2)
        tokenizer = SimpleNamespace(end_token="[END]")

        def simulate(self, _prefix):
            return [
                prefix + ["D10", "condition"],
                prefix + ["D30", "[END]"],
            ]

    times, events, incomplete, trajectories = generate_diagnosis_rollouts(
        Predictor(), prefix, {"condition"}, 365, 2, 1
    )

    assert times == [10.0, 365.0]
    assert events == [True, False]
    assert incomplete == 0
    assert trajectories == [
        {
            "tokens": ["D10", "condition"],
            "completed": True,
            "completion_reason": "condition",
            "outcome_event": "condition",
            "time_to_event_days": 10.0,
            "generated_elapsed_days": 10.0,
        },
        {
            "tokens": ["D30", "[END]"],
            "completed": True,
            "completion_reason": "end_token",
            "outcome_event": None,
            "time_to_event_days": 365.0,
            "generated_elapsed_days": 30.0,
        },
    ]


def test_trajectory_writer_saves_nested_records_to_parquet(tmp_path):
    writer = TrajectoryParquetWriter(tmp_path, buffer_size=1)
    writer.add(
        {
            "sample_index": 0,
            "person_id": 123,
            "cutoff_timestamp": 100.0,
            "history_visit_count": 2,
            "followup_days": 365.0,
            "prompt_tokens": ["[VS]", "[VE]"],
            "generated_trajectories": [
                {
                    "tokens": ["D10", "condition"],
                    "completed": True,
                    "completion_reason": "condition",
                    "outcome_event": "condition",
                    "time_to_event_days": 10.0,
                    "generated_elapsed_days": 10.0,
                }
            ],
        }
    )

    rows = pq.read_table(tmp_path / "part_00000.parquet").to_pylist()

    assert len(rows) == 1
    assert rows[0]["prompt_tokens"] == ["[VS]", "[VE]"]
    assert rows[0]["generated_trajectories"][0]["tokens"] == [
        "D10",
        "condition",
    ]


def test_resolve_device_accepts_cpu():
    device = resolve_device("cpu")
    assert device.type == "cpu"
    assert device.index is None


def test_resolve_gpu_ids_maps_through_visible_devices():
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3,5"}):
        assert resolve_gpu_ids("0,2") == ["1", "5"]
        assert resolve_gpu_ids("all") == ["1", "3", "5"]


def test_strip_cli_option_supports_both_spellings():
    arguments = ["--gpu-ids", "0,1", "--x", "1", "--gpu-ids=all"]
    assert strip_cli_option(arguments, "--gpu-ids") == ["--x", "1"]


def test_multiple_gpu_run_uses_disjoint_partitions_and_merges_results(tmp_path):
    selected = [
        {
            "sample_index": index,
            "person_id": 100 + index,
            "cutoff_timestamp": 0.0,
            "history_visit_count": 2,
            "followup_days": 365.0,
            "prefix": ["[VS]", "[VE]"],
            "observed_days": float(index + 1),
            "observed_event": True,
            "observed_condition_concept_id": "condition",
        }
        for index in range(6)
    ]
    args = SimpleNamespace(
        model="model",
        tokenizer=None,
        concept="concept",
        diagnosis_concept_id=None,
        concept_ancestor=None,
        sequences="sequences",
        output=str(tmp_path / "result.json"),
        patients=6,
        rollouts=2,
        horizon_days=365,
        min_history_visits=2,
        max_new_tokens=512,
        top_p=1.0,
        top_k=300,
        temperature=1.0,
        seed=42,
        bootstrap_resamples=0,
        trajectory_output=None,
        trajectory_buffer_size=100,
    )
    calls = []
    partitions = []

    def fake_popen(command, env, stdout, stderr):
        partition_path = command[command.index("--partition-input") + 1]
        output_path = command[command.index("--output") + 1]
        partition_payload = json.loads(open(partition_path).read())
        partition = partition_payload["patients"]
        assert partition_payload["condition_tokens"] == ["condition"]
        partitions.append([record["sample_index"] for record in partition])
        patients_detail = []
        for record in partition:
            patients_detail.append(
                {
                    **record,
                    "predicted_event_probability": 1.0,
                    "predicted_restricted_mean_days": record["observed_days"],
                    "predicted_risk": -record["observed_days"],
                    "valid_rollouts": 2,
                    "imputed_incomplete_rollouts": 0,
                    "discarded_incomplete_attempts": 0,
                }
            )
        with open(output_path, "w") as output_file:
            json.dump({"patients_detail": patients_detail}, output_file)
        calls.append((command, env))
        process = mock.Mock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        return process

    with mock.patch.object(
        evaluation.CehrGptTokenizer, "from_pretrained", return_value=mock.Mock()
    ), mock.patch.object(
        evaluation, "load_condition_tokens", return_value={"condition"}
    ), mock.patch.object(
        evaluation, "select_cutoffs", return_value=selected
    ), mock.patch.object(
        evaluation.subprocess, "Popen", side_effect=fake_popen
    ), mock.patch.object(
        evaluation.sys, "argv", ["program", "--gpu-ids", "0,1"]
    ):
        result = evaluation.run_on_multiple_gpus(
            args, ["2", "3"], time.monotonic()
        )

    assert result["patients"] == 6
    assert result["c_index"] == 1.0
    assert [item["sample_index"] for item in result["patients_detail"]] == list(
        range(6)
    )
    assert sorted(partitions[0] + partitions[1]) == list(range(6))
    assert set(partitions[0]).isdisjoint(partitions[1])
    assert [call[1]["CUDA_VISIBLE_DEVICES"] for call in calls] == ["2", "3"]
    assert not (tmp_path / "result.json.partitions").exists()
