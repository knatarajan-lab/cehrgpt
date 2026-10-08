import inspect
from types import SimpleNamespace

from transformers.generation.utils import GenerationMixin

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.tools.evaluate_diagnosis_rollouts import (
    bootstrap_c_index,
    generate_diagnosis_rollouts,
    harrell_c_index,
    resolve_device,
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

    times, events, incomplete = generate_diagnosis_rollouts(
        Predictor(), prefix, {"condition"}, 365, 2, 1
    )

    assert times == [10.0, 365.0]
    assert events == [True, False]
    assert incomplete == 0


def test_resolve_device_accepts_cpu():
    device = resolve_device("cpu")
    assert device.type == "cpu"
    assert device.index is None
