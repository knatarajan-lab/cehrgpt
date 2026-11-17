import glob
import os
from types import SimpleNamespace

import datasets
import pandas as pd
import pytest

from cehrgpt.generation import generate_batch_hf_gpt_sequence as gen


class RecordingBar:
    """Stands in for tqdm and records what the generation loop reports."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.total = kwargs["total"]
        self.n = 0
        self.postfix = {}
        self.closed = False
        RecordingBar.instances.append(self)

    def update(self, amount):
        self.n += amount

    def set_postfix(self, **kwargs):
        self.postfix = kwargs

    def close(self):
        self.closed = True


class FakeModel:
    generation_config = SimpleNamespace(pad_token_id=None, eos_token_id=None, bos_token_id=None)
    config = SimpleNamespace(n_positions=100)

    def eval(self):
        return self

    def to(self, _device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    end_token_id = 1

    def encode(self, prompt):
        return list(range(len(prompt)))


@pytest.fixture
def run(monkeypatch, tmp_path):
    RecordingBar.instances = []
    monkeypatch.setattr(gen, "tqdm", RecordingBar)
    monkeypatch.setattr(gen.CehrGptTokenizer, "from_pretrained", lambda *_a, **_k: FakeTokenizer())
    monkeypatch.setattr(gen.CEHRGPT2LMHeadModel, "from_pretrained", lambda *_a, **_k: FakeModel())
    monkeypatch.setattr(gen, "read_backbone", lambda *_a, **_k: "gpt2")
    monkeypatch.setattr(gen, "resolve_attn_implementation", lambda **_k: "eager")
    monkeypatch.setattr(gen, "get_cehrgpt_output_folder", lambda *_a, **_k: "run")
    monkeypatch.setattr(
        gen,
        "load_parquet_as_dataset",
        lambda *_a, **_k: datasets.Dataset.from_dict({"concept_ids": [["a", "b", "c", "d", "e", "f"]] * 20}),
    )

    def fake_batch(model, tokenizer, prompts, **_kwargs):
        n = len(prompts)
        return {"sequences": [["x", "y"]] * n, "value_indicators": [None] * n, "values": [None] * n}

    monkeypatch.setattr(gen, "generate_single_batch", fake_batch)
    monkeypatch.setattr(gen, "normalize_value", lambda ids, *_a, **_k: (ids, None, None, None, None))

    def call(num_of_patients, batch_size, buffer_size, **extra):
        args = SimpleNamespace(
            tokenizer_folder="t", model_folder="m", output_folder=str(tmp_path), demographic_data_path="d",
            drop_long_sequences=False, context_window=64, temperature=1.0, repetition_penalty=1.0,
            sampling_strategy="TopPStrategy", num_beams=1, num_beam_groups=1, epsilon_cutoff=0.0, top_p=1.0,
            top_k=100, meds_format=False, num_of_patients=num_of_patients, batch_size=batch_size,
            min_num_of_concepts=1, buffer_size=buffer_size, **extra,
        )
        gen.main(args)
        files = glob.glob(os.path.join(str(tmp_path), "run", "generated_sequences", "*.parquet"))
        return RecordingBar.instances[-1], files

    return call


def test_the_bar_counts_patients_and_stops_at_the_requested_total(run):
    # 10 patients in batches of 4 take 3 batches and generate 12
    bar, files = run(10, 4, buffer_size=100)
    assert bar.kwargs["total"] == 10
    assert bar.kwargs["unit"] == "patient"
    assert bar.n == 10
    assert bar.closed
    assert sum(len(pd.read_parquet(f)) for f in files) == 12  # the output is unchanged


def test_the_postfix_reports_the_buffer_and_the_files_written(run):
    bar, files = run(8, 2, buffer_size=4)
    # 5 batches (8 // 2 + 1) of 2 patients: the buffer is flushed every 2 batches, the rest at the end
    assert bar.postfix == {"batch": "5/5", "buffered": 0, "files": len(files)}
    assert len(files) == 3
    assert any(f.endswith("-last.parquet") for f in files)


def test_the_bar_can_be_disabled(run):
    bar, _ = run(4, 2, buffer_size=100, disable_progress_bar=True)
    assert bar.kwargs["disable"] is True
    bar, _ = run(4, 2, buffer_size=100)
    assert bar.kwargs["disable"] is False  # a caller without the option gets the bar


def test_the_argument_parser_has_the_flag():
    parser = gen.create_arg_parser()
    base = ["--num_of_patients", "1", "--demographic_data_path", "d", "--model_folder", "m",
            "--tokenizer_folder", "t", "--output_folder", "o", "--batch_size", "1", "--buffer_size", "1",
            "--context_window", "8", "--min_num_of_concepts", "1", "--sampling_strategy", "TopPStrategy"]
    assert parser.parse_args(base).disable_progress_bar is False
    assert parser.parse_args(base + ["--disable_progress_bar"]).disable_progress_bar is True
