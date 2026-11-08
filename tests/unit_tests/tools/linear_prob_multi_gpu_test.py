import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

from cehrgpt.models.special_tokens import LINEAR_PROB_TOKEN, RANDOM_TOKEN
from cehrgpt.tools.linear_prob import compute_cehrgpt_features as cf


def make_dataset(n, offset=0):
    return Dataset.from_dict(
        {
            "person_id": list(range(offset, offset + n)),
            "num_of_concepts": [10 + (i % 7) for i in range(n)],
        }
    )


def make_processed(with_validation=True):
    splits = {"train": make_dataset(11), "test": make_dataset(5, offset=1000)}
    if with_validation:
        splits["validation"] = make_dataset(4, offset=500)
    return DatasetDict(splits)


@pytest.mark.parametrize("with_validation", [True, False])
@pytest.mark.parametrize("num_shards", [1, 2, 3])
def test_shards_are_disjoint_and_cover_everything(tmp_path, with_validation, num_shards):
    processed = make_processed(with_validation)
    paths = cf.shard_processed_dataset(processed, num_shards, str(tmp_path))
    assert [os.path.basename(p) for p in paths] == [f"shard_{i}" for i in range(num_shards)]
    shards = [load_from_disk(p) for p in paths]
    for shard in shards:
        assert set(shard.keys()) == {"train", "test"}

    expected_train = set(processed["train"]["person_id"]) | (
        set(processed["validation"]["person_id"]) if with_validation else set()
    )
    train_ids = [i for shard in shards for i in shard["train"]["person_id"]]
    test_ids = [i for shard in shards for i in shard["test"]["person_id"]]
    assert sorted(train_ids) == sorted(expected_train)  # no duplicates, nothing lost
    assert sorted(test_ids) == sorted(processed["test"]["person_id"])


def test_shards_are_round_robin_and_balanced(tmp_path):
    paths = cf.shard_processed_dataset(make_processed(False), 3, str(tmp_path))
    first = load_from_disk(paths[0])["train"]["person_id"]
    assert first == [0, 3, 6, 9]
    sizes = [len(load_from_disk(p)["train"]) for p in paths]
    assert max(sizes) - min(sizes) <= 1


@pytest.fixture
def parent_setup(tmp_path, monkeypatch):
    processed = make_processed()
    monkeypatch.setattr(
        cf,
        "prepare_processed_dataset",
        lambda *a, **k: (processed, "tokenizer-obj", "/resolved/tokenizer"),
    )
    tokenizer = mock.MagicMock()
    tokenizer.get_vocab.return_value = {}
    monkeypatch.setattr(cf.CehrGptTokenizer, "from_pretrained", lambda *_: tokenizer)
    calls = {}

    def fake_run_workers(**kwargs):
        calls.update(kwargs)
        calls["shards_existed"] = all(os.path.isdir(p) for p in kwargs["shard_paths"])
        n = len(kwargs["shard_paths"])
        return [calls.get("exit_code", 0)] * n, [f"log{i}" for i in range(n)]

    monkeypatch.setattr(cf, "run_workers", fake_run_workers)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--gpu_ids",
            "0,1",
            "--workers_per_gpu=2",
            "--tokenizer_name_or_path",
            "/original/tokenizer",
            "--output_dir",
            str(tmp_path),
        ],
    )
    args = (
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(tokenizer_name_or_path="/original/tokenizer"),
        SimpleNamespace(output_dir=str(tmp_path), dataloader_num_workers=0),
    )
    return args, calls, tokenizer, tmp_path


def test_parent_launches_one_worker_per_device_with_clean_argv(parent_setup):
    args, calls, tokenizer, tmp_path = parent_setup
    assert cf.run_on_multiple_gpus(*args, cuda_devices=["0", "1", "0", "1"]) == 0
    assert calls["module"] == "cehrgpt.tools.linear_prob.compute_cehrgpt_features"
    assert calls["shard_option"] == "--shard_dataset_path"
    assert calls["cuda_devices"] == ["0", "1", "0", "1"]
    assert len(calls["shard_paths"]) == 4 and calls["shards_existed"]
    argv = calls["worker_argv"]
    assert "--gpu_ids" not in argv and not any(a.startswith("--workers_per_gpu") for a in argv)
    # the tokenizer the dataset was tokenized with replaces the original one
    assert argv.count("--tokenizer_name_or_path") == 1
    assert argv[argv.index("--tokenizer_name_or_path") + 1] == "/resolved/tokenizer"
    assert [c.args[0] for c in tokenizer.add_tokens.call_args_list] == [
        LINEAR_PROB_TOKEN,
        RANDOM_TOKEN,
    ]
    # shards are cleaned up after success
    assert not (tmp_path / "temp" / "shards").exists()


def test_parent_keeps_shards_and_fails_when_a_worker_fails(parent_setup):
    args, calls, _, tmp_path = parent_setup
    calls["exit_code"] = 3
    assert cf.run_on_multiple_gpus(*args, cuda_devices=["0", "1"]) == 1
    assert (tmp_path / "temp" / "shards" / "shard_0").exists()


def test_parent_caps_workers_by_dataset_size(parent_setup, monkeypatch):
    args, calls, _, _ = parent_setup
    tiny = DatasetDict({"train": make_dataset(2), "validation": make_dataset(0), "test": make_dataset(1)})
    monkeypatch.setattr(
        cf, "prepare_processed_dataset", lambda *a, **k: (tiny, None, "/t")
    )
    cf.run_on_multiple_gpus(*args, cuda_devices=["0", "1", "2", "3"])
    assert calls["cuda_devices"] == ["0"]  # 1 test sample -> 1 worker
    assert len(calls["shard_paths"]) == 1


def test_parent_discards_stale_shards(parent_setup):
    args, calls, _, tmp_path = parent_setup
    stale = tmp_path / "temp" / "shards" / "shard_9"
    stale.mkdir(parents=True)
    cf.run_on_multiple_gpus(*args, cuda_devices=["0", "1"])
    assert not stale.exists()


class StopMain(Exception):
    pass


def run_main(monkeypatch, **cehrgpt_kwargs):
    cehrgpt_args = SimpleNamespace(
        gpu_ids=None, workers_per_gpu=1, shard_dataset_path=None, **cehrgpt_kwargs
    )
    monkeypatch.setattr(
        cf,
        "parse_runner_args",
        lambda: (cehrgpt_args, SimpleNamespace(), SimpleNamespace(), SimpleNamespace()),
    )
    return cehrgpt_args


def test_main_without_gpu_options_does_not_shard(monkeypatch):
    run_main(monkeypatch)
    monkeypatch.setattr(cf, "run_on_multiple_gpus", mock.Mock(side_effect=AssertionError))
    monkeypatch.setattr(cf.torch.cuda, "is_available", mock.Mock(side_effect=StopMain))
    with pytest.raises(StopMain):
        cf.main()


def test_main_with_several_workers_runs_the_parent_and_returns(monkeypatch):
    args = run_main(monkeypatch)
    args.gpu_ids, args.workers_per_gpu = "0,1", 2
    monkeypatch.setattr(cf, "resolve_gpu_ids", lambda ids: ["0", "1"])
    parent = mock.Mock(return_value=0)
    monkeypatch.setattr(cf, "run_on_multiple_gpus", parent)
    monkeypatch.setattr(cf.torch.cuda, "is_available", mock.Mock(side_effect=AssertionError))
    cf.main()
    assert parent.call_args.args[-1] == ["0", "1", "0", "1"]


def test_main_exits_nonzero_when_the_parent_fails(monkeypatch):
    args = run_main(monkeypatch)
    args.gpu_ids = "0,1"
    monkeypatch.setattr(cf, "resolve_gpu_ids", lambda ids: ["0", "1"])
    monkeypatch.setattr(cf, "run_on_multiple_gpus", mock.Mock(return_value=1))
    with pytest.raises(SystemExit) as exit_info:
        cf.main()
    assert exit_info.value.code == 1


def test_main_pins_a_single_requested_gpu(monkeypatch):
    args = run_main(monkeypatch)
    args.gpu_ids = "1"
    monkeypatch.setattr(cf, "resolve_gpu_ids", lambda ids: ["5"])
    monkeypatch.setattr(cf.torch.cuda, "is_available", mock.Mock(side_effect=StopMain))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unset-by-test")
    with pytest.raises(StopMain):
        cf.main()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"


def test_main_in_a_shard_worker_never_spawns_workers(monkeypatch):
    args = run_main(monkeypatch)
    args.gpu_ids, args.workers_per_gpu, args.shard_dataset_path = "0,1", 2, "/shard"
    monkeypatch.setattr(cf, "run_on_multiple_gpus", mock.Mock(side_effect=AssertionError))
    monkeypatch.setattr(cf.torch.cuda, "is_available", mock.Mock(side_effect=StopMain))
    with pytest.raises(StopMain):
        cf.main()
