import json
import os
import tempfile
import unittest

import numpy as np
import polars as pl
import torch
from transformers import TrainingArguments

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import setup_open_vocab
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments
from cehrgpt.runners.sample_packing_trainer import CehrGptTrainer

from .model_integration_test import QUESTION_DIM, QUESTION_ROWS, batch, tiny_model


def train(model, start_step, steps=5):
    """Train on batches that carry QA labels and return the logs of every step."""
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp, max_steps=steps, per_device_train_batch_size=1, logging_steps=1, report_to="none",
            use_cpu=True, remove_unused_columns=False, save_strategy="no", learning_rate=1e-3, seed=0,
            disable_tqdm=True,
        )
        trainer = CehrGptTrainer(
            model=model, args=args, train_dataset=[batch()] * steps,
            data_collator=lambda examples: examples[0], open_vocab_start_step=start_step,
        )
        trainer.train()
        return [log for log in trainer.state.log_history if "loss" in log], trainer


class OpenVocabStartStepTest(unittest.TestCase):
    def test_the_qa_loss_starts_at_the_given_step(self):
        logs, _ = train(tiny_model(), start_step=3)
        # the logs are for steps 1..5; the trainer's global step is 0 during the first one
        active = [log["open_vocab_active"] for log in logs]
        self.assertEqual(active, [0.0, 0.0, 0.0, 1.0, 1.0])
        self.assertEqual(["open_vocab_loss" in log for log in logs], [False, False, False, True, True])
        # MOTOR and the language model are trained from the start
        for log in logs:
            self.assertIn("token_loss", log)

    def test_zero_trains_it_from_the_first_step(self):
        logs, _ = train(tiny_model(), start_step=0, steps=3)
        self.assertEqual([log["open_vocab_active"] for log in logs], [1.0, 1.0, 1.0])
        self.assertTrue(all("open_vocab_loss" in log for log in logs))

    def test_the_head_does_not_move_before_it_starts(self):
        model = tiny_model()
        before = {n: p.detach().clone() for n, p in model.open_vocab_head.named_parameters()}
        train(model, start_step=100, steps=3)
        for name, parameter in model.open_vocab_head.named_parameters():
            torch.testing.assert_close(parameter, before[name], msg=name)

    def test_the_head_learns_after_it_starts(self):
        model = tiny_model()
        before = model.open_vocab_head.question_bias.bias.detach().clone()
        train(model, start_step=2, steps=5)
        self.assertFalse(torch.equal(model.open_vocab_head.question_bias.bias, before))

    def test_a_model_without_the_head_trains_as_before(self):
        model = tiny_model(include_open_vocab=False, open_vocab_num_question_rows=None, open_vocab_question_dim=None)
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(output_dir=tmp, max_steps=2, per_device_train_batch_size=1, logging_steps=1,
                                     report_to="none", use_cpu=True, remove_unused_columns=False, save_strategy="no",
                                     disable_tqdm=True)
            trainer = CehrGptTrainer(model=model, args=args, train_dataset=[batch(with_open_vocab=False)] * 2,
                                     data_collator=lambda examples: examples[0], open_vocab_start_step=5)
            trainer.train()
        logs = [log for log in trainer.state.log_history if "loss" in log]
        self.assertTrue(all("open_vocab_active" not in log for log in logs))


def write_assets(folder):
    """An OHDSI folder, a universe and embeddings of QUESTION_ROWS conditions; none of it involves a tokenizer."""
    ids = list(range(100, 100 + QUESTION_ROWS))
    ohdsi = os.path.join(folder, "ohdsi")
    concept = pl.DataFrame({
        "concept_id": ids, "concept_name": [f"q{i}" for i in ids], "domain_id": ["Condition"] * len(ids),
        "vocabulary_id": ["SNOMED"] * len(ids), "concept_class_id": ["Clinical Finding"] * len(ids),
    })
    ancestor = pl.DataFrame({
        "ancestor_concept_id": ids, "descendant_concept_id": ids, "min_levels_of_separation": [0] * len(ids),
    })
    for name, frame in (("concept", concept), ("concept_ancestor", ancestor)):
        os.makedirs(os.path.join(ohdsi, name))
        frame.write_parquet(os.path.join(ohdsi, name, "part-0.parquet"))
    universe_dir = os.path.join(folder, "universes")
    os.makedirs(universe_dir)
    pl.DataFrame({"concept_id": ids}).write_parquet(os.path.join(universe_dir, "condition_universe.parquet"))
    embeddings_dir = os.path.join(folder, "embeddings")
    os.makedirs(embeddings_dir)
    rng = np.random.default_rng(0)
    np.save(os.path.join(embeddings_dir, "embeddings.npy"), rng.normal(size=(QUESTION_ROWS, QUESTION_DIM)).astype(np.float32))
    pl.DataFrame({"row": list(range(QUESTION_ROWS)), "concept_id": ids,
                  "text": [f"q{i}" for i in ids]}).write_parquet(os.path.join(embeddings_dir, "question_texts.parquet"))
    with open(os.path.join(embeddings_dir, "question_embedder.json"), "w") as f:
        json.dump({"model": "test", "template": "{text}"}, f)
    return universe_dir, ohdsi, embeddings_dir


class StubTokenizer:
    """What setup_open_vocab reads of a tokenizer: its vocabulary."""

    def __init__(self, concept_ids):
        self.vocab = {"[PAD]": 0, **{str(c): i + 1 for i, c in enumerate(concept_ids)}}

    def get_vocab(self):
        return self.vocab


class SetupOpenVocabTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.universe_dir, self.ohdsi, self.embeddings_dir = write_assets(self.tmp.name)
        self.ids = list(range(100, 100 + QUESTION_ROWS))
        self.tokenizer = StubTokenizer(self.ids)
        self.tokenizer_dir = os.path.join(self.tmp.name, "tokenizer")
        os.makedirs(self.tokenizer_dir)

    def args(self, **kwargs):
        defaults = dict(
            include_open_vocab=True, open_vocab_universe_dir=self.universe_dir, vocab_dir=self.ohdsi,
            open_vocab_embeddings_dir=self.embeddings_dir, open_vocab_loss_weight=0.25, open_vocab_residual_dim=16,
        )
        defaults.update(kwargs)
        return CehrGPTArguments(**defaults)

    def setup(self, model, args=None, tokenizer=None):
        return setup_open_vocab(model, args or self.args(), tokenizer or self.tokenizer, self.tokenizer_dir)

    def motor_only(self):
        return tiny_model(include_open_vocab=False, open_vocab_num_question_rows=None, open_vocab_question_dim=None)

    def test_nothing_happens_when_it_is_off(self):
        model = self.motor_only()
        self.assertIsNone(self.setup(model, CehrGPTArguments()))
        self.assertIsNone(model.open_vocab_head)

    def test_the_head_is_added_to_a_trained_motor_model(self):
        model = self.motor_only()
        assets = self.setup(model)
        self.assertEqual(assets.num_rows, QUESTION_ROWS)
        self.assertIsNotNone(model.open_vocab_head)
        self.assertEqual(model.config.include_open_vocab, True)
        self.assertEqual(model.config.open_vocab_loss_weight, 0.25)
        self.assertEqual((model.config.open_vocab_num_question_rows, model.config.open_vocab_question_dim), (QUESTION_ROWS, QUESTION_DIM))
        np.testing.assert_allclose(model.question_embeddings.numpy(), assets.embeddings)
        # it works, and the config survives a save and load
        out = model(**batch())
        self.assertTrue(torch.isfinite(out.open_vocab_loss))
        model.save_pretrained(self.tmp.name + "/model")
        loaded = CEHRGPT2LMHeadModel.from_pretrained(self.tmp.name + "/model")
        self.assertTrue(torch.equal(loaded.question_embeddings, model.question_embeddings))

    def test_only_the_questions_of_the_vocabulary_get_embeddings(self):
        # the vocabulary has half of the concepts: the model holds the embeddings of those questions only
        half = StubTokenizer(self.ids[: QUESTION_ROWS // 2])
        model = self.motor_only()
        assets = self.setup(model, tokenizer=half)
        self.assertEqual(assets.num_rows, QUESTION_ROWS // 2)
        self.assertEqual(model.config.open_vocab_num_question_rows, QUESTION_ROWS // 2)
        self.assertEqual(tuple(model.question_embeddings.shape), (QUESTION_ROWS // 2, QUESTION_DIM))
        self.assertEqual(assets.question_concept_ids.tolist(), self.ids[: QUESTION_ROWS // 2])

    def test_the_minimum_patient_count_reads_the_statistics_of_the_tokenizer(self):
        # 1000 patients; the first five concepts have 500, the next five 10, the others 1
        shares = {str(c): (0.5 if i < 5 else 0.01 if i < 10 else 0.001) for i, c in enumerate(self.ids)}
        with open(os.path.join(self.tokenizer_dir, "cehrgpt_concept_stats.json"), "w") as f:
            json.dump(shares, f)
        assets = self.setup(self.motor_only(), self.args(open_vocab_min_patient_count=100))
        self.assertEqual(assets.question_concept_ids.tolist(), self.ids[:5])

    def test_a_model_that_has_the_head_only_gets_the_embeddings_again(self):
        model = tiny_model()
        head = model.open_vocab_head
        model.question_embeddings.zero_()
        self.setup(model)
        self.assertIs(model.open_vocab_head, head)
        self.assertGreater(model.question_embeddings.abs().sum().item(), 0)

    def test_the_paths_and_the_motor_head_are_required(self):
        with self.assertRaises(ValueError):
            self.setup(self.motor_only(), CehrGPTArguments(include_open_vocab=True))
        with self.assertRaises(ValueError):
            self.setup(self.motor_only(), self.args(vocab_dir=None))
        no_motor = tiny_model(include_open_vocab=False, include_motor_time_to_event=False,
                              open_vocab_num_question_rows=None, open_vocab_question_dim=None)
        with self.assertRaises(ValueError):
            self.setup(no_motor)

    def test_wrong_sized_embeddings_are_rejected(self):
        model = tiny_model()
        model.config.open_vocab_num_question_rows = QUESTION_ROWS + 3
        model.question_embeddings = torch.zeros(QUESTION_ROWS + 3, QUESTION_DIM)
        with self.assertRaises(ValueError):
            self.setup(model)


if __name__ == "__main__":
    unittest.main()
