import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Trainer, TrainerControl

from cehrgpt.data.sample_packing_sampler import SamplePackingBatchSampler
from cehrgpt.runners.sample_packing_trainer import CehrGptTrainer


def make_trainer():
    trainer = object.__new__(CehrGptTrainer)
    trainer.aux_loss_start_step = 0
    trainer.aux_loss_warmup_steps = 0
    trainer._sub_loss_accumulators = {
        "train": {"sums": {}, "counts": {}},
        "eval": {"sums": {}, "counts": {}},
    }
    trainer._sub_loss_context = "train"
    trainer.args = SimpleNamespace()
    trainer.state = SimpleNamespace(global_step=10, log_history=[])
    trainer.control = TrainerControl()
    trainer.model = SimpleNamespace(
        config=SimpleNamespace(
            next_token_prediction_loss_weight=1.0,
            freeze_cehrgpt_generation_model=False,
        )
    )
    trainer.optimizer = None
    trainer.lr_scheduler = None
    return trainer


class CehrGptTrainerLossAccumulatorTest(unittest.TestCase):
    def test_training_and_evaluation_sub_losses_are_isolated(self):
        trainer = make_trainer()

        def model(**_inputs):
            return SimpleNamespace(
                loss=torch.tensor(5.0),
                token_loss=torch.tensor(2.0),
                time_token_loss=None,
                time_to_visit_loss=None,
                token_value_loss=None,
                motor_tte_loss=torch.tensor(3.0),
                age_at_vs_loss=None,
                year_at_vs_loss=None,
            )

        trainer.compute_loss(model, {})
        trainer._sub_loss_context = "eval"
        trainer._reset_sub_losses("eval")
        trainer.compute_loss(model, {})

        eval_logs = {}
        trainer._flush_sub_losses(eval_logs, "eval")
        self.assertEqual(eval_logs["token_loss"], 2.0)
        self.assertEqual(eval_logs["motor_tte_loss"], 3.0)

        train_logs = {}
        trainer._flush_sub_losses(train_logs, "train")
        self.assertEqual(train_logs["token_loss"], 2.0)
        self.assertEqual(train_logs["motor_tte_loss"], 3.0)

    def test_reset_discards_stale_evaluation_values(self):
        trainer = make_trainer()
        trainer._sub_loss_accumulators["eval"] = {
            "sums": {"motor_tte_loss": 46_733.0},
            "counts": {"motor_tte_loss": 1},
        }

        trainer._reset_sub_losses("eval")

        self.assertEqual(
            trainer._sub_loss_accumulators["eval"], {"sums": {}, "counts": {}}
        )


class CehrGptTrainerEvaluationTest(unittest.TestCase):
    def test_evaluation_runs_once_with_fresh_accumulators(self):
        trainer = make_trainer()
        trainer._sub_loss_accumulators["eval"] = {
            "sums": {"motor_tte_loss": 46_931.0},
            "counts": {"motor_tte_loss": 1},
        }

        with patch.object(
            Trainer,
            "evaluate",
            return_value={"eval_loss": 3.2},
        ) as evaluate:
            metrics = trainer.evaluate()

        evaluate.assert_called_once()
        self.assertEqual(metrics, {"eval_loss": 3.2})
        self.assertEqual(
            trainer._sub_loss_accumulators["eval"], {"sums": {}, "counts": {}}
        )
        self.assertEqual(trainer._sub_loss_context, "train")


class SamplePackingBatchSamplerTest(unittest.TestCase):
    def test_evaluation_order_is_constant_across_epochs(self):
        sampler = SamplePackingBatchSampler(
            lengths=[1, 1, 1, 1, 1],
            max_tokens_per_batch=6,
            max_position_embeddings=10,
            seed=42,
            shuffle=False,
        )

        epoch_zero = list(sampler)
        sampler.sampler.set_epoch(7)
        epoch_seven = list(sampler)

        self.assertEqual(epoch_zero, epoch_seven)
        self.assertEqual(epoch_zero, [[0, 1], [2, 3], [4]])


if __name__ == "__main__":
    unittest.main()
