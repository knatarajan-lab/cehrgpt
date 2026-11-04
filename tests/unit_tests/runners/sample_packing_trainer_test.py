import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainerControl

from cehrgpt.runners.sample_packing_trainer import CehrGptTrainer


class RecordingEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self):
        super().__init__(early_stopping_patience=1)
        self.metrics = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.metrics.append(dict(metrics))
        return control


def make_trainer():
    trainer = object.__new__(CehrGptTrainer)
    trainer.aux_loss_start_step = 0
    trainer.aux_loss_warmup_steps = 0
    trainer.eval_loss_retry_factor = 100.0
    trainer._sub_loss_accumulators = {
        "train": {"sums": {}, "counts": {}},
        "eval": {"sums": {}, "counts": {}},
    }
    trainer._sub_loss_context = "train"
    trainer._last_accepted_eval_losses = {}
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


class CehrGptTrainerEvaluationRetryTest(unittest.TestCase):
    def test_implausible_loss_is_retried_before_early_stopping(self):
        trainer = make_trainer()
        trainer._last_accepted_eval_losses["eval_loss"] = 3.0
        early_stopping = RecordingEarlyStoppingCallback()
        other_callback = object()
        trainer.callback_handler = SimpleNamespace(
            callbacks=[other_callback, early_stopping]
        )

        with patch.object(
            Trainer,
            "evaluate",
            side_effect=[{"eval_loss": 46_931.0}, {"eval_loss": 3.2}],
        ) as evaluate:
            metrics = trainer.evaluate()

        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(metrics["eval_loss"], 3.2)
        self.assertEqual(metrics["eval_loss_first_attempt"], 46_931.0)
        self.assertEqual(metrics["eval_loss_retry_triggered"], 1.0)
        self.assertEqual(early_stopping.metrics, [metrics])
        self.assertEqual(
            trainer.callback_handler.callbacks, [other_callback, early_stopping]
        )
        self.assertEqual(trainer._last_accepted_eval_losses["eval_loss"], 3.2)
        self.assertEqual(trainer._sub_loss_context, "train")

    def test_stable_loss_is_not_retried(self):
        trainer = make_trainer()
        trainer._last_accepted_eval_losses["eval_loss"] = 3.0
        early_stopping = RecordingEarlyStoppingCallback()
        trainer.callback_handler = SimpleNamespace(callbacks=[early_stopping])

        with patch.object(
            Trainer, "evaluate", return_value={"eval_loss": 3.1}
        ) as evaluate:
            metrics = trainer.evaluate()

        evaluate.assert_called_once()
        self.assertEqual(metrics, {"eval_loss": 3.1})
        self.assertEqual(early_stopping.metrics, [metrics])


if __name__ == "__main__":
    unittest.main()
