import tempfile
import unittest

import torch

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel

BINS = [0.0, 1.0, 7.0, 365.0, float("inf")]
N_EMBD, QUESTION_ROWS, QUESTION_DIM, LENGTH = 32, 20, 8, 12
PREDICTION_POSITIONS = [3, 6, 9]


def config(**overrides):
    kwargs = dict(
        vocab_size=64, n_positions=64, n_embd=N_EMBD, n_layer=2, n_head=4, n_inner=64,
        bos_token_id=0, eos_token_id=1, pad_token_id=0, include_values=False,
        use_sub_time_tokenization=False, include_ttv_prediction=False,
        include_motor_time_to_event=True, motor_tte_vocab_size=40,
        motor_num_time_pieces=4, motor_time_bins=list(BINS), linear_prob_n_layer=1,
        linear_prob_token_id=8, ve_token_id=5, vs_token_id=4, att_token_ids=[6, 7],
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        include_open_vocab=True, open_vocab_num_question_rows=QUESTION_ROWS,
        open_vocab_question_dim=QUESTION_DIM, open_vocab_loss_weight=0.5, open_vocab_residual_dim=16,
    )
    kwargs.update(overrides)
    return CEHRGPTConfig(**kwargs)


def tiny_model(**overrides):
    torch.manual_seed(0)
    model = CEHRGPT2LMHeadModel(config(**overrides))
    if model.open_vocab_head is not None:
        generator = torch.Generator().manual_seed(1)
        model.set_question_embeddings(torch.randn(QUESTION_ROWS, QUESTION_DIM, generator=generator))
    return model


def batch(with_open_vocab=True):
    generator = torch.Generator().manual_seed(2)
    input_ids = torch.randint(9, 64, (1, LENGTH), generator=generator)
    indicators = torch.zeros(1, LENGTH, dtype=torch.bool)
    indicators[0, PREDICTION_POSITIONS] = True
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones(1, LENGTH),
        "labels": input_ids.clone(),
        "motor_tte_task_indicators": indicators,
        "motor_censor_times": torch.full((1, LENGTH), 50.0),
    }
    if with_open_vocab:
        inputs.update(
            open_vocab_question_rows=torch.tensor([3, 5, 7, 11]),
            # three prediction rows, four questions: row 0 asks slots 0 and 2, row 2 asks slot 1
            open_vocab_event_rows=torch.tensor([0, 0, 2]),
            open_vocab_event_questions=torch.tensor([0, 2, 1]),
            open_vocab_event_times=torch.tensor([4.0, 12.0, 30.0]),
        )
    return inputs


class OpenVocabModelTest(unittest.TestCase):
    def test_the_loss_is_added_with_its_weight(self):
        model = tiny_model().train()
        inputs = batch()
        on = model(**inputs)
        self.assertTrue(torch.isfinite(on.open_vocab_loss))
        model.open_vocab_active = False
        off = model(**inputs)
        self.assertIsNone(off.open_vocab_loss)
        torch.testing.assert_close(on.loss - off.loss, on.open_vocab_loss * 0.5)

    def test_without_the_labels_the_head_is_not_used(self):
        model = tiny_model().train()
        out = model(**batch(with_open_vocab=False))
        self.assertIsNone(out.open_vocab_loss)
        self.assertTrue(torch.isfinite(out.loss))

    def test_gradients_reach_the_head_only_while_it_is_on(self):
        model = tiny_model().train()
        model(**batch()).loss.backward()
        self.assertGreater(model.open_vocab_head.question_bias.bias.grad.abs().sum(), 0)
        self.assertGreater(model.open_vocab_head.question_weight.weight.grad.abs().sum(), 0)
        model.zero_grad()
        model.open_vocab_active = False
        model(**batch()).loss.backward()
        for name, parameter in model.open_vocab_head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)  # in the graph, as distributed training needs
            self.assertEqual(parameter.grad.abs().sum().item(), 0.0, name)

    def test_the_question_embeddings_are_not_trained(self):
        model = tiny_model().train()
        model(**batch()).loss.backward()
        self.assertFalse(model.question_embeddings.requires_grad)
        self.assertNotIn("question_embeddings", dict(model.named_parameters()))
        self.assertIn("question_embeddings", model.state_dict())

    def test_a_checkpoint_keeps_the_head_and_the_embeddings(self):
        model = tiny_model().eval()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            loaded = CEHRGPT2LMHeadModel.from_pretrained(tmp).eval()
        self.assertTrue(torch.equal(loaded.question_embeddings, model.question_embeddings))
        inputs = batch()
        torch.testing.assert_close(loaded(**inputs).open_vocab_loss, model(**inputs).open_vocab_loss)

    def test_open_vocab_can_be_turned_on_for_a_motor_checkpoint(self):
        # pretrain MOTOR and CEHR-GPT first ...
        motor_only = tiny_model(include_open_vocab=False, open_vocab_num_question_rows=None, open_vocab_question_dim=None)
        self.assertIsNone(motor_only.open_vocab_head)
        with tempfile.TemporaryDirectory() as tmp:
            motor_only.save_pretrained(tmp)
            # ... then continue with the QA head, which is new in the checkpoint
            resumed = CEHRGPT2LMHeadModel.from_pretrained(tmp, config=config())
        head = resumed.open_vocab_head
        self.assertIsNotNone(head)
        # the head keeps its own start: zero question weights and a low hazard, not the GPT-2 initialisation
        self.assertEqual(head.question_weight.weight.abs().sum().item(), 0.0)
        self.assertTrue(torch.all(head.question_bias.bias == -10.0))
        self.assertEqual(head.weight_residual[-1].weight.abs().sum().item(), 0.0)
        # the MOTOR part is what was saved
        torch.testing.assert_close(resumed.motor_tte.task_layer.weight, motor_only.motor_tte.task_layer.weight)

    def test_embeddings_of_the_wrong_shape_are_rejected(self):
        model = tiny_model()
        with self.assertRaises(ValueError):
            model.set_question_embeddings(torch.zeros(QUESTION_ROWS + 1, QUESTION_DIM))

    def test_the_config_needs_motor_and_the_embedding_size(self):
        with self.assertRaises(RuntimeError):
            config(include_motor_time_to_event=False)
        with self.assertRaises(RuntimeError):
            config(open_vocab_num_question_rows=None)

    def test_a_model_without_open_vocab_is_unchanged(self):
        model = tiny_model(include_open_vocab=False, open_vocab_num_question_rows=None, open_vocab_question_dim=None).train()
        out = model(**batch(with_open_vocab=False))
        self.assertIsNone(out.open_vocab_loss)
        self.assertNotIn("question_embeddings", model.state_dict())


if __name__ == "__main__":
    unittest.main()
