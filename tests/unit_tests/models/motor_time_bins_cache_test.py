import unittest

import torch

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel

VOCAB_SIZE = 64
MOTOR_VOCAB = 40
TIME_BINS = [0.0, 1.0, 7.0, 365.0, float("inf")]


def build_model() -> CEHRGPT2LMHeadModel:
    torch.manual_seed(0)
    config = CEHRGPTConfig(
        vocab_size=VOCAB_SIZE,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=64,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        include_values=False,
        use_sub_time_tokenization=False,
        include_ttv_prediction=False,
        include_motor_time_to_event=True,
        motor_tte_vocab_size=MOTOR_VOCAB,
        motor_num_time_pieces=4,
        motor_time_bins=list(TIME_BINS),
        linear_prob_n_layer=1,
        linear_prob_token_id=8,
        ve_token_id=5,
        vs_token_id=4,
        att_token_ids=[6, 7],
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return CEHRGPT2LMHeadModel(config)


class TestMotorTimeBinsCache(unittest.TestCase):
    def test_bins_are_copied_to_the_device_once_and_reused(self):
        model = build_model()
        first = model._get_motor_time_bins(torch.device("cpu"))
        self.assertIs(model._get_motor_time_bins(torch.device("cpu")), first)
        torch.testing.assert_close(first, torch.tensor(TIME_BINS))

    def test_enabling_motor_again_resets_the_cache(self):
        model = build_model()
        first = model._get_motor_time_bins(torch.device("cpu"))
        model.config.motor_time_bins = [0.0, 2.0, 9.0, 400.0, float("inf")]
        model.enable_motor_tte()
        self.assertIsNone(model._motor_time_bins_on_device)
        refreshed = model._get_motor_time_bins(torch.device("cpu"))
        self.assertIsNot(refreshed, first)
        self.assertEqual(refreshed[3].item(), 400.0)

    def test_casting_the_model_does_not_round_the_bin_edges(self):
        model = build_model().to(torch.bfloat16)
        bins = model._get_motor_time_bins(torch.device("cpu"))
        self.assertEqual(bins.dtype, torch.float32)
        self.assertEqual(bins[3].item(), 365.0)  # 365 is not representable in bfloat16


if __name__ == "__main__":
    unittest.main()
