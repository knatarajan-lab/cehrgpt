import unittest
from unittest import mock

import torch
from transformers import PreTrainedModel

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.gpt2 import GPT2FlashAttention
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel

ATTN_PDROP = 0.1
N_LAYER = 2


def build_flash_attention_model() -> CEHRGPT2LMHeadModel:
    torch.manual_seed(7)
    config = CEHRGPTConfig(
        vocab_size=24,
        n_positions=32,
        n_embd=32,
        n_layer=N_LAYER,
        n_head=4,
        n_inner=64,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        include_values=False,
        use_sub_time_tokenization=False,
        include_ttv_prediction=False,
        include_motor_time_to_event=False,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=ATTN_PDROP,
    )
    config._attn_implementation = "flash_attention_2"
    # flash-attn is not installed where the unit tests run, the tests patch its functions
    with mock.patch.object(
        PreTrainedModel,
        "_check_and_enable_flash_attn_2",
        classmethod(lambda cls, config, *args, **kwargs: config),
    ):
        return CEHRGPT2LMHeadModel(config)


def fake_flash_attn_func(query, key, value, causal=False):
    """flash_attn_func on (batch, length, heads, head_dim) tensors, computed with SDPA."""
    query, key, value = (
        states.float().transpose(1, 2) for states in (query, key, value)
    )
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=causal
    )
    return output.transpose(1, 2).to(torch.bfloat16)


class TestFlashAttentionDropout(unittest.TestCase):
    def _dropout_rates(self, training: bool, with_attention_mask: bool):
        """The dropout rates the flash-attn functions are called with by a forward pass."""
        rates = []

        def record_flash(
            query, key, value, dropout_p=0.0, softmax_scale=None, causal=False
        ):
            rates.append(dropout_p)
            return fake_flash_attn_func(query, key, value, causal=causal)

        def record_varlen(query, key, value, dropout_p=0.0, causal=False, **kwargs):
            rates.append(dropout_p)
            # One unpadded sequence, the un-padded states are (length, heads, head_dim)
            return fake_flash_attn_func(
                query[None], key[None], value[None], causal=causal
            )[0]

        model = build_flash_attention_model().train(training)
        self.assertTrue(
            all(isinstance(block.attn, GPT2FlashAttention) for block in model.cehrgpt.h)
        )
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 8]])
        attention_mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if with_attention_mask
            else None
        )
        with mock.patch(
            "cehrgpt.models.gpt2.flash_attn_func", record_flash, create=True
        ), mock.patch(
            "cehrgpt.models.gpt2.flash_attn_varlen_func", record_varlen, create=True
        ), mock.patch(
            "cehrgpt.models.gpt2.index_first_axis",
            lambda states, indices: states[indices],
            create=True,
        ), mock.patch(
            "cehrgpt.models.gpt2.pad_input",
            lambda states, indices, batch_size, length: states[None],
            create=True,
        ):
            model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
        return rates

    def test_attention_dropout_is_off_outside_training(self):
        for with_attention_mask in (True, False):
            with self.subTest(with_attention_mask=with_attention_mask):
                self.assertEqual(
                    self._dropout_rates(
                        training=False, with_attention_mask=with_attention_mask
                    ),
                    [0.0] * N_LAYER,
                )

    def test_attention_dropout_is_kept_in_training(self):
        for with_attention_mask in (True, False):
            with self.subTest(with_attention_mask=with_attention_mask):
                self.assertEqual(
                    self._dropout_rates(
                        training=True, with_attention_mask=with_attention_mask
                    ),
                    [ATTN_PDROP] * N_LAYER,
                )


if __name__ == "__main__":
    unittest.main()
