import unittest

import torch
from torch import nn
from transformers.utils import is_flash_attn_2_available

from cehrgpt.models.activations import RMSNorm
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.gpt2 import GPT2Block, LlamaMLP
from cehrgpt.models.hf_cehrgpt import (
    CEHRGPT2LMHeadModel,
    CEHRGPT2Model,
    create_sample_packing_attention_mask,
)
from cehrgpt.models.qwen2 import Qwen2Block
from cehrgpt.runners.gpt_runner_util import read_backbone, resolve_attn_implementation

VOCAB_SIZE = 64


def build_config(backbone: str, **overrides) -> CEHRGPTConfig:
    """A tiny CoMET-S-shaped config: the ratios match, the sizes are shrunk for speed."""
    kwargs = dict(
        backbone=backbone,
        vocab_size=VOCAB_SIZE,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=128,
        activation_function="silu" if backbone == "qwen2" else "gelu_new",
        decoder_mlp="LlamaMLP" if backbone == "qwen2" else "GPT2MLP",
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        # Keep the auxiliary objectives off; they are backbone-independent.
        include_values=False,
        use_sub_time_tokenization=False,
        include_ttv_prediction=False,
        include_motor_time_to_event=False,
        # Deterministic forward passes.
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    kwargs.update(overrides)
    return CEHRGPTConfig(**kwargs)


class TestQwen2ConfigPlumbing(unittest.TestCase):
    def test_backbone_default_is_gpt2(self):
        self.assertEqual(CEHRGPTConfig().backbone, "gpt2")

    def test_rejects_unknown_backbone(self):
        with self.assertRaises(ValueError):
            build_config("mamba")

    def test_kv_heads_default_to_mha(self):
        config = build_config("qwen2")
        self.assertEqual(config.num_key_value_heads, config.num_attention_heads)

    def test_rejects_indivisible_kv_heads(self):
        with self.assertRaises(ValueError):
            build_config("qwen2", n_head=4, num_key_value_heads=3)

    def test_config_round_trips(self):
        config = build_config("qwen2", num_key_value_heads=2, rope_theta=50000.0)
        restored = CEHRGPTConfig.from_dict(config.to_dict())
        self.assertEqual(restored.backbone, "qwen2")
        self.assertEqual(restored.num_key_value_heads, 2)
        self.assertEqual(restored.rope_theta, 50000.0)
        self.assertEqual(restored.rms_norm_eps, config.rms_norm_eps)


class TestQwen2BackboneAssembly(unittest.TestCase):
    def test_qwen2_backbone_builds_qwen2_blocks(self):
        model = CEHRGPT2Model(build_config("qwen2"))
        self.assertEqual(len(model.h), 2)
        for block in model.h:
            self.assertIsInstance(block, Qwen2Block)
            self.assertIsInstance(block.ln_1, RMSNorm)
            self.assertIsInstance(block.ln_2, RMSNorm)
            self.assertIsInstance(block.mlp, LlamaMLP)
        self.assertIsInstance(model.ln_f, RMSNorm)

        names = dict(model.named_parameters())
        self.assertIn("h.0.attn.q_proj.weight", names)
        self.assertIn("h.0.attn.o_proj.weight", names)
        self.assertIn("h.0.mlp.gate_proj.weight", names)
        self.assertNotIn("h.0.attn.c_attn.weight", names)
        # Qwen2 convention: bias on q/k/v, none on the output projection.
        self.assertIn("h.0.attn.q_proj.bias", names)
        self.assertNotIn("h.0.attn.o_proj.bias", names)

    def test_gpt2_backbone_is_unchanged(self):
        model = CEHRGPT2Model(build_config("gpt2"))
        for block in model.h:
            self.assertIsInstance(block, GPT2Block)
            self.assertIsInstance(block.ln_1, nn.LayerNorm)
        self.assertIsInstance(model.ln_f, nn.LayerNorm)
        self.assertIn("h.0.attn.c_attn.weight", dict(model.named_parameters()))

    def test_mlp_inner_dim_is_explicit(self):
        model = CEHRGPT2Model(build_config("qwen2", n_inner=128))
        self.assertEqual(model.h[0].mlp.gate_proj.out_features, 128)
        self.assertEqual(model.h[0].mlp.down_proj.in_features, 128)

    def test_update_attn_bias_still_works(self):
        # CEHRGPT2Model.update_attn_bias re-registers `bias` on every block's attention;
        # the Qwen2 attention must expose the same buffer for sample packing to widen it.
        model = CEHRGPT2Model(build_config("qwen2"))
        model.update_attn_bias(128)
        for block in model.h:
            self.assertEqual(block.attn.bias.shape[-1], 128)

    def test_residual_init_scaling_applies_to_qwen2_projections(self):
        config = build_config("qwen2", n_layer=8, initializer_range=0.02)
        model = CEHRGPT2Model(config)
        expected = config.initializer_range / (2 * config.n_layer) ** 0.5
        # o_proj/down_proj should be scaled down relative to a non-residual projection.
        self.assertLess(model.h[0].attn.o_proj.weight.std().item(), config.initializer_range)
        self.assertAlmostEqual(
            model.h[0].attn.o_proj.weight.std().item(), expected, delta=expected * 0.5
        )


class TestQwen2ForwardPass(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _inputs(self, batch_size=2, seq_len=10):
        input_ids = torch.randint(2, VOCAB_SIZE, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        ages = torch.randint(20, 80, (batch_size, seq_len))
        return input_ids, attention_mask, ages

    def test_backbone_forward_shapes(self):
        model = CEHRGPT2Model(build_config("qwen2")).eval()
        input_ids, attention_mask, ages = self._inputs()
        with torch.no_grad():
            output = model(
                input_ids, attention_mask=attention_mask, position_ids=ages
            )
        self.assertEqual(output.last_hidden_state.shape, (2, 10, 32))
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

    def test_lm_head_forward_and_loss(self):
        model = CEHRGPT2LMHeadModel(build_config("qwen2")).eval()
        input_ids, attention_mask, ages = self._inputs()
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ages=ages,
                labels=input_ids,
            )
        self.assertEqual(output.logits.shape, (2, 10, VOCAB_SIZE))
        self.assertTrue(torch.isfinite(output.loss))

    def test_ages_do_not_change_qwen2_outputs(self):
        """The Qwen2 backbone must ignore `ages` and use sequential positions."""
        model = CEHRGPT2Model(build_config("qwen2")).eval()
        input_ids, attention_mask, ages = self._inputs()
        with torch.no_grad():
            with_ages = model(
                input_ids, attention_mask=attention_mask, position_ids=ages
            ).last_hidden_state
            without_ages = model(
                input_ids, attention_mask=attention_mask, position_ids=None
            ).last_hidden_state
            other_ages = model(
                input_ids, attention_mask=attention_mask, position_ids=ages * 0 + 3
            ).last_hidden_state
        torch.testing.assert_close(with_ages, without_ages)
        torch.testing.assert_close(with_ages, other_ages)

    def test_rotary_embeddings_are_applied(self):
        """
        Rotary embeddings must make attention position-dependent.

        Fed a sequence of identical tokens, every query and key vector is identical
        *before* the rotation, so without RoPE each query would spread its attention
        perfectly uniformly over the keys it is allowed to see. Any deviation from uniform
        is therefore attributable to the rotation. Note this cannot be tested via the
        hidden states: RoPE does not touch the value vectors, so an average of identical
        values is identical regardless of the weights.
        """
        # Needs the eager path: SDPA never exposes attention weights.
        model = CEHRGPT2Model(
            build_config("qwen2", attn_implementation="eager")
        ).eval()
        seq_len = 8
        repeated = torch.full((1, seq_len), 5, dtype=torch.long)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        with torch.no_grad():
            attentions = model(
                repeated, attention_mask=attention_mask, output_attentions=True
            ).attentions

        # Last query position attends over all `seq_len` keys.
        last_row = attentions[0][0, 0, -1, :]
        uniform = torch.full_like(last_row, 1.0 / seq_len)
        self.assertFalse(torch.allclose(last_row, uniform, atol=1e-4))

    def test_rope_theta_changes_outputs(self):
        """A different rotary base must produce different hidden states."""
        input_ids = torch.randint(2, VOCAB_SIZE, (1, 8))
        attention_mask = torch.ones((1, 8), dtype=torch.long)

        torch.manual_seed(7)
        default = CEHRGPT2Model(build_config("qwen2")).eval()
        torch.manual_seed(7)
        retuned = CEHRGPT2Model(build_config("qwen2", rope_theta=500.0)).eval()

        with torch.no_grad():
            a = default(input_ids, attention_mask=attention_mask).last_hidden_state
            b = retuned(input_ids, attention_mask=attention_mask).last_hidden_state
        self.assertFalse(torch.allclose(a, b, atol=1e-5))


class TestQwen2Causality(unittest.TestCase):
    """Risk A: causality must not depend on the additive attention mask."""

    def setUp(self):
        torch.manual_seed(0)

    def _assert_causal(self, backbone: str, attn_implementation: str = "eager"):
        model = CEHRGPT2Model(
            build_config(backbone, attn_implementation=attn_implementation)
        ).eval()
        seq_len = 12
        input_ids = torch.randint(2, VOCAB_SIZE, (1, seq_len))
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        ages = torch.randint(20, 80, (1, seq_len))

        with torch.no_grad():
            baseline = model(
                input_ids, attention_mask=attention_mask, position_ids=ages
            ).last_hidden_state
            # Change the LAST token only; nothing before it may move.
            perturbed_ids = input_ids.clone()
            perturbed_ids[0, -1] = (perturbed_ids[0, -1] + 7) % VOCAB_SIZE
            perturbed = model(
                perturbed_ids, attention_mask=attention_mask, position_ids=ages
            ).last_hidden_state

        torch.testing.assert_close(
            baseline[:, :-1], perturbed[:, :-1], msg=f"{backbone} leaked future tokens"
        )
        self.assertFalse(torch.allclose(baseline[:, -1], perturbed[:, -1], atol=1e-6))

    def test_qwen2_is_causal_eager(self):
        self._assert_causal("qwen2", "eager")

    def test_qwen2_is_causal_sdpa(self):
        self._assert_causal("qwen2", "sdpa")

    def test_gpt2_is_causal(self):
        self._assert_causal("gpt2")

    def test_attention_weights_are_lower_triangular(self):
        model = CEHRGPT2Model(
            build_config("qwen2", attn_implementation="eager")
        ).eval()
        input_ids = torch.randint(2, VOCAB_SIZE, (1, 9))
        attention_mask = torch.ones((1, 9), dtype=torch.long)
        with torch.no_grad():
            output = model(
                input_ids, attention_mask=attention_mask, output_attentions=True
            )
        for layer_attentions in output.attentions:
            upper = torch.triu(layer_attentions[0, 0], diagonal=1)
            self.assertLess(upper.abs().max().item(), 1e-9)


class TestQwen2SamplePacking(unittest.TestCase):
    """Risk B: packed segments must not attend across their boundaries."""

    def setUp(self):
        torch.manual_seed(0)

    def _assert_segments_isolated(self, attn_implementation: str):
        model = CEHRGPT2Model(
            build_config("qwen2", attn_implementation=attn_implementation)
        ).eval()
        # Two patients packed into one row, separated by a single padding token.
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.long)
        input_ids = torch.randint(2, VOCAB_SIZE, (1, 9))
        input_ids[0, 4] = 0  # pad token at the separator

        with torch.no_grad():
            baseline = model(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state
            # Perturb the SECOND segment; the first segment must be untouched.
            perturbed_ids = input_ids.clone()
            perturbed_ids[0, 6] = (perturbed_ids[0, 6] + 5) % VOCAB_SIZE
            perturbed = model(
                perturbed_ids, attention_mask=attention_mask
            ).last_hidden_state

        first_segment = slice(0, 4)
        torch.testing.assert_close(
            baseline[:, first_segment],
            perturbed[:, first_segment],
            msg=f"{attn_implementation}: second packed segment leaked into the first",
        )
        self.assertTrue(torch.isfinite(baseline[:, first_segment]).all())
        self.assertTrue(torch.isfinite(baseline[:, 5:]).all())

    def test_packed_segments_are_isolated_eager(self):
        self._assert_segments_isolated("eager")

    def test_packed_segments_are_isolated_sdpa(self):
        self._assert_segments_isolated("sdpa")

    def test_sdpa_mask_has_no_fully_masked_rows(self):
        """
        Every row of the mask handed to SDPA must attend somewhere.

        Sample packing zeroes a separator position's whole row. SDPA's fused kernels
        return NaN for such rows, and that NaN then spreads to every position in later
        layers via `0 * NaN` when the separator's column is weighted. The math backend
        returns a finite average instead, so this is invisible on CPU - hence an
        invariant test on the mask itself rather than on the output.
        """
        captured = {}
        model = CEHRGPT2Model(
            build_config("qwen2", attn_implementation="sdpa")
        ).eval()
        original_forward = model.h[0].attn.forward

        def capture(hidden_states, position_ids=None, layer_past=None,
                    attention_mask=None, **kwargs):
            captured["mask"] = attention_mask
            return original_forward(
                hidden_states,
                position_ids=position_ids,
                layer_past=layer_past,
                attention_mask=attention_mask,
                **kwargs,
            )

        model.h[0].attn.forward = capture

        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.long)
        input_ids = torch.randint(2, VOCAB_SIZE, (1, 9))
        input_ids[0, 4] = 0
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)

        mask = captured["mask"]
        self.assertIsNotNone(mask, "attention mask was not passed to the attention")
        self.assertEqual(mask.dim(), 4)
        floor = torch.finfo(mask.dtype).min
        attends_somewhere = (mask > floor).any(dim=-1)
        self.assertTrue(
            attends_somewhere.all(),
            f"fully masked rows at {(~attends_somewhere).nonzero().tolist()}",
        )
        # The separator must attend only to itself, so nothing can leak into it.
        separator_row = mask[0, 0, 4]
        self.assertEqual(int((separator_row > floor).sum()), 1)
        self.assertGreater(separator_row[4], floor)
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

    def test_packing_mask_is_block_diagonal(self):
        mask = create_sample_packing_attention_mask(
            torch.tensor([[1, 1, 0, 1, 1, 1]])
        )
        expected = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
            ]
        )
        torch.testing.assert_close(mask[0], expected)


class TestSdpaMatchesEager(unittest.TestCase):
    """SDPA and eager must be numerically equivalent, mask handling included."""

    def _assert_parity(self, attention_mask, seq_len, batch_size=1):
        torch.manual_seed(3)
        eager = CEHRGPT2Model(
            build_config("qwen2", attn_implementation="eager")
        ).eval()
        torch.manual_seed(3)
        sdpa = CEHRGPT2Model(build_config("qwen2", attn_implementation="sdpa")).eval()

        input_ids = torch.randint(2, VOCAB_SIZE, (batch_size, seq_len))
        with torch.no_grad():
            a = eager(input_ids, attention_mask=attention_mask).last_hidden_state
            b = sdpa(input_ids, attention_mask=attention_mask).last_hidden_state

        # Compare only real-token positions: fully masked padding rows are -inf
        # everywhere and produce NaN in both paths (pre-existing behaviour shared with
        # the gpt2 backbone).
        real = attention_mask.to(torch.bool)
        torch.testing.assert_close(a[real], b[real], atol=1e-5, rtol=1e-5)

    def test_parity_no_padding(self):
        self._assert_parity(torch.ones((2, 10), dtype=torch.long), 10, batch_size=2)

    def test_parity_with_left_padding(self):
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
        self._assert_parity(mask, 8)

    def test_parity_with_sample_packing(self):
        mask = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.long)
        self._assert_parity(mask, 9)

    def test_sdpa_rejects_output_attentions(self):
        model = CEHRGPT2Model(
            build_config("qwen2", attn_implementation="sdpa")
        ).eval()
        input_ids = torch.randint(2, VOCAB_SIZE, (1, 6))
        with self.assertRaises(ValueError):
            model(
                input_ids,
                attention_mask=torch.ones((1, 6), dtype=torch.long),
                output_attentions=True,
            )

    def test_sdpa_kv_cache_matches_full_forward(self):
        torch.manual_seed(5)
        model = CEHRGPT2Model(
            build_config("qwen2", attn_implementation="sdpa")
        ).eval()
        seq_len = 6
        input_ids = torch.randint(2, VOCAB_SIZE, (1, seq_len))
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        with torch.no_grad():
            full = model(input_ids, attention_mask=attention_mask).last_hidden_state
            prefix = model(
                input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1],
                use_cache=True,
            )
            step = model(
                input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )
        torch.testing.assert_close(
            full[:, -1], step.last_hidden_state[:, -1], atol=1e-5, rtol=1e-5
        )


class TestQwen2Training(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_backward_reaches_all_qwen2_parameters(self):
        model = CEHRGPT2LMHeadModel(build_config("qwen2")).train()
        input_ids = torch.randint(2, VOCAB_SIZE, (2, 10))
        attention_mask = torch.ones((2, 10), dtype=torch.long)
        output = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )
        output.loss.backward()
        for name in (
            "cehrgpt.h.0.attn.q_proj.weight",
            "cehrgpt.h.0.attn.o_proj.weight",
            "cehrgpt.h.0.mlp.gate_proj.weight",
            "cehrgpt.h.0.mlp.down_proj.weight",
            "cehrgpt.h.0.ln_1.weight",
            "cehrgpt.ln_f.weight",
        ):
            parameter = dict(model.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, f"{name} received no gradient")
            self.assertTrue(torch.isfinite(parameter.grad).all(), f"{name} grad not finite")
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, f"{name} grad is zero")

    def test_gradient_checkpointing_forward(self):
        # The checkpointing branch calls the block with POSITIONAL arguments, so the
        # Qwen2Block signature order has to match GPT2Block exactly.
        model = CEHRGPT2LMHeadModel(build_config("qwen2")).train()
        model.gradient_checkpointing_enable()
        self.assertTrue(model.cehrgpt.gradient_checkpointing)
        input_ids = torch.randint(2, VOCAB_SIZE, (2, 10))
        attention_mask = torch.ones((2, 10), dtype=torch.long)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=False,
        )
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))


class TestQwen2KVCache(unittest.TestCase):
    """The legacy `(key, value)` tuple cache the generation loop assumes must work."""

    def setUp(self):
        torch.manual_seed(0)

    def test_incremental_decoding_matches_full_forward(self):
        model = CEHRGPT2Model(build_config("qwen2")).eval()
        seq_len = 6
        input_ids = torch.randint(2, VOCAB_SIZE, (1, seq_len))
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)

        with torch.no_grad():
            full = model(input_ids, attention_mask=attention_mask).last_hidden_state

            prefix = model(
                input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1],
                use_cache=True,
            )
            self.assertEqual(len(prefix.past_key_values), 2)
            self.assertEqual(prefix.past_key_values[0][0].shape[-2], seq_len - 1)

            step = model(
                input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )

        torch.testing.assert_close(
            full[:, -1], step.last_hidden_state[:, -1], atol=1e-5, rtol=1e-5
        )


class TestQkNorm(unittest.TestCase):
    """QK-norm (Qwen3) applied per head over head_dim, before the rotary embedding."""

    def setUp(self):
        torch.manual_seed(0)

    def test_absent_by_default(self):
        model = CEHRGPT2Model(build_config("qwen2"))
        self.assertFalse(model.h[0].attn.use_qk_norm)
        self.assertFalse(hasattr(model.h[0].attn, "q_norm"))
        self.assertNotIn("h.0.attn.q_norm.weight", dict(model.named_parameters()))

    def test_present_and_shaped_per_head(self):
        config = build_config("qwen2", use_qk_norm=True)
        model = CEHRGPT2Model(config)
        attention = model.h[0].attn
        self.assertIsInstance(attention.q_norm, RMSNorm)
        self.assertIsInstance(attention.k_norm, RMSNorm)
        head_dim = config.n_embd // config.n_head
        self.assertEqual(attention.q_norm.weight.shape, (head_dim,))
        self.assertEqual(attention.k_norm.weight.shape, (head_dim,))
        # Initialised to unit gain, so the norm starts as a pure rescale.
        self.assertTrue(torch.allclose(attention.q_norm.weight, torch.ones(head_dim)))

    def test_changes_outputs_and_is_not_a_no_op(self):
        input_ids = torch.randint(2, VOCAB_SIZE, (1, 10))
        attention_mask = torch.ones((1, 10), dtype=torch.long)
        torch.manual_seed(7)
        without = CEHRGPT2Model(build_config("qwen2")).eval()
        torch.manual_seed(7)
        with_norm = CEHRGPT2Model(build_config("qwen2", use_qk_norm=True)).eval()
        with torch.no_grad():
            a = without(input_ids, attention_mask=attention_mask).last_hidden_state
            b = with_norm(input_ids, attention_mask=attention_mask).last_hidden_state
        self.assertFalse(torch.allclose(a, b, atol=1e-5))
        self.assertTrue(torch.isfinite(b).all())

    def test_bounds_attention_logit_scale(self):
        """
        The point of QK-norm: query/key magnitude no longer scales the logits.

        Inflating the q/k projections by 10x must leave the attention distribution
        unchanged with QK-norm on, and change it without.
        """
        for use_qk_norm, expect_stable in ((True, True), (False, False)):
            torch.manual_seed(11)
            model = CEHRGPT2Model(
                build_config(
                    "qwen2", use_qk_norm=use_qk_norm, attn_implementation="eager"
                )
            ).eval()
            input_ids = torch.randint(2, VOCAB_SIZE, (1, 9))
            attention_mask = torch.ones((1, 9), dtype=torch.long)
            with torch.no_grad():
                before = model(
                    input_ids, attention_mask=attention_mask, output_attentions=True
                ).attentions[0]
                for block in model.h:
                    block.attn.q_proj.weight.mul_(10.0)
                    block.attn.k_proj.weight.mul_(10.0)
                after = model(
                    input_ids, attention_mask=attention_mask, output_attentions=True
                ).attentions[0]
            stable = torch.allclose(before, after, atol=1e-4)
            self.assertEqual(
                stable,
                expect_stable,
                f"use_qk_norm={use_qk_norm}: attention scale stability was {stable}",
            )

    def test_sdpa_matches_eager_with_qk_norm(self):
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.long)
        input_ids = torch.randint(2, VOCAB_SIZE, (1, 9))
        torch.manual_seed(5)
        eager = CEHRGPT2Model(
            build_config("qwen2", use_qk_norm=True, attn_implementation="eager")
        ).eval()
        torch.manual_seed(5)
        sdpa = CEHRGPT2Model(
            build_config("qwen2", use_qk_norm=True, attn_implementation="sdpa")
        ).eval()
        with torch.no_grad():
            a = eager(input_ids, attention_mask=attention_mask).last_hidden_state
            b = sdpa(input_ids, attention_mask=attention_mask).last_hidden_state
        real = attention_mask.to(torch.bool)
        torch.testing.assert_close(a[real], b[real], atol=1e-5, rtol=1e-5)

    def test_kv_cache_consistent_with_qk_norm(self):
        """Cached keys are normalised when written; a later step must not re-normalise."""
        torch.manual_seed(9)
        model = CEHRGPT2Model(
            build_config("qwen2", use_qk_norm=True, attn_implementation="eager")
        ).eval()
        seq_len = 6
        input_ids = torch.randint(2, VOCAB_SIZE, (1, seq_len))
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        with torch.no_grad():
            full = model(input_ids, attention_mask=attention_mask).last_hidden_state
            prefix = model(
                input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1],
                use_cache=True,
            )
            step = model(
                input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )
        torch.testing.assert_close(
            full[:, -1], step.last_hidden_state[:, -1], atol=1e-5, rtol=1e-5
        )

    def test_causality_preserved_with_qk_norm(self):
        model = CEHRGPT2Model(
            build_config("qwen2", use_qk_norm=True, attn_implementation="sdpa")
        ).eval()
        input_ids = torch.randint(2, VOCAB_SIZE, (1, 12))
        attention_mask = torch.ones((1, 12), dtype=torch.long)
        with torch.no_grad():
            baseline = model(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state
            perturbed_ids = input_ids.clone()
            perturbed_ids[0, -1] = (perturbed_ids[0, -1] + 7) % VOCAB_SIZE
            perturbed = model(
                perturbed_ids, attention_mask=attention_mask
            ).last_hidden_state
        torch.testing.assert_close(baseline[:, :-1], perturbed[:, :-1])

    def test_gradients_reach_qk_norm_gains(self):
        model = CEHRGPT2LMHeadModel(
            build_config("qwen2", use_qk_norm=True)
        ).train()
        input_ids = torch.randint(2, VOCAB_SIZE, (2, 10))
        attention_mask = torch.ones((2, 10), dtype=torch.long)
        model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        ).loss.backward()
        for name in ("cehrgpt.h.0.attn.q_norm.weight", "cehrgpt.h.0.attn.k_norm.weight"):
            gain = dict(model.named_parameters())[name]
            self.assertIsNotNone(gain.grad, f"{name} got no gradient")
            self.assertTrue(torch.isfinite(gain.grad).all())
            self.assertGreater(gain.grad.abs().sum().item(), 0.0)

    def test_config_round_trips(self):
        config = build_config("qwen2", use_qk_norm=True)
        restored = CEHRGPTConfig.from_dict(config.to_dict())
        self.assertTrue(restored.use_qk_norm)


class TestAttnImplementationResolution(unittest.TestCase):
    def test_qwen2_defaults_to_sdpa(self):
        self.assertEqual(resolve_attn_implementation("qwen2"), "sdpa")

    def test_gpt2_rejects_sdpa(self):
        with self.assertRaises(ValueError):
            resolve_attn_implementation("gpt2", "sdpa")

    def test_gpt2_backbone_normalises_autoselected_sdpa_to_eager(self):
        # transformers auto-selects sdpa because the class advertises _supports_sdpa;
        # the gpt2 block has no sdpa path, so the model must fall back to eager.
        model = CEHRGPT2Model(build_config("gpt2", attn_implementation="sdpa"))
        self.assertEqual(model.config._attn_implementation, "eager")

    def test_explicit_request_wins(self):
        self.assertEqual(resolve_attn_implementation("qwen2", "eager"), "eager")
        self.assertEqual(resolve_attn_implementation("gpt2", "eager"), "eager")

    def test_gpt2_default_tracks_flash_availability(self):
        expected = "flash_attention_2" if is_flash_attn_2_available() else "eager"
        self.assertEqual(resolve_attn_implementation("gpt2"), expected)

    def test_requesting_unavailable_flash_raises(self):
        if is_flash_attn_2_available():
            self.skipTest("flash attention 2 is installed in this environment")
        with self.assertRaises(ValueError):
            resolve_attn_implementation("qwen2", "flash_attention_2")

    def test_read_backbone_round_trips(self):
        import tempfile

        for backbone in ("gpt2", "qwen2"):
            with tempfile.TemporaryDirectory() as tmp:
                build_config(backbone).save_pretrained(tmp)
                self.assertEqual(read_backbone(tmp), backbone)

    def test_read_backbone_falls_back_for_missing_config(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_backbone(tmp), "gpt2")


if __name__ == "__main__":
    unittest.main()
