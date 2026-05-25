"""
Unit tests for GPT2FlashAttention focusing on the use_local_attention routing.

When use_local_attention=True, GPT2FlashAttention bypasses flash_attn_func entirely
and routes to:
  - xformers.ops.memory_efficient_attention  (if xformers is installed)
  - torch.nn.functional.scaled_dot_product_attention  (fallback, always available)

These tests exercise the SDPA/xformers path so they run without flash-attn installed.
They are structured to be stepped through in a debugger: concrete small shapes,
descriptive variable names, and intermediate tensors are kept visible.
"""

import unittest

import torch

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.gpt2 import GPT2FlashAttention, HAS_XFORMERS

NEG_INF = torch.finfo(torch.float32).min


def _make_config(use_local_attention: bool = True, apply_rotary: bool = False) -> CEHRGPTConfig:
    """Return a minimal CEHRGPTConfig suitable for direct unit testing."""
    cfg = CEHRGPTConfig(
        vocab_size=100,
        n_embd=32,        # hidden size
        n_head=4,         # head_dim = 32 // 4 = 8
        n_layer=2,
        n_positions=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_local_attention=use_local_attention,
        apply_rotary=apply_rotary,
    )
    # Mark as flash_attention_2 so GPT2Block would pick GPT2FlashAttention.
    # We set it directly here because we instantiate the class manually.
    cfg._attn_implementation = "flash_attention_2"
    return cfg


def _causal_mask(B: int, L: int) -> torch.Tensor:
    """Build a (B, 1, L, L) additive causal mask: 0.0 allowed, NEG_INF blocked."""
    mask = torch.zeros(B, 1, L, L)
    causal_block = torch.full((L, L), NEG_INF).triu(diagonal=1)  # upper triangle
    mask[:, 0, :, :] = causal_block
    return mask


def _local_causal_mask(B: int, L: int, window: int) -> torch.Tensor:
    """
    Build a (B, 1, L, L) additive mask that is causal AND restricts each
    query to attend only to the previous `window` key positions.
    """
    mask = torch.full((B, 1, L, L), NEG_INF)
    for i in range(L):
        lo = max(0, i - window)
        hi = i + 1  # inclusive of self
        mask[:, 0, i, lo:hi] = 0.0
    return mask


class TestGPT2FlashAttentionLocalPath(unittest.TestCase):
    """
    Tests for the use_local_attention=True branch of GPT2FlashAttention.

    Neither flash_attn_func nor flash_attn_varlen_func are called here because
    use_local_attention=True routes exclusively to xformers or torch SDPA.
    """

    B = 2   # batch size
    L = 8   # sequence length
    N_EMB = 32  # n_embd
    N_HEAD = 4  # number of heads
    HEAD_DIM = 8  # N_EMB // N_HEAD

    def setUp(self):
        self.config = _make_config(use_local_attention=True)
        self.attn = GPT2FlashAttention(self.config, layer_idx=0, apply_rotary=False)
        self.attn.eval()  # disable dropout for deterministic output

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _hidden(self, B=None, L=None) -> torch.Tensor:
        B = B or self.B
        L = L or self.L
        torch.manual_seed(0)
        return torch.randn(B, L, self.N_EMB)

    # ------------------------------------------------------------------
    # 1. Output shape — 4D additive causal mask
    # ------------------------------------------------------------------
    def test_output_shape_with_causal_mask(self):
        """SDPA/xformers path: output matches (B, L, n_embd)."""
        hidden_states = self._hidden()
        attn_mask = _causal_mask(self.B, self.L)  # (2, 1, 8, 8)

        # Set a breakpoint here to inspect hidden_states and attn_mask
        attn_output, present = self.attn(hidden_states, attention_mask=attn_mask)

        self.assertEqual(attn_output.shape, (self.B, self.L, self.N_EMB))
        self.assertIsNone(present)  # use_cache=False by default

    # ------------------------------------------------------------------
    # 2. No NaN / Inf in output
    # ------------------------------------------------------------------
    def test_no_nan_inf_with_causal_mask(self):
        """Output must be finite under a standard causal mask."""
        hidden_states = self._hidden()
        attn_mask = _causal_mask(self.B, self.L)

        attn_output, _ = self.attn(hidden_states, attention_mask=attn_mask)

        self.assertFalse(torch.isnan(attn_output).any(), "NaN in output")
        self.assertFalse(torch.isinf(attn_output).any(), "Inf in output")

    # ------------------------------------------------------------------
    # 3. No attention mask (attn_mask=None)
    # ------------------------------------------------------------------
    def test_output_shape_no_mask(self):
        """Local-attention path with attention_mask=None still produces correct shape."""
        hidden_states = self._hidden()

        attn_output, _ = self.attn(hidden_states, attention_mask=None)

        self.assertEqual(attn_output.shape, (self.B, self.L, self.N_EMB))
        self.assertFalse(torch.isnan(attn_output).any())

    # ------------------------------------------------------------------
    # 4. Windowed local mask — blocked positions produce different output
    # ------------------------------------------------------------------
    def test_local_window_changes_output(self):
        """
        With a narrow window mask (window=2), tokens cannot attend beyond their
        2-step neighbourhood. This should produce a different output than full
        causal attention.
        """
        hidden_states = self._hidden()

        mask_full = _causal_mask(self.B, self.L)
        mask_local = _local_causal_mask(self.B, self.L, window=2)

        out_full, _ = self.attn(hidden_states, attention_mask=mask_full)
        out_local, _ = self.attn(hidden_states, attention_mask=mask_local)

        # Set a breakpoint here to compare out_full vs out_local row by row
        self.assertFalse(
            torch.allclose(out_full, out_local, atol=1e-4),
            "Windowed mask produced identical output to full causal — "
            "the mask is not being applied.",
        )

    # ------------------------------------------------------------------
    # 5. KV-cache: present key/value returned when use_cache=True
    # ------------------------------------------------------------------
    def test_use_cache_returns_present(self):
        """use_cache=True returns (key, value) as present."""
        hidden_states = self._hidden()
        attn_mask = _causal_mask(self.B, self.L)

        attn_output, present = self.attn(
            hidden_states, attention_mask=attn_mask, use_cache=True
        )

        self.assertIsNotNone(present)
        past_key, past_value = present
        # key/value shapes: (B, n_head, L, head_dim)
        self.assertEqual(past_key.shape, (self.B, self.N_HEAD, self.L, self.HEAD_DIM))
        self.assertEqual(past_value.shape, (self.B, self.N_HEAD, self.L, self.HEAD_DIM))

    # ------------------------------------------------------------------
    # 6. Decode step: extend sequence via layer_past
    # ------------------------------------------------------------------
    def test_decode_step_with_layer_past(self):
        """
        Simulate a KV-cache decode step:
          prefill L=7 tokens, then decode 1 new token with layer_past.
        The output for the new token should have shape (B, 1, n_embd).
        """
        L_prefix = 7
        hidden_prefix = self._hidden(L=L_prefix)
        mask_prefix = _causal_mask(self.B, L_prefix)

        # Prefill step — collect past key/value
        _, layer_past = self.attn(
            hidden_prefix, attention_mask=mask_prefix, use_cache=True
        )

        # Decode step — single new token
        hidden_new = self._hidden(L=1)
        # Decode mask: (B, 1, 1, KV_len) where KV_len = L_prefix + 1
        decode_mask = torch.zeros(self.B, 1, 1, L_prefix + 1)

        # Set a breakpoint here to inspect layer_past and decode_mask
        attn_output, new_past = self.attn(
            hidden_new,
            attention_mask=decode_mask,
            layer_past=layer_past,
            use_cache=True,
        )

        self.assertEqual(attn_output.shape, (self.B, 1, self.N_EMB))
        past_key, past_value = new_past
        self.assertEqual(past_key.shape[2], L_prefix + 1)  # KV length extended by 1

    # ------------------------------------------------------------------
    # 7. output_attentions=True — attn_weights slot is None (SDPA doesn't return them)
    # ------------------------------------------------------------------
    def test_output_attentions_returns_none_weights(self):
        """SDPA/xformers path sets attn_weights=None even when output_attentions=True."""
        hidden_states = self._hidden()
        attn_mask = _causal_mask(self.B, self.L)

        result = self.attn(
            hidden_states, attention_mask=attn_mask, output_attentions=True
        )
        # result = (attn_output, present, attn_weights)
        self.assertEqual(len(result), 3)
        attn_weights = result[2]
        self.assertIsNone(
            attn_weights,
            "Expected attn_weights=None for SDPA/xformers local-attention path",
        )

    # ------------------------------------------------------------------
    # 8. Verify xformers vs SDPA path is taken (informational)
    # ------------------------------------------------------------------
    def test_which_backend_is_used(self):
        """
        Prints which backend will handle local attention in this environment.
        Not a correctness assertion — useful when stepping in a debugger.
        """
        backend = "xformers" if HAS_XFORMERS else "torch.nn.functional.scaled_dot_product_attention"
        print(f"\n[test_which_backend_is_used] use_local_attention backend: {backend}")
        # Just a sanity check: the module is set up correctly
        self.assertTrue(
            getattr(self.config, "use_local_attention", False),
            "Config should have use_local_attention=True",
        )


class TestGPT2FlashAttentionEagerPath(unittest.TestCase):
    """
    Tests for use_local_attention=False, which falls through to _flash_attention_forward.
    These tests skip automatically if flash-attn is not installed.
    """

    B = 1
    L = 6
    N_EMB = 32

    def setUp(self):
        from transformers.utils import is_flash_attn_2_available
        if not is_flash_attn_2_available():
            self.skipTest("flash-attn not installed; skipping flash path tests")
        self.config = _make_config(use_local_attention=False)
        self.attn = GPT2FlashAttention(self.config, layer_idx=0, apply_rotary=False)
        self.attn.eval()

    def test_output_shape_binary_mask(self):
        """Flash path with binary (B, L) attention_mask returns (B, L, n_embd)."""
        torch.manual_seed(1)
        hidden_states = torch.randn(self.B, self.L, self.N_EMB)
        attn_mask = torch.ones(self.B, self.L)  # no padding

        attn_output, _ = self.attn(hidden_states, attention_mask=attn_mask)

        self.assertEqual(attn_output.shape, (self.B, self.L, self.N_EMB))
        self.assertFalse(torch.isnan(attn_output).any())


if __name__ == "__main__":
    unittest.main()
