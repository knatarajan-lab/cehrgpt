import numpy as np
import pytest
import torch

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import (
    CEHRGPT2LMHeadModel,
    extract_features_from_packed_sequence,
)
from cehrgpt.tools.linear_prob import attention_rollout as ar
from cehrgpt.tools.linear_prob import compute_attention_rollout as tool

VOCAB_SIZE = 64


def tiny_model(backbone="gpt2", linear_prob_n_layer=None, n_layer=3, **overrides):
    torch.manual_seed(0)
    kwargs = dict(
        backbone=backbone,
        vocab_size=VOCAB_SIZE,
        n_positions=64,
        n_embd=32,
        n_layer=n_layer,
        n_head=4,
        n_inner=64,
        activation_function="silu" if backbone == "qwen2" else "gelu_new",
        decoder_mlp="LlamaMLP" if backbone == "qwen2" else "GPT2MLP",
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        include_values=False,
        use_sub_time_tokenization=False,
        include_ttv_prediction=False,
        include_motor_time_to_event=True,
        motor_tte_vocab_size=4,
        motor_num_time_pieces=2,
        motor_time_bins=[0, 30, 60],
        linear_prob_n_layer=linear_prob_n_layer,
        linear_prob_token_id=63,
        ve_token_id=5,
        vs_token_id=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        attn_implementation="eager",
    )
    kwargs.update(overrides)
    model = CEHRGPT2LMHeadModel(CEHRGPTConfig(**kwargs)).eval()
    # The default initialisation attends almost uniformly, which would hide a wrong formula
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() >= 2:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.4)
    return model


def packed_batch():
    """Row 0 packs two patients (5 and 4 tokens) and ends with padding, row 1 is left padded."""
    generator = torch.Generator().manual_seed(2)
    input_ids = torch.randint(6, 60, (2, 12), generator=generator)
    attention_mask = torch.ones((2, 12))
    attention_mask[0, 5] = 0
    attention_mask[0, 10:] = 0
    attention_mask[1, :4] = 0
    input_ids[attention_mask == 0] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def random_attention(num_layers, attention_mask, causal, seed, heads=3):
    """Random attention weights that respect the mask, as (batch, heads, T, T) per layer."""
    generator = torch.Generator().manual_seed(seed)
    allowed = ar._allowed_attention(attention_mask, causal=causal)
    layers = []
    for _ in range(num_layers):
        weights = torch.rand((attention_mask.shape[0], heads) + allowed.shape[1:], generator=generator)
        weights = weights * allowed.unsqueeze(1)
        layers.append(weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9))
    return layers


def test_hand_computed_one_layer():
    self_attention = torch.tensor([[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]])
    cross_attention = torch.tensor([[[1.0, 0.0, 0.0], [0.6, 0.4, 0.0], [0.2, 0.3, 0.5]]])
    result = ar.linear_prob_attention_rollout(
        [self_attention], [cross_attention], torch.ones((1, 3))
    )
    # 0.5 * A + 0.5 * I = [[1, 0, 0], [0.25, 0.75, 0], [0.1, 0.15, 0.75]]
    # [0.2, 0.3, 0.5] @ that = [0.2 + 0.075 + 0.05, 0.225 + 0.075, 0.375]
    assert result.query_positions.tolist() == [2]
    torch.testing.assert_close(result.scores, torch.tensor([[0.325, 0.3, 0.375]]))


def test_hand_computed_two_probe_blocks_read_two_decoder_layers():
    first = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
    second = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    cross_first = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    cross_second = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
    result = ar.linear_prob_attention_rollout(
        [first, second], [cross_first, cross_second], torch.ones((1, 2))
    )
    # R_1 = [[1, 0], [0.25, 0.75]], R_2 = [[1, 0], [0.5, 0.5]] @ R_1 = [[1, 0], [0.625, 0.375]]
    # The first probe block reads R_1 at the last token: [0.25, 0.75], weight 0.25 / 0.75
    # The second reads [0.5, 0.5] @ R_2 = [0.8125, 0.1875], weight 0.5 / 0.75
    expected = (0.25 * torch.tensor([0.25, 0.75]) + 0.5 * torch.tensor([0.8125, 0.1875])) / 0.75
    torch.testing.assert_close(result.scores, expected.unsqueeze(0))

    uniform = ar.linear_prob_attention_rollout(
        [first, second],
        [cross_first, cross_second],
        torch.ones((1, 2)),
        probe_layer_weighting="uniform",
    )
    expected = 0.5 * torch.tensor([0.25, 0.75]) + 0.5 * torch.tensor([0.8125, 0.1875])
    torch.testing.assert_close(uniform.scores, expected.unsqueeze(0))


def test_probe_layer_weights():
    torch.testing.assert_close(
        ar.probe_layer_weights(3, 0.5, "rollout"), torch.tensor([1.0, 2.0, 4.0]) / 7.0
    )
    torch.testing.assert_close(ar.probe_layer_weights(4, 0.5, "uniform"), torch.full((4,), 0.25))
    with pytest.raises(ValueError):
        ar.probe_layer_weights(2, 0.5, "last")


def test_prediction_positions_are_where_the_features_are_taken():
    attention_mask = packed_batch()["attention_mask"]
    positions = ar.find_prediction_positions(attention_mask)
    assert positions.tolist() == [[0, 4], [0, 9], [1, 11]]
    # The packed row alone: the positions extract_features_from_packed_sequence reads
    hidden_state = torch.arange(12.0).view(1, 12, 1)
    features = extract_features_from_packed_sequence(hidden_state, attention_mask[:1])
    assert features.flatten().tolist() == [4.0, 9.0]


@pytest.mark.parametrize("num_probe_layers", [0, 1, 2, 3])
@pytest.mark.parametrize("weighting", ["rollout", "uniform"])
def test_packed_batch_matches_the_matrix_rollout(num_probe_layers, weighting):
    attention_mask = packed_batch()["attention_mask"]
    self_attentions = random_attention(3, attention_mask, causal=True, seed=3)
    cross_attentions = random_attention(num_probe_layers, attention_mask, causal=False, seed=4)
    result = ar.linear_prob_attention_rollout(
        self_attentions,
        cross_attentions,
        attention_mask,
        residual_weight=0.3,
        probe_layer_weighting=weighting,
    )
    scores = result.scores
    assert (scores >= 0).all()
    torch.testing.assert_close(scores.sum(dim=-1), torch.ones(3))
    # Nothing on padding, on the other patient of the packed row, or after the prediction time
    assert result.batch_indices.tolist() == [0, 0, 1]
    assert scores[0, 5:].abs().max() == 0
    assert scores[1, :6].abs().max() == 0 and scores[1, 10:].abs().max() == 0
    assert scores[2, :4].abs().max() == 0
    assert (scores[0, :5] > 0).all() and (scores[1, 6:10] > 0).all() and (scores[2, 4:] > 0).all()
    assert result.token_mask.sum(dim=-1).tolist() == [5, 4, 8]

    rollouts = ar.decoder_attention_rollout(self_attentions, attention_mask, residual_weight=0.3)
    for index, (row, position) in enumerate(ar.find_prediction_positions(attention_mask).tolist()):
        if num_probe_layers == 0:
            expected = rollouts[-1][row, position]
        else:
            weights = ar.probe_layer_weights(num_probe_layers, 0.3, weighting)
            expected = sum(
                weights[i] * cross_attentions[i].mean(dim=1)[row, position]
                @ rollouts[3 - num_probe_layers + i][row]
                for i in range(num_probe_layers)
            )
        torch.testing.assert_close(scores[index], expected, atol=1e-6, rtol=1e-5)


def test_decoder_rollout_never_reaches_the_future_or_another_patient():
    attention_mask = packed_batch()["attention_mask"]
    self_attentions = random_attention(4, attention_mask, causal=True, seed=5)
    allowed = ar._allowed_attention(attention_mask, causal=True)
    real = attention_mask.bool()
    for rollout in ar.decoder_attention_rollout(self_attentions, attention_mask):
        assert rollout[~allowed].abs().max() == 0
        torch.testing.assert_close(rollout.sum(dim=-1), real.float())


def test_an_earlier_prediction_time_with_a_causal_probe_gets_nothing_from_the_future():
    attention_mask = torch.ones((1, 6))
    self_attentions = random_attention(2, attention_mask, causal=True, seed=6)
    cross_attentions = random_attention(2, attention_mask, causal=True, seed=7)
    result = ar.linear_prob_attention_rollout(
        self_attentions,
        cross_attentions,
        attention_mask,
        query_positions=torch.tensor([[0, 2], [0, 5]]),
    )
    assert result.scores[0, 3:].abs().max() == 0
    torch.testing.assert_close(result.scores.sum(dim=-1), torch.ones(2))


def test_attention_that_ignores_the_mask_is_an_error():
    attention_mask = packed_batch()["attention_mask"]
    self_attentions = random_attention(2, attention_mask, causal=False, seed=8)
    cross_attentions = random_attention(1, attention_mask, causal=False, seed=9)
    with pytest.raises(ValueError, match="self attention of the decoder layer"):
        ar.linear_prob_attention_rollout(self_attentions, cross_attentions, attention_mask)

    self_attentions = random_attention(2, attention_mask, causal=True, seed=8)
    leaking = torch.full((2, 12, 12), 1.0 / 12)
    with pytest.raises(ValueError, match="cross attention of the probe block 0"):
        ar.linear_prob_attention_rollout(self_attentions, [leaking], attention_mask)


def test_invalid_arguments():
    attention_mask = torch.ones((1, 3))
    attention = random_attention(1, attention_mask, causal=True, seed=10)
    with pytest.raises(ValueError, match="every probe block reads its own decoder layer"):
        ar.linear_prob_attention_rollout(attention, attention * 2, attention_mask)
    with pytest.raises(ValueError, match="residual_weight"):
        ar.linear_prob_attention_rollout(attention, attention, attention_mask, residual_weight=1.0)
    with pytest.raises(ValueError, match="padding"):
        ar.linear_prob_attention_rollout(
            attention,
            attention,
            torch.tensor([[1.0, 1.0, 0.0]]),
            query_positions=torch.tensor([[0, 2]]),
            leak_tolerance=None,
        )
    with pytest.raises(ValueError, match="head_fusion"):
        ar.fuse_heads(torch.rand(1, 2, 3, 3), "median")


def test_uniform_baseline_favours_the_first_tokens():
    baseline = ar.uniform_attention_baseline(torch.ones((1, 6)), num_layers=3, num_probe_layers=3)
    scores = baseline.scores[0]
    torch.testing.assert_close(scores.sum(), torch.tensor(1.0))
    assert (scores[:-1] >= scores[1:]).all() and scores[0] > 2 * scores[-1]


@pytest.mark.parametrize("backbone", ["gpt2", "qwen2"])
@pytest.mark.parametrize("linear_prob_n_layer", [None, 2])
def test_model_rollout(backbone, linear_prob_n_layer):
    model = tiny_model(backbone, linear_prob_n_layer)
    batch = packed_batch()
    num_probe_layers = linear_prob_n_layer or 3
    assert len(model.linear_prob.linear_prob) == num_probe_layers

    with torch.no_grad():
        expected = model(**batch, output_attentions=True, output_hidden_states=True)
        with ar.AttentionCapture(model) as capture:
            captured = model(**batch, output_hidden_states=True)
    # Capturing changes nothing, and what is captured is what the model returns itself
    torch.testing.assert_close(captured.logits, expected.logits)
    torch.testing.assert_close(
        captured.linear_prob_hidden_states, expected.linear_prob_hidden_states
    )
    assert captured.attentions is None
    assert len(capture.self_attentions) == 3
    assert len(capture.cross_attentions) == num_probe_layers
    real = batch["attention_mask"].bool()
    for attention, expected_attention in zip(capture.self_attentions, expected.attentions):
        torch.testing.assert_close(attention[real], expected_attention.mean(dim=1)[real])
    # The hooks are gone
    for block in model.cehrgpt.h:
        assert not block.attn._forward_hooks and not block.attn._forward_pre_hooks

    result = ar.compute_token_importance(model, batch)
    scores = result.scores
    assert (scores >= 0).all()
    torch.testing.assert_close(scores.sum(dim=-1), torch.ones(3))
    assert scores[0, 5:].abs().max() == 0
    assert scores[1, :6].abs().max() == 0 and scores[1, 10:].abs().max() == 0
    assert scores[2, :4].abs().max() == 0

    manual = ar.linear_prob_attention_rollout(
        expected.attentions, capture.cross_attentions, batch["attention_mask"]
    )
    torch.testing.assert_close(scores, manual.scores)

    # A patient of a packed row has the scores it has alone
    for index, (start, end) in enumerate([(0, 5), (6, 10)]):
        alone = ar.compute_token_importance(
            model,
            {
                "input_ids": batch["input_ids"][:1, start:end],
                "attention_mask": batch["attention_mask"][:1, start:end],
            },
        )
        torch.testing.assert_close(
            alone.scores[0], scores[index, start:end], atol=1e-5, rtol=1e-4
        )

    # The local representation is the last decoder state: the row of the decoder rollout
    local = ar.compute_token_importance(model, batch, target="local")
    rollouts = ar.decoder_attention_rollout(expected.attentions, batch["attention_mask"])
    torch.testing.assert_close(local.scores[0], rollouts[-1][0, 4], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(local.scores, scores)


def test_model_without_the_probe_rolls_out_the_last_decoder_state():
    model = tiny_model(include_motor_time_to_event=False)
    assert model.linear_prob is None
    batch = packed_batch()
    result = ar.compute_token_importance(model, batch, target="global")
    local = ar.compute_token_importance(model, batch, target="local")
    torch.testing.assert_close(result.scores, local.scores)
    torch.testing.assert_close(result.scores.sum(dim=-1), torch.ones(3))


def test_capture_needs_the_eager_attention_and_eval_mode():
    model = tiny_model("qwen2")
    model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        ar.AttentionCapture(model)
    model = tiny_model()
    model.config.causal_sfm = True
    with pytest.raises(NotImplementedError):
        ar.AttentionCapture(model)
    model = tiny_model().train()
    with pytest.raises(ValueError, match="eval"):
        ar.compute_token_importance(model, packed_batch())


def test_capture_keeps_the_weights_the_caller_asked_for():
    model = tiny_model()
    batch = packed_batch()
    with torch.no_grad(), ar.AttentionCapture(model):
        outputs = model(**batch, output_attentions=True)
    assert len(outputs.attentions) == 3
    assert outputs.attentions[0].shape == (2, 4, 12, 12)


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return [f"c{token_id}" for token_id in token_ids]


def test_per_sample_token_importance_is_aligned_to_the_tokens():
    model = tiny_model()
    batch = packed_batch()
    importance = ar.compute_token_importance(model, batch)
    baseline = ar.uniform_attention_baseline(batch["attention_mask"], 3, 3)
    samples = tool.per_sample_token_importance(
        importance, baseline, batch["input_ids"], FakeTokenizer()
    )
    assert [len(sample["concept_ids"]) for sample in samples] == [5, 4, 8]
    assert samples[1]["concept_ids"] == [f"c{i}" for i in batch["input_ids"][0, 6:10].tolist()]
    for sample in samples:
        assert len(sample["importance"]) == len(sample["concept_ids"])
        np.testing.assert_allclose(sample["importance"].sum(), 1.0, rtol=1e-5)
        np.testing.assert_allclose(sample["baseline_importance"].sum(), 1.0, rtol=1e-5)


def test_decode_tokens_falls_back_to_the_ids():
    class JoiningTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            return ["joined"]

    assert tool.decode_tokens(JoiningTokenizer(), [3, 4]) == ["3", "4"]
    assert tool.decode_tokens(FakeTokenizer(), [3, 4]) == ["c3", "c4"]


def test_concept_importance_aggregator():
    aggregator = tool.ConceptImportanceAggregator()
    aggregator.add(["a", "b", "a"], [0.2, 0.5, 0.3], [0.5, 0.25, 0.25], label=True)
    aggregator.add(["b", "c"], [0.4, 0.6], [0.5, 0.5], label=False)
    frame = aggregator.to_frame().set_index("concept_id")
    assert frame.index.tolist() == ["b", "c", "a"]
    np.testing.assert_allclose(frame["mean_importance"].sum(), 1.0)
    assert frame.loc["a", "num_samples"] == 1 and frame.loc["b", "num_samples"] == 2
    np.testing.assert_allclose(frame.loc["a", "mean_importance"], 0.25)
    np.testing.assert_allclose(frame.loc["a", "mean_importance_when_present"], 0.5)
    np.testing.assert_allclose(frame.loc["a", "mean_baseline_when_present"], 0.75)
    np.testing.assert_allclose(frame.loc["a", "lift"], 0.5 / 0.75)
    np.testing.assert_allclose(frame.loc["b", "lift"], 0.9 / 0.75)
    np.testing.assert_allclose(frame.loc["b", "mean_importance_positive"], 0.5)
    np.testing.assert_allclose(frame.loc["b", "mean_importance_negative"], 0.4)
    np.testing.assert_allclose(frame.loc["c", "mean_importance_positive"], 0.0)


DAY = 86400.0
PERSON_IDS = [7, 8, 7, 7, 9]
# 2020-09-13 12:26:40 UTC is 1.6e9
INDEX_DATES = [1.6e9, 1.6e9, 1.6e9 + 3600.0, 1.6e9 + 5 * DAY, 1.6e9]


def test_parse_prediction_time_is_a_day_or_a_second_in_utc():
    assert tool.parse_prediction_time("2020-09-13") == (1.6e9 - 44800.0, 1.6e9 + 41600.0, False)
    assert tool.parse_prediction_time("2020-09-13 12:26:40") == (1.6e9, 1.6e9 + 1.0, True)
    assert tool.parse_prediction_time("2020-09-13T12:26:40") == (1.6e9, 1.6e9 + 1.0, True)


def test_select_sample_indices_of_a_subject():
    assert tool.select_sample_indices(PERSON_IDS, INDEX_DATES, 7) == [0, 2, 3]
    assert tool.select_sample_indices(PERSON_IDS, INDEX_DATES, 7, "2020-09-13") == [0, 2]
    assert tool.select_sample_indices(PERSON_IDS, INDEX_DATES, 7, "2020-09-13 13:26:40") == [2]
    assert tool.select_sample_indices(PERSON_IDS, INDEX_DATES, 7, "2020-09-18") == [3]
    assert tool.select_sample_indices(PERSON_IDS, INDEX_DATES, 7, "2021-01-01") == []
    assert tool.select_sample_indices(PERSON_IDS, INDEX_DATES, 10) == []
    # The collator and the datasets give the ids as a column of lists or of numbers
    assert tool.select_sample_indices([[p] for p in PERSON_IDS], INDEX_DATES, 9) == [4]


def test_select_sample_indices_falls_back_to_the_day():
    # No sample at that second: the cohort may be stored with dates only
    assert tool.select_sample_indices(PERSON_IDS, INDEX_DATES, 7, "2020-09-13 08:00:00") == [0, 2]
    assert tool.select_sample_indices(PERSON_IDS, INDEX_DATES, 7, "2020-09-14 08:00:00") == []


def test_token_importance_frame():
    frame = tool.token_importance_frame(
        {
            "concept_ids": ["a", "b", "a"],
            "importance": np.asarray([0.5, 0.3, 0.2], dtype=np.float32),
            "baseline_importance": np.asarray([0.25, 0.6, 0.0], dtype=np.float32),
        }
    )
    assert frame.columns.tolist() == [
        "position",
        "concept_id",
        "importance",
        "baseline_importance",
        "lift",
    ]
    assert frame["position"].tolist() == [0, 1, 2]
    assert frame["concept_id"].tolist() == ["a", "b", "a"]
    np.testing.assert_allclose(frame["lift"].to_numpy()[:2], [2.0, 0.5], rtol=1e-6)
    assert np.isnan(frame["lift"].to_numpy()[2])
