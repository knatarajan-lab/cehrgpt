import pytest

from cehrgpt.time_to_event.time_to_event_prediction import get_max_input_length


def test_context_window_caps_input_length():
    assert get_max_input_length(128, 1024, 256) == 128


def test_model_capacity_caps_input_length_after_reserving_generation_tokens():
    assert get_max_input_length(1024, 512, 128) == 384


@pytest.mark.parametrize(
    ("context_window", "model_max_length", "max_new_tokens"),
    [(0, 512, 128), (-1, 512, 128), (128, 128, 128), (128, 64, 128)],
)
def test_invalid_input_budgets_are_rejected(
    context_window, model_max_length, max_new_tokens
):
    with pytest.raises(ValueError):
        get_max_input_length(context_window, model_max_length, max_new_tokens)
