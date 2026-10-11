from types import SimpleNamespace

import torch

from cehrgpt.generation.generate_batch_hf_gpt_sequence import generate_single_batch


def test_generate_single_batch_uses_tokenizer_special_token_ids():
    class Model:
        generation_config = SimpleNamespace(
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=None,
        )

        def generate(self, **kwargs):
            self.received_generation_config = kwargs["generation_config"]
            return SimpleNamespace(
                sequences=torch.tensor([[1, 2, 3]]),
                sequence_vals=None,
                sequence_val_masks=None,
            )

    tokenizer = SimpleNamespace(
        end_token_id=3,
        pad_token_id=4,
        decode=lambda *_args, **_kwargs: ["a", "b", "c"],
    )
    model = Model()

    generate_single_batch(
        model=model,
        cehrgpt_tokenizer=tokenizer,
        prompts=torch.tensor([[1, 2]]),
        max_length=10,
        max_new_tokens=2,
    )

    config = model.received_generation_config
    assert config.bos_token_id == tokenizer.end_token_id
    assert config.eos_token_id == tokenizer.end_token_id
    assert config.pad_token_id == tokenizer.pad_token_id
