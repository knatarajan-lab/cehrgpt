import argparse

import torch
from transformers import AutoTokenizer, GenerationConfig
from transformers.utils import is_flash_attn_2_available

from cehrgpt.models.encoder_decoder.instruct_hf_cehrgpt import InstructCEHRGPTModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer


def setup_model(tokenizer_path, model_path, device):
    encoder_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_path)
    model = (
        InstructCEHRGPTModel.from_pretrained(
            model_path,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
        )
        .eval()
        .to(device)
    )

    return encoder_tokenizer, cehrgpt_tokenizer, model


def generate_response(
    query, encoder_tokenizer, cehrgpt_tokenizer, model, device, generation_config
):
    encoder_inputs = encoder_tokenizer(query, return_tensors="pt")
    encoder_input_ids = encoder_inputs["input_ids"]
    encoder_attention_mask = encoder_inputs["attention_mask"]
    batch_size = encoder_input_ids.shape[0]
    batched_inputs = torch.tile(
        torch.tensor([[cehrgpt_tokenizer.start_token_id]]), (batch_size, 1)
    ).to(device)

    output = model.generate(
        inputs=batched_inputs,
        encoder_input_ids=encoder_input_ids.to(device),
        encoder_attention_mask=encoder_attention_mask.to(device),
        generation_config=generation_config,
        lab_token_ids=cehrgpt_tokenizer.lab_token_ids,
    )

    return [
        cehrgpt_tokenizer.decode(seq.cpu().numpy(), skip_special_tokens=True)
        for seq in output.sequences
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses using InstructCEHRGPTModel"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", required=True, help="Path to tokenizer"
    )
    parser.add_argument("--model_name_or_path", required=True, help="Path to model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Setting up the model and tokenizer...")

    encoder_tokenizer, cehrgpt_tokenizer, model = setup_model(
        args.tokenizer_name_or_path, args.model_name_or_path, device
    )

    # Configure model generation settings
    generation_config = GenerationConfig(
        max_length=1024,
        min_length=20,
        bos_token_id=cehrgpt_tokenizer.start_token_id,
        eos_token_id=cehrgpt_tokenizer.end_token_id,
        pad_token_id=cehrgpt_tokenizer.pad_token_id,
        do_sample=True,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=False,
        output_hidden_states=False,
        output_scores=False,
        renormalize_logits=True,
    )

    while True:
        query = input("Enter your query (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        response = generate_response(
            query,
            encoder_tokenizer,
            cehrgpt_tokenizer,
            model,
            device,
            generation_config,
        )
        print("Generated Response:", response)


if __name__ == "__main__":
    main()
