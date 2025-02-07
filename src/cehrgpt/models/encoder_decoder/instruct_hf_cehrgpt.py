from dataclasses import asdict
from typing import List, Optional, Tuple, Union

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.models.encoder_decoder import EncoderDecoderModel

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.hf_modeling_outputs import CehrGptGenerateDecoderOnlyOutput

from .instruct_hf_modeling_outputs import InstructCehrGptCausalLMOutput


class InstructCEHRGPTModel(EncoderDecoderModel):

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[CEHRGPT2LMHeadModel] = None,
    ):
        super().__init__(config, encoder, decoder)
        # Put the encoder in the eval mode
        self.encoder.eval()

    def tie_weights(self):
        """
        The tie_weights will do nothing since encoder is a classic LLM and decoder is cehr-gpt.

        :return:
        """

    def forward(
        self,
        encoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, InstructCehrGptCausalLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # We don't want to update the encoder model
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                return_dict=return_dict,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        encoder_hidden_states = encoder_outputs[0]
        # optionally project encoder_hidden_states
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        value_indicators = kwargs.get("value_indicators", None)
        values = kwargs.get("values", None)
        past_key_values = kwargs.get("past_key_values", None)
        position_ids = kwargs.get("position_ids", None)
        random_vectors = kwargs.get("random_vectors", None)
        head_mask = kwargs.get("head_mask", None)
        use_cache = kwargs.get("use_cache", None)
        output_attentions = kwargs.get("output_attentions", None)
        output_hidden_states = kwargs.get("output_hidden_states", None)

        decoder_output = self.decoder(
            input_ids=input_ids,
            value_indicators=value_indicators,
            values=values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            random_vectors=random_vectors,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cross_attention=False,
        )

        decoder_output_dict = asdict(decoder_output)
        decoder_output_dict.update(
            {
                "encoder_last_hidden_state": encoder_hidden_states,
                "encoder_attentions": encoder_attention_mask,
            }
        )
        return InstructCehrGptCausalLMOutput(
            **decoder_output_dict,
        )

    def _sample(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[CehrGptGenerateDecoderOnlyOutput, torch.LongTensor]:

        encoder_input_ids = model_kwargs.get("encoder_attention_mask", None)
        encoder_attention_mask = model_kwargs.get("encoder_attention_mask", None)
        encoder_batch_size, encoder_sequence_length = encoder_input_ids.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_shape, device=encoder_input_ids.device
            )
        # We don't want to update the encoder model
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
            )
        encoder_hidden_states = encoder_outputs[0]

        generated_output = self.decoder._sample(
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **model_kwargs,
        )

        return generated_output
