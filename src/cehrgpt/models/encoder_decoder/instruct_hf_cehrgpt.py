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
from .monkey_patch_cehrgpt import register_cehrgpt_in_hf


class InstructCEHRGPTModel(EncoderDecoderModel):

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[CEHRGPT2LMHeadModel] = None,
    ):
        super().__init__(config, encoder, decoder)
        if not getattr(config.encoder, "encoder_trainable", False):
            # Set the whole model to be non-trainable
            for param in self.encoder.parameters():
                param.requires_grad = False

    def tie_weights(self):
        """
        The tie_weights will do nothing since encoder is a classic LLM and decoder is cehr-gpt.

        :return:
        """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        value_indicators: Optional[torch.BoolTensor] = None,
        values: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        random_vectors: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        true_value_indicators: Optional[torch.BoolTensor] = None,
        true_values: Optional[torch.LongTensor] = None,
        time_to_visits: Optional[torch.FloatTensor] = None,
        time_token_indicators: Optional[torch.BoolTensor] = None,
        sub_time_tokens: Optional[torch.LongTensor] = None,
        # This is added so person_id can be passed to the collator
        person_id: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
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
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        encoder_hidden_states = encoder_outputs[0]
        # optionally project encoder_hidden_states
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        decoder_output = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            value_indicators=value_indicators,
            values=values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            labels=labels,
            true_value_indicators=true_value_indicators,
            true_values=true_values,
            time_to_visits=time_to_visits,
            time_token_indicators=time_token_indicators,
            sub_time_tokens=sub_time_tokens,
            past_key_values=past_key_values,
            position_ids=position_ids,
            random_vectors=random_vectors,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_output + encoder_outputs

        return InstructCehrGptCausalLMOutput(
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            **decoder_output,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        register_cehrgpt_in_hf()
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        lab_token_ids=None,
        **kwargs,
    ):
        return self.decoder.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            lab_token_ids=lab_token_ids,
            **kwargs,
        )

    def _sample(
        self,
        inputs: Optional[torch.LongTensor],
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

        encoder_outputs = model_kwargs.get("encoder_outputs")
        encoder_hidden_states = encoder_outputs[0]
        # This is important so that the decoder will not use attention_mask as its own mask
        encoder_attention_mask = model_kwargs.pop("attention_mask")

        # optionally project encoder_hidden_states
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        generated_output = self.decoder._sample(
            input_ids=inputs,
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
