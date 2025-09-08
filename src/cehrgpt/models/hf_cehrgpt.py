import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn
from torch.distributions import Gamma
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.activations import gelu_new
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.streamers import BaseStreamer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.pytorch_utils import Conv1D
from transformers.utils import is_flash_attn_2_available, logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from cehrgpt.gpt_utils import (
    construct_age_sequence,
    encode_demographics,
    extract_time_interval_in_days,
    is_att_token,
    multiple_of_10,
)
from cehrgpt.models.activations import RMSNorm
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.gpt2 import GPT2Block, is_sample_pack
from cehrgpt.models.hf_modeling_outputs import (
    CehrGptCausalLMOutput,
    CehrGptGenerateDecoderOnlyOutput,
    CehrGptOutputWithPast,
    CehrGptSequenceClassifierOutput,
)

logger = logging.get_logger(__name__)


def extract_features_from_packed_sequence(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    max_index = attention_mask.nonzero(as_tuple=False).flatten()[-1]
    padded_attention_mask = f.pad(attention_mask[:, : max_index + 1], (0, 1))
    feature_indices = torch.nonzero(padded_attention_mask == 0)[:, 1] - 1
    return hidden_state[:, feature_indices]


def create_sample_packing_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Create a block-diagonal attention mask for packed sequences within a batch.

    Args:
        attention_mask (torch.Tensor): (batch_size, seq_len) binary mask where 1 = token, 0 = padding

    Returns:
        torch.Tensor: (batch_size, seq_len, seq_len) attention mask where entries are 1 if tokens
                      can attend to each other (within same packed segment), 0 otherwise.
    """
    # Step 1: Identify segments within each sample
    cumsum_mask = (attention_mask == 0).cumsum(dim=-1)
    segment_ids = cumsum_mask * attention_mask  # zeros remain zero

    # Step 2: Compare segment IDs pairwise per batch element
    # Shape: (batch_size, seq_len, seq_len)
    attn_matrix = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).int()

    # Step 3: Mask out padding tokens
    mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
    attn_matrix = attn_matrix * mask

    return attn_matrix


class MotorTaskHead(nn.Module):
    def __init__(
        self,
        input_dim,
        motor_tte_vocab_size,
        motor_num_time_pieces,
        eps=1e-6,
    ):
        super(MotorTaskHead, self).__init__()
        self.input_dim = input_dim
        self.motor_tte_vocab_size = motor_tte_vocab_size
        self.motor_num_time_pieces = motor_num_time_pieces
        self.final_layer = nn.Linear(input_dim, input_dim * motor_num_time_pieces)
        self.norm = RMSNorm(input_dim, eps)
        self.task_layer = nn.Linear(input_dim, motor_tte_vocab_size)
        self.task_time_bias = nn.Parameter(
            torch.zeros(1, self.motor_num_time_pieces, motor_tte_vocab_size)
        )

    def forward(self, x):
        # Ensure scale is positive
        length = x.shape[0]
        # (num_visits_in_batch, motor_tte_vocab_size * motor_num_time_pieces)
        x = self.final_layer(x).reshape(
            length, self.motor_num_time_pieces, self.input_dim
        )
        x = self.norm(x)
        x = self.task_layer(x) + self.task_time_bias
        # lambda_p = f.softplus(x)

        # # Check for NaN values
        # if torch.isnan(lambda_p).any():
        #     logger.warning(f"NaN values found in scale_param. x: {x}")
        # # (num_visits_in_batch,  motor_num_time_pieces, motor_tte_vocab_size,)
        return x


class VisitTimeToEventHead(nn.Module):
    def __init__(self, input_dim):
        super(VisitTimeToEventHead, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), gelu_new, nn.Linear(input_dim // 2, 1)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), gelu_new, nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        lambda_param = f.softplus(self.linear1(x))  # Ensure scale is positive
        k_param = f.softplus(self.linear2(x))  # Ensure shape is positive
        # Check for NaN values
        if torch.isnan(lambda_param).any():
            logger.warning(f"NaN values found in scale_param. x: {x}")
        if torch.isnan(k_param).any():
            logger.warning(f"NaN values found in k_param. x: {x}")
        return lambda_param, k_param


class ConceptValueTransformationLayer(nn.Module):
    def __init__(self, embedding_size):
        super(ConceptValueTransformationLayer, self).__init__()
        self.embedding_size = embedding_size
        self.merge_value_transformation_layer = nn.Sequential(
            nn.Linear(
                2 * embedding_size, embedding_size
            )  # +1 for the concept_values concatenation
        )

    def forward(
        self,
        concept_embeddings: Optional[torch.FloatTensor],
        value_indicators: Optional[torch.BoolTensor] = None,
        value_embeddings: Optional[torch.FloatTensor] = None,
    ):
        value_indicators = value_indicators.unsqueeze(-1)

        # Concatenate concept_embeddings and concept_values
        concept_embeddings_with_val = torch.cat(
            [concept_embeddings, value_embeddings], dim=-1
        )

        # Transform concatenated embeddings back to embedding_size
        transformed_embeddings = self.merge_value_transformation_layer(
            concept_embeddings_with_val
        )

        # Apply mask using torch.where
        concept_embeddings = torch.where(
            value_indicators, transformed_embeddings, concept_embeddings
        )

        return concept_embeddings


class CEHRGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained.

    models.
    """

    config_class = CEHRGPTConfig
    base_model_prefix = "cehrgpt"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(
                    mean=0.0,
                    std=(
                        self.config.initializer_range
                        / math.sqrt(2 * self.config.n_layer)
                    ),
                )

    def tie_weights(self):
        # We only want to tie weights when we DO NOT use use_pretrained_embeddings
        if not getattr(self.config, "use_pretrained_embeddings", False):
            super().tie_weights()
            # We want to tie the weights for value tokens regardless of the value of use_pretrained_embeddings
            if getattr(self.config, "tie_word_embeddings", True):
                output_value_embeddings = self.get_value_output_embeddings()
                if output_value_embeddings is not None:
                    self._tie_or_clone_weights(
                        output_value_embeddings, self.get_value_input_embeddings()
                    )

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        if getattr(self.config, "use_pretrained_embeddings", False):
            base_model = getattr(self, self.base_model_prefix, self)
            if base_model is not self:
                old_embeddings = base_model.pretrained_wte[0]
                new_embeddings = self._get_resized_embeddings(
                    old_embeddings, new_num_tokens, pad_to_multiple_of
                )
                old_embeddings_requires_grad = old_embeddings.weight.requires_grad
                new_embeddings.requires_grad_(old_embeddings_requires_grad)
                base_model.pretrained_wte[0] = new_embeddings
            else:
                raise NotImplementedError
        return super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def update_pretrained_embeddings(self, token_ids, pretrained_embeddings):
        if getattr(self.config, "use_pretrained_embeddings", False):
            new_pretrained_token_ids = []
            new_pretrained_embeddings = []
            for token_id, vector in zip(token_ids, pretrained_embeddings):
                if token_id not in self.config.pretrained_token_ids:
                    new_pretrained_token_ids.append(token_id)
                    new_pretrained_embeddings.append(vector)

            if new_pretrained_token_ids:
                self.pretrained_wte[0].weight.requires_grad = False
                self.pretrained_wte[0].weight[new_pretrained_token_ids] = torch.tensor(
                    np.asarray(new_pretrained_embeddings),
                    dtype=self.pretrained_wte[0].weight.dtype,
                    device=self.pretrained_wte[0].weight.device,
                )
                self.config.pretrained_token_ids.extend(new_pretrained_token_ids)

    def resize_value_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> Optional[nn.Embedding]:
        """
        Resizes value token embeddings matrix of the model if `new_num_tokens != config.value_vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        # If the model did not include values, we don't want to resize the value embeddings
        if not self.config.include_values:
            return None

        # check if the value vocab_size is less than the number of tokens
        # we need to resize the value_embeddings if necessary
        if self.config.value_vocab_size < new_num_tokens:
            # Update the embedding size
            old_value_embeddings = self.get_value_input_embeddings()
            new_value_embeddings = self._get_resized_embeddings(
                old_value_embeddings, new_num_tokens, pad_to_multiple_of
            )
            old_embeddings_requires_grad = old_value_embeddings.weight.requires_grad
            new_value_embeddings.requires_grad_(old_embeddings_requires_grad)
            self.set_value_input_embeddings(new_value_embeddings)
            is_quantized = (
                hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
            )
            # Update new_num_tokens with the actual size of new_value_embeddings
            if pad_to_multiple_of is not None:
                if is_deepspeed_zero3_enabled() and not is_quantized:
                    import deepspeed

                    with deepspeed.zero.GatheredParameters(
                        new_value_embeddings.weight, modifier_rank=None
                    ):
                        new_num_tokens = new_value_embeddings.weight.shape[0]
                else:
                    new_num_tokens = new_value_embeddings.weight.shape[0]

            # make sure that lm head is resized as well
            if (
                self.get_value_output_embeddings() is not None
                and not self.config.tie_word_embeddings
            ):
                old_value_head = self.get_value_output_embeddings()
                new_value_head = self._get_resized_lm_head(
                    old_value_head, new_num_tokens
                )
                old_value_head_requires_grad = old_value_head.weight.requires_grad
                new_value_head.requires_grad_(old_value_head_requires_grad)
                self.set_value_output_embeddings(new_value_head)
            # Update base model and current model config
            self.config.value_vocab_size = (
                self.get_value_input_embeddings().weight.shape[0]
            )

        # Return the input value embeddings
        return self.get_value_input_embeddings()

    def get_value_input_embeddings(self) -> nn.Embedding:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_value_input_embeddings()
        else:
            raise NotImplementedError

    def set_value_input_embeddings(self, value: nn.Embedding):
        """
        Set model's input embeddings.

        Args:
            value (`nn.Module`): A module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_value_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_value_output_embeddings(self) -> Optional[nn.Linear]:
        """
        Returns the model's output embeddings.

        Returns:
            `nn.Module`: A torch module mapping hidden states to vocabulary.
        """
        return None  # Overwrite for models with output embeddings

    def set_value_output_embeddings(self, output_embeddings: nn.Module) -> None:
        """
        Returns the model's output embeddings.

        Returns:
            `nn.Module`: A torch module mapping hidden states to vocabulary.
        """

    def set_position_embeddings(
        self, position_embeddings: Union[nn.Embedding, Tuple[nn.Embedding]]
    ) -> None:
        raise NotImplementedError(
            f"`set_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def update_attn_bias(self, new_num_position_embeddings: Optional[int]) -> None:
        raise NotImplementedError(
            f"`update_attn_bias` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def resize_position_embeddings(self, new_num_position_embeddings: Optional[int]):
        if new_num_position_embeddings is not None:
            is_quantized = (
                hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
            )
            wpe = self.get_position_embeddings()
            if wpe is not None:
                max_position, embed_dim = wpe.weight.shape
                if new_num_position_embeddings > max_position:
                    new_embeddings = nn.Embedding(
                        new_num_position_embeddings,
                        embed_dim,
                        device=wpe.weight.device,
                        dtype=wpe.weight.dtype,
                    )

                    # initialize all new embeddings (in particular added tokens)
                    self._init_weights(new_embeddings)
                    if is_deepspeed_zero3_enabled() and not is_quantized:
                        import deepspeed

                        params = [wpe.weight, new_embeddings.weight]
                        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                            new_embeddings.weight.data[:max_position, :] = (
                                wpe.weight.data[:max_position, :]
                            )
                    else:
                        new_embeddings.weight.data[:max_position, :] = wpe.weight.data[
                            :max_position, :
                        ]
                    self.set_position_embeddings(new_embeddings)
                    self.config.max_position_embeddings = new_num_position_embeddings
                    self.update_attn_bias(new_num_position_embeddings)


class CEHRGPT2Model(CEHRGPTPreTrainedModel):

    def __init__(self, config: CEHRGPTConfig):
        super().__init__(config)

        self.include_values = config.include_values
        self.include_ttv_prediction = config.include_ttv_prediction
        self.embed_dim = config.hidden_size

        if config.use_pretrained_embeddings:
            self.initialize_pretrained_embeddings()
        else:
            self.pretrained_wte = None

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        if self.include_values:
            self.vte = nn.Embedding(config.value_vocab_size, self.embed_dim)
            self.concept_value_transformation_layer = ConceptValueTransformationLayer(
                self.embed_dim
            )

        self.drop = nn.Dropout(config.embd_pdrop)
        gpt_blocks = []
        for i in range(config.num_hidden_layers):
            gpt_block = GPT2Block(config, layer_idx=i)
            gpt_block.is_causal = True
            gpt_blocks.append(gpt_block)
        self.h = nn.ModuleList(gpt_blocks)
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        # We do need to update the pre-computed attention bias matrix if sample packing requires a larger context window
        if self.config.sample_packing_max_positions > self.config.n_positions:
            logger.info(
                "Updated attn_bias to %s according to sample_packing_max_positions",
                config.sample_packing_max_positions,
            )
            self.update_attn_bias(self.config.sample_packing_max_positions)

    def initialize_pretrained_embeddings(self):
        layers = [
            nn.Embedding(self.config.vocab_size, self.config.pretrained_embedding_dim),
            nn.Linear(self.config.pretrained_embedding_dim, self.embed_dim),
            gelu_new,
        ]
        for _ in range(self.config.n_pretrained_embeddings_layers - 1):
            layers.extend(
                [
                    nn.Linear(self.embed_dim, self.embed_dim),
                    gelu_new,
                ]
            )
        self.pretrained_wte = nn.Sequential(*layers)
        # Disable the weight of the pretrained embeddings
        self.pretrained_wte[0].weight.requires_grad = False

    def parallelize(self, device_map=None):
        # Check validity of device_map
        warnings.warn(
            "`CEHRGPT2Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = (
            "cpu"
            if "cpu" in self.device_map.keys()
            else "cuda:" + str(min(self.device_map.keys()))
        )
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        if self.config.use_pretrained_embeddings:
            self.pretrained_wte = self.pretrained_wte.to(self.first_device)
        if self.include_values:
            self.vte = self.vte.to(self.first_device)
            self.concept_value_transformation_layer = (
                self.concept_value_transformation_layer.to(self.first_device)
            )
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        if self.config.use_pretrained_embeddings:
            self.pretrained_wte = self.pretrained_wte.to("cpu")
        self.vte = self.vte.to("cpu")
        self.concept_value_transformation_layer = (
            self.concept_value_transformation_layer.to("cpu")
        )
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def update_attn_bias(self, max_position_embeddings: int):
        for i in range(len(self.h)):
            self.h[i].attn.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(
                        (max_position_embeddings, max_position_embeddings),
                        dtype=torch.bool,
                    )
                )
                .view(1, 1, max_position_embeddings, max_position_embeddings)
                .to(self.h[i].attn.bias.device),
                persistent=False,
            )

    def get_position_embeddings(
        self,
    ) -> Optional[Union[nn.Embedding, Tuple[nn.Embedding]]]:
        return None

    def set_position_embeddings(self, new_embeddings: nn.Embedding):
        self.wpe = new_embeddings

    def get_value_input_embeddings(self) -> nn.Module:
        return self.vte

    def set_value_input_embeddings(self, value: nn.Module):
        self.vte = value

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        value_indicators: Optional[torch.BoolTensor] = None,
        values: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        random_vectors: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CehrGptOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        # When causal SFM is enabled, we need to expand the context window by one to make room for the random vector
        if (
            self.config.causal_sfm
            and input_ids.shape[1] >= self.config.demographics_size
        ):
            # Convert torch.Size to a list
            shape_list = list(input_shape)
            # Increment the last dimension
            shape_list[-1] += 1
            # Convert list back to torch.Size if needed
            input_shape = torch.Size(shape_list)

        input_ids.device

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        else:
            past_key_values[0][0].size(-2)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")

            if (
                self.config.causal_sfm
                and attention_mask.shape[-1] >= self.config.demographics_size
            ):
                # # Specify the indices to set to 0
                # rows = [1, 2, 2, 3, 3, 3]
                # cols = [0, 0, 1, 0, 1, 2]
                # # Set the specified indices to 0
                # attention_mask[rows, cols] = 0.0
                attention_mask = torch.concat(
                    [
                        attention_mask[..., : self.config.demographics_size],
                        attention_mask.new_ones(attention_mask.shape[:-1] + (1,)),
                        attention_mask[..., self.config.demographics_size :],
                    ],
                    dim=-1,
                )

            # The flash attention requires the original attention_mask
            if (
                not getattr(self.config, "_attn_implementation", "eager")
                == "flash_attention_2"
            ):
                attention_mask = attention_mask.view(batch_size, -1)

                # If this is sample packing, we need to great the
                if is_sample_pack(attention_mask):
                    attention_mask = create_sample_packing_attention_mask(
                        attention_mask
                    )[:, None, :, :]
                else:
                    # We create a 3D attention mask from a 2D tensor mask.
                    # Sizes are [batch_size, 1, 1, to_seq_length]
                    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                    # this attention mask is more simple than the triangular masking of causal attention
                    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                    attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(
                    dtype=self.dtype
                )  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if self.config.use_pretrained_embeddings:
            pretrained_token_id_indicators = torch.isin(
                input_ids,
                torch.tensor(self.config.pretrained_token_ids).to(input_ids.device),
            )
            input_embeddings = torch.where(
                pretrained_token_id_indicators.unsqueeze(-1),
                self.pretrained_wte(input_ids),
                self.wte(input_ids),
            )
        else:
            input_embeddings = self.wte(input_ids)

        if self.config.causal_sfm and input_shape[1] >= self.config.demographics_size:
            demographic_embeddings = input_embeddings[
                :, : self.config.demographics_size
            ]
            medical_event_embeddings = input_embeddings[
                :, self.config.demographics_size :
            ]
            if random_vectors is None:
                random_vectors = torch.rand_like(input_embeddings[:, :1])

            input_embeddings = torch.concat(
                [demographic_embeddings, random_vectors, medical_event_embeddings],
                dim=1,
            )
            position_ids = torch.concat(
                [
                    position_ids[:, : self.config.demographics_size],
                    position_ids[:, :1],
                    position_ids[:, self.config.demographics_size :],
                ],
                dim=1,
            )

        if self.include_values:
            if (
                self.config.causal_sfm
                and input_shape[1] >= self.config.demographics_size
            ):
                values = torch.concat(
                    [torch.zeros_like(values[:, :1], dtype=torch.int32), values],
                    dim=1,
                )
                value_indicators = torch.concat(
                    [
                        torch.zeros_like(value_indicators[:, :1]).to(torch.bool),
                        value_indicators,
                    ],
                    dim=1,
                )
            value_embeddings = self.vte(values)
            # Combine the value and concept embeddings together
            input_embeddings = self.concept_value_transformation_layer(
                concept_embeddings=input_embeddings,
                value_indicators=value_indicators,
                value_embeddings=value_embeddings,
            )

        hidden_states = self.drop(input_embeddings)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device) for past_state in layer_past
                    )
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    position_ids,
                    None,
                    attention_mask,
                    head_mask[i],
                    None,
                    None,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    position_ids=position_ids,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return CehrGptOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CEHRGPT2LMHeadModel(CEHRGPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight", "value_head.weight"]

    def __init__(self, config: CEHRGPTConfig):
        super().__init__(config)
        self.cehrgpt = CEHRGPT2Model(config)
        if self.config.include_ttv_prediction:
            self.tte_head = VisitTimeToEventHead(config.n_embd)

        if self.config.use_sub_time_tokenization:
            self.time_token_lm_head = nn.Linear(
                config.n_embd // 3, config.time_token_vocab_size, bias=False
            )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if self.config.include_values:
            self.value_head = nn.Linear(
                config.n_embd, config.value_vocab_size, bias=False
            )

        if self.config.include_motor_time_to_event:
            self.motor_tte = MotorTaskHead(
                input_dim=config.n_embd,
                motor_tte_vocab_size=config.motor_tte_vocab_size,
                motor_num_time_pieces=config.motor_num_time_pieces,
            )

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def parallelize(self, device_map=None):
        warnings.warn(
            "`GPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.cehrgpt.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.cehrgpt.h))
        self.cehrgpt.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.cehrgpt.first_device)
        if self.config.include_values:
            self.value_head = self.value_head.to(self.cehrgpt.first_device)
        if self.config.include_ttv_prediction:
            self.tte_head = self.tte_head.to(self.cehrgpt.first_device)
        if self.config.include_motor_time_to_event:
            self.motor_tte = self.motor_tte.to(self.cehrgpt.first_device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.cehrgpt.deparallelize()
        self.cehrgpt = self.cehrgpt.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        if self.config.include_values:
            self.value_head = self.value_head.to("cpu")
        if self.config.include_ttv_prediction:
            self.tte_head = self.tte_head.to("cpu")
        if self.config.include_motor_time_to_event:
            self.motor_tte = self.motor_tte.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_value_output_embeddings(self):
        if self.config.include_values:
            return self.value_head
        return None

    def set_value_output_embeddings(self, new_embeddings):
        if self.config.include_values:
            self.value_head = new_embeddings

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        return self.cehrgpt.get_position_embeddings()

    def set_position_embeddings(self, new_embeddings: nn.Embedding):
        self.cehrgpt.set_position_embeddings(new_embeddings)

    def update_attn_bias(self, max_position_embeddings: int):
        self.cehrgpt.update_attn_bias(max_position_embeddings)

    def update_motor_tte_vocab_size(
        self, motor_tte_vocab_size: Optional[int] = None
    ) -> None:
        update_motor_tte_layer = False
        if motor_tte_vocab_size and motor_tte_vocab_size > 0:
            if self.config.include_motor_time_to_event:
                if self.config.motor_tte_vocab_size != motor_tte_vocab_size:
                    self.config.include_motor_time_to_event = True
                    self.config.motor_tte_vocab_size = motor_tte_vocab_size
                    update_motor_tte_layer = True
            else:
                self.config.include_motor_time_to_event = True
                self.config.motor_tte_vocab_size = motor_tte_vocab_size
                update_motor_tte_layer = True

        if update_motor_tte_layer:
            self.motor_tte = MotorTaskHead(
                self.config.n_embd,
                self.config.motor_tte_vocab_size,
                self.config.motor_num_time_pieces,
            )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        cehrgpt_tokenizer,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        ages = kwargs.get("ages")
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            # Subtract the past_length by 1 due to the random vector
            if self.cehrgpt.config.causal_sfm:
                past_length -= 1
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            ages = ages[:, remove_prefix_length:]

        attention_mask = kwargs.get("attention_mask", None)
        random_vectors = kwargs.get("random_vectors", None)
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if self.cehrgpt.include_values:
            value_indicators = kwargs.get(
                "value_indicators", torch.zeros_like(input_ids).to(torch.bool)
            )
            values = kwargs.get(
                "values",
                torch.zeros_like(
                    input_ids,
                    dtype=torch.int32,
                ),
            )
            # Omit tokens covered by past_key_values
            if past_key_values:
                past_length = past_key_values[0][0].shape[2]
                # Some generation methods already pass only the last input ID
                if value_indicators.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # Default to old behavior: keep only final ID
                    remove_prefix_length = value_indicators.shape[1] - 1
                value_indicators = value_indicators[:, remove_prefix_length:]
                values = values[:, remove_prefix_length:]

            model_inputs.update(
                {"value_indicators": value_indicators, "values": values}
            )

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "ages": ages,
                "attention_mask": attention_mask,
                "random_vectors": random_vectors,
            }
        )

        return model_inputs

    def motor_nll_loss(
        self,
        hidden_states,
        motor_tte_times,
        motor_tte_event_indicators,
        motor_tte_task_indicators,
        motor_tte_masks,
        motor_end_index,
    ):
        """
        Computes the negative log-likelihood (NLL) loss using the LogNormal distribution.

        for modeling time-to-event data at each visit.

        Args:
            hidden_states (Tensor): Hidden representations for sequence tokens [num_of_concepts, hidden_dim].
            motor_tte_times (Tensor): Raw time-to-event durations [B, T, motor_vocab_size] (flattened).
            motor_tte_task_indicators: (Tensor): Bool indicators (True if included, False if not included).
            motor_tte_event_indicators (Tensor): Binary indicators (1 if censored, 0 if event occurred).
            motor_tte_masks (Tensor): Binary indicators whether the prediction should be masked
            (1 if not masked, 0 if masked).
            motor_end_index (Tensor): Tensor indicating the number of valid [VE] tokens in the batch.

        Returns:
            Tensor: Scalar loss value (mean negative log-likelihood).
        """
        motor_end_index = motor_end_index.sum().item()
        motor_tte_times = motor_tte_times.view(
            (-1, self.config.motor_num_time_pieces, self.config.motor_tte_vocab_size)
        )[:motor_end_index].clamp(min=1e-3)
        motor_tte_event_indicators = motor_tte_event_indicators.reshape(
            (-1, self.config.motor_num_time_pieces, self.config.motor_tte_vocab_size)
        )[:motor_end_index]
        # motor_tte_masks = motor_tte_masks.view(
        #     (-1, self.config.motor_num_time_pieces, self.config.motor_tte_vocab_size)
        # )[:motor_end_index]

        tte_features = hidden_states[motor_tte_task_indicators].view(
            (-1, self.config.n_embd)
        )

        assert tte_features.shape[0] == motor_tte_times.shape[0], (
            "The number of VE tokens in the labels needs to match up "
            "with the first dimension of motor_time_to_event_vectors. "
            f"Received ve_token_features.shape[0]: {tte_features.shape[0]}, "
            f"motor_time_to_event_vectors.shape[0]: {motor_tte_times.shape[0]}"
        )

        # Get Exponential parameters from model
        time_dependent_logits = self.motor_tte(tte_features)

        # Compute event loss
        # Calculate the accumulative hazard
        # exp(-sum_{j} lambda_j)
        survival_loss = torch.exp2(time_dependent_logits + motor_tte_times).mean()
        event_loss = (
            -math.log(2)
            * torch.where(motor_tte_event_indicators, time_dependent_logits, 0).mean()
        )

        # survival_loss = (
        #     torch.where(motor_tte_masks, lambda_p * motor_tte_times, 0)
        #     .sum(dim=1)
        #     .mean()
        # )
        # event_loss = (
        #     -torch.where(motor_tte_event_indicators, torch.log(lambda_p), 0)
        #     .sum(dim=1)
        #     .mean()
        # )
        return survival_loss + event_loss

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        value_indicators: Optional[torch.BoolTensor] = None,
        values: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        random_vectors: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        true_value_indicators: Optional[torch.BoolTensor] = None,
        true_values: Optional[torch.LongTensor] = None,
        time_to_visits: Optional[torch.FloatTensor] = None,
        time_token_indicators: Optional[torch.BoolTensor] = None,
        sub_time_tokens: Optional[torch.LongTensor] = None,
        motor_tte_times: Optional[torch.FloatTensor] = None,
        motor_tte_event_indicators: Optional[torch.BoolTensor] = None,
        motor_tte_task_indicators: Optional[torch.BoolTensor] = None,
        motor_tte_masks: Optional[torch.BoolTensor] = None,
        motor_end_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ages: Optional[torch.FloatTensor] = None,
        epoch_times: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CehrGptCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.cehrgpt(
            input_ids,
            value_indicators=value_indicators,
            values=values,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=ages,
            random_vectors=random_vectors,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        # get rid of the random vector:
        if (
            self.config.causal_sfm
            and hidden_states.shape[1] > self.config.demographics_size + 1
        ):
            hidden_states = torch.concat(
                [
                    hidden_states[:, : self.config.demographics_size],
                    hidden_states[:, self.config.demographics_size + 1 :],
                ],
                dim=1,
            )

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.cehrgpt.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        if self.cehrgpt.include_values:
            value_logits = self.value_head(hidden_states)
        else:
            value_logits = None

        loss = None
        token_loss = None
        time_token_loss = None
        time_to_visit_loss = None
        token_value_loss = None
        motor_tte_loss = None

        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)

            if self.config.causal_sfm:
                # Ensure demographic_labels matches the dtype of original labels
                demographic_labels = torch.full(
                    (labels.shape[0], self.config.demographics_size),
                    -100,
                    dtype=labels.dtype,  # Match the original labels' dtype
                    device=labels.device,  # Ensure on the same device
                )
                # Concatenate the demographic labels with the rest of the original labels
                labels = torch.cat(
                    (demographic_labels, labels[:, self.config.demographics_size :]),
                    dim=1,
                )

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_tokens: torch.BoolTensor = shift_labels != 100
            total_num_tokens = valid_tokens.sum()
            if (
                self.cehrgpt.config.lab_token_penalty
                and self.cehrgpt.config.lab_token_exists
            ):
                lab_index = torch.isin(
                    shift_labels.view(-1),
                    torch.tensor(self.cehrgpt.config.lab_token_ids).to(
                        lm_logits.device
                    ),
                )
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="none")
                token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                token_loss = torch.where(
                    lab_index,
                    token_loss * self.cehrgpt.config.lab_token_loss_weight,
                    token_loss,
                )

                token_loss = token_loss.sum() / total_num_tokens
            else:
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="none")
                token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                token_loss = token_loss.sum() / total_num_tokens

            loss = token_loss * self.cehrgpt.config.next_token_prediction_loss_weight

            if self.cehrgpt.config.entropy_penalty:
                # Compute probabilities using softmax
                probs = torch.softmax(shift_logits, dim=-1)
                # Compute negative entropy: sum(p * log(p))
                entropy = torch.sum(
                    probs * torch.log(probs + 1e-9), dim=-1
                )  # Add epsilon for numerical stability
                entropy = torch.where(valid_tokens, entropy, 0)
                # Regularization term: mean entropy scaled by alpha
                entropy_penalty = entropy.sum() / total_num_tokens
                loss += entropy_penalty * self.cehrgpt.config.entropy_penalty_alpha

            if (
                self.config.include_motor_time_to_event
                and motor_tte_times is not None
                and motor_tte_event_indicators is not None
                and motor_tte_task_indicators is not None
                and motor_tte_masks is not None
                and motor_end_index is not None
            ):
                motor_tte_loss = self.motor_nll_loss(
                    hidden_states=hidden_states,
                    motor_tte_times=motor_tte_times,
                    motor_tte_event_indicators=motor_tte_event_indicators,
                    motor_tte_task_indicators=motor_tte_task_indicators,
                    motor_tte_masks=motor_tte_masks,
                    motor_end_index=motor_end_index,
                )
                loss += motor_tte_loss * self.config.motor_time_to_event_weight

            # We add another loss term when use_sub_time_tokenization is enabled, we need to recover the sub time token
            # predictions for year/month/token
            if (
                self.config.use_sub_time_tokenization
                and sub_time_tokens is not None
                and time_token_indicators is not None
            ):
                # Split the last dimensions into three parts
                time_loss_fct = CrossEntropyLoss(reduction="none")
                time_token_logits = self.time_token_lm_head(
                    torch.unflatten(hidden_states, 2, (3, -1))
                )
                shifted_time_token_logits = time_token_logits[
                    ..., :-1, :, :
                ].contiguous()
                shifted_time_token_indicators = (
                    time_token_indicators[..., 1:].contiguous().to(lm_logits.device)
                )
                shifted_time_token_labels = (
                    sub_time_tokens[:, 1:, ...].contiguous().to(lm_logits.device)
                )
                time_token_loss = time_loss_fct(
                    shifted_time_token_logits.view(
                        -1, self.config.time_token_vocab_size
                    ),
                    shifted_time_token_labels.view(-1),
                )
                time_token_loss = torch.where(
                    shifted_time_token_indicators.view(-1, 1).to(torch.bool),
                    time_token_loss.view(-1, 3),
                    0,
                )
                time_token_loss = time_token_loss.sum() / total_num_tokens
                loss += time_token_loss * self.config.time_token_loss_weight

            if time_to_visits is not None and time_to_visits is not None:
                # Get lambda and k parameters
                lambda_param, k_param = self.tte_head(hidden_states)

                # Perform slicing before tensors are split across GPUs
                shifted_lambda_param = lambda_param[..., :-1, :].contiguous()
                shifted_k_param = k_param[..., :-1, :].contiguous()
                shift_time_to_visits = time_to_visits[..., 1:].contiguous()

                # Move to the same device as lambda_param
                shift_time_to_visits = shift_time_to_visits.to(lambda_param.device)
                time_to_visit_indicator = shift_time_to_visits >= 0
                # Define the Gamma distribution
                dist = Gamma(
                    shifted_k_param.squeeze(-1), shifted_lambda_param.squeeze(-1)
                )
                # Compute log-probs and apply the time_to_visit_indicator
                log_probs = dist.log_prob(
                    torch.clamp(shift_time_to_visits, min=1e-3) + 1e-6
                )
                log_probs = torch.where(time_to_visit_indicator, log_probs, 0)
                time_to_visit_loss = -log_probs.sum() / total_num_tokens
                # Compute the loss
                loss += time_to_visit_loss * self.config.time_to_visit_loss_weight

            if true_values is not None and true_value_indicators is not None:
                true_values = true_values.to(value_logits.device)
                shift_value_logits = value_logits[..., :-1, :].contiguous()
                shift_value_indicators = true_value_indicators[..., :-1].contiguous()
                shift_next_values = true_values[..., 1:].contiguous()
                value_loss_fct = CrossEntropyLoss(reduction="none")
                token_value_loss = value_loss_fct(
                    shift_value_logits.view(-1, shift_value_logits.size(-1)),
                    shift_next_values.view(-1),
                )
                token_value_loss = torch.where(
                    shift_value_indicators.view(-1), token_value_loss, 0
                )
                token_value_loss = token_value_loss.sum() / total_num_tokens
                if (
                    self.cehrgpt.config.lab_token_penalty
                    and self.cehrgpt.config.lab_token_exists
                ):
                    token_value_loss = (
                        token_value_loss * self.config.lab_token_loss_weight
                    )
                loss += token_value_loss * self.config.value_prediction_loss_weight

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CehrGptCausalLMOutput(
            loss=loss,
            logits=lm_logits,
            value_indicators=value_indicators,
            next_value_logits=value_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            token_loss=token_loss,
            time_token_loss=time_token_loss,
            time_to_visit_loss=time_to_visit_loss,
            token_value_loss=token_value_loss,
            motor_tte_loss=motor_tte_loss,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or.

        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
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
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        logits_warper = (
            logits_warper if logits_warper is not None else LogitsProcessorList()
        )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.generation_config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.generation_config.eos_token_id
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device)
            if eos_token_id is not None
            else None
        )
        output_scores = (
            output_scores
            if output_scores is not None
            else self.generation_config.output_scores
        )
        output_logits = (
            output_logits
            if output_logits is not None
            else self.generation_config.output_logits
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        if "cehrgpt_tokenizer" not in model_kwargs:
            raise RuntimeError(
                "The cehr-gpt tokenizer must be provided to the "
                "model.generate(..., cehrgpt_tokenizer=cehrgpt_tokenizer)"
            )

        # Remove this from the model_kwargs and will pass it to other functions explicitly
        cehrgpt_tokenizer = model_kwargs.pop("cehrgpt_tokenizer")

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        model_kwargs["attention_mask"] = input_ids != pad_token_id
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
        # Getting the lab token ids
        lab_token_ids = torch.tensor(
            cehrgpt_tokenizer.lab_token_ids,
            dtype=torch.int32,
        )
        if model_kwargs.get("value_indicators", None) is not None:
            value_indicators = model_kwargs.get("value_indicators")
        else:
            value_indicators = torch.zeros_like(input_ids).to(torch.bool)

        if model_kwargs.get("values", None) is not None:
            values = model_kwargs.get("values")
        else:
            values = torch.zeros_like(
                input_ids,
                dtype=torch.int32,
            )
        # Generate initial random_vectors
        if self.cehrgpt.config.causal_sfm:
            model_kwargs["random_vectors"] = torch.rand(
                [batch_size, 1, self.cehrgpt.embed_dim],
                dtype=(
                    torch.bfloat16 if is_flash_attn_2_available() else torch.float32
                ),
                device=input_ids.device,
            )
        else:
            model_kwargs["random_vectors"] = None
        model_kwargs["value_indicators"] = value_indicators
        model_kwargs["values"] = values

        # A variable to keep track of time and initialize it to zero
        batched_time_delta = np.zeros((batch_size,), dtype=np.float32)
        batched_ages = model_kwargs.get("ages", None)
        if batched_ages is None:
            batched_ages = []
            for token_ids in input_ids.detach().cpu():
                concept_ids = cehrgpt_tokenizer.decode(
                    token_ids.numpy(), skip_special_tokens=False
                )
                batched_ages.append(construct_age_sequence(concept_ids))
            # Turn this to a numpy array for easy manipulation
            batched_ages = np.asarray(batched_ages)
        else:
            batched_ages = batched_ages.cpu().numpy()
        # This is the base to which we will add the time delta
        base_ages = np.asarray([ages[-1] for ages in batched_ages])
        # Update the keyword arguments for the prepare_inputs_for_generation
        model_kwargs["ages"] = torch.tensor(batched_ages).to(input_ids.device)

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, cehrgpt_tokenizer, **model_kwargs
            )

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # TODO: decode to get time tokens and recalculate the age at this time step
            # Look for a potential time token
            for batch_i, next_concept_id in enumerate(
                cehrgpt_tokenizer.decode(
                    next_tokens.detach().cpu().numpy(), skip_special_tokens=False
                )
            ):
                if is_att_token(next_concept_id):
                    batched_time_delta[batch_i] += extract_time_interval_in_days(
                        next_concept_id
                    )

            next_age = (base_ages + batched_time_delta // 365).astype(int)[..., None]
            batched_ages = np.concatenate([batched_ages, next_age], axis=-1)
            model_kwargs["ages"] = torch.tensor(batched_ages).to(input_ids.device)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if self.cehrgpt.include_values:
                next_value_indicators = torch.isin(
                    next_tokens, lab_token_ids.to(next_tokens.device)
                )
                next_value_logits = outputs.next_value_logits[:, -1]
                # sample
                next_value_probs = nn.functional.softmax(next_value_logits, dim=-1)
                next_value_tokens = torch.multinomial(next_value_probs, num_samples=1)

                # update value_indicators
                value_indicators = torch.cat(
                    [value_indicators, next_value_indicators[:, None]], dim=-1
                )

                # update values
                values = torch.cat([values, next_value_tokens], dim=-1)

                model_kwargs["value_indicators"] = value_indicators
                model_kwargs["values"] = values

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        return CehrGptGenerateDecoderOnlyOutput(
            sequences=input_ids,
            sequence_val_masks=(
                value_indicators.to(torch.bool) if self.cehrgpt.include_values else None
            ),
            sequence_vals=(values if self.cehrgpt.include_values else None),
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = f.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CehrGptForClassification(CEHRGPTPreTrainedModel):
    _keep_in_fp32_modules = ["age_batch_norm", "dense_layer", "classifier"]

    def __init__(self, config: CEHRGPTConfig):
        super().__init__(config)

        self.cehrgpt = CEHRGPT2Model(config)
        self.age_batch_norm = torch.nn.BatchNorm1d(1)

        # Workaround
        self.age_batch_norm.weight.data = self.age_batch_norm.weight.data.float()
        self.age_batch_norm.bias.data = self.age_batch_norm.bias.data.float()

        self.dropout = nn.Dropout(config.summary_first_dropout)
        self.dense_layer = nn.Linear(config.hidden_size + 1, config.hidden_size // 2)
        self.dense_dropout = nn.Dropout(config.summary_first_dropout)
        self.classifier = nn.Linear(config.hidden_size // 2, 1)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def resize_position_embeddings(self, new_num_position_embeddings: Optional[int]):
        return self.cehrgpt.resize_position_embeddings(new_num_position_embeddings)

    def _apply_age_norm(
        self,
        age_at_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies batch normalization to the input age tensor.

        If the batch contains more than one example,
        standard batch normalization is applied. If the batch size is 1, batch normalization is applied
        without updating the running statistics, ensuring that the normalization uses the stored running
        mean and variance without modification.

        Args:
            age_at_index (torch.FloatTensor): A tensor containing the age values to normalize.
            The tensor has shape `(batch_size, num_features)` where `batch_size` is the number of samples in the batch.

        Returns:
            torch.FloatTensor: A tensor with the normalized age values.
        """
        if age_at_index.shape[0] > 1:
            normalized_age = self.age_batch_norm(age_at_index.float())
        else:
            self.age_batch_norm.eval()
            # Apply batch norm without updating running stats
            with (
                torch.no_grad(),
            ):  # Prevent tracking gradients, since we don't want to update anything
                normalized_age = self.age_batch_norm(age_at_index)
            # Optionally, set the layer back to training mode if needed later
            self.age_batch_norm.train()
        return normalized_age

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        age_at_index: torch.FloatTensor,
        classifier_label: Optional[torch.FloatTensor],
        value_indicators: Optional[torch.BoolTensor] = None,
        values: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ages: Optional[torch.FloatTensor] = None,
        epoch_times: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CehrGptSequenceClassifierOutput:

        cehrgpt_output = self.cehrgpt(
            input_ids=input_ids,
            value_indicators=value_indicators,
            values=values,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=ages,
        )

        if is_sample_pack(attention_mask):
            features = extract_features_from_packed_sequence(
                cehrgpt_output.last_hidden_state, attention_mask
            )
            assert features.shape[1] == classifier_label.shape[1], (
                "the length of the features need to be the same as the length of classifier_label. "
                f"features.shape[1]: {features.shape[1]}, "
                f"classifier_label.shape[1]: {classifier_label.shape[1]}"
            )
            assert features.shape[1] == age_at_index.shape[1], (
                "the length of the features need to be the same as the length of age_at_index. "
                f"features.shape[1]: {features.shape[1]}, "
                f"age_at_index.shape[1]: {age_at_index.shape[1]}"
            )
            num_samples = age_at_index.shape[1]
            features = features.view((num_samples, -1))
            classifier_label = classifier_label.view((num_samples, -1))
            with torch.autocast(device_type="cuda", enabled=False):
                normalized_age = self._apply_age_norm(
                    age_at_index.view((num_samples, 1))
                )
        else:
            features = cehrgpt_output.last_hidden_state[..., -1, :]
            # Disable autocasting for precision-sensitive operations
            with torch.autocast(device_type="cuda", enabled=False):
                normalized_age = self._apply_age_norm(age_at_index)

        # In case the model is in bfloat16
        if features.dtype != normalized_age.dtype:
            normalized_age = normalized_age.to(features.dtype)

        # In fine-tuning, the sequences are left-padded, so we use the last element as the pooler
        next_input = self.dropout(features)
        next_input = torch.cat([next_input, normalized_age], dim=1)
        next_input = self.dense_layer(next_input)
        next_input = nn.functional.relu(next_input)
        next_input = self.dense_dropout(next_input)
        logits = self.classifier(next_input)

        loss = None
        if classifier_label is not None:
            if self.config.class_weights:
                class_weights = torch.tensor(
                    [self.config.class_weights[1] / self.config.class_weights[0]],
                    dtype=torch.float32,
                ).to(logits.device)
            else:
                class_weights = None
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            loss = loss_fct(logits, classifier_label)

        return CehrGptSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=cehrgpt_output.last_hidden_state,
            attentions=cehrgpt_output.attentions,
        )

    def parallelize(self, device_map=None):
        self.cehrgpt.parallelize(device_map=device_map)

    def deparallelize(self):
        self.cehrgpt.deparallelize()
