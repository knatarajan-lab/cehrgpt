from typing import Dict, List

from transformers import PretrainedConfig


class CEHRGPTConfig(PretrainedConfig):
    """
    Args:

        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            [`GPT2DoubleHeadsModel`].

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        bos_token_id (`int`, *optional*, defaults to 50256):
            Id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            Id of the end of sentence token in the vocabulary.
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.
    """

    model_type = "cehrgpt"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    @property
    def token_to_time_token_mapping(self) -> Dict[int, List[int]]:
        # The saved _token_to_time_token_mapping converts the key to string, so we need to convert it back to int
        return {
            int(token): list(map(int, sub_tokens))
            for token, sub_tokens in self._token_to_time_token_mapping.items()
        }

    def __init__(
        self,
        vocab_size=50257,
        time_token_vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        decoder_mlp="GPT2MLP",
        mlp_bias=False,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        lab_token_ids=None,
        ve_token_id=None,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        apply_rotary=False,
        include_values=False,
        value_vocab_size=None,
        include_ttv_prediction=False,
        use_sub_time_tokenization=True,
        include_motor_time_to_event=True,
        motor_tte_vocab_size=None,
        motor_time_to_event_weight=1.0,
        motor_num_time_pieces=16,
        token_to_time_token_mapping: Dict[int, List] = None,
        use_pretrained_embeddings=False,
        n_pretrained_embeddings_layers=2,
        pretrained_embedding_dim=768,
        pretrained_token_ids: List[int] = None,
        next_token_prediction_loss_weight=1.0,
        time_token_loss_weight=1.0,
        time_to_visit_loss_weight=1.0,
        causal_sfm=False,
        demographics_size=4,
        lab_token_penalty=False,
        lab_token_loss_weight=0.9,
        value_prediction_loss_weight=1.0,
        entropy_penalty=False,
        entropy_penalty_alpha=0.01,
        sample_packing_max_positions=None,
        class_weights=None,
        **kwargs,
    ):
        if token_to_time_token_mapping is None:
            token_to_time_token_mapping = {}
        if pretrained_token_ids is None:
            pretrained_token_ids = list()
        self.vocab_size = vocab_size
        self.time_token_vocab_size = time_token_vocab_size
        self.n_positions = n_positions
        self.sample_packing_max_positions = (
            sample_packing_max_positions
            if sample_packing_max_positions
            else n_positions
        )
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.decoder_mlp = decoder_mlp
        self.mlp_bias = mlp_bias
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.lab_token_ids = lab_token_ids

        self.apply_rotary = apply_rotary
        self.include_values = include_values
        self.value_vocab_size = value_vocab_size

        self.next_token_prediction_loss_weight = next_token_prediction_loss_weight
        self.include_ttv_prediction = include_ttv_prediction
        self.use_sub_time_tokenization = use_sub_time_tokenization
        self._token_to_time_token_mapping = token_to_time_token_mapping
        self.time_token_loss_weight = time_token_loss_weight
        self.time_to_visit_loss_weight = time_to_visit_loss_weight

        # MOTOR TTE configuration
        self.motor_tte_vocab_size = motor_tte_vocab_size
        self.include_motor_time_to_event = (
            include_motor_time_to_event
            and self.motor_tte_vocab_size
            and self.motor_tte_vocab_size > 0
        )
        if self.include_motor_time_to_event and not ve_token_id:
            raise RuntimeError(
                f"ve_token_id must be provided when include_motor_time_to_event is True"
            )
        self.ve_token_id = ve_token_id
        self.motor_time_to_event_weight = motor_time_to_event_weight
        self.motor_num_time_pieces = motor_num_time_pieces

        self.causal_sfm = causal_sfm
        self.demographics_size = demographics_size
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.pretrained_embedding_dim = pretrained_embedding_dim
        self.pretrained_token_ids = pretrained_token_ids
        self.n_pretrained_embeddings_layers = n_pretrained_embeddings_layers
        # self.tie_word_embeddings = not use_pretrained_embeddings

        self.lab_token_penalty = lab_token_penalty
        self.lab_token_loss_weight = lab_token_loss_weight
        self.entropy_penalty = entropy_penalty
        self.entropy_penalty_alpha = entropy_penalty_alpha
        self.value_prediction_loss_weight = value_prediction_loss_weight

        # Class weights for fine-tuning
        self.class_weights = class_weights

        kwargs["tie_word_embeddings"] = not use_pretrained_embeddings

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def lab_token_exists(self) -> bool:
        return self.lab_token_ids is not None and len(self.lab_token_ids) > 0
