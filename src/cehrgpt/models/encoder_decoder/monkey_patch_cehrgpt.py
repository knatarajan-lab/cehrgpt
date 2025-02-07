from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForCausalLM

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel

original_from_config = AutoModelForCausalLM.from_config


def for_model(model_type: str, *args, **kwargs):
    if model_type == "cehrgpt":
        return CEHRGPTConfig(*args, **kwargs)
    if model_type in CONFIG_MAPPING:
        config_class = CONFIG_MAPPING[model_type]
        return config_class(*args, **kwargs)
    raise ValueError(
        f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
    )


def from_config(config, **kwargs):
    if isinstance(config, CEHRGPTConfig):
        return CEHRGPT2LMHeadModel(config)
    else:
        return original_from_config(config, **kwargs)


def register_cehrgpt_in_hf():
    # Monkey patch to work around the validation for cehrgpt
    AutoConfig.for_model = for_model
    AutoModelForCausalLM.from_config = from_config
