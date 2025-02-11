import os


class Config:
    VOCABULARY_DIR = os.getenv("vocabulary_dir")
    TOKENIZER_PATH = os.getenv("tokenizer_path")
    MODEL_PATH = os.getenv("model_path")
    USE_LLM_PARSER = bool(int(os.getenv("use_llm_parser", "0")))
    DEV_MODE = bool(int(os.getenv("dev_model", "0")))
