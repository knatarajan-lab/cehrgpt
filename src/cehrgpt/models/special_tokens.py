# Backward-compatibility shim — do not remove.
# special_tokens has moved to cehrgpt.tokenization.special_tokens.

from cehrgpt.tokenization.special_tokens import (  # noqa: F401
    DISCHARGE_CONCEPT_IDS,
    END_TOKEN,
    LINEAR_PROB_TOKEN,
    OUT_OF_VOCABULARY_TOKEN,
    PAD_TOKEN,
    RANDOM_TOKEN,
    START_TOKEN,
    VISIT_CONCEPT_IDS,
)
