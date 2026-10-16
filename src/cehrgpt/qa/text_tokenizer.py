from typing import Iterable

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"


def train_text_tokenizer(
    texts: Iterable[str], vocab_size: int = 8000, min_frequency: int = 2
) -> PreTrainedTokenizerFast:
    """
    Train a byte level BPE tokenizer on the question/answer text.

    The language model is trained from scratch, so its vocabulary only needs to cover the
    generated questions and answers. Byte level BPE never produces unknown tokens.
    """
    backend = ByteLevelBPETokenizer()
    backend.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend._tokenizer,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
    )
