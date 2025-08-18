import torch
from torch.nn import functional as f


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
