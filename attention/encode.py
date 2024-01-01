from typing import Callable, Tuple, List
import math

import torch


def positional_encode(sequence_length: int, embedding_dim: int) -> torch.Tensor:
    """
    Generate sinusoidal positional embeddings in PyTorch for a given sequence length and embedding dimension.

    :param sequence_length: Length of the input sequence.
    :param embedding_dim: The dimensionality of the embeddings.
    :return: A PyTorch tensor of shape (sequence_length, embedding_dim) containing the positional embeddings.
    """
    position = torch.arange(sequence_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
    sinusoidal_embeddings = torch.zeros((sequence_length, embedding_dim))
    sinusoidal_embeddings[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_embeddings[:, 1::2] = torch.cos(position * div_term)

    return sinusoidal_embeddings

