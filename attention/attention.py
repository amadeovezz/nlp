from typing import Callable
import torch
import numpy as np
from torch.functional import F


class BatchedSelfAttentionHead:
    def __init__(self
                 , emb_dim: int
                 , out_dimension: int
                 , block_type: str = "encoder"
                 , generator: torch.Generator = None
                 ):
        self.Query = torch.randn(emb_dim, out_dimension, dtype=torch.float64, generator=generator)
        self.Key = torch.randn(emb_dim, out_dimension, dtype=torch.float64, generator=generator)
        self.Value = torch.randn(emb_dim, out_dimension, dtype=torch.float64, generator=generator)
        self.block_type = block_type

    def __call__(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
       :param input: A tensor of dim [Batch, Token, Embedding dim]
       :return:
       """
        # Pass along
        queries = input @ self.Query
        keys = input @ self.Key
        values = input @ self.Value

        # Compute attention scores
        matrix = (queries @ keys.transpose(-2, -1))

        if self.block_type == "decoder":
            num_of_tokens = input.shape[1]
            # Create auto-regressive token communication
            tril = torch.tril(torch.ones(num_of_tokens, num_of_tokens, dtype=torch.float64))
            masked = matrix.masked_fill(tril == 0, float(-1e9))
            scores = F.softmax(masked, dim=2)
        elif self.block_type == "encoder":
            scores = F.softmax(matrix, dim=2)

        # Compute weighted embeddings
        new_embeddings = scores @ values
        return new_embeddings, scores

    def tune(self, learning_rate: float):
        self.Query.data += learning_rate * (-1 * self.Query.grad)
        self.Key.data += learning_rate * (-1 * self.Key.grad)
        self.Value.data += learning_rate * (-1 * self.Value.grad)

    def require_grad(self):
        self.Query.requires_grad = True
        self.Key.requires_grad = True
        self.Value.requires_grad = True

    def zero_grad(self):
        self.Query.grad = None
        self.Key.grad = None
        self.Value.grad = None


class SelfAttentionHead:

    def __init__(self
                 , emd_dim: int
                 , out_dimension: int
                 , block_type: str = "encoder"
                 , generator: torch.Generator = None
                 ):

        self.Query = torch.randn(emd_dim, out_dimension, dtype=torch.float64, generator=generator)
        self.Key = torch.randn(emd_dim, out_dimension, dtype=torch.float64, generator=generator)
        self.Value = torch.randn(emd_dim, out_dimension, dtype=torch.float64, generator=generator)
        self.block_type = block_type

    def __call__(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        :param input: A tensor of dim [Token, Embedding dim]
        :return:
        """

        # Pass along
        queries = input @ self.Query
        keys = input @ self.Key
        values = input @ self.Value

        # Compute attention scores
        matrix = (queries @ keys.T)

        if self.block_type == "decoder":
            num_of_tokens = input.shape[0]
            tril = torch.tril(torch.ones(num_of_tokens, num_of_tokens, dtype=torch.float64))
            masked = matrix.masked_fill(tril == 0, float('-inf'))
            scores = F.softmax(masked, dim=1)
        elif self.block_type == "encoder":
            scores = F.softmax(matrix, dim=1)

        # Compute weighted embeddings
        new_embeddings = scores @ values
        return new_embeddings, scores

    def require_grad(self):
        self.Query.requires_grad = True
        self.Key.requires_grad = True
        self.Value.requires_grad = True


class CrossAttentionHead:
    # TODO: not sure this is really cross attention

    def __init__(self
                 , num_of_possible_inputs: int
                 , out_dimension: int
                 , generator: torch.Generator = None
                 ):
        self.Query = torch.randn(num_of_possible_inputs, out_dimension, generator=generator)
        self.Key = torch.randn(num_of_possible_inputs, out_dimension, generator=generator)
        self.Value = torch.randn(num_of_possible_inputs, out_dimension, generator=generator)

    def __call__(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Fetch embeddings
        queries = self.Query[input]
        keys = self.Key[input]
        values = self.Value[input]

        # Compute attention scores
        matrix = (queries @ keys.T).exp()
        scores = matrix / matrix.sum(1, keepdims=True)

        # Compute weighted embeddings
        new_embeddings = scores @ values
        return new_embeddings, scores


    def require_grad(self):
        self.Query.requires_grad = True
        self.Key.requires_grad = True
        self.Value.requires_grad = True

