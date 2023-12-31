
from typing import Callable
import torch


class BatchedAttentionHead:
    def __init__(self
                 , emb_dim: int
                 , out_dimension: int
                 , g=torch.Generator().manual_seed(2147483647)
                 ):
        self.Query = torch.randn(emb_dim, out_dimension, dtype=torch.float64, generator=g)
        self.Key = torch.randn(emb_dim, out_dimension, dtype=torch.float64, generator=g)
        self.Value = torch.randn(emb_dim, out_dimension,dtype=torch.float64, generator=g)

    def __call__(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Pass along
        queries = input @ self.Query
        keys = input @ self.Key
        values = input @ self.Value

        # Compute attention scores
        matrix = (queries @ keys.transpose(-2, -1)).exp()
        scores = matrix / matrix.sum(1, keepdims=True)

        # Compute weighted embeddings
        new_embeddings = scores @ values
        return new_embeddings

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


class AttentionHead:

    def __init__(self
                 , emd_dim: int
                 , out_dimension: int
                 , g=torch.Generator().manual_seed(2147483647)
                 ):
        self.Query = torch.randn(emd_dim, out_dimension, dtype=torch.float64, generator=g)
        self.Key = torch.randn(emd_dim, out_dimension, dtype=torch.float64, generator=g)
        self.Value = torch.randn(emd_dim, out_dimension, dtype=torch.float64, generator=g)

    def __call__(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Pass along
        queries = input @ self.Query
        keys = input @ self.Key
        values = input @ self.Value

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


class StaticAttentionHead:

    def __init__(self
                 , num_of_possible_inputs: int
                 , out_dimension: int
                 , g=torch.Generator().manual_seed(2147483647)
                 ):
        self.Query = torch.randn(num_of_possible_inputs, out_dimension, generator=g)
        self.Key = torch.randn(num_of_possible_inputs, out_dimension, generator=g)
        self.Value = torch.randn(num_of_possible_inputs, out_dimension, generator=g)

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

