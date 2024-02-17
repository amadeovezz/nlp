from typing import List, Dict

from base.abstract import Model
import torch


class LayerNorm(Model):

    def __init__(self, normalized_shape: int, eps=1e-5):
        """
        :param normalized_shape:
        """
        self.weights = torch.ones(normalized_shape, dtype=torch.float64)
        self.bias = torch.zeros(normalized_shape, dtype=torch.float64)
        self.epsilon = eps  # Small constant for numerical stability

    def __call__(self, input: torch.Tensor):
        """

        :param input: Tensor with dim [Batch, Embedding] or [Batch, Token Length, Embedding_dim]
        :return:
        """
        # Determine the dimension along which to calculate mean and variance
        dim = -1 if input.dim() == 2 else 2

        mean = torch.mean(input, dim=dim, keepdim=True)
        variance = torch.var(input, dim=dim, keepdim=True)
        normalized = (input - mean) / torch.sqrt(variance + self.epsilon)

        return normalized * self.weights + self.bias

    def zero_grad(self):
        self.weights.grad = None
        self.bias.grad = None

    def tune(self, learning_rate: float):
        self.weights.data += learning_rate * (-1 * self.weights.grad)
        self.bias.data += learning_rate * (-1 * self.bias.grad)

    def require_grad(self):
        self.weights.requires_grad = True
        self.bias.requires_grad = True

    def params(self) -> List[Dict]:
        return [{
            "layer": self.__class__.__name__
            , "weights": self.weights
            , "bias": self.bias
            }]
