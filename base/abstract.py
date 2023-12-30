from abc import ABC

from typing import List, Dict, Callable
import torch


class Layer(ABC):

    def __init__(self):
        raise NotImplementedError

    def __call__(self, inputs: torch.Tensor):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def require_grad(self):
        raise NotImplementedError


class Model(ABC):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, logits: torch.Tensor, targets: torch.Tensor):
        raise NotImplementedError

    def tune(self, learning_rate: float) -> None:
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def require_grad(self):
        raise NotImplementedError
