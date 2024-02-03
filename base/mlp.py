from typing import List, Dict, Callable
import torch
from torch.functional import F

from base import abstract


class LinearLayer(abstract.Layer):

    def __init__(self
                 , num_of_inputs: int
                 , num_of_neurons: int
                 , activation_func: Callable = None
                 , generator: torch.Generator = None
                 , init_scale_factor_weights: float = 1
                 , init_scale_factor_biases: float = 1
                 , append_hidden_layer: bool = False
                 , append_pre_activation_layer: bool = False
                 ):
        """

        our weights matrix gets initialized as: [Weights for each input, Number of nodes in layer]
        """
        # Params
        self.weights = torch.randn(num_of_inputs, num_of_neurons, dtype=torch.float64,
                                   generator=generator) * init_scale_factor_weights

        self.biases = torch.randn(num_of_neurons, dtype=torch.float64, generator=generator) * init_scale_factor_biases

        # Activation
        self.activation_func = activation_func

        # Debugging and logging
        self.append_hidden_layer = append_hidden_layer
        self.append_pre_activation_layer = append_pre_activation_layer
        self.hidden_layers = [] if append_hidden_layer else None
        self.pre_activation_layers = [] if append_pre_activation_layer else None

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        [Batch, Inputs in] @  [Weights for each input, Number of nodes in layer]

        :param inputs: tensor with dim: [Batch, Inputs in]
        :return: [Batch, Number of nodes in layer]
        """
        pre_activation = inputs @ self.weights + self.biases

        out = self.activation_func(pre_activation) if self.activation_func is not None else pre_activation

        if self.append_hidden_layer:
            self.hidden_layers.append(out)

        if self.append_pre_activation_layer:
            self.pre_activation_layers.append(pre_activation)

        return out

    def tune(self, learning_rate: float) -> None:
        self.weights.data += learning_rate * (-1 * self.weights.grad)
        self.biases.data += learning_rate * (-1 * self.biases.grad)

    def zero_grad(self) -> None:
        self.weights.grad = None
        self.biases.grad = None

    def require_grad(self) -> None:
        self.weights.requires_grad = True
        self.biases.requires_grad = True


class MLP(abstract.Model):

    def __init__(self, layers: List[LinearLayer]):
        self.layers = layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: tensor with dim: [Batch, Inputs]
        :return: tensor with dim [Batch, Logits]
        """
        out = inputs
        for layer in self.layers:
            out = layer(out)
        return out

    def tune(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.tune(learning_rate)

    def loss(self, out: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out, targets)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def require_grad(self):
        for layer in self.layers:
            layer.require_grad()

    @torch.no_grad
    def dataset_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        out = self.forward(inputs)
        return self.loss(out, targets)

