from typing import Callable
import torch

class LinearLayer:

    def __init__(self
                 , num_of_inputs: int
                 , num_of_neurons: int
                 , activation_func: Callable = None
                 , g=torch.Generator().manual_seed(2147483647)
                 , init_scale_factor_weights: float = 1
                 , init_scale_factor_biases: float = 1
                 , append_hidden_layer: bool = False
                 , append_pre_activation_layer: bool = False
                 ):

        # Assumes we are mini-batching
        self.weights = torch.randn(num_of_inputs, num_of_neurons, dtype=torch.float64, generator=g) * init_scale_factor_weights
        self.biases = torch.randn(num_of_neurons,dtype=torch.float64, generator=g) * init_scale_factor_biases
        self.activation_func = activation_func
        self.append_hidden_layer = append_hidden_layer
        self.append_pre_activation_layer = append_pre_activation_layer
        self.hidden_layers = [] if append_hidden_layer else None
        self.pre_activation_layers = [] if append_pre_activation_layer else None


    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        pre_activation = inputs @ self.weights + self.biases
        out = self.activation_func(pre_activation) if self.activation_func is not None else pre_activation

        if self.append_hidden_layer:
            self.hidden_layers.append(out)

        if self.append_pre_activation_layer:
            self.pre_activation_layers.append(pre_activation)

        return out

    def zero_grad(self) -> None:
        self.weights.grad = None
        self.biases.grad = None

    def require_grad(self) -> None:
        self.weights.requires_grad = True
        self.biases.requires_grad = True
