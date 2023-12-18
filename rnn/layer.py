from typing import Callable, List
import torch

class RecurrentLayer:

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

        # Params
        # Assumes we are mini-batching
        self.weights = torch.randn(num_of_inputs, num_of_neurons, dtype=torch.float64, generator=g) * init_scale_factor_weights
        self.biases = torch.randn(num_of_neurons, dtype=torch.float64, generator=g, ) * init_scale_factor_biases
        self.activation_func = activation_func
        # Save our previous activation (required for our RNN)
        self.previous_pre_activation = None

        # Debugging and logging
        self.append_hidden_layer = append_hidden_layer
        self.append_pre_activation_layer = append_pre_activation_layer
        self.hidden_layers = [] if append_hidden_layer else None
        self.pre_activation_layers = [] if append_pre_activation_layer else None

    def params(self) -> List[torch.Tensor]:
        return [self.weights, self.biases, self.previous_pre_activation]

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:

        # Regular layer
        pre_activation = inputs @ self.weights + self.biases

        # Recurrent step
        weighted_pre_activation = pre_activation
        if self.previous_pre_activation is not None:
            weighted_pre_activation = pre_activation + self.previous_pre_activation

        self.previous_pre_activation = weighted_pre_activation

        out = self.activation_func(weighted_pre_activation) if self.activation_func is not None else weighted_pre_activation

        # Logging
        if self.append_hidden_layer:
            self.hidden_layers.append(out)

        if self.append_pre_activation_layer:
            self.pre_activation_layers.append(pre_activation)

        return out

    def zero_grad(self) -> None:
        self.weights.grad = None
        self.biases.grad = None
        self.previous_pre_activation = None

    def require_grad(self) -> None:
        self.weights.requires_grad = True
        self.biases.requires_grad = True
