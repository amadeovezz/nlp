from typing import Callable, List
import torch

from base.abstract import Layer

class RecurrentLayer(Layer):

    def __init__(self
                 , num_of_inputs: int
                 , num_of_neurons: int
                 , activation_func: Callable = None
                 , generator: torch.Generator = None
                 , init_scale_factor_weights: float = 1
                 , init_scale_factor_biases: float = 1
                 , append_pre_activation_layer: bool = False
                 , append_activation_layer: bool = False
                 , append_activation_gradients: bool = False
                 ):

        # Params
        # Creates dim: [Weights for each input, Number of nodes in layer]
        self.weights = torch.randn(num_of_inputs, num_of_neurons, dtype=torch.float64,
                                   generator=generator) * init_scale_factor_weights
        self.biases = torch.randn(num_of_neurons, dtype=torch.float64, generator=generator) * init_scale_factor_biases
        self.activation_func = activation_func

        # Debugging and logging
        self.append_pre_activation_layer = append_pre_activation_layer
        self.append_activation_layer = append_activation_layer
        self.pre_activation_layers = [] if append_pre_activation_layer else None
        self.activation_layers = [] if append_activation_layer else None

        # Specific to RecurrentLayer
        self.previous_pre_activation = None

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        pre_activation = inputs @ self.weights + self.biases

        # Recurrent step
        aggregated_pre_activation = pre_activation
        if self.previous_pre_activation is not None:
            aggregated_pre_activation = pre_activation + self.previous_pre_activation

        self.previous_pre_activation = aggregated_pre_activation

        # Apply activation func
        out = self.activation_func(aggregated_pre_activation) if self.activation_func is not None else aggregated_pre_activation

        # Logging
        if self.append_pre_activation_layer:
            self.pre_activation_layers.append(pre_activation)

        if self.append_activation_layer:
            self.activation_layers.append(out)

        return out

    def tune(self, learning_rate: float) -> None:
        return None

    def zero_grad(self) -> None:
        self.weights.grad = None
        self.biases.grad = None

    def require_grad(self) -> None:
        self.weights.requires_grad = True
        self.biases.requires_grad = True

    def reset_previous_activations(self):
        self.previous_pre_activation = None

    def params(self) -> List[torch.Tensor]:
        return None