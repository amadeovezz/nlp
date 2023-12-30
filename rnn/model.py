from typing import Callable, List, Dict
import torch
import torch.nn.functional as F

from layer import RecurrentLayer

from base.abstract import Model

class RNN(Model):

    def __init__(self, hp: Dict, emd: torch.Tensor, layers: List[RecurrentLayer]):
        self.hp = hp
        self.embedding = emd
        self.layers = layers

    def require_grad(self):
        for layer in self.layers:
            layer.require_grad()
        self.embedding.requires_grad = True

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
        self.embedding.grad = None

    def reset_previous_activations(self):
        for layer in self.layers:
            layer.reset_previous_activations()

    def forward(self, inputs_idx: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: Pytorch tensor with dim: [Batch, Index of inputs]
        :return: [Batch, Logits]
        """

        # reset previous activations
        self.reset_previous_activations()

        # embedding
        batched_embedding = self.embedding[inputs_idx]

        # [Batch, Token, Embedding] -> [Token, Batch, Embedding]
        tokens_by_batch = batched_embedding.transpose(0,1)

        # forward
        for tokens in tokens_by_batch:
            out = tokens
            for layer in self.layers:
                out = layer(out)  # final output is our logits

        return out


    @torch.no_grad
    def generate(self, max_num_of_tokens: int) -> List:
        sampled_text = []

        # start with '/n' characters
        # extra brackets is so we can use our batched forward pass
        context = torch.tensor([self.hp["token_length"] * [0]], dtype=torch.int)

        for _ in range(0, max_num_of_tokens):
            out = self.forward(context)
            sampled_char = torch.multinomial(F.softmax(out, dim=1), num_samples=1).item()
            sampled_text.append(sampled_char)
            context = torch.tensor([context[0][1:].tolist() + [sampled_char]], dtype=torch.int)

        return sampled_text


    def tune(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.weights.data += learning_rate * (-1 * layer.weights.grad)
            layer.biases.data += learning_rate * (-1 * layer.biases.grad)
        self.embedding.data += learning_rate * (-1 * self.embedding.grad)

    def loss(self, out: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out, targets)

    @torch.no_grad
    def dataset_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        out = self.forward(inputs)
        return self.loss(out, targets)
