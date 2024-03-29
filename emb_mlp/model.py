from typing import Callable, List, Dict
import torch
import torch.nn.functional as F

from base.abstract import Model
from base.mlp import MLP


class EmbMLP(Model):

    def __init__(self, hp: Dict, model: MLP, generator: torch.Generator = None, **kwargs):
        self.hp = hp
        self.embedding = torch.randn(
            (kwargs["num_of_unique_chars"], hp["dim_of_embedding"])
            , generator=generator
            , dtype=torch.float64)
        self.mlp = model

    def require_grad(self):
        self.embedding.requires_grad = True
        self.mlp.require_grad()

    def zero_grad(self):
        self.embedding.grad = None
        self.mlp.zero_grad()

    def forward(self, inputs_idx: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: 2D tensor with dim: [Batch, Index of inputs]
        :return: [Batch, Logits]
        """
        embedding = self.embedding[inputs_idx]
        inputs_encoded = embedding.view(-1, self.hp["token_length"] * self.hp["dim_of_embedding"])
        return self.mlp.forward(inputs_encoded)

    def tune(self, learning_rate: float) -> None:
        self.embedding.data += learning_rate * (-1 * self.embedding.grad)
        self.mlp.tune(learning_rate)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.mlp.loss(logits, targets)

    def params(self) -> List[Dict]:
        layer_dict = {
            "layer": self.__class__.__name__
            , "layer_num": 1
            , "embedding": self.embedding
        }
        params = self.mlp.params()
        for param in params:
            param["layer_num"] += 1

        return [layer_dict] + params

    def disable_logging(self) -> None:
        self.mlp.disable_logging()

    @torch.no_grad
    def generate(self, max_num_of_tokens: int) -> List:
        sampled_text = []

        # start with '/n' characters ([0])
        context = torch.tensor(self.hp["token_length"] * [0], dtype=torch.int)

        for _ in range(0, max_num_of_tokens):
            out = self.forward(context) # only one batch
            sampled_char = torch.multinomial(F.softmax(out, dim=1), num_samples=1).item()
            sampled_text.append(sampled_char)
            context = torch.tensor(context[1:].tolist() + [sampled_char], dtype=torch.int)

        return sampled_text

    @torch.no_grad
    def dataset_loss(self, all_inputs: torch.Tensor, all_targets: torch.Tensor) -> torch.Tensor:
        out = self.forward(all_inputs)
        return self.loss(out, all_targets)