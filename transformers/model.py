from typing import Callable, List, Dict
import torch
import torch.nn.functional as F

from base.abstract import Model
from base.mlp import MLP

from attention import BatchedAttentionHead


class Attention(Model):

    def __init__(self, hp: Dict, mlp: MLP, **kwargs):
        self.hp = hp
        self.embedding = torch.randn(
            (kwargs["num_of_unique_chars"], hp["dim_of_embedding"])
            , requires_grad=True
            , dtype=torch.float64)
        self.attention_head = BatchedAttentionHead(
            emb_dim=hp['dim_of_embedding'],
            out_dimension=hp["dim_of_attention_embedding"] ,
        )
        self.mlp = mlp

    def require_grad(self):
        self.embedding.requires_grad = True
        self.attention_head.require_grad()
        self.mlp.require_grad()

    def zero_grad(self):
        self.mlp.zero_grad()
        self.embedding.grad = None

    def forward(self, inputs_idx: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: tensor with dim: [Batch, Index of inputs]
        :return: [Batch, Logits]
        """
        embedding = self.embedding[inputs_idx]
        inputs_encoded = embedding.view(-1, self.hp["token_length"] * self.hp["dim_of_embedding"])
        return self.mlp.forward(inputs_encoded)

    def tune(self, learning_rate: float) -> None:
        self.mlp.tune(learning_rate)
        self.embedding.data += learning_rate * (-1 * self.embedding.grad)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.mlp.loss(logits, targets)

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