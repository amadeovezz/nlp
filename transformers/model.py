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
        self.positional_encoding_func = kwargs['positional_encoding_func']

    def require_grad(self):
        self.embedding.requires_grad = True
        self.attention_head.require_grad()
        self.mlp.require_grad()

    def zero_grad(self):
        self.embedding.grad = None
        self.attention_head.zero_grad()
        self.mlp.zero_grad()

    def forward(self, inputs_idx: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: tensor with dim: [Batch, Index of inputs]
        :return: [Batch, Logits]
        """
        # embedding
        token_embedding = self.embedding[inputs_idx]

        # positional embedding
        positional_embedding = self.positional_encoding_func(token_embedding[0].shape[0], self.hp["dim_of_embedding"])

        # this will be broadcasted
        embedding = token_embedding + positional_embedding

        # attention head
        attention_embedding = self.attention_head(embedding)

        # forward
        return self.mlp.forward(attention_embedding)

    def tune(self, learning_rate: float) -> None:
        self.embedding.data += learning_rate * (-1 * self.embedding.grad)
        self.attention_head.tune(learning_rate)
        self.mlp.tune(learning_rate)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor):
        # Grab the last token in the sequence
        Batch, Token, Prob = logits.shape
        logits = logits.view(Batch*Token, Prob)
        targets = targets.view(Batch*Token)
        return F.cross_entropy(logits, targets)

    @torch.no_grad
    def generate(self, max_num_of_tokens: int) -> List:
        sampled_text = []

        # start with '/n' characters
        # extra brackets is so we can use our batched forward pass
        context = torch.tensor([self.hp["token_length"] * [0]], dtype=torch.int)

        for _ in range(0, max_num_of_tokens):
            out = self.forward(context)
            last = out[:, self.hp["token_length"] - 1, :]
            sampled_char = torch.multinomial(F.softmax(last, dim=1), num_samples=1).item()
            sampled_text.append(sampled_char)
            context = torch.tensor([context[0][1:].tolist() + [sampled_char]], dtype=torch.int)

        return sampled_text

    @torch.no_grad
    def dataset_loss(self, all_inputs: torch.Tensor, all_targets: torch.Tensor) -> torch.Tensor:
        out = self.forward(all_inputs)
        return self.loss(out, all_targets)