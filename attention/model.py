from typing import Callable, List, Dict
import torch
import torch.nn.functional as F
from torch import nn

from base.abstract import Model
from base.mlp import MLP, LinearLayer
from base.norm import LayerNorm

from attention import BatchedSelfAttentionHead


class SimpleTransformer(Model):

    def __init__(self, hp: Dict, mlp: MLP, **kwargs):
        """
        :param hp: hyperparameters. Some notworthy ones:
            - dim_of_embedding: the dimension of the original embedding + positional encoding
            - dim_of_attention_embedding: the desired output of the embedding after an attention head. Note for
            multi-headed attention, the increase or decrease in dimensionality occurs with projection of a linear layer
            instead of within the attention head itself.
        :param mlp:
        :param kwargs: 
        """
        self.hp = hp
        self.positional_encoding_func = kwargs['positional_encoding_func']
        self.embedding = torch.randn(
            (kwargs["num_of_unique_chars"], hp["dim_of_embedding"])
            , dtype=torch.float64)

        self.attention_head = BatchedSelfAttentionHead(
                emb_dim=hp["dim_of_embedding"],
                out_dimension=["dim_of_attention_embedding"],
                block_type=kwargs["attention_block_type"]
        )

        self.layer_norm = LayerNorm(hp["dim_of_attention_embedding"], eps=0)
        self.mlp = mlp

    def require_grad(self):
        self.embedding.requires_grad = True
        self.attention_head.require_grad()
        self.layer_norm.require_grad()
        self.mlp.require_grad()

    def zero_grad(self):
        self.embedding.grad = None
        self.attention_head.zero_grad()
        self.layer_norm.zero_grad()
        self.mlp.zero_grad()

    def forward(self, inputs_idx: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: tensor with dim: [Batch, Index of inputs (Tokens)]
        :return: [Batch, Token, Logits]
        """
        # embedding [Batch, Tokens, Embedding]
        token_embedding = self.embedding[inputs_idx]

        # positional embedding
        positional_embedding = self.positional_encoding_func(token_embedding[0].shape[0], self.hp["dim_of_embedding"])

        embedding = token_embedding + positional_embedding # this will be broadcasted

        # single headed attention
        attention_embedding = self.attention_head(embedding)[0]

        # residual connection + layer norm
        add_norm_embedding = attention_embedding + self.layer_norm(attention_embedding)

        # mlp forward
        return self.mlp.forward(add_norm_embedding)

    def inspect_attention(self, inputs_idx: torch.Tensor) -> torch.Tensor:
        # embedding [Batch, Tokens, Embedding]
        token_embedding = self.embedding[inputs_idx]

        # positional embedding
        positional_embedding = self.positional_encoding_func(token_embedding[0].shape[0], self.hp["dim_of_embedding"])

        embedding = token_embedding + positional_embedding  # this will be broadcasted

        # attention head
        return self.attention_head(embedding)

    def tune(self, learning_rate: float) -> None:
        self.embedding.data += learning_rate * (-1 * self.embedding.grad)
        self.attention_head.tune(learning_rate)
        self.layer_norm.tune(learning_rate)
        self.mlp.tune(learning_rate)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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

class Transformer(SimpleTransformer):

    def __init__(self, hp: Dict, mlp: MLP, **kwargs):
        """
        A transformer with more bells and whistles.
        Assumes multi-headed attention.

        :param hp: hyperparameters. Some noteworthy ones:
            - dim_of_embedding: the dimension of the original embedding + positional encoding
            - dim_of_attention_embedding: the desired output of the embedding after multi-headed attention.
            This is determined by a projection after the attention blocks.
        :param mlp:
        :param kwargs:
        """
        self.hp = hp
        self.positional_encoding_func = kwargs['positional_encoding_func']
        self.embedding = torch.randn(
            (kwargs["num_of_unique_chars"], hp["dim_of_embedding"])
            , dtype=torch.float64)

        self.dim_of_attention_head = hp['dim_of_embedding'] // hp["num_of_attention_heads"]
        assert self.dim_of_attention_head % 2 == 0
        self.attention_heads = [
            BatchedSelfAttentionHead(
                emb_dim=self.dim_of_attention_head,
                out_dimension=self.dim_of_attention_head,
                block_type=kwargs["attention_block_type"]
            ) for _ in range(hp["num_of_attention_heads"])]

        self.linear_projection = LinearLayer(
            num_of_inputs=hp['dim_of_embedding'],
            num_of_neurons=hp["dim_of_attention_embedding"],
            activation_func=torch.tanh,
        )

        self.layer_norm_1 = LayerNorm(hp["dim_of_attention_embedding"], eps=0)
        #self.layer_norm_2 = LayerNorm(kwargs['num_of_unique_chars'], eps=0)
        self.mlp = mlp

    def require_grad(self):
        self.embedding.requires_grad = True
        for attn in self.attention_heads:
            attn.require_grad()
        self.linear_projection.require_grad()
        self.layer_norm_1.require_grad()
       #self.layer_norm_2.require_grad()
        self.mlp.require_grad()

    def zero_grad(self):
        self.embedding.grad = None
        for attn in self.attention_heads:
            attn.zero_grad()
        self.linear_projection.zero_grad()
        self.layer_norm_1.zero_grad()
        #self.layer_norm_2.zero_grad()
        self.mlp.zero_grad()

    def forward(self, inputs_idx: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: tensor with dim: [Batch, Index of inputs (Tokens)]
        :return: [Batch, Token, Logits]
        """
        # embedding [Batch, Tokens, Embedding]
        token_embedding = self.embedding[inputs_idx]

        # positional embedding
        positional_embedding = self.positional_encoding_func(token_embedding[0].shape[0], self.hp["dim_of_embedding"])

        embedding = token_embedding + positional_embedding # this will be broadcasted

        # attention blocks
        for _ in range(0, self.hp["num_of_attention_blocks"]):

            # multi-headed attention
            # we need to break up the dimensionality of our vector
            split_embedding = torch.split(embedding, self.dim_of_attention_head, dim=-1)
            split_attention_embedding = [head(emb)[0] for head, emb in zip(self.attention_heads, split_embedding)]
            attention_embedding_cat = torch.cat(split_attention_embedding, dim=-1)
            attention_embedding = self.linear_projection(attention_embedding_cat)

            # residual connection + layer norm
            add_norm_embedding = attention_embedding + self.layer_norm_1(attention_embedding)

            # mlp
            out = self.mlp.forward(add_norm_embedding)

            # residual connection + layer norm
            #final = out + self.layer_norm_2(out)

        return out

    def tune(self, learning_rate: float) -> None:
        self.embedding.data += learning_rate * (-1 * self.embedding.grad)
        for attn in self.attention_heads:
            attn.tune(learning_rate)
        self.linear_projection.tune(learning_rate)
        self.layer_norm_1.tune(learning_rate)
        #self.layer_norm_2.tune(learning_rate)
        self.mlp.tune(learning_rate)
