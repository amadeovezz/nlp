from typing import Callable, List, Dict
import torch
import torch.nn.functional as F
from torch import nn

from base.abstract import Model
from base.mlp import MLP, LinearLayer
from base.norm import LayerNorm

from attention import BatchedSelfAttentionHead


class SimpleTransformer(Model):

    def __init__(self
                 , hp: Dict
                 , mlp: MLP
                 , generator: torch.Generator = None
                 , **kwargs):
        """
        A single block and single headed transformer model.

        :param hp: hyperparameters. Some noteworthy ones:
            - dim_of_embedding: the dimensionality of the embedding pre/post attention
        :param mlp:
        :param kwargs:

        Note: For simplicity (and consistency with original transformer paper) we retain the dimensionality of the
        embedding through the attention layers here.

        """
        self.hp = hp
        self.positional_encoding_func = kwargs['positional_encoding_func']
        self.embedding = torch.randn(
            (kwargs["num_of_unique_chars"], hp["dim_of_embedding"])
            , dtype=torch.float64
            , generator=generator)

        self.attention_head = BatchedSelfAttentionHead(
                emb_dim=hp["dim_of_embedding"],
                out_dimension=hp["dim_of_embedding"],
                block_type=kwargs["attention_block_type"],
                generator=generator
        )

        self.mlp = mlp
        self.layer_norm_1 = LayerNorm(hp["dim_of_embedding"], eps=0)
        self.layer_norm_2 = LayerNorm(hp["dim_of_embedding"], eps=0)
        self.linear_proj = LinearLayer(
             num_of_inputs=hp["dim_of_embedding"]
           , num_of_neurons=kwargs["num_of_unique_chars"]
        )

    def require_grad(self):
        self.embedding.requires_grad = True
        self.attention_head.require_grad()
        self.layer_norm_1.require_grad()
        self.layer_norm_2.require_grad()
        self.linear_proj.require_grad()
        self.mlp.require_grad()

    def zero_grad(self):
        self.embedding.grad = None
        self.attention_head.zero_grad()
        self.layer_norm_1.zero_grad()
        self.layer_norm_2.zero_grad()
        self.linear_proj.zero_grad()
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

        # residual connection + layer norm 1
        add_norm_embedding = self.layer_norm_1(embedding + attention_embedding)

        # feed forward
        mlp_out = self.mlp.forward(add_norm_embedding)

        # residual connection + layer norm 2
        add_norm_mlp_out = self.layer_norm_2(add_norm_embedding + mlp_out)

        # final linear projection
        return self.linear_proj(add_norm_mlp_out)

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
        self.layer_norm_1.tune(learning_rate)
        self.layer_norm_2.tune(learning_rate)
        self.linear_proj.tune(learning_rate)
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

    def __init__(self
                 , hp: Dict
                 , mlp: MLP
                 , generator: torch.Generator() = None
                 , **kwargs):
        """
        A multi-headed, multi-block transformer model

        :param hp: hyperparameters. Some noteworthy ones:

            - dim_of_embedding: the dimensionality of the embedding pre/post attention
        :param mlp:
        :param kwargs:

         Note: For simplicity (and consistency with original transformer paper) we retain the dimensionality of the
        embedding through the attention layers here.
        """
        self.hp = hp
        self.positional_encoding_func = kwargs['positional_encoding_func']
        self.embedding = torch.randn(
            (kwargs["num_of_unique_chars"], hp["dim_of_embedding"])
            , dtype=torch.float64
            , generator=generator
        )

        assert (hp['dim_of_embedding'] / hp["num_of_attention_heads"]) % 1 == 0
        self.dim_of_attention_head = hp['dim_of_embedding'] // hp["num_of_attention_heads"]
        self.attention_heads = [
            BatchedSelfAttentionHead(
                emb_dim=self.dim_of_attention_head,
                out_dimension=self.dim_of_attention_head,
                block_type=kwargs["attention_block_type"],
                generator=generator
            ) for _ in range(hp["num_of_attention_heads"])]

        self.linear_proj_1 = LinearLayer(
            num_of_inputs=hp['dim_of_embedding'],
            num_of_neurons=hp["dim_of_embedding"],
            generator=generator
        )

        self.linear_proj_2 = LinearLayer(
            num_of_inputs=hp['dim_of_embedding'],
            num_of_neurons=kwargs["num_of_unique_chars"],
            generator=generator
        )

        self.layer_norm_1 = LayerNorm(hp["dim_of_embedding"], eps=0)
        self.layer_norm_2 = LayerNorm(hp['dim_of_embedding'], eps=0)
        self.mlp = mlp

    def require_grad(self):
        self.embedding.requires_grad = True
        for attn in self.attention_heads:
            attn.require_grad()
        self.linear_proj_1.require_grad()
        self.linear_proj_2.require_grad()
        self.layer_norm_1.require_grad()
        self.layer_norm_2.require_grad()
        self.mlp.require_grad()

    def zero_grad(self):
        self.embedding.grad = None
        for attn in self.attention_heads:
            attn.zero_grad()
        self.linear_proj_1.zero_grad()
        self.linear_proj_2.zero_grad()
        self.layer_norm_1.zero_grad()
        self.layer_norm_2.zero_grad()
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
            attention_embedding = self.linear_proj_1(attention_embedding_cat)

            # residual connection + layer norm
            add_norm_embedding = self.layer_norm_1(embedding + attention_embedding)

            # mlp
            mlp_out = self.mlp.forward(add_norm_embedding)

            # residual connection + layer norm
            add_norm_mlp_out = self.layer_norm_2(add_norm_embedding + mlp_out)

            # set the out embedding from the previous block as input to the next attention block
            embedding = add_norm_mlp_out

        # final linear projection to generate our logits
        return self.linear_proj_2(embedding)

    def tune(self, learning_rate: float) -> None:
        self.embedding.data += learning_rate * (-1 * self.embedding.grad)
        for attn in self.attention_heads:
            attn.tune(learning_rate)
        self.linear_proj_1.tune(learning_rate)
        self.linear_proj_2.tune(learning_rate)
        self.layer_norm_1.tune(learning_rate)
        self.layer_norm_2.tune(learning_rate)
        self.mlp.tune(learning_rate)
