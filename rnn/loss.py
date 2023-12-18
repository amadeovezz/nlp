from typing import Dict, List
import torch
from torch.functional import F

import layer


@torch.no_grad()
def loss(data: torch.Tensor
         , targets: torch.Tensor
         , emd: torch.Tensor
         , layers: List[layer.RecurrentLayer]
         , hp: Dict) -> torch.Tensor:

    # [Batch, Token, Embedding]
    batched_embedding = emd[data]

    # [Batch, Token, Embedding] -> [Token, Batch, Embedding]
    tokens_by_batch = batched_embedding.transpose(0,1)

    ## Forward ##
    for token in tokens_by_batch:
        out = token
        for layer in layers:
            out = layer(out)  # final output is our logits

    return F.cross_entropy(out, targets)
