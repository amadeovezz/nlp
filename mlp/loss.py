from typing import Dict, List
import torch
from torch.functional import F

import layer


@torch.no_grad()
def loss(data: torch.Tensor
         , targets: torch.Tensor
         , emd: torch.Tensor
         , layers: List[layer.LinearLayer]
         , hp: Dict) -> torch.Tensor:

    embedding = emd[data]
    inputs_encoded = embedding.view(-1, hp["token_length"] * hp["dim_of_embedding"])

    ## Forward ##
    out = inputs_encoded
    for layer in layers:
        out = layer(out) # final output is our logits

    return F.cross_entropy(out, targets)


