from typing import Dict, List
import torch
from torch.functional import F

import layer

def generate(
          emd: torch.Tensor
        , layers: List[layer.RecurrentLayer]
        , hp: Dict
        , max_num_of_tokens: int) -> List:

    sampled_text = []
    # start with '/n' characters
    context = hp["token_length"] * [0]

    for _ in range(0, max_num_of_tokens):

        embedding = emd[context]
        inputs_encoded = embedding.view(-1, hp["token_length"] * hp["dim_of_embedding"])

        ## Forward ##
        out = inputs_encoded
        for layer in layers:
            out = layer(out) # final output is our logits

        sampled_char = torch.multinomial(F.softmax(out, dim=1), num_samples=1).item()
        sampled_text.append(sampled_char)
        context = context[1:] + [sampled_char]

    return sampled_text

