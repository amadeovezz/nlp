from typing import Callable, Tuple, List
import torch


def build_dataset(encoded_data: List, block_size: int):
    Inputs, Targets = [], []
    context = [0] * block_size

    for ix in encoded_data:
        Inputs.append(context)
        Targets.append(ix)

        # Update the context
        context = context[1:] + [ix]

    return torch.tensor(Inputs), torch.tensor(Targets)
