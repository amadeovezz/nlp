from typing import Callable, Tuple, List, Dict
import torch

def get_dataset(encoded_data: List, token_length: int, context_window: str = "fixed") -> Dict:

    idx = round(0.85*len(encoded_data))
    train = encoded_data[:idx]
    validation = encoded_data[idx:]

    def fixed_dataset(dataset: List[int]) -> Tuple:
        Inputs, Targets = [], []
        context = [0] * token_length

        for data in dataset:
            Inputs.append(context)
            Targets.append(data)

            # Update the context
            context = context[1:] + [data]

        return torch.tensor(Inputs), torch.tensor(Targets)

    def expanding_dataset(dataset: List[int]) -> Tuple:
        Inputs, Targets = [], []
        context = [0] * (token_length + 1)

        for data in dataset:
            Inputs.append(context[:-1])
            Targets.append(context[1:])

            # Update the context
            context = context[1:] + [data]

        return torch.tensor(Inputs), torch.tensor(Targets)


    if context_window == "fixed":
        return {
            "train": fixed_dataset(train),
            "validation": fixed_dataset(validation)
        }

    elif context_window == "expanding":
        return {
            "train": expanding_dataset(train),
            "validation": expanding_dataset(validation)
        }