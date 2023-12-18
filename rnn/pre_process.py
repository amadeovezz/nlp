from typing import Callable, Tuple, List, Dict
import torch

def build_dataset(encoded_data: List, token_length: int) -> Dict:

    idx = round(0.85*len(encoded_data))
    train = encoded_data[:idx]
    validation = encoded_data[idx:]

    def generate_data(dataset: List, token_length: int) -> Tuple:
        Inputs, Targets = [], []
        context = [0] * token_length

        for data in dataset:
            Inputs.append(context)
            Targets.append(data)

            # Update the context
            context = context[1:] + [data]

        return torch.tensor(Inputs), torch.tensor(Targets)

    return {
        "train": generate_data(train, token_length),
        "validation": generate_data(validation, token_length)
    }
