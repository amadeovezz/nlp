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

        for i in range(len(dataset) - 1):  # Iterate until the second last element
            # Create the input sequence for the current index
            input_sequence = dataset[max(0, i + 1 - token_length):i + 1]
            target = dataset[i + 1]
            Inputs.append(input_sequence)
            Targets.append(target)

        # Pad shorter sequences at the beginning
        Inputs = [([0] * (token_length - len(seq)) + seq) for seq in Inputs]

        return torch.tensor(Inputs), torch.Tensor(Targets)

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