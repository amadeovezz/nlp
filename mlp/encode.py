from typing import Callable, Tuple, List


def get_encoder_decoder(training_data: str, type: str = "character") -> Tuple[Callable, Callable]:
    with open(training_data, 'r', encoding='utf-8') as f:
        text = f.read()

    if type == "character":
        chars = sorted(list(set(text)))
        return char_encoder_decoder(chars)

    if type == "word":
        raise NotImplemented


def char_encoder_decoder(chars: List) -> Tuple[Callable, Callable]:
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encoder = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decoder = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
    return encoder, decoder
