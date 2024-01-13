from typing import List
import matplotlib.pyplot as plt
import torch


# visualize dimensions 0 and 1 of the embedding matrix C for all characters
def plot_2d_char_embedding(unique_chars: List, embedding: torch.Tensor) -> None:
    itos = {i: ch for i, ch in enumerate(unique_chars)}
    plt.figure(figsize=(8,8))
    plt.scatter(embedding[:,0].data, embedding[:,1].data, s=200)
    for i in range(embedding.shape[0]):
        plt.text(embedding[i,0].item(), embedding[i,1].item(), itos[i], ha="center", va="center", color='white')
    plt.grid('minor')