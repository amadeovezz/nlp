from typing import List, Dict
import matplotlib.pyplot as plt
import torch

from base import abstract

# visualize dimensions 0 and 1 of the embedding matrix C for all characters
def plot_2d_char_embedding(unique_chars: List, embedding: torch.Tensor) -> None:
    itos = {i: ch for i, ch in enumerate(unique_chars)}
    plt.figure(figsize=(8,8))
    plt.scatter(embedding[:,0].data, embedding[:,1].data, s=200)
    for i in range(embedding.shape[0]):
        plt.text(embedding[i,0].item(), embedding[i,1].item(), itos[i], ha="center", va="center", color='white')
    plt.grid('minor')


def plot_activation_distributions(layers: List[abstract.Layer]) -> None:
    plt.figure(figsize=(20, 4))  # width and height of the plot
    legends = []
    for i, layer in enumerate(layers):
        if layer.activation_layers is None:
            print("No activation layers found, make sure to enable: append_activation_layer=True")
            return
        t = layer.activation_layers[-1]
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (
        i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean() * 100))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__}')
    plt.legend(legends);
    plt.title('Activation distribution')


def plot_activation_grad_distributions(layers: List[abstract.Layer]) -> None:
    # visualize histograms
    plt.figure(figsize=(20, 4))  # width and height of the plot
    legends = []
    for i, layer in enumerate(layers):
        if layer.activation_layers is None:
            print("No activation layers found, make sure to enable: append_activation_layer=True")
            return
        if layer.activation_layers[-1].grad is None:
            print("No activation layer gradients found, make sure to enable: retain_activations=True")
            return
        t = layer.activation_layers[-1].grad
        print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__}')
    plt.legend(legends);
    plt.title('Activation gradient distribution')


def plot_parameters_grads(params: List[Dict]) -> None:
    # visualize histograms
    plt.figure(figsize=(20, 4))  # width and height of the plot
    legends = []
    for i, param in enumerate(params):
        if "weights" in param.keys() and "LinearLayer" in param.values():
            data = param["weights"]
            grad = data.grad
            print('layer: %s | weight %10s | mean %+f | std %e | grad:data ratio %e' % (
            param["layer"], tuple(data.shape), grad.mean(), grad.std(), grad.std() / data.std()))
            hy, hx = torch.histogram(grad, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'param {i} {tuple(data.shape)}')
    plt.legend(legends)
    plt.title('weights gradient distribution');