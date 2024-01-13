import torch

class LayerNorm:

    def __init__(self, normalized_shape: int, eps=1e-5):
        """
        :param normalized_shape:
        """
        self.weights = torch.ones(normalized_shape)
        self.bias = torch.zeros(normalized_shape)
        self.epsilon = eps # Small constant for numerical stability

    def __call__(self, input: torch.Tensor):
        """

        :param input: Tensor with dim [Batch, Token Length, Embedding_dim]
        :return:
        """
        mean = torch.mean(input, dim=2, keepdim=True)
        variance = torch.var(input, dim=2, keepdim=True)
        normalized = (input - mean) / torch.sqrt(variance + self.epsilon)
        return self.weights * normalized + self.bias

    def zero_grad(self):
        self.weights.grad = None
        self.bias.grad = None

    def tune(self, learning_rate: float):
        self.weights.data += learning_rate * (-1 * self.weights.grad)
        self.bias.data += learning_rate * (-1 * self.bias.grad)

    def require_grad(self):
        self.weights.requires_grad = True
        self.bias.requires_grad = True
