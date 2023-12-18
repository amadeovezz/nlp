import torch
from torch.functional import F
from typing import List, Dict
from layer import RecurrentLayer


def sgd(hp: Dict
        , emb: torch.Tensor
          , layers: List[RecurrentLayer]
          , training_inputs: torch.Tensor
          , training_targets: torch.Tensor
          , loss_list: List):
    for epoch in range(0, hp['epochs']):

        ## Random mini batch ##

        # select 32 random inputs (the indexes) from our training data
        batch_idxs = torch.randint(0, training_inputs.shape[0], (hp['mini_batch_size'],))

        # fetch the actual inputs
        inputs = training_inputs[batch_idxs]

        # embedding
        batched_embedding = emb[inputs]

        # [Batch, Token, Embedding] -> [Token, Batch, Embedding]
        tokens_by_batch = batched_embedding.transpose(0,1)

        ## Forward ##
        for tokens in tokens_by_batch:
            out = tokens
            for layer in layers:
                out = layer(out)  # final output is our logits

        loss = F.cross_entropy(out, training_targets[batch_idxs])
        #loss_list.append(loss.log10().item())
        loss_list.append(loss.item())

        ## Backwards ##
        for layer in layers:
            layer.zero_grad()

        loss.backward()

        ## Tune ##
        learning_rate = hp['init_learning_rate'] if epoch < 100000 else hp['converging_learning_rate']
        emb.data += learning_rate * (-1 * emb.grad)
        for layer in layers:
            layer.weights.data += learning_rate * (-1 * layer.weights.grad)
            layer.biases.data += learning_rate * (-1 * layer.biases.grad)

        if epoch % 10000 == 0:
            print(f'epoch: {epoch}, loss: {loss.item():.4f}')