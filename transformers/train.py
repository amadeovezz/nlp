import torch
from torch.functional import F
from typing import List, Dict, Callable


from mlp.layer import LinearLayer
from attention import BatchedAttentionHead



def sgd(hp: Dict
        , emb: torch.Tensor
        , position_encoding_func:  Callable
        , attention_head: BatchedAttentionHead
        , layers: List[LinearLayer]
        , training_inputs: torch.Tensor
        , training_targets: torch.Tensor
        , loss_list: List):

    for epoch in range(0,hp['epochs']):

        ## Random mini batch ##

        # select 32 random inputs (the indexes) from our training data
        batch_idxs = torch.randint(0, training_inputs.shape[0], (hp['mini_batch_size'],))

        # fetch the actual inputs
        inputs = training_inputs[batch_idxs]

        ## Embeddings ##

        # embedding
        token_embedding = emb[inputs]

        # positional embedding
        positional_embedding = position_encoding_func(token_embedding[0].shape[0], hp["dim_of_embedding"])

        embedding = token_embedding + positional_embedding  # This will be broadcasted

        ## Attention head ##
        attention_embedding = attention_head(embedding)

        ## Forward ##
        out = attention_embedding
        for layer in layers:
            out = layer(out) # final output is our logits

        # Grab the last token in the sequence
        last_token_logits = out[:, hp["token_length"] - 1, :]
        loss = F.cross_entropy(last_token_logits, training_targets[batch_idxs])

        ## Stats ##
        loss_list.append(loss.item())
        #loss_list.append(loss.log10().item())

        ## Backwards ##
        emb.grad = None
        for layer in layers:
            layer.zero_grad()

        loss.backward()

        ## Tune ##
        learning_rate = hp['init_learning_rate'] if epoch < 100000 else hp['converging_learning_rate']
        emb.data += learning_rate * (-1 * emb.grad)
        for layer in layers:
            layer.weights.data += learning_rate * (-1 * layer.weights.grad)
            layer.biases.data += learning_rate * (-1 * layer.biases.grad)

