import torch
from typing import List, Dict

from base import abstract


def sgd(hp: Dict
        , model: abstract.Model
        , training_inputs: torch.Tensor
        , training_targets: torch.Tensor
        , loss_list: List):
    """

    :param hp: hyper-params
    :param model:
    :param training_inputs: a Tensor [[ Context window ]]
    :param training_targets:
    :param loss_list:
    :return:
    """

    for epoch in range(0, hp['epochs']):

        ## Random mini batch ##

        # select random inputs (the indexes) from our training data
        batch_idxs = torch.randint(0, training_inputs.shape[0], (hp['mini_batch_size'],))

        # fetch the actual inputs
        inputs = training_inputs[batch_idxs]

        ## Forward ##
        out = model.forward(inputs)
        loss = model.loss(out, training_targets[batch_idxs])

        ## Stats ##
        loss_list.append(loss.item())
        # loss_list.append(loss.log10().item())

        ## Backwards ##
        model.zero_grad()
        loss.backward()

        ## Tune ##
        learning_rate = hp['init_learning_rate'] if epoch < 100000 else hp['converging_learning_rate']
        model.tune(learning_rate)

        if epoch % 10000 == 0:
            print(f'epoch: {epoch} / {hp["epochs"]}, loss: {loss.item():.4f}')