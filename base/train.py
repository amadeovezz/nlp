import torch
from typing import List, Dict

from base import abstract


def sgd(hp: Dict
        , model: abstract.Model
        , training_inputs: torch.Tensor
        , training_targets: torch.Tensor
        , loss_list: List
        , stat_list: List
        ):
    """
    :param hp: hyper-params
    :param model: an object that implements the methods in abstract.Model
    :param training_inputs: a 2D Tensor where each row represent a batch and each batch contains
    a set of indexes that represent characters from training data. Ie: [[1,2,3], [3,4,5]]
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

        ## More stats ##
        stat_list.append(
            [
                {
                    "epoch": epoch
                    , "layer": p['layer']
                    , "layer_num": p['layer_num']
                    , "update:data ratio": ((hp['init_learning_rate'] * p['weights'].grad).std() / p['weights'].std())
                } for p in model.params() if p['layer'] in ['LinearLayer']]
        )

        if epoch % hp['epochs_log_interval'] == 0:
            print(f'epoch: {epoch} / {hp["epochs"]}, loss: {loss.item():.4f}')