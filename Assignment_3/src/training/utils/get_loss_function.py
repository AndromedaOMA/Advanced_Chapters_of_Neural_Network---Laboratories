import sys

import torch


def get_loss_function(name):
    name = name.lower()
    if name == 'crossentropyloss':
        print('Loss function loaded!')
        return torch.nn.CrossEntropyLoss()
    elif name == 'mseloss':
        print('Loss function loaded!')
        return torch.nn.MSELoss()
    else:
        print('The loss function name you have entered is not supported!')
        sys.exit()
