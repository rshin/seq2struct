import torch

from seq2struct.utils import registry

registry.register('optimizer', 'adadelta')(torch.optim.Adadelta)
registry.register('optimizer', 'adam')(torch.optim.Adam)
registry.register('optimizer', 'sgd')(torch.optim.SGD)