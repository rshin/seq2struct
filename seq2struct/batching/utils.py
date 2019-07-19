import inspect

import torch


def get_signature(f):
    # If it's a torch.nn.Module and if __call__ isn't overridden, look through to forward
    if isinstance(f, torch.nn.Module) and f.__class__.__call__ is torch.nn.Module.__call__:
        return inspect.signature(f.forward)
    else:
        try:
            return inspect.signature(f)
        except ValueError:
            return None
