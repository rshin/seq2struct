import operator

import attr
import torch

from . import scheduling

class Concat(scheduling.BatchKey):
    '''Default BatchKey for where everything has shape [1, ...].'''
    # TODO: Make usable for torch.nn.Linear by making the number of non-batch dimensions configurable.

    @classmethod
    def create(cls, *args, **kwargs):
        return cls()

    @property
    def iterable_keys(self):
        return None

    def _combine(self, items):
        return torch.cat(list(items), dim=0)

    def _separate(self, combined):
        return torch.split(combined, dim=0)


scheduling.BatchingPolicy.register_default_for_type(torch.nn.Embedding, Concat)
scheduling.BatchingPolicy.register_default_for_type(torch.nn.Linear, Concat)
scheduling.BatchingPolicy.register_default_for_type(torch.nn.CrossEntropyLoss, Concat)
scheduling.BatchingPolicy.register_default_for_type(torch.nn.functional.logsigmoid, Concat)
scheduling.BatchingPolicy.register_default_for_type(operator.sub, Concat)
scheduling.BatchingPolicy.register_default_for_type(operator.neg, Concat)


class Unbatched(scheduling.BatchKey):

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls()
    
    def call_batched(self, existing_results, callable, invocations):
        return [callable(*i.args, **i.kwargs) for i in invocations]


scheduling.BatchingPolicy.register_default_for_type(torch.cat, Unbatched)
scheduling.BatchingPolicy.register_default_for_type(torch.logsumexp, Unbatched)
scheduling.BatchingPolicy.register_default_for_type(torch.mean, Unbatched)