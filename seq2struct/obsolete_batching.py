# To use:
# - Instantiate a BatchedModule in batch_mode=True
# - Use as normal to compute the loss
# - Figure out what we need to 

import abc
import collections
import functools
import itertools
import operator
import unittest.mock

import attr
import torch


@attr.s(frozen=True)
class Concat(BatchKey):
    orig_type = attr.ib()
    ref_path = attr.ib()
    module_method_name = attr.ib()
    args = attr.ib()
    kwargs = attr.ib()

    @property
    def iterable_keys(self):
        return None

    @classmethod
    def batch_key(cls, orig_type, ref_path, module_method_name, *args, **kwargs):
        return cls(
            orig_type,
            ref_path,
            len(args),
            tuple(kwargs.keys()))
            #tuple(arg.shape for arg in args),
            #tuple((k, v.shape)
            #      for k, v in sorted(kwargs.items())))

    def _combine(self, items):
        return torch.cat(list(items), dim=0)

    def _separate(self, combined):
        return torch.split(combined, dim=0)


@attr.s(frozen=True)
class BatchingForTorchCat(BatchCollator, BatchKey):
    # TODO: Save shapes of tensor
    num_tensors = attr.ib()
    dim = attr.ib()

    @classmethod
    def batch_key(cls, orig_type, ref_path, tensors, **kwargs):
        if 'out' in kwargs:
            assert kwargs['out'] is None
        return cls(num_tensors=len(tensors), dim=kwargs.get('dim', 0))

    def call_batched(self, existing_results, callable, tensors, **kwargs):
        batched_tensors = tuple(
            torch.cat(self._to_value(existing_results, v) for v in arg) for arg in tensors)
        return torch.split(callable(batched_tensors, **kwargs), dim=0)


def stub_function_creator(orig_function, collator):
    @functools.wraps(orig_function)
    def wrapped(*args, **kwargs):
        return schedule_node(collator, orig_function, None, args, kwargs)
    return wrapped