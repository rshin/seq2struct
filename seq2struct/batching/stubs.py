import functools
import inspect
import itertools

import attr
import torch

from . import scheduling
from . import graph
from . import utils


class Stub:
    '''Replaces torch.nn.Module attributes in BatchedModules.
    
    When called, schedules computation in the graph.'''

    def __init__(self, orig_value, callable_ref):
        self.callable_ref = callable_ref
        self.batch_key_type = scheduling.BatchingPolicy.active().get_batch_key_type(
            orig_value, callable_ref)

    def __call__(self, *args, **kwargs):
        if not scheduling.BatchScheduler.active():
            raise Exception('This module can only be invoked with a BatchScheduler active.')

        batch_key = self.batch_key_type.create(*args, **kwargs)
        return scheduling.BatchScheduler.active().schedule_node(
                self.callable_ref, batch_key, args, kwargs)

    def __getattr__(self, name):
        # TODO: Save original object around so that we can get signature?
        raise NotImplementedError
        return Stub(
            callable_ref=self.callable_ref.extend(None, name), # XXX update signature
            batching_policy=scheduling.BatchingPolicy.active())


@attr.s
class StubModuleListOrDict:
    callable_ref = attr.ib()
    items = attr.ib(factory=dict, init=False)
    
    def __setitem__(self, key, value):
        self.items[key] = Stub(
            orig_value=value,
            callable_ref=self.callable_ref.extend(utils.get_signature(value), key))
    
    def __getitem__(self, key):
        return self.items[key]


class BatchedModule(torch.nn.Module):
    '''A batching-aware torch.nn.Module.
    
    If instantiated while a BatchingPolicy is active, all of its
    torch.nn.Module attributes are replaced with StubModules.
    '''
    
    def __init__(self, batch_mode=False):
        super().__init__()
        self._ref_path = ()

    def __setattr__(self, name, value):
        # TODO: Perform something special for torch.nn.Sequential?
        if isinstance(value, BatchedModule):
            if value._ref_path != ():
                value._ref_path = self._ref_path + (name,)
        elif scheduling.BatchingPolicy.active():
            if isinstance(value, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                assert len(value) == 0  # TODO: Support non-empty modules
                value = StubModuleListOrDict(
                        callable_ref=scheduling.AttrRef(None, self._ref_path + (name,)))
            elif isinstance(value, torch.nn.Module):
                value = Stub(
                    orig_value=value,
                    callable_ref=scheduling.AttrRef(utils.get_signature(value), self._ref_path + (name,)))
        super().__setattr__(name, value)