import copy
import functools
import operator

import attr

from . import scheduling
from . import utils

@attr.s
class Invoke:
    args = attr.ib()
    kwargs = attr.ib()

    def __call__(self, function):
        return function(*args, **kwargs)


@attr.s(cmp=False)
class Node:
    # args and kwargs passed in at the time of invocation.
    args = attr.ib()
    kwargs = attr.ib()

    callable_ref = attr.ib()
    batch_key = attr.ib()
    depth = attr.ib(default=0)

    outgoing = attr.ib(default=attr.Factory(list))
    num_incoming = attr.ib(default=0)


def batched_func(fn):
    '''Schedule a call to `fn` when in batching mode.'''
    if not scheduling.BatchScheduler.active():
        return fn
    
    batching_policy = scheduling.BatchingPolicy.active()
    callable_ref = scheduling.WrapperRef(utils.get_signature(fn), fn)
    batch_key_type = batching_policy.get_batch_key_type(
            fn, callable_ref)
    @functools.wraps(fn)
    def stub(*args, **kwargs):
        batch_key = batch_key_type.create(*args, **kwargs)
        return scheduling.BatchScheduler.active().schedule_node(
                callable_ref, batch_key, args, kwargs)

    return stub


@attr.s
class ResultHandle:
    node = attr.ib()
    accessor = attr.ib(factory=tuple)

    def with_shape(self, *shape):
        copied = copy.copy(self)
        copied.shape = shape
        return copied

    def split(self, num_splits):
        result = []
        for i in range(num_splits):
            copied = copy.copy(self)
            copied.accessor = self.accessor + (operator.itemgetter(i),)
            result.append(copied)
        return tuple(result)

    def __getattr__(self, name):
        return ResultHandle(self.node, self.accessor + (operator.attrgetter(name),))
    
    def __getitem__(self, key):
        return ResultHandle(self.node, self.accessor + (operator.itemgetter(key),))

    def __call__(self, *args, **kwargs):
        return ResultHandle(self.node, self.accessor + (Invoke(args, kwargs),))

    def __iter__(self):
        keys = self.node.batch_key.iterable_keys
        if not keys:
            raise NotImplementedError
        for key in keys:
            yield self[key]
    
    def __sub__(self, other):
        return batched_func(operator.sub)(self, other)
    
    def __neg__(self):
        return batched_func(operator.neg)(self)



def batched_value(tensor):
    scheduler = scheduling.BatchScheduler.active()
    if not scheduler:
        return tensor

    if isinstance(tensor, ResultHandle):
        return tensor
    
    return scheduler.add_constant(tensor)