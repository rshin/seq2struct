import attr
import collections
import inspect
import itertools

import torch

from . import graph

@attr.s(frozen=True)
class CallableRef:
    signature = attr.ib()

    def find(self, root_module):
        raise NotImplementedError
    
    def extend(self, key, signature):
        '''Get a reference to a subpart of the callable.
        
        Examples:
        - For a class, an attribute of the class
        - For a class instance, an attribute of the instance
        '''

        raise NotImplementedError


@attr.s(frozen=True)
class AttrRef(CallableRef):
    ref_path = attr.ib()

    def find(self, root_module):
        result = root_module
        for elem in self.ref_path:
            result = getattr(result, elem)
        return result

    def extend(self, key, signature):
        return AttrRef(signature, self.ref_path + (key,))


@attr.s(frozen=True)
class WrapperRef(CallableRef):
    value = attr.ib()

    def find(self, root_module):
        return self.value


@attr.s(frozen=True)
class BatchKey:
    '''Determines which computations for a given callable_ref to batch together, and how.'''

    @classmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError

    @property
    def iterable_keys(self):
        '''Valid keys to use in iteration of an associated ResultHandle.'''
        # TODO: Move out of this class.
        raise NotImplementedError

    def call_batched(self, existing_results, callable, callable_ref, invocations):
        args, kwargs = self._regroup_invocations(invocations)

        batched_args = [
            self._combine(self._to_value(existing_results, v) for v in arg) for arg in args
        ]
        batched_kwargs = {
            k: self._combine(self._to_value(existing_results, v) for v in arg)
            for k, arg in kwargs.items()
        }
        
        return self._separate(callable(*args, **kwargs))
    
    def _regroup_invocations(self, invocations):
        args_count = set(len(i.args) for i in invocations)
        kwargs_keys = set(i.kwargs.keys() for i in invocations)
        assert len(args_count) == 1
        assert len(kwargs_keys) == 1
        args_count = args_count.pop()
        kwargs_keys = kwargs_keys.pop()
        args = [[invocation.args[i] for invocation in invocations]
                for i in range(args_count)]
        kwargs = {
            k: [invocation.kwargs[k] for invocation in invocations]
            for k in kwargs_keys}
        return args, kwargs
        
    def _regroup_invocations_with_signature(self, signature, invocations):
        # Assume there are no *args, **kwargs in signature
        assert not any(
            param.kind in (VAR_POSITIONAL, VAR_KEYWROD) for param in signature.parameters.values())

        combined = inspect.BoundArguments(
            signature,
            collections.OrderedDict((param_name, []) for param_name in signature.parameters))
        for invocation in invocations:
            bound = signature.bind(*invocation.args, **invocation.kawrgs)
            bound.apply_defaults()
            for k, v in combined.items():
                v.append(bound.arguments[k])

        return combined

    def _to_value(self, existing_results, handle_or_value):
        if isinstance(handle_or_value, ResultHandle):
            value = existing_results[handle_or_value.node]
            for f in handle_or_value.accessor:
                value = f(value)
            return value
        return handle_or_value

    def _combine(self, items):
        raise NotImplementedError

    def _separate(self, combined):
        raise NotImplementedError


class BatchingPolicy:

    _STACK = []
    _DEFAULTS = {}

    def __init__(
            self, 
#            default_collator=ConcatBatchCollator, 
#            custom_collators={},
        ):
        #self.default_collator = default_collator
        #self.custom_collators = custom_collators
        pass

     # TODO: Convert this to "get_batch_key_factory"?
    def get_batch_key_type(self, value, callable_ref):
        # BatchKey member
        result = getattr(value, 'BatchKey', None)
        if result is not None:
            return result
        
        # Defaults
        result = self._DEFAULTS.get(value)
        if result is not None:
            return result
        result = self._DEFAULTS.get(type(value))
        if result is not None:
            return result

        ## - Registered here
        #result = self.custom_collators.get(orig_type)
        #if result is not None:
        #    return result
        #result = self.custom_collators.get(ref_path)
        #if result is not None:
        #    return result
        raise KeyError
        #return self.default_collator

    # Use as a context manager
    def __enter__(self):
        BatchingPolicy._STACK.append(self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        BatchingPolicy._STACK.pop()

    @classmethod
    def active(cls):
        if cls._STACK:
            return cls._STACK[-1]
        return None
    
    @classmethod
    def register_default_for_type(cls, orig_type, batch_key_type):
        cls._DEFAULTS[orig_type] = batch_key_type


def with_batch_key(value, batch_key_type):
    batching_policy = BatchingPolicy.active()
    if not batching_policy:
        return value
    
    value.BatchKey = batch_key_type
    return value


@attr.s(frozen=True)
class Schedule:
    actions = attr.ib()
    precomputed = attr.ib()


class BatchScheduler:

    _STACK = []

    # TODO: Specify relevant Torch functions more robustly.
    # TODO: Allow customization of overrides.
    #_patches = [
    #    unittest.mock.patch('torch.cat', stub_function_creator(torch.cat, BatchingForTorchCat)),
    #    unittest.mock.patch('torch.nn.functional.logsigmoid', stub_function_creator(torch.nn.functional.logsigmoid, ConcatBatchCollator)),
    #]

    def __init__(self, policy):
        self.enqueued_nodes = []
        self.precomputed_values = {}
        self.policy = policy

    def serialize(self):
        depths_by_key = collections.defaultdict(list)
        for node in self.enqueued_nodes:
            depths_by_key[node.callable_ref, node.batch_key].append(depth)
        mean_depth_by_key = {k: sum(v) / len(v) for k, v in depths_by_key.items()}

        schedule = []
        agenda = collections.defaultdict(list)
        while self.enqueued_nodes or agenda:
            # Update agenda
            remaining_nodes = []
            for node in self.enqueued_nodes:
                if node.num_incoming == 0:
                    agenda[node.callable_ref, node.batch_key].append(node)
                else:
                    remaining_nodes.append(node)
            self.enqueued_nodes = remaining_nodes

            # Run computation on best group of nodes from agenda
            callable_ref, batch_key = min(agenda, key=lambda k: mean_depth_by_key[k])

            # TODO: Precompute how to merge these nodes and execute the computation.
            schedule.append((callable_ref, batch_key, agenda[batch_key]))

            # Mark these nodes as computed
            for node in nodes:
                for next_node in node.outgoing:
                    next_node.num_incoming -= 1
            del agenda[callable_ref, batch_key]
        
        return schedule

    def schedule_node(self, callable_ref, batch_key, args, kwargs):
        # TODO: Normalize args and kwargs to specific named arguments in the original module.
        # TODO: see if we've already computed this value somewhere else
        node = graph.Node(args, kwargs, callable_ref, batch_key)

        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, graph.ResultHandle):
                node.num_incoming += 1
                node.depth = max(node.depth, arg.node.depth + 1)
                arg.node.outgoing.append(node)

            # Check that no Tensors have requires_grad set to True
            elif isinstance(arg, torch.Tensor):
                if arg.requires_grad:
                    raise ValueError('Only constant tensors should be used')

        self.enqueued_nodes.append(node)
        return graph.ResultHandle(node)
    
    def add_constant(self, value):
        node = graph.Node(None, None, None, None)
        self.precomputed_values[node] = value
        return graph.ResultHandle(node)

    # Use as a context manager
    def __enter__(self):
        BatchScheduler._STACK.append(self)
        self.policy.__enter__()
        #for p in self._patches:
        #    p.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        BatchScheduler._STACK.pop()
        self.policy.__exit__(exc_type, exc_value, traceback)
        #for p in self._patches:
        #    p.stop()

    @classmethod
    def active(cls):
        if cls._STACK:
            return cls._STACK[-1]
        return None
    
#@attr.s(frozen=True)
#class MethodRef(CallableRef):
#    method_name = attr.ib()
#
#    def find(self, root_module):
#        return getattr(super().find(root_module), self.method_name)
