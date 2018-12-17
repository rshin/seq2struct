import collections
import collections.abc
import inspect

_REGISTRY = collections.defaultdict(dict)


def register(kind, name):
    kind_registry = _REGISTRY[kind]

    def decorator(obj):
        if name in kind_registry:
            raise LookupError('{} already registered as kind {}'.format(name, kind))
        kind_registry[name] = obj
        return obj

    return decorator


def lookup(kind, name):
    if isinstance(name, collections.abc.Mapping):
        name = name['name']

    if kind not in _REGISTRY:
        raise KeyError('Nothing registered under "{}"'.format(kind))
    return _REGISTRY[kind][name]


def construct(kind, config, **kwargs):
    return instantiate(lookup(kind, config), config, **kwargs)


def instantiate(callable, config, **kwargs):
    merged = {**config, **kwargs}
    signature = inspect.signature(callable)
    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL):
            raise ValueError('Unsupported kind for param {}: {}'.format(name, param.kind))

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return callable(**merged)

    for key in list(merged.keys()):
        if key not in signature.parameters:
            merged.pop(key)
    return callable(**merged)