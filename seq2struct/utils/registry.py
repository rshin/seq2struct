import collections


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
    return _REGISTRY[kind][name]


def construct(kind, config):
    return lookup(kind, config['name'])(config)
