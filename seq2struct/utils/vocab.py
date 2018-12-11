import collections.abc


class Sentinel(object):
    '''Used to represent special values like UNK.'''
    # pylint: disable=too-few-public-methods
    __slots__ = ('name', )

    def __init__(self, name):
        # type: (str) -> None
        self.name = name

    def __repr__(self):
        # type: () -> str
        return '<' + self.name + '>'

    def __lt__(self, other):
        # type: (object) -> bool
        if isinstance(other, IndexedSet.Sentinel):
            return self.name < other.name
        return True


UNK = Sentinel('UNK')
BOS = Sentinel('BOS')
EOS = Sentinel('EOS')


class Vocab(collections.abc.Set):

    def __init__(self, iterable, special_elems=(UNK, BOS, EOS)):
        # type: (Iterable[T]) -> None
        elements = list(special_elems)

        iterable_list = list(iterable)
        assert len(iterable_list) == len(set(iterable_list))
        elements.extend(iterable_list)

        self.id_to_elem = {i: elem for i, elem in enumerate(elements)}
        self.elem_to_id = {elem: i for i, elem in enumerate(elements)}

    def __iter__(self):
        # type: () -> Iterator[Union[T, IndexedSet.Sentinel]]
        for i in xrange(len(self)):
            yield self.id_to_elem[i]

    def __contains__(self, value):
        # type: (object) -> bool
        return value in self.elem_to_id

    def __len__(self):
        # type: () -> int
        return len(self.elem_to_id)

    def __getitem__(self, key):
        # type: (int) -> Union[T, IndexedSet.Sentinel]
        if isinstance(key, slice):
            raise TypeError('Slices not supported.')
        return self.id_to_elem[key]

    def index(self, value):
        # type: (T) -> int
        try:
            return self.elem_to_id[value]
        except KeyError:
            return self.elem_to_id[UNK]

    def indices(self, values):
        # type: (Iterable[T]) -> List[int]
        return [self.index(value) for value in values]

    def __hash__(self):
        # type: () -> int
        return id(self)
