"Handle AST objects."

import ast
# pylint: disable=unused-import
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union
# pylint: enable=unused-import

import asdl


class ASTWrapperVisitor(asdl.VisitorBase):
    '''Used by ASTWrapper to collect information.

    - put constructors in one place.
    - checks that all fields have names.
    - get all optional fields.
    '''

    def __init__(self):
        # type: () -> None
        super(ASTWrapperVisitor, self).__init__()
        self.constructors = {}  # type: Dict[str, asdl.Constructor]
        self.sum_types = {}  # type: Dict[str, asdl.Sum]
        self.product_types = {}  # type: Dict[str, asdl.Product]
        self.fieldless_constructors = {}  # type: Dict[str, asdl.Constructor]
        self.optional_fields = []  # type: List[Tuple[str, str]]
        self.sequential_fields = []  # type: List[Tuple[str, str]]

    def visitModule(self, mod):
        # type: (asdl.Module) -> None
        for dfn in mod.dfns:
            self.visit(dfn)

    def visitType(self, type_):
        # type: (asdl.Type) -> None
        self.visit(type_.value, str(type_.name))

    def visitSum(self, sum_, name):
        # type: (asdl.Sum, str) -> None
        self.sum_types[name] = sum_
        for t in sum_.types:
            self.visit(t, name)

    def visitConstructor(self, cons, _name):
        # type: (asdl.Constructor, str) -> None
        self.constructors[cons.name] = cons
        if not cons.fields:
            self.fieldless_constructors[cons.name] = cons
        for f in cons.fields:
            self.visit(f, cons.name)

    def visitField(self, field, name):
        # type: (asdl.Field, str) -> None
        # pylint: disable=no-self-use
        if field.name is None:
            raise ValueError('Field of type {} in {} lacks name'.format(
                field.type, name))
        if field.opt:
            self.optional_fields.append((name, field.name))
        elif field.seq:
            self.sequential_fields.append((name, field.name))

    def visitProduct(self, prod, name):
        # type: (asdl.Product, str) -> None
        self.product_types[name] = prod
        for f in prod.fields:
            self.visit(f, name)


SingularType = Union[asdl.Constructor, asdl.Product]


class ASTWrapper(object):
    '''Provides helper methods on the ASDL AST.'''

    # pylint: disable=too-few-public-methods

    def __init__(self, ast_def, root_type=None):
        # type: (asdl.Module, str) -> None
        self.ast_def = ast_def
        self._root_type = root_type

        visitor = ASTWrapperVisitor()
        visitor.visit(ast_def)

        self.constructors = visitor.constructors
        self.sum_types = visitor.sum_types
        self.product_types = visitor.product_types

        # Product types and constructors:
        # no need to decide upon a further type for these.
        self.singular_types = {}  # type: Dict[str, SingularType]
        self.singular_types.update(self.constructors)
        self.singular_types.update(self.product_types)

        # IndexedSets for each sum type
        self.sum_type_vocabs = {
            name: sorted(t.name for t in sum_type.types)
            for name, sum_type in self.sum_types.items()
        }
        self.fieldless_constructors = sorted(
            visitor.fieldless_constructors.keys())
        self.optional_fields = sorted(visitor.optional_fields)
        self.sequential_fields = sorted(visitor.sequential_fields)

    @property
    def types(self):
        # type: () -> Dict[str, Union[asdl.Sum, asdl.Product]]
        return self.ast_def.types

    @property
    def root_type(self):
        # type: () -> str
        return self._root_type


# Improve this when mypy supports recursive types.
Node = Dict[str, Any]


def verify_ast(ast_def, node, expected_type=None, field_path=()):
    # type: (ASTWrapper, Node, Optional[str], Tuple[str, ...]) -> None
    # pylint: disable=too-many-branches
    '''Checks that `node` conforms to the ASDL provided in `ast_def`.'''
    if node is None:
        raise ValueError('node is None. path: {}'.format(field_path))
    if not isinstance(node, dict):
        raise ValueError('node is type {}. path: {}'.format(
            type(node), field_path))

    node_type = node['_type']  # type: str
    if expected_type is not None:
        sum_product = ast_def.types[expected_type]
        if isinstance(sum_product, asdl.Product):
            if node_type != expected_type:
                raise ValueError(
                    'Expected type {}, but instead saw {}. path: {}'.format(
                        expected_type, node_type, field_path))
        elif isinstance(sum_product, asdl.Sum):
            possible_names = [t.name
                              for t in sum_product.types]  # type: List[str]
            if node_type not in possible_names:
                raise ValueError(
                    'Expected one of {}, but instead saw {}. path: {}'.format(
                        ', '.join(possible_names), node_type, field_path))

        else:
            raise ValueError('Unexpected type in ASDL: {}'.format(sum_product))

    if node_type in ast_def.types:
        # Either a product or a sum type; we want it to be a product type
        sum_product = ast_def.types[node_type]
        if isinstance(sum_product, asdl.Sum):
            raise ValueError('sum type {} not allowed as node type. path: {}'.
                             format(node_type, field_path))
        fields_to_check = sum_product.fields
    elif node_type in ast_def.constructors:
        fields_to_check = ast_def.constructors[node_type].fields
    else:
        raise ValueError('Unknown node_type {}. path: {}'.format(node_type,
                                                                 field_path))

    for field in fields_to_check:
        if not field.opt and not field.seq and (field.name not in node or
                                                node[field.name] is None):
            raise ValueError('required field {} is missing. path: {}'.format(
                field.name, field_path))
        if field.seq and field.name in node and not isinstance(
                node[field.name], (list, tuple)):  # noqa: E125
            raise ValueError('sequential field {} is not sequence. path: {}'.
                             format(field.name, field_path))

        # Check that each item in this field has the expected type.
        items = node.get(field.name,
                         ()) if field.seq else (node.get(field.name), )
        item_type = {
            'identifier': lambda x: isinstance(x, str),
            'int': lambda x: isinstance(x, int),
            'string': lambda x: isinstance(x, str),
            'bytes': lambda x: isinstance(x, bytes),
            'object': lambda x: isinstance(x, object),
            'singleton': lambda x: x is True or x is False or x is None
        }

        # pylint: disable=cell-var-from-loop
        if field.type in item_type:
            check = item_type[field.type]
        else:
            # pylint: disable=line-too-long
            check = lambda n: verify_ast(ast_def, n, field.type, field_path + (field.name, ))  # noqa: E731,E501

        for item in items:
            if item is None:
                continue
            check(item)


def convert_native_ast(node):
    # type: (ast.AST) -> Dict[str, Any]
    result = {'_type': node.__class__.__name__}  # type: Dict[str, Any]
    for field, value in ast.iter_fields(node):
        if isinstance(value, (list, tuple)):
            result[field] = [convert_native_ast(v) for v in value]
        elif not isinstance(value, ast.AST):
            result[field] = value
        else:
            result[field] = convert_native_ast(value)
    return result
