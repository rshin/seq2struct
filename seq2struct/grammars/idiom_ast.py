import collections
import copy
import enum
import functools
import json
import re

import asdl

from seq2struct import ast_util
from seq2struct.utils import registry


class HoleType(enum.Enum):
    ReplaceSelf = 1
    AddChild = 2


class MissingValue:
    pass


@registry.register('grammar', 'idiom_ast')
class IdiomAstGrammar:

    def __init__(self, base_grammar, template_file, root_type=None):
        self.base_grammar = registry.construct('grammar', base_grammar)
        self.templates = json.load(open(template_file))

        self.ast_wrapper = self.base_grammar.ast_wrapper
        self.base_ast_wrapper = copy.deepcopy(self.ast_wrapper)
        self.root_type = self.base_grammar.root_type
        if base_grammar['name'] == 'python':
            self.root_type = 'mod'
        if root_type is not None:
            self.root_type = root_type

        singular_types_with_single_seq_field = set(
            name for name, type_info in self.ast_wrapper.singular_types.items()
            if len(type_info.fields) == 1 and type_info.fields[0].seq)

        templates_by_head_type = collections.defaultdict(list)
        for template in self.templates:
            head_type = template['idiom'][0]
            # head_type can be one of the following:
            # 1. name of a constructor/product with a single seq field. 
            # 2. name of any other constructor/product
            # 3. name of a seq field (e.g. 'Dict-keys'),
            #    when the containing constructor/product contains more than one field
            #    (not yet implemented)
            # For 1 and 3, the template should be treated as a 'seq fragment'
            # which can occur in any seq field of the corresponding sum/product type.
            # However, the NL2Code model has no such notion currently.
            if head_type in singular_types_with_single_seq_field:
                # seq_type could be sum type or product type, but not constructor
                seq_type = self.ast_wrapper.singular_types[head_type].fields[0].type
                templates_by_head_type[seq_type].append((template, True))
                templates_by_head_type[head_type].append((template, False))
            else:
                templates_by_head_type[head_type].append((template, False))
        
        types_to_replace = {}

        for head_type, templates in templates_by_head_type.items():
            constructors, seq_fragment_constructors = [], []
            for template, is_seq_fragment in templates:
                if is_seq_fragment:
                    if head_type in self.ast_wrapper.product_types:
                        seq_type = '{}_plus_templates'.format(head_type)
                    else:
                        seq_type = head_type

                    seq_fragment_constructors.append(
                        self._template_to_constructor(template, '_{}_seq'.format(seq_type)))
                else:
                    constructors.append(self._template_to_constructor(template))

            # head type can be:
            # constructor (member of sum type)
            if head_type in self.ast_wrapper.constructors:
                assert constructors
                assert not seq_fragment_constructors

                self.ast_wrapper.add_constructors_to_sum_type(
                    self.ast_wrapper.constructor_to_sum_type[head_type],
                    constructors)
            
            # sum type
            elif head_type in self.ast_wrapper.sum_types:
                assert not constructors
                assert seq_fragment_constructors
                self.ast_wrapper.add_seq_fragment_type(head_type, seq_fragment_constructors)

            # product type
            elif head_type in self.ast_wrapper.product_types:
                #assert constructors
                #if seq_fragment_constructors:
                #    raise NotImplementedError('seq fragments of product types not supported: {}'.format(head_type))

                # Replace Product with Constructor
                # - make a Constructor
                orig_prod_type = self.ast_wrapper.product_types[head_type]
                new_constructor_for_prod_type = asdl.Constructor(
                    name=head_type, fields=orig_prod_type.fields)
                # - remove Product in ast_wrapper
                self.ast_wrapper.remove_product_type(head_type)

                # Define a new sum type
                # Add the original product type and template as constructors
                name = '{}_plus_templates'.format(head_type)
                self.ast_wrapper.add_sum_type(
                    name,
                    asdl.Sum(types=constructors + [new_constructor_for_prod_type]))
                # Add seq fragment constructors
                self.ast_wrapper.add_seq_fragment_type(name, seq_fragment_constructors)

                # Replace every occurrence of the product type in the grammar
                types_to_replace[head_type] = name
            
            # built-in type
            elif head_type in self.ast_wrapper.primitive_types:
                raise NotImplementedError(
                    'built-in type as head type of idiom unsupported: {}'.format(head_type))
                # Define a new sum type
                # Add the original built-in type and template as constructors
                # Replace every occurrence of the product type in the grammar
            
            else:
                raise NotImplementedError('Unable to handle head type of idiom: {}'.format(head_type))
            
        # Replace occurrences of product types which have been used as idiom head types
        for constructor_or_product in self.ast_wrapper.singular_types.values():
            for field in constructor_or_product.fields:
                if field.type in types_to_replace:
                    field.type = types_to_replace[field.type]

    def parse(self, code, section):
        if section == 'train':
            return self.convert_idiom_ast(code, template_id=None)()
        else:
            return self.base_grammar.parse(code, section)

    def unparse(self, tree):
        expanded_tree = self._expand_templates(tree)
        self.base_ast_wrapper.verify_ast(expanded_tree)
        return self.base_grammar.unparse(expanded_tree)

    def tokenize_field_value(self, field_value):
        return self.base_grammar.tokenize_field_value(field_value)

    #
    #
    #

    def _expand_templates(self, tree):
        if not isinstance(tree, dict):
            return tree

        node_type = tree['_type']
        constructor = self.ast_wrapper.constructors.get(node_type)

        expanded_fields = {}
        for field, value in tree.items():
            if field == '_type':
                continue
            if isinstance(value, (list, tuple)):
                result = []
                for item in value:
                    converted = self._expand_templates(item)
                    if isinstance(item, dict) and re.match('^Template\d+_.*_seq$', item['_type']):
                        item_type_info = self.ast_wrapper.constructors[converted['_type']]

                        assert len(item_type_info.fields) == 1
                        assert item_type_info.fields[0].seq
                        result += converted.get(item_type_info.fields[0].name, [])
                    else:
                        result.append(converted)
                expanded_fields[field] = result
            else:
                expanded_fields[field] = self._expand_templates(value)

        if constructor is None or not hasattr(constructor, 'template'):
            return {'_type': node_type, **expanded_fields}
 
        template = constructor.template
        hole_values = {}
        for field, expanded_value in expanded_fields.items():
            m = re.match('^hole(\d+)$', field)
            if not m:
                raise ValueError('Unexpected field name: {}'.format(field))
            hole_id = int(m.group(1))

            # Do something special if we have a seq fragment
            hole_values[hole_id] = expanded_value
        return template(hole_values)
    
    def _template_to_constructor(self, template_dict, suffix=''):
        hole_node_types = {}

        # Find where the holes occur
        stack = [(None, template_dict['idiom'], None)]
        while stack:
            parent, node, child_index = stack.pop()
            node_type, ref_symbols, hole_id, children = node
            if hole_id is not None:
                assert hole_id not in hole_node_types
                # node_type could be:
                # - name of field
                #   => hole type is same as field's type
                # - name of type, if it only has one child
                # - binarizer
                hyphenated_node_type = None
                unhyphenated_node_type = None

                hole_type_str = template_dict['holes'][hole_id]['type']
                if hole_type_str == 'AddChild':
                    node_type_for_field_type = node_type

                elif hole_type_str == 'ReplaceSelf':
                    # Two types of ReplaceSelf
                    # 1. Node has a hyphen: should be a repeated field
                    # 2. Node lacks a hyphen, and
                    #    2a. node is same as parent: a repeated field
                    #    2b. node is not the same as parent: an elem
                    if '-' in node_type:
                        node_type_for_field_type = node_type
                    else:
                        node_type_for_field_type = parent[0]
                        #if '-' in parent_type:
                        #    hyphenated_node_type = parent_type
                        #else:
                        #    unhyphenated_node_type = parent_type

                field_info = self._get_field_info_from_name(node_type_for_field_type)

                # Check for situations like 
                # Call-args
                # |       \
                # List[0] Call-args[1]
                #
                # Tuple
                # |       \
                # Tuple[0] Tuple[1]
                # where hole 0 should not be a sequence.

                if field_info.seq and hole_type_str == 'ReplaceSelf' and '-' not in node_type:
                    assert child_index in (0, 1)
                    seq = child_index == 1
                else:
                    seq = field_info.seq
                hole_node_types[hole_id] = (field_info.type, seq, field_info.opt)
            stack += [(node, child, i) for i, child in enumerate(children)]
        
        # Create fields for the holes
        fields = []
        for hole in template_dict['holes']:
            i = hole['id']
            field_type, seq, opt = hole_node_types[i]
            field = asdl.Field(type=field_type, name='hole{}'.format(i), seq=seq, opt=opt)
            field.hole_type = HoleType[hole['type']]
            fields.append(field)

        constructor = asdl.Constructor('Template{}{}'.format(template_dict['id'], suffix), fields)
        constructor.template = self.convert_idiom_ast(template_dict['idiom'], template_id=template_dict['id'])

        return constructor

    def _get_field_info_from_name(self, node_type):
        if '-' in node_type:
            type_name, field_name = node_type.split('-') 
            type_info = self.ast_wrapper.singular_types[type_name]
            field_info, = [field for field in type_info.fields if field.name == field_name]
        else:
            type_info = self.ast_wrapper.singular_types[node_type]
            assert len(type_info.fields) == 1
            field_info = type_info.fields[0]
        return field_info

    def convert_idiom_ast(self, idiom_ast, template_id=None, seq_fragment_type=None):
        if template_id is not  None:
            node_type, ref_symbols, hole, children = idiom_ast
        else:
            node_type, ref_symbols, children = idiom_ast

        if isinstance(node_type, dict):
            assert 'template_id' in node_type
            if seq_fragment_type:
                suffix = '_{}_seq'.format(seq_fragment_type)
            else:
                suffix = '' 
            node_type = 'Template{}{}'.format(node_type['template_id'], suffix)
            is_template_node = True
        else:
            is_template_node = False

        type_info = self.ast_wrapper.singular_types[node_type]

        # Each element of this list is a tuple (field, child)
        # - field: asdl.Field object
        # - child: an idiom_ast node
        #   If field.seq then child will be a binarizer node, or a template headed by a binarizer
        #   Otherwise, child will be a node whose type indicates the field's name (e.g. Call-func),
        #   and with a single child that contains the content of the field
        children_to_convert = []
        if is_template_node:
            assert len(children) == len(type_info.fields)
            for field, child in zip(type_info.fields, children):
                if field.hole_type == HoleType.ReplaceSelf and  field.seq:
                    children_to_convert.append((field, child))
                else:
                    assert not field.seq
                    dummy_node = list(idiom_ast)
                    dummy_node[0] = '{}-{}'.format(node_type, field.name)
                    dummy_node[-1] = [child]
                    children_to_convert.append((field, dummy_node))
               # else:
               #     raise ValueError('Unexpected hole_type: {}'.format(field.hole_type))
        else:
            fields_by_name = {f.name: f for f in type_info.fields}
            if len(type_info.fields) == 0:
                pass
            elif len(type_info.fields) == 1:
                children_to_convert.append((type_info.fields[0], idiom_ast))
            else:
                prefix_len = len(node_type) + 1
                for child in children:
                    field_name = child[0][prefix_len:]
                    children_to_convert.append((fields_by_name[field_name], child))
        assert set(field.name for field, _ in children_to_convert) == set(field.name for field in type_info.fields)

        def result_creator(hole_values={}):
            if template_id is not None and hole is not None and self.templates[template_id]['holes'][hole]['type'] == 'ReplaceSelf':
                return hole_values.get(hole, MissingValue)

            result = {}
            for field, child_node in children_to_convert:
                # field: ast.Field object representing the field in the ASDL Constructor/Product
                # child_node:
                #   the idiom_ast nodes which specify the field
                #   len(child_children)
                #   - 0: should never happen
                #   - 1: for regular fields, opt fields, seq fields if length is 0 or represented by a template node
                #   - 2: for seq fields of length >= 1
                if field.type in self.ast_wrapper.primitive_types:
                    convert = lambda node: (lambda hole_values: self.convert_builtin_type(field.type, node[0]))
                else:
                    convert = functools.partial(
                        self.convert_idiom_ast, template_id=template_id)

                if field.seq:
                    value = []
                    while True:
                        # child_node[2]: ID of hole
                        if template_id is not None and child_node[2] is not None:
                            hole_value =  hole_values.get(child_node[2], [])
                            assert isinstance(hole_value, list)
                            value += hole_value
                            #if value:
                            #    assert isinstance(hole_value, list)
                            #    value += hole_value
                            #else:
                            #    return hole_value
                            assert len(child_node[-1]) == 0
                            break

                        child_type, child_children = child_node[0], child_node[-1]
                        if isinstance(child_type, dict):
                            # Another template
                            value.append(convert(child_node, seq_fragment_type=field.type if field.seq else None)(hole_values))
                            break
                        # If control reaches here, child_node is a binarizer node
                        if len(child_children) == 1:
                            assert child_children[0][0] == 'End'
                            break
                        elif len(child_children) == 2:
                            # TODO: Sometimes we need to value.extend?
                            value.append(convert(child_children[0])(hole_values))
                            child_node = child_children[1]
                        else:
                            raise ValueError('Unexpected number of children: {}'.format(len(child_children)))
                    present = bool(value)
                elif field.opt:
                    # child_node[2]: ID of hole
                    if template_id is not None and child_node[2] is not None:
                        assert len(child_node[-1]) == 0
                        present = child_node[2] in hole_values
                        value = hole_values.get(child_node[2])
                    else:
                        assert len(child_node[-1]) == 1
                        # type of first (and only) child of child_node
                        if child_node[-1][0][0] == 'Null':
                            value = None
                            present = False
                        else:
                            value = convert(child_node[-1][0])(hole_values)
                            present = value is not MissingValue
                else:
                    if template_id is not None and child_node[2] is not None:
                        assert len(child_node[-1]) == 0
                        value = hole_values[child_node[2]]
                        present = True
                    else:
                        assert len(child_node[-1]) == 1
                        value = convert(child_node[-1][0])(hole_values)
                        present = True
                if present:
                    result[field.name] = value

            result['_type'] = node_type
            return result

        return result_creator

    def convert_builtin_type(self, field_type, value):
        if field_type == 'singleton' and value == 'Null':
            return None
        return value
