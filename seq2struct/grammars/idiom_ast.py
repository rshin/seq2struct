import collections
import enum
import functools
import json

import asdl

from seq2struct import ast_util
from seq2struct.utils import registry


class HoleType(enum.Enum):
    ReplaceSelf = 1
    AddChild = 2


@registry.register('grammar', 'idiom_ast')
class IdiomAstGrammar:

    def __init__(self, base_grammar, template_file):
        self.base_grammar = registry.construct('grammar', base_grammar)
        self.templates = json.load(open(template_file))

        self.ast_wrapper = self.base_grammar.ast_wrapper
        self.root_type = self.base_grammar.root_type
        if base_grammar['name'] == 'python':
            self.root_type = 'mod'

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
                    seq_fragment_constructors.append(
                        self._template_to_constructor(template, '_{}_seq'.format(head_type)))
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
                assert constructors
                if seq_fragment_constructors:
                    raise NotImplementedError('seq fragments of product types not supported: {}'.format(head_type))

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

                # Replace every occurrence of the product type in the grammar
                types_to_replace[head_type] = name
            
            # built-in type
            elif head_type in asdl.builtin_types:
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
            return self.convert_idiom_ast(code, has_hole_info=False)
        else:
            return self.base_grammar.parse(code, section)

    def unparse(self, tree):
        raise NotImplementedError

    def tokenize_field_value(self, field_value):
        return self.base_grammar.tokenize_field_value(field_value)

    #
    #
    #
    
    def _template_to_constructor(self, template_dict, suffix=''):
        hole_node_types = {}

        # Find where the holes occur
        stack = [template_dict['idiom']]
        while stack:
            node_type, ref_symbols, hole_id, children = stack.pop()
            if hole_id is not None:
                assert hole_id not in hole_node_types
                # node_type could be:
                # - name of field
                #   => hole type is same as field's type
                # - name of type, if it only has one child
                # - binarizer
                if '-' in node_type:
                    type_name, field_name = node_type.split('-') 
                    type_info = self.ast_wrapper.singular_types[type_name]
                    field_info, = [field for field in type_info.fields if field.name == field_name]
                else:
                    type_info = self.ast_wrapper.singular_types[node_type]
                    assert len(type_info.fields) == 1
                    field_info = type_info.fields[0]
                hole_node_types[hole_id] = (field_info.type, field_info.seq, field_info.opt)
            stack += children
        
        # Create fields for the holes
        fields = []
        for i, hole in enumerate(template_dict['holes']):
            field_type, seq, opt = hole_node_types[i]
            field = asdl.Field(type=field_type, name='hole{}'.format(i), seq=seq, opt=opt)
            field.hole_type = HoleType[hole['type']]
            fields.append(field)

        constructor = asdl.Constructor('Template{}{}'.format(template_dict['id'], suffix), fields)
        # TODO: Also save information about the actual idiom trees and how to fill them
        return constructor

    def convert_idiom_ast(self, idiom_ast, has_hole_info, seq_fragment_type=None):
        if has_hole_info:
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
                if field.hole_type == HoleType.ReplaceSelf:
                    children_to_convert.append((field, child))
                elif field.hole_type == HoleType.AddChild:
                    # TODO: Determine node_type_for_field from where the hole appears in the idiom tree
                    dummy_node = list(idiom_ast)
                    dummy_node[0] = '{}-{}'.format(node_type, field.name)
                    dummy_node[-1] = [child]
                    children_to_convert.append((field, dummy_node))
                else:
                    raise ValueError('Unexpected hole_type: {}'.format(field.hole_type))
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

        result = {}
        #for field, node_type_for_field, child_nodes in children_to_convert:
        for field, child_node in children_to_convert:
            # field: ast.Field object representing the field in the ASDL Constructor/Product
            # node_type_for_field: 
            #   name of the node in idiom_ast format which specifies the field
            #   (e.g. 'Import-names', 'List')
            # child_nodes:
            #   the idiom_ast nodes which specify the field
            #   len(child_nodes)
            #   - 0: should never happen
            #   - 1: for regular fields, opt fields, seq fields if length is 0 or represented by a template node
            #   - 2: for seq fields of length >= 1
            if field.type in asdl.builtin_types:
                convert = lambda node: node[0]
            else:
                convert = functools.partial(
                    self.convert_idiom_ast, has_hole_info=has_hole_info)

            if field.seq:
                value = []
                while True:
                    child_type, child_children = child_node[0], child_node[-1]
                    if isinstance(child_type, dict):
                        value.append(convert(child_node, seq_fragment_type=field.type))
                        break
                    # If control reaches here, child_node is a binarizer node
                    if len(child_children) == 1:
                        assert child_children[0][0] == 'End'
                        break
                    elif len(child_children) == 2:
                        value.append(convert(child_children[0]))
                        child_node = child_children[-1]
                    else:
                        raise ValueError('Unexpected number of children: {}'.format(len(child_children)))
                present = bool(value)
            elif field.opt:
                assert len(child_node[-1]) == 1
                # type of first (and only) child of child_node
                if child_node[-1][0][0] == 'Null':  
                    value = None
                    present = False
                else:
                    value = convert(child_node[-1][0])
                    present = True
            else:
                assert len(child_node[-1]) == 1
                value = convert(child_node[-1][0])
                present = True
            if present:
                result[field.name] = value

        result['_type'] = node_type
        return result