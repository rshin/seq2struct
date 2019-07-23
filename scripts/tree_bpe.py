import argparse
import collections
import copy
import json

import _jsonnet
import asdl
import attr

from seq2struct import datasets
from seq2struct import models
from seq2struct.utils import registry

# Initial units: node and its required product-type fields, recursively
# Eligible triples: node, field_name then
# - opt: None, or type of field
# - seq: None, or if len > 0:
#   - sum type: type of first element
#   - product type: type of first element (indicates there's more than one element)
#   - constant: value
# - neither: type of field, or constant value
#
# Merging
# 1. Replace type of node with something else
# 2. Promote fields of the involved field
#   - None: nothing to promote
#   - seq, type of first element: all fields of that type
#   - neither: type of field


class IdentitySet(collections.abc.MutableSet):
    def __init__(self, iterable=()):
        self.map = {id(x): x for x in iterable}

    def __contains__(self, value):
        return id(value) in self.map
    
    def __iter__(self):
        return self.map.values()
    
    def __len__(self):
        return len(self.map)
    
    def add(self, value):
        self.map[id(value)] = value
    
    def discard(self, value):
        self.map.pop(id(value))


@attr.s
class TypeInfo:
    name = attr.ib()
    base_name = attr.ib()
    predecessor_name = attr.ib()
    unset_fields = attr.ib()
    preset_fields = attr.ib()
    preset_seq_elem_counts = attr.ib(factory=lambda: collections.defaultdict(int))


@attr.s(frozen=True)
class Primitive:
    value = attr.ib()


def count_subtrees(grammar, trees, num_rewrites):
    ast_wrapper = grammar.ast_wrapper

    # TODO: Revive the following
    ## Preprocess the grammar
    ## Create initial units: node and its required product-type fields, recursively
    #units = {name: {} for name in ast_wrapper.singular_types}
    #for name, cons in ast_wrapper.singular_types.items():
    #    unit_fields = units[name]
    #    for field in cons.fields:
    #        if not field.seq and not field.opt and field.type in ast_wrapper.singular_types:
    #            unit_fields[field.name] = units[field.type]

    # field name syntax:
    # (field_name{1}, ..., field_name{k}, i, field_name{k+1}, ..., field_name{n})
    
    type_infos = {
        k: TypeInfo(
            name=k,
            base_name=k,
            predecessor_name=k,
            unset_fields={field.name: field for field in v.fields},
            preset_fields={}
        ) for k, v in ast_wrapper.singular_types.items()
    }

    # Count types
    for iteration in range(100):
        triple_occurrences = count_triples(grammar.ast_wrapper, type_infos, trees)

        # Most frequent
        most_freq_triple, most_freq_occurrences = max(
                triple_occurrences.items(), key=lambda kv: len(kv[1]))

        existing_type_name, field_name, field_info = most_freq_triple
        tuple_name = field_name if isinstance(field_name, tuple) else (field_name,)
        existing_type = type_infos[existing_type_name]
        existing_field = existing_type.unset_fields[field_name]

        promoted_fields = []
        if isinstance(field_info, Primitive) or field_info is None:
            pass
        else:
            # Figure out which fields of type `field_info` should be promoted
            # Example:
            #   most_freq_triple = ('Call', 'func', 'Name')
            #   field_info = 'Name'
            #   type_infos['Name'].unset_fields = {'id': Field(identifier, id)}
            for field_field in type_infos[field_info].unset_fields.values():
                if isinstance(field_field.name, tuple):
                    field_field_tuple_name = field_field.name
                else:
                    field_field_tuple_name = (field_field.name,)
                if existing_field.seq:
                    suffix = (existing_type.preset_seq_elem_counts[tuple_name],) + field_field_tuple_name
                else:
                    suffix = field_field_tuple_name
                promoted_fields.append((
                    field_field,
                    asdl.Field(
                        type=field_field.type,
                        name=tuple_name + suffix,
                        seq=field_field.seq,
                        opt=field_field.opt)
                ))

            #promoted_fields = [
            #    asdl.Field(
            #        name=new_name,
            #        seq=unset_field.seq,
            #        opt=unset_field.opt,
            #    ) for new_name, unset_field in zip(new_field_names, type_infos[field_info].unset_fields)
            #]

        # Create a new type
        new_preset_fields = dict(existing_type.preset_fields)
        new_preset_seq_elem_counts = copy.copy(existing_type.preset_seq_elem_counts)
        if existing_field.seq and field_info is not None:
            new_preset_fields[
                tuple_name + (new_preset_seq_elem_counts[tuple_name],)] = field_info
            new_preset_seq_elem_counts[tuple_name] += 1
        else:
            new_preset_fields[tuple_name] = field_info

        new_unset_fields = {
            **{f.name: f for old_field, f in promoted_fields},
            **existing_type.unset_fields
        }
        if field_info is None or not existing_field.seq:
            # Only unset if...
            # - field is not sequential
            # - field has been set to None, meaning the end of a sequence
            del new_unset_fields[field_name]

        new_type = TypeInfo(
            name='Type{}'.format(len(type_infos)),
            base_name=existing_type.base_name,
            predecessor_name=existing_type.name,
            unset_fields=new_unset_fields,
            preset_fields=new_preset_fields,
            preset_seq_elem_counts = new_preset_seq_elem_counts
        )
        type_infos[new_type.name] = new_type

        # Tracks which occurrences have been removed due to promotion.
        discarded = IdentitySet()
        for occ in most_freq_occurrences:
            if occ in discarded:
                continue

            occ['_type'] = new_type.name
            def delete_obsoleted_field():
                if existing_field.seq:
                    # todo: change 0 if we can promote other elements
                    del occ[field_name][0]
                    if not occ[field_name]:
                        del occ[field_name]
                else:
                    del occ[field_name]

            if isinstance(field_info, Primitive):
                delete_obsoleted_field()
            elif field_info is None:
                pass
            else:
                if existing_field.seq:
                    # todo: change 0 if we can promote other elements
                    value_to_promote = occ[field_name][0]
                else:
                    value_to_promote = occ[field_name]
                delete_obsoleted_field()
                discarded.add(value_to_promote)

                for old_field, new_field in promoted_fields:
                    if old_field.name not in value_to_promote:
                        assert old_field.opt or old_field.seq
                        continue
                    occ[new_field.name] = value_to_promote[old_field.name]
                    assert occ[new_field.name]


def count_triples(ast_wrapper, type_infos, trees):
    triple_occurrences = collections.defaultdict(list)
    for tree in trees:
        queue = collections.deque([tree])
        while queue:
            node = queue.pop()
            for field_name, field in type_infos[node['_type']].unset_fields.items():
                if field_name in node:
                    field_value = node[field_name]
                    is_primitive = field.type in ast_wrapper.primitive_types

                    if field.seq:
                        relevant_value = field_value[0]
                        if not is_primitive:
                            queue.extend(field_value)
                    else:
                        relevant_value = field_value
                        if not is_primitive:
                            queue.append(field_value)

                    if is_primitive:
                        field_info = Primitive(relevant_value)
                    else:
                        field_info = relevant_value['_type']
                else:
                    assert field.seq or field.opt
                    field_info = None

                triple_occurrences[node['_type'], field_name, field_info].append(node)

            for field_name in type_infos[node['_type']].preset_fields:
                assert field_name not in node
    
    return triple_occurrences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    parser.add_argument('--section', default='train')
    args = parser.parse_args()

    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    # 0. Construct preprocessors
    model_preproc = registry.instantiate(
        registry.lookup('model', config['model']).Preproc,
        config['model'])
    model_preproc.load()

    # 3. Get training data somewhere
    preproc_data = model_preproc.dataset(args.section)
    count_subtrees(model_preproc.dec_preproc.grammar, [dec.tree for enc, dec in preproc_data], num_rewrites=1000)
    

if __name__ == '__main__':
    main()