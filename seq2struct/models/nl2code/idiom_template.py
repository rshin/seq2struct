import attr
import enum
import itertools
import pyrsistent

from seq2struct import ast_util
from seq2struct.models.nl2code import decoder
from seq2struct.utils import vocab


class TemplateTraversalType(enum.Enum):
    DEFAULT = 0
    CHILDREN_APPLY = 1
    LIST_LENGTH_APPLY = 2


@attr.s(frozen=True)
class TemplateTraversalState:
    node = attr.ib()
    parent_field_type = attr.ib()
    type = attr.ib(default=TemplateTraversalType.DEFAULT)


@attr.s(frozen=True)
class TemplateActionProvider:
    model = attr.ib()
    queue = attr.ib()
    buffer = attr.ib(factory=pyrsistent.pdeque)
    last_returned = attr.ib(default=None)

    @classmethod
    def build(cls, model, tree, parent_field_type):
        return cls(model, pyrsistent.pdeque([TemplateTraversalState(tree, parent_field_type)]))

    @property
    def finished(self):
        return not self.queue and not self.buffer

    # TODO reduce duplication with compute_loss
    def step(self, last_choice):
        queue = self.queue
        buffer = self.buffer
        def rv():
            return buffer.left, TemplateActionProvider(
                    self.model, queue, buffer.popleft(), buffer.left)

        if buffer:
            if not isinstance(self.last_returned, ast_util.HoleValuePlaceholder):
                assert (last_choice == self.last_returned or
                        isinstance(self.last_returned, tuple) and last_choice in self.last_returned)
            return rv()

        to_return = False

        while queue:
            item = queue.right
            node = item.node
            parent_field_type = item.parent_field_type

            queue = queue.pop()
            to_return = False
            result_finished = False
            new_last_choice = None

            if item.type == TemplateTraversalType.DEFAULT and isinstance(node, (list, tuple)):
                hvps = [elem for elem in node if isinstance(elem,
                    ast_util.HoleValuePlaceholder)]
                num_seq_hvps = sum(hvp.is_seq for hvp in hvps)
                assert num_seq_hvps in (0, 1)

                node_type = parent_field_type + '*'
                if num_seq_hvps:
                    allowed_lengths = [
                        l for l in self.model.preproc.seq_lengths[node_type]
                        if l >= len(node) - 1
                    ]
                    rule_indices = tuple(
                        self.model.rules_index[node_type, length]
                        for length in allowed_lengths)
                else:
                    rule = (node_type, len(node))
                    rule_indices = (self.model.rules_index[rule],)
                if len(rule_indices) == 1:
                    buffer = buffer.append(rule_indices[0])
                    new_last_choice = rule_indices[0]
                else:
                    to_return = True
                    buffer = buffer.append(rule_indices)

                queue = queue.append(
                    TemplateTraversalState(
                        type=TemplateTraversalType.LIST_LENGTH_APPLY,
                        node=node,
                        parent_field_type=parent_field_type,
                    ))
                result_finished = True
 
            elif item.type == TemplateTraversalType.LIST_LENGTH_APPLY and isinstance(node, (list, tuple)):
                list_type, num_children = self.model.preproc.all_rules[last_choice]
                assert list_type == parent_field_type + '*'
                assert num_children > 0

                if num_children < len(node):
                    assert isinstance(node[-1], ast_util.HoleValuePlaceholder)
                    assert node[-1].is_seq
                    assert num_children + 1 == len(node)
                    node = node[:-1]
                elif len(node) < num_children:
                    assert isinstance(node[-1], ast_util.HoleValuePlaceholder)
                    assert node[-1].is_seq
                    node = node + [node[-1]] * (num_children - len(node))

                if self.model.preproc.use_seq_elem_rules and parent_field_type in self.model.ast_wrapper.sum_types:
                    parent_field_type += '_seq_elem'

                for i, elem in reversed(list(enumerate(node))):
                    queue = queue.append(
                        TemplateTraversalState(
                            node=elem,
                            parent_field_type=parent_field_type,
                        ))
                result_finished = True
            
            elif isinstance(node, ast_util.HoleValuePlaceholder):
                buffer = buffer.append(node)
                result_finished = True

            elif parent_field_type in self.model.preproc.grammar.pointers:
                assert isinstance(node, int)
                buffer = buffer.append(node)
                result_finished = True

            elif parent_field_type in self.model.ast_wrapper.primitive_types:
                # identifier, int, string, bytes, object, singleton
                # - could be bytes, str, int, float, bool, NoneType
                # - terminal tokens vocabulary is created by turning everything into a string (with `str`)
                # - at decoding time, cast back to str/int/float/bool
                field_type = type(node).__name__
                field_value_split = self.model.preproc.grammar.tokenize_field_value(node) + [
                        vocab.EOS]

                buffer = buffer.extend(field_value_split)
                result_finished = True
            
            if result_finished:
                if to_return:
                    return rv()
                else:
                    last_choice = new_last_choice
                    continue

            type_info = self.model.ast_wrapper.singular_types[node['_type']]
            if item.type == TemplateTraversalType.CHILDREN_APPLY:
                node_type, children_presence = self.model.preproc.all_rules[last_choice]
                assert node_type == node['_type']

                # reversed so that we perform a DFS in left-to-right order
                for field_info, present in reversed(
                        list(zip(type_info.fields, children_presence))):
                    
                    if present:
                        assert field_info.name in node
                    elif field_info.name not in node:
                        continue
                    else:
                        field_value = node[field_info.name]
                        if isinstance(field_value, ast_util.HoleValuePlaceholder):
                            assert field_value.is_opt
                        elif isinstance(field_value, list):
                            assert len(field_value) == 1
                            assert isinstance(field_value[0], ast_util.HoleValuePlaceholder)
                            assert field_value[0].is_seq
                        else:
                            raise ValueError(field_value)
                        continue

                    field_value = node[field_info.name]
                    queue = queue.append(
                        TemplateTraversalState(
                            node=field_value,
                            parent_field_type=field_info.type,
                        ))

                last_choice = new_last_choice
                continue

            if parent_field_type in self.model.preproc.sum_type_constructors:
                # ApplyRule, like expr -> Call
                rule = (parent_field_type, type_info.name)
                rule_idx = self.model.rules_index[rule]
                assert not node.get('_extra_types', ())
                buffer = buffer.append(rule_idx)

            if type_info.fields:
                # ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
                # Figure out which rule needs to be applied
                present = decoder.get_field_presence_info(self.model.ast_wrapper, node, type_info.fields)

                # Are any of the fields HoleValuePlaceholders?
                hvp_present = False
                presence_values = []
                for i, field_info in enumerate(type_info.fields):
                    if field_info.name not in node:
                        presence_values.append((False,))
                        continue

                    field_value = node[field_info.name]

                    if isinstance(field_value, ast_util.HoleValuePlaceholder) or (
                        isinstance(field_value, list) and
                        len(field_value) == 1 and
                        isinstance(field_value[0], ast_util.HoleValuePlaceholder)):

                        # If field is a primitive type, then we need to ask the model what type it is
                        # If field is optional, it may actually be missing
                        presence = tuple(set(info[i] for info in self.model.preproc.field_presence_infos[node['_type']]))
                        presence_values.append(presence)
                        hvp_present = True
                    else:
                        presence_values.append((present[i],))
                
                if hvp_present:
                    rule_indices = tuple(
                        rule_idx for rule_idx in (
                            self.model.rules_index.get((node['_type'], p))
                            for p in itertools.product(*presence_values))
                        if rule_idx is not None
                    )
                    if len(rule_indices) == 1:
                        buffer = buffer.append(rule_indices[0])
                        new_last_choice = rule_indices[0]
                    else:
                        to_return = True
                        buffer = buffer.append(rule_indices)
                else:
                    rule = (node['_type'], tuple(present))
                    rule_idx = self.model.rules_index[rule]
                    buffer = buffer.append(rule_idx)
                    new_last_choice = rule_idx

                queue = queue.append(
                    TemplateTraversalState(
                        type=TemplateTraversalType.CHILDREN_APPLY,
                        node=node,
                        parent_field_type=parent_field_type,
                    ))

            if to_return:
                return rv()
            else:
                last_choice = new_last_choice
                continue

        return rv()
