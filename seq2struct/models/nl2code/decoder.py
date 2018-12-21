import ast
import collections
import collections.abc
import enum
import json
import os
import re

import asdl
import attr
import torch

from seq2struct import ast_util
from seq2struct import models
from seq2struct.models import abstract_preproc
from seq2struct.models import attention
from seq2struct.utils import registry
from seq2struct.utils import vocab


def lstm_init(device, num_layers, hidden_size, *batch_sizes):
    init_size = (num_layers, ) + batch_sizes + (hidden_size, )
    init = torch.zeros(*init_size, device=device)
    return (init, init)


def split_string_whitespace_and_camelcase(s):
    split_space = s.split(' ')
    result = []
    for token in split_space:
        if token: # \uE012 is an arbitrary glue character, from the Private Use Area.
            camelcase_split_token = re.sub('([a-z])([A-Z])', '\\1\uE012\\2', token).split('\uE012')
            result.extend(camelcase_split_token)
        result.append(' ')
    return result[:-1]


def maybe_stack(items, dim=None):
    to_stack = [item for item in items if item is not None]
    if not to_stack:
        return None
    elif len(to_stack) == 1:
        return to_stack[0].unsqueeze(dim)
    else:
        return torch.stack(to_stack, dim)


def to_dict_with_sorted_values(d):
    return {k: sorted(v) for k, v in d.items()}


def to_dict_with_set_values(d):
    result = {}
    for k, v in d.items():
        hashable_v = []
        for v_elem in v:
            if isinstance(v_elem, list):
                hashable_v.append(tuple(v_elem))
            else:
                hashable_v.append(v_elem)
        result[k] = set(hashable_v)
    return result


def tuplify(x):
    if not isinstance(x, (tuple, list)):
        return x
    return tuple(tuplify(elem) for elem in x)


@attr.s
class NL2CodeDecoderPreprocItem:
    tree = attr.ib()
    orig_code = attr.ib()


class NL2CodeDecoderPreproc(abstract_preproc.AbstractPreproc):
    def __init__(
            self,
            save_path,
            min_freq=3,
            max_count=5000):
        # TODO: don't hardcode Python.asdl
        self.ast_wrapper = ast_util.ASTWrapper(
                asdl.parse(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        '..', '..',
                        'Python.asdl')))

        self.vocab_path = os.path.join(save_path, 'dec_vocab.json')
        self.observed_productions_path = os.path.join(save_path, 'observed_productions.json')
        self.grammar_rules_path = os.path.join(save_path, 'grammar_rules.json')
        self.data_dir = os.path.join(save_path, 'dec')

        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        # TODO: Write 'train', 'val', 'test' somewhere else
        self.items = {'train': [], 'val': [], 'test': []}
        self.sum_type_constructors = collections.defaultdict(set)
        self.field_presence_infos = collections.defaultdict(set)
        self.seq_lengths = collections.defaultdict(set)
        self.primitive_types = set()

        self.vocab = None
        self.all_rules = None
        self.rules_mask = None

    def validate_item(self, item, section):
        try:
            py_ast = ast.parse(item.code)
            root = ast_util.convert_native_ast(py_ast)
        except SyntaxError:
            return section != 'train', None
        return True, root

    def add_item(self, item, section, validation_info):
        root = validation_info
        if section == 'train':
            for token in self._all_tokens(root):
                self.vocab_builder.add_word(token)
            self._record_productions(root)

        self.items[section].append(
            NL2CodeDecoderPreprocItem(
                tree=root,
                orig_code=item.code))

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab = self.vocab_builder.finish()
        self.vocab.save(self.vocab_path)

        for section, items in self.items.items():
            with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                for item in items:
                    f.write(json.dumps(attr.asdict(item)) + '\n')

        # observed_productions
        self.sum_type_constructors = to_dict_with_sorted_values(
            self.sum_type_constructors)
        self.field_presence_infos = to_dict_with_sorted_values(
            self.field_presence_infos)
        self.seq_lengths = to_dict_with_sorted_values(
            self.seq_lengths)
        self.primitive_types = sorted(self.primitive_types)
        with open(self.observed_productions_path, 'w') as f:
            json.dump({
                'sum_type_constructors': self.sum_type_constructors,
                'field_presence_infos': self.field_presence_infos,
                'seq_lengths': self.seq_lengths,
                'primitive_types': self.primitive_types,
            }, f, indent=2, sort_keys=True)

        # grammar
        self.all_rules, self.rules_mask = self._calculate_rules()
        with open(self.grammar_rules_path, 'w') as f:
            json.dump({
                'all_rules': self.all_rules,
                'rules_mask': self.rules_mask,
            }, f, indent=2, sort_keys=True)

    def load(self):
        self.vocab = vocab.Vocab.load(self.vocab_path)

        observed_productions = json.load(open(self.observed_productions_path))
        self.sum_type_constructors = observed_productions['sum_type_constructors']
        self.field_presence_infos = observed_productions['field_presence_infos']
        self.seq_lengths = observed_productions['seq_lengths']
        self.primitive_types = observed_productions['primitive_types']

        grammar = json.load(open(self.grammar_rules_path))
        self.all_rules = tuplify(grammar['all_rules'])
        self.rules_mask = grammar['rules_mask']

    def dataset(self, section):
        return [
            NL2CodeDecoderPreprocItem(**json.loads(line))
            for line in open(os.path.join(self.data_dir, section + '.jsonl'))]

    def _record_productions(self, tree):
        queue = [tree]
        while queue:
            node = queue.pop()
            node_type = node['_type']

            # Rules of the form:
            # expr -> Attribute | Await | BinOp | BoolOp | ...
            if node_type in self.ast_wrapper.constructors:
                constructor = self.ast_wrapper.constructor_to_sum_type[node_type]
                self.sum_type_constructors[constructor].add(node_type)

            # Rules of the form:
            # FunctionDef
            # -> identifier name, arguments args
            # |  identifier name, arguments args, stmt* body
            # |  identifier name, arguments args, expr* decorator_list
            # |  identifier name, arguments args, expr? returns
            # ...
            # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
            assert node_type in self.ast_wrapper.singular_types
            field_presence_info = get_field_presence_info(
                    node,
                    self.ast_wrapper.singular_types[node_type].fields)
            self.field_presence_infos[node_type].add(field_presence_info)

            # Rules of the form:
            # stmt* -> stmt
            #        | stmt stmt
            #        | stmt stmt stmt
            for field_info in self.ast_wrapper.singular_types[node_type].fields:
                field_value = node.get(field_info.name)
                to_enqueue = []
                if field_info.seq:
                    self.seq_lengths[field_info.type + '*'].add(len(field_value))
                    to_enqueue = field_value
                else:
                    to_enqueue = [field_value]
                for child in to_enqueue:
                    if isinstance(child, collections.abc.Mapping) and '_type' in child:
                        queue.append(child)
                    else:
                        self.primitive_types.add(type(child).__name__)

    def _calculate_rules(self):
        offset = 0

        all_rules = []
        rules_mask = {}

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        for parent, children in sorted(self.sum_type_constructors.items()):
            assert not isinstance(children, set)
            rules_mask[parent] = (offset, offset + len(children))
            offset += len(children)
            all_rules += [(parent, child) for child in children]

        # Rules of the form:
        # FunctionDef
        # -> identifier name, arguments args
        # |  identifier name, arguments args, stmt* body
        # |  identifier name, arguments args, expr* decorator_list
        # |  identifier name, arguments args, expr? returns
        # ...
        # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
        for name, field_presence_infos in sorted(self.field_presence_infos.items()):
            assert not isinstance(field_presence_infos, set)
            rules_mask[name] = (offset, offset + len(field_presence_infos))
            offset += len(field_presence_infos)
            all_rules += [(name, presence) for presence in field_presence_infos]

        # Rules of the form:
        # stmt* -> stmt
        #        | stmt stmt
        #        | stmt stmt stmt
        for seq_type_name, lengths in sorted(self.seq_lengths.items()):
            assert not isinstance(lengths, set)
            rules_mask[seq_type_name] = (offset, offset + len(lengths))
            offset += len(lengths)
            all_rules += [(seq_type_name, i) for i in lengths]

        return tuple(all_rules), rules_mask


    def _all_tokens(self, root):
        queue = [root]
        while queue:
            node = queue.pop()
            type_info = self.ast_wrapper.singular_types[node['_type']]

            for field_info in reversed(type_info.fields):
                field_value = node.get(field_info.name)
                if field_info.type in asdl.builtin_types:
                    for token in self._tokenize_field_value(field_value):
                        yield token
                elif isinstance(field_value, (list, tuple)):
                    queue.extend(field_value)
                elif field_value is not None:
                    queue.append(field_value)

    def _tokenize_field_value(self, field_value):
        if isinstance(field_value, bytes):
            field_value = field_value.encode('latin1')
        else:
            field_value = str(field_value)
        return split_string_whitespace_and_camelcase(field_value)


@attr.s
class TreeState:
    node = attr.ib()
    parent_action_emb = attr.ib()
    parent_h = attr.ib()
    parent_field_type = attr.ib()
    debug_path = attr.ib()




@registry.register('decoder', 'NL2Code')
class NL2CodeDecoder(torch.nn.Module):

    Preproc = NL2CodeDecoderPreproc

    def __init__(
            self, 
            device,
            preproc,
            #
            max_seq_length=10,
            rule_emb_size=128,
            node_embed_size=64,
            # TODO: This should be automatically inferred from encoder
            enc_recurrent_size=256,
            recurrent_size=256,
            desc_attn=None,
            copy_pointer=None):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.ast_wrapper = preproc.ast_wrapper
        self.terminal_vocab = preproc.vocab

        self.rule_emb_size = rule_emb_size
        self.node_emb_size = node_embed_size
        self.enc_recurrent_size = enc_recurrent_size
        self.recurrent_size = recurrent_size

        self.rules_index = {v: idx for idx, v in enumerate(self.preproc.all_rules)}

        self.node_type_vocab = vocab.Vocab(
                sorted(self.preproc.primitive_types) +
                sorted(self.ast_wrapper.sum_types.keys()) +
                sorted(self.ast_wrapper.singular_types.keys()) +
                sorted(self.preproc.seq_lengths.keys()),
                special_elems=())

        self.state_update = torch.nn.LSTM(
                self.rule_emb_size * 2 + self.enc_recurrent_size + self.recurrent_size + self.node_emb_size,
                self.recurrent_size,
                num_layers=1)
        if desc_attn is None:
            self.desc_attn = attention.BahdanauAttention(
                    query_size=self.recurrent_size,
                    value_size=self.enc_recurrent_size,
                    proj_size=50)
        else:
            # TODO: Figure out how to get right sizes (query, value) to module
            self.desc_attn = desc_attn

        self.rule_logits = torch.nn.Sequential(
                torch.nn.Linear(self.recurrent_size, self.rule_emb_size),
                torch.nn.Tanh(),
                torch.nn.Linear(self.rule_emb_size, len(self.rules_index)))
        self.rule_embedding = torch.nn.Embedding(
                num_embeddings=len(self.rules_index),
                embedding_dim=self.rule_emb_size)

        self.gen_logodds = torch.nn.Linear(self.recurrent_size, 1)
        self.terminal_logits = torch.nn.Sequential(
                torch.nn.Linear(self.recurrent_size, self.rule_emb_size),
                torch.nn.Tanh(),
                torch.nn.Linear(self.rule_emb_size, len(self.terminal_vocab)))
        self.terminal_embedding = torch.nn.Embedding(
                num_embeddings=len(self.terminal_vocab),
                embedding_dim=self.rule_emb_size)
        if copy_pointer is None:
            self.copy_pointer = attention.BahdanauPointer(
                    query_size=self.recurrent_size,
                    key_size=self.enc_recurrent_size,
                    proj_size=50)
        else:
            # TODO: Figure out how to get right sizes (query, key) to module
            self.copy_pointer = copy_pointer

        self.node_type_embedding = torch.nn.Embedding(
                num_embeddings=len(self.node_type_vocab),
                embedding_dim=self.node_emb_size)

        # TODO batching
        self.zero_rule_emb = torch.zeros(1, self.rule_emb_size, device=self._device)
        self.zero_recurrent_emb = torch.zeros(1, self.recurrent_size, device=self._device)
        self.xent_loss = torch.nn.CrossEntropyLoss(reduction='none')

    @classmethod
    def _calculate_rules(cls, preproc):
        offset = 0

        all_rules = []
        rules_mask = {}

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        for parent, children in sorted(preproc.sum_type_constructors.items()):
            assert parent not in rules_mask
            rules_mask[parent] = (offset, offset + len(children))
            offset += len(children)
            all_rules += [(parent, child) for child in children]

        # Rules of the form:
        # FunctionDef
        # -> identifier name, arguments args
        # |  identifier name, arguments args, stmt* body
        # |  identifier name, arguments args, expr* decorator_list
        # |  identifier name, arguments args, expr? returns
        # ...
        # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
        for name, field_presence_infos in sorted(preproc.field_presence_infos.items()):
            assert name not in rules_mask
            rules_mask[name] = (offset, offset + len(field_presence_infos))
            offset += len(field_presence_infos)
            all_rules += [(name, presence) for presence in field_presence_infos]

        # Rules of the form:
        # stmt* -> stmt
        #        | stmt stmt
        #        | stmt stmt stmt
        for seq_type_name, lengths in sorted(preproc.seq_lengths.items()):
            assert seq_type_name not in rules_mask
            rules_mask[seq_type_name] = (offset, offset + len(lengths))
            offset += len(lengths)
            all_rules += [(seq_type_name, i) for i in lengths]

        return all_rules, rules_mask

    def compute_loss(self, example, desc_enc):
        # - field is required, singular, sum type
        #   choose among the possibilities afforded by the sum type
        # - field is required, singular, primitive type
        #   Use GenToken
        # - field is repeated
        #   Original NL2Code:
        #     The number of children is decided ahead of time.
        #     E.g. there would be rules like
        #       expr* -> expr
        #       expr* -> expr expr
        #       expr* -> expr expr expr
        # - field is optional and it's missing
        #   Original NL2Code:
        #     This never happens because all missing fields are
        #     omitted from the node.
        state = lstm_init(self._device, 1, self.recurrent_size, 1)
        prev_action_emb = self.zero_rule_emb

        loss = []

        # Perform a DFS
        queue = [
            TreeState(
                node=example.tree,
                parent_action_emb=self.zero_rule_emb,
                parent_h=self.zero_recurrent_emb,
                parent_field_type=None,
                debug_path=example.tree['_type'])
        ]
        while queue:
            item = queue.pop()
            node = item.node
            parent_action_emb = item.parent_action_emb
            parent_h = item.parent_h
            parent_field_type = item.parent_field_type

            if parent_field_type in asdl.builtin_types:
                # identifier, int, string, bytes, object, singleton
                # - could be bytes, str, int, float, bool, NoneType
                # - terminal tokens vocabulary is created by turning everything into a string (with `str`)
                # - at decoding time, cast back to str/int/float/bool
                field_type = type(node).__name__
                field_value_split = self.preproc._tokenize_field_value(node) + [
                        vocab.EOS]

                for token in field_value_split:
                    state, prev_action_emb, loss_piece = self.gen_token(
                            field_type,
                            token,
                            state,
                            prev_action_emb,
                            parent_h, 
                            parent_action_emb,
                            desc_enc)
                    loss.append(loss_piece)
                continue
 
            if isinstance(node, (list, tuple)):
                # parent_field_type* -> [parent_field_type] * len(list)
                node_type = parent_field_type + '*'
                rule = (node_type, len(node))
                output, state, prev_action_emb, loss_piece = self.apply_rule_loss(
                        node_type,
                        rule,
                        state,
                        prev_action_emb,
                        parent_h, 
                        parent_action_emb,
                        desc_enc)
                loss.append(loss_piece)

                for i, elem in reversed(list(enumerate(node))):
                    queue.append(
                        TreeState(
                            node=elem,
                            parent_action_emb=prev_action_emb,
                            parent_h=output,
                            parent_field_type=parent_field_type,
                            debug_path=item.debug_path + '[{}]'.format(i)))
                continue

            type_info = self.ast_wrapper.singular_types[node['_type']]

            if parent_field_type in self.ast_wrapper.sum_types:
                # ApplyRule, like expr -> Call
                rule = (parent_field_type, type_info.name)
                output, state, prev_action_emb, loss_piece = self.apply_rule_loss(
                        parent_field_type,
                        rule,
                        state,
                        prev_action_emb,
                        parent_h, 
                        parent_action_emb,
                        desc_enc)

                parent_h = output
                parent_action_emb = prev_action_emb
                loss.append(loss_piece)

            if type_info.fields:
                # ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
                # Figure out which rule needs to be applied
                present = get_field_presence_info(node, type_info.fields)
                # TODO: put type name in type_info for product types
                rule = (node['_type'], tuple(present))
                output, state, prev_action_emb, loss_piece = self.apply_rule_loss(
                        node['_type'],
                        rule,
                        state,
                        prev_action_emb,
                        parent_h,
                        parent_action_emb,
                        desc_enc)
                loss.append(loss_piece)

            # reversed so that we perform a DFS in left-to-right order
            for field_info in reversed(type_info.fields):
                field_value = node.get(field_info.name)
                if field_info.type not in asdl.builtin_types:
                    if field_value is None or field_value == []:
                        continue

                queue.append(
                    TreeState(
                        node=field_value,
                        parent_action_emb=prev_action_emb,
                        parent_h=output,
                        # "identifier" for example
                        parent_field_type=field_info.type,
                        debug_path='{} -> {}'.format(item.debug_path, field_info.name)))

        return torch.sum(torch.stack(loss, dim=0), dim=0)
    
    def infer(self, desc_enc):
        return TreeTraversal(self, desc_enc)

    def _desc_attention(self, prev_state, desc_enc):
        # prev_state shape:
        # - h_n: num_layers * num_directions (=1) x batch (=1) x emb_size
        # - c_n: num_layers * num_directions (=1) x batch (=1) x emb_size
        prev_state_h, prev_state_c = prev_state
        query = prev_state_h[-1]
        return self.desc_attn(query, desc_enc.memory, attn_mask=None)
    
    def _tensor(self, data, dtype=None):
        return torch.tensor(data, dtype=dtype, device=self._device)
    
    def _index(self, vocab, word):
        return self._tensor([vocab.index(word)])

    def apply_rule(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        # desc_context shape: batch (=1) x emb_size
        desc_context, _ = self._desc_attention(prev_state, desc_enc)
        # node_type_emb shape: batch (=1) x emb_size
        node_type_emb = self.node_type_embedding(
                self._index(self.node_type_vocab, node_type))

        state_input = torch.cat(
            (
                prev_action_emb,  # a_{t-1}: rule_emb_size
                desc_context,  # c_t: enc_recurrent_size
                parent_h,  # s_{p_t}: recurrent_size
                parent_action_emb,  # a_{p_t}: rule_emb_size
                node_type_emb,  # n_{f-t}: node_emb_size
            ),
            dim=-1)
        output, new_state = self.state_update(
                # state_input shape: batch (=1) x (emb_size * 5)
                state_input.unsqueeze(dim=0), prev_state)
        # output shape: batch (=1) x emb_size
        output = output.squeeze(0)
        # rule_logits shape: batch (=1) x num choices
        rule_logits = self.rule_logits(output)

        return output, new_state, rule_logits
    
    def rule_infer(self, node_type, rule_logits):
        rule_logprobs = torch.nn.functional.log_softmax(rule_logits, dim=-1)
        rules_start, rules_end = self.preproc.rules_mask[node_type]

        # TODO: Mask other probabilities first?
        return list(zip(
            range(rules_start, rules_end),
            rule_logprobs[0, rules_start:rules_end]))

    def apply_rule_loss(
            self,
            node_type,
            rule,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        output, new_state, rule_logits = self.apply_rule(
                node_type, prev_state, prev_action_emb, parent_h, parent_action_emb, desc_enc)

        # rule_idx shape: batch (=1)
        rule_idx = self._tensor([self.rules_index[rule]])
        # action_emb shape: batch (=1) x emb_size
        action_emb = self.rule_embedding(rule_idx)
        # loss_piece shape: batch (=1)
        loss_piece = self.xent_loss(rule_logits, rule_idx)
        return output, new_state, action_emb, loss_piece
    
    def gen_token(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        # desc_context shape: batch (=1) x emb_size
        desc_context, _ = self._desc_attention(prev_state, desc_enc)
        # node_type_emb shape: batch (=1) x emb_size
        node_type_emb = self.node_type_embedding(
                self._index(self.node_type_vocab, node_type))
                
        state_input = torch.cat(
            (
                prev_action_emb,  # a_{t-1}: rule_emb_size
                desc_context,  # c_t: enc_recurrent_size
                parent_h,  # s_{p_t}: recurrent_size
                parent_action_emb,  # a_{p_t}: rule_emb_size
                node_type_emb,  # n_{f-t}: node_emb_size
            ),
            dim=-1)
        # state_input shape: batch (=1) x (emb_size * 5)
        output, new_state = self.state_update(
                state_input.unsqueeze(dim=0), prev_state)
        # output shape: batch (=1) x emb_size
        output = output.squeeze(0)

        # gen_logodds shape: batch (=1)
        gen_logodds = self.gen_logodds(output).squeeze(1)

        return new_state, output, gen_logodds
    
    def gen_token_loss(
            self,
            node_type,
            token,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        new_state, output, gen_logodds = self.gen_token(
            node_type, prev_state, prev_action_emb, parent_h, parent_action_emb, desc_enc)

        # token_idx shape: batch (=1), LongTensor
        token_idx = self._index(self.terminal_vocab, token)
        # action_emb shape: batch (=1) x emb_size
        action_emb = self.terminal_embedding(token_idx)

        # +unk, +in desc: copy
        # +unk, -in desc: gen (an unk token)
        # -unk, +in desc: copy, gen
        # -unk, -in desc: gen
        # gen_logodds shape: batch (=1)
        desc_locs = desc_enc.find_word_occurrences(token)
        if desc_locs:
            # copy: if the token appears in the description at least once
            # copy_loc_logits shape: batch (=1) x desc length
            copy_loc_logits = self.copy_pointer(output, desc_enc.memory)
            copy_logprob = (
                # log p(copy | output)
                # shape: batch (=1)
                torch.nn.functional.logsigmoid(-gen_logodds) -
                # xent_loss: -log p(location | output)
                # TODO: sum the probability of all occurrences
                # shape: batch (=1)
                self.xent_loss(copy_loc_logits, self._tensor(desc_locs[0:1])))
        else:
            copy_logprob = None

        # gen: ~(unk & in desc), equivalent to  ~unk | ~in desc
        if token in self.terminal_vocab or copy_logprob is None:
            token_logits = self.terminal_logits(output)
            # shape: 
            gen_logprob = (
                # log p(gen | output)
                # shape: batch (=1)
                torch.nn.functional.logsigmoid(gen_logodds) -
                # xent_loss: -log p(token | output)
                # shape: batch (=1)
                self.xent_loss(token_logits, token_idx))
        else:
            gen_logprob = None

        # loss should be -log p(...), so negate
        loss_piece = -torch.logsumexp(
            maybe_stack([copy_logprob, gen_logprob], dim=1),
            dim=1)

        return new_state, action_emb, loss_piece
    
    def token_infer(self, output, gen_logodds, desc_enc):
        # Copy tokens
        # log p(copy | output)
        # shape: batch (=1)
        copy_logprob = torch.nn.functional.logsigmoid(-gen_logodds)
        copy_loc_logits = self.copy_pointer(output, desc_enc.memory)
        # log p(loc_i | copy, output)
        # shape: batch (=1) x seq length
        copy_loc_logprobs = torch.nn.functional.log_softmax(copy_loc_logits, dim=-1)
        # log p(loc_i, copy | output)
        copy_loc_logprobs += copy_logprob

        log_prob_by_word = collections.defaultdict(
            int,
            zip(desc_enc.words, copy_loc_logprobs.squeeze(0)))
        
        # Generate tokens
        # log p(~copy | output)
        # shape: batch (=1)
        gen_logprob = torch.nn.functional.logsigmoid(-gen_logodds)
        token_logits = self.terminal_logits(output)
        # log p(v | ~copy, output)
        # shape: batch (=1) x vocab size
        token_logprobs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        # log p(v, ~copy| output)
        # shape: batch (=1) x vocab size
        token_logprobs += gen_logprob

        for idx in range(token_logprobs.shape[1]):
            log_prob_by_word[self.terminal_vocab[idx]] += token_logprobs[0, idx]
        
        return list(log_prob_by_word.items())


def get_field_presence_info(node, field_infos):
    present = []
    for field_info in field_infos:
        field_value = node.get(field_info.name)
        is_present = field_value is not None and field_value != []

        maybe_missing = field_info.opt or field_info.seq
        is_builtin_type = field_info.type in asdl.builtin_types

        if maybe_missing and is_builtin_type:
            # TODO: make it posible to deal with "singleton?"
            present.append(is_present and type(field_value).__name__)
        elif maybe_missing and not is_builtin_type:
            present.append(is_present)
        elif not maybe_missing and is_builtin_type:
            present.append(type(field_value).__name__)
        elif not maybe_missing and not is_builtin_type:
            assert is_present
            present.append(True)
    return tuple(present)


class TreeTraversal:

    @attr.s
    class QueueItem:
        state = attr.ib()
        node_type = attr.ib()
        parent_action_emb = attr.ib()
        parent_h = attr.ib()
        parent_field_name = attr.ib()
    
    class State(enum.Enum):
        SUM_TYPE_INQUIRE = 0
        SUM_TYPE_APPLY = 1

        CHILDREN_INQUIRE = 2
        CHILDREN_APPLY = 3

        LIST_LENGTH_INQUIRE = 4
        LIST_LENGTH_APPLY = 5

        GEN_TOKEN = 6

        NODE_FINISHED = 7

    class TreeAction:
        pass
    
    @attr.s
    class SetParentField(TreeAction):
        parent_field_name = attr.ib()
        value = attr.ib()
    
    @attr.s
    class CreateParentFieldList(TreeAction):
        parent_field_name = attr.ib()

    @attr.s
    class AppendParentField(TreeAction):
        parent_field_name = attr.ib()
        value = attr.ib()
        terminal_type = attr.ib()
    
    @attr.s
    class NodeFinished(TreeAction):
        pass
    
    def __init__(self, model, desc_enc):
        self.model = model
        self.ast_wrapper = model.ast_wrapper

        self.state = lstm_init(model._device, 1, self.model.recurrent_size, 1)
        self.prev_action_emb = model.zero_rule_emb
        self.desc_enc = desc_enc

        self.queue = []

        self.cur_item = TreeTraversal.QueueItem(
                state=TreeTraversal.State.CHILDREN_INQUIRE,
                node_type='Module',
                parent_action_emb=self.model.zero_rule_emb,
                parent_h=self.model.zero_recurrent_emb,
                parent_field_name=None)
        self.parent_h = None

        self.actions = []
        self.update_prev_action_emb = self.update_prev_action_emb_apply_rule
    
    def step(self, last_choice):
        SUM_TYPE_INQUIRE    = TreeTraversal.State.SUM_TYPE_INQUIRE
        SUM_TYPE_APPLY      = TreeTraversal.State.SUM_TYPE_APPLY
        CHILDREN_INQUIRE    = TreeTraversal.State.CHILDREN_INQUIRE
        CHILDREN_APPLY      = TreeTraversal.State.CHILDREN_APPLY
        LIST_LENGTH_INQUIRE = TreeTraversal.State.LIST_LENGTH_INQUIRE
        LIST_LENGTH_APPLY   = TreeTraversal.State.LIST_LENGTH_APPLY
        GEN_TOKEN           = TreeTraversal.State.GEN_TOKEN
        NODE_FINISHED       = TreeTraversal.State.NODE_FINISHED

        self.update_prev_action_emb(last_choice)

        while True:
            # 1. ApplyRule, like expr -> Call
            if self.cur_item.state == SUM_TYPE_INQUIRE:
                # a. Ask which one to choose
                output, self.state, rule_logits = self.model.apply_rule(
                        self.cur_item.node_type, 
                        self.state,
                        self.prev_action_emb,
                        self.cur_item.parent_h, 
                        self.cur_item.parent_action_emb,
                        self.desc_enc)
                self.parent_h = output
                self.cur_item.state = SUM_TYPE_APPLY

                self.update_prev_action_emb = self.update_prev_action_emb_apply_rule
                return self.model.rule_infer(self.cur_item.node_type, rule_logits)

            elif self.cur_item.state == SUM_TYPE_APPLY:
                # b. Add action, prepare for #2
                sum_type, singular_type = self.model.preproc.all_rules[last_choice]
                assert sum_type == self.cur_item.node_type

                self.cur_item.node_type = singular_type
                self.cur_item.parent_action_emb = self.prev_action_emb
                self.cur_item.state = CHILDREN_INQUIRE

                last_choice = None
                continue

            # 2. ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
            elif self.cur_item.state == CHILDREN_INQUIRE:
                self.actions.append(
                    TreeTraversal.SetParentField(
                        self.cur_item.parent_field_name, {'_type': self.cur_item.node_type}))

                # Check if we have no children
                type_info = self.ast_wrapper.singular_types[self.cur_item.node_type]
                if not type_info.fields:
                    self.actions.append(TreeTraversal.NodeFinished())
                    if self.pop():
                        continue
                    else:
                        return None

                # a. Ask about presence
                output, self.state, rule_logits = self.model.apply_rule(
                        self.cur_item.node_type, 
                        self.state,
                        self.prev_action_emb,
                        self.cur_item.parent_h, 
                        self.cur_item.parent_action_emb,
                        self.desc_enc)
                self.parent_h = output
                self.cur_item.state = CHILDREN_APPLY

                self.update_prev_action_emb = self.update_prev_action_emb_apply_rule
                return self.model.rule_infer(self.cur_item.node_type, rule_logits)
            
            elif self.cur_item.state == CHILDREN_APPLY:
                # b. Create the children
                node_type, children_presence = self.model.preproc.all_rules[last_choice]
                assert node_type == self.cur_item.node_type

                self.queue.append(TreeTraversal.QueueItem(
                    state=NODE_FINISHED,
                    node_type=None,
                    parent_action_emb=None,
                    parent_h=None,
                    parent_field_name=None
                ))
                for field_info, present in reversed(
                        list(zip(self.ast_wrapper.singular_types[node_type].fields, children_presence))):
                    if not present:
                        continue

                    # seq field: LIST_LENGTH_INQUIRE x
                    # sum type: SUM_TYPE_INQUIRE x
                    # product type: 
                    #   no children: not possible
                    #   children: CHILDREN_INQUIRE
                    # constructor type: not possible x
                    # builtin type: GEN_TOKEN x
                    child_type = field_type = field_info.type
                    if field_info.seq:
                        child_state = LIST_LENGTH_INQUIRE
                    elif field_type in self.ast_wrapper.sum_types:
                        child_state = SUM_TYPE_INQUIRE
                    elif field_type in self.ast_wrapper.product_types:
                        assert self.ast_wrapper.product_types[field_type].fields
                        child_state = CHILDREN_INQUIRE
                    elif field_type in asdl.builtin_types:
                        child_state = GEN_TOKEN
                        child_type = present
                    else:
                        raise ValueError('Unable to handle field type {}'.format(field_type))

                    self.queue.append(TreeTraversal.QueueItem(
                        state=child_state,
                        node_type=child_type,
                        parent_action_emb=self.prev_action_emb,
                        parent_h=self.parent_h,
                        parent_field_name=field_info.name,
                    ))
                
                advanced = self.pop()
                assert advanced
                last_choice = None
                continue
            
            elif self.cur_item.state == LIST_LENGTH_INQUIRE:
                list_type = self.cur_item.node_type + '*'
                output, self.state, rule_logits = self.model.apply_rule(
                        list_type,
                        self.state,
                        self.prev_action_emb,
                        self.cur_item.parent_h, 
                        self.cur_item.parent_action_emb,
                        self.desc_enc)
                self.parent_h = output
                self.cur_item.state = LIST_LENGTH_APPLY

                self.update_prev_action_emb = self.update_prev_action_emb_apply_rule
                return self.model.rule_infer(list_type, rule_logits)
            
            elif self.cur_item.state == LIST_LENGTH_APPLY:
                list_type, num_children = self.model.preproc.all_rules[last_choice]
                elem_type = self.cur_item.node_type
                assert list_type == elem_type + '*'

                for i in range(num_children):
                    if elem_type in self.ast_wrapper.sum_types:
                        child_state = SUM_TYPE_INQUIRE
                    elif elem_type in self.ast_wrapper.product_types:
                        child_state = CHILDREN_INQUIRE
                    elif elem_type in asdl.builtin_types:
                        raise ValueError('sequential builtin types not supported')
                    else:
                        raise ValueError('Unable to handle seq field type {}'.format(elem_type))
                    self.queue.append(TreeTraversal.QueueItem(
                        state=child_state,
                        node_type=elem_type,
                        parent_action_emb=self.prev_action_emb,
                        parent_h=self.parent_h,
                        parent_field_name=self.cur_item.parent_field_name,
                    ))
                self.actions.append(TreeTraversal.CreateParentFieldList(self.cur_item.parent_field_name))

                advanced = self.pop()
                assert advanced

            elif self.cur_item.state == GEN_TOKEN:
                if last_choice == vocab.EOS:
                    if self.pop():
                        continue
                    else:
                        return None
                elif last_choice is not None:
                    # token_idx shape: batch (=1), LongTensor
                    token_idx = self.model._index(self.model.terminal_vocab, last_choice)
                    # action_emb shape: batch (=1) x emb_size
                    self.prev_action_emb = self.model.terminal_embedding(token_idx)
                    self.actions.append(TreeTraversal.AppendParentField(
                        self.cur_item.parent_field_name,
                        last_choice,
                        self.cur_item.node_type))

                self.state, output, gen_logodds = self.model.gen_token(
                        self.cur_item.node_type,
                        self.state,
                        self.prev_action_emb,
                        self.cur_item.parent_h,
                        self.cur_item.parent_action_emb,
                        self.desc_enc)
                log_probs_by_word = self.model.token_infer(output, gen_logodds, self.desc_enc)

                self.update_prev_action_emb = self.update_prev_action_emb_gen_token
                return log_probs_by_word

            elif self.cur_item.state == NODE_FINISHED:
                self.actions.append(TreeTraversal.NodeFinished())
                if self.pop():
                    continue
                else:
                    return None
            
            else:
                raise ValueError('Unknown state {}'.format(self.cur_item.state))
    
    def update_prev_action_emb_gen_token(self, last_choice):
        if last_choice is None:
            return
        # token_idx shape: batch (=1), LongTensor
        token_idx = self.model._index(self.model.terminal_vocab, last_choice)
        # action_emb shape: batch (=1) x emb_size
        self.prev_action_emb = self.model.terminal_embedding(token_idx)
    
    def update_prev_action_emb_apply_rule(self, last_choice):
        if last_choice is None:
            return
        # rule_idx shape: batch (=1)
        rule_idx = self.model._tensor([last_choice])
        # action_emb shape: batch (=1) x emb_size
        self.prev_action_emb = self.model.rule_embedding(rule_idx)
    
    def pop(self):
        self.parent_h = None
        if self.queue:
           self.cur_item = self.queue.pop()
           return True
        return False
