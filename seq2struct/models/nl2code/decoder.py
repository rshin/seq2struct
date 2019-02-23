import ast
import collections
import collections.abc
import enum
import itertools
import json
import os
import operator
import re

import asdl
import attr
import pyrsistent
import torch

from seq2struct import ast_util
from seq2struct import grammars
from seq2struct.models import abstract_preproc
from seq2struct.models import attention
from seq2struct.models import lstm
from seq2struct.utils import registry
from seq2struct.utils import vocab
from seq2struct.utils import serialization


def lstm_init(device, num_layers, hidden_size, *batch_sizes):
    init_size = batch_sizes + (hidden_size, )
    if num_layers is not None:
        init_size = (num_layers, ) + init_size
    init = torch.zeros(*init_size, device=device)
    return (init, init)


def maybe_stack(items, dim=None):
    to_stack = [item for item in items if item is not None]
    if not to_stack:
        return None
    elif len(to_stack) == 1:
        return to_stack[0].unsqueeze(dim)
    else:
        return torch.stack(to_stack, dim)


def accumulate_logprobs(d, keys_and_logprobs):
    for key, logprob in keys_and_logprobs:
        existing = d.get(key)
        if existing is None:
            d[key] = logprob
        else:
            d[key] = torch.logsumexp(
                torch.stack((logprob, existing), dim=0),
                dim=0)


@attr.s
class NL2CodeDecoderPreprocItem:
    tree = attr.ib()
    orig_code = attr.ib()


class NL2CodeDecoderPreproc(abstract_preproc.AbstractPreproc):
    def __init__(
            self,
            grammar,
            save_path,
            min_freq=3,
            max_count=5000,
            use_seq_elem_rules=False):
        
        self.grammar = registry.construct('grammar', grammar)
        self.ast_wrapper = self.grammar.ast_wrapper

        self.vocab_path = os.path.join(save_path, 'dec_vocab.json')
        self.observed_productions_path = os.path.join(save_path, 'observed_productions.json')
        self.grammar_rules_path = os.path.join(save_path, 'grammar_rules.json')
        self.data_dir = os.path.join(save_path, 'dec')

        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        self.use_seq_elem_rules = use_seq_elem_rules

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
        parsed = self.grammar.parse(item.code, section)
        if parsed:
            self.ast_wrapper.verify_ast(parsed)
            return True, parsed
        return section != 'train', None

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
        self.sum_type_constructors = serialization.to_dict_with_sorted_values(
            self.sum_type_constructors)
        self.field_presence_infos = serialization.to_dict_with_sorted_values(
            self.field_presence_infos, key=str)
        self.seq_lengths = serialization.to_dict_with_sorted_values(
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
        self.all_rules = serialization.tuplify(grammar['all_rules'])
        self.rules_mask = grammar['rules_mask']

    def dataset(self, section):
        return [
            NL2CodeDecoderPreprocItem(**json.loads(line))
            for line in open(os.path.join(self.data_dir, section + '.jsonl'))]

    def _record_productions(self, tree):
        queue = [(tree, False)]
        while queue:
            node, is_seq_elem = queue.pop()
            node_type = node['_type']

            # Rules of the form:
            # expr -> Attribute | Await | BinOp | BoolOp | ...
            # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
            for type_name in [node_type] + node.get('_extra_types', []):
                if type_name in self.ast_wrapper.constructors:
                    sum_type_name = self.ast_wrapper.constructor_to_sum_type[type_name]
                    if is_seq_elem and self.use_seq_elem_rules:
                        self.sum_type_constructors[sum_type_name + '_seq_elem'].add(type_name)
                    else:
                        self.sum_type_constructors[sum_type_name].add(type_name)
            
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
                    self.ast_wrapper,
                    node,
                    self.ast_wrapper.singular_types[node_type].fields)
            self.field_presence_infos[node_type].add(field_presence_info)

            for field_info in self.ast_wrapper.singular_types[node_type].fields:
                field_value = node.get(field_info.name, [] if field_info.seq else None)
                to_enqueue = []
                if field_info.seq:
                    # Rules of the form:
                    # stmt* -> stmt
                    #        | stmt stmt
                    #        | stmt stmt stmt
                    self.seq_lengths[field_info.type + '*'].add(len(field_value))
                    to_enqueue = field_value
                else:
                    to_enqueue = [field_value]
                for child in to_enqueue:
                    if isinstance(child, collections.abc.Mapping) and '_type' in child:
                        queue.append((child, field_info.seq))
                    else:
                        self.primitive_types.add(type(child).__name__)

    def _calculate_rules(self):
        offset = 0

        all_rules = []
        rules_mask = {}

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
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
                if field_info.type in self.ast_wrapper.primitive_types:
                    for token in self.grammar.tokenize_field_value(field_value):
                        yield token
                elif isinstance(field_value, (list, tuple)):
                    queue.extend(field_value)
                elif field_value is not None:
                    queue.append(field_value)


@attr.s
class TreeState:
    node = attr.ib()
    parent_field_type = attr.ib()


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
            dropout=0.,
            desc_attn='bahdanau',
            copy_pointer=None,
            multi_loss_type='logsumexp'):
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

        if self.preproc.use_seq_elem_rules:
            self.node_type_vocab = vocab.Vocab(
                    sorted(self.preproc.primitive_types) +
                    sorted(self.ast_wrapper.custom_primitive_types) +
                    sorted(self.preproc.sum_type_constructors.keys()) +
                    sorted(self.preproc.field_presence_infos.keys()) +
                    sorted(self.preproc.seq_lengths.keys()),
                    special_elems=())
        else:
            self.node_type_vocab = vocab.Vocab(
                    sorted(self.preproc.primitive_types) +
                    sorted(self.ast_wrapper.custom_primitive_types) +
                    sorted(self.ast_wrapper.sum_types.keys()) +
                    sorted(self.ast_wrapper.singular_types.keys()) +
                    sorted(self.preproc.seq_lengths.keys()),
                    special_elems=())

        self.state_update = lstm.RecurrentDropoutLSTMCell(
                input_size=self.rule_emb_size * 2 + self.enc_recurrent_size + self.recurrent_size + self.node_emb_size,
                hidden_size=self.recurrent_size,
                dropout=dropout)
        if desc_attn == 'bahdanau':
            self.desc_attn = attention.BahdanauAttention(
                    query_size=self.recurrent_size,
                    value_size=self.enc_recurrent_size,
                    proj_size=50)
        elif desc_attn == 'mha':
            self.desc_attn = attention.MultiHeadedAttention(
                    h=8,
                    query_size=self.recurrent_size,
                    value_size=self.enc_recurrent_size)
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
        if multi_loss_type == 'logsumexp':
            self.multi_loss_reduction = lambda logprobs: -torch.logsumexp(logprobs, dim=1)
        elif multi_loss_type == 'mean':
            self.multi_loss_reduction = lambda logprobs: -torch.mean(logprobs, dim=1)
        
        self.pointers = torch.nn.ModuleDict()
        self.pointer_action_emb_proj = torch.nn.ModuleDict()
        for pointer_type in self.preproc.grammar.pointers:
            self.pointers[pointer_type] = attention.ScaledDotProductPointer(
                    query_size=self.recurrent_size,
                    key_size=self.enc_recurrent_size)
            self.pointer_action_emb_proj[pointer_type] = torch.nn.Linear(
                    self.recurrent_size, self.rule_emb_size)

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
        # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
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

    def compute_loss(self, example, desc_enc, debug=False):
        traversal = TrainTreeTraversal(self, desc_enc, debug)
        traversal.step(None)

        queue = [
            TreeState(
                node=example.tree,
                parent_field_type=self.preproc.grammar.root_type,
            )
        ]
        while queue:
            item = queue.pop()
            node = item.node
            parent_field_type = item.parent_field_type

            if isinstance(node, (list, tuple)):
                node_type = parent_field_type + '*'
                rule = (node_type, len(node))
                rule_idx = self.rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.LIST_LENGTH_APPLY
                traversal.step(rule_idx)

                if self.preproc.use_seq_elem_rules and parent_field_type in self.ast_wrapper.sum_types:
                    parent_field_type += '_seq_elem'

                for i, elem in reversed(list(enumerate(node))):
                    queue.append(
                        TreeState(
                            node=elem,
                            parent_field_type=parent_field_type,
                        ))
                continue

            if parent_field_type in self.preproc.grammar.pointers:
                assert isinstance(node, int)
                assert traversal.cur_item.state == TreeTraversal.State.POINTER_APPLY
                pointer_map = desc_enc.pointer_maps.get(parent_field_type)
                if pointer_map:
                    values = pointer_map[node]
                    traversal.step(values[0], values[1:])
                else:
                    traversal.step(node)
                continue

            if parent_field_type in self.ast_wrapper.primitive_types:
                # identifier, int, string, bytes, object, singleton
                # - could be bytes, str, int, float, bool, NoneType
                # - terminal tokens vocabulary is created by turning everything into a string (with `str`)
                # - at decoding time, cast back to str/int/float/bool
                field_type = type(node).__name__
                field_value_split = self.preproc.grammar.tokenize_field_value(node) + [
                        vocab.EOS]

                for token in field_value_split:
                    assert traversal.cur_item.state == TreeTraversal.State.GEN_TOKEN
                    traversal.step(token)
                continue
            
            type_info = self.ast_wrapper.singular_types[node['_type']]

            if parent_field_type in self.preproc.sum_type_constructors:
                # ApplyRule, like expr -> Call
                rule = (parent_field_type, type_info.name)
                rule_idx = self.rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.SUM_TYPE_APPLY
                extra_rules = [
                    self.rules_index[parent_field_type, extra_type]
                    for extra_type in node.get('_extra_types', [])]
                traversal.step(rule_idx, extra_rules)

            if type_info.fields:
                # ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
                # Figure out which rule needs to be applied
                present = get_field_presence_info(self.ast_wrapper, node, type_info.fields)
                rule = (node['_type'], tuple(present))
                rule_idx = self.rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.CHILDREN_APPLY
                traversal.step(rule_idx)

            # reversed so that we perform a DFS in left-to-right order
            for field_info in reversed(type_info.fields):
                if field_info.name not in node:
                    continue

                queue.append(
                    TreeState(
                        node=node[field_info.name],
                        parent_field_type=field_info.type,
                    ))

        loss = torch.sum(torch.stack(tuple(traversal.loss), dim=0), dim=0)
        if debug:
            return loss, [attr.asdict(entry) for entry in traversal.history]
        else:
            return loss
        

    def begin_inference(self, desc_enc, example):
        traversal = InferenceTreeTraversal(self, desc_enc, example)
        choices = traversal.step(None)
        return traversal, choices

    def _desc_attention(self, prev_state, desc_enc):
        # prev_state shape:
        # - h_n: batch (=1) x emb_size
        # - c_n: batch (=1) x emb_size
        query = prev_state[0]
        return self.desc_attn(query, desc_enc.memory, attn_mask=None)
    
    def _tensor(self, data, dtype=None):
        return torch.tensor(data, dtype=dtype, device=self._device)
    
    def _index(self, vocab, word):
        return self._tensor([vocab.index(word)])
    
    def _update_state(
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
        new_state = self.state_update(
                # state_input shape: batch (=1) x (emb_size * 5)
                state_input, prev_state)
        return new_state

    def apply_rule(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        new_state = self._update_state(
            node_type, prev_state, prev_action_emb, parent_h, parent_action_emb, desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]
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

    def gen_token(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        new_state = self._update_state(
            node_type, prev_state, prev_action_emb, parent_h, parent_action_emb, desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]

        # gen_logodds shape: batch (=1)
        gen_logodds = self.gen_logodds(output).squeeze(1)

        return new_state, output, gen_logodds
    
    def gen_token_loss(
            self,
            output,
            gen_logodds,
            token,
            desc_enc):
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
        return loss_piece
    
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

        log_prob_by_word = {}
        # accumulate_logprobs is needed because the same word may appear
        # multiple times in desc_enc.words.
        accumulate_logprobs(
            log_prob_by_word,
            zip(desc_enc.words, copy_loc_logprobs.squeeze(0)))
        
        # Generate tokens
        # log p(~copy | output)
        # shape: batch (=1)
        gen_logprob = torch.nn.functional.logsigmoid(gen_logodds)
        token_logits = self.terminal_logits(output)
        # log p(v | ~copy, output)
        # shape: batch (=1) x vocab size
        token_logprobs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        # log p(v, ~copy| output)
        # shape: batch (=1) x vocab size
        token_logprobs += gen_logprob

        accumulate_logprobs(
            log_prob_by_word,
            ((self.terminal_vocab[idx], token_logprobs[0, idx]) for idx in range(token_logprobs.shape[1])))
        
        return list(log_prob_by_word.items())
    
    def compute_pointer(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        new_state = self._update_state(
            node_type, prev_state, prev_action_emb, parent_h, parent_action_emb, desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]
        # pointer_logits shape: batch (=1) x num choices
        pointer_logits = self.pointers[node_type](
            output, desc_enc.pointer_memories[node_type])

        return output, new_state, pointer_logits
    
    def pointer_infer(self, node_type, logits):
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        return list(zip(
            # TODO batching
            range(logits.shape[1]),
            logprobs[0]))


def get_field_presence_info(ast_wrapper, node, field_infos):
    present = []
    for field_info in field_infos:
        field_value = node.get(field_info.name)
        is_present = field_value is not None and field_value != []

        maybe_missing = field_info.opt or field_info.seq
        is_builtin_type = field_info.type in ast_wrapper.primitive_types

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

    @attr.s(frozen=True)
    class QueueItem:
        state = attr.ib()
        node_type = attr.ib()
        parent_action_emb = attr.ib()
        parent_h = attr.ib()
        parent_field_name = attr.ib()

        def to_str(self):
            return '<state: {}, node_type: {}, parent_field_name: {}>'.format(self.state, self.node_type, self.parent_field_name)
    
    class State(enum.Enum):
        SUM_TYPE_INQUIRE = 0
        SUM_TYPE_APPLY = 1
        CHILDREN_INQUIRE = 2
        CHILDREN_APPLY = 3
        LIST_LENGTH_INQUIRE = 4
        LIST_LENGTH_APPLY = 5
        GEN_TOKEN = 6
        POINTER_INQUIRE = 7
        POINTER_APPLY = 8
        NODE_FINISHED = 9

    class PrevActionTypes(enum.Enum):
        APPLY_RULE = 0
        GEN_TOKEN = 1
    
    def __init__(self, model, desc_enc):
        if model is None:
            return

        self.model = model
        self.desc_enc = desc_enc

        model.state_update.set_dropout_masks(batch_size=1)
        self.recurrent_state = lstm_init(model._device, None, self.model.recurrent_size, 1)
        self.prev_action_emb = model.zero_rule_emb

        root_type = model.preproc.grammar.root_type 
        if root_type in model.preproc.ast_wrapper.sum_types:
            initial_state = TreeTraversal.State.SUM_TYPE_INQUIRE
        else:
            initial_state = TreeTraversal.State.CHILDREN_INQUIRE

        self.queue = pyrsistent.pvector()
        self.cur_item = TreeTraversal.QueueItem(
                state=initial_state,
                node_type=root_type,
                parent_action_emb=self.model.zero_rule_emb,
                parent_h=self.model.zero_recurrent_emb,
                parent_field_name=None)

        self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_apply_rule

    def clone(self):
        other = self.__class__(None, None)
        other.model = self.model
        other.desc_enc = self.desc_enc
        other.recurrent_state = self.recurrent_state
        other.prev_action_emb = self.prev_action_emb
        other.queue = self.queue
        other.cur_item = self.cur_item
        other.actions = self.actions
        other.update_prev_action_emb = self.update_prev_action_emb
        return other

    def step(self, last_choice, extra_choice_info=None):
        SUM_TYPE_INQUIRE    = TreeTraversal.State.SUM_TYPE_INQUIRE
        SUM_TYPE_APPLY      = TreeTraversal.State.SUM_TYPE_APPLY
        CHILDREN_INQUIRE    = TreeTraversal.State.CHILDREN_INQUIRE
        CHILDREN_APPLY      = TreeTraversal.State.CHILDREN_APPLY
        LIST_LENGTH_INQUIRE = TreeTraversal.State.LIST_LENGTH_INQUIRE
        LIST_LENGTH_APPLY   = TreeTraversal.State.LIST_LENGTH_APPLY
        GEN_TOKEN           = TreeTraversal.State.GEN_TOKEN
        POINTER_INQUIRE     = TreeTraversal.State.POINTER_INQUIRE
        POINTER_APPLY       = TreeTraversal.State.POINTER_APPLY
        NODE_FINISHED       = TreeTraversal.State.NODE_FINISHED

        while True:
            self.update_using_last_choice(last_choice, extra_choice_info)

            # 1. ApplyRule, like expr -> Call
            if self.cur_item.state == SUM_TYPE_INQUIRE:
                # a. Ask which one to choose
                output, self.recurrent_state, rule_logits = self.model.apply_rule(
                        self.cur_item.node_type, 
                        self.recurrent_state,
                        self.prev_action_emb,
                        self.cur_item.parent_h, 
                        self.cur_item.parent_action_emb,
                        self.desc_enc)
                self.cur_item = attr.evolve(
                        self.cur_item,
                        state=SUM_TYPE_APPLY,
                        parent_h=output)

                self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_apply_rule
                return self.rule_choice(self.cur_item.node_type, rule_logits)

            elif self.cur_item.state == SUM_TYPE_APPLY:
                # b. Add action, prepare for #2
                sum_type, singular_type = self.model.preproc.all_rules[last_choice]
                assert sum_type == self.cur_item.node_type

                self.cur_item = attr.evolve(
                        self.cur_item,
                        node_type=singular_type,
                        parent_action_emb=self.prev_action_emb,
                        state=CHILDREN_INQUIRE)

                last_choice = None
                continue

            # 2. ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
            elif self.cur_item.state == CHILDREN_INQUIRE:
                # Check if we have no children
                type_info = self.model.ast_wrapper.singular_types[self.cur_item.node_type]
                if not type_info.fields:
                    if self.pop():
                        last_choice = None
                        continue
                    else:
                        return None

                # a. Ask about presence
                output, self.recurrent_state, rule_logits = self.model.apply_rule(
                        self.cur_item.node_type, 
                        self.recurrent_state,
                        self.prev_action_emb,
                        self.cur_item.parent_h, 
                        self.cur_item.parent_action_emb,
                        self.desc_enc)
                self.cur_item = attr.evolve(
                        self.cur_item,
                        state=CHILDREN_APPLY,
                        parent_h=output)

                self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_apply_rule
                return self.rule_choice(self.cur_item.node_type, rule_logits)
            
            elif self.cur_item.state == CHILDREN_APPLY:
                # b. Create the children
                node_type, children_presence = self.model.preproc.all_rules[last_choice]
                assert node_type == self.cur_item.node_type

                self.queue = self.queue.append(TreeTraversal.QueueItem(
                    state=NODE_FINISHED,
                    node_type=None,
                    parent_action_emb=None,
                    parent_h=None,
                    parent_field_name=None
                ))
                for field_info, present in reversed(
                        list(zip(self.model.ast_wrapper.singular_types[node_type].fields, children_presence))):
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
                    elif field_type in self.model.ast_wrapper.sum_types:
                        child_state = SUM_TYPE_INQUIRE
                    elif field_type in self.model.ast_wrapper.product_types:
                        assert self.model.ast_wrapper.product_types[field_type].fields
                        child_state = CHILDREN_INQUIRE
                    elif field_type in self.model.preproc.grammar.pointers:
                        child_state = POINTER_INQUIRE
                    elif field_type in self.model.ast_wrapper.primitive_types:
                        child_state = GEN_TOKEN
                        child_type = present
                    else:
                        raise ValueError('Unable to handle field type {}'.format(field_type))

                    self.queue = self.queue.append(TreeTraversal.QueueItem(
                        state=child_state,
                        node_type=child_type,
                        parent_action_emb=self.prev_action_emb,
                        parent_h=self.cur_item.parent_h,
                        parent_field_name=field_info.name,
                    ))
                
                advanced = self.pop()
                assert advanced
                last_choice = None
                continue
            
            elif self.cur_item.state == LIST_LENGTH_INQUIRE:
                list_type = self.cur_item.node_type + '*'
                output, self.recurrent_state, rule_logits = self.model.apply_rule(
                        list_type,
                        self.recurrent_state,
                        self.prev_action_emb,
                        self.cur_item.parent_h, 
                        self.cur_item.parent_action_emb,
                        self.desc_enc)
                self.cur_item = attr.evolve(
                        self.cur_item,
                        state=LIST_LENGTH_APPLY,
                        parent_h=output)

                self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_apply_rule
                return self.rule_choice(list_type, rule_logits)
            
            elif self.cur_item.state == LIST_LENGTH_APPLY:
                list_type, num_children = self.model.preproc.all_rules[last_choice]
                elem_type = self.cur_item.node_type
                assert list_type == elem_type + '*'

                child_node_type = elem_type
                if elem_type in self.model.ast_wrapper.sum_types:
                    child_state = SUM_TYPE_INQUIRE
                    if self.model.preproc.use_seq_elem_rules:
                        child_node_type = elem_type + '_seq_elem'
                elif elem_type in self.model.ast_wrapper.product_types:
                    child_state = CHILDREN_INQUIRE
                elif elem_type == 'identifier':
                    child_state = GEN_TOKEN
                    child_node_type = 'str'
                elif elem_type in self.model.ast_wrapper.primitive_types:
                    # TODO: Fix this
                    raise ValueError('sequential builtin types not supported')
                else:
                    raise ValueError('Unable to handle seq field type {}'.format(elem_type))

                for i in range(num_children):
                    self.queue = self.queue.append(TreeTraversal.QueueItem(
                        state=child_state,
                        node_type=child_node_type,
                        parent_action_emb=self.prev_action_emb,
                        parent_h=self.cur_item.parent_h,
                        parent_field_name=self.cur_item.parent_field_name,
                    ))

                advanced = self.pop()
                assert advanced
                last_choice = None
                continue

            elif self.cur_item.state == GEN_TOKEN:
                if last_choice == vocab.EOS:
                    if self.pop():
                        last_choice = None
                        continue
                    else:
                        return None

                self.recurrent_state, output, gen_logodds = self.model.gen_token(
                        self.cur_item.node_type,
                        self.recurrent_state,
                        self.prev_action_emb,
                        self.cur_item.parent_h,
                        self.cur_item.parent_action_emb,
                        self.desc_enc)
                self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_gen_token
                return self.token_choice(output, gen_logodds)
            
            elif self.cur_item.state == POINTER_INQUIRE:
                # a. Ask which one to choose
                output, self.recurrent_state, logits = self.model.compute_pointer(
                        self.cur_item.node_type, 
                        self.recurrent_state,
                        self.prev_action_emb,
                        self.cur_item.parent_h, 
                        self.cur_item.parent_action_emb,
                        self.desc_enc)
                self.cur_item = attr.evolve(
                        self.cur_item,
                        state=POINTER_APPLY,
                        parent_h=output)

                self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_pointer
                return self.pointer_choice(self.cur_item.node_type, logits)
            
            elif self.cur_item.state == POINTER_APPLY:
                if self.pop():
                    last_choice = None
                    continue
                else:
                    return None

            elif self.cur_item.state == NODE_FINISHED:
                if self.pop():
                    last_choice = None
                    continue
                else:
                    return None
            
            else:
                raise ValueError('Unknown state {}'.format(self.cur_item.state))
    
    def update_using_last_choice(self, last_choice, extra_choice_info):
        if last_choice is None:
            return
        self.update_prev_action_emb(self, last_choice, extra_choice_info)
    
    @classmethod
    def _update_prev_action_emb_apply_rule(cls, self, last_choice, extra_choice_info):
        # rule_idx shape: batch (=1)
        rule_idx = self.model._tensor([last_choice])
        # action_emb shape: batch (=1) x emb_size
        self.prev_action_emb = self.model.rule_embedding(rule_idx)
    
    @classmethod
    def _update_prev_action_emb_gen_token(cls, self, last_choice, extra_choice_info):
        # token_idx shape: batch (=1), LongTensor
        token_idx = self.model._index(self.model.terminal_vocab, last_choice)
        # action_emb shape: batch (=1) x emb_size
        self.prev_action_emb = self.model.terminal_embedding(token_idx)
    
    @classmethod
    def _update_prev_action_emb_pointer(cls, self, last_choice, extra_choice_info):
        # TODO batching
        self.prev_action_emb = self.model.pointer_action_emb_proj[self.cur_item.node_type](
                self.desc_enc.pointer_memories[self.cur_item.node_type][:, last_choice])

    def pop(self):
        if self.queue:
           self.cur_item = self.queue[-1]
           self.queue = self.queue.delete(-1)
           return True
        return False

    def rule_choice(self, node_type, rule_logits):
        raise NotImplementedError

    def token_choice(self, output, gen_logodds):
        raise NotImplementedError

    def pointer_choice(self, node_type, logits):
        raise NotImplementedError


@attr.s
class ChoiceHistoryEntry:
  rule_left = attr.ib()
  choices = attr.ib()
  probs = attr.ib()
  valid_choices = attr.ib()


class TrainTreeTraversal(TreeTraversal):

    @attr.s(frozen=True)
    class XentChoicePoint:
        logits = attr.ib()
        def compute_loss(self, outer, idx, extra_indices):
            if extra_indices:
                logprobs = torch.nn.functional.log_softmax(self.logits, dim=1)
                valid_logprobs = logprobs[:, [idx] + extra_indices]
                return outer.model.multi_loss_reduction(valid_logprobs)
            else:
                # idx shape: batch (=1)
                idx = outer.model._tensor([idx])
                # loss_piece shape: batch (=1)
                return outer.model.xent_loss(self.logits, idx)

    @attr.s(frozen=True)
    class TokenChoicePoint:
        lstm_output = attr.ib()
        gen_logodds = attr.ib()
        def compute_loss(self, outer, token, extra_tokens):
            return outer.model.gen_token_loss(
                    self.lstm_output,
                    self.gen_logodds,
                    token,
                    outer.desc_enc)

    def __init__(self, model, desc_enc, debug=False):
        super().__init__(model, desc_enc)
        self.choice_point = None
        self.loss = pyrsistent.pvector()

        self.debug = debug
        self.history = pyrsistent.pvector()

    def clone(self):
        super_clone = super().clone()
        super_clone.choice_point = self.choice_point
        super_clone.loss = self.loss
        super_clone.debug = self.debug
        super_clone.history = self.history
        return super_clone

    def rule_choice(self, node_type, rule_logits):
        self.choice_point = self.XentChoicePoint(rule_logits)
        if self.debug:
            choices = []
            probs = []
            for rule_idx, logprob in sorted(
                    self.model.rule_infer(node_type, rule_logits),
                    key=operator.itemgetter(1),
                    reverse=True):
                _, rule = self.model.preproc.all_rules[rule_idx]
                choices.append(rule)
                probs.append(logprob.exp().item())
            self.history = self.history.append(
                    ChoiceHistoryEntry(node_type, choices, probs, None))

    def token_choice(self, output, gen_logodds):
        self.choice_point = self.TokenChoicePoint(output, gen_logodds)
    
    def pointer_choice(self, node_type, logits):
        self.choice_point = self.XentChoicePoint(logits)

    def update_using_last_choice(self, last_choice, extra_choice_info):
        super().update_using_last_choice(last_choice, extra_choice_info)
        if last_choice is None:
            return

        if self.debug and isinstance(self.choice_point, self.XentChoicePoint):
            valid_choice_indices = [last_choice] + ([] if extra_choice_info is None
                else extra_choice_info)
            self.history[-1].valid_choices = [
                self.model.preproc.all_rules[rule_idx][1]
                for rule_idx in valid_choice_indices]

        self.loss = self.loss.append(
                self.choice_point.compute_loss(self, last_choice, extra_choice_info))
        self.choice_point = None


class InferenceTreeTraversal(TreeTraversal):
    class TreeAction:
        pass

    @attr.s(frozen=True)
    class SetParentField(TreeAction):
        parent_field_name = attr.ib()
        node_type = attr.ib()
        node_value = attr.ib(default=None)

    @attr.s(frozen=True)
    class CreateParentFieldList(TreeAction):
        parent_field_name = attr.ib()

    @attr.s(frozen=True)
    class AppendTerminalToken(TreeAction):
        parent_field_name = attr.ib()
        value = attr.ib()

    @attr.s(frozen=True)
    class FinalizeTerminal(TreeAction):
        parent_field_name = attr.ib()
        terminal_type = attr.ib()

    @attr.s(frozen=True)
    class NodeFinished(TreeAction):
        pass

    SIMPLE_TERMINAL_TYPES = {
        'str': str,
        'int': int,
        'float': float,
        'bool': lambda n: {'True': True, 'False': False}.get(n, False),
    }
 
    SIMPLE_TERMINAL_TYPES_DEFAULT = {
        'str': '',
        'int': 0,
        'float': 0,
        'bool': True,
    }

    def __init__(self, model, desc_enc, example=None):
        super().__init__(model, desc_enc)
        self.example = example
        self.actions = pyrsistent.pvector()

    def clone(self):
        super_clone = super().clone()
        super_clone.actions = self.actions
        super_clone.example = self.example
        return super_clone

    def rule_choice(self, node_type, rule_logits):
        return self.model.rule_infer(node_type, rule_logits)

    def token_choice(self, output, gen_logodds):
        return self.model.token_infer(output, gen_logodds, self.desc_enc)

    def pointer_choice(self, node_type, logits):
        # Group them based on pointer map
        pointer_logprobs = self.model.pointer_infer(node_type, logits)
        pointer_map = self.desc_enc.pointer_maps.get(node_type)
        if not pointer_map:
            return pointer_logprobs

        pointer_logprobs = dict(pointer_logprobs)
        return [
            (orig_index, torch.logsumexp(
                torch.stack(
                    tuple(pointer_logprobs[i] for i in mapped_indices),
                    dim=0),
                dim=0))
            for orig_index, mapped_indices in pointer_map.items()
        ]

    def update_using_last_choice(self, last_choice, extra_choice_info):
        super().update_using_last_choice(last_choice, extra_choice_info)

        # Record actions
        # CHILDREN_INQUIRE
        if self.cur_item.state == TreeTraversal.State.CHILDREN_INQUIRE:
            self.actions = self.actions.append(
                self.SetParentField(
                    self.cur_item.parent_field_name,  self.cur_item.node_type))
            type_info = self.model.ast_wrapper.singular_types[self.cur_item.node_type]
            if not type_info.fields:
                self.actions = self.actions.append(self.NodeFinished())

        # LIST_LENGTH_APPLY
        elif self.cur_item.state == TreeTraversal.State.LIST_LENGTH_APPLY:
            self.actions = self.actions.append(self.CreateParentFieldList(self.cur_item.parent_field_name))

        # GEN_TOKEN
        elif self.cur_item.state == TreeTraversal.State.GEN_TOKEN:
            if last_choice == vocab.EOS:
                self.actions = self.actions.append(self.FinalizeTerminal(
                    self.cur_item.parent_field_name,
                    self.cur_item.node_type))
            elif last_choice is not None:
                self.actions = self.actions.append(self.AppendTerminalToken(
                    self.cur_item.parent_field_name,
                    last_choice))

        elif self.cur_item.state == TreeTraversal.State.POINTER_APPLY:
            self.actions = self.actions.append(self.SetParentField(
                    self.cur_item.parent_field_name,
                    node_type=None,
                    node_value=last_choice))

        # NODE_FINISHED
        elif self.cur_item.state == TreeTraversal.State.NODE_FINISHED:
            self.actions = self.actions.append(self.NodeFinished())

    def finalize(self):
        root = current = None
        stack = []
        for i, action in enumerate(self.actions):
            if isinstance(action, self.SetParentField):
                if action.node_value is None:
                    new_node = {'_type': action.node_type}
                else:
                    new_node = action.node_value

                if action.parent_field_name is None:
                    # Initial node in tree.
                    assert root is None
                    root = current = new_node
                    stack.append(root)
                    continue

                existing_list = current.get(action.parent_field_name)
                if existing_list is None:
                    current[action.parent_field_name] = new_node
                else:
                    assert isinstance(existing_list, list)
                    current[action.parent_field_name].append(new_node)

                if action.node_value is None:
                    stack.append(current)
                    current = new_node

            elif isinstance(action, self.CreateParentFieldList):
                current[action.parent_field_name] = []

            elif isinstance(action, self.AppendTerminalToken):
                tokens = current.get(action.parent_field_name)
                if tokens is None:
                    tokens = current[action.parent_field_name] = []
                tokens.append(action.value)

            elif isinstance(action, self.FinalizeTerminal):
                terminal = ''.join(current.get(action.parent_field_name, []))
                constructor = self.SIMPLE_TERMINAL_TYPES.get(action.terminal_type)
                if constructor:
                    try:
                        value = constructor(terminal)
                    except ValueError:
                        value = self.SIMPLE_TERMINAL_TYPES_DEFAULT[action.terminal_type]
                elif action.terminal_type == 'bytes':
                    value = terminal.decode('latin1')
                elif action.terminal_type == 'NoneType':
                    value = None
                else:
                    raise ValueError('Unknown terminal type: {}'.format(action.terminal_type))
                current[action.parent_field_name] = value

            elif isinstance(action, self.NodeFinished):
                current = stack.pop()

            else:
                raise ValueError(action)

        assert not stack
        return root, self.model.preproc.grammar.unparse(root, self.example)
