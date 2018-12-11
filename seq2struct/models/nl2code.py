import torch

import attr

from seq2struct.utils import registry
from seq2struct.utils import vocab
from seq2struct.cobatch import torch_batcher


@attr.ib
class TreeState:
    node = attr.ib()
    parent_action_emb = attr.ib()
    parent_h = attr.ib()
    sum_type = attr.ib()


def lstm_init(device, num_layers, hidden_size, *batch_sizes):
    init_size = (num_layers, ) + batch_sizes + (hidden_size, )
    init = torch.zeros(*init_size, device=device)
    return (init, init)


def split_string_whitespace_and_camelcase(s):
    split_space = s.split(' ')
    result = []
    for token in split_space:
        if token:
            camelcase_split_token = re.sub('([a-z])([A-Z])', '\\1\uE012\\2', token).split('\uE012')
            result.extend(camelcase_split_token)
        result.append(' ')
    return result[:-1]


@registry.register('decoder', 'NL2Code')
class NL2CodeModel(torch.nn.Module):

    def __init__(self, config):
        self._device = torch.device('cpu') # FIXME
        self.ast_wrapper = None  # FIXME
        self.terminal_vocab = None  # FIXME
        self.emb_size = config['emb_size']

        self.all_rules, self.rules_mask = self._calculate_rules(
                config, self.ast_wrapper)
        self.rules_index = {v: idx for idx, v in enumerate(all_rules)}

        self.node_type_vocab = Vocab(
                ['str', 'int', 'float'] +
                list(ast_wrapper.sum_types.keys()) +
                list(ast_wrapper.singular_types.keys()))

        self.state_update = torch.nn.LSTM(
                self.emb_size * 4,
                self.emb_size,
                num_layers=2)

        self.rule_logits = torch.nn.Sequential(
                torch.nn.Linear(self.emb_size, self.emb_size),
                torch.nn.Tanh(),
                torch.nn.Linear(self.emb_size, len(all_rules)))
        self.rule_embedding = torch.nn.Embedding(
                num_embeddings=len(all_rules),
                embeding_dim=self.emb_size)

        self.use_gen_token_logodds = torch.nn.Linear(self.emb_size, 1)

        self.node_type_embedding = torch.nn.Embedding(
                num_embeddings=len(self.node_type_vocab),                embedding_dim=self.emb_size)

        self.zero_emb = torch.zeros(1, self.emb.size, device=self._device)


    @classmethod
    def _calculate_rules(cls, config, ast_wrapper):
        offset = 0

        all_rules = []
        rules_mask = {}

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        for parent, children in ast_wrapper.sum_type_vocabs.items():
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
        all_node_types = set()
        all_seq_types = set()
        for name, info in ast_wrapper.singular_types.items():
            presence_prod = list(itertools.product(
                    *((True, False) if field.opt or field.seq else (True,)
                        for field in info.fields)))
            rules_mask[name] = (offset, offset + len(presence_prod))
            offset += len(presence_prod)
            all_rules += [(name, presence) for presence in presence_prod]

            for field in info.fields:
                if field.seq:
                    all_seq_types.add(field.type)

        # Rules of the form:
        # stmt* -> stmt
        #        | stmt stmt
        #        | stmt stmt stmt
        max_seq_length = config['max_seq_length']
        for seq_type in sorted(all_seq_types):
            rules_mask[seq_type + '*'] = (offset, offset + max_seq_length)
            offset += max_seq_length
            all_rules += [(name, i) for i in range(1, max_seq_length + 1)]

        return all_rules, rules_mask

    def compute_loss(self, root):
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
        state = lstm_init(self._device, 2, self.emb_size, 1)
        prev_action_emb = self.zero_emb

        loss = []

        # Perform a DFS
        queue = [
            TreeState(
                node=root,
                parent_action_emb=self.zero_emb,
                parent_h=self.zero_emb,
                sum_type=None)
        ]
        while queue:
            node, parent_action_emb, parent_h, sum_type = queue.pop()
            type_info = self.ast_wrapper.singular_types[node['_type']]

            if sum_type is not None:
                # ApplyRule, like expr -> Call
                # TODO: rule_idx needs to be torch.Tensor
                rule = (sum_type, type_info.name)
                output, state, prev_action_emb, loss_piece = self.apply_rule(
                        sum_type,
                        rule,
                        prev_action_emb,
                        parent_h, 
                        parent_action_emb)
                
                parent_h = output
                parent_action_emb = prev_action_emb
                loss.append(loss_piece)

            # ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
            # Figure out which rule needs to be applied
            present = []
            for field_info in type_info.fields:
                field_value = node.get(field_info.name)
                present.append(field_value is not None and field_value != [])
            rule = (type_info.name, tuple(present))
            output, state, prev_action_emb, loss_piece = self.apply_rule(
                    node['_type'],
                    rule_idx,
                    prev_action_emb,
                    parent_h,
                    parent_action_emb)
            loss.append(loss_piece)

            # reversed so that we perform a DFS in left-to-right order
            for field_info in reversed(type_info.fields):
                field_value = node.get(field_info.name)
                if field_value is None or field_value == []:
                    continue

                # - identifier, int, string, bytes, object, singleton
                #   - could be str, int, float, object, bool
                #   - terminal tokens vocabulary is created by turning everything into a string (with `str`)
                #   - at decoding time, cast back to str/int/float/bool
                if isinstance(field_value, str, bytes, int, float):
                    if isinstance(field_value, bytes):
                        field_value = field_value.encode('latin1')
                    else:
                        field_value = str(field_value)
                        field_type = type(field_value).__name__
                    field_value_split = split_string_whitespace_and_camelcase(field_value)

                    # Handle literal
                    for token in field_value_split:
                        output, state, prev_action_emb, loss_piece = self.gen_token(
                                field_type,
                                token,
                                prev_action_emb,
                                parent_h, 
                                parent_action_emb)

                    continue

                if field_info.type in ast_wrapper.sum_types:
                    # - sum type
                    child_sum_type = field_info.type
                else:
                    # - product type
                    # No special handling needed.
                    child_sum_type = None

                queue.append(
                    TreeState(
                        node=field_value,
                        parent_action_emb=prev_action_emb,
                        parent_h=output,
                        # FIXME
                        sum_type=type_info.name))


            # TODO
            #context = None
            #state_input = torch.cat((
            #    prev_action_emb,
            #    parent_action_emb,
            #    parent_state,
            #    node_type_emb,
            #    context), dim=-1)
            #output, new_state = self.state_update(
            #    state_input.unsqueeze(dim=0), prev_state).squeeze(0)

            #for field_info in reversed(type_info.fields):
            #    pass


    def apply_rule(self, node_type, rule, prev_action_emb, parent_h,
             parent_action_emb):
        context = None # TODO
        node_type_emb = self.node_type_embedding(
                self.node_type_vocab.index(node_type))
        rule_idx = self.rules_index[rule]

        state_input = torch.cat(
            (
                prev_action_emb,  # a_{t-1}
                context,  # c_t
                parent_h,  # s_{p_t}
                parent_action_emb,  # a_{p_t}
                node_type_emb,  # n_{f-t}
            ),
            dim=-1)
        output, new_state = self.state_update(
                state_input.unsqueeze(dim=0), prev_state).squeeze(0)

        action_emb = self.rule_embedding(rule_idx)
        loss_piece = torch.nn.functional.cross_entropy(
                self.rule_logits(output),
                rule_idx)
        return output, new_state, action_emb, loss_piece


    def gen_token(self, node, token, prev_action_emb, parent_h, parent_action_emb):
        # TODO implement copying
        context = None # TODO
                
        state_input = torch.cat(
            (
                prev_action_emb,  # a_{t-1}
                context,  # c_t
                parent_h,  # s_{p_t}
                parent_action_emb,  # a_{p_t}
                node_type_emb,  # n_{f-t}
            ),
            dim=-1)

