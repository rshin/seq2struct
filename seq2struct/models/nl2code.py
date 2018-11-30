import torch

from seq2struct.utils import registry
from seq2struct.cobatch import torch_batcher


@registry.register('decoder', 'NL2Code')
class NL2CodeModel(torch.nn.Module):

    def __init__(self, config):
        self.ast_wrapper = None  # FIXME
        self.emb_size = config['emb_size']

        rules_mask = {}
        all_rules = []
        offset = 0

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        for parent, children in self.ast_wrapper.sum_type_vocabs.items():
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
        all_seq_types = set()
        for name, info in self.ast_wrapper.singular_types.items():
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

        self.all_rules = all_rules
        self.rules_mask = rules_mask
        self.rules_index = {v: idx for idx, v in enumerate(all_rules)}

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
        prev_state      = None
        prev_action_emb = None

        # Perform a DFS
        queue = [(root, None, None)]
        while queue:
            node, parent_action_emb, sum_type = queue.pop()
            type_info = self.ast_wrapper.singular_types[node._type]

            if sum_type is not None:
                rule_idx = self.rules_index[sum_type, type_info.name]

            present = []
            for field_info in type_info.fields:
                field_value = getattr(node, field_info.name, default=None)
                present.append(field_value is not None and field_value != [])

            # Figure out which rule needs to be applied
            rule_idx = self.rules_index[type_info.name, tuple(present)]
            rule_emb = self.rule_embedding(rule_idx)

            # TODO
            context = None

            state_input = torch.cat((
                prev_action_emb,
                parent_action_emb,
                parent_state,
                node_type_emb,
                context), dim=-1)
            output, new_state = self.state_update(
                state_input.unsqueeze(dim=0), prev_state).squeeze(0)

            for field_info in reversed(type_info.fields):
                pass


    def step(self, node_type,
            prev_state,
            prev_action,
            parent_action,
            parent_state):

        # Update LSTM state
        prev_action_emb = None
        parent_action_emb = None
        node_type_emb = None
        context = None
        # state_input shape: batch x emb size
        state_input = torch.cat((
            prev_action_emb,
            parent_action_emb,
            parent_state,
            node_type_emb,
            context), dim=-1)

        output, new_state = self.state_update(
            state_input.unsqueeze(dim=0), prev_state).squeeze(0)
