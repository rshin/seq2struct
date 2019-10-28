class TemplateTreeTraversal:

    def __init__(self, model, desc_enc):
        super().__init__(model, desc_enc)
        self.steps = []
        self.holes = []

    def step(self, last_choice, extra_choice_info=None):
        index = len(self.steps)
        self.steps.append(last_choice)
        return index

    # TODO reduce duplication with compute_loss
    def traverse_tree(self, tree, parent_field_type=None):
        queue = [
            TreeState(
                node=tree,
                parent_field_type=parent_field_type,
            )
        ]
        while queue:
            item = queue.pop()
            node = item.node
            parent_field_type = item.parent_field_type

            if isinstance(node, (list, tuple)):
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
                # TODO: Put back the following line
                #assert self.cur_item.state == TreeTraversal.State.LIST_LENGTH_APPLY
                if len(rule_indices) == 1:
                    self.step(rule_indices[0])
                else:
                    self.step(rule_indices)

                if self.model.preproc.use_seq_elem_rules and parent_field_type in self.model.ast_wrapper.sum_types:
                    parent_field_type += '_seq_elem'

                for i, elem in reversed(list(enumerate(node))):
                    queue.append(
                        TreeState(
                            node=elem,
                            parent_field_type=parent_field_type,
                        ))
                continue
            
            if isinstance(node, ast_util.HoleValuePlaceholder):
                self.step(node)
                continue

            if parent_field_type in self.model.preproc.grammar.pointers:
                assert isinstance(node, int)
                # TODO: Put back the following line
                #assert self.cur_item.state == TreeTraversal.State.POINTER_APPLY
                self.step(node)
                continue

            if parent_field_type in self.model.ast_wrapper.primitive_types:
                # identifier, int, string, bytes, object, singleton
                # - could be bytes, str, int, float, bool, NoneType
                # - terminal tokens vocabulary is created by turning everything into a string (with `str`)
                # - at decoding time, cast back to str/int/float/bool
                field_type = type(node).__name__
                field_value_split = self.model.preproc.grammar.tokenize_field_value(node) + [
                        vocab.EOS]

                for token in field_value_split:
                    # TODO: Put back the following line
                    #assert self.cur_item.state == TreeTraversal.State.GEN_TOKEN
                    self.step(token)
                continue
            
            type_info = self.model.ast_wrapper.singular_types[node['_type']]

            if parent_field_type in self.model.preproc.sum_type_constructors:
                # ApplyRule, like expr -> Call
                rule = (parent_field_type, type_info.name)
                rule_idx = self.model.rules_index[rule]
                # TODO: Put back the following line
                #assert self.cur_item.state == TreeTraversal.State.SUM_TYPE_APPLY
                extra_rules = [
                    self.model.rules_index[parent_field_type, extra_type]
                    for extra_type in node.get('_extra_types', [])]
                self.step(rule_idx, extra_rules)

            if type_info.fields:
                # ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
                # Figure out which rule needs to be applied
                present = get_field_presence_info(self.model.ast_wrapper, node, type_info.fields)

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
                        self.step(rule_indices[0])
                    else:
                        self.step(rule_indices)
                else:
                    rule = (node['_type'], tuple(present))
                    rule_idx = self.model.rules_index[rule]
                    # TODO: Put back following line
                    #assert self.cur_item.state == TreeTraversal.State.CHILDREN_APPLY
                    self.step(rule_idx)

            # reversed so that we perform a DFS in left-to-right order
            for field_info in reversed(type_info.fields):
                if field_info.name not in node:
                    continue

                queue.append(
                    TreeState(
                        node=node[field_info.name],
                        parent_field_type=field_info.type,
                    ))
