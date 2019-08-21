import collections
import itertools
import json
import os

import attr
import torch
import torchtext

from seq2struct.models import abstract_preproc

try:
    from seq2struct.models import lstm
except ImportError:
    pass
from seq2struct.models import spider_enc_modules
from seq2struct.utils import registry
from seq2struct.utils import vocab
from seq2struct.utils import serialization


@attr.s
class SpiderEncoderState:
    state = attr.ib()
    memory = attr.ib()
    question_memory = attr.ib()
    schema_memory = attr.ib()
    words = attr.ib()

    pointer_memories = attr.ib()
    pointer_maps = attr.ib()

    def find_word_occurrences(self, word):
        return [i for i, w in enumerate(self.words) if w == word]


@attr.s
class PreprocessedSchema:
    column_names = attr.ib(factory=list)
    table_names = attr.ib(factory=list)
    table_bounds = attr.ib(factory=list)
    column_to_table = attr.ib(factory=dict)
    table_to_columns = attr.ib(factory=dict)
    foreign_keys = attr.ib(factory=dict)
    foreign_keys_tables = attr.ib(factory=lambda: collections.defaultdict(set))
    primary_keys = attr.ib(factory=list)


class SpiderEncoderV2Preproc(abstract_preproc.AbstractPreproc):

    def __init__(
            self,
            save_path,
            min_freq=3,
            max_count=5000,
            include_table_name_in_column=True,
            word_emb=None,
            count_tokens_in_word_emb_for_vocab=False):
        if word_emb is None:
            self.word_emb = None
        else:
            self.word_emb = registry.construct('word_emb', word_emb)

        self.data_dir = os.path.join(save_path, 'enc')
        self.include_table_name_in_column = include_table_name_in_column
        self.count_tokens_in_word_emb_for_vocab = count_tokens_in_word_emb_for_vocab
        # TODO: Write 'train', 'val', 'test' somewhere else
        self.texts = {'train': [], 'val': [], 'test': []}

        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        self.vocab_path = os.path.join(save_path, 'enc_vocab.json')
        self.vocab = None
        self.counted_db_ids = set()
        self.preprocessed_schemas = {}

    def validate_item(self, item, section):
        return True, None
 
    def add_item(self, item, section, validation_info):
        preprocessed = self.preprocess_item(item, validation_info)
        self.texts[section].append(preprocessed)

        if section == 'train':
            if item.schema.db_id in self.counted_db_ids:
                to_count = preprocessed['question']
            else:
                self.counted_db_ids.add(item.schema.db_id)
                to_count = itertools.chain(
                        preprocessed['question'],
                        *preprocessed['columns'],
                        *preprocessed['tables'])

            for token in to_count:
                count_token = (
                    self.word_emb is None or
                    self.count_tokens_in_word_emb_for_vocab or
                    self.word_emb.lookup(token) is None)
                if count_token:
                    self.vocab_builder.add_word(token)

    def preprocess_item(self, item, validation_info):
        if self.word_emb:
            question = self.word_emb.tokenize(item.orig['question'])
        else:
            question = item.text

        preproc_schema = self._preprocess_schema(item.schema)

        return {
            'question': self._tokenize(item.text, item.orig['question']),
            'db_id': item.schema.db_id,
            'columns': preproc_schema.column_names,
            'tables': preproc_schema.table_names,
            'table_bounds': preproc_schema.table_bounds,
            'column_to_table': preproc_schema.column_to_table,
            'table_to_columns': preproc_schema.table_to_columns,
            'foreign_keys': preproc_schema.foreign_keys,
            'foreign_keys_tables': preproc_schema.foreign_keys_tables,
            'primary_keys': preproc_schema.primary_keys,
        }

    def _preprocess_schema(self, schema):
        if schema.db_id in self.preprocessed_schemas:
            return self.preprocessed_schemas[schema.db_id]
        result = self._preprocess_schema_uncached(schema)
        self.preprocessed_schemas[schema.db_id] = result
        return result
    
    def _preprocess_schema_uncached(self, schema):
        r = PreprocessedSchema()

        last_table_id = None
        for i, column in enumerate(schema.columns):
            column_name = ['<type: {}>'.format(column.type)] + self._tokenize(
                    column.name, column.unsplit_name)
            if self.include_table_name_in_column:
                if column.table is None:
                    table_name = ['<any-table>']
                else:
                    table_name = self._tokenize(
                        column.table.name, column.table.unsplit_name)
                column_name += ['<table-sep>'] + table_name
            r.column_names.append(column_name)

            table_id = None if column.table is None else column.table.id
            r.column_to_table[str(i)] = table_id
            if table_id is not None:
                columns = r.table_to_columns.setdefault(str(table_id), [])
                columns.append(i)
            if last_table_id != table_id:
                r.table_bounds.append(i)
                last_table_id = table_id

            if column.foreign_key_for is not None:
                r.foreign_keys[str(column.id)] = column.foreign_key_for.id
                r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)

        r.table_bounds.append(len(schema.columns))
        assert len(r.table_bounds) == len(schema.tables) + 1

        for i, table in enumerate(schema.tables):
            r.table_names.append(self._tokenize(
                table.name, table.unsplit_name))

        r.foreign_keys_tables = serialization.to_dict_with_sorted_values(r.foreign_keys_tables)
        r.primary_keys = [
            column.id
            for column in table.primary_keys
            for table in schema.tables
        ]

        return r

    def _tokenize(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize(unsplit)
        return presplit

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab = self.vocab_builder.finish()
        self.vocab.save(self.vocab_path)

        for section, texts in self.texts.items():
            with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                for text in texts:
                    f.write(json.dumps(text) + '\n')

    def load(self):
        self.vocab = vocab.Vocab.load(self.vocab_path)

    def dataset(self, section):
        return [
            json.loads(line)
            for line in open(os.path.join(self.data_dir, section + '.jsonl'))]


@registry.register('encoder', 'spiderv2')
class SpiderEncoderV2(torch.nn.Module):

    batched = True
    Preproc = SpiderEncoderV2Preproc

    def __init__(
            self,
            device,
            preproc,
            word_emb_size=128,
            recurrent_size=256,
            dropout=0.,
            question_encoder=('emb', 'bilstm'),
            column_encoder=('emb', 'bilstm'),
            table_encoder=('emb', 'bilstm'),
            update_config={},
            include_in_memory=('question', 'column', 'table'),
            batch_encs_update=True,
            ):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.vocab = preproc.vocab

        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size
        assert self.recurrent_size % 2 == 0
        self.include_in_memory = set(include_in_memory)
        self.dropout = dropout

        self.question_encoder = self._build_modules(question_encoder)
        self.column_encoder = self._build_modules(column_encoder)
        self.table_encoder = self._build_modules(table_encoder)

        update_modules = {
            'relational_transformer': 
            spider_enc_modules.RelationalTransformerUpdate,
            'none':
            spider_enc_modules.NoOpUpdate,
        }

        self.encs_update = registry.instantiate(
            update_modules[update_config['name']],
            update_config,
            device=self._device,
            hidden_size=recurrent_size,
            )
        self.batch_encs_update = batch_encs_update

    def _build_modules(self, module_types):
        module_builder = {
            'emb': lambda: spider_enc_modules.LookupEmbeddings(
                self._device,
                self.vocab,
                self.preproc.word_emb,
                self.word_emb_size),
            'linear': lambda: spider_enc_modules.EmbLinear(
                input_size=self.word_emb_size,
                output_size=self.word_emb_size),
            'bilstm': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=False),
            'bilstm-native': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=False,
                use_native=True),
            'bilstm-summarize': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=True),
            'bilstm-native-summarize': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=True,
                use_native=True),
        }

        modules = []
        for module_type in module_types:
            modules.append(module_builder[module_type]())
        return torch.nn.Sequential(*modules)


    def forward_unbatched(self, desc):
        # Encode the question
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?

        # q_enc: question len x batch (=1) x recurrent_size
        q_enc, (_, _) = self.question_encoder([desc['question']])

        # Encode the columns
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?
        # - Summarize each column into one?
        # c_enc: sum of column lens x batch (=1) x recurrent_size
        c_enc, c_boundaries = self.column_encoder(desc['columns'])
        column_pointer_maps = {
            i: list(range(left, right))
            for i, (left, right) in enumerate(zip(c_boundaries, c_boundaries[1:]))
        }

        # Encode the tables
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?
        # - Summarize each table into one?
        # t_enc: sum of table lens x batch (=1) x recurrent_size
        t_enc, t_boundaries = self.table_encoder(desc['tables'])
        c_enc_length = c_enc.shape[0]
        table_pointer_maps = {
            i: [
                idx 
                for col in desc['table_to_columns'][str(i)]
                for idx in column_pointer_maps[col]
            ] +  list(range(left + c_enc_length, right + c_enc_length))
            for i, (left, right) in enumerate(zip(t_boundaries, t_boundaries[1:]))
        }

        # Update each other using self-attention
        # q_enc_new, c_enc_new, and t_enc_new now have shape
        # batch (=1) x length x recurrent_size
        q_enc_new, c_enc_new, t_enc_new = self.encs_update(
                desc, q_enc, c_enc, c_boundaries, t_enc, t_boundaries)
        
        memory = []
        if 'question' in self.include_in_memory:
            memory.append(q_enc_new)
        if 'column' in self.include_in_memory:
            memory.append(c_enc_new)
        if 'table' in self.include_in_memory:
            memory.append(t_enc_new)
        memory = torch.cat(memory, dim=1)

        return SpiderEncoderState(
            state=None,
            memory=memory,
            # TODO: words should match memory
            words=desc['question'],
            pointer_memories={
                'column': c_enc_new,
                'table': torch.cat((c_enc_new, t_enc_new), dim=1),
            },
            pointer_maps={
                'column': column_pointer_maps,
                'table': table_pointer_maps,
            }
        )

    def forward(self, descs):
        # Encode the question
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?

        # q_enc: PackedSequencePlus, [batch, question len, recurrent_size]
        q_enc, _ = self.question_encoder([[desc['question']] for desc in descs])

        # Encode the columns
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?
        # - Summarize each column into one?
        # c_enc: PackedSequencePlus, [batch, sum of column lens, recurrent_size]
        c_enc, c_boundaries = self.column_encoder([desc['columns'] for desc in descs])

        column_pointer_maps = [
            {
                i: list(range(left, right))
                for i, (left, right) in enumerate(zip(c_boundaries_for_item, c_boundaries_for_item[1:]))
            }
            for batch_idx, c_boundaries_for_item in enumerate(c_boundaries)
        ]

        # Encode the tables
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?
        # - Summarize each table into one?
        # t_enc: PackedSequencePlus, [batch, sum of table lens, recurrent_size]
        t_enc, t_boundaries = self.table_encoder([desc['tables'] for desc in descs])

        c_enc_lengths = list(c_enc.orig_lengths())
        table_pointer_maps = [
            {
                i: [
                    idx
                    for col in desc['table_to_columns'][str(i)]
                    for idx in column_pointer_maps[batch_idx][col]
                ] +  list(range(left + c_enc_lengths[batch_idx], right + c_enc_lengths[batch_idx]))
                for i, (left, right) in enumerate(zip(t_boundaries_for_item, t_boundaries_for_item[1:]))
            }
            for batch_idx, (desc, t_boundaries_for_item) in enumerate(zip(descs, t_boundaries))
        ]

        # Update each other using self-attention
        # q_enc_new, c_enc_new, and t_enc_new are PackedSequencePlus with shape
        # batch (=1) x length x recurrent_size
        if self.batch_encs_update:
            q_enc_new, c_enc_new, t_enc_new = self.encs_update(
                    descs, q_enc, c_enc, c_boundaries, t_enc, t_boundaries)
 
        result = []
        for batch_idx, desc in enumerate(descs):
            if self.batch_encs_update:
                q_enc_new_item = q_enc_new.select(batch_idx).unsqueeze(0)
                c_enc_new_item = c_enc_new.select(batch_idx).unsqueeze(0)
                t_enc_new_item = t_enc_new.select(batch_idx).unsqueeze(0)
            else:
                q_enc_new_item, c_enc_new_item, t_enc_new_item = \
                        self.encs_update.forward_unbatched(
                                desc,
                                q_enc.select(batch_idx).unsqueeze(1),
                                c_enc.select(batch_idx).unsqueeze(1),
                                c_boundaries[batch_idx],
                                t_enc.select(batch_idx).unsqueeze(1),
                                t_boundaries[batch_idx])

            memory = []
            if 'question' in self.include_in_memory:
                memory.append(q_enc_new_item)
            if 'column' in self.include_in_memory:
                memory.append(c_enc_new_item)
            if 'table' in self.include_in_memory:
                memory.append(t_enc_new_item)
            memory = torch.cat(memory, dim=1)

            result.append(SpiderEncoderState(
                state=None,
                memory=memory,
                question_memory=q_enc_new_item,
                schema_memory=torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                # TODO: words should match memory
                words=desc['question'],
                pointer_memories={
                    'column': c_enc_new_item,
                    'table': torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                },
                pointer_maps={
                    'column': column_pointer_maps[batch_idx],
                    'table': table_pointer_maps[batch_idx],
                }
            ))
        return result

