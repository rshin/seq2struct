import json
import os

import attr
import torch

from seq2struct.models import abstract_preproc
from seq2struct.models import lstm
from seq2struct.utils import registry
from seq2struct.utils import vocab

@attr.s
class SpiderEncoderState:
    state = attr.ib()
    memory = attr.ib()
    words = attr.ib()

    pointer_memories = attr.ib()

    def find_word_occurrences(self, word):
        return [i for i, w in enumerate(self.words) if w == word]


@registry.register('encoder', 'spider')
class SpiderEncoder(torch.nn.Module):
    class Preproc(abstract_preproc.AbstractPreproc):
        def __init__(
                self,
                save_path,
                min_freq=3,
                max_count=5000):
            self.vocab_path = os.path.join(save_path, 'enc_vocab.json')
            self.data_dir = os.path.join(save_path, 'enc')

            self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
            # TODO: Write 'train', 'val', 'test' somewhere else
            self.texts = {'train': [], 'val': [], 'test': []}

            self.vocab = None

        def validate_item(self, item, section):
            return True, None
        
        def add_item(self, item, section, validation_info):
            if section == 'train':
                for token in item.text:
                    self.vocab_builder.add_word(token)
                for column in item.schema.columns:
                    for token in column.name:
                        self.vocab_builder.add_word(token)

            self.texts[section].append(self.preprocess_item(item, validation_info))

        def preprocess_item(self, item, validation_info):
            column_names = []
            for column in item.schema.columns:
                table_name = ['all'] if column.table is None else column.table.name
                column_names.append([column.type] + table_name + column.name)

            return {
                'question': item.text,
                'columns': column_names,
            }

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

    def __init__(
            self,
            device,
            preproc,
            word_emb_size=128,
            recurrent_size=256,
            dropout=0.):
        super().__init__()
        self._device = device
        self.vocab = preproc.vocab

        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size
        assert self.recurrent_size % 2 == 0

        self.embedding = torch.nn.Embedding(
                num_embeddings=len(self.vocab),
                embedding_dim=self.word_emb_size)

        self.question_encoder = lstm.LSTM(
                input_size=self.word_emb_size,
                hidden_size=self.recurrent_size // 2,
                bidirectional=True,
                dropout=dropout)
                
        self.column_name_encoder = lstm.LSTM(
                input_size=self.word_emb_size,
                hidden_size=self.recurrent_size // 2,
                bidirectional=True,
                dropout=dropout)

        #self.column_set_encoder = lstm.LSTM(
        #        input_size=self.recurrent_size,
        #        hidden_size=self.recurrent_size // 2,
        #        bidirectional=True,
        #        dropout=dropout)

    def forward(self, desc):
        # emb shape: desc length x batch (=1) x word_emb_size
        question_emb = self._embed_words(desc['question'], self.embedding)

        # outputs shape: desc length x batch (=1) x recurrent_size
        # state shape:
        # - h: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
        # - c: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
        question_outputs, question_state = self.question_encoder(question_emb)

        # column_embs: list of batch (=1) x recurrent size
        column_embs = []
        for column in desc['columns']:
            column_name_embs = self._embed_words(column, self.embedding)

            # outputs shape: desc length x batch (=1) x recurrent_size
            # state shape:
            # - h: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
            # - c: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
            _, (h, c) = self.column_name_encoder(column_name_embs)
            column_embs.append(torch.cat((h[0], h[1]), dim=-1))

        #columns_outputs, columns_state = self.column_set_encoder(torch.stack(column_embs, dim=0))
        #columns_outputs = columns_outputs.transpose(0, 1)
        columns_outputs = torch.stack(column_embs, dim=1)

        return SpiderEncoderState(
            state=question_state,
            memory=question_outputs.transpose(0, 1),
            words=desc['question'],
            pointer_memories={'column': columns_outputs})
    
    def _embed_words(self, tokens, emb_module):
        # token_indices shape: batch (=1) x length
        token_indices = torch.tensor(
                self.vocab.indices(tokens),
                device=self._device).unsqueeze(0)

        # emb shape: batch (=1) x length x word_emb_size
        emb = emb_module(token_indices)

        # return value shape: desc length x batch (=1) x word_emb_size
        return emb.transpose(0, 1)
