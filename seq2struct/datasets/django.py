import itertools
import json

import attr
import torch.utils.data

from seq2struct.utils import registry


@attr.s
class DjangoItem:
    text = attr.ib()
    code = attr.ib()
    str_map = attr.ib()


@registry.register('dataset', 'django')
class DjangoDataset(torch.utils.data.Dataset): 
    def __init__(self, path, limit=None):
        self.path = path
        self.examples = []
        for line in itertools.islice(open(self.path), limit):
            example = json.loads(line)
            self.examples.append(DjangoItem(
                text=example['text']['tokens'],
                code=example['orig'],
                str_map=example['text']['str_map']))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
