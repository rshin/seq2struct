import json

import attr
import torch

from seq2struct.utils import registry


@attr.s
class SpiderItem:
    text = attr.ib()
    code = attr.ib()
    orig = attr.ib()


@registry.register('dataset', 'spider')
class SpiderDataset(torch.utils.data.Dataset): 
    def __init__(self, paths, tables_path, limit=None):
        self.paths = paths
        self.examples = []
        for path in paths:
            raw_data = json.load(open(path))
            for entry in raw_data:
                self.examples.append(SpiderItem(
                    text=entry['question_toks'],
                    code=entry['sql'],
                    orig=entry))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
