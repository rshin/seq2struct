import json

import attr
import torch

from seq2struct.utils import registry


@attr.s
class SpiderItem:
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()


@attr.s
class Column:
    table = attr.ib()
    name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()


@attr.s
class Table:
    name = attr.ib()
    orig_name = attr.ib()


@attr.s
class Schema:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()


@registry.register('dataset', 'spider')
class SpiderDataset(torch.utils.data.Dataset): 
    def __init__(self, paths, tables_path, limit=None):
        self.paths = paths
        self.examples = []
        self.schemas = {}

        for schema_dict in json.load(open(tables_path)):
            tables = tuple(Table(name.split(), orig_name) for name, orig_name in zip(
                schema_dict['table_names'], schema_dict['table_names_original']))
            columns = tuple(
                Column(
                    table=tables[table_id] if table_id >= 0 else None,
                    name=col_name.split(),
                    orig_name=orig_col_name,
                    type=col_type,
                )
                for (table_id, col_name), (_, orig_col_name), col_type in zip(
                    schema_dict['column_names'], 
                    schema_dict['column_names_original'],
                    schema_dict['column_types'])
            )
            db_id = schema_dict['db_id']
            self.schemas[db_id] = Schema(db_id, tables, columns)

        for path in paths:
            raw_data = json.load(open(path))
            for entry in raw_data:
                item = SpiderItem(
                    text=entry['question_toks'],
                    code=entry['sql'],
                    schema=self.schemas[entry['db_id']],
                    orig=entry)
                self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
