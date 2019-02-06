import os, sys
import json
import sqlite3
import traceback
import argparse

from third_party.spider.process_sql import get_sql
from third_party.spider.preprocess.schema import Schema

#TODO: update the following dirs
#sql_path = 'spider/train.json'
#db_dir = 'database/'
#output_file = 'train_new.json'
#table_file = 'spider/tables.json'


def get_schemas_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    db_names = [db['db_id'] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db['db_id']
        schema = {} #{'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db['column_names_original']
        table_names_original = db['table_names_original']
        tables[db_id] = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--tables', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    sql_path = args.input
    output_file = args.output
    table_file = args.tables

    schemas, db_names, tables = get_schemas_from_json(table_file)

    with open(sql_path) as inf:
        sql_data = json.load(inf)

    sql_data_new = []
    for data in sql_data:
        try:
            db_id = data["db_id"]
            schema = schemas[db_id]
            table = tables[db_id]
            schema = Schema(schema, table)
            sql = data["query"]
            sql_label = get_sql(schema, sql)
            data["sql"] = sql_label
            sql_data_new.append(data)
        except:
            print("db_id: ", db_id)
            print("sql: ", sql)
    
    with open(output_file, 'wt') as out:
        json.dump(sql_data_new, out, sort_keys=True, indent=4, separators=(',', ': '))
