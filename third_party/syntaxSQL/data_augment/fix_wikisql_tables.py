import re
import json

BANNED = re.compile(r'[\'"\[\]().:]')

tables = json.load(open('wikisql_tables.json'))
for schema in tables:
    for i, (table_id, column_name) in enumerate(schema['column_names_original']):
        if table_id == -1:
            continue
        column_name = BANNED.sub('', column_name)
        if column_name in ('', '#', '!', '%', '$'):
            column_name = schema['column_names'][i][1]
            column_name = re.sub('( +)', '_', column_name)
            column_name = column_name.replace('#', 'hash')
            column_name = column_name.replace('!', 'exclamation_point')
            column_name = column_name.replace('%', 'percent')
            column_name = column_name.replace('$', 'dollar')
            column_name = BANNED.sub('', column_name)
        schema['column_names_original'][i] = (table_id, column_name)
    
    for i, table_name in enumerate(schema['table_names_original']):
        table_name = BANNED.sub('', table_name)
        if table_name == 'as':
            table_name = 'as_'
        if table_name == '':
            table_name = 'empty'
        schema['table_names_original'][i] = table_name

with open('wikisql_tables.json', 'w') as f:
    json.dump(tables, f, sort_keys=True, indent=4, separators=(',', ': '))