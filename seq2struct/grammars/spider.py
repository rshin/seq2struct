import collections
import copy
import itertools
import os

import asdl
import attr
import networkx as nx

from seq2struct import ast_util
from seq2struct.utils import registry


def bimap(first, second):
    return {f: s for f, s in  zip(first, second)}, {s: f for f, s in zip(first, second)}


def filter_nones(d):
    return {k: v for k, v in d.items() if v is not None and v != []}


def join(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x


def intersperse(delimiter, seq):
    return itertools.islice(
        itertools.chain.from_iterable(
            zip(itertools.repeat(delimiter), seq)), 1, None)


@registry.register('grammar', 'spider')
class SpiderLanguage:

    root_type = 'sql'
    #pointers = {'column', 'table'}
    pointers = {'column'}

    def __init__(self, output_from=False):
        self.ast_wrapper = ast_util.ASTWrapper(
                asdl.parse(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        'Spider.asdl')),
                custom_primitive_type_checkers = {
                    'column': lambda x: isinstance(x, int),
                    #'table': lambda x: isinstance(x, int),
                })
        self.output_from = output_from
        if not self.output_from:
            sql_fields = self.ast_wrapper.product_types['sql'].fields
            assert sql_fields[1].name == 'from'
            del sql_fields[1]

    def parse(self, code, section):
        return self.parse_sql(code)

    def unparse(self, tree, item):
        unparser = SpiderUnparser(self.ast_wrapper, item.schema)
        return unparser.unparse_sql(tree)

    @classmethod
    def tokenize_field_value(cls, field_value):
        if isinstance(field_value, bytes):
            field_value_str = field_value.encode('latin1')
        elif isinstance(field_value, str):
            field_value_str = field_value
        else:
            field_value_str = str(field_value)
            if field_value_str[0] == '"' and field_value_str[-1] == '"':
                field_value_str = field_value_str[1:-1]
        # TODO: Get rid of surrounding quotes
        return [field_value_str]

    #
    #
    #

    def parse_val(self, val):
        if isinstance(val, str):
            return {
                    '_type': 'String',
                    's': val,
            }
        elif isinstance(val, list):
            return {
                    '_type': 'ColUnit',
                    'c': self.parse_col_unit(val),
            }
        elif isinstance(val, float):
            return {
                    '_type': 'Number',
                    'f': val,
            }
        elif isinstance(val, dict):
            return {
                    '_type': 'ValSql',
                    's': self.parse_sql(val),
            }
        else:
            raise ValueError(val)

    def parse_col_unit(self, col_unit):
        agg_id, col_id, is_distinct = col_unit
        return {
                '_type': 'col_unit',
                'agg_id': {'_type': self.AGG_TYPES_F[agg_id]},
                'col_id': col_id,
                'is_distinct': is_distinct,
        }

    def parse_val_unit(self, val_unit):
        unit_op, col_unit1, col_unit2 = val_unit
        result =  {
                '_type': self.UNIT_TYPES_F[unit_op],
                'col_unit1': self.parse_col_unit(col_unit1),
        }
        if unit_op != 0:
            result['col_unit2'] = self.parse_col_unit(col_unit2)
        return result

    def parse_table_unit(self, table_unit):
        table_type, value = table_unit
        if table_type == 'sql':
            return {
                    '_type': 'TableUnitSql',
                    's': self.parse_sql(value),
            }
        elif table_type == 'table_unit':
            return {
                    '_type': 'Table',
                    'table_id': value,
            }
        else:
            raise ValueError(table_type)

    def parse_cond(self, cond, optional=False):
        if optional and not cond:
            return None

        if len(cond) > 1:
            return {
                    '_type': self.LOGIC_OPERATORS_F[cond[1]],
                    'left': self.parse_cond(cond[:1]),
                    'right': self.parse_cond(cond[2:]),
            }

        (not_op, op_id, val_unit, val1, val2),  = cond
        result = {
                '_type': self.COND_TYPES_F[op_id],
                'val_unit': self.parse_val_unit(val_unit),
                'val1': self.parse_val(val1),
        }
        if op_id == 1:  # between
            result['val2'] = self.parse_val(val2)
        if not_op:
            result = {
                    '_type': 'Not',
                    'c': result,
            }
        return result

    def parse_sql(self, sql, optional=False):
        if optional and sql is None:
            return None
        return filter_nones({
                '_type': 'sql',
                'select': self.parse_select(sql['select']),
                'where': self.parse_cond(sql['where'], optional=True),
                'group_by': [self.parse_col_unit(u) for u in sql['groupBy']],
                'order_by': self.parse_order_by(sql['orderBy']),
                'having': self.parse_cond(sql['having'], optional=True),
                'limit': sql['limit'],
                'intersect': self.parse_sql(sql['intersect'], optional=True),
                'except': self.parse_sql(sql['except'], optional=True),
                'union': self.parse_sql(sql['union'], optional=True),
                **({
                    'from': self.parse_from(sql['from']),
                } if self.output_from else {})
        })

    def parse_select(self, select):
        is_distinct, aggs = select
        return {
                '_type': 'select',
                'is_distinct': is_distinct,
                'aggs': [self.parse_agg(agg) for agg in aggs],
        }

    def parse_agg(self, agg):
        agg_id, val_unit = agg
        return {
                '_type': 'agg',
                'agg_id': {'_type': self.AGG_TYPES_F[agg_id]},
                'val_unit': self.parse_val_unit(val_unit),
        }

    def parse_from(self, from_):
        return filter_nones({
                '_type': 'from',
                'table_units': [
                    self.parse_table_unit(u) for u in from_['table_units']],
                'conds': self.parse_cond(from_['conds'], optional=True),
        })

    def parse_order_by(self, order_by):
        if not order_by:
            return None

        order, val_units = order_by
        return {
                '_type': 'order_by',
                'order': {'_type': self.ORDERS_F[order]},
                'val_units': [self.parse_val_unit(v) for v in val_units]
        }

    #
    #
    #

    COND_TYPES_F, COND_TYPES_B = bimap(
        #('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists'),
        #(None, 'Between', 'Eq', 'Gt', 'Lt', 'Ge', 'Le', 'Ne', 'In', 'Like', 'Is', 'Exists'))
        range(1, 10),
        ('Between', 'Eq', 'Gt', 'Lt', 'Ge', 'Le', 'Ne', 'In', 'Like'))

    UNIT_TYPES_F, UNIT_TYPES_B = bimap(
        #('none', '-', '+', '*', '/'),
        range(5),
        ('Column', 'Minus', 'Plus', 'Times', 'Divide'))

    AGG_TYPES_F, AGG_TYPES_B = bimap(
        range(6),
        ('NoneAggOp', 'Max', 'Min', 'Count', 'Sum', 'Avg'))

    ORDERS_F, ORDERS_B = bimap(
        ('asc', 'desc'),
        ('Asc', 'Desc'))

    LOGIC_OPERATORS_F, LOGIC_OPERATORS_B = bimap(
            ('and', 'or'),
            ('And', 'Or'))

@attr.s
class SpiderUnparser:
    ast_wrapper = attr.ib()
    schema = attr.ib()

    UNIT_TYPES_B = {
        'Minus': '-',
        'Plus': '+',
        'Times': '*',
        'Divide': '/',
    }
    COND_TYPES_B = {
        'Between': 'BETWEEN',
        'Eq': '=',
        'Gt': '>',
        'Lt': '<',
        'Ge': '>=',
        'Le': '<=',
        'Ne': '!=', 
        'In': 'IN',
        'Like': 'LIKE'
    }

    @classmethod
    def conjoin_conds(cls, conds):
        if not conds:
            return None
        if len(conds) == 1:
            return conds[0]
        return {'_type': 'And', 'left': conds[0], 'right': cls.conjoin_conds(conds[1:])}
    
    @classmethod
    def linearize_cond(cls, cond):
        if cond['_type'] in ('And', 'Or'):
            conds, keywords = cls.linearize_cond(cond['right'])
            return [cond['left']] + conds, [cond['_type']] + keywords
        else:
            return [cond], []

    def unparse_val(self, val):
        if val['_type'] == 'String':
            return val['s']
        if val['_type'] == 'ColUnit':
            return self.unparse_col_unit(val['c'])
        if val['_type'] == 'Number':
            return str(val['f'])
        if val['_type'] == 'ValSql':
            return '({})'.format(self.unparse_sql(val['s']))

    def unparse_col_unit(self, col_unit):
        column = self.schema.columns[col_unit['col_id']]
        if column.table is None:
            column_name = column.orig_name
        else:
            column_name = '{}.{}'.format(column.table.orig_name, column.orig_name)

        if col_unit['is_distinct']:
            column_name = 'DISTINCT {}'.format(column_name)
        agg_type = col_unit['agg_id']['_type']
        if agg_type == 'NoneAggOp':
            return column_name
        else:
            return '{}({})'.format(agg_type, column_name)

    def unparse_val_unit(self, val_unit):
        if val_unit['_type'] == 'Column':
            return self.unparse_col_unit(val_unit['col_unit1'])
        col1 = self.unparse_col_unit(val_unit['col_unit1'])
        col2 = self.unparse_col_unit(val_unit['col_unit2'])
        return '{} {} {}'.format(col1, self.UNIT_TYPES_B[val_unit['_type']], col2)

    #def unparse_table_unit(self, table_unit):
    #    raise NotImplementedError

    def unparse_cond(self, cond, negated=False):
        if cond['_type'] == 'And':
            assert not negated
            return '{} AND {}'.format(
                self.unparse_cond(cond['left']), self.unparse_cond(cond['right']))
        elif cond['_type'] == 'Or':
            assert not negated
            return '{} OR {}'.format(
                self.unparse_cond(cond['left']), self.unparse_cond(cond['right']))
        elif cond['_type'] == 'Not':
            return self.unparse_cond(cond['c'], negated=True)
        elif cond['_type'] == 'Between':
            tokens = [self.unparse_val_unit(cond['val_unit'])]
            if negated:
                tokens.append('NOT')
            tokens += [
                'BETWEEN',
                self.unparse_val(cond['val1']),
                'AND',
                self.unparse_val(cond['val2']),
            ]
            return ' '.join(tokens)
        tokens = [self.unparse_val_unit(cond['val_unit'])]
        if negated:
            tokens.append('NOT')
        tokens += [self.COND_TYPES_B[cond['_type']], self.unparse_val(cond['val1'])]
        return ' '.join(tokens)

    def unparse_sql(self, tree):
        # First, fix 'from'
        if 'from' not in tree:
            tree = copy.deepcopy(tree)

            # Get all candidate columns
            candidate_column_ids = set(self.ast_wrapper.find_all_descendants_of_type(
                tree, 'column', lambda field: field.type != 'sql'))
            candidate_columns = [self.schema.columns[i] for i in candidate_column_ids]
            all_from_table_ids = set(
                column.table.id for column in candidate_columns if column.table is not None)
            if not all_from_table_ids:
                # Copied from SyntaxSQLNet
                all_from_table_ids = {0}

            covered_tables = set()
            candidate_table_ids = sorted(all_from_table_ids)
            start_table_id = candidate_table_ids[0]
            conds = []
            for table_id in candidate_table_ids[1:]:
                if table_id in covered_tables:
                    continue
                try:
                    path = nx.shortest_path(
                        self.schema.foreign_key_graph, source=start_table_id, target=table_id)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    covered_tables.add(table_id)
                    continue
                
                for source_table_id, target_table_id in zip(path, path[1:]):
                    if target_table_id in covered_tables:
                        continue
                    all_from_table_ids.add(target_table_id)
                    col1, col2 = self.schema.foreign_key_graph[source_table_id][target_table_id]['columns']
                    conds.append({
                        '_type': 'Eq',
                        'val_unit': {
                            '_type': 'Column',
                            'col_unit1': {
                                '_type': 'col_unit',
                                'agg_id': {'_type': 'NoneAggOp'},
                                'col_id': col1,
                                'is_distinct': False
                            },
                        }, 
                        'val1': {
                            '_type': 'ColUnit',
                            'c': {
                                '_type': 'col_unit',
                                'agg_id': {'_type': 'NoneAggOp'},
                                'col_id': col2,
                                'is_distinct': False
                            }
                        }
                    })
            table_units = [{'_type': 'Table', 'table_id': i} for i in sorted(all_from_table_ids)]

            tree['from'] = {
                '_type': 'from',
                'table_units': table_units,
            }
            cond_node = self.conjoin_conds(conds)
            if cond_node is not None:
                tree['from']['conds'] = cond_node
        result = [
            # select select,
            self.unparse_select(tree['select']),
            # from from,
            self.unparse_from(tree['from']),
        ]
        # cond? where,
        if 'where' in tree:
            result += [
                'WHERE',
                self.unparse_cond(tree['where'])
            ]
        # col_unit* group_by,
        if 'group_by' in tree:
            result += [
                'GROUP BY',
                ', '.join(self.unparse_col_unit(c) for c in tree['group_by'])
            ]
        # cond? having,
        if 'having' in tree:
            result += ['HAVING', self.unparse_cond(tree['having'])]
        # order_by? order_by,
        if 'order_by' in tree:
            result.append(self.unparse_order_by(tree['order_by']))
        # int? limit, 
        if 'limit' in tree:
            result += ['LIMIT', str(tree['limit'])]

        # sql? intersect,
        if 'intersect' in tree:
            result += ['INTERSECT', self.unparse_sql(tree['intersect'])]
        # sql? except,
        if 'except' in tree:
            result += ['EXCEPT', self.unparse_sql(tree['except'])]
        # sql? union
        if 'union' in tree:
            result += ['UNION', self.unparse_sql(tree['union'])]
        
        return ' '.join(result)

    def unparse_select(self, select):
        tokens = ['SELECT']
        if select['is_distinct']:
            tokens.append('DISTINCT')
        tokens.append(', '.join(self.unparse_agg(agg) for agg in select.get('aggs', [])))
        return ' '.join(tokens)

    def unparse_agg(self, agg):
        unparsed_val_unit = self.unparse_val_unit(agg['val_unit'])
        agg_type = agg['agg_id']['_type']
        if agg_type == 'NoneAggOp':
            return unparsed_val_unit
        else:
            return '{}({})'.format(agg_type, unparsed_val_unit)

    def unparse_from(self, from_):
        if 'conds' in from_:
            all_conds, keywords = self.linearize_cond(from_['conds'])
        else:
            all_conds, keywords = [], []
        assert all(keyword == 'And' for keyword in keywords)

        cond_indices_by_table = collections.defaultdict(set)
        tables_involved_by_cond_idx = collections.defaultdict(set)
        for i, cond in enumerate(all_conds):
            for column in self.ast_wrapper.find_all_descendants_of_type(cond, 'column'):
                table = self.schema.columns[column].table
                if table is None:
                    continue
                cond_indices_by_table[table.id].add(i)
                tables_involved_by_cond_idx[i].add(table.id)

        output_table_ids = set()
        output_cond_indices = set()
        tokens = ['FROM']
        for i, table_unit in enumerate(from_.get('table_units', [])):
            if i > 0:
                tokens += ['JOIN']

            if table_unit['_type'] == 'TableUnitSql':
                tokens.append('({})'.format(self.unparse_sql(table_unit['s'])))
            elif table_unit['_type'] == 'Table':
                table_id = table_unit['table_id']
                tokens += [self.schema.tables[table_id].orig_name]
                output_table_ids.add(table_id)

                # Output "ON <cond>" if all tables involved in the condition have been output
                conds_to_output = []
                for cond_idx in sorted(cond_indices_by_table[table_id]):
                    if cond_idx in output_cond_indices:
                        continue
                    if tables_involved_by_cond_idx[cond_idx] <= output_table_ids:
                        conds_to_output.append(all_conds[cond_idx])
                        output_cond_indices.add(cond_idx)
                if conds_to_output:
                    tokens += ['ON']
                    tokens += list(intersperse(
                        'AND', 
                        (self.unparse_cond(cond) for cond in conds_to_output)))
        return ' '.join(tokens)

    def unparse_order_by(self, order_by):
        return 'ORDER BY {} {}'.format(
            ', '.join(self.unparse_val_unit(v) for v in order_by['val_units']),
            order_by['order']['_type'])
