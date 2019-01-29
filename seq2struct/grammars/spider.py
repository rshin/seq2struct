import os

import asdl

from seq2struct import ast_util
from seq2struct.utils import registry


def bimap(first, second):
    return {f: s for f, s in  zip(first, second)}, {s: f for f, s in zip(first, second)}


def filter_nones(d):
    return {k: v for k, v in d.items() if v is not None and v != []}


@registry.register('grammar', 'spider')
class SpiderLanguage:

    root_type = 'sql'

    def __init__(self):
        self.ast_wrapper = ast_util.ASTWrapper(
                asdl.parse(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        'Spider.asdl')),
                custom_primitive_type_checkers = {
                    'column': lambda x: isinstance(x, int)
                })

        self.pointers = {'column'}

    def parse(self, code, section):
        return self.parse_sql(code)

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
                #'from': self.parse_from(sql['from']),
                'where': self.parse_cond(sql['where'], optional=True),
                'group_by': [self.parse_col_unit(u) for u in sql['groupBy']],
                'order_by': self.parse_order_by(sql['orderBy']),
                'having': self.parse_cond(sql['having'], optional=True),
                'limit': sql['limit'],
                'intersect': self.parse_sql(sql['intersect'], optional=True),
                'except': self.parse_sql(sql['except'], optional=True),
                'union': self.parse_sql(sql['union'], optional=True),
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
