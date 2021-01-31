import itertools
import math
import os
import pprint as pp
import sys

import numpy as np
import pandas as pd

MARKET_ORDER = 'Net Payment Paid/Received for Market Orders(USD)'
MARKET_LIMIT_ORDER = 'Net Payment Paid/ Received for Marketable Limit Orders(USD)'
NON_MARKET_LIMIT_ORDER = 'Net Payment Paid/ Received for Non- Marketable Limit Orders(USD)'

SP500 = 'sp500_stock'
NSP = 'non_sp500_stock'
OPTIONS = 'options'

def name_cleanup(name):
    name = name.replace('LLC', '')
    name = name.replace(',', '')
    name = name.replace('Securities', '')
    name = name.replace('Execution Services', '')
    name = name.replace('Americas', '')
    name = name.replace('&', '')
    name = name.replace('Co.', '')
    name = name.replace('LP', '')
    name = name.strip()

    return name

def read_table(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    dataframes = []
    table_lines = []
    for l in lines:
        l = l.strip()
        if l == '====':
            df = pd.DataFrame(table_lines[1:], columns=table_lines[0])
            df.index = df[table_lines[0][0]]
            dataframes.append(df)
            table_lines = []
        else:
            content = l.split('\t')
            table_lines.append([name_cleanup(c) for c in content if len(c.strip()) > 0])

    assert len(dataframes) == 3

    return {SP500: dataframes[0],
            NSP: dataframes[1],
            OPTIONS: dataframes[2],}

# https://stackoverflow.com/questions/320929/currency-formatting-in-python
def money_format(num):
    return ('{:20,.2f}'.format(num)).strip()


def operate(method, digits, numbers):
    if type(method) == str:
        result = np.__dict__[method](numbers)
    elif hasattr(method, '__call__'):
        result = method(numbers)
    else:
        raise Exception(f'Method {method} not supported')

    if type(result) == float:
        return round(result, digits)
    else:
        return result

    return result


def check_tuple_contain(full, partial):
    for i, f in enumerate(partial):
        if partial[i] != None:
            if partial[i] != f:
                return False
        else:
            continue


class DataLoader:
    def __init__(self, data_dict):
        self.all_data = dict()
        
        self.month_set = set()
        self.table_set = set()
        self.venue_set = set()
        self.payment_set = set()

        for month in data_dict:
            for table_type in data_dict[month]:
                for i, row in data_dict[month][table_type].iterrows():
                    for c in [MARKET_ORDER, MARKET_LIMIT_ORDER, NON_MARKET_LIMIT_ORDER]:
                        key = (month, table_type, i, c)
                        value = row[c]
            
                        self.month_set.add(month)
                        self.table_set.add(table_type)
                        self.venue_set.add(i)
                        self.payment_set.add(c)
                        self.all_data[key] = float(value.replace(',', ''))

        self.param_order = ['month', 'table_type', 'venue', 'payment']
        self.param_names = {
            'month' : self.month_set,
            'table_type': self.table_set,
            'venue': self.venue_set,
            'payment': self.payment_set
            }

    def get_data(self, missing="skip", **kwargs):
        arg_set = dict()
        for k in kwargs:
            assert k in self.param_names

            if type(kwargs[k]) != tuple and \
                type(kwargs[k]) != list and \
                type(kwargs[k]) != set:
                arg_set[k] = [kwargs[k]]
            else:
                arg_set[k] = sorted(list(kwargs[k]))
        
        for pn in self.param_names:
            if pn not in arg_set:
                arg_set[pn] = sorted(list(self.param_names[pn]))

        dims = [arg_set[pn] for pn in self.param_order]
        flat_fims = itertools.product(*dims)

        results = []
        for fl in flat_fims:
            if fl not in self.all_data:
                if missing == "skip":
                    continue
                elif missing == "None":
                    v = math.nan
                else:
                    raise KeyError(f'Cannot find key {fl}')
            else:
                v = self.all_data[fl]

            results.append((fl, v))
        
        return results
    
    def get_simple_aggregate(self, method, digits=2, missing="skip", **kwargs):
        data = self.get_data(missing, **kwargs)
        numbers = [d[1] for d in data]

        return operate(method, digits, numbers)
