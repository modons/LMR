__author__ = 'frodre'

"""
General data loading functions
"""

import pandas
import pickle
from functools import lru_cache

def query_dataframe(filters):
    pass

def grab_record_dataframe(columns):
    pass

def load_data_frame(data_src):
    return pandas.read_pickle(data_src)

@lru_cache(maxsize=64)
def load_cpickle(file):

    with open(file, 'rb') as f:
        return pickle.load(f)
