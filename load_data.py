__author__ = 'frodre'

"""
General data loading functions
"""

import pandas
import pickle

def query_dataframe(filters):
    pass

def grab_record_dataframe(columns):
    pass

def load_data_frame(data_src):
    return pandas.read_pickle(data_src)

def load_cpickle(file):

    with open(file, 'rb') as f:
        return pickle.load(f)
