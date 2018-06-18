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

def load_data_frame(data_src, key=None):
    """Read Pandas dataframe from HDF or pickle file

    Parameters
    ----------
    data_src : path (string), buffer or path object
    key : str, optional
        Group identifier in the HDF store. Can be omitted if the HDF file
        contains a single pandas object or if reading pickle.

    Returns
    -------
    Pandas dataframe.
    """
    try:
        out = pandas.read_hdf(data_src, key=key)
    except OSError:  # Thrown if file is pickle.
        out = pandas.read_pickle(data_src)
    return out

def load_cpickle(file):

    with open(file, 'rb') as f:
        return pickle.load(f)
