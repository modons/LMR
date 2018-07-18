__author__ = 'wperkins'
"""
Simple script to change PAGES dataset from xcel to pandas dataframe.  This
makes data easier/snappier to work with in the LMR.

Author: Andre Perkins
"""

import pandas as pd

def pages_xcel_to_dataframes(filename, metaout, dataout):
    """
    Takes in Pages2K CSV and converts it to dataframe storage.  This increases
    size on disk due to the joining along the time index (lots of null values).

    Makes it easier to query and grab data for the proxy experiments.

    :param filename:
    :param metaout:
    :param dataout:
    :return:
    """

    meta_sheet_name = 'Metadata'
    metadata = pd.read_excel(filename, meta_sheet_name)
    # rename 'PAGES ID' column header to more general 'Proxy ID'
    metadata.rename(columns = {'PAGES ID':'Proxy ID'},inplace=True)
    metadata.to_pickle(metaout)

    record_sheet_names = ['AntProxies', 'ArcProxies', 'AsiaProxies',
                          'AusProxies', 'EurProxies', 'NAmPol', 'NAmTR',
                          'SAmProxies']

    for i, sheet in enumerate(record_sheet_names):
        tmp = pd.read_excel(filename, sheet)
        # for key, series in tmp.iteritems():
        #     h5store[key] = series[series.notnull()]

        if i == 0:
            df = tmp
        else:
            # SQL like table join along index
            df = df.merge(tmp, how='outer', on='PAGES 2k ID')

    #fix index and column name
    col0 = df.columns[0]
    newcol0 = df[col0][0]
    df.set_index(col0, drop=True, inplace=True)
    df.index.name = newcol0
    df = df.ix[1:]
    df.sort_index(inplace=True)

    # TODO: make sure year index is consecutive
    #write data to file
    df.to_pickle(dataout)

if __name__ == "__main__":
    work_dir = '/home/chaos2/wperkins/data/LMR/proxies/'
    fname = work_dir + 'Pages2k_DatabaseS1-All-proxy-records.xlsx'
    meta_outfile = work_dir + 'Pages2kv1_Metadata.df.pckl'
    outfile = work_dir + 'Pages2kv1_Proxies.df.pckl'

    pages_xcel_to_dataframes(fname, meta_outfile, outfile)
