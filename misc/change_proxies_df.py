__author__ = 'wperkins'

import pandas as pd
from numpy.testing import assert_array_equal

in_meta = '/home/chaos2/wperkins/data/LMR/data/proxies/Pages2k_Metadata.df.pckl'
out_meta = '/home/chaos2/wperkins/data/LMR/data/proxies/Pages2k_Metadata_0pt5res.df.pckl'

in_data = '/home/chaos2/wperkins/data/LMR/data/proxies/Pages2k_Proxies.df.pckl'
out_data = '/home/chaos2/wperkins/data/LMR/data/proxies/Pages2k_Proxies_0pt5res.df.pckl'

lat_lim = 20
meta = pd.read_pickle(in_meta)
data = pd.read_pickle(in_data)

is_res = meta['Resolution (yr)'] == 1.0
is_tree = (meta['Archive type'] == 'Tree ring') & is_res
is_ice = (meta['Archive type'] == 'Ice core') & is_res

ids = []

ids += meta['PAGES ID'][is_tree].values.tolist()
ids += meta['PAGES ID'][is_ice].values.tolist()

new_meta = meta.copy()
new_data = data.copy()

for i,id in enumerate(ids):
    print 'updating site {}'.format(id)
    site_meta = new_meta[new_meta['PAGES ID'] == id]
    row_idx = site_meta.index.values[0]
    if site_meta['Archive type'].iloc[0] == 'Tree ring':
        if site_meta['Lat (N)'].iloc[0] > 20:
            # measure season in april
            new_meta['Resolution (yr)'][row_idx] = 0.5

        elif site_meta['Lat (N)'].iloc[0] < -20:
            # measure season in october
            new_meta['Resolution (yr)'][row_idx] = 0.5
            new_meta['Youngest (C.E.)'][row_idx] = site_meta['Youngest (C.E.)'].iloc[0] + 0.5
            new_meta['Oldest (C.E.)'][row_idx] = site_meta['Oldest (C.E.)'].iloc[0] + 0.5

            tmp_dat = data[id]
            new_idx = data.index.values + 0.5
            tmp_dat = tmp_dat.reindex(new_idx, method='ffill', limit=1)
            tmp_df = pd.DataFrame(tmp_dat)

            new_data = new_data.drop(id, axis=1)
            new_data = new_data.merge(tmp_df,
                                      how='outer',
                                      left_index=True,
                                      right_index=True)

            assert_array_equal(new_data[id][new_data[id].notnull()].values,
                               data[id][data[id].notnull()].values)

    elif site_meta['Archive type'].iloc[0] == 'Ice core':
        if site_meta['Lat (N)'].iloc[0] < 0:
            # measure season in april
            new_meta['Resolution (yr)'][row_idx] = 0.5

        elif site_meta['Lat (N)'].iloc[0] > 0:
            # measure season in october
            new_meta['Resolution (yr)'][row_idx] = 0.5
            new_meta['Youngest (C.E.)'][row_idx] = site_meta['Youngest (C.E.)'].iloc[0] + 0.5
            new_meta['Oldest (C.E.)'][row_idx] = site_meta['Oldest (C.E.)'].iloc[0] + 0.5

            tmp_dat = data[id]
            new_idx = data.index.values + 0.5
            tmp_dat = tmp_dat.reindex(new_idx, method='ffill', limit=1)
            tmp_df = pd.DataFrame(tmp_dat)

            new_data = new_data.drop(id, axis=1)
            new_data = new_data.merge(tmp_df,
                                      how='outer',
                                      left_index=True,
                                      right_index=True)

            assert_array_equal(new_data[id][new_data[id].notnull()].values,
                               data[id][data[id].notnull()].values)

print 'Total sites updated ', i
pd.to_pickle(new_meta, out_meta)
pd.to_pickle(new_data, out_data)


