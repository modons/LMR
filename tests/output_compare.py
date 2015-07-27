__author__ = 'wperkins'
"""
Compares LMR output between two archive directories.
"""

import glob
import cPickle
import numpy as np
import os.path as path

from itertools import izip

# proxy-first version vs time-first version comparison
off_v_on = True

#dir1 = '/home/disk/ekman/rtardif/kalman3/LMR/output/validationWithAndre/r0/'
dir1 = '/home/chaos2/wperkins/data/LMR/output/archive/proxyfirstloop_10yr_100members_1itr_100pct/r0/'
dir2 = '/home/chaos2/wperkins/data/LMR/output/archive/timefirst_10yr_100members_1itr_100pct/r0/'

d1_npfiles = glob.glob(dir1 + '*.npz')
d1_pckl_files = glob.glob(dir1 + '*.pckl')
d1_files = d1_npfiles + d1_pckl_files
d1_names = [path.split(fpath)[1] for fpath in d1_files]

d2_npfiles = glob.glob(dir2 + '*.npz')
d2_pckl_files = glob.glob(dir2 + '*.pckl')
d2_files = d2_npfiles + d2_pckl_files
d2_names = [path.split(fpath)[1] for fpath in d2_files]

# Match assimilated proxies
assim_fname = 'assimilated_proxies.npy'

assim1 = np.load(dir1 + assim_fname)
assim2 = np.load(dir2 + assim_fname)

assim2_idx_match_to1 = []
for proxy2 in assim2:
    id2 = proxy2[proxy2.keys()[0]][0]
    for i, proxy in enumerate(assim1):
        id1 = proxy[proxy.keys()[0]][0]
        if id1 == id2:
            assim2_idx_match_to1.append(i)
            break

# for i, idx in enumerate(assim2_idx_match_to1):
#     print assim1[idx].keys()
#     print assim2[i].keys()

nproxies = len(assim2)
assert len(assim2) == len(assim1)

#Test that they created the same files.
for name in d1_names:
    assert name in d2_names

# Test equivalence of file contents
for idx, file in enumerate(d1_files):

    idx2 = d2_names.index(d1_names[idx])
    print d1_names[idx]

    if path.splitext(file)[1] == '.npz':
        f1 = np.load(file)
        f2 = np.load(d2_files[idx2])

        for key, values in f1.iteritems():
            values2 = f2[key]

            print '\t', key, ': ', values.dtype

            if values.dtype.kind == 'S':
                for i1, i2 in izip(values, values2):
                    assert i1 == i2
            elif not values.dtype == np.object:
                if off_v_on:
                    if key == 'Xb_one_aug':
                        # Need to re arrange proxie order
                        p1 = values[:-nproxies]
                        p2 = values2[:-nproxies]
                        np.testing.assert_allclose(p1, p2)

                        ye1 = values[-nproxies:]
                        ye1_rearr = ye1[assim2_idx_match_to1]
                        ye2 = values2[-nproxies:]
                        np.testing.assert_allclose(ye1_rearr, ye2)
                    elif key == 'gmt_save':
                        # Proxies assimilated in different order check ending
                        np.testing.assert_allclose(values[-1], values2[-1],
                                                   rtol=1e-4,
                                                   atol=1e-4)
                elif len(values.shape) > 1:
                    np.testing.assert_allclose(values.mean(axis=1),
                                               values2.mean(axis=1),
                                               rtol=1e-4,
                                               atol=1e-4)
                else:
                    np.testing.assert_allclose(values, values2, rtol=1e-4)
            else:
                assert values == values2

    elif path.splitext(file)[1] == '.pckl':
        with open(file, 'r') as load_file:
            f1 = cPickle.load(load_file)
        with open(d2_files[idx2], 'r') as load_file:
            f2 = cPickle.load(load_file)

        for key, dict in f1.iteritems():
            dict2 = f2[key]

            for sub_key, value in dict.iteritems():
                value2 = dict2[sub_key]
                if off_v_on and sub_key == 'HXa':
                    # Proxy order is different between the two right now
                    continue
                elif type(value) == np.ndarray:
                    np.testing.assert_allclose(value, value2)
                else:
                    assert value == value2
    else:
        continue
