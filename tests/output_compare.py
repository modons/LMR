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
is_proxy_first_driver_loop = False

# Use this if the two reconstructions use different state variables
# Also use the
diff_state_vectors = True

# All close testing tolerances (checks for diff > atol + rtol*desired_value)
rtol = 1e-4
atol = 1e-6

#base_dir = '/home/chaos2/wperkins/data/LMR/output/testing'
base_dir = '/home/katabatic2/wperkins/LMR_output/testing'
# sim_dir1 = 'testdev_production_comparison_1900_1960_seed0_nens10/r0/'
# sim_dir2 = 'testdev_ncdc_add_comparison_seed0_1900_1960_10nens/r0/'
#sim_dir1 = 'testdev_precalcye_pages_linearTorP_no_tas/r0/'
#sim_dir2 = 'testdev_production_pages_linearTorP_comparison/r0/'
# --
#sim_dir1 = 'test_ncdc_bilinear_precalc_tp/r0'
#sim_dir1 = 'test_ncdc_bilinear_precalc_t/r0'
# sim_dir1 = 'test_ncdc_bilinear_precalc_p/r0'
# sim_dir2 = 'test_ncdc_bilinear_noprecalc_tp/r0'

sim_dir1 = 'testdev_yaml_config_update/r0'
sim_dir2 = 'reference_pages/r0'


dir1 = path.join(base_dir, sim_dir1)
dir2 = path.join(base_dir, sim_dir2)

d1_npfiles = glob.glob(path.join(dir1, '*.npz'))
d1_pckl_files = glob.glob(path.join(dir1, '*.pckl'))
d1_files = d1_npfiles + d1_pckl_files
d1_names = [path.split(fpath)[1] for fpath in d1_files]

d2_npfiles = glob.glob(path.join(dir2, '*.npz'))
d2_pckl_files = glob.glob(path.join(dir1, '*.pckl'))
d2_files = d2_npfiles + d2_pckl_files
d2_names = [path.split(fpath)[1] for fpath in d2_files]

# Match assimilated proxies
assim_fname = 'assimilated_proxies.npy'

assim1 = np.load(path.join(dir1, assim_fname))
assim2 = np.load(path.join(dir2, assim_fname))

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

#Test that they created the same files. NOTE: if first dir is missing files
# they will not be checked.  Reference (first directory) should be the benchmark
for name in d1_names:
    assert name in d2_names

# Test equivalence of file contents
for idx, file in enumerate(d1_files):

    idx2 = d2_names.index(d1_names[idx])
    print d1_names[idx]

    if diff_state_vectors:
        exclude = ['Xb_one.npz', 'gmt.npz', 'gmt_ensemble.npz']
        if d1_names[idx] in exclude:
            continue

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
                if is_proxy_first_driver_loop:
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
                                                   rtol=rtol,
                                                   atol=atol)
                elif len(values.shape) > 1:
                    np.testing.assert_allclose(values.mean(axis=1),
                                               values2.mean(axis=1),
                                               rtol=rtol,
                                               atol=atol)
                else:
                    np.testing.assert_allclose(values, values2, rtol=rtol,
                                               atol=atol)
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
                if is_proxy_first_driver_loop and sub_key == 'HXa':
                    # Proxy order is different between the two right now
                    continue
                elif type(value) == np.ndarray:
                    np.testing.assert_allclose(value, value2)
                else:
                    assert value == value2
    else:
        continue
