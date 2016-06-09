import sys
import numpy as np
import os
from itertools import izip

sys.path.append('../')

import LMR_prior
import LMR_psms
import LMR_proxy_pandas_rework
import LMR_config

# This script gives a method for pre-calculating ye values for our linear psm.
# It outputs this file in a subdirectory of the prior source directory which
# is then in turn checked for by the driver.  Files created will be unique to
# the prior, psm calibration source, and state variable used to calculate it.
# I'm choosing to forego on the fly creation of these files for now.  If a user
# would like to create one for their current configuration (and currently
# hard-coded for the tas_sfc_Amon state variable), this script should just be
# an easy one-off run with no editing required.

cfg = LMR_config.Config()

print ('Starting Ye precalculation using prior data from '
       '{}'.format(cfg.prior.prior_source))
X = LMR_prior.prior_assignment(cfg.prior.prior_source)
X.prior_datadir = cfg.prior.datadir_prior
X.prior_datafile = cfg.prior.datafile_prior
X.statevars = cfg.prior.psm_required_variables
X.detrend = cfg.prior.detrend

cfg.psm.linear.psm_r_crit = 0.0
print 'Loading proxies...'
proxy_database = cfg.proxies.use_from[0]
psm_key = cfg.psm.use_psm[proxy_database]
psm_class = LMR_psms.get_psm_class(psm_key)
psm_kwargs = psm_class.get_kwargs(cfg)

proxy_class = LMR_proxy_pandas_rework.get_proxy_class(proxy_database)
proxy_ids_by_grp, proxy_objects = proxy_class.load_all(cfg,
                                                       [0, 2000],
                                                       **psm_kwargs)

X.read_prior()

for state_var in cfg.prior.psm_required_variables:
    print 'Working on prior variable {}'.format(state_var)
    annual_data = X.prior_dict[state_var]['value']

    ye_out = np.zeros((len(proxy_objects), annual_data.shape[0]))
    lon = X.prior_dict[state_var]['lon']
    lat = X.prior_dict[state_var]['lat']

    print ('Calculating ye values for {:d} proxies.'.format(len(proxy_objects)))
    for i, pobj in enumerate(proxy_objects):
        tmp_dat = pobj.psm_obj.get_close_grid_point_data(annual_data,
                                                         lon,
                                                         lat)
        ye_out[i] = pobj.psm_obj.basic_psm(tmp_dat)

    pid_map = {pobj.id: idx
               for pobj, idx in izip(proxy_objects, xrange(len(proxy_objects)))}
    out_fname = '{}_{}_{}_{}.npz'.format(cfg.prior.prior_source,
                                         cfg.psm.linear.datatag_calib,
                                         proxy_database,
                                         state_var)

    precalc_ye_dir = 'precalc_ye_files'
    out_dir = os.path.join(cfg.prior.datadir_prior, precalc_ye_dir)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_full = os.path.join(out_dir, out_fname)
    print 'Writing precalculated ye file: {}'.format(out_full)
    np.savez(out_full,
             pid_index_map=pid_map,
             ye_vals=ye_out)







