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

print ('Loading prior data from {}'.format(cfg.prior.prior_source))
X = LMR_prior.prior_assignment(cfg.prior.prior_source)
X.prior_datadir = cfg.prior.datadir_prior
X.prior_datafile = cfg.prior.datafile_prior
X.statevars = cfg.prior.psm_required_variables

X.read_prior()
annual_data = X.prior_dict['tas_sfc_Amon']['value']

cfg.psm.linear.psm_r_crit = 0.0
print 'Loading proxies...'
psm_kwargs = LMR_psms.LinearPSM.get_kwargs(cfg)
pages_pids, pages_pobjs =\
    LMR_proxy_pandas_rework.ProxyPages.load_all(cfg,
                                                [0, 2000],
                                                **psm_kwargs)

print pages_pids

ye_out = np.zeros((len(pages_pobjs), annual_data.shape[0]))
lon = X.prior_dict['tas_sfc_Amon']['lon']
lat = X.prior_dict['tas_sfc_Amon']['lat']

print ('Calculating ye values for {:d} proxies.'.format(len(pages_pobjs)))
for i, pobj in enumerate(pages_pobjs):
    tmp_dat = pobj.psm_obj.get_close_grid_point_data(annual_data,
                                                     lon,
                                                     lat)
    ye_out[i] = pobj.psm_obj.basic_psm(tmp_dat)

pid_map = {pobj.id: idx
           for pobj, idx in izip(pages_pobjs, xrange(len(pages_pobjs)))}
out_fname = '{}_{}_{}.npz'.format(cfg.prior.prior_source,
                                  cfg.psm.linear.datatag_calib,
                                  'tas_sfc_Amon')

precalc_ye_dir = 'precalc_ye_files'
out_dir = os.path.join(cfg.prior.datadir_prior, precalc_ye_dir)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out_full = os.path.join(out_dir, out_fname)
print 'Writing precalculated ye file: {}'.format(out_full)
np.savez(out_full,
         pid_index_map=pid_map,
         ye_vals=ye_out)







