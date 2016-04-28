import sys
import numpy as np
import os
from itertools import izip

import LMR_prior
import LMR_psms
import LMR_proxy_pandas_rework
import LMR_config

sys.path.append('../')


cfg = LMR_config.Config()

X = LMR_prior.prior_assignment(cfg.prior.prior_source)
X.prior_datadir = cfg.prior.datadir_prior
X.prior_datafile = cfg.prior.datafile_prior
X.statevars = cfg.prior.psm_required_variables

X.read_prior()
annual_data = X.prior_dict['tas_sfc_Amon']['value']

cfg.psm.linear.psm_r_crit = 0.0

psm_kwargs = LMR_psms.LinearPSM.get_kwargs(cfg)
pages_pids, pages_pobjs =\
    LMR_proxy_pandas_rework.ProxyPages.load_all(cfg,
                                                [0, 2000],
                                                **psm_kwargs)

ye_out = np.zeros((len(pages_pobjs), annual_data.shape[0]))
lon = X.prior_dict['tas_sfc_Amon']['lon']
lat = X.prior_dict['tas_sfc_Amon']['lat']

for i, pobj in enumerate(pages_pobjs):
    tmp_dat = pobj.psm_obj.get_close_grid_point_data(annual_data,
                                                     lon,
                                                     lat)
    ye_out[i] = pobj.psm_obj.basic_psm(tmp_dat)

pid_map = {pid: idx for pid, idx in izip(pages_pids, xrange(len(pages_pids)))}
out_fname = '{}_{}_{}.npz'.format(cfg.prior.prior_source,
                                  cfg.psm.linear.datatag_calib,
                                  'tas_sfc_Amon')

precalc_ye_dir = 'precalc_ye_files'
out_dir = os.path.join(cfg.prior.datadir_prior, precalc_ye_dir)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

np.savez(os.path.join(out_dir, out_fname),
         pid_index_map=pid_map,
         ye_vals=ye_out)







