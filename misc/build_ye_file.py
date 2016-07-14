import sys
import numpy as np
import os
from itertools import izip

sys.path.append('../')

import LMR_prior
import LMR_psms
import LMR_proxy_pandas_rework
import LMR_config
from LMR_utils import create_precalc_ye_filename

# This script gives a method for pre-calculating ye values for our linear psm.
# It outputs this file in a subdirectory of the prior source directory which
# is then in turn checked for by the driver.  Files created will be unique to
# the prior, psm calibration source, proxy source, and state variable used to
# calculate it. I'm choosing to forego on the fly creation of these files for
# now.  If a user would like to create one for their current configuration  this
#  script should just be an easy one-off run with no editing required.

cfg = LMR_config.Config()

# Create prior source object
print ('Starting Ye precalculation using prior data from '
       '{}'.format(cfg.prior.prior_source))
X = LMR_prior.prior_assignment(cfg.prior.prior_source)
X.prior_datadir = cfg.prior.datadir_prior
X.prior_datafile = cfg.prior.datafile_prior
X.statevars = cfg.prior.psm_required_variables
X.detrend = cfg.prior.detrend
X.kind = cfg.prior.state_kind
X.Nens = None  # Load entire prior

#  Load the proxy data
cfg.psm.linear.psm_r_crit = 0.0
print 'Loading proxies...'
proxy_database = cfg.proxies.use_from[0]
psm_key = cfg.psm.use_psm[proxy_database]
psm_class = LMR_psms.get_psm_class(psm_key)
psm_kwargs = psm_class.get_kwargs(cfg)

proxy_class = LMR_proxy_pandas_rework.get_proxy_class(proxy_database)
proxy_objects = proxy_class.load_all_annual_no_filtering(cfg,
                                                         **psm_kwargs)

# Load the prior data
X.populate_ensemble(cfg.prior.prior_source, cfg.prior)

# Calculate the Ye values
num_proxy = len(proxy_objects)
len_prior_dat = X.prior_dict[X.statevars[0]]['values'].shape[1]
print 'Calculating ye values for {:d} proxies'.format(num_proxy)
ye_out = np.zeros((num_proxy, len_prior_dat))

for i, pobj in enumerate(proxy_objects):
    ye_out[i] = pobj.psm(X.ens, X.full_state_info, X.coords)

# Create a mapping for each proxy id to an index of the array
pid_map = {pobj.id: idx for idx, pobj in enumerate(proxy_objects)}

# Create filename for current experiment
out_dir = os.path.join(cfg.core.lmr_path, 'ye_precalc_files')
out_fname = create_precalc_ye_filename(cfg)

assert len(out_fname) <= 255, 'Filename is too long...'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Write precalculated ye file
out_full = os.path.join(out_dir, out_fname)
print 'Writing precalculated ye file: {}'.format(out_full)
np.savez(out_full,
         pid_index_map=pid_map,
         ye_vals=ye_out)
