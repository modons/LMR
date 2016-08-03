import sys
import numpy as np
import os
from itertools import izip

import timeit

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
X.Nens = None  # None => Load entire prior


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

if psm_key == 'linear':
    if cfg.psm.linear.avgPeriod == 'annual':
        X.avgInterval = cfg.psm.linear.avgPeriod
        psm_avg = 'annual'
    elif cfg.psm.linear.avgPeriod == 'season':
        X.avgInterval = 'monthly'
        psm_avg = 'season'
    else:
        print 'ERROR in specification of averaging period.'
        raise SystemExit()        
elif psm_key == 'linear_TorP':
    if cfg.psm.linear_TorP.avgPeriod == 'annual':
        X.avgInterval = cfg.psm.linear_TorP.avgPeriod
        psm_avg = 'annual'
    elif cfg.psm.linear_TorP.avgPeriod == 'season':
        X.avgInterval = 'monthly'
        psm_avg = 'season'
    else:
        print 'ERROR in specification of averaging period.'
        raise SystemExit()
elif psm_key == 'bilinear':
    if cfg.psm.bilinear.avgPeriod == 'annual':
        X.avgInterval = cfg.psm.bilinear.avgPeriod
        psm_avg = 'annual'
    elif cfg.psm.linear_TorP.avgPeriod == 'season':
        X.avgInterval = 'monthly'
        psm_avg = 'season'
    else:
        print 'ERROR in specification of averaging period.'
        raise SystemExit()
else:
    X.avgInterval = 'annual'


masterstarttime = timeit.default_timer()

# Load the prior data
X.populate_ensemble(cfg.prior.prior_source, cfg.prior)

statedim = X.ens.shape[0]
ntottime = X.ens.shape[1]

dates = X.prior_dict[X.statevars[0]]['years']
# How many years in prior data?
years = list(set([d.year for d in X.prior_dict[X.statevars[0]]['years']]))
len_prior_dat = len(years)


# Calculate the Ye values
num_proxy = len(proxy_objects)
print 'Calculating ye values for {:d} proxies'.format(num_proxy)
ye_out = np.zeros((num_proxy, len_prior_dat))

if psm_avg == 'season':
    # Ye calculation on the basis of seasonally-averaged prior data

    # map out all possible seasonality vectors that will have to be considered
    season_vects = []
    for pobj in proxy_objects: season_vects.append(pobj.seasonality)
    season_unique = list(set(map(tuple, season_vects)))

    # Loop over seasonality definitions found in the proxy set
    for season in season_unique:

        print 'Processing proxies with seasonality metadata:', season

        nbmonths = len(season)
        year_current = [m for m in season if m>0 and m<=12]
        year_before  = [abs(m) for m in season if m < 0]        
        year_follow  = [m-12 for m in season if m > 12]

        # Identify prior array indices corresponding to months in proxy seasonality
        maskind = np.zeros((nbmonths,len_prior_dat),dtype='int64')
        # define for year 1
        yr1 = years[1]
        tindsyr   = [k for k,d in enumerate(dates) if d.year == yr1    and d.month in year_current]
        tindsyrm1 = [k for k,d in enumerate(dates) if d.year == yr1-1. and d.month in year_before]
        tindsyrp1 = [k for k,d in enumerate(dates) if d.year == yr1+1. and d.month in year_follow]
        indsyr1 = tindsyrm1+tindsyr+tindsyrp1
        # repeat pattern for other years over entire time dimension
        maskind[:,0] = [x-12 for x in indsyr1]
        for kyr in range(1,len_prior_dat):
            cyr = kyr-1
            maskind[:,kyr] = [x+(cyr*12) for x in indsyr1]
        maskind[maskind < 0] = 0
        maskind[maskind > ntottime-1] = 0
        
        # Extract the slice of prior data corresponding to proxy seasonality
        prior_seasonSlice = np.take(X.ens, maskind, axis=1)
        # mean over appropriate season
        prior_state = np.mean(prior_seasonSlice, axis=1)
        
        # loop over proxies
        for i, pobj in enumerate(proxy_objects):
            # restrict to proxy records with seasonality corresponding to
            # current "season" loop variable
            if pobj.seasonality == list(season):
                print '{:10d} (...of {:d})'.format(i, num_proxy), pobj.id
                ye_out[i] = pobj.psm(prior_state, X.full_state_info, X.coords)

else:
    # Ye calculation on the basis of annual (calendar) prior data
    for i, pobj in enumerate(proxy_objects):
        print '{:10d} (...of {:d})'.format(i, num_proxy), pobj.id
        ye_out[i] = pobj.psm(X.ens, X.full_state_info, X.coords)


elapsedtot = timeit.default_timer() - masterstarttime
print 'Total elapsed time:', elapsedtot


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
