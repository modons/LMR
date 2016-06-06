import sys

sys.path.append('../')

import LMR_config as cfg
import LMR_prior
import numpy as np
import pytest


def test_prior_seed():
    cfg_obj = cfg.Config()
    prior_cfg = cfg_obj.prior
    prior_cfg.seed = 2
    prior_source = '20cr'
    datadir_prior = 'data'
    datafile_prior = '[vardef_template]_gridded_dat.nc'
    state_variables = ['air']

    X = LMR_prior.prior_assignment(prior_source)

    X.prior_datadir = datadir_prior
    X.prior_datafile = datafile_prior
    X.statevars = state_variables
    X.Nens = 1
    X.detrend = False

    X.populate_ensemble(prior_source, prior_cfg)

    X2 = LMR_prior.prior_assignment(prior_source)

    X2.prior_datadir = datadir_prior
    X2.prior_datafile = datafile_prior
    X2.statevars = state_variables
    X2.Nens = 1
    X2.detrend = False

    X2.populate_ensemble(prior_source, prior_cfg)

    np.testing.assert_equal(X2.ens, X.ens)