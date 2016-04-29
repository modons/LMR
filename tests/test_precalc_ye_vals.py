import pytest
import numpy as np
import sys

sys.path.append('../.')

import LMR_proxy_pandas_rework
import LMR_prior
import LMR_config
import LMR_utils

LMR_config.core.seed = 0

@pytest.fixture(scope='module')
def annual_prior():
    cfg = LMR_config.Config()
    X = LMR_prior.prior_assignment(cfg.prior.prior_source)
    X.prior_datadir = cfg.prior.datadir_prior
    X.prior_datafile = cfg.prior.datafile_prior
    X.statevars = cfg.prior.psm_required_variables

    X.read_prior()
    return X

@pytest.fixture(scope='module')
def sampled_annual_prior():
    cfg = LMR_config.Config()
    X = LMR_prior.prior_assignment(cfg.prior.prior_source)
    X.prior_datadir = cfg.prior.datadir_prior
    X.prior_datafile = cfg.prior.datafile_prior
    X.statevars = cfg.prior.psm_required_variables
    X.Nens = cfg.core.nens

    X.populate_ensemble(cfg.prior.prior_source, cfg.prior)
    return X

@pytest.fixture(scope='module')
def proxy_manager():
    cfg = LMR_config.Config()
    pmanager = LMR_proxy_pandas_rework.ProxyManager(cfg, [0, 2000])

    return pmanager


def test_load_precalc_file_not_exist(proxy_manager,
                                     sampled_annual_prior):
    cfg = LMR_config.Config()
    cfg.prior.prior_source = 'not_a_prior_source'

    with pytest.raises(IOError):
        LMR_utils.load_precalculated_ye_vals(cfg,
                                             proxy_manager,
                                             sampled_annual_prior.prior_sample_indices)


def test_written_file_against_realtime_ye_calc(proxy_manager,
                                               sampled_annual_prior):
    cfg = LMR_config.Config()

    loaded_ye = LMR_utils.load_precalculated_ye_vals(cfg,
                                                     proxy_manager,
                                                     sampled_annual_prior.prior_sample_indices)

    for i, proxy in enumerate(proxy_manager.sites_assim_proxy_objs()):
        calc_ye = proxy.psm(sampled_annual_prior.ens,
                            sampled_annual_prior.full_state_info,
                            sampled_annual_prior.coords)
        np.testing.assert_equal(loaded_ye[i], calc_ye)

def test_build_ye_from_annual(annual_prior, proxy_manager, sampled_annual_prior):

    annual_data = annual_prior.prior_dict['tas_sfc_Amon']['value']
    pobjs = [pobj for pobj in proxy_manager.sites_assim_proxy_objs()]

    ye_out = np.zeros((len(pobjs), annual_data.shape[0]))
    lon = annual_prior.prior_dict['tas_sfc_Amon']['lon']
    lat = annual_prior.prior_dict['tas_sfc_Amon']['lat']

    for i, pobj in enumerate(pobjs):
        tmp_dat = pobj.psm_obj.get_close_grid_point_data(annual_data,
                                                         lon,
                                                         lat)
        basic_ye = pobj.psm_obj.basic_psm(tmp_dat)
        sampled_basic_ye = basic_ye[sampled_annual_prior.prior_sample_indices]
        calc_ye = pobj.psm(sampled_annual_prior.ens,
                           sampled_annual_prior.full_state_info,
                           sampled_annual_prior.coords)
        np.testing.assert_equal(sampled_basic_ye, calc_ye)


if __name__ == 'main':

    X = sampled_annual_prior()
    pman = proxy_manager()

    test_written_file_against_realtime_ye_calc(pman, X)




