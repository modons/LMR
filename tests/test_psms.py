__author__ = 'frodre'

# Temporary import until it's turned into a package
import sys
sys.path.append('../')

import pytest
import pickle
import yaml
import LMR_psms as psms
import LMR_config
import numpy as np
from LMR_utils import haversine


@pytest.fixture()
def psm_dat(request):

    fname = ('/home/katabatic/wperkins/data/LMR/PSM/'
             'PSMs_GISTEMP.pckl')
    f = open(fname)
    dat = pickle.load(f)
    f.close()

    return dat


@pytest.fixture(scope='module')
def config(request):
    with open('test_config.yml', 'r') as f:
        cfg_update_dict = yaml.load(f)

    LMR_config.update_config_class_yaml(cfg_update_dict, LMR_config)

    def fin():
        reload(LMR_config)

    request.addfinalizer(fin)
    return LMR_config.Config()


# All tests using Aus_01 unless specified otherwise
def dummy_proxy(pid='Aus_01', ptype='Tree ring_Width'):
    class Proxy():
        pass

    p = Proxy()
    p.id = pid
    p.type = ptype
    p.lat = -42
    p.lon = 147
    p.elev = 0

    return p



def test_abstract_class_creation():
    with pytest.raises(TypeError):
        x = psms.BasePSM()


def test_linear_with_psm_data(psm_dat, config):
    proxy = dummy_proxy()

    lpsm = psms.LinearPSM(config, dummy_proxy(), psm_data=psm_dat)
    site_dat = psm_dat[(proxy.type, proxy.id)]
    assert lpsm.lat == site_dat['lat']
    assert lpsm.lon == site_dat['lon']
    np.testing.assert_allclose(lpsm.corr, site_dat['PSMcorrel'])
    np.testing.assert_allclose(lpsm.slope, site_dat['PSMslope'])
    np.testing.assert_allclose(lpsm.intercept, site_dat['PSMintercept'])
    np.testing.assert_allclose(lpsm.R, site_dat['PSMmse'])
    # assert lpsm.corr == site_dat['PSMcorrel']
    # assert lpsm.slope == site_dat['PSMslope']
    # assert lpsm.intercept == site_dat['PSMintercept']
    # assert lpsm.R == site_dat['PSMmse']
    assert hasattr(lpsm, 'psm')


def test_linear_load_from_config(psm_dat, config):
    proxy = dummy_proxy()

    lpsm = psms.LinearPSM(config, dummy_proxy())

    site_dat = psm_dat[(proxy.type, proxy.id)]
    assert lpsm.lat == site_dat['lat']
    assert lpsm.lon == site_dat['lon']
    np.testing.assert_allclose(lpsm.corr, site_dat['PSMcorrel'])
    np.testing.assert_allclose(lpsm.slope, site_dat['PSMslope'])
    np.testing.assert_allclose(lpsm.intercept, site_dat['PSMintercept'])
    np.testing.assert_allclose(lpsm.R, site_dat['PSMmse'])
    assert hasattr(lpsm, 'psm')


def test_linear_calibrate_proxy():
    pass


def test_linear_calibrate_psm_data_no_key_match():
    pass


def test_linear_calibrate_pasm_data_config_file_not_found():
    pass



def test_linear_corr_below_rcrit(psm_dat, config):
    config.psm.linear.psm_r_crit = 0.2
    proxy = dummy_proxy(pid='SAm_19', ptype='Tree ring_Width')
    with pytest.raises(ValueError):
        psms.LinearPSM(config, proxy, psm_data=psm_dat)


def test_linear_psm_ye_val():
    proxy = dummy_proxy()
    pass


def test_linear_get_kwargs(psm_dat, config):
    psm_kwargs = psms.LinearPSM.get_kwargs(config)

    assert 'psm_data' in psm_kwargs.keys()
    assert psm_kwargs['psm_data'] == psm_dat


def test_get_psm_class():
    assert type(psms.get_psm_class('linear')) is type(psms.LinearPSM)


def test_get_close_grid_point_data(psm_dat):
    cfg = LMR_config.Config()
    proxy = dummy_proxy()

    lpsm = psms.LinearPSM(cfg, dummy_proxy(), psm_data=psm_dat)

    lats = np.linspace(-90, 90, 192)
    lons = np.linspace(0, 360, 288, endpoint=False)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_flatgrid = lon_grid.ravel()
    lat_flatgrid = lat_grid.ravel()
    test_dat = np.empty((10, 192, 288))
    test_dat_flat = test_dat.reshape(10, 192*288)

    # Find row index of X for which [X_lat,X_lon] corresponds to closest
    # grid point to
    dist = haversine(proxy.lon, proxy.lat, lon_flatgrid, lat_flatgrid)

    min_flat_idx = dist.argmin()
    ref_lon = lon_flatgrid[min_flat_idx]
    ref_lat = lat_flatgrid[min_flat_idx]

    ref_data = test_dat_flat[:, min_flat_idx]

    # test with lat/lon vectors
    func_data = lpsm.get_close_grid_point_data(test_dat, lons, lats)
    np.testing.assert_equal(ref_data, func_data)

    # test with lat/lon flattened
    func_data = lpsm.get_close_grid_point_data(test_dat, lon_flatgrid,
                                               lat_flatgrid)
    np.testing.assert_equal(ref_data, func_data)

    # test with lat/lon grid
    func_data = lpsm.get_close_grid_point_data(test_dat, lon_grid, lat_grid)
    np.testing.assert_equal(ref_data, func_data)

    # test with flattened data
    func_data = lpsm.get_close_grid_point_data(test_dat_flat, lon_grid,
                                               lat_grid)
    np.testing.assert_equal(ref_data, func_data)


if __name__ == '__main__':
    test_linear_corr_below_rcrit(psm_dat(None))
