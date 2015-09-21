__author__ = 'frodre'

# Temporary import until it's turned into a package
import sys
sys.path.append('../')

import pytest
import cPickle
from copy import deepcopy
import LMR_psms as psms
import LMR_proxy2 as proxy2
import test_config


@pytest.fixture()
def psm_dat(request):

    fname = ('/home/chaos2/wperkins/data/LMR/PSM/'
             'PSMs_GISTEMP.pckl')
    f = open(fname)
    dat = cPickle.load(f)
    f.close()

    return dat

# All tests using Aus_01 unless specified otherwise
def dummy_proxy(pid='Aus_01', ptype='Tree ring_Width'):
    class Proxy():
        pass

    p = Proxy()
    p.id = pid
    p.type = ptype

    return p


@pytest.mark.xfail
def test_abstract_class_creation():
    x = psms.BasePSM()


def test_linear_with_psm_data(psm_dat):
    proxy = dummy_proxy()

    lpsm = psms.LinearPSM(test_config, dummy_proxy(), psm_data=psm_dat)
    site_dat = psm_dat[(proxy.type, proxy.id)]
    assert lpsm.lat == site_dat['lat']
    assert lpsm.lon == site_dat['lon']
    assert lpsm.corr == site_dat['PSMcorrel']
    assert lpsm.slope == site_dat['PSMslope']
    assert lpsm.intercept == site_dat['PSMintercept']
    assert lpsm.R == site_dat['PSMmse']
    assert hasattr(lpsm, 'psm')


def test_linear_load_from_config(psm_dat):
    proxy = dummy_proxy()

    lpsm = psms.LinearPSM(test_config, dummy_proxy())

    site_dat = psm_dat[(proxy.type, proxy.id)]
    assert lpsm.lat == site_dat['lat']
    assert lpsm.lon == site_dat['lon']
    assert lpsm.corr == site_dat['PSMcorrel']
    assert lpsm.slope == site_dat['PSMslope']
    assert lpsm.intercept == site_dat['PSMintercept']
    assert lpsm.R == site_dat['PSMmse']
    assert hasattr(lpsm, 'psm')


def test_linear_calibrate_proxy():
    pass


def test_linear_calibrate_psm_data_no_key_match():
    pass


# TODO: Figure out why it's not calibrating exactly the same
@pytest.mark.xfail
def test_linear_calibrate_psm_data_config_file_not_found(psm_dat):
    cfg_cpy = deepcopy(test_config)
    cfg_cpy.psm.linear.pre_calib_datafile = 'not_found_lol'
    site = proxy2.ProxyPages.load_site(cfg_cpy, 'Aus_02', [1850, 2000])
    site_psm = site.psm_obj

    site_dat = psm_dat[(site.type, site.id)]
    try:
        assert site.lat == site_dat['lat']
        assert site.lon == site_dat['lon']
        assert site_psm.corr == site_dat['PSMcorrel']
        assert site_psm.slope == site_dat['PSMslope']
        assert site_psm.intercept == site_dat['PSMintercept']
        assert site_psm.R == site_dat['PSMmse']
    finally:
        reload(test_config)



def test_linear_corr_below_rcrit(psm_dat):
    proxy = dummy_proxy(pid='SAm_19', ptype='Tree ring_Width')
    with pytest.raises(ValueError):
        psms.LinearPSM(test_config, proxy, psm_data=psm_dat)


def test_linear_psm_ye_val():
    proxy = dummy_proxy()
    pass


def test_linear_get_kwargs(psm_dat):
    reload(test_config)
    psm_kwargs = psms.LinearPSM.get_kwargs(test_config)

    assert 'psm_data' in psm_kwargs.keys()
    assert psm_kwargs['psm_data'] == psm_dat


def test_get_psm_class():
    assert type(psms.get_psm_class('linear')) is type(psms.LinearPSM)

if __name__ == '__main__':
    test_linear_calibrate_psm_data_config_file_not_found(psm_dat(None))
