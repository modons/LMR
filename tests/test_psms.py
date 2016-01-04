__author__ = 'frodre'

# Temporary import until it's turned into a package
import sys
sys.path.append('../')

import pytest
import cPickle
import LMR_psms as psms
import LMR_config as cfg


@pytest.fixture()
def psm_dat(request):

    f = open(cfg.psm.linear.pre_calib_datafile)
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
    p.lat = -42
    p.lon = 147

    return p


@pytest.mark.xfail
def test_abstract_class_creation():
    x = psms.BasePSM()


def test_linear_with_psm_data(psm_dat):
    proxy = dummy_proxy()

    lpsm = psms.LinearPSM(cfg, dummy_proxy(), psm_data=psm_dat)
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

    lpsm = psms.LinearPSM(cfg, dummy_proxy())

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


def test_linear_calibrate_pasm_data_config_file_not_found():
    pass



def test_linear_corr_below_rcrit(psm_dat):
    proxy = dummy_proxy(pid='SAm_19', ptype='Tree ring_Width')
    with pytest.raises(ValueError):
        psms.LinearPSM(cfg, proxy, psm_data=psm_dat)


def test_linear_psm_ye_val():
    proxy = dummy_proxy()
    pass


def test_linear_get_kwargs(psm_dat):
    psm_kwargs = psms.LinearPSM.get_kwargs(cfg)

    assert 'psm_data' in psm_kwargs.keys()
    assert psm_kwargs['psm_data'] == psm_dat


def test_get_psm_class():
    assert type(psms.get_psm_class('linear')) is type(psms.LinearPSM)

if __name__ == '__main__':
    test_linear_corr_below_rcrit(psm_dat(None))
