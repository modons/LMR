__author__ = 'frodre'


# Temporary import until it's turned into a package
import sys
sys.path.append('../')

import pytest
import cPickle
import numpy as np
import LMR_proxy2 as proxy2
import LMR_config as cfg

@pytest.fixture()
def psm_dat(request):

    f = open(cfg.psm.linear.pre_calib_datafile)
    dat = cPickle.load(f)
    f.close()

    return dat

@pytest.fixture()
def dummy_proxy(request):
    class proxy:
        pass

    p = proxy()
    p.pid = 'Aus_01'
    p.ptype = 'Tree ring_Width'
    p.start_yr = 1950
    p.end_yr = 1960
    p.values = range(11)
    p.lat = 45
    p.lon = 210
    p.time = range(1950, 1961)

    return p

@pytest.fixture(scope='function')
def meta(request):
    with open('test_meta.pckl', 'rb') as f:
        dat = cPickle.load(f)
    return dat


@pytest.fixture(scope='function')
def pdata(request):
    with open('test_pdata.pckl', 'rb') as f:
        return cPickle.load(f)


@pytest.mark.xfail
def test_abstract_class_creation():
    x = proxy2.BaseProxyObject()


def test_lon_fix():
    assert proxy2.fix_lon(0) == 0
    assert proxy2.fix_lon(-180) == 180
    assert proxy2.fix_lon(-90) == 270


def test_proxies_get_class():
    assert type(proxy2.get_proxy_class('pages')) == type(proxy2.ProxyPages)


def test_pages_init(psm_dat, dummy_proxy):
    p = dummy_proxy
    pclass = proxy2.ProxyPages(cfg, p.pid, p.ptype, p.start_yr, p.end_yr,
                               p.lat, p.lon, p.values, p.time,
                               **{'psm_data': psm_dat})

    assert pclass.id == p.pid
    assert pclass.type == p.ptype
    assert pclass.start_yr == p.start_yr
    assert pclass.end_yr == p.end_yr
    assert pclass.values == p.values
    assert pclass.lat == p.lat
    assert pclass.lon == p.lon
    assert pclass.time == p.time
    assert hasattr(pclass, 'psm')


def test_pages_empty_values(psm_dat, dummy_proxy):
    with pytest.raises(ValueError):
        p = dummy_proxy
        pclass = proxy2.ProxyPages(cfg, p.pid, p.ptype, p.start_yr, p.end_yr,
                                   p.lat, p.lon, [], p.time,
                                   psm_kwargs={'psm_data': psm_dat})


def test_pages_none_values(psm_dat, dummy_proxy):
    with pytest.raises(ValueError):
        p = dummy_proxy
        pclass = proxy2.ProxyPages(cfg, p.pid, p.ptype, p.start_yr, p.end_yr,
                                   p.lat, p.lon, None, p.time,
                                   psm_kwargs={'psm_data': psm_dat})


def test_pages_time_values_len_mismatch(psm_dat, dummy_proxy):
    with pytest.raises(AssertionError):
        p = dummy_proxy
        time = p.time[0:-2]
        pclass = proxy2.ProxyPages(cfg, p.pid, p.ptype, p.start_yr, p.end_yr,
                                   p.lat, p.lon, p.values, time,
                                   psm_kwargs={'psm_data': psm_dat})


def test_pages_load_site(meta, pdata, psm_dat):
    psm_kwargs = {'psm_data': psm_dat}
    drange = [1980, 2000]
    start, end = drange
    pclass = proxy2.ProxyPages.load_site(cfg, 'Aus_16', meta, pdata,
                                         drange, **psm_kwargs)

    assert pclass.id == 'Aus_16'
    assert np.alltrue((pclass.time >= start) & (pclass.time <= end))
    assert pclass.type == r'Coral_d18O'
    assert pclass.lat == -21
    assert pclass.lon == proxy2.fix_lon(-160)
    np.testing.assert_array_equal(pclass.values.values,
        pdata['Aus_16'][(pdata.index >= start) &
                        (pdata.index <= end) &
                        pdata['Aus_16'].notnull()].values)


def test_pages_load_site_no_obs(meta, pdata, psm_dat):
    psm_kwargs = {'psm_data': psm_dat}
    drange = [2000, 3000]
    start, end = drange
    with pytest.raises(ValueError):
        pclass = proxy2.ProxyPages.load_site(cfg, 'Aus_16', meta, pdata,
                                             drange, **psm_kwargs)


def test_pages_load_site_non_consectutive(meta, pdata, psm_dat):
    dat = pdata['Aus_16'][(pdata.index >= 1982) &
                          (pdata.index <= 1991)]
    dat[1984:1988] = np.nan
    dat = dat[dat.notnull()]

    psm_kwargs = {'psm_data': psm_dat}
    drange = [1982, 1991]
    start, end = drange

    pclass = proxy2.ProxyPages.load_site(cfg, 'Aus_16', meta, pdata,
                                         drange, **psm_kwargs)
    np.testing.assert_array_equal(dat.index.values, pclass.time)
    np.testing.assert_array_equal(dat.values, pclass.values.values)


def test_pages_load_all():
    pass

if __name__ == '__main__':
    test_pages_load_site(meta(None), pdata(None), psm_dat(None))