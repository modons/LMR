__author__ = 'frodre'


# Temporary import until it's turned into a package
import sys
sys.path.append('../')

import pytest
import cPickle
import numpy as np
import LMR_proxy_pandas_rework as proxy2
import LMR_config as cfg
from itertools import izip


@pytest.fixture()
def psm_dat(request):

    cfg_obj = cfg.Config()
    f = open(cfg_obj.psm.linear.pre_calib_datafile)
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
    cfg_obj = cfg.Config()
    p = dummy_proxy
    pclass = proxy2.ProxyPages(cfg_obj, p.pid, p.ptype, p.start_yr, p.end_yr,
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
    cfg_obj = cfg.Config()
    with pytest.raises(ValueError):
        p = dummy_proxy
        pclass = proxy2.ProxyPages(cfg_obj, p.pid, p.ptype, p.start_yr, p.end_yr,
                                   p.lat, p.lon, [], p.time,
                                   psm_kwargs={'psm_data': psm_dat})


def test_pages_none_values(psm_dat, dummy_proxy):
    cfg_obj = cfg.Config()
    with pytest.raises(ValueError):
        p = dummy_proxy
        pclass = proxy2.ProxyPages(cfg_obj, p.pid, p.ptype, p.start_yr, p.end_yr,
                                   p.lat, p.lon, None, p.time,
                                   psm_kwargs={'psm_data': psm_dat})


def test_pages_time_values_len_mismatch(psm_dat, dummy_proxy):
    cfg_obj = cfg.Config()
    with pytest.raises(AssertionError):
        p = dummy_proxy
        time = p.time[0:-2]
        pclass = proxy2.ProxyPages(cfg_obj, p.pid, p.ptype, p.start_yr, p.end_yr,
                                   p.lat, p.lon, p.values, time,
                                   psm_kwargs={'psm_data': psm_dat})


def test_pages_load_site(meta, pdata, psm_dat):
    cfg_obj = cfg.Config()
    psm_kwargs = {'psm_data': psm_dat}
    drange = [1980, 2000]
    start, end = drange
    pclass = proxy2.ProxyPages.load_site(cfg_obj, 'Aus_16', drange, meta, pdata,
                                         **psm_kwargs)

    assert pclass.id == 'Aus_16'
    assert np.alltrue((pclass.time >= start) & (pclass.time <= end))
    assert pclass.type == r'Coral_d18O'
    assert pclass.lat == -21
    assert pclass.lon == proxy2.fix_lon(-160)
    np.testing.assert_array_equal(pclass.values.values,
        pdata['Aus_16'][(pdata.index >= start) &
                        (pdata.index <= end) &
                        pdata['Aus_16'].notnull()].values)

def test_pages_load_site_no_preloaded(meta, pdata, psm_dat):
    cfg_obj = cfg.Config()
    psm_kwargs = {'psm_data': psm_dat}
    drange = [1980, 2000]
    start, end = drange
    pclass = proxy2.ProxyPages.load_site(cfg_obj, 'Aus_16', drange, **psm_kwargs)

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
    cfg_obj = cfg.Config()
    psm_kwargs = {'psm_data': psm_dat}
    drange = [2000, 3000]
    with pytest.raises(ValueError):
        pclass = proxy2.ProxyPages.load_site(cfg_obj, 'Aus_16', drange, meta,
                                             pdata, **psm_kwargs)


def test_pages_load_site_non_consectutive(meta, pdata, psm_dat):
    cfg_obj = cfg.Config()
    dat = pdata['Aus_16'][(pdata.index >= 1982) &
                          (pdata.index <= 1991)]
    dat[1984:1988] = np.nan
    dat = dat[dat.notnull()]

    psm_kwargs = {'psm_data': psm_dat}
    drange = [1982, 1991]

    pclass = proxy2.ProxyPages.load_site(cfg_obj, 'Aus_16', drange, meta, pdata,
                                         **psm_kwargs)
    np.testing.assert_array_equal(dat.index.values, pclass.time)
    np.testing.assert_array_equal(dat.values, pclass.values.values)


def test_pages_load_all(meta, pdata, psm_dat):
    psm_kwargs = {'psm_data': psm_dat}
    drange = [1970, 2000]
    cfg_obj = cfg.Config()

    # Everything here depends upon default configuration
    bytype, allp = proxy2.ProxyPages.load_all(cfg_obj, drange, meta, pdata,
                                             **psm_kwargs)

    assert len(allp) == 4
    assert bytype['Tree ring_Width'] == ['Aus_04']
    assert bytype['Coral_d18O'] == ['Aus_16']
    assert bytype['Tree ring_Density'] == ['NAm-TR_065']
    assert bytype['Ice core_d18O'] == ['Arc_35']

    # Check values and order
    order = cfg_obj.proxies.pages.proxy_order

    for i, p in enumerate(allp):
        if i < len(allp) - 1:
            assert order.index(p.type) <= order.index(allp[i+1].type)

        pdata_site = pdata[p.id][p.time]
        np.testing.assert_array_equal(p.values.values, pdata_site.values)


def test_pages_load_all_specific_regions(meta, pdata, psm_dat):
    psm_kwargs = {'psm_data': psm_dat}
    drange = [1970, 2000]
    cfg_obj = cfg.Config()

    cfg_obj.proxies.pages.regions = ['Australasia', 'Arctic', 'Europe']
    cfg_obj.proxies.pages.simple_filters['PAGES 2k Region'] = \
        cfg_obj.proxies.pages.regions

    # Everything here depends upon default configuration
    bytype, allp = proxy2.ProxyPages.load_all(cfg_obj, drange, meta, pdata,
                                              **psm_kwargs)

    assert len(allp) == 3

    not_list = ['Antarctica', 'Asia', 'Europe', 'North America',
                'South America']
    for p in allp:
        region = meta['PAGES 2k Region'][meta['PAGES ID'] == p.id].iloc[0]
        assert region not in not_list


def test_pages_load_all_specific_type(meta, pdata, psm_dat):
    psm_kwargs = {'psm_data': psm_dat}
    drange = [1970, 2000]
    cfg_obj = cfg.Config()

    cfg_obj.proxies.pages.proxy_assim2 = {
        'Tree ring_Width': ['Ring width',
                            'Tree ring width',
                            'Total ring width',
                            'TRW'],
        'Ice core_d18O': ['d18O']
    }
    cfg_obj.proxies.pages.proxy_order = ['Tree ring_Width', 'Ice core_d18O']

    # Everything here depends upon default configuration
    bytype, allp = proxy2.ProxyPages.load_all(cfg_obj, drange, meta, pdata,
                                              **psm_kwargs)

    assert len(allp) == 2
    assert bytype['Ice core_d18O'] == ['Arc_35']
    assert bytype['Tree ring_Width'] == ['Aus_04']

    with pytest.raises(KeyError):
        bytype['Coral_d18O']

    with pytest.raises(KeyError):
        bytype['Tree ring_Density']


def test_pages_proxy_manager_all(psm_dat):
    psm_kwargs = {'psm_data': psm_dat}
    drange = [1970, 2000]
    cfg_obj = cfg.Config()

    cfg_obj.proxies.pages.datafile_proxy = 'test_pdata.pckl'
    cfg_obj.proxies.pages.metafile_proxy = 'test_meta.pckl'
    cfg_obj.proxies.proxy_frac = 1.0

    pmanager = proxy2.ProxyManager(cfg_obj, drange)
    assert pmanager.ind_eval is None
    assert len(pmanager.ind_assim) == 4

    bytype, allp = proxy2.ProxyPages.load_all(cfg_obj, drange, **psm_kwargs)

    for mobj, pobj in izip(pmanager.sites_assim_proxy_objs(), allp):
        assert mobj.id == pobj.id


def test_pages_proxy_manager_proxy_fracs(psm_dat):
    drange = [1970, 2000]
    cfg_obj = cfg.Config()

    cfg_obj.proxies.pages.datafile_proxy = 'test_pdata.pckl'
    cfg_obj.proxies.pages.metafile_proxy = 'test_meta.pckl'
    cfg_obj.proxies.proxy_frac = 0.5

    pmanager = proxy2.ProxyManager(cfg_obj, drange)
    assert len(pmanager.ind_assim) == 2
    assert len(pmanager.ind_eval) == 2

    cfg_obj.proxies.proxy_frac = 0.0
    pmanager = proxy2.ProxyManager(cfg_obj, drange)
    assert len(pmanager.ind_assim) == 0
    assert len(pmanager.ind_eval) == 4


def test_pages_proxy_manager_seeded(psm_dat):
    drange = [1970, 2000]
    cfg_obj = cfg.Config()

    cfg_obj.proxies.pages.datafile_proxy = 'test_pdata.pckl'
    cfg_obj.proxies.pages.metafile_proxy = 'test_meta.pckl'
    cfg_obj.proxies.proxy_frac = 0.75
    cfg_obj.proxies.seed = 1

    pmanager = proxy2.ProxyManager(cfg_obj, drange)
    pmanager2 = proxy2.ProxyManager(cfg_obj, drange)

    assert len(pmanager.ind_assim) == len(pmanager2.ind_assim)

    for pid in pmanager.ind_assim:
        assert pid in pmanager2.ind_assim

    for pid in pmanager.ind_eval:
        assert pid in pmanager2.ind_eval


if __name__ == '__main__':
    test_pages_proxy_manager_proxy_fracs(psm_dat(None))