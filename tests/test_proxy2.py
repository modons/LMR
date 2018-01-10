__author__ = 'frodre'


# Temporary import until it's turned into a package
import sys
sys.path.append('../')

import pytest
import copy
import pickle
import numpy as np
import LMR_proxy_pandas_rework as proxy2
import LMR_config as cfg
import yaml



@pytest.fixture()
def dummy_proxy(request):
    class proxy:
        pass

    p = proxy()
    p.pid = 'Aus_01'
    p.ptype = 'Tree ring_Width'
    p.start_yr = 1950
    p.end_yr = 1960
    p.values = list(range(11))
    p.lat = 45
    p.lon = 210
    p.elev = 0
    p.time = list(range(1950, 1961))
    p.seasonality = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    return p

@pytest.fixture(scope='module')
def config(request):
    with open('test_config.yml', 'r') as f:
        cfg_update_dict = yaml.load(f)

    cfg.update_config_class_yaml(cfg_update_dict, cfg)

    def fin():
        reload(cfg)

    request.addfinalizer(fin)
    return cfg.Config()

@pytest.fixture(scope='module')
def meta(request):
    with open('test_meta.pckl', 'rb') as f:
        dat = pickle.load(f)
    return dat


@pytest.fixture(scope='function')
def pdata(request):
    with open('test_pdata.pckl', 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope='module')
def seasons(request):
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    return data


def test_abstract_class_creation():
    with pytest.raises(TypeError):
        x = proxy2.BaseProxyObject()


def test_lon_fix():
    assert proxy2.fix_lon(0) == 0
    assert proxy2.fix_lon(-180) == 180
    assert proxy2.fix_lon(-90) == 270


def test_proxies_get_class():
    assert type(proxy2.get_proxy_class('pages')) == type(proxy2.ProxyPages)


def test_pages_init(dummy_proxy, config):
    p = dummy_proxy
    pclass = proxy2.ProxyPages(config, p.pid, p.ptype, p.start_yr, p.end_yr,
                               p.lat, p.lon, p.elev, p.seasonality,
                               p.values,
                               p.time)

    assert pclass.id == p.pid
    assert pclass.type == p.ptype
    assert pclass.start_yr == p.start_yr
    assert pclass.end_yr == p.end_yr
    assert pclass.values == p.values
    assert pclass.lat == p.lat
    assert pclass.lon == p.lon
    assert pclass.elev == p.elev
    assert pclass.time == p.time
    assert pclass.seasonality == p.seasonality
    assert hasattr(pclass, 'psm')


def test_pages_empty_values(seasons, dummy_proxy, config):
    with pytest.raises(ValueError):
        p = dummy_proxy
        pclass = proxy2.ProxyPages(config, p.pid, p.ptype, p.start_yr,
                                   p.end_yr,
                                   p.lat, p.lon, p.elev, seasons, [], p.time)


def test_pages_none_values(seasons, dummy_proxy, config):
    with pytest.raises(ValueError):
        p = dummy_proxy
        pclass = proxy2.ProxyPages(config, p.pid, p.ptype, p.start_yr,
                                   p.end_yr,
                                   p.lat, p.lon, p.elev, seasons, None, p.time)


def test_pages_time_values_len_mismatch(seasons, dummy_proxy, config):
    with pytest.raises(AssertionError):
        p = dummy_proxy
        time = p.time[0:-2]
        pclass = proxy2.ProxyPages(config, p.pid, p.ptype, p.start_yr,
                                   p.end_yr,
                                   p.lat, p.lon, p.elev, seasons, p.values,
                                   time)


def test_pages_load_site(meta, pdata, config):
    drange = [1980, 2000]
    start, end = drange
    pclass = proxy2.ProxyPages.load_site(config, 'Aus_16', drange, meta, pdata)

    assert pclass.id == 'Aus_16'
    assert np.alltrue((pclass.time >= start) & (pclass.time <= end))
    assert pclass.type == r'Coral_d18O'
    assert pclass.lat == -21
    assert pclass.lon == proxy2.fix_lon(-160)
    assert pclass.elev == 0
    np.testing.assert_array_equal(pclass.values.values,
        pdata['Aus_16'][(pdata.index >= start) &
                        (pdata.index <= end) &
                        pdata['Aus_16'].notnull()].values)

def test_pages_load_site_no_preloaded(pdata, config):
    drange = [1980, 2000]
    start, end = drange
    pclass = proxy2.ProxyPages.load_site(config, 'Aus_16', drange)

    assert pclass.id == 'Aus_16'
    assert np.alltrue((pclass.time >= start) & (pclass.time <= end))
    assert pclass.type == r'Coral_d18O'
    assert pclass.lat == -21
    assert pclass.lon == proxy2.fix_lon(-160)
    assert pclass.elev == 0
    np.testing.assert_array_equal(pclass.values.values,
        pdata['Aus_16'][(pdata.index >= start) &
                        (pdata.index <= end) &
                        pdata['Aus_16'].notnull()].values)


def test_pages_load_site_no_obs(meta, pdata, config):
    drange = [2000, 3000]
    with pytest.raises(ValueError):
        pclass = proxy2.ProxyPages.load_site(config, 'Aus_16', drange, meta,
                                             pdata)


def test_pages_load_site_non_consectutive(meta, pdata, config):
    dat = pdata['Aus_16'][(pdata.index >= 1982) &
                          (pdata.index <= 1991)]
    dat[1984:1988] = np.nan
    dat = dat[dat.notnull()]
    drange = [1982, 1991]

    pclass = proxy2.ProxyPages.load_site(config, 'Aus_16', drange, meta, pdata)
    np.testing.assert_array_equal(dat.index.values, pclass.time)
    np.testing.assert_array_equal(dat.values, pclass.values.values)


def test_pages_load_all(meta, pdata, config):
    drange = [1970, 2000]

    # Everything here depends upon default configuration
    bytype, allp = proxy2.ProxyPages.load_all(config, drange, meta, pdata)

    assert len(allp) == 4
    assert bytype['Tree ring_Width'] == ['Aus_04']
    assert bytype['Coral_d18O'] == ['Aus_16']
    assert bytype['Tree ring_Density'] == ['NAm-TR_065']
    assert bytype['Ice core_d18O'] == ['Arc_35']

    # Check values and order
    order = config.proxies.pages.proxy_order

    for i, p in enumerate(allp):
        if i < len(allp) - 1:
            assert order.index(p.type) <= order.index(allp[i+1].type)

        pdata_site = pdata[p.id][p.time]
        np.testing.assert_array_equal(p.values.values, pdata_site.values)


def test_pages_load_all_specific_regions(meta, pdata):
    drange = [1970, 2000]

    new_regions = ['Australasia', 'Arctic', 'Europe']
    update_dict = {'proxies': {'pages': {'regions': new_regions}}}
    config = cfg.Config(**update_dict)

    # Everything here depends upon default configuration
    bytype, allp = proxy2.ProxyPages.load_all(config, drange, meta, pdata)

    assert len(allp) == 3

    not_list = ['Antarctica', 'Asia', 'Europe', 'North America',
                'South America']
    for p in allp:
        region = meta['PAGES 2k Region'][meta['PAGES ID'] == p.id].iloc[0]
        assert region not in not_list


def test_pages_load_all_specific_type(meta, pdata):
    drange = [1970, 2000]
    proxy_assim2 = {
        'Tree ring_Width': ['Ring width',
                            'Tree ring width',
                            'Total ring width',
                            'TRW'],
        'Ice core_d18O': ['d18O']
    }
    update_dict = {'proxies': {'pages': {'proxy_assim2': proxy_assim2,
                                         'proxy_order': ['Tree ring_Width',
                                                         'Ice core_d18O']}}}
    cfg_obj = cfg.Config(**update_dict)

    # Everything here depends upon default configuration
    bytype, allp = proxy2.ProxyPages.load_all(cfg_obj, drange, meta, pdata)

    assert len(allp) == 2
    assert bytype['Ice core_d18O'] == ['Arc_35']
    assert bytype['Tree ring_Width'] == ['Aus_04']

    with pytest.raises(KeyError):
        bytype['Coral_d18O']

    with pytest.raises(KeyError):
        bytype['Tree ring_Density']


def test_pages_load_all_annual_no_filtering(meta, pdata):

    cfg_obj = cfg.Config(**{'proxies': {'use_from': ['pages']},
                            'psm': {'linear': {'psm_r_crit': 0.0}}})

    # Erase Filter parameters
    cfg_obj.proxies.pages.proxy_assim2 = {}
    cfg_obj.proxies.pages.proxy_blacklist = []
    cfg_obj.proxies.pages.proxy_order = []
    cfg_obj.proxies.pages.regions = []

    pobjs = proxy2.ProxyPages.load_all_annual_no_filtering(cfg_obj, meta, pdata)

    # There are six proxy records, one of them has no values
    assert len(pobjs) == 5


def test_pages_proxy_manager_all():
    drange = [1970, 2000]
    test_dir = '/home/disk/p/wperkins/Research/LMR/tests'
    update_dict = {'core': {'recon_period': drange},
                   'proxies': {'pages': {'datadir_proxy': test_dir,
                                         'datafile_proxy': 'test_pdata.pckl',
                                         'metafile_proxy': 'test_meta.pckl'},
                               'proxy_frac': 1.0}}
    config = cfg.Config(**update_dict)

    pmanager = proxy2.ProxyManager(config, drange)
    assert pmanager.ind_eval is None
    assert len(pmanager.ind_assim) == 4

    bytype, allp = proxy2.ProxyPages.load_all(config, drange)

    for mobj, pobj in zip(pmanager.sites_assim_proxy_objs(), allp):
        assert mobj.id == pobj.id


def test_pages_proxy_manager_proxy_fracs():
    drange = [1970, 2000]
    test_dir = '/home/disk/p/wperkins/Research/LMR/tests'
    update_dict = {'core': {'recon_period': drange},
                   'proxies': {'pages': {'datadir_proxy': test_dir,
                                         'datafile_proxy': 'test_pdata.pckl',
                                         'metafile_proxy': 'test_meta.pckl'},
                               'proxy_frac': 0.5}}
    config = cfg.Config(**update_dict)

    pmanager = proxy2.ProxyManager(config, drange)
    assert len(pmanager.ind_assim) == 2
    assert len(pmanager.ind_eval) == 2

    update_dict['proxies']['proxy_frac'] = 0.0
    config = cfg.Config(**update_dict)
    pmanager = proxy2.ProxyManager(config, drange)
    assert len(pmanager.ind_assim) == 0
    assert len(pmanager.ind_eval) == 4


def test_pages_proxy_manager_seeded():
    drange = [1970, 2000]
    update_dict = {'proxies':
                              {'pages': {'datadir_proxy': '/home/disk/p/wperkins/Research/LMR/tests',
                                         'datafile_proxy': 'test_pdata.pckl',
                                         'metafile_proxy': 'test_meta.pckl'},
                               'proxy_frac': 0.75},
                   'core': {'seed': 1}}
    cfg_obj = cfg.Config(**update_dict)

    pmanager = proxy2.ProxyManager(cfg_obj, drange)
    pmanager2 = proxy2.ProxyManager(cfg_obj, drange)

    assert len(pmanager.ind_assim) == len(pmanager2.ind_assim)

    for pid in pmanager.ind_assim:
        assert pid in pmanager2.ind_assim

    for pid in pmanager.ind_eval:
        assert pid in pmanager2.ind_eval


if __name__ == '__main__':
    test_pages_proxy_manager_proxy_fracs(psm_dat(None))