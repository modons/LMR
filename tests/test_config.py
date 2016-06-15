import sys
sys.path.append('../')

import pytest
import LMR_config
import os.path as opath

@pytest.fixture(scope='function')
def cfg(request):
    def fin():
        reload(LMR_config)
    request.addfinalizer(fin)

    return LMR_config


# has right attributes
# test defaults
def test_default_configuration_core(cfg):
    cfg_object = cfg.Config()

    attrs = ['wrapper', 'core', 'proxies', 'psm', 'prior']
    for attr in attrs:
        assert hasattr(cfg_object, attr)

    assert cfg.core.lmr_path == cfg_object.core.lmr_path
    assert hasattr(cfg_object.core, 'curr_iter')

    assert hasattr(cfg_object.prior, 'datadir_prior')


def test_default_configuration_seed(cfg):
    new_seed = 1234
    cfg.core.seed = new_seed
    cfg_object = cfg.Config()

    assert hasattr(cfg_object.core, 'seed')
    assert hasattr(cfg_object.proxies, 'seed')
    assert hasattr(cfg_object.prior, 'seed')

    assert cfg_object.proxies.seed == new_seed
    assert cfg_object.prior.seed == new_seed


def test_default_configuration_proxies_pages(cfg):
    cfg_object = cfg.Config()

    assert hasattr(cfg_object.proxies, 'pages')
    assert hasattr(cfg_object.proxies.pages, 'datadir_proxy')
    assert hasattr(cfg_object.proxies.pages, 'datafile_proxy')
    assert hasattr(cfg_object.proxies.pages, 'metafile_proxy')
    assert hasattr(cfg_object.proxies.pages, 'proxy_type_mapping')
    assert hasattr(cfg_object.proxies.pages, 'simple_filters')

    assert cfg_object.core.lmr_path in cfg_object.proxies.pages.datadir_proxy


def test_default_configuration_psm_linear(cfg):
    cfg_object = cfg.Config()

    assert hasattr(cfg_object.psm, 'linear')
    assert hasattr(cfg_object.psm.linear, 'datadir_calib')
    assert hasattr(cfg_object.psm.linear, 'pre_calib_datafile')

    assert cfg_object.core.lmr_path in cfg_object.psm.linear.datadir_calib


def test_default_configuration_psm_linear_t_or_p(cfg):
    cfg_object = cfg.Config()

    assert hasattr(cfg_object.psm, 'linear_TorP')
    assert hasattr(cfg_object.psm.linear_TorP, 'datadir_calib')
    assert hasattr(cfg_object.psm.linear_TorP, 'pre_calib_datafile_T')
    assert hasattr(cfg_object.psm.linear_TorP, 'pre_calib_datafile_P')


# test default and then changed default
def test_default_configuration_change_default_path(cfg):
    orig_path = cfg.core.lmr_path
    cfg1 = cfg.Config()
    new_path = 'new_path/is/here'

    cfg.core.lmr_path = new_path
    cfg2 = cfg.Config()

    assert cfg1.core.lmr_path != cfg2.core.lmr_path
    assert new_path in cfg2.prior.datadir_prior
    assert new_path in cfg2.psm.linear.datadir_calib
    assert new_path in cfg2.proxies.pages.datadir_proxy
    assert orig_path not in cfg2.prior.datadir_prior
    assert orig_path not in cfg2.psm.linear.datadir_calib
    assert orig_path not in cfg2.proxies.pages.datadir_proxy

    cfg.core.lmr_path = orig_path

# test keyword args
@pytest.mark.xfail()
def test_configuration_keyword_arg_usage(cfg):
    core_dir = cfg.core.lmr_path

    new_dir = 'testdir/'
    proxy_config = cfg.proxies(datadir_proxy=new_dir)
    assert new_dir in proxy_config.pages.datadir_proxy
    assert core_dir not in proxy_config.pages.datadir_proxy

    new_file = 'test.pckl'
    psm_config = cfg.psm(datadir_calib=new_dir, pre_calib_datafile=new_file)
    assert new_dir in psm_config.linear.datadir_calib
    assert new_file in psm_config.linear.pre_calib_datafile

