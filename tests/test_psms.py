__author__ = 'frodre'

# Temporary import until it's turned into a package
import sys.path as path
path.append('../')

import pytest
import LMR_psms as psms


@pytest.mark.xfail
def test_abstract_class_creation():
    x = psms.BasePSM()


def test_linear_with_psm_data():
    pass


def test_linear_load_from_config():
    pass


def test_linear_calibrate_proxy():
    pass


def test_linear_calibrate_psm_data_no_key_match():
    pass


def test_linear_calibrate_pasm_data_config_file_not_found():
    pass


@pytest.mark.xfail
def test_linear_corr_below_rcrit():
    pass


def test_linear_psm_ye_val():
    pass


def test_get_kwargs():
    pass


def test_get_psm_class():
    pass
