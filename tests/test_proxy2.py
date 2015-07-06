__author__ = 'frodre'


# Temporary import until it's turned into a package
import sys.path as path
path.append('../')

import pytest
import LMR_proxy2 as proxy2

@pytest.mark.xfail
def test_abstract_class_creation():
    x = proxy2.BaseProxyObject()


def test_lon_fix():
    pass


def test_pages_init():
    pass


def test_pages_load_site():
    pass


def test_pages_load_site_no_obs():
    pass


def test_pages_load_all():
    pass
