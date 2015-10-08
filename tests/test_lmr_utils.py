import sys
sys.path.append('../')

import pytest
import LMR_utils2 as Utils
import netCDF4 as ncf
import numpy as np

@pytest.fixture(scope='module')
def ncf_data(request):
    f_obj = ncf.Dataset('data/gridded_dat.nc', 'r')

    def fin():
        f_obj.close()

    request.addfinalizer(fin)
    return f_obj


@pytest.mark.parametrize("doc", [
    """This is class docstring""",
    None])
def test_class_doc_inherit(doc):
    class foo:
        __doc__ = doc
        pass

    @Utils.class_docs_fixer
    class bar(foo):
        pass

    assert bar.__doc__ == doc


@pytest.mark.parametrize("doc", [
    """This is func docstring""",
    None])
def test_function_doc_inherit(doc):
    class foo:
        def lol(self):
            pass

    foo.lol.__func__.__doc__ = doc

    @Utils.class_docs_fixer
    class bar(foo):
        def lol(self):
            pass

    assert bar.lol.__func__.__doc__ == doc


def test_function_doc_augment():
    parent_doc = """This is the parents lol docstr"""
    child_doc = """%%aug%%
            The childs doc is here
            """

    class foo:
        def lol(self):
            pass

    foo.lol.__func__.__doc__ = parent_doc

    @Utils.class_docs_fixer
    class bar(foo):
        @Utils.augment_docstr
        def lol(self):
            """%%aug%%
            The childs doc is here
            """
            pass

    assert bar.lol.__func__.__doc__ == (parent_doc +
                                        child_doc.replace('%%aug%%', ''))


def test_global_mean2(ncf_data):

    dat = ncf_data['air'][0:4]
    lat = ncf_data['lat'][:]
    lon = ncf_data['lon'][:]

    longrid, latgrid = np.meshgrid(lon, lat)

    gm_time = Utils.global_mean(dat, lat, lon)
    gm0 = Utils.global_mean(dat[0], lat, lon)

    # with time
    gm_test = Utils.global_mean2(dat, lat)
    np.testing.assert_allclose(gm_test, gm_time)

    # flattened lat w/ time
    flat_dat = dat.reshape(4, 94*192)
    gm_test = Utils.global_mean2(flat_dat, latgrid.flatten())
    np.testing.assert_allclose(gm_test, gm_time)

    # no time
    gm_test = Utils.global_mean2(dat[0], lat)
    np.testing.assert_allclose(gm_test, gm0)

    # no time flattened spatial
    gm_test = Utils.global_mean2(dat[0].flatten(), latgrid.flatten())
    np.testing.assert_allclose(gm_test, gm0)


