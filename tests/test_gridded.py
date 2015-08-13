__author__ = 'wperkins'

"""
Test LMR_gridded classes and methods
"""

# Temporary import until it's turned into a package
import sys
sys.path.append('../')

import pytest
import LMR_gridded as lmrgrid
import numpy as np
import netCDF4 as ncf
import random


@pytest.fixture(scope='module')
def ncf_data(request):
    f_obj = ncf.Dataset('data/gridded_dat.nc', 'r')

    def fin():
        f_obj.close()

    request.addfinalizer(fin)
    return f_obj


@pytest.mark.xfail
def test_gridded_data_no_abstract_instance():
    """
    Test abstract variable instantiation fails
    """
    lmrgrid.GriddedVariable('tas', ['time', 'lat', 'lon'], np.zeros((3, 3, 3)),
                            [1.0], time=range(3), lat=range(3), lon=range(3))


def test_priorvar_init(ncf_data):
    """
    Test basic initialization with time, lat, lon data
    """
    dat = ncf_data.variables
    lat = dat['lat']
    lon = dat['lon']
    time = dat['time']
    data = dat['air']

    prior_var = lmrgrid.PriorVariable('air', data.dimensions,
                                      data[:], 1.0, time=time[:],
                                      lat=lat[:], lon=lon[:])

    assert prior_var.name == 'air'
    assert prior_var.dim_order == ('time', 'lat', 'lon')
    assert prior_var.ndim == 3
    np.testing.assert_array_equal(prior_var.data, data[:])
    assert prior_var.resolution == 1.0
    np.testing.assert_array_equal(prior_var.time, time[:])
    assert prior_var.nsamples == len(time)
    assert prior_var.lev is None
    np.testing.assert_array_equal(prior_var.lat, lat[:])
    np.testing.assert_array_equal(prior_var.lon, lon[:])
    assert prior_var._space_dims == ['lat', 'lon']
    assert prior_var._space_shp == [94, 192]
    assert prior_var.type == 'horizontal'


@pytest.mark.parametrize('init_args, kw_args, type',
     [(('tseries', ['time'], np.arange(2), 1.0),
       {'time': np.arange(2)},
       'timeseries'),
      (('vert', ['time', 'lev', 'lat'], np.arange(20).reshape(2, 2, 5), 1.0),
       {'time': np.arange(2), 'lev': np.arange(2), 'lat': np.arange(5)},
       'meridional_vertical'),
      pytest.mark.xfail(raises=ValueError)((('wrong_ndim', ['time', 'lev'],
                                             np.arange(10), 1.0),
          {'time': np.arange(2), 'lev': np.arange(5)},
          'failure')),
      pytest.mark.xfail(raises=ValueError)((('dim_missvals', ['time', 'lev'],
                          np.arange(10).reshape(2,5), 1.0),
          {'time': np.arange(2)},
          'failure')),
      pytest.mark.xfail(raises=ValueError)((('dim_mismatch', ['time', 'lev'],
                          np.arange(10).reshape(2,5), 1.0),
          {'time': np.arange(2), 'lev': np.arange(4)},
          'failure')),
      pytest.mark.xfail(raises=KeyError)((('unrecog', ['time', 'notadim'],
                                           np.arange(4).reshape(2, 2), 1.0),
                         {'time': np.arange(2)},
                         'failed')),
      pytest.mark.xfail(raises=ValueError)((('unrecog_combo',
                                             ['time', 'lev', 'lon'],
                                             np.arange(20).reshape(2, 2, 5),
                                             1.0),
                         {'time': np.arange(2), 'lev': np.arange(2),
                          'lon': np.arange(5)},
                         'failed')),
      pytest.mark.xfail(raises=ValueError)((('2manydims', ['time', 'lev', 'lat',
                                                           'lon'],
                          np.arange(20).reshape(1, 2, 2, 5), 1.0),
                         {'time': np.arange(1), 'lev': np.arange(2),
                          'lat': np.arange(2), 'lon': np.arange(5)},
                         'failed'))
     ])
def test_priorvar_types(init_args, kw_args, type):
    """
    Test basic gridded variable types designation. Timeseries, horizontal,
    meridional/vertical.
    """


    prior_var = lmrgrid.PriorVariable(*init_args, **kw_args)

    assert prior_var.type == type
    if type == 'timeseries':
        assert prior_var._space_shp == [1]
        assert prior_var.data.shape[1] == 1

def test_priorvar_trunc(ncf_data):
    """
    Test regridding of grid_obj data
    """

    dat = ncf_data.variables
    lat = dat['lat']
    lon = dat['lon']
    time = dat['time']
    data = dat['air']

    prior_var = lmrgrid.PriorVariable('air', data.dimensions,
                                      data[:], 1.0, time=time[:],
                                      lat=lat[:], lon=lon[:])
    new_prior = prior_var.truncate(42)

    assert new_prior.data.shape == (len(time), 44, 66)


def test_priorvar_flattened(ncf_data):
    """
    Test flattening spatial fields
    """

    dat = ncf_data.variables
    lat = dat['lat']
    lon = dat['lon']
    time = dat['time']
    data = dat['air']

    prior_var = lmrgrid.PriorVariable('air', data.dimensions,
                                      data[:], 1.0, time=time[:],
                                      lat=lat[:], lon=lon[:])

    flat_dat, flat_coords = prior_var.flattened_spatial()

    longrd, latgrd = np.meshgrid(lon[:], lat[:])
    flat = data[:].reshape(data.shape[0], np.product(data.shape[1:]))

    assert 'lat' in flat_coords.keys()
    assert 'lon' in flat_coords.keys()
    assert 'time' not in flat_coords.keys()
    assert 'lev' not in flat_coords.keys()
    assert data.shape[0] == flat_dat.shape[0]
    np.testing.assert_array_equal(flat, flat_dat)
    np.testing.assert_array_equal(longrd.flatten(), flat_coords['lon'])
    np.testing.assert_array_equal(latgrd.flatten(), flat_coords['lat'])


def test_priorvar_sample(ncf_data):
    """
    Test sampling of prior variable
    """

    dat = ncf_data.variables
    lat = dat['lat']
    lon = dat['lon']
    time = dat['time']
    data = dat['air']

    prior_var = lmrgrid.PriorVariable('air', data.dimensions,
                                      data[:], 1.0, time=time[:],
                                      lat=lat[:], lon=lon[:])
    seed = 15
    nens = 5
    random.seed(seed)
    smp_idx = random.sample(range(len(time)), nens)
    dat_sample = np.zeros([nens] + list(data.shape[1:]))
    for k, idx in enumerate(smp_idx):
        dat_sample[k] = data[idx]

    time_sample = time[:][smp_idx]

    sampled_prior = prior_var.sample(nens, seed=seed)

    assert len(sampled_prior.data) == nens
    assert dat_sample.shape == sampled_prior.data.shape
    np.testing.assert_array_equal(dat_sample, sampled_prior.data)
    np.testing.assert_array_equal(time_sample, sampled_prior.time)
