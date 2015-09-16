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
import os
import random
from itertools import izip



@pytest.fixture(scope='module')
def ncf_data(request):
    f_obj = ncf.Dataset('data/gridded_dat.nc', 'r')

    def fin():
        f_obj.close()

    request.addfinalizer(fin)
    return f_obj

@pytest.fixture(scope='function')
def ncf_dates(request, ncf_data):
    time = ncf_data.variables['time']
    return ncf.num2date(time[:], time.units)

@pytest.fixture(scope='function')
def ncf_temps(request, ncf_data):
    return ncf_data.variables['air'][:]


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
    assert prior_var.space_shp == [94, 192]
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
        assert prior_var.space_shp == [1]
        assert prior_var.data.shape[1] == 1


def test_priorvar_save(ncf_data):
    """
    Test object saving
    """

    dat = ncf_data.variables
    lat = dat['lat']
    lon = dat['lon']
    time = dat['time']
    data = dat['air']

    prior_var = lmrgrid.PriorVariable('air', data.dimensions,
                                      data[:], 1.0, time=time[:],
                                      lat=lat[:], lon=lon[:])

    prior_var.save('data/test_save')
    assert os.path.exists('data/test_save.h5')


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

    sampled_prior = prior_var.sample_from_idx(smp_idx)

    assert len(sampled_prior.data) == nens
    assert dat_sample.shape == sampled_prior.data.shape
    np.testing.assert_array_equal(dat_sample, sampled_prior.data)
    np.testing.assert_array_equal(time_sample, sampled_prior.time)


def test_griddedvar_netcdf_timeconvert(ncf_data):

    time = ncf_data['time']
    dates = lmrgrid.GriddedVariable._netcdf_datetime_convert(time)

    assert len(time) == len(dates)


def test_griddedvar_netcdf_timeconvert_units_outofrange(ncf_data):

    time = ncf_data['time_inv_ref']
    dates = lmrgrid.GriddedVariable._netcdf_datetime_convert(time)


    time_invalid = ncf_data['time_invalid']
    cmp_dates = lmrgrid.GriddedVariable._netcdf_datetime_convert(time_invalid)

    for orig, compare in zip(dates, cmp_dates):
        assert orig.year == compare.year
        assert orig.month == compare.month


@pytest.mark.xfail(raises=ValueError)
def test_griddedvar_timeavg_year_determine_fail(ncf_dates, ncf_temps):

    time_yrs, avg_data = \
        lmrgrid.GriddedVariable._time_avg_gridded_to_resolution(ncf_dates[:8],
                                                                ncf_temps,
                                                                1.0,
                                                                0)

@pytest.mark.xfail(raises=ValueError)
def test_griddedvar_timeavg_not_divisible_res(ncf_dates, ncf_temps):

    time_yrs, avg_data = \
        lmrgrid.GriddedVariable._time_avg_gridded_to_resolution(ncf_dates,
                                                                ncf_temps,
                                                                1.2,
                                                                0)


@pytest.mark.parametrize('resolution', [0.25, 0.5, 1.0, 2.0])
def test_griddedvar_timeavg_variable_resolution(resolution, ncf_dates, ncf_temps):

    time_yrs, avg_data = \
        lmrgrid.GriddedVariable._time_avg_gridded_to_resolution(ncf_dates,
                                                                ncf_temps,
                                                                resolution,
                                                                0)
    num_yrs = 4
    n_units = int(num_yrs/resolution)

    assert len(time_yrs) == n_units
    assert len(avg_data) == n_units

    start_year = ncf_dates[0].year
    yrs = np.array([start_year + i*resolution for i in xrange(n_units)])
    np.testing.assert_array_equal(yrs, time_yrs)

    dat_shp = ncf_temps.shape
    avg_cmp = ncf_temps.reshape(n_units, dat_shp[0]/n_units, *dat_shp[1:])
    avg_cmp = avg_cmp.mean(axis=1)
    np.testing.assert_array_equal(avg_data, avg_cmp)


@pytest.mark.parametrize('yr_shift', [0, 3, 6, 9, 12, 15])
def test_griddedvar_timeavg_variable_yrshift(yr_shift, ncf_dates, ncf_temps):

    resolution = 0.5
    time_yrs, avg_data = \
        lmrgrid.GriddedVariable._time_avg_gridded_to_resolution(ncf_dates,
                                                                ncf_temps,
                                                                resolution,
                                                                yr_shift)

    units_per_res = 6
    res_unit_per_yr = 2

    assert len(avg_data) % res_unit_per_yr == 0
    mod_yr_shift = yr_shift % 12
    cmp_dat = ncf_temps[mod_yr_shift:(mod_yr_shift+units_per_res)].mean(axis=0)
    np.testing.assert_array_equal(avg_data[0], cmp_dat)

    if yr_shift % 12 != 0:
        assert len(time_yrs) == 3*res_unit_per_yr
        assert len(avg_data) == 3*res_unit_per_yr


@pytest.mark.parametrize('resolution', [0.5, 1.0, 2.0])
def test_priorvar_timeavg_anomaly(resolution, ncf_dates, ncf_temps):

    _, avg_data = \
        lmrgrid.GriddedVariable._time_avg_gridded_to_resolution(ncf_dates,
                                                                ncf_temps,
                                                                resolution)

    time_yrs, anom_avg_data = \
        lmrgrid.PriorVariable._time_avg_gridded_to_resolution(ncf_dates,
                                                              ncf_temps,
                                                              resolution)

    step = np.ceil(1/resolution)
    anom0 = avg_data[0] - avg_data[0::step].mean(axis=0)

    np.testing.assert_array_equal(anom_avg_data[0], anom0)


def test_priorvar_subannual_res_separation(ncf_data):

    dat = ncf_data.variables
    lat = dat['lat']
    lon = dat['lon']
    time = dat['time']
    time = ncf.num2date(time[:], time.units)
    data = dat['air']

    resolution = 0.5

    time_yrs, anom_avg_data = \
        lmrgrid.PriorVariable._time_avg_gridded_to_resolution(time,
                                                              data[:],
                                                              resolution)

    prior_var = lmrgrid.PriorVariable('air', data.dimensions,
                                      anom_avg_data, resolution, time=time_yrs,
                                      lat=lat[:], lon=lon[:])

    new_dat, new_time = \
        lmrgrid.GriddedVariable._subannual_decomp(prior_var.data,
                                                  prior_var.time,
                                                  resolution)

    for i, (dat, time) in enumerate(izip(new_dat, new_time)):
        np.testing.assert_array_equal(time_yrs[i::2], time)
        np.testing.assert_array_equal(anom_avg_data[i::2], dat)


@pytest.mark.parametrize('varname, savefile',
                         [pytest.mark.xfail(raises=ValueError)(('notime',
                                                                False)),
                          pytest.mark.xfail(raises=ValueError)(('outoforder',
                                                                False)),
                          ('air', True),
                          ('alternatedims', False),
                          ('vertical_use_this', False),
                          ('tseries', False)])
def test_priorvar_load_from_netcdf(varname, savefile):

    dirname = '/home/disk/p/wperkins/Research/LMR/tests/data/'
    filename = 'gridded_dat.nc'
    resolution = 1.0
    yr_shift = 0
    pre_avg_fname = '.pre_avg_{}_res{:02.1f}'.format(varname,
                                                     resolution)
    prior_obj = lmrgrid.PriorVariable._load_from_netcdf(dirname,
                                                        filename,
                                                        varname,
                                                        resolution,
                                                        save=savefile)[0]

    assert len(prior_obj.data) == 4
    if savefile:
        assert os.path.exists(dirname+filename+pre_avg_fname+'.h5')

    if varname == 'tseries':
        assert prior_obj.dim_order == ['time']
    elif varname == 'vertical_use_this':
        assert prior_obj.dim_order == ['time', 'lev', 'lat']
    else:
        assert prior_obj.dim_order == ['time', 'lat', 'lon']


def test_priorvar_load_from_netcdf_truncate():
    dirname = '/home/disk/p/wperkins/Research/LMR/tests/data/'
    filename = 'gridded_dat.nc'
    resolution = 1.0
    yr_shift = 0
    varname = 'air'
    pre_avg_fname = '.pre_avg_{}_res{:02.1f}'.format(varname,
                                                     resolution)
    prior_obj = lmrgrid.PriorVariable._load_from_netcdf(dirname,
                                                        filename,
                                                        varname,
                                                        resolution,
                                                        truncate=True,
                                                        save=True)[0]

    assert len(prior_obj.time) == 4
    assert prior_obj.data.shape == (4, 44, 66)
    assert os.path.exists(os.path.join(dirname,
                                       filename+pre_avg_fname+'.trnc.h5'))


@pytest.mark.parametrize('varname, trunc',
                         [('air', False),
                          ('air', True)])
def test_priorvar_load_preavg_data(varname, trunc):
    dirname = '/home/disk/p/wperkins/Research/LMR/tests/data/'
    filename = 'gridded_dat.nc'
    resolution = 1.0
    yr_shift = 0
    pre_avg_fname = '.pre_avg_{}_res{:02.1f}'.format(varname,
                                                     resolution)

    if trunc:
        pre_avg_fname += '.trnc'

    if os.path.exists(dirname + filename + pre_avg_fname + '.h5'):
        os.remove(dirname + filename + pre_avg_fname + '.h5')

    ncf_prior = lmrgrid.PriorVariable._load_from_netcdf(dirname,
                                                        filename,
                                                        varname,
                                                        resolution,
                                                        truncate=trunc,
                                                        save=True)[0]

    pre_avg_prior = lmrgrid.PriorVariable._load_pre_avg_obj(dirname,
                                                            filename,
                                                            varname,
                                                            resolution,
                                                            truncate=trunc)[0]

    np.testing.assert_array_equal(ncf_prior.data, pre_avg_prior.data)
    np.testing.assert_array_equal(ncf_prior.lat, pre_avg_prior.lat)
    np.testing.assert_array_equal(ncf_prior.lon, pre_avg_prior.lon)
    np.testing.assert_array_equal(ncf_prior.time, pre_avg_prior.time)


def test_priorvar_load_preavg_full_exist_no_truncated():
    dirname = '/home/disk/p/wperkins/Research/LMR/tests/data/'
    filename = 'gridded_dat.nc'
    resolution = 1.0
    yr_shift = 0
    varname = 'air'
    pre_avg_fname = '.pre_avg_{}_res{:02.1f}'.format(varname,
                                                     resolution)
    if os.path.exists(dirname + filename + pre_avg_fname + '.trnc'):
        os.remove(dirname + filename + pre_avg_fname + '.trnc')

    ncf_prior = lmrgrid.PriorVariable._load_from_netcdf(dirname,
                                                        filename,
                                                        varname,
                                                        resolution,
                                                        truncate=True,
                                                        save=False)[0]

    pre_avg_prior = lmrgrid.PriorVariable._load_pre_avg_obj(dirname,
                                                            filename,
                                                            varname,
                                                            resolution,
                                                            truncate=True)[0]

    np.testing.assert_array_equal(ncf_prior.data, pre_avg_prior.data)
    np.testing.assert_array_equal(ncf_prior.lat, pre_avg_prior.lat)
    np.testing.assert_array_equal(ncf_prior.lon, pre_avg_prior.lon)
    np.testing.assert_array_equal(ncf_prior.time, pre_avg_prior.time)
    assert os.path.exists(dirname + filename + pre_avg_fname + '.trnc.h5')


@pytest.mark.xfail(raises=IOError)
def test_priorvar_load_preavg_not_exist():
    dirname = '/home/disk/p/wperkins/Research/LMR/tests/data/'
    filename = 'gridded_dat.nc'
    resolution = 1.0
    yr_shift = 0
    varname = 'not_a_var'
    pre_avg_prior = lmrgrid.PriorVariable._load_pre_avg_obj(dirname,
                                                            filename,
                                                            varname,
                                                            resolution,
                                                            truncate=True)[0]


@pytest.mark.parametrize('res', [0.5, 1.])
def test_state(res):
    import test_config

    test_config.core.assimilation_time_res = [res]
    dirname = test_config.prior.datadir_prior
    filename = test_config.prior.datafile_prior
    sample_idxs = test_config.prior.prior_sample_idx

    test_config.core.sub_base_res = res
    state_obj = lmrgrid.State.from_config(test_config)

    num_priors = int(np.ceil(1/res))

    assert len(state_obj.state_list) == num_priors

    for var in test_config.prior.state_variables:
        fname = filename.replace('[vardef_template]', var)
        pvars = lmrgrid.PriorVariable._load_from_netcdf(dirname,
                                                        fname,
                                                        var,
                                                        res,
                                                        sample=sample_idxs)
        for i in range(num_priors):
            prior_var = pvars[i]
            np.testing.assert_array_equal(state_obj.get_var_data(var, idx=i),
                prior_var.flattened_spatial()[0].T)



if __name__ == '__main__':

    # with ncf.Dataset('/home/disk/p/wperkins/Research/LMR/tests/data/gridded_dat.nc', 'r') as f:
    #     time = f.variables['time']
    #     time = ncf.num2date(time[:], time.units)
    #     data = f.variables['air'][:]
        #test_priorvar_subannual_res_separation(f)

        # import test_config
        # test_state(f, 1.0)
    #test_griddedvar_timeavg_variable_yrshift(3, time, data)

    #test_priorvar_load_preavg_data('air', True)
    test_state(0.5)

