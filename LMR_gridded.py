"""
A module containing classes and methods for gridded data

Author: Andre
Adapted from load_gridded_data, LMR_prior, LMR_calibrate
"""

from abc import abstractmethod, ABCMeta
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
import numpy as np
import os.path
import cPickle
import random

from LMR_config import constants
from LMR_utils2 import regrid_sphere2
_LAT = 'lat'
_LON = 'lon'
_LEV = 'lev'
_TIME = 'time'

_DEFAULT_DIM_ORDER = [_TIME, _LEV, _LAT, _LON]
_ALT_DIMENSION_DEFS = {'latitude': _LAT,
                       'longitude': _LON,
                       'plev': _LEV}

_ftypes = constants.file_types


class GriddedVariable(object):

    __metaclass__ = ABCMeta

    def __init__(self, name, dims_ordered, data, resolution, time=None,
                 lev=None, lat=None, lon=None):
        self.name = name
        self.dim_order = dims_ordered
        self.ndim = len(dims_ordered)
        self.data = data
        self.resolution = resolution
        self.time = time
        self.nsamples = len(self.time)
        self.lev = lev
        self.lat = lat
        self.lon = lon

        self._space_dims = [dim for dim in dims_ordered if dim != _TIME]
        self._dim_coord_map = {_TIME: self.time,
                               _LEV: self.lev,
                               _LAT: self.lat,
                               _LON: self.lon}
        self._space_shp = [len(self._dim_coord_map[dim])
                           for dim in self._space_dims]

        if len(self._space_dims) > 2:
            raise ValueError('Class cannot handle 3D data yet!')

        if not self._space_dims:
            self.type = 'timeseries'
            self._space_shp = [1]
            self.data = self.data.reshape(self.nsamples, 1)
        elif _LAT in self._space_dims and _LON in self._space_dims:
            self.type = 'horizontal'
        elif _LAT in self._space_dims and _LEV in self._space_dims:
            self.type = 'meridional_vertical'
        else:
            raise ValueError('Unrecognized dimension combination.')

    @classmethod
    @abstractmethod
    def load(cls, config, *args):
        pass

    def save(self, filename):
        cPickle.dump(self, filename)

    def truncate(self, trunc_type='T42'):

        """
        Return a new truncated version of the gridded object.  Only works
        on horizontal data for now.
        """
        assert self.type == 'horizontal'
        class_obj = type(self)

        regrid_data, new_lat, new_lon = regrid_sphere2(self, 'T42')
        return class_obj(self.name, self.dim_order, regrid_data,
                         self.resolution,
                         time=self.time,
                         lev=self.lev,
                         lat=new_lat,
                         lon=new_lon)

    def flattened_spatial(self):

        flat_data = self.data.reshape(len(self.time),
                                      np.product(self._space_shp))

        # Get dimensions of data
        coords = [self._dim_coord_map[key] for key in self._space_dims]
        grids = np.meshgrid(*coords, indexing='ij')
        flat_coords = {dim: grid.flatten()
                       for dim, grid in zip(self._space_dims, grids)}

        return flat_data, flat_coords

    @classmethod
    def get_loader_for_filetype(cls, file_type):
        ftype_map = {_ftypes.netcdf: cls._load_from_netcdf}
        return ftype_map[file_type]

    def sample(self, nens, seed=None):
        """
        Random sample ensemble of the
        :param nens:
        :param seed:
        :return:
        """

        if seed:
            random.seed(seed)

        ind_ens = random.sample(range(len(self.time)), nens)
        self.time = np.array(self.time)[ind_ens]
        self.data = self.data[ind_ens]
        self.nsamples = len(self.time)

    @classmethod
    def _load_pre_avg_obj(cls, dir_name, filename, resolution,
                          yr_shift, truncate=False):

        # Check if pre-processed averages file exists
        pre_avg_tag = '.pre_avg_res{:02.1f}_shift{:d}'.format(resolution,
                                                              yr_shift)
        trunc_name = pre_avg_tag + '.trnc'

        if truncate:
            pre_avg_name = pre_avg_tag + trunc_name
        else:
            pre_avg_name = pre_avg_tag

        # Look for pre_averaged_file
        if os.path.exists(dir_name + filename + pre_avg_name):
            filename += pre_avg_name

            with open(filename, 'r') as f:
                prior_obj = cPickle.loads(f)

        elif truncate and os.path.exists(dir_name + filename +
                                         pre_avg_tag):
            # If truncate requested and truncate not found look for
            # pre-averaged full version
            filename += pre_avg_tag
            with open(filename, 'r') as f:
                prior_obj = cPickle.load(f)

            prior_obj = prior_obj.truncate()
            prior_obj.save()
        else:
            prior_obj = None

        return prior_obj

    @classmethod
    def _load_from_netcdf(cls, dir_name, filename, varname, resolution,
                          yr_shift, truncate=False):

        # Check if pre-processed averages file exists
        pre_avg_name = '.pre_avg_res{:02.1f}_shift{:d}'

        with Dataset(dir_name+filename, 'r') as f:
            var = f.variables[varname]
            data_shp = var.shape

            # Convert to key names defined in _DEFAULT_DIM_ORDER
            dims = []
            dim_exclude = []
            for i, dim in enumerate(var.dimensions):
                if data_shp[i] == 1:
                    dim_exclude.append(dim)
                elif dim in _DEFAULT_DIM_ORDER:
                    dims.append(dim.lower())
                else:
                    dims.append(_ALT_DIMENSION_DEFS[dim.lower()])

            # Make sure it has time dimension
            if _TIME not in dims:
                raise ValueError('No time dimension for specified prior data.')

            # Check order of all dimensions
            idx_order = [_DEFAULT_DIM_ORDER.index(dim) for dim in dims]
            if idx_order != idx_order.sort():
                raise ValueError('Input file dimensions do not match default'
                                 ' ordering.')

            # Load dimension values
            dim_vals = {dim_name: f.variables[dim_key]
                        for dim_name, dim_key in zip(dims, var.dimensions)
                        if not dim_key in dim_exclude}

            # Convert time to datetimes
            dim_vals[_TIME] = cls._netcdf_datetime_convert(dim_vals[_TIME])

            # Extract data for each dimension
            dim_vals = {k: val[:] for k, val in dim_vals.iteritems()}

            # Average to correct time resolution
            dim_vals[_TIME], avg_data = \
                cls._time_avg_gridded_to_resolution(dim_vals[_TIME],
                                                    var[:],
                                                    resolution,
                                                    yr_shift)

            # TODO: Replace with logger statement
            print (varname, ' res ', resolution, ': Global: mean=',
                   np.nanmean(avg_data),
                   ' , std-dev=', np.nanstd(avg_data))

            prior_obj = cls(varname, dims, avg_data, resolution, **dim_vals)
            prior_obj.save(pre_avg_name.format(resolution, yr_shift))

            if truncate:
                try:
                    prior_obj = prior_obj.truncate()
                    prior_obj.save(pre_avg_name.format(resolution, yr_shift) +
                                   '.trnc')
                except AssertionError:
                    pass

            return prior_obj

    @staticmethod
    def _netcdf_datetime_convert(time_var):
        """
        Converts netcdf time variable into date-times.

        Used as a static method in case necesary to overwrite with subclass
        :param time_var:
        :return:
        """
        try:
            time = num2date(time_var[:], units=time_var.units,
                                calendar=time_var.calendar)
            return time.tolist()
        except ValueError:
            # num2date needs calendar year start >= 0001 C.E. (bug submitted
            # to unidata about this
            tunits = time_var.units
            since_yr_idx = tunits.index('since ') + 6
            year = int(tunits[since_yr_idx:since_yr_idx+4])
            year_diff = year - 0001

            new_units = tunits[:since_yr_idx] + '0001-01-01 00:00:00'
            time = num2date(time_var[:], new_units, calendar=time_var.calendar)
            return [datetime(d.year + year_diff, d.month, d.day,
                             d.hour, d.minute, d.second)
                    for d in time]

    @staticmethod
    def _time_avg_gridded_to_resolution(time_vals, data, resolution, yr_shift):

        # Calculate number of elements in 1 resolution unit
        start = time_vals[yr_shift].date()
        end = start.replace(year=start.year+1)

        for i, dt_obj in enumerate(time_vals[yr_shift:]):
            if dt_obj.date() == end:
                break
        else:
            raise ValueError('Could not determine number of elements in a '
                             'single year')
        i += 1  # shift from index values to actual values

        if resolution % i != 0:
            raise ValueError('Elements in yr not evenly divisible by given '
                             'resolution')
        else:
            nelem_in_unit_res = resolution / i

        # Find cutoff to keep only full years in data
        end_cutoff = -(len(time_vals[yr_shift:]) % nelem_in_unit_res)

        # Find number of units in new resolution
        tot_units = len(time_vals[yr_shift:end_cutoff]) / nelem_in_unit_res
        spatial_shp = data.shape[1:]

        # Average data and create year list
        avg_data = data[yr_shift:end_cutoff].reshape(tot_units,
                                                     nelem_in_unit_res,
                                                     *spatial_shp)
        avg_data = np.nanmean(avg_data, axis=1)

        start_yr = start.year
        time_yrs = [start_yr + i*resolution for i in xrange(tot_units)]

        return time_yrs, avg_data


class PriorVariable(GriddedVariable):

    @classmethod
    def load(cls, config, varname):
        file_dir = config.prior.datadir_prior
        file_name = config.prior.datafile_prior
        file_type = config.prior.dataformat_prior
        truncate = config.prior.truncate
        base_resolution = config.core.assimilation_time_res[0]
        yr_shift = config.core.year_start_idx_shift
        nens = config.core.nens
        seed = config.core.seed

        try:
            ftype_loader = cls.get_loader_for_filetype(file_type)
        except KeyError:
            raise TypeError('Specified file type not supported yet.')

        fname = file_name.replace('[vardef_template]', varname)
        var_obj = cls._load_pre_avg_obj(file_dir, fname, varname,
                                            base_resolution, yr_shift)
        if not var_obj:
            var_obj = ftype_loader(file_dir, fname, varname, base_resolution,
                                   yr_shift)

        # Sample from loaded data if desired
        if nens:
            var_obj.sample(nens, seed=seed)

        return var_obj

    @classmethod
    def load_allvars(cls, config):
        var_names = config.prior.state_variables

        prior_dict = {}
        for vname in var_names:
            prior_dict[vname] = cls.load(config, vname)

        return prior_dict

    # TODO: This might not work for removing anomaly
    @staticmethod
    def _time_avg_gridded_to_resolution(time_vals, data, resolution, yr_shift):

        # Call Base class to get correct time average
        time, avg_data = \
            super(PriorVariable, PriorVariable).\
            _time_avg_gridded_to_resolution(time_vals, data, resolution,
                                            yr_shift)
        # Calculate anomaly
        if resolution < 1:
            units_in_yr = 1/resolution
            old_shp = avg_data.shape
            new_shape = [old_shp[0]/units_in_yr, units_in_yr] + old_shp[1:]
            avg_data = avg_data.reshape(new_shape)
            anom_data = avg_data - np.nanmean(avg_data, axis=0)
            anom_data.reshape(old_shp)
        else:
            anom_data = avg_data - np.nanmean(avg_data, axis=0)

        # TODO: Replace with logger statement
        print ('Removing the temporal mean (for every gridpoint) from the '
               'prior...')
        return time, anom_data

    def subannual_resolution_prior(self):
        """
        Divides sub-annual resolution priors into a set of PriorVar objects
        for each season
        :return:
        """

        num_priors = np.ciel(1./self.resolution)
        class_obj = type(self)
        seasonal_priors = []
        for i in xrange(num_priors):
            new_prior = class_obj(self.name,
                                  self.dim_order,
                                  self.data[i::num_priors],
                                  self.resolution,
                                  time=self.time[i::num_priors],
                                  lev=self.lev,
                                  lat=self.lat,
                                  lon=self.lon)
            seasonal_priors.append(new_prior)

        return seasonal_priors


class State(object):
    """
    Class to create state vector and information
    """

    def __init__(self, config):

        prior_vars = PriorVariable.load_allvars(config)
        base_res = config.core.assimilation_time_res[0]

        self.state_list = []
        self.var_coords = {}
        self.var_view_range = {}

        len_state = 0
        for var, prior_obj in prior_vars.iteritems():
            # If sub-annual split up into seasons, multiple state vectors
            if base_res < 1:
                pobjs = prior_obj.subannual_resolution_prior()
            else:
                pobjs = [prior_obj]

            var_start = len_state
            for i, pobj in enumerate(pobjs):

                # Store range of data in state dimension
                flat_data, flat_coords = pobj.flattened_spatial()
                var_end = flat_data.shape[0] + var_start
                self.var_view_range[var] = (var_start, var_end)
                var_start += flat_data.shape[0]

                # Add prior to state vector, transposed to make state the first
                # dimension, and ensemble members along the 2nd
                try:
                    np.concatenate((self.state_list[i], flat_data.T), axis=0)
                except IndexError:
                    self.state_list.append(flat_data.T)

            # Save variable view information
            self.var_coords[var] = flat_coords

    def get_var_data_view(self, idx, var_name):
        """
        Returns a view (NOT A COPY) of the variable in the state vector
        """
        start, end = self.var_view_range[var_name]

        return self.state_list[idx][start:end]
