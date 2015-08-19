"""
A module containing classes and methods for gridded data

Author: Andre
Adapted from load_gridded_data, LMR_prior, LMR_calibrate
"""

from abc import abstractmethod, ABCMeta
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np
import os
from copy import deepcopy
from os.path import join
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
        self.lev = lev
        self.lat = lat
        self.lon = lon

        self._dim_coord_map = {_TIME: self.time,
                               _LEV: self.lev,
                               _LAT: self.lat,
                               _LON: self.lon}

        # Make sure ndimensions specified match data
        if self.ndim != len(self.data.shape):
            raise ValueError('Number of dimensions given do not match data'
                             ' dimensions.')

        # Make sure each dimension has consistent number of values as data
        for i, dim in enumerate(self.dim_order):
            if self._dim_coord_map[dim] is None:
                raise ValueError('Dimension specified but no values provided '
                                 'for initialization')

            if data.shape[i] != len(self._dim_coord_map[dim]):
                raise ValueError('Dimension values provided do not match in '
                                 'length with dimension axis of data')

        # Right now it shouldn't happen for loaders, may change in future
        if time is not None:
            self.nsamples = len(self.time)
        else:
            self.nsamples = 1
            self.dim_order.insert(_TIME, 0)
            self.data = self.data.reshape(1, *self.data.shape)

        self._space_dims = [dim for dim in dims_ordered if dim != _TIME]
        self.space_shp = [len(self._dim_coord_map[dim])
                           for dim in self._space_dims]

        if len(self._space_dims) > 2:
            raise ValueError('Class cannot handle 3D data yet!')

        if not self._space_dims:
            self.type = 'timeseries'
            self.space_shp = [1]
            self.data = self.data.reshape(self.nsamples, 1)
        elif _LAT in self._space_dims and _LON in self._space_dims:
            self.type = 'horizontal'
        elif _LAT in self._space_dims and _LEV in self._space_dims:
            self.type = 'meridional_vertical'
        else:
            raise ValueError('Unrecognized dimension combination.')

    def save(self, filename):
        with open(filename, 'w') as f:
            cPickle.dump(self, f)

    def truncate(self, ntrunc=42):

        """
        Return a new truncated version of the gridded object.  Only works
        on horizontal data for now.
        """
        assert self.type == 'horizontal'
        class_obj = type(self)

        regrid_data, new_lat, new_lon = regrid_sphere2(self, ntrunc)
        return class_obj(self.name, self.dim_order, regrid_data,
                         self.resolution,
                         time=self.time,
                         lev=self.lev,
                         lat=new_lat[:, 0],
                         lon=new_lon[0])

    def flattened_spatial(self):

        flat_data = self.data.reshape(len(self.time),
                                      np.product(self.space_shp))

        # Get dimensions of data
        coords = [self._dim_coord_map[key] for key in self._space_dims]
        grids = np.meshgrid(*coords, indexing='ij')
        flat_coords = {dim: grid.flatten()
                       for dim, grid in zip(self._space_dims, grids)}

        return flat_data, flat_coords

    def sample(self, nens, seed=None):
        """
        Random sample ensemble of current gridded variable
        """

        cls = type(self)

        if seed:
            random.seed(seed)

        ind_ens = random.sample(range(len(self.time)), nens)
        time_sample = np.array(self.time)[ind_ens]

        data_sample = np.zeros([nens] + list(self.data.shape[1:]))
        for k, idx in enumerate(ind_ens):
            data_sample[k] = self.data[idx]

        return cls(self.name, self.dim_order, data_sample, self.resolution,
                   time=time_sample,
                   lev=self.lev,
                   lat=self.lat,
                   lon=self.lon)

    @abstractmethod
    def load(cls, config, *args):
        pass

    @classmethod
    def get_loader_for_filetype(cls, file_type):
        ftype_map = {_ftypes.netcdf: cls._load_from_netcdf}
        return ftype_map[file_type]

    @classmethod
    def _load_pre_avg_obj(cls, dir_name, filename, varname, resolution,
                          yr_shift, truncate=False):

        # Check if pre-processed averages file exists
        pre_avg_tag = '.pre_avg_{}_res{:02.1f}_shift{:d}'.format(varname,
                                                                 resolution,
                                                                 yr_shift)
        trunc_tag = '.trnc'

        path = join(dir_name, filename + pre_avg_tag)

        if truncate:
            path += trunc_tag

        # Look for pre_averaged_file
        if os.path.exists(path):

            with open(path, 'r') as f:
                prior_obj = cPickle.load(f)

        elif truncate and os.path.exists(path.strip(trunc_tag)):
            # If truncate requested and truncate not found look for
            # pre-averaged full version
            with open(path.strip(trunc_tag), 'r') as f:
                prior_obj = cPickle.load(f)

            prior_obj = prior_obj.truncate()
            prior_obj.save(path)
        else:
            raise IOError('No pre-averaged file found for given specifications')

        return prior_obj

    @classmethod
    def _load_from_netcdf(cls, dir_name, filename, varname, resolution,
                          yr_shift, truncate=False, save=False):

        # Check if pre-processed averages file exists
        pre_avg_name = '.pre_avg_{}_res{:02.1f}_shift{:d}'

        with Dataset(join(dir_name, filename), 'r') as f:
            ncf_varname = varname.split('_')[0]
            var = f.variables[ncf_varname]
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
            if idx_order != sorted(idx_order):
                raise ValueError('Input file dimensions do not match default'
                                 ' ordering.')

            # Load dimension values
            dim_vals = {dim_name: f.variables[dim_key]
                        for dim_name, dim_key in zip(dims, var.dimensions)
                        if dim_key not in dim_exclude}

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

            # Filename for saving pre-averaged pickle
            new_fname = join(dir_name,
                             filename + pre_avg_name.format(varname,
                                                            resolution,
                                                            yr_shift))

            # Create GriddedVariable object
            prior_obj = cls(varname, dims, avg_data, resolution, **dim_vals)
            if save:
                prior_obj.save(new_fname)

            if truncate:
                try:
                    prior_obj = prior_obj.truncate()
                    if save:
                        prior_obj.save(new_fname + '.trnc')
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
        if not hasattr(time_var, 'calendar'):
            cal = 'standard'
        else:
            cal = time_var.calendar

        try:
            time = num2date(time_var[:], units=time_var.units,
                                calendar=cal)
            return time
        except ValueError:
            # num2date needs calendar year start >= 0001 C.E. (bug submitted
            # to unidata about this
            # TODO: Add a warning about likely inaccuracy day res and smaller
            tunits = time_var.units
            since_yr_idx = tunits.index('since ') + 6
            year = int(tunits[since_yr_idx:since_yr_idx+4])
            year_diff = year - 0001

            new_units = tunits[:since_yr_idx] + '0001-01-01 00:00:00'
            time = num2date(time_var[:], new_units, calendar=cal)
            reshifted_time = [datetime(d.year + year_diff, d.month, d.day,
                                       d.hour, d.minute, d.second)
                              for d in time]
            return np.array(reshifted_time)

    @staticmethod
    def _time_avg_gridded_to_resolution(time_vals, data, resolution,
                                        yr_shift=0):
        """
        Converts to time units of years at specified resolution and shift
        :param time_vals:
        :param data:
        :param resolution:
        :param yr_shift:
        :return:
        """

        # Calculate number of elements in 1 resolution unit
        start = time_vals[yr_shift]
        start = datetime(start.year, start.month, start.day)
        end = start.replace(year=start.year+1)

        for elem_in_yr, dt_obj in enumerate(time_vals[yr_shift:]):
            # NetCDF.datetime to regular datetime for equivalence check
            if datetime(dt_obj.year, dt_obj.month, dt_obj.day) == end:
                break
        else:
            raise ValueError('Could not determine number of elements in a '
                             'single year')
        yr_shift %= elem_in_yr  # shouldn't have yr_shift larger then elem_in_yr
        end_cutoff = -((elem_in_yr - yr_shift) % elem_in_yr)
        if end_cutoff == 0:
            end_cutoff = None

        # Find number of units in new resolution
        nelem_in_unit_res = resolution * elem_in_yr
        if not nelem_in_unit_res.is_integer():
            raise ValueError('Elements in yr not evenly divisible by given '
                             'resolution')
        tot_units = int(len(time_vals[yr_shift:end_cutoff]) /
                        nelem_in_unit_res)
        spatial_shp = data.shape[1:]

        # Average data and create year list
        avg_data = data[yr_shift:end_cutoff].reshape(tot_units,
                                                     nelem_in_unit_res,
                                                     *spatial_shp)
        avg_data = np.nanmean(avg_data, axis=1)

        start_yr = start.year
        time_yrs = [start_yr + i*resolution
                    for i in xrange(tot_units)]

        return np.array(time_yrs), avg_data


class PriorVariable(GriddedVariable):

    @classmethod
    def load(cls, config, varname):
        file_dir = config.prior.datadir_prior
        file_name = config.prior.datafile_prior
        file_type = config.prior.dataformat_prior
        base_resolution = config.core.assimilation_time_res[0]
        yr_shift = config.core.year_start_idx_shift
        nens = config.core.nens
        seed = config.core.seed

        try:
            ftype_loader = cls.get_loader_for_filetype(file_type)
        except KeyError:
            raise TypeError('Specified file type not supported yet.')

        fname = file_name.replace('[vardef_template]', varname)

        try:
            var_obj = cls._load_pre_avg_obj(file_dir, fname, varname,
                                            base_resolution, yr_shift)
            print 'Loaded pre-averaged file.'
        except IOError:
            print 'No pre-averaged file found... Loading directly from file.'
            var_obj = ftype_loader(file_dir, fname, varname, base_resolution,
                                   yr_shift, save=True)

        if base_resolution < 1.0:
            var_objs = var_obj.subannual_resolution_prior()
        else:
            var_objs = [var_obj]

        # Sample from loaded data if desired
        if nens:
            var_objs = [obj.sample(nens, seed=seed)
                        for obj in var_objs]

        return var_objs

    @classmethod
    def load_allvars(cls, config):
        var_names = config.prior.state_variables

        prior_dict = OrderedDict()
        for vname in var_names:
            prior_dict[vname] = cls.load(config, vname)

        return prior_dict

    # TODO: This might not work for removing anomaly
    @staticmethod
    def _time_avg_gridded_to_resolution(time_vals, data, resolution,
                                        yr_shift=0):

        # Call Base class to get correct time average
        time, avg_data = \
            super(PriorVariable, PriorVariable).\
            _time_avg_gridded_to_resolution(time_vals, data, resolution,
                                            yr_shift)
        # Calculate anomaly
        if resolution < 1:
            units_in_yr = 1/resolution
            old_shp = list(avg_data.shape)
            new_shape = [old_shp[0]/units_in_yr, units_in_yr] + old_shp[1:]
            avg_data = avg_data.reshape(new_shape)
            anom_data = avg_data - np.nanmean(avg_data, axis=0)
            anom_data = anom_data.reshape(old_shp)
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

        num_priors = int(np.ceil(1./self.resolution))
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

    def __init__(self, prior_vars, base_res):

        self._prior_vars = prior_vars
        self._base_res = base_res
        self.state_list = []
        self.var_coords = {}
        self.var_view_range = {}
        self.var_space_shp = {}
        self.augmented = False

        self.len_state = 0
        for var, pobjs in prior_vars.iteritems():

            self.var_space_shp[var] = pobjs[0].space_shp

            var_start = self.len_state
            for i, pobj in enumerate(pobjs):

                flat_data, flat_coords = pobj.flattened_spatial()

                # Store range of data in state dimension
                if i == 0:
                    var_end = flat_data.T.shape[0] + var_start
                    self.var_view_range[var] = (var_start, var_end)
                    self.len_state += var_end

                # Add prior to state vector, transposed to make state the first
                # dimension, and ensemble members along the 2nd
                try:
                    self.state_list[i] = \
                        np.concatenate((self.state_list[i], flat_data.T),
                                       axis=0)
                except IndexError:
                    self.state_list.append(flat_data.T)

            # Save variable view information
            self.var_coords[var] = flat_coords

        self.shape = self.state_list[0].shape
        self.old_state_info = self.get_old_state_info()

    @classmethod
    def from_config(cls, config):
        pvars = PriorVariable.load_allvars(config)
        base_res = config.core.assimilation_time_res[0]

        return cls(pvars, base_res)

    def get_var_data(self, var_name, idx=None):
        """
        Returns a view (or a copy) of the variable in the state vector
        """
        # TODO: change idx to optional, if none does average
        # probably switch statelist to numpy array for easy averaging.
        start, end = self.var_view_range[var_name]

        if idx is not None:
            var_data = self.state_list[idx][start:end]
        else:
            var_data = [dat[start:end] for dat in self.state_list]

        return var_data

    def truncate_state(self):
        """
        Create a truncated copy of the current state
        """
        trunc_pvars = OrderedDict()
        for var_name, pvar in self._prior_vars.iteritems():
            trunc_pvars[var_name] = [pobj.truncate() for pobj in pvar]
        state_class = type(self)
        return state_class(trunc_pvars, self._base_res)

    def copy_state(self):

        return deepcopy(self)

    def augment_state(self, ye_vals):

        aug_state_list = []
        for i, state in enumerate(self.state_list):
            aug_state_list.append(np.append(state, ye_vals, axis=0))

        self.state_list = aug_state_list
        self.augmented = True

        self.var_view_range['state'] = (0, self.len_state)
        self.var_view_range['ye_vals'] = (self.len_state,
                                          self.len_state + len(ye_vals))

    def annual_avg(self, var_name=None):

        if var_name is None:
            subannual_data = np.array(self.state_list)
        else:
            # Get variable data for each subannual state vector, creat ndarray
            subannual_data = np.array(self.get_var_data(var_name))

        avg = subannual_data.mean(axis=0)
        return avg

    def avg_to_res(self, res):

        if res == self._base_res:
            return

        if res < 1:
            chunk = int(res / self._base_res)
            new_state = [self.state_list[i:i+chunk]
                         for i in xrange(0, len(self.state_list), chunk)]
            new_state = [np.array(state_group).mean(axis=0)
                         for state_group in new_state]
            self.state_list = new_state
        elif res == 1:
            new_state = np.array(self.state_list).mean(axis=0)
            self.state_list = [new_state]
        else:
            raise ValueError('Cannot handle resolutions larger than 1 yet.')

    def replace_subannual_avg(self, avg):
        data = np.array(self.state_list)
        data_mean = self.annual_avg()
        self.state_list = data - data_mean + avg

    def get_old_state_info(self):

        state_info = {}
        for var in self.var_view_range.keys():
            var_info = {'pos': self.var_view_range[var]}

            space_dims = [dim for dim in _DEFAULT_DIM_ORDER
                          if dim in self.var_coords[var].keys()]
            var_info['spacecoords'] = space_dims
            var_info['spacedims'] = self.var_space_shp[var]
            state_info[var] = var_info

        return state_info




