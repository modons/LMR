"""
A module containing classes and methods for gridded data

Author: Andre
Adapted from load_gridded_data, LMR_prior, LMR_calibrate
"""

from abc import abstractmethod, ABCMeta
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from collections import OrderedDict
from itertools import izip
import numpy as np
import os
from copy import deepcopy
from os.path import join
import cPickle
import random
import tables as tb

from LMR_config import constants
from LMR_utils2 import regrid_sphere2, var_to_hdf5_carray, empty_hdf5_carray
from LMR_utils2 import fix_lon
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
                 lev=None, lat=None, lon=None, fill_val=None,
                 sampled=None):
        self.name = name
        self.dim_order = dims_ordered
        self.ndim = len(dims_ordered)
        self.data = data
        self.resolution = resolution
        self.time = time
        self.lev = self._cnvt_to_float_64(lev)
        self.lat = self._cnvt_to_float_64(lat)
        self.lon = self._cnvt_to_float_64(fix_lon(lon))
        self._fill_val = fill_val
        self._idx_used_for_sample = sampled

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

    def save(self, filename, position=0):

        filename += '.h5'
        data_grp = '/data'

        # Overwrites file everytime save position 0 is invoked
        if position == 0:
            mode = 'w'
        else:
            mode = 'a'

        # Open file to write to
        with tb.open_file(filename, mode,
                          filters=tb.Filters(complib='blosc',
                                             complevel=2)) as h5f:
            if '/grid_objects' in h5f:
                pobj_array = h5f.get_node('/grid_objects')
            else:
                # Write the grid_object
                pobj_array = h5f.create_vlarray('/', 'grid_objects',
                                                atom=tb.ObjectAtom(),
                                                createparents=True)

            if not data_grp in h5f:
                h5f.create_group('/', 'data')

            # Write the data
            self.nan_to_fill_val()
            nd_name = 'obj' + str(position)
            var_to_hdf5_carray(h5f, data_grp, nd_name, self.data)
            self.fill_val_to_nan()

            # Remove data from object before pickling
            tmp_dat = self.data
            del self.data
            pobj_array.append(self)
            self.data = tmp_dat

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
                         lon=new_lon[0],
                         fill_val=self._fill_val)

    def fill_val_to_nan(self):
        self.data[self.data == self._fill_val] = np.nan

    def nan_to_fill_val(self):
        self.data[~np.isfinite(self.data)] = self._fill_val

    def flattened_spatial(self):

        flat_data = self.data.reshape(len(self.time),
                                      np.product(self.space_shp))

        # Get dimensions of data
        coords = [self._dim_coord_map[key] for key in self._space_dims]
        grids = np.meshgrid(*coords, indexing='ij')
        flat_coords = {dim: grid.flatten()
                       for dim, grid in zip(self._space_dims, grids)}

        return flat_data, flat_coords

    def sample_from_idx(self, sample_idxs):
        """
        Random sample ensemble of current gridded variable
        """

        cls = type(self)
        nsamples = len(sample_idxs)
        time_sample = self.time[sample_idxs]

        data_sample = np.zeros([nsamples] + list(self.data.shape[1:]))
        for k, idx in enumerate(sample_idxs):
            data_sample[k] = self.data[idx]

        # Account for timeseries trailing singleton dimension
        data_sample = np.squeeze(data_sample)

        return cls(self.name, self.dim_order, data_sample, self.resolution,
                   time=time_sample,
                   lev=self.lev,
                   lat=self.lat,
                   lon=self.lon,
                   fill_val=self._fill_val,
                   sampled=sample_idxs)

    @abstractmethod
    def load(cls, config, *args):
        pass

    @classmethod
    def _main_load_helper(cls, file_dir, file_name, varname, file_type,
                          base_resolution, nens=None, seed=None, sample=None,
                          split_varname=True, data_req_frac=None, save=True,
                          ignore_pre_avg=False):

        try:
            ftype_loader = cls.get_loader_for_filetype(file_type)
        except KeyError:
            raise TypeError('Specified file type not supported yet.')

        if split_varname:
            fname = file_name.replace('[vardef_template]', varname)
            varname = varname.split('_')[0]
        else:
            fname = file_name

        try:
            if ignore_pre_avg:
                raise IOError('Ignore pre_averaged files')
            var_objs = cls._load_pre_avg_obj(file_dir, fname, varname,
                                             base_resolution,
                                             sample=sample,
                                             nens=nens,
                                             seed=seed)
            print 'Loaded pre-averaged file.'
        except IOError:
            print 'No pre-averaged file found or ignore specified ... ' \
                  'Loading directly from file.'
            var_objs = ftype_loader(file_dir, fname, varname, base_resolution,
                                    sample=sample, save=save,
                                    nens=nens, seed=seed,
                                    data_req_frac=data_req_frac)

        return var_objs



    @classmethod
    def get_loader_for_filetype(cls, file_type):
        ftype_map = {_ftypes.netcdf: cls._load_from_netcdf}
        return ftype_map[file_type]

    @classmethod
    def _load_pre_avg_obj(cls, dir_name, filename, varname, resolution,
                          truncate=False, sample=None, nens=None, seed=None):
        """
        General structure for load pre-averaged:
        1. Load data (done so into sub_annual groups if applicable)
            a. If truncate is desired it searches for pre_avg truncated data
               but if not found, then it searches for full pre_avg data
        5. Sample if desired
        6. Return list of GriddedVar objects
        """

        # Check if pre-processed averages file exists
        pre_avg_tag = '.pre_avg_{}_res{:02.2f}'.format(varname,
                                                       resolution)
        trunc_tag = '.trnc'
        ftype_tag = '.h5'

        path = join(dir_name, filename + pre_avg_tag)

        if truncate:
            path += trunc_tag

        path += ftype_tag

        # Look for pre_averaged_file
        if os.path.exists(path):
            do_trunc = False
            dat_path = path
        elif truncate and os.path.exists(path.strip(trunc_tag)):
            # If truncate requested and truncate not found look for
            # pre-averaged full version
            do_trunc = True
            dat_path = path.strip(trunc_tag)
        else:
            raise IOError('No pre-averaged file found for given specifications')

        # Load prior objects
        with tb.open_file(dat_path, 'r') as h5f:
            obj_arr = h5f.root.grid_objects
            srange = h5f.root.data.obj0.shape[0]

            sample_idxs = cls._sample_gen(sample, srange, nens, seed)

            gobjs = []
            for i, obj in enumerate(obj_arr):
                data = h5f.get_node('/data/' + 'obj'+str(i))

                if sample_idxs:

                    obj.data = data
                    obj = obj.sample_from_idx(sample_idxs)
                else:
                    obj_data = data.read()
                    obj.data = obj_data

                obj.fill_val_to_nan()

                if do_trunc:
                    obj.truncate()
                    obj.save(path, position=i)

                gobjs.append(obj)

        return gobjs

    @classmethod
    def _load_from_netcdf(cls, dir_name, filename, varname, resolution,
                          truncate=False, sample=None, nens=None, seed=None,
                          save=False, data_req_frac=None):
        """
        General structure for load origininal:
        1. Load data
        2. Avg to base resolution
        3. Separate into subannual groups
        4. Create GriddedVar Object for each group and save pre-averaged
        5. Sample if desired
        6. Return list of GriddedVar objects
        """

        # Check if pre-processed averages file exists
        pre_avg_name = '.pre_avg_{}_res{:02.2f}'

        with Dataset(join(dir_name, filename), 'r') as f:
            var = f.variables[varname]
            data_shp = var.shape
            try:
                fill_val = var._FillValue
            except AttributeError:
                fill_val = 2**15 - 1

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
                                                    data_req_frac=data_req_frac)

            # TODO: Replace with logger statement
            print (varname, ' res ', resolution, ': Global: mean=',
                   np.nanmean(avg_data),
                   ' , std-dev=', np.nanstd(avg_data))

            # Filename for saving pre-averaged pickle
            new_fname = join(dir_name,
                             filename + pre_avg_name.format(varname,
                                                            resolution))

            #Separate into subannual objects if res < 1.0
            new_avg_data, new_time = cls._subannual_decomp(avg_data,
                                                           dim_vals[_TIME],
                                                           resolution)

            srange = new_avg_data[0].shape[0]
            sample_idx = cls._sample_gen(sample, srange, nens, seed)

            # Create gridded objects
            grid_objs = []
            for i, (new_dat, new_t) in enumerate(izip(new_avg_data, new_time)):
                dim_vals[_TIME] = new_t
                grid_obj = cls(varname, dims, new_dat, resolution,
                               fill_val=fill_val, **dim_vals)
                if save:
                    grid_obj.save(new_fname, position=i)

                if truncate:
                    try:
                        grid_obj = grid_obj.truncate()

                        if save:
                            grid_obj.save(new_fname + '.trnc', position=i)
                    except AssertionError:
                        # If datatype is not horizontal it won't be truncated
                        pass

                if sample_idx:
                    grid_obj = grid_obj.sample_from_idx(sample_idx)

                grid_objs.append(grid_obj)

            return grid_objs

    @staticmethod
    def _cnvt_to_float_64(data):
        if data is not None:
            data = data.astype(np.float64)
        return data

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
    def _subannual_decomp(data, time, resolution):

        num_subann_chunks = int(np.ceil(1.0/resolution))
        tlen = len(time) / num_subann_chunks

        new_data = np.zeros([num_subann_chunks, tlen] + list(data.shape[1:]),
                            dtype=data.dtype)
        new_time = np.zeros((num_subann_chunks, tlen),
                            dtype=time.dtype)
        for i in xrange(num_subann_chunks):
            new_data[i] = data[i::num_subann_chunks]
            new_time[i] = time[i::num_subann_chunks]

        return new_data, new_time

    @staticmethod
    def _sample_gen(sample, srange, nens, seed):

        if sample is not None:
            return sample
        elif nens is not None:
            # Defaults to sys time if seed=None
            random.seed(seed)
            return random.sample(range(srange), nens)
        else:
            return None

    @staticmethod
    def _time_avg_gridded_to_resolution(time_vals, data, resolution,
                                        yr_shift=0, data_req_frac=None):
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

        # Find number of units in new resolution
        nelem_in_unit_res = resolution * elem_in_yr
        if not nelem_in_unit_res.is_integer():
            raise ValueError('Elements in yr not evenly divisible by given '
                             'resolution')

        end_cutoff = -(len(time_vals[yr_shift:]) % elem_in_yr)
        if end_cutoff == 0:
            end_cutoff = None
        tot_units = int(len(time_vals[yr_shift:end_cutoff]) /
                        nelem_in_unit_res)
        spatial_shp = data.shape[1:]

        # Average data and create year list
        avg_data = data[yr_shift:end_cutoff].reshape(tot_units,
                                                     nelem_in_unit_res,
                                                     *spatial_shp)

        # If desired check for minimum number of data points
        if data_req_frac is not None:
            non_nan_frac = \
                np.isfinite(avg_data).sum(axis=1) / float(nelem_in_unit_res)
            req_met = non_nan_frac >= data_req_frac
            expand_shp = [nelem_in_unit_res] + list(spatial_shp)
            expand_mat = np.ones(expand_shp, dtype=np.bool)
            req_met = np.expand_dims(req_met, axis=1)
            req_met = req_met & expand_mat

            avg_data[~req_met] = np.nan

        avg_data = np.nanmean(avg_data, axis=1)

        start_yr = start.year
        time_yrs = [start_yr + i*resolution
                    for i in xrange(tot_units)]

        return np.array(time_yrs), avg_data


class PriorVariable(GriddedVariable):

    @classmethod
    def load(cls, config, varname, sample=None):
        file_dir = config.prior.datadir_prior
        file_name = config.prior.datafile_prior
        file_type = config.prior.dataformat_prior
        base_resolution = config.core.sub_base_res
        nens = config.core.nens
        seed = config.core.seed
        overwrite = config.core.overwrite_pre_avg_file
        ignore_pre_avg = config.core.ignore_pre_avg_file

        return cls._main_load_helper(file_dir, file_name, varname, file_type,
                                     base_resolution,
                                     nens=nens,
                                     seed=seed,
                                     sample=sample,
                                     save=overwrite,
                                     ignore_pre_avg=ignore_pre_avg)

    @classmethod
    def load_allvars(cls, config):
        var_names = config.prior.state_variables

        sample = None
        prior_dict = OrderedDict()
        for vname in var_names:
            pobjs = cls.load(config, vname, sample)
            # Assure that same sample is used for all variables of a prior
            sample = pobjs[0]._idx_used_for_sample
            prior_dict[vname] = pobjs

        return prior_dict

    @staticmethod
    def _time_avg_gridded_to_resolution(time_vals, data, resolution,
                                        yr_shift=0, data_req_frac=None):

        # Call Base class to get correct time average
        time, avg_data = \
            super(PriorVariable, PriorVariable).\
            _time_avg_gridded_to_resolution(time_vals, data, resolution,
                                            yr_shift=yr_shift,
                                            data_req_frac=data_req_frac)
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


class AnalysisVariable(GriddedVariable):

    @classmethod
    def load(cls, psm_config):

        # TODO: change this to switch based on PSM
        file_dir = psm_config.datadir_calib
        file_name = psm_config.datafile_calib
        file_type = psm_config.dataformat_calib
        base_resolution = psm_config.sub_base_res
        varname = psm_config.varname_calib
        dat_frac = psm_config.min_data_req_frac
        overwrite = psm_config.overwrite_pre_avg_file
        ignore = psm_config.ignore_pre_avg_file

        return cls._main_load_helper(file_dir, file_name, varname, file_type,
                                     base_resolution, split_varname=False,
                                     data_req_frac=dat_frac,
                                     save=overwrite,
                                     ignore_pre_avg=ignore)

    @classmethod
    def load_allvars(cls):
        pass

    @staticmethod
    def avg_calib_to_res(calib_objs, resolutions, shifts):
        class_type = type(calib_objs[0])
        calib_res_dict = {}
        for res in resolutions:
            shift = shifts[res]
            if shift > 0:
                shift_idx = int(calib_objs[0].resolution/shift)
            else:
                shift_idx = 0
            num_obj_out = int(np.ceil(1/res))
            nobjs_to_avg = len(calib_objs)/num_obj_out

            if num_obj_out == len(calib_objs):
                # TODO: no shift, but not a current usage case...
                calib_res_dict[res] = calib_objs
                continue

            shift_calib_objs = np.roll(calib_objs, shift_idx)

            aobjs = []
            for i in xrange(num_obj_out):
                start = i*nobjs_to_avg
                end = start+nobjs_to_avg

                curr_objs = shift_calib_objs[start:end]

                # Determine mask for missing sub_annual locations
                mask = ~np.isfinite(curr_objs[0].data)
                for obj in curr_objs:
                    mask |= ~np.isfinite(obj.data)

                data = []
                for obj in curr_objs:
                    tmp = obj.data.copy()
                    tmp[mask] = np.nan
                    data.append(tmp)

                new_data = np.nanmean(data, axis=0)

                new_time = calib_objs[start].time

                curr_obj = curr_objs[0]
                new_obj = class_type(curr_obj.name,
                                     curr_obj.dim_order,
                                     new_data,
                                     res,
                                     time=new_time,
                                     lat=curr_obj.lat,
                                     lon=curr_obj.lon,
                                     lev=curr_obj.lev,
                                     fill_val=curr_obj._fill_val)

                aobjs.append(new_obj)
            calib_res_dict[res] = aobjs

        return calib_res_dict


class BerkeleyEarthAnalysisVariable(AnalysisVariable):

    @classmethod
    def load(cls, psm_config):
        return super(BerkeleyEarthAnalysisVariable, cls)

    @staticmethod
    def _netcdf_datetime_convert(time_var):
        """
        Converts netcdf time variable into date-times.

        Used as a static method in case necesary to overwrite with subclass
        :param time_var:
        :return:
        """

        time_yrs = []
        for yrAD in time_var[:]:

            year = int(yrAD)
            rem = yrAD - year
            base = datetime(year, 1, 1)
            diff_yr = base.replace(year=base.year + 1) - base
            diff_to_yr_secs = diff_yr.total_seconds()
            tdel = timedelta(seconds=(diff_to_yr_secs * rem))
            time_yrs.append(base + tdel)

        return np.array(time_yrs)


class State(object):
    """
    Class to create state vector and information
    """

    def __init__(self, prior_vars, base_res):

        self._prior_vars = prior_vars
        self._base_res = base_res
        self.resolution = base_res
        self.state_list = []
        self.var_coords = {}
        self.var_view_range = {}
        self.var_space_shp = {}
        self.augmented = False

        # Attr for h5 container use
        self.h5f_out = None
        self.xb_out = None
        self._yr_len = None
        self._orig_state = None
        self._tmp_state = {}

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
        base_res = config.core.sub_base_res

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

    def augment_state(self, ye_vals):

        aug_state_list = []
        for i, state in enumerate(self.state_list):
            aug_state_list.append(np.append(state, ye_vals, axis=0))

        self.state_list = aug_state_list
        self.augmented = True

        self.var_view_range['state'] = (0, self.len_state)
        self.var_view_range['ye_vals'] = (self.len_state,
                                          self.len_state + len(ye_vals))

    def reset_augmented_ye(self, ye_vals):

        ye_state = self.get_var_data('ye_vals')
        for i, subann_ye in enumerate(ye_state):
            subann_ye[:] = ye_vals

    def annual_avg(self, var_name=None):

        if var_name is None:
            subannual_data = np.array(self.state_list)
        else:
            # Get variable data for each subannual state vector, creat ndarray
            subannual_data = np.array(self.get_var_data(var_name))

        avg = subannual_data.mean(axis=0)
        return avg

    def stash_state_list(self, name):

        self._tmp_state[name] = (deepcopy(self.state_list), self.resolution)

    def stash_recall_state_list(self, name, pop=False):

        if name in self._tmp_state:
            if pop:
                state, res = self._tmp_state.pop(name)
            else:
                state, res = self._tmp_state[name]

            self.state_list = state
            self.resolution = res
        else:
            print 'No currently stashed state with name {}....'.format(name)

    def stash_pop_state_list(self, name):
        self.stash_recall_state_list(name, pop=True)

    def avg_to_res(self, res, shift):

        """Average current state list to resolution"""

        if res == self.resolution:
            return

        if res < 1:
            chunk = int(res / self._base_res)
            shift_idx = int(shift / self._base_res)
            tmp_dat = np.roll(self.state_list, shift_idx, axis=1)

            end_idx = len(self.state_list)
            new_state = [tmp_dat[i:i+chunk].mean(axis=0)
                         for i in xrange(0, end_idx, chunk)]
            self.state_list = new_state
        elif res == 1:
            new_state = np.array(self.state_list).mean(axis=0)
            self.state_list = [new_state]
        else:
            raise ValueError('Cannot handle resolutions larger than 1 yet.')

        self.resolution = res

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

    def create_h5_state_container(self, fname, nyrs_in_recon):

        """Initialize pytables output container"""

        self.h5f_out = tb.open_file(fname, 'w',
                                    filters=tb.Filters(complib='blosc',
                                                       complevel=2))
        atom = tb.Atom.from_dtype(self.state_list[0].dtype)
        num_subann = len(self.state_list)
        tdim_len = (nyrs_in_recon + 1) * num_subann
        shape = [tdim_len] + list(self.state_list[0].shape)
        self.xb_out = empty_hdf5_carray(self.h5f_out, '/', 'output', atom,
                                        shape)

        self._orig_state = deepcopy(self.state_list)
        self._yr_len = len(self._orig_state)
        self.xb_out[0:self._yr_len] = np.array(self._orig_state)
        self.xb_out[self._yr_len:(self._yr_len*2)] = np.array(self._orig_state)

    def insert_upcoming_prior(self, curr_yr_idx, use_curr=False):

        """
        Insert as we go to prevent huge upfront write cost
        """
        istart = curr_yr_idx*self._yr_len + self._yr_len
        iend = istart + self._yr_len

        if not use_curr:
            self.xb_out[istart:iend] = self._orig_state
        else:
            self.xb_out[istart:iend] = self.xb_out[curr_yr_idx:istart]

    def xb_from_h5(self, yr_idx, res, shift):
        ishift = int(shift / self._base_res)
        istart = yr_idx*self._yr_len + ishift
        iend = istart + self._yr_len

        self.state_list = self.xb_out[istart:iend]
        self.avg_to_res(res, 0)

    def propagate_avg_to_h5(self, yr_idx, shift):
        nchunks = len(self.state_list)
        chk_size = self._yr_len / nchunks
        ishift = int(shift / self._base_res)

        for i in xrange(nchunks):
            avg = self.state_list[i]

            istart = yr_idx*self._yr_len + i*chk_size + ishift
            iend = istart + chk_size
            tmp_dat = self.xb_out[istart:iend]

            if len(tmp_dat) > 1:
                tmp_dat = tmp_dat - tmp_dat.mean(axis=0)
                tmp_dat += avg

                self.xb_out[istart:iend] = tmp_dat
            else:
                # same size as _base_res, just replace
                self.xb_out[istart:iend] = avg

    def close_xb_container(self):
        self.h5f_out.close()


_analysis_var_classes = {'BerkeleyEarth': BerkeleyEarthAnalysisVariable}


def get_analysis_var_class(analysis_source):
    return _analysis_var_classes.get(analysis_source, AnalysisVariable)
