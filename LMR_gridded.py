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

from LMR_config import constants

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

        space_dims = [dim for dim in dims_ordered if dim != _TIME]

        if len(space_dims) > 2:
            raise ValueError('Class cannot handle 3D data yet!')

        if not space_dims:
            self.type = 'timeseries'
        elif _LAT in space_dims and _LON in space_dims:
            self.type = 'horizontal'
        elif _LAT in space_dims and _LEV in space_dims:
            self.type = 'meridional_vertical'
        else:
            raise ValueError('Unrecognized dimension combination.')

    @classmethod
    @abstractmethod
    def load(cls, config):
        pass

    @classmethod
    def get_loader_for_filetype(cls, file_type):
        ftype_map = {_ftypes.netcdf: cls._load_from_netcdf}
        return ftype_map[file_type]

    @classmethod
    def _load_from_netcdf(cls, dir_name, filename, varname, resolution,
                          yr_shift):

        # Check if pre-processed averages file exists
        pre_avg_name = '.pre_avg_res{:02.1f}_shift{:d}'.format(resolution,
                                                               yr_shift)
        if os.path.exists(dir_name + filename + pre_avg_name):
            filename += pre_avg_name
            pre_processed = True
        else:
            pre_processed = False

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
                    dims.append(_ALT_DIMENSION_DEFS(dim.lower()))

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
            dim_vales = {k: val[:] for k, val in dim_vals.iteritems()}

            # Average to correct time resolution
            dim_vals[_TIME], avg_data = \
                cls._time_avg_gridded_to_resolution(dim_vals[_TIME],
                                                    var[:],
                                                    resolution,
                                                    yr_shift)

            # TODO: Replace with logger statement
            print (varname, ': Global: mean=', np.nanmean(avg_data),
                   ' , std-dev=', np.nanstd(avg_data))

            return cls(varname, dims, avg_data, resolution, **dim_vals)

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
            fmt = '%Y-%d-%m %H:%M:%S'
            tunits = time_var.units
            since_yr_idx = tunits.index('since ') + 6
            year = int(tunits[since_yr_idx:since_yr_idx+4])
            year_diff = year - 0001
            new_start_date = datetime(0001, 01, 01, 0, 0, 0)

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
        time_yrs = [start_yr + i*resolution for i in range(tot_units)]

        return time_yrs, avg_data


class PriorVariable(GriddedVariable):

    @classmethod
    def load(cls, config):
        file_dir = config.prior.datadir_prior
        file_name = config.prior.datafile_prior
        yr_resolu = config.core.assimilation_time_res

        if file_type == constants.file_types.netcdf:
            cls._load_from_netcdf(config)
        else:
            raise TypeError('Specified file type not supported yet.')

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
