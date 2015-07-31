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


class GriddedVariable(object):

    def __init__(self, name, dims_ordered, data, time=None,
                 lev=None, lat=None, lon=None):
        self.name = name
        self.dim_order = dims_ordered
        self.ndim = len(dims_ordered)
        self.data = data
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
    def _load_from_netcdf(cls, config, varname):

        dir_name = config.prior.datadir_prior
        filename = config.prior.datafile_prior
        pre_avg_name = '.pre_avg'

        pre_avg_exists = False
        # Check for pre-averaged
        if os.path.exists(dir_name+filename+pre_avg_name):
            pre_avg_exists = True

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

            # Convert time to years
            dim_vals[_TIME] = cls.netcdf_datetime_convert(dim_vals[_TIME])

            # Extract data for each dimension
            dim_vales = {k: val[:] for k, val in dim_vals.iteritems()}

            return cls(varname, dims, var[:], **dim_vals)

    @classmethod
    def load(cls, config, file_type):
        if file_type == constants.file_types.netcdf:
            cls._load_from_netcdf(config)
        else:
            raise TypeError('Specified file type not supported yet.')

    @staticmethod
    def netcdf_datetime_convert(time_var):
        try:
            time = num2date(time_var[:], units=time_var.units,
                                calendar=time_var.calendar)
            return  time.tolist()
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
            time_yrs = num2date(time_var[:], new_units, calendar=time_var.calendar)
            return [datetime(d.year + year_diff, d.month, d.day,
                             d.hour, d.minute, d.second)
                    for d in time_yrs]
