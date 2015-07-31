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


class GriddedData:
    __metaclass__ = ABCMeta

    def __init__(self, name, vartype, dim_order, data, time,
                 lev=None, lat=None, lon=None):
        self.name = name
        self.vartype = vartype
        self.dim_order = dim_order
        self.data = data
        self.time = time
        self.lev = lev
        self.lat = lat
        self.lon = lon

    @classmethod
    @abstractmethod
    def _load_from_netcdf(cls):
        pass

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


class PriorVariable(GriddedData):

    @classmethod
    def _load_from_netcdf(cls, config, varname):

        dir = config.prior.datadir_prior
        file = config.prior.datafile_prior

        # Check for pre-averaged

        with Dataset(dir+file, 'r') as f:
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

            # Determine Prior TYPE

            return cls(varname, )





def _unload_netcdf_dims(dat_shp, ncf_dimensions):
