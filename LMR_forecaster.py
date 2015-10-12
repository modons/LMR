from abc import ABCMeta, abstractmethod
import numpy as  np

import pylim.LIM as LIM
import pylim.DataTools as DT
from LMR_utils2 import class_docs_fixer, augment_docstr, regrid_sphere


class BaseForecaster:
    """
    Class defining methods for LMR forecasting
    """

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def forecast(self, t0_data):
        """
        Perform forecast

        Parameters
        ----------
        t0_data: ndarray-like
            Initial data to forecast from (stateDim x nens)

        Returns
        -------
        ndarray-like
            Forecast results (stateDim x nens)
        :return:
        """
        pass


@class_docs_fixer
class LIMForecaster:
    """
    Linear Inverse Model Forecaster
    """

    def __init__(self, config):
        cfg = config.forecaster.LIM
        infile = cfg.calib_filename
        varname = cfg.calib_varname
        fmt = cfg.dataformat

        if fmt == 'NCD':
            data_obj = DT.netcdf_to_data_obj(infile, varname, force_flat=True)
        else:
            raise TypeError('Unsupported calibration data'
                            ' type for LIM: {}'.format(fmt))

        coords = data_obj.get_dim_coords(['lat', 'lon', 'time'])

        # TODO: May want to tie this more into LMR regridding
        # Truncate the calibration data
        dat_new, lat_new, lon_new = regrid_sphere(len(coords['lat'][1]),
                                                  len(coords['lon'][1]),
                                                  len(coords['time'][1]),
                                                  data_obj.data.T,
                                                  42)

        new_coords = {'time': coords['time'],
                      'lat': (1, lat_new[:, 0]),
                      'lon': (2, lon_new[0])}
        new_shp = (len(new_coords['time'][1]),
                   len(new_coords['lat'][1]),
                   len(new_coords['lon'][1]))
        dat_new = dat_new.T.reshape(new_shp)

        calib_obj = DT.BaseDataObject(dat_new, dim_coords=new_coords,
                                      force_flat=True)

        self.lim = LIM.LIM(calib_obj, cfg.wsize, cfg.fcast_times,
                           cfg.fcast_num_pcs, detrend_data=cfg.detrend)

    def forecast(self, state_obj):

        assert state_obj.resolution == 1.0, 'Annual data required to forecast.'

        state_var = 'tas_sfc_Amon'
        var_data = state_obj.get_var_data(state_var, idx=0)

        fcast_data = self._forecast_helper(var_data)

        # var_data is returned as a view for annual, so this re-assigns
        var_data[:] = fcast_data.T

    def _forecast_helper(self, t0_data):

        # TODO: Check anomaly stuff
        # dummy time coordinate
        time_coord = {'time': (0, range(t0_data.shape[1]))}
        fcast_obj = DT.BaseDataObject(t0_data.T, dim_coords=time_coord,
                                      force_flat=True,
                                      is_run_mean=True,
                                      is_anomaly=True,
                                      is_detrended=True)

        fcast, eofs = self.lim.forecast(fcast_obj)

        # return physical forecast (dimensions of stateDim x nens)
        return np.dot(eofs, np.squeeze(fcast))


_forecaster_classes = {'lim': LIMForecaster}


def get_forecaster_class(key):
    """
    Retrieve forecaster class type to be instantiated.

    Parameters
    ----------
    key: str
        Dict key to retrieve correct Forecaster class type.

    Returns
    -------
    BaseForecaster-like:
        Forecaster class to be instantiated
    """
    return _forecaster_classes[key]




