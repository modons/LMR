"""
Module containing methods for various PSMs

Adapted from LMR_proxy and LMR_calibrate by Andre Perkins
"""

import matplotlib.pyplot as pylab
import numpy as np
import logging
import LMR_calibrate
from LMR_utils2 import haversine, smooth2D, class_docs_fixer

from scipy.stats import linregress
from abc import ABCMeta, abstractmethod
from load_data import load_cpickle

# Logging output utility, configuration controlled by driver
logger = logging.getLogger(__name__)

class BasePSM:
    """
    Proxy system model.

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to
    psm_kwargs: dict (unpacked)
        Specfic arguments for the target PSM
    """
    __metaclass__ = ABCMeta

    def __init__(self, config, proxy_obj, **psm_kwargs):
        pass

    @abstractmethod
    def psm(self, prior_obj):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        prior_obj BasePriorObject
            Prior to be mapped to observation space (Ye).

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """
        pass

    @abstractmethod
    def error(self):
        """
        Error model for given PSM.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_kwargs(config):
        """
        Returns keyword arguments required for instantiation of given PSM.

        Parameters
        ----------
        config: LMR_config
            Configuration module used for current LMR run.

        Returns
        -------
        kwargs: dict
            Keyword arguments for given PSM
        """
        pass


@class_docs_fixer
class LinearPSM(BasePSM):
    """
    PSM based on linear regression.

    Attributes
    ----------
    lat: float
        Latitude of associated proxy site
    lon: float
        Longitude of associated proxy site
    corr: float
        Correlation of proxy record against calibration data
    slope: float
        Linear regression slope of proxy/calibration fit
    intercept: float
        Linear regression y-intercept of proxy/calibration fit
    R: float
        Mean-squared error of proxy/calibration fit

    Parameters
    ----------
    config: LMR_config
        Configuration module used for current LMR run.
    proxy_obj: BaseProxyObject like
        Proxy object that this PSM is being attached to
    psm_data: dict
        Pre-calibrated PSM dictionary containing current associated
        proxy site's calibration information

    Raises
    ------
    ValueError
        If PSM is below critical correlation threshold.
    """

    def __init__(self, config, proxy_obj, psm_data=None):

        proxy = proxy_obj.type
        site = proxy_obj.id
        r_crit = config.psm.linear.psm_r_crit

        try:
            # Try using pre-calibrated psm_data
            if psm_data is None:
                psm_data = self._load_psm_data(config)

            psm_site_data = psm_data[(proxy, site)]

            self.lat = psm_site_data['lat']
            self.lon = psm_site_data['lon']
            self.corr = psm_site_data['PSMcorrel']
            self.slope = psm_site_data['PSMslope']
            self.intercept = psm_site_data['PSMintercept']
            self.R = psm_site_data['PSMmse']

        except (KeyError, IOError) as e:
            # No precalibration found, have to do it for this proxy
            logger.error(e)
            logger.info('PSM not calibrated for:' + str((proxy, site)))

            # TODO: Fix call and Calib Module
            datag_calib = config.psm.linear.datatag_calib
            C = LMR_calibrate.calibration_assignment(datag_calib)
            C.datadir_calib = config.psm.linear.datadir_calib
            C.read_calibration()
            self.calibrate(C, proxy_obj)

        # Raise exception if critical correlation value not met
        if abs(self.corr) < r_crit:
            raise ValueError(('Proxy model correlation ({:.2f}) does not meet '
                              'critical threshold ({:.2f}).'
                              ).format(self.corr, r_crit))

    # TODO: Ideally prior state info and coordinates should all be in single obj
    def psm(self, Xb, state_idx):
        """
        Maps a given state to observations for the given proxy

        Parameters
        ----------
        Xb: ndarray
            State vector to be mapped into observation space (stateDim x ensDim)
        X_state_info: dict
            Information pertaining to variables in the state vector
        X_coords: ndarray
            Coordinates for the state vector (stateDim x 2)

        Returns
        -------
        Ye:
            Equivalent observation from prior
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------
        # TODO: state variable is hard coded for now...
        state_var = 'tas_sfc_Amon'
        if state_var not in Xb.var_view_range.keys():
            raise KeyError('Needed variable not in state vector for Ye'
                           ' calculation.')

        # Find row index of X for which [X_lat,X_lon] corresponds to closest
        # grid point to
        # location of proxy site [self.lat,self.lon]
        # Calclulate distances from proxy site.
        dist = haversine(self.lon, self.lat,
                          Xb.var_coords[state_var]['lon'],
                          Xb.var_coords[state_var]['lat'])

        # row index of nearest grid pt. in prior (minimum distance)
        kind = np.unravel_index(dist.argmin(), dist.shape)[0]

        Ye = (self.slope *
              np.squeeze(Xb.get_var_data(state_idx, state_var)[kind, :]) +
              self.intercept)

        return Ye

    # Define the error model for this proxy
    @staticmethod
    def error():
        return 0.1

    # TODO: Simplify a lot of the actions in the calibration
    def calibrate(self, C, proxy, diag_output=False, diag_output_figs=False):
        """
        Calibrate given proxy record against observation data and set relevant
        PSM attributes.

        Parameters
        ----------
        C: calibration_master like
            Calibration object containing data, time, lat, lon info
        proxy: BaseProxyObject like
            Proxy object to fit to the calibration data
        diag_output, diag_output_figs: bool, optional
            Diagnostic output flags for calibration method
        """

        calib_spatial_avg = False
        Npts = 9  # nb of neighboring pts used in smoothing

        # --------------------------------------------
        # Use linear model (regression) as default PSM
        # --------------------------------------------

        # TODO: Distance calculation should probably be a function
        # Look for indices of calibration grid point closest in space (in 2D)
        # to proxy site
        dist = np.empty([C.lat.shape[0], C.lon.shape[0]])
        tmp = np.array(
            [haversine(proxy.lon, proxy.lat, C.lon[k], C.lat[j]) for j
             in xrange(len(C.lat)) for k in xrange(len(C.lon))])
        dist = np.reshape(tmp, [len(C.lat), len(C.lon)])
        # indices of nearest grid pt.
        jind, kind = np.unravel_index(dist.argmin(), dist.shape)

        # TODO: need to fix this to use proxy2 style time
        # overlapping years
        sc = set(C.time)
        sp = set(proxy.time)

        common_time = sorted(list(sc.intersection(sp)))
        # respective indices in calib and proxy times
        ind_c = [j for j, k in enumerate(C.time) if k in common_time]
        #ind_p = [j for j, k in enumerate(proxy.time) if k in common_time]
        ind_p = common_time

        if calib_spatial_avg:
            C2Dsmooth = np.zeros(
                [C.time.shape[0], C.lat.shape[0], C.lon.shape[0]])
            for m in ind_c:
                C2Dsmooth[m, :, :] = smooth2D(C.temp_anomaly[m, :, :], n=Npts)
            reg_x = [C2Dsmooth[m, jind, kind] for m in ind_c]
        else:
            reg_x = [C.temp_anomaly[m, jind, kind] for m in ind_c]

        reg_y = [proxy.values[m] for m in ind_p]
        # check for valid values
        ind_c_ok = [j for j, k in enumerate(reg_x) if np.isfinite(k)]
        ind_p_ok = [j for j, k in enumerate(reg_y) if np.isfinite(k)]
        common_ok = sorted(list(set(ind_c_ok).intersection(set(ind_p_ok))))
        # fill numpy arrays with values used in regression
        reg_xa = np.array([reg_x[k] for k in common_ok])  # calib. values
        reg_ya = np.array([reg_y[k] for k in common_ok])  # proxy values

        # -------------------------
        # Perform linear regression
        # -------------------------
        nobs = reg_ya.shape[0]
        if nobs < 10:  # skip rest if insufficient overlapping data
            raise(ValueError('Insufficent observation/calibration overlap'
                             ' to calibrate psm.'))

        # START NEW (GH) 21 June 2015
        # detrend both the proxy and the calibration data
        #
        # save copies of the original data for residual estimates later
        reg_xa_all = np.copy(reg_xa)
        reg_ya_all = np.copy(reg_ya)
        # proxy detrend: (1) linear regression, (2) fit, (3) detrend
        xvar = range(len(reg_ya))
        proxy_slope, proxy_intercept, r_value, p_value, std_err = \
            linregress(xvar, reg_ya)
        proxy_fit = proxy_slope*np.squeeze(xvar) + proxy_intercept
        # reg_ya = reg_ya - proxy_fit # expt 4: no detrend for proxy
        # calibration detrend: (1) linear regression, (2) fit, (3) detrend
        xvar = range(len(reg_xa))
        calib_slope, calib_intercept, r_value, p_value, std_err = \
            linregress(xvar, reg_xa)
        calib_fit = calib_slope*np.squeeze(xvar) + calib_intercept
        # reg_xa = reg_xa - calib_fit
        # END NEW (GH) 21 June 2015

        print 'Calib stats (x)              [min, max, mean, std]:', np.nanmin(
            reg_xa), np.nanmax(reg_xa), np.nanmean(reg_xa), np.nanstd(reg_xa)
        print 'Proxy stats (y:original)     [min, max, mean, std]:', np.nanmin(
            reg_ya), np.nanmax(reg_ya), np.nanmean(reg_ya), np.nanstd(reg_ya)
        # standardize proxy values over period of overlap with calibration data
        # reg_ya = (reg_ya - np.nanmean(reg_ya))/np.nanstd(reg_ya)
        # print 'Proxy stats (y:standardized) [min, max, mean, std]:', np.nanmin
        # (reg_ya), np.nanmax(reg_ya), np.nanmean(reg_ya), np.nanstd(reg_ya)
        # GH: note that std_err pertains to the slope, not the residuals!!!

        self.slope, self.intercept, r_value, p_value, std_err = linregress(
            reg_xa, reg_ya)

        # Calculate stats on regression residuals
        # GH: residuals have to be computed "by hand"
        fit = self.slope * np.squeeze(reg_xa) + self.intercept
        residuals = fit - reg_ya
        MSE = np.mean((residuals) ** 2)
        self.R = MSE
        self.corr = r_value

        if diag_output:
            # Some other diagnostics
            varproxy = np.var(reg_ya)
            varresiduals = np.var(residuals)
            varratio = np.divide(varproxy, varresiduals)
            snr = abs(r_value) / (np.sqrt(1.0 - r_value ** 2))
            # Diagnostic output
            print "***PSM stats:"
            print "Nobs                 :", nobs
            print "slope                :", self.slope
            print "intercept            :", self.intercept
            print "correl.              :", r_value
            print "r-squared            :", r_value ** 2
            print "p_value              :", p_value
            print "std_err              :", std_err
            print "resid MSE            :", MSE
            print "var(proxy)/var(resid):", varratio
            print "SNR                  :", snr

            if diag_output_figs:
                # Figure (scatter plot w/ summary statistics)
                line = self.slope * reg_xa + self.intercept
                pylab.plot(reg_xa, line, 'r-', reg_xa, reg_ya, 'o',
                           markersize=7, markerfacecolor='#5CB8E6',
                           markeredgecolor='black', markeredgewidth=1)
                # GH: I don't know how this ran in the first place; must exploit
                #  some global namespace
                pylab.title('%s: %s' % (proxy.type, proxy.id))
                pylab.xlabel('Calibration data')
                pylab.ylabel('Proxy data')
                xmin, xmax, ymin, ymax = pylab.axis()
                # Annotate with summary stats
                ypos = ymax - 0.05 * (ymax - ymin)
                xpos = xmin + 0.025 * (xmax - xmin)
                pylab.text(xpos, ypos, 'Nobs = %s' % str(nobs), fontsize=12,
                           fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                pylab.text(xpos, ypos,
                           'Slope = %s' % "{:.4f}".format(self.slope),
                           fontsize=12, fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                pylab.text(xpos, ypos,
                           'Intcpt = %s' % "{:.4f}".format(self.intercept),
                           fontsize=12, fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                pylab.text(xpos, ypos, 'Corr = %s' % "{:.4f}".format(r_value),
                           fontsize=12, fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                pylab.text(xpos, ypos,
                           'p_value = %s' % "{:.4f}".format(p_value),
                           fontsize=12, fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                pylab.text(xpos, ypos, 'StdErr = %s' % "{:.4f}".format(std_err),
                           fontsize=12, fontweight='bold')
                ypos = ypos - 0.05 * (ymax - ymin)
                pylab.text(xpos, ypos, 'Res.MSE = %s' % "{:.4f}".format(MSE),
                           fontsize=12, fontweight='bold')
                pylab.savefig('proxy_%s_%s_LinearModel_calib.png' % (
                    proxy.type.replace(" ", "_"), proxy.id),
                              bbox_inches='tight')
                pylab.close()

    @staticmethod
    def get_kwargs(config):
        psm_data = LinearPSM._load_psm_data(config)
        return {'psm_data': psm_data}

    @staticmethod
    def _load_psm_data(config):
        """Helper method for loading from dataframe"""
        pre_calib_file = config.psm.linear.pre_calib_datafile

        return load_cpickle(pre_calib_file)


# Mapping dict to PSM object type, this is where proxy_type/psm relations
# should be specified (I think.) - AP
_psm_classes = {'linear': LinearPSM}

def get_psm_class(psm_type):
    """
    Retrieve psm class type to be instantiated.

    Parameters
    ----------
    psm_type: str
        Dict key to retrieve correct PSM class type.

    Returns
    -------
    BasePSM like:
        Class type to be instantiated and attached to a proxy.
    """
    return _psm_classes[psm_type]

