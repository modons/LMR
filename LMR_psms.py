"""
Module containing methods for various PSMs

Adapted from LMR_proxy and LMR_calibrate by Andre Perkins
"""

import matplotlib.pyplot as pylab
import numpy as np
import logging
import LMR_utils
import LMR_calibrate

from scipy.stats import linregress
from abc import ABCMeta, abstractmethod
from load_data import load_cpickle

# Logging output utility, configuration controlled by driver
logger = logging.getLogger(__name__)

class BasePSM:
    """
    Abstract class defining what a PSM should look like.  Minimizes need to
    adjust code in actual driver.
    """
    __metaclass__ = ABCMeta

    def __init__(self, config, proxy_obj, **psm_kwargs):
        pass

    @abstractmethod
    def psm(self, prior_obj):
        pass

    @abstractmethod
    def error(self):
        pass

    @staticmethod
    @abstractmethod
    def get_kwargs(config):
        pass


class LinearPSM(BasePSM):

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
            self.calibrate(C)

        if self.corr < r_crit:
            raise ValueError(('Proxy model correlation ({:.2f}) does not meet '
                              'critical threshold ({:.2f}).'
                              ).format(self.corr, r_crit))

    def psm(self, X):

        """
        Main function to be exposed to each proxy object

        :param X:
        :return:
        """

        # ----------------------
        # Calculate the Ye's ...
        # ----------------------

        # Find row index of X for which [X_lat,X_lon] corresponds to closest grid point to
        # location of proxy site [self.lat,self.lon]
        # Calclulate distances from proxy site.
        stateDim = X.shape[0]
        ensDim = X.shape[1]
        X_lat = X.lat
        X_lon = X.lon
        dist = np.empty(stateDim)
        dist = np.array(
            [LMR_utils.haversine(self.lon, self.lat, X_lon[k], X_lat[k]) for k
             in xrange(stateDim)])

        # row index of nearest grid pt. in prior (minimum distance)
        kind = np.unravel_index(dist.argmin(), dist.shape)

        Ye = self.slope * np.squeeze(X[kind, :]) + self.intercept

        return Ye

    # Define the error model for this proxy
    @staticmethod
    def error():
        return 0.1

    # TODO: Simplify a lot of the actions in the calibration
    def calibrate(self, C, proxy_data,
                  diag_output=False, diag_output_figs=False):

        calib_spatial_avg = False
        Npts = 9  # nb of neighboring pts used in smoothing

        # --------------------------------------------
        # Use linear model (regression) as default PSM
        # --------------------------------------------

        """ GH code block for random calibration year. not yet implemented in this new code
        GH: new: random sample of calibration period. ncalib is the sample size (max = len(self.time))
        ncalib = 75
        rantimes = sample(range(0,len(self.time)),ncalib)
        for i in rantimes:
        """

        # Look for indices of calibration grid point closest in space (in 2D) to proxy site
        dist = np.empty([C.lat.shape[0], C.lon.shape[0]])
        tmp = np.array(
            [LMR_utils.haversine(self.lon, self.lat, C.lon[k], C.lat[j]) for j
             in xrange(len(C.lat)) for k in xrange(len(C.lon))])
        dist = np.reshape(tmp, [len(C.lat), len(C.lon)])
        # indices of nearest grid pt.
        jind, kind = np.unravel_index(dist.argmin(), dist.shape)

        # TODO: need to fix this to use proxy2 style time
        # overlapping years
        sc = set(C.time)
        sp = set(proxy_data.time)

        common_time = list(sc.intersection(sp))
        # respective indices in calib and proxy times
        ind_c = [j for j, k in enumerate(C.time) if k in common_time]
        ind_p = [j for j, k in enumerate(proxy_data.time) if k in common_time]

        if calib_spatial_avg:
            C2Dsmooth = np.zeros(
                [C.time.shape[0], C.lat.shape[0], C.lon.shape[0]])
            for m in ind_c:
                C2Dsmooth[m, :, :] = LMR_utils.smooth2D(C.temp_anomaly[m, :, :],
                                                        n=Npts)
            reg_x = [C2Dsmooth[m, jind, kind] for m in ind_c]
        else:
            reg_x = [C.temp_anomaly[m, jind, kind] for m in ind_c]

        reg_y = [proxy_data.values[m] for m in ind_p]
        # check for valid values
        ind_c_ok = [j for j, k in enumerate(reg_x) if np.isfinite(k)]
        ind_p_ok = [j for j, k in enumerate(reg_y) if np.isfinite(k)]
        common_ok = list(set(ind_c_ok).intersection(set(ind_p_ok)))
        # fill numpy arrays with values used in regression
        reg_xa = np.array([reg_x[k] for k in common_ok])  # calib. values
        reg_ya = np.array([reg_y[k] for k in common_ok])  # proxy values

        # -------------------------
        # Perform linear regression
        # -------------------------
        nobs = reg_ya.shape[0]
        if nobs < 10:  # skip rest if insufficient overlapping data
            return

        print 'Calib stats (x)              [min, max, mean, std]:', np.nanmin(
            reg_xa), np.nanmax(reg_xa), np.nanmean(reg_xa), np.nanstd(reg_xa)
        print 'Proxy stats (y:original)     [min, max, mean, std]:', np.nanmin(
            reg_ya), np.nanmax(reg_ya), np.nanmean(reg_ya), np.nanstd(reg_ya)
        # standardize proxy values over period of overlap with calibration data
        # reg_ya = (reg_ya - np.nanmean(reg_ya))/np.nanstd(reg_ya)
        # print 'Proxy stats (y:standardized) [min, max, mean, std]:', np.nanmin(reg_ya), np.nanmax(reg_ya), np.nanmean(reg_ya), np.nanstd(reg_ya)
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
                # GH: I don't know how this ran in the first place; must exploit some global namespace
                pylab.title('%s: %s' % (proxy_data.type, proxy_data.id))
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
                    proxy_data.type.replace(" ", "_"), proxy_data.id),
                              bbox_inches='tight')
                pylab.close()

    @staticmethod
    def get_kwargs(config):
        psm_data = LinearPSM._load_psm_data(config)
        return {'psm_data': psm_data}

    @staticmethod
    def _load_psm_data(config):
        pre_calib_file = config.psm.linear.pre_calib_datafile

        return load_cpickle(pre_calib_file)


_psm_classes = {'linear': LinearPSM}

def get_psm_class(psm_type):
    return _psm_classes[psm_type]

