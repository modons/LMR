"""
Module: LMR_verify_proxy.py

Purpose: Contains functions used to evaluate the LMR paleoclimate reconstructions 
         on the basis of proxy data set aside for verification (non-assimilated).
         Error statistics of the reconstructions are evaluated using an independent 
         set of proxy chronologies. 

 Originator: Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
                           | March 2015

Revisions:
           - From original LMR_diagnostics_proxy code, adapted to new reconstruction 
             output (ensemble mean only) and use of pre-built PSMs.
             [R. Tardif, U. of Washington, July 2015]
           - Adapted to the Pandas DataFrames version of the proxy database.
             [R. Tardif, U. of Washington, December 2015]
           - Addded the use of NCDC proxy database as possible proxy data input.
             [R. Tardif, U. of Washington, March 2016]
           - Adjustments made to code to reflect recent changes to how config classes
             are defined and instanciated + general code re-org. for improving speed.
             [R. Tardif, U. of Washington, July 2016]
"""
import os
import numpy as np
import cPickle    
import pandas as pd
from time import time
from os.path import join
from copy import deepcopy

# LMR specific imports
import LMR_proxy_pandas_rework
import LMR_psms
import LMR_calibrate
import LMR_prior
from LMR_utils import coefficient_efficiency, rmsef

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.basemap import Basemap

# =============================================================================
def roundup(x):
    if x <=100:
        n = 1
    elif 1000 > x > 100:
        n = 2
    elif 10000 > x >= 1000:
        n = 3
    else:
        n = 4      
    return int(round(x,-n))



# =============================================================================
# START:  set user parameters here
# =============================================================================

# ------------------------------
# Section 1: Plotting parameters
# ------------------------------

make_plots = False
make_plots_individual_sites = False

# set the default size of the figure in inches. ['figure.figsize'] = width, height;  
plt.rcParams['figure.figsize'] = 9, 7  # that's default image size for this interactive session
plt.rcParams['axes.linewidth'] = 2.0 #set the value globally
plt.rcParams['font.weight'] = 'bold' #set the font weight globally
#plt.rc('text', usetex=True)
plt.rc('text', usetex=False)

# Assign symbol to proxy types for plotting
proxy_verif = {\
    'Tree ring_Width'       :'o',\
    'Tree ring_Density'     :'s',\
    'Ice core_d18O'         :'v',\
    'Ice core_d2H'          :'^',\
    'Ice core_Accumulation' :'D',\
    'Coral_d18O'            :'p',\
    'Coral_Luminescence'    :'8',\
    'Lake sediment_All'     :'<',\
    'Marine sediment_All'   :'>',\
    'Speleothem_All'        :'h',\
    }

class v_core(object):
    """
    High-level parameters of reconstruction experiment

    Attributes
    ----------
    nexp: str
        Name of reconstruction experiment
    lmr_path: str
        Absolute path for the experiment
    verif_period: list(int)
        Time period over which verification is applied
    iter_range: list(int)
        Number of Monte-Carlo iterations to consider
    datadir_input: str
        Absolute path to working directory output for LMR
    datadir_output: str
        Absolute path to directory where verification results are written
    write_full_verif_dict: boolean
        Whether to write the full eval python dictionary to output directory
    """

    #nexp = 'p3rlrc0_CCSM4_LastMillenium_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
    #nexp = 'p3rlrc0_CCSM4_PiControl_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
    #nexp = 'p3rlrc0_GFDLCM3_PiControl_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
    #nexp = 'p3rlrc0_MPIESMP_LastMillenium_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
    #nexp = 'p3rlrc0_20CR_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
    #nexp = 'p3rlrc0_ERA20C_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
    #nexp = 'p3rlrc0_CCSM4_LastMillenium_ens100_cMLOST_allAnnualProxyTypes_pf0.75'
    #nexp = 'p3rlrc0_GFDLCM3_PiControl_ens100_cMLOST_allAnnualProxyTypes_pf0.75'
    #nexp = 'p3rlrc0_MPIESMP_LastMillenium_ens100_cMLOST_allAnnualProxyTypes_pf0.75'
    #nexp = 'p3rlrc0_20CR_ens100_cMLOST_allAnnualProxyTypes_pf0.75'
    #nexp = 'p3rlrc0_ERA20C_ens100_cMLOST_allAnnualProxyTypes_pf0.75'
    #nexp = 'p4rlrc0_CCSM4_LastMillenium_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
    #nexp = 'p4rlrc0_GFDLCM3_PiControl_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
    #
    # -- production recons for paper 1
    #nexp = 'production_gis_ccsm4_pagesall_0.75/'
    #nexp = 'production_mlost_ccsm4_pagesall_0.75/'
    #nexp = 'production_cru_ccsm4_pagesall_0.75/'
    #
    # -- test recons with NCDC proxy data
    #nexp = 't2_2k_CCSM4_LastMillenium_ens100_cMLOST_NCDCproxiesPAGES1_pf0.75'
    #nexp = 't2_2k_CCSM4_LastMillenium_ens100_cGISTEMP_NCDCproxiesPagesTrees_pf0.75'
    #nexp = 't2_2k_CCSM4_LastMillenium_ens100_cGISTEMP_NCDCproxiesBreitTrees_pf0.75'
    nexp = 'TasPrcpPslZW500_2k_CCSM4lm_cGISTEMP_NCDCprxTreesPagesOnly_pf0.75'
    
    
    # lmr_path: where all the data is located ... model (prior), analyses (GISTEMP, HAdCRUT...) and proxies.
    #lmr_path = '/home/chaos2/wperkins/data/LMR'
    lmr_path = '/home/disk/kalman3/rtardif/LMR'

    #verif_period = [0, 2000]
    #verif_period = [0, 1849]
    #verif_period = [1700,1849]
    verif_period = [0, 1879]
    #verif_period = [1880, 2000]

    #iter_range = [0, 0]
    iter_range = [0, 100]

    # Input directory, where to find the reconstruction data
    #datadir_input  = '/home/disk/kalman3/hakim/LMR/'
    #datadir_input  = '/home/disk/kalman2/wperkins/LMR_output/archive' # production recons
    #datadir_input  = '/home/disk/kalman3/rtardif/LMR/output'
    datadir_input  = '/home/disk/ekman4/rtardif/LMR/output'
    
    # Output directory, where the verification results & figs will be dumped.
    datadir_output = datadir_input # if want to keep things tidy
    #datadir_output = '/home/disk/ekman/rtardif/kalman3/LMR/output/verification_production_runs'

    # Whether to write the full eval python dictionary to output directory ("summary" dict. has most of what we want)
    write_full_verif_dict = False


    def __init__(self):
        self.nexp = self.nexp
        self.lmr_path = self.lmr_path
        self.verif_period = self.verif_period
        self.iter_range = self.iter_range
        self.datadir_input = self.datadir_input
        self.datadir_output = self.datadir_output
        self.write_full_verif_dict = self.write_full_verif_dict

        
class v_proxies(object):
    """
    Parameters for proxy data

    Attributes
    ----------
    use_from: list(str)
        A list of keys for proxy classes to load from.  Keys available are
        stored in LMR_proxy_pandas_rework.
    proxy_frac: float
        Fraction of available proxy data (sites) to assimilate
    """

    #use_from = ['pages']
    use_from = ['NCDC']
    proxy_frac = 1.0

    # ---------------
    # PAGES2k proxies
    # ---------------
    class _pages(object):
        """
        Parameters for PagesProxy class

        Attributes
        ----------
        datadir_proxy: str
            Absolute path to proxy data
        datafile_proxy: str
            Absolute path to proxy records file
        metafile_proxy: str
            Absolute path to proxy meta data
        dataformat_proxy: str
            File format of the proxy data
        regions: list(str)
            List of proxy data regions (data keys) to use.
        proxy_resolution: list(float)
            List of proxy time resolutions to use
        proxy_blacklist : list
            A blacklist on proxy records, to eliminate specific records from
            processing
        proxy_order: list(str):
            Order of assimilation by proxy type key
        proxy_assim2: dict{ str: list(str)}
            Proxy types to be assimilated.
            Uses dictionary with structure {<<proxy type>>: [.. list of measuremant
            tags ..] where "proxy type" is written as
            "<<archive type>>_<<measurement type>>"
        proxy_type_mapping: dict{(str,str): str}
            Maps proxy type and measurement to our proxy type keys.
            (e.g. {('Tree ring', 'TRW'): 'Tree ring_Width'} )
        simple_filters: dict{'str': Iterable}
            List mapping Pages2k metadata sheet columns to a list of values
            to filter by.
        """

        datadir_proxy = None
        datafile_proxy = 'Pages2k_Proxies.df.pckl'
        metafile_proxy = 'Pages2k_Metadata.df.pckl'
        dataformat_proxy = 'DF'

        regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
                   'North America', 'South America']
        proxy_resolution = [1.0]
        proxy_timeseries_kind = 'asis' # 'anom' for anomalies (temporal mean removed) or 'asis' to keep unchanged
        

        # DO NOT CHANGE FORMAT BELOW

        proxy_order = ['Tree ring_Width',
                       'Tree ring_Density',
                       'Ice core_d18O',
                       'Ice core_d2H',
                       'Ice core_Accumulation',
                       'Coral_d18O',
                       'Coral_Luminescence',
                       'Lake sediment_All',
                       'Marine sediment_All',
                       'Speleothem_All']

        proxy_assim2 = {
            'Tree ring_Width': ['Ring width',
                                'Tree ring width',
                                'Total ring width',
                                'TRW'],
            'Tree ring_Density': ['Maximum density',
                                  'Minimum density',
                                  'Earlywood density',
                                  'Latewood density',
                                  'MXD'],
            'Ice core_d18O': ['d18O'],
            'Ice core_d2H': ['d2H'],
            'Ice core_Accumulation': ['Accumulation'],
            'Coral_d18O': ['d18O'],
            'Coral_Luminescence': ['Luminescence'],
            'Lake sediment_All': ['Varve thickness',
                                  'Thickness',
                                  'Mass accumulation rate',
                                  'Particle-size distribution',
                                  'Organic matter',
                                  'X-ray density'],
            'Marine sediment_All': ['Mg/Ca'],
            'Speleothem_All': ['Lamina thickness'],
            }

        # A blacklist on proxy records, to prevent assimilation of chronologies known to be duplicates
        proxy_blacklist = []


        def __init__(self):
            if self.datadir_proxy is None:
                self.datadir_proxy = join(v_core.lmr_path, 'data', 'proxies')
            else:
                self.datadir_proxy = self.datadir_proxy
                
            self.datafile_proxy = join(self.datadir_proxy,
                                       self.datafile_proxy)
            self.metafile_proxy = join(self.datadir_proxy,
                                       self.metafile_proxy)

            self.dataformat_proxy = self.dataformat_proxy
            self.regions = list(self.regions)
            self.proxy_resolution = self.proxy_resolution
            self.proxy_timeseries_kind = self.proxy_timeseries_kind
            self.proxy_order = list(self.proxy_order)
            self.proxy_assim2 = deepcopy(self.proxy_assim2)
            self.proxy_blacklist = list(self.proxy_blacklist)
            
            # Create mapping for Proxy Type/Measurement Type to type names above
            self.proxy_type_mapping = {}
            for type, measurements in self.proxy_assim2.iteritems():
                # Fetch proxy type name that occurs before underscore
                type_name = type.split('_', 1)[0]
                for measure in measurements:
                    self.proxy_type_mapping[(type_name, measure)] = type

            self.simple_filters = {'PAGES 2k Region': self.regions,
                                   'Resolution (yr)': self.proxy_resolution}


    # ---------------
    # NCDC proxies
    # ---------------
    class _ncdc(object):
        """
        Parameters for NCDC proxy class

        Attributes
        ----------
        datadir_proxy: str
            Absolute path to proxy data
        datafile_proxy: str
            Absolute path to proxy records file
        metafile_proxy: str
            Absolute path to proxy meta data
        dataformat_proxy: str
            File format of the proxy data
        regions: list(str)
            List of proxy data regions (data keys) to use.
        proxy_resolution: list(float)
            List of proxy time resolutions to use
        proxy_blacklist : list
            A blacklist on proxy records, to eliminate specific records from
            processing
        database_filter: list(str)
            List of databases from which to limit the selection of proxies.
            Use [] (empty list) if no restriction, or ['db_name1', db_name2'] to limit to 
            proxies contained in "db_name1" OR "db_name2". 
            Possible choices are: 'PAGES1', 'PAGES2', 'LMR_FM'
        proxy_order: list(str):
            Order of assimilation by proxy type key
        proxy_assim2: dict{ str: list(str)}
            Proxy types to be assimilated.
            Uses dictionary with structure {<<proxy type>>: [.. list of measuremant
            tags ..] where "proxy type" is written as
            "<<archive type>>_<<measurement type>>"
        proxy_type_mapping: dict{(str,str): str}
            Maps proxy type and measurement to our proxy type keys.
            (e.g. {('Tree ring', 'TRW'): 'Tree ring_Width'} )
        simple_filters: dict{'str': Iterable}
            List mapping proxy metadata sheet columns to a list of values
            to filter by.
        """

        datadir_proxy = None
        datafile_proxy = 'NCDC_Proxies.df.pckl'
        metafile_proxy = 'NCDC_Metadata.df.pckl'
        dataformat_proxy = 'DF'

        regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
                   'North America', 'South America']

        proxy_resolution = [1.0]
        proxy_timeseries_kind = 'asis' # 'anom' for anomalies (temporal mean removed) or 'asis' to keep unchanged
        
        # Limit proxies to those included in the following databases
        #database_filter = ['PAGES1']
        database_filter = []

        # A blacklist on proxy records, to prevent processing of chronologies known to be duplicates.
        proxy_blacklist = []
        #proxy_blacklist = ['00aust01a', '06cook02a', '06cook03a', '09japa01a', '10guad01a', '99aust01a', '99fpol01a']

        
        # DO NOT CHANGE FORMAT BELOW
        proxy_order = [
#            'Tree Rings_All',
            'Tree Rings_WoodDensity',
            'Tree Rings_WidthPages',
            'Tree Rings_WidthBreit',
            'Corals and Sclerosponges_d18O',
#            'Corals and Sclerosponges_d13C',
#            'Corals and Sclerosponges_d14C',
            'Corals and Sclerosponges_SrCa',
#            'Corals and Sclerosponges_BaCa',
#            'Corals and Sclerosponges_CdCa',
#            'Corals and Sclerosponges_MgCa',
#            'Corals and Sclerosponges_UCa',
#            'Corals and Sclerosponges_Sr',
#            'Corals and Sclerosponges_Pb',
            'Ice Cores_d18O',
            'Ice Cores_dD',
            'Ice Cores_Accumulation',
            'Ice Cores_MeltFeature',
            'Lake Cores_Varve',
#            'Speleothems_d18O',
#            'Speleothems_d13C'
            ]

        proxy_assim2 = {
            'Corals and Sclerosponges_d18O': ['d18O','delta18O','d18o','d18O_stk','d18O_int','d18O_norm',
                                              'd18o_avg','d18o_ave','dO18','d18O_4'],
            'Corals and Sclerosponges_d14C': ['d14C','d14c','ac_d14c'],
            'Corals and Sclerosponges_d13C': ['d13C','d13c','d13c_ave','d13c_ann_ave','d13C_int'],
            'Corals and Sclerosponges_SrCa': ['Sr/Ca','Sr/Ca_norm','Sr/Ca_anom','Sr/Ca_int'],
            'Corals and Sclerosponges_Sr'  : ['Sr'],
            'Corals and Sclerosponges_BaCa': ['Ba/Ca'],
            'Corals and Sclerosponges_CdCa': ['Cd/Ca'],
            'Corals and Sclerosponges_MgCa': ['Mg/Ca'],
            'Corals and Sclerosponges_UCa' : ['U/Ca','U/Ca_anom'],
            'Corals and Sclerosponges_Pb'  : ['Pb'],
            'Ice Cores_d18O'               : ['d18O','delta18O','delta18o','d18o','d18o_int','d18O_int','d18O_norm',
                                              'd18o_norm','dO18','d18O_anom'],
            'Ice Cores_dD'                 : ['deltaD','delD'],
            'Ice Cores_Accumulation'       : ['accum','accumu'],
            'Ice Cores_MeltFeature'        : ['MFP'],
            'Lake Cores_Varve'             : ['varve', 'varve_thickness', 'varve thickness'],
            'Speleothems_d18O'             : ['d18O'],
            'Speleothems_d13C'             : ['d13C'],
            'Tree Rings_All'               : ['clim_signal'],
            'Tree Rings_WidthBreit'        : ['trsgi'],
            'Tree Rings_WidthPages'        : ['TRW',
                                              'ERW',
                                              'LRW'],
            'Tree Rings_WoodDensity'       : ['max_d',
                                              'min_d',
                                              'early_d',
                                              'late_d',
                                              'MXD'],
            }


        def __init__(self):
            if self.datadir_proxy is None:
                self.datadir_proxy = join(v_core.lmr_path, 'data', 'proxies')
            else:
                self.datadir_proxy = self.datadir_proxy

            self.datafile_proxy = join(self.datadir_proxy,
                                       self.datafile_proxy)
            self.metafile_proxy = join(self.datadir_proxy,
                                       self.metafile_proxy)

            self.dataformat_proxy = self.dataformat_proxy
            self.regions = list(self.regions)
            self.proxy_resolution = self.proxy_resolution
            self.proxy_timeseries_kind = self.proxy_timeseries_kind
            self.proxy_order = list(self.proxy_order)
            self.proxy_assim2 = deepcopy(self.proxy_assim2)
            self.database_filter = list(self.database_filter)
            self.proxy_blacklist = list(self.proxy_blacklist)

            self.proxy_type_mapping = {}
            for ptype, measurements in self.proxy_assim2.iteritems():
                # Fetch proxy type name that occurs before underscore
                type_name = ptype.split('_', 1)[0]
                for measure in measurements:
                    self.proxy_type_mapping[(type_name, measure)] = ptype

            self.simple_filters = {'Resolution (yr)': self.proxy_resolution}


    # Initialize subclasses with all attributes
    def __init__(self, **kwargs):
        self.use_from = self.use_from
        self.proxy_frac = self.proxy_frac
        self.pages = self._pages(**kwargs)
        self.ncdc = self._ncdc(**kwargs)



class v_psm(object):
    """
    Parameters for PSM classes

    Attributes
    ----------
    use_psm: dict{str: str}
        Maps proxy class key to psm class key.  Used to determine which psm
        is associated with what Proxy type.
    """

    use_psm = {'pages': 'linear', 'NCDC': 'linear'}
    #use_psm = {'pages': 'linear_TorP', 'NCDC': 'linear_TorP'}
    #use_psm = {'pages': 'bilinear', 'NCDC': 'bilinear'}

    
    class _linear(object):
        """
        Parameters for the linear fit PSM.

        Attributes
        ----------
        datatag_calib: str
            Source of calibration data for PSM
        datadir_calib: str
            Absolute path to calibration data
        datafile_calib: str
            Filename for calibration data
        dataformat_calib: str
            Data storage type for calibration data
        pre_calib_datafile: str
            Absolute path to precalibrated Linear PSM data
        psm_r_crit: float
            Usage threshold for correlation of linear PSM
        """

        datadir_calib = None

        # Choice between:
        # --------------
        datatag_calib = 'GISTEMP'
        datafile_calib = 'gistemp1200_ERSST.nc'
        # or
        #datatag_calib = 'MLOST'
        #datafile_calib = 'MLOST_air.mon.anom_V3.5.4.nc'
        # or 
        #datatag_calib = 'HadCRUT'
        #datafile_calib = 'HadCRUT.4.4.0.0.median.nc'
        # or 
        #datatag_calib = 'BerkeleyEarth'
        #datafile_calib = 'Land_and_Ocean_LatLong1.nc'
        # or 
        #datatag_calib = 'GPCC'
        #datafile_calib = 'GPCC_precip.mon.total.1x1.v6.nc'
        # or
        #datatag_calib_P = 'DaiPDSI'
        #datafile_calib_P = 'Dai_pdsi.mon.mean.selfcalibrated_185001-201412.nc'

        
        pre_calib_datafile = None

        psm_r_crit = 0.0


        def __init__(self):
            self.datatag_calib = self.datatag_calib
            self.datafile_calib = self.datafile_calib
            self.psm_r_crit = self.psm_r_crit

            if self.datadir_calib is None:
                self.datadir_calib = join(v_core.lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = self.datadir_calib

            if self.pre_calib_datafile is None:
                filename = 'PSMs_'+'-'.join(v_proxies.use_from)+'_'+self.datatag_calib+'.pckl'
                self.pre_calib_datafile = join(v_core.lmr_path,
                                               'PSM',
                                               filename)
            else:
                self.pre_calib_datafile = self.pre_calib_datafile


    class _linear_TorP(_linear):
        """
        Parameters for the linear fit PSM.

        Attributes
        ----------
        datatag_calib_T: str
            Source of temperature calibration data for linear PSM
        datadir_calib_T: str
            Absolute path to temperature calibration data *or* None if using
            default lmr_path
        datafile_calib_T: str
            Filename for temperature calibration data
        datatag_calib_P: str
            Source of precipitation calibration data for linear PSM
        datadir_calib_P: str
            Absolute path to precipitation calibration data *or* None if using
            default lmr_path
        datafile_calib_P: str
            Filename for precipitation calibration data
        dataformat_calib: str
            Data storage type for calibration data
        pre_calib_datafile_T: str
            Absolute path to precalibrated Linear temperature PSM data
        pre_calib_datafile_P: str
            Absolute path to precalibrated Linear precipitation PSM data
        psm_r_crit: float
            Usage threshold for correlation of linear PSM

        """

        datadir_calib = None
        
        # linear PSM w.r.t. temperature
        # -----------------------------
        # Choice between:
        # ---------------
        datatag_calib_T = 'GISTEMP'
        datafile_calib_T = 'gistemp1200_ERSST.nc'
        # or
        # datatag_calib_T = 'MLOST'
        # datafile_calib_T = 'MLOST_air.mon.anom_V3.5.4.nc'
        # or
        # datatag_calib_T = 'HadCRUT'
        # datafile_calib_T = 'HadCRUT.4.4.0.0.median.nc'
        # or
        # datatag_calib_T = 'BerkeleyEarth'
        # datafile_calib_T = 'Land_and_Ocean_LatLong1.nc'
        #
        # linear PSM w.r.t. precipitation/moisture
        # ----------------------------------------
        # Choice between:
        # ---------------
        # datatag_calib_P = 'GPCC'
        # datafile_calib_P = 'GPCC_precip.mon.total.1x1.v6.nc'
        # or
        datatag_calib_P = 'GPCC'
        datafile_calib_P = 'GPCC_precip.mon.flux.1x1.v6.nc'
        # or
        #datatag_calib_P = 'DaiPDSI'
        #datafile_calib_P = 'Dai_pdsi.mon.mean.selfcalibrated_185001-201412.nc'
        
        dataformat_calib = 'NCD'

        pre_calib_datafile_T = None
        pre_calib_datafile_P = None

        psm_r_crit = 0.0

        def __init__(self):
            self.datatag_calib_T = self.datatag_calib_T
            self.datafile_calib_T = self.datafile_calib_T
            self.datatag_calib_P = self.datatag_calib_P
            self.datafile_calib_P = self.datafile_calib_P
            self.dataformat_calib = self.dataformat_calib
            self.psm_r_crit = self.psm_r_crit

            if self.datadir_calib is None:
                self.datadir_calib = join(v_core.lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = self.datadir_calib

            if self.pre_calib_datafile_T is None:
                filename_t = 'PSMs_' + '-'.join(v_proxies.use_from) + '_' + \
                             self.datatag_calib_T + '.pckl'
                self.pre_calib_datafile_T = join(v_core.lmr_path,
                                                 'PSM',
                                                 filename_t)
            else:
                self.pre_calib_datafile_T = self.pre_calib_datafile_T

            if self.pre_calib_datafile_P is None:
                filename_p = 'PSMs_' + '-'.join(v_proxies.use_from) + '_' + \
                             self.datatag_calib_P + '.pckl'
                self.pre_calib_datafile_P = join(v_core.lmr_path,
                                                 'PSM',
                                                 filename_p)
            else:
                self.pre_calib_datafile_P = self.pre_calib_datafile_P


    class _bilinear(object):
        """
        Parameters for the bilinear fit PSM.

        Attributes
        ----------
        datatag_calib_T: str
            Source of calibration temperature data for PSM
        datadir_calib_T: str
            Absolute path to calibration temperature data
        datafile_calib_T: str
            Filename for calibration temperature data
        dataformat_calib_T: str
            Data storage type for calibration temperature data
        datatag_calib_P: str
            Source of calibration precipitation/moisture data for PSM
        datadir_calib_P: str
            Absolute path to calibration precipitation/moisture data
        datafile_calib_P: str
            Filename for calibration precipitation/moisture data
        dataformat_calib_P: str
            Data storage type for calibration precipitation/moisture data
        pre_calib_datafile: str
            Absolute path to precalibrated Linear PSM data
        psm_r_crit: float
            Usage threshold for correlation of linear PSM
        """

        datadir_calib = None

        # calibration w.r.t. temperature
        # -----------------------------
        # Choice between:
        #
        datatag_calib_T = 'GISTEMP'
        datafile_calib_T = 'gistemp1200_ERSST.nc'
        # or
        #datatag_calib_T = 'MLOST'
        #datafile_calib_T = 'MLOST_air.mon.anom_V3.5.4.nc'
        # or 
        #datatag_calib_T = 'HadCRUT'
        #datafile_calib_T = 'HadCRUT.4.4.0.0.median.nc'
        # or 
        #datatag_calib_T = 'BerkeleyEarth'
        #datafile_calib_T = 'Land_and_Ocean_LatLong1.nc'

        # calibration w.r.t. precipitation/moisture
        # ----------------------------------------
        #datatag_calib_P = 'GPCC'
        #datafile_calib_P = 'GPCC_precip.mon.total.1x1.v6.nc'
        # or 
        datatag_calib_P = 'GPCC'
        datafile_calib_P = 'GPCC_precip.mon.flux.1x1.v6.nc'
        # or
        #datatag_calib_P = 'DaiPDSI'
        #datafile_calib_P = 'Dai_pdsi.mon.mean.selfcalibrated_185001-201412.nc'


        pre_calib_datafile = None

        psm_r_crit = 0.0
         

        def __init__(self):
            self.datatag_calib_T = self.datatag_calib_T
            self.datafile_calib_T = self.datafile_calib_T
            self.datatag_calib_P = self.datatag_calib_P
            self.datafile_calib_P = self.datafile_calib_P
            self.psm_r_crit = self.psm_r_crit

            if self.datadir_calib is None:
                self.datadir_calib = join(v_core.lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = self.datadir_calib


            if self.pre_calib_datafile is None:
                filename = 'PSMs_'+'-'.join(v_proxies.use_from)+'_'+self.datatag_calib_T+'_'+self.datatag_calib_P+'.pckl'
                self.pre_calib_datafile = join(v_core.lmr_path,
                                               'PSM',
                                               filename)
            else:
                self.pre_calib_datafile = self.pre_calib_datafile
    


                
    
    # Initialize subclasses with all attributes
    def __init__(self, **kwargs):
        self.use_psm = self.use_psm
        self.linear = self._linear(**kwargs)
        self.linear_TorP = self._linear_TorP(**kwargs)
        self.bilinear = self._bilinear(**kwargs)
                

# =============================================================================
# END:  set user parameters here
# =============================================================================
class config(object):
    def __init__(self,core,proxies,psm):
        self.core = core()
        self.proxies = proxies()
        self.psm = psm()

Cfg = config(v_core, v_proxies,v_psm)

#==============================================================================


# =============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Main code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# =============================================================================
def main():


    #
    # TODO: re-define "verif" proxies as those in the database (all proxy objects given by "load_all") that are NOT in the assimilated set
    # 


    
    verbose = 1

    complement_set = True
    
    begin_time = time()

    proxy_database = Cfg.proxies.use_from[0]
    # get psm type (linear, linear_torP, bilinear, h_interp)
    psm_type = Cfg.psm.use_psm[proxy_database]
    # import related psm class
    psm_class =  LMR_psms.get_psm_class(psm_type)
    psm_kwargs = psm_class.get_kwargs(Cfg)

    verif_period = Cfg.core.verif_period
    
    print 'Proxies             :', proxy_database
    print 'PSM type            :', psm_type
    print 'Verif. period       :', verif_period

    if proxy_database == 'pages':
        print 'Proxy data location :', Cfg.proxies.pages.datadir_proxy
    elif proxy_database == 'NCDC':
        print 'Proxy data location :', Cfg.proxies.ncdc.datadir_proxy
    else:
        print 'ERROR in specification of proxy database. Exiting!'
        exit(1)

    # get proxy class
    proxy_class = LMR_proxy_pandas_rework.get_proxy_class(proxy_database)
    # load proxy data
    proxy_ids_by_grp, proxy_objects = proxy_class.load_all(Cfg,
                                                       verif_period,
                                                       **psm_kwargs)

    master_proxy_list = []
    for proxy_idx, Y in enumerate(proxy_objects): master_proxy_list.append((Y.type,Y.id))
    
    print '--------------------------------------------------------------------'
    print 'Uploaded proxies : counts per proxy type:'
    # count the total number of proxies
    total_proxy_count = len(proxy_objects)
    for pkey, plist in proxy_ids_by_grp.iteritems():
        print('%45s : %5d' % (pkey, len(plist)))
    print('%45s : %5d' % ('TOTAL', total_proxy_count))
    print '--------------------------------------------------------------------'
    print ' '

    """
    print psm_kwargs.keys()
    print ' '
    print psm_kwargs['psm_data']
    print ' '
    """
    #print psm_kwargs['psm_data'][('Tree Rings_WidthBreit', 'europe_pola017B:trsgi')]
    #print psm_kwargs['psm_data_T'][('Tree Rings_WidthBreit', 'europe_pola017B:trsgi')]
    #print psm_kwargs['psm_data_P'][('Tree Rings_WidthBreit', 'europe_pola017B:trsgi')]
        

    # psm type
    if psm_type == 'linear':
        print 'Calibration source  :', Cfg.psm.linear.datatag_calib
        datatag_calib = 'linear_'+Cfg.psm.linear.datatag_calib
        psm_file = Cfg.psm.linear.pre_calib_datafile
        required_state_variables = ['tas_sfc_Amon']
    elif psm_type == 'linear_TorP':
        print 'Calibration sources :', Cfg.psm.linear_TorP.datatag_calib_T, '+', Cfg.psm.linear_TorP.datatag_calib_P
        datatag_calib_T = Cfg.psm.linear_TorP.datatag_calib_T
        datatag_calib_P = Cfg.psm.linear_TorP.datatag_calib_P
        datatag_calib = 'linear_'+datatag_calib_T+'or'+datatag_calib_P
        #psm_file = Cfg.psm.bilinear.pre_calib_datafile
        if datatag_calib_P == 'GPCC':
            required_state_variables = ['tas_sfc_Amon','pr_sfc_Amon']
        elif datatag_calib_P == 'DaiPDSI':
            required_state_variables = ['tas_sfc_Amon','scpdsi_sfc_Amon']
        else:
            print 'Unrecognized value for datatag_calib_P. Exiting!'
            exit(1)
    elif psm_type == 'bilinear':
        print 'Calibration sources :', Cfg.psm.bilinear.datatag_calib_T, '+', Cfg.psm.bilinear.datatag_calib_P
        datatag_calib_T = Cfg.psm.bilinear.datatag_calib_T
        datatag_calib_P = Cfg.psm.bilinear.datatag_calib_P
        datatag_calib = 'bilinear_'+datatag_calib_T+'and'+datatag_calib_P
        psm_file = Cfg.psm.bilinear.pre_calib_datafile
        if datatag_calib_P == 'GPCC':
            required_state_variables = ['tas_sfc_Amon','pr_sfc_Amon']
        elif datatag_calib_P == 'DaiPDSI':
            required_state_variables = ['tas_sfc_Amon','scpdsi_sfc_Amon']
        else:
            print 'Unrecognized value for datatag_calib_P. Exiting!'
            exit(1)
    elif psm_type == 'h_interp':
        datatag_calib = 'interp'
        required_state_variables = ['d18O_sfc_Amon']
    else:
        print 'ERROR: problem with the type of psm!'
        exit(1)

    load_time = time() - begin_time
    print '======================================================='
    print 'Loading completed in '+ str(load_time/60.0)+' mins'
    print '======================================================='

    
    # ==========================================================================
    # Loop over the Monte-Carlo reconstructions
    # ==========================================================================
    nexp = Cfg.core.nexp

    datadir_input = Cfg.core.datadir_input
    datadir_output = Cfg.core.datadir_output

    verif_dict = []
    assim_dict = []

    Nbiter = Cfg.core.iter_range[1]
    MCiters = np.arange(Cfg.core.iter_range[0], Cfg.core.iter_range[1]+1)
    for iter in MCiters:

        # Experiment data directory
        workdir = datadir_input+'/'+nexp+'/r'+str(iter)

        print '==> Working on: ' + workdir

        
        # Information on assimilated and non-assimilated proxies
        # ------------------------------------------------------

        # List of assimilated proxies
        # ---------------------------
        loaded_proxies = np.load(workdir+'/'+'assimilated_proxies.npy')
        nb_loaded_proxies = len(loaded_proxies)

        # Filter to match proxy sites in set of calibrated PSMs
        indp = [k for k in range(nb_loaded_proxies) if (loaded_proxies[k].keys()[0],loaded_proxies[k][loaded_proxies[k].keys()[0]][0]) in master_proxy_list]
        assim_proxies = loaded_proxies[indp]
        nb_assim_proxies = len(assim_proxies)

        # in list form
        assim_proxies_list = [(assim_proxies[k].keys()[0],assim_proxies[k][assim_proxies[k].keys()[0]][0]) for k in range(nb_assim_proxies)]

        print '------------------------------------------------'
        print 'Number of proxy sites in assimilated set :',  nb_assim_proxies


        # List of non-assimilated proxies
        # -------------------------------
        # complement_set: option to use proxies in master_proxy_list that are NOT in assimilated set as the verification set,
        #                 rather than those found in the recon "nonassimilated_proxies.npy" file ... ... ...

        if complement_set:
            verif_proxies_list = list(set(master_proxy_list) - set(assim_proxies_list))
            nb_verif_proxies = len(verif_proxies_list)
        else:            
            loaded_proxies = np.load(workdir+'/'+'nonassimilated_proxies.npy')
            nb_loaded_proxies = len(loaded_proxies)

            # Filter to match proxy sites in set of calibrated PSMs
            indp = [k for k in range(nb_loaded_proxies) if (loaded_proxies[k].keys()[0],loaded_proxies[k][loaded_proxies[k].keys()[0]][0]) in master_proxy_list]
            verif_proxies = loaded_proxies[indp]
            nb_verif_proxies = len(verif_proxies)
            
            # in list form
            verif_proxies_list = [(verif_proxies[k].keys()[0],verif_proxies[k][verif_proxies[k].keys()[0]][0]) for k in range(nb_verif_proxies)]
        
        print 'Number of proxy sites in verification set:',  nb_verif_proxies
        print '------------------------------------------------'

        # ------------------------------------------------------
        # Load in the prior data
        # ------------------------------------------------------
        file_prior = workdir+'/Xb_one.npz'
        Xprior_statevector = np.load(file_prior)
        state_info = Xprior_statevector['state_info'].item()
        available_state_variables = state_info.keys()

        
        # -------------------------------------------------------
        # Load in the ensemble-mean reconstruction data
        # -------------------------------------------------------

        # extract required data from state vector
        nb_state_var = len(required_state_variables)
        
        for state_var in required_state_variables:
            # Check for presence of file containing reconstruction of state variable
            filein = workdir+'/ensemble_mean_'+state_var+'.npz'
            if not os.path.isfile(filein):
                print 'ERROR in specification of reconstruction data'
                print 'File ', filein, ' does not exist! - Exiting!'
                exit(1)

            Xrecon = np.load(filein)
            recon_times = np.asarray(Xrecon['years'],dtype='int32')          
            recon_lat = Xrecon['lat']
            recon_lon = Xrecon['lon']
            recon_val = Xrecon['xam']

            lat =  recon_lat[:,0]
            lon =  recon_lon[0,:]
            annual_data = recon_val


            # prior
            posbeg = state_info[state_var]['pos'][0]
            posend = state_info[state_var]['pos'][1]
            Xb_one = Xprior_statevector['Xb_one']
        
            nlat = state_info[state_var]['spacedims'][0]
            nlon = state_info[state_var]['spacedims'][1]
            var_prior = Xb_one[posbeg:posend+1,:]
            [_,Nens] = var_prior.shape
            Xprior = var_prior.reshape(nlat,nlon,Nens)
            # Broadcast prior ensemble state over time dimension
            ntime = annual_data.shape[0]
            tmp = np.repeat(Xprior[None,:,:,:],ntime,axis=0)
            Xprior_ens = np.rollaxis(tmp,3,0)
            
            print state_var, Xprior.shape, annual_data.shape, Xprior_ens.shape


        # -------------------------------------------------------
        # Load the full ensemble Ye's updated during assimilation
        # (appended state vector)
        # -------------------------------------------------------
        fnameYe = workdir+'/'+'analysis_Ye.pckl'
        try:
            assimYe.clear()
        except:
            pass
        if os.path.isfile(fnameYe):
            infile  = open(fnameYe,'r')
            assimYe = cPickle.load(infile)
            infile.close()


        # ----------------------------------------------------------
        # loop over proxy objects and calculate proxy estimates (Ye)
        # from reconstruction and from prior
        # ----------------------------------------------------------
        pcount_tot   = 0
        pcount_verif = 0
        pcount_assim = 0
        evald_verif = {}
        evald_assim = {}
        for i, pobj in enumerate(proxy_objects):
            sitetag = (pobj.type,pobj.id)
            pstatus = None            
            
            if (sitetag in verif_proxies_list) or (sitetag in assim_proxies_list):

                # reconstruction (ensemble mean) Ye
                recon_dat = pobj.psm_obj.get_close_grid_point_data(annual_data,
                                                                   lon,lat)
                ye_recon = pobj.psm_obj.basic_psm(recon_dat)
                
                # prior (ensemble mean) Ye 
                xp = np.mean(Xprior,axis=2)
                Xprior_ensmean = np.repeat(xp[None,:,:],ntime,axis=0)
                prior_dat = pobj.psm_obj.get_close_grid_point_data(Xprior_ensmean,
                                                                   lon,lat)
                ye_prior = pobj.psm_obj.basic_psm(prior_dat)

                # Ye from prior full ensemble ...
                prior_dat_ens = pobj.psm_obj.get_close_grid_point_data(Xprior_ens[0:,:,:,:],
                                                                       lon,lat)
                prior_dat_ens = np.rollaxis(prior_dat_ens,1,0)
                ye_prior_ens = pobj.psm_obj.basic_psm(prior_dat_ens[:,0:])

                
                # assign 'verif' or 'assim' status to proxy record
                if sitetag in verif_proxies_list:
                    pcount_verif += 1
                    pstatus = 'verif'
                if sitetag in assim_proxies_list:
                    pcount_assim += 1
                    pstatus = 'assim'

            if pstatus:
                # Use pandas DataFrame to store proxy & Ye data side-by-side
                headers = ['time', 'Ye_recon', 'Ye_prior'] + ['Ye_prior_%s' %str(k+1) for k in range(Nens)]
                
                # Merge Ye values from recon (ensemble-mean), prior(ensemble-mean) and prior (full sensemble)
                # in single array & convert to dataframe
                ye_data = np.c_[recon_times.T,ye_recon.T,ye_prior.T,ye_prior_ens]
                df = pd.DataFrame(ye_data)
                df.columns = headers

                # Add proxy data
                frame = pd.DataFrame({'time':pobj.time, 'y': pobj.values})
                df = df.merge(frame, how='outer', on='time')

                # define dataframe indexing
                col0 = df.columns[0]
                df.set_index(col0, drop=True, inplace=True)
                df.index.name = 'time'
                df.sort_index(inplace=True)
                
                # Reconstruction (ensemble-mean) error
                df['Ye_recon_error'] = df['Ye_recon'] - df['y']
                # Prior  (ensemble-mean) error
                df['Ye_prior_error'] = df['Ye_prior'] - df['y']
            
                obcount = df['Ye_recon_error'].count()
                if obcount < 10:
                    continue

                pcount_tot += 1
                indok = df['y'].notnull()

                if verbose > 0:
                    print '================================================'
                    print 'Site:', sitetag
                    print 'status:', pstatus
                    print 'Number of verification points  :', obcount            
                    print 'Mean of proxy values           :', np.mean(df['y'][indok])
                    print 'Mean ensemble-mean             :', np.mean(df['Ye_recon'][indok])
                    print 'Mean ensemble-mean error       :', np.mean(df['Ye_recon_error'][indok])
                    print 'Ensemble-mean RMSE             :', rmsef(df['Ye_recon'][indok],df['y'][indok])
                    print 'Correlation                    :', np.corrcoef(df['y'][indok],df['Ye_recon'][indok])[0,1]
                    print 'CE                             :', coefficient_efficiency(df['y'][indok],df['Ye_recon'][indok])
                    print 'Mean ensemble-mean(prior)      :', np.mean(df['Ye_prior'][indok])
                    print 'Mean ensemble-mean error(prior):', np.mean(df['Ye_prior_error'][indok])
                    print 'Ensemble-mean RMSE(prior)      :', rmsef(df['Ye_prior'][indok],df['y'][indok])
                    corr = np.corrcoef(df['y'][indok],df['Ye_prior'][indok])[0,1]
                    if not np.isfinite(corr): corr = 0.0
                    print 'Correlation (prior)            :', corr
                    print 'CE (prior)                     :', coefficient_efficiency(df['y'][indok],df['Ye_prior'][indok])
                    print '================================================'            

                
                # Fill "verif" and "assim" dictionaries with data generated above
                if pstatus == 'assim':
                    evald_assim[sitetag] = {}

                    evald_assim[sitetag]['MCiter'] = iter
                    # Site info
                    evald_assim[sitetag]['lat'] = pobj.lat
                    evald_assim[sitetag]['lon'] = pobj.lon
                    evald_assim[sitetag]['alt'] = pobj.elev
                    # PSM info
                    evald_assim[sitetag]['PSMinfo'] = pobj.psm_obj.__dict__
                    # Stats data
                    evald_assim[sitetag]['NbEvalPts']         = obcount
                    evald_assim[sitetag]['EnsMean_MeanError'] = np.mean(df['Ye_recon_error'][indok])
                    evald_assim[sitetag]['EnsMean_RMSE']      = rmsef(df['Ye_recon'][indok],df['y'][indok])
                    evald_assim[sitetag]['EnsMean_Corr']      = np.corrcoef(df['y'][indok],df['Ye_recon'][indok])[0,1]
                    evald_assim[sitetag]['EnsMean_CE']        = coefficient_efficiency(df['y'][indok],df['Ye_recon'][indok])
                    corr = np.corrcoef(df['y'][indok],df['Ye_prior'][indok])[0,1]
                    if not np.isfinite(corr): corr = 0.0
                    evald_assim[sitetag]['PriorEnsMean_Corr'] = corr
                    evald_assim[sitetag]['PriorEnsMean_CE']   = coefficient_efficiency(df['y'][indok],df['Ye_prior'][indok])
                    evald_assim[sitetag]['ts_years']          = df.index[indok].values
                    evald_assim[sitetag]['ts_ProxyValues']    = df['y'][indok].values
                    evald_assim[sitetag]['ts_EnsMean']        = df['Ye_recon'][indok].values

                    # ... assim_Ye info ...
                    Ntime = len(pobj.time)
                    truth = np.zeros(shape=[Ntime]) 
                    R = assimYe[sitetag]['R']
                    [_,Nens] = assimYe[sitetag]['HXa'].shape
                    YeFullEns = np.zeros(shape=[Ntime,Nens])
                    YeFullEns_error = np.zeros(shape=[Ntime,Nens])
                    Ye_prior_FullEns = np.zeros(shape=[Ntime,Nens])
                    YePriorFullEns_error = np.zeros(shape=[Ntime,Nens])

                    # extract prior full ensemble from dataframe
                    Ye_prior_FullEns = df.ix[indok,2:Nens+2].values
                    
                    Ye_time = assimYe[sitetag]['years']
                    obcount = 0
                    for t in pobj.time:
                        truth[obcount] = pobj.values[t]
                        # array time index for recon_values
                        indt_recon = np.where(Ye_time==t)
                        YeFullEns[obcount,:] = assimYe[sitetag]['HXa'][indt_recon]
                        YeFullEns_error[obcount,:] = YeFullEns[obcount,:]-truth[obcount,None] # i.e. full ensemble innovations
                        YePriorFullEns_error[obcount,:] = Ye_prior_FullEns[obcount,:]-truth[obcount,None]
                        obcount = obcount + 1

                    # Reconstruction
                    mse = np.mean(np.square(YeFullEns_error),axis=1)
                    varYe = np.var(YeFullEns,axis=1,ddof=1)
                    # time series of DA ensemble calibration ratio
                    evald_assim[sitetag]['ts_DAensCalib']   =  mse/(varYe+R)

                    # Prior
                    mse = np.mean(np.square(YePriorFullEns_error),axis=1)
                    varYe = np.var(Ye_prior_FullEns,axis=1,ddof=1)
                    # time series of prior ensemble calibration ratio
                    evald_assim[sitetag]['ts_PriorEnsCalib']   =  mse/(varYe+R)

                    
                elif pstatus == 'verif':
                    evald_verif[sitetag] = {}

                    evald_verif[sitetag]['MCiter'] = iter
                    # Site info
                    evald_verif[sitetag]['lat'] = pobj.lat
                    evald_verif[sitetag]['lon'] = pobj.lon
                    evald_verif[sitetag]['alt'] = pobj.elev
                    # PSM info
                    evald_verif[sitetag]['PSMinfo'] = pobj.psm_obj.__dict__
                    # Stats data
                    evald_verif[sitetag]['NbEvalPts']         = obcount
                    evald_verif[sitetag]['EnsMean_MeanError'] = np.mean(df['Ye_recon_error'][indok])
                    evald_verif[sitetag]['EnsMean_RMSE']      = rmsef(df['Ye_recon'][indok],df['y'][indok])
                    evald_verif[sitetag]['EnsMean_Corr']      = np.corrcoef(df['y'][indok],df['Ye_recon'][indok])[0,1]
                    evald_verif[sitetag]['EnsMean_CE']        = coefficient_efficiency(df['y'][indok],df['Ye_recon'][indok])
                    corr = np.corrcoef(df['y'][indok],df['Ye_prior'][indok])[0,1]
                    if not np.isfinite(corr): corr = 0.0
                    evald_verif[sitetag]['PriorEnsMean_Corr'] = corr
                    evald_verif[sitetag]['PriorEnsMean_CE']   = coefficient_efficiency(df['y'][indok],df['Ye_prior'][indok])
                    evald_verif[sitetag]['ts_years']          = df.index[indok].values
                    evald_verif[sitetag]['ts_ProxyValues']    = df['y'][indok].values
                    evald_verif[sitetag]['ts_EnsMean']        = df['Ye_recon'][indok].values

                else:
                    print 'pstatus undefined...'

        

        verif_dict.append(evald_verif)
        assim_dict.append(evald_assim)

    
    # ==========================================================================
    # End of loop on MC iterations => Now calculate summary statistics ---------
    # ==========================================================================

    outdir = datadir_output+'/'+nexp+'/verifProxy_PSM_'+datatag_calib+'_'+str(verif_period[0])+'to'+str(verif_period[1])
    if not os.path.isdir(outdir):
        os.system('mkdir %s' % outdir)
    
    # -----------------------
    # With *** verif_dict ***
    # -----------------------
    if Cfg.core.write_full_verif_dict:
        # Dump dictionary to pickle files
        outfile = open('%s/reconstruction_eval_verif_proxy_full.pckl' % (outdir),'w')
        cPickle.dump(verif_dict,outfile)
        outfile.close()

    # For each site :    
    # List of sites in the verif dictionary
    list_tmp = []
    for i in range(len(verif_dict)):
        for j in range(len(verif_dict[i].keys())):
            list_tmp.append(verif_dict[i].keys()[j])
    list_sites = list(set(list_tmp)) # filter to unique elements
    
    summary_stats_verif = {}
    for k in range(len(list_sites)):
        # indices in verif_dict where this site is present
        inds  = [j for j in range(len(verif_dict)) if list_sites[k] in verif_dict[j].keys()]

        
        summary_stats_verif[list_sites[k]] = {}
        summary_stats_verif[list_sites[k]]['lat']            = verif_dict[inds[0]][list_sites[k]]['lat']
        summary_stats_verif[list_sites[k]]['lon']            = verif_dict[inds[0]][list_sites[k]]['lon']
        summary_stats_verif[list_sites[k]]['alt']            = verif_dict[inds[0]][list_sites[k]]['alt']
        summary_stats_verif[list_sites[k]]['NbPts']          = len(inds)
        
        #summary_stats_verif[list_sites[k]]['PSMslope']       = verif_dict[inds[0]][list_sites[k]]['PSMslope']
        #summary_stats_verif[list_sites[k]]['PSMintercept']   = verif_dict[inds[0]][list_sites[k]]['PSMintercept']
        #summary_stats_verif[list_sites[k]]['PSMcorrel']      = verif_dict[inds[0]][list_sites[k]]['PSMcorrel']
        summary_stats_verif[list_sites[k]]['PSMinfo']      = verif_dict[inds[0]][list_sites[k]]['PSMinfo']
        
        
        # These contain data for the "grand ensemble" (i.e. ensemble of realizations) for "kth" site
        nbpts = [verif_dict[j][list_sites[k]]['NbEvalPts'] for j in inds]
        me    = [verif_dict[j][list_sites[k]]['EnsMean_MeanError'] for j in inds]
        rmse  = [verif_dict[j][list_sites[k]]['EnsMean_RMSE'] for j in inds]
        corr  = [verif_dict[j][list_sites[k]]['EnsMean_Corr'] for j in inds]
        ce    = [verif_dict[j][list_sites[k]]['EnsMean_CE'] for j in inds]
        corr_prior  = [verif_dict[j][list_sites[k]]['PriorEnsMean_Corr'] for j in inds]
        ce_prior    = [verif_dict[j][list_sites[k]]['PriorEnsMean_CE'] for j in inds]

        # Scores on grand ensemble
        summary_stats_verif[list_sites[k]]['GrandEnsME']        = me
        summary_stats_verif[list_sites[k]]['GrandEnsRMSE']      = rmse
        summary_stats_verif[list_sites[k]]['GrandEnsCorr']      = corr
        summary_stats_verif[list_sites[k]]['GrandEnsCE']        = ce
        # prior
        summary_stats_verif[list_sites[k]]['PriorGrandEnsCorr'] = corr_prior
        summary_stats_verif[list_sites[k]]['PriorGrandEnsCE']   = ce_prior

        # Summary across grand ensemble
        summary_stats_verif[list_sites[k]]['MeanME']          = np.mean(me)
        summary_stats_verif[list_sites[k]]['SpreadME']        = np.std(me)
        summary_stats_verif[list_sites[k]]['MeanRMSE']        = np.mean(rmse)
        summary_stats_verif[list_sites[k]]['SpreadRMSE']      = np.std(rmse)
        summary_stats_verif[list_sites[k]]['MeanCorr']        = np.mean(corr)
        summary_stats_verif[list_sites[k]]['SpreadCorr']      = np.std(corr)
        summary_stats_verif[list_sites[k]]['MeanCE']          = np.mean(ce)
        summary_stats_verif[list_sites[k]]['SpreadCE']        = np.std(ce)
        # prior
        summary_stats_verif[list_sites[k]]['PriorMeanCorr']   = np.nanmean(corr_prior)
        summary_stats_verif[list_sites[k]]['PriorSpreadCorr'] = np.nanstd(corr_prior)
        summary_stats_verif[list_sites[k]]['PriorMeanCE']     = np.mean(ce_prior)
        summary_stats_verif[list_sites[k]]['PriorSpreadCE']   = np.std(ce_prior)

        # for time series
        summary_stats_verif[list_sites[k]]['ts_years']       = verif_dict[inds[0]][list_sites[k]]['ts_years']
        summary_stats_verif[list_sites[k]]['ts_ProxyValues'] = verif_dict[inds[0]][list_sites[k]]['ts_ProxyValues']
        ts_recon = [verif_dict[j][list_sites[k]]['ts_EnsMean'] for j in inds]
        summary_stats_verif[list_sites[k]]['ts_MeanRecon']   = np.mean(ts_recon,axis=0)
        summary_stats_verif[list_sites[k]]['ts_SpreadRecon'] = np.std(ts_recon,axis=0)
        # ens. calibration
        R = [verif_dict[inds[0]][list_sites[k]]['PSMinfo']['R']]
        ensVar = np.mean(np.var(ts_recon,axis=0,ddof=1)) # !!! variance in grand ensemble (realizations, not DA ensemble) !!! 
        mse = np.mean(np.square(rmse))
        calib = mse/(ensVar+R)
        summary_stats_verif[list_sites[k]]['EnsCalib'] = calib[0]
        ## without R
        #calib = mse/(ensVar)
        #summary_stats_verif[list_sites[k]]['EnsCalib'] = calib

    # Dump data to pickle file
    outfile = open('%s/reconstruction_eval_verif_proxy_summary.pckl' % (outdir),'w')
    cPickle.dump(summary_stats_verif,outfile)
    outfile.close()

    # -----------------------
    # With *** assim_dict ***
    # -----------------------
    # Dump dictionary to pickle files
    if Cfg.core.write_full_verif_dict:
        outfile = open('%s/reconstruction_eval_assim_proxy_full.pckl' % (outdir),'w')
        cPickle.dump(assim_dict,outfile)
        outfile.close()

    # For each site :    
    # List of sites in the assim dictionary
    list_tmp = []
    for i in range(len(assim_dict)):
        for j in range(len(assim_dict[i].keys())):
            list_tmp.append(assim_dict[i].keys()[j])
    list_sites = list(set(list_tmp)) # filter to unique elements

    summary_stats_assim = {}
    for k in range(len(list_sites)):
        # indices in assim_dict where this site is present
        inds  = [j for j in range(len(assim_dict)) if list_sites[k] in assim_dict[j].keys()]

        summary_stats_assim[list_sites[k]] = {}
        summary_stats_assim[list_sites[k]]['lat']            = assim_dict[inds[0]][list_sites[k]]['lat']
        summary_stats_assim[list_sites[k]]['lon']            = assim_dict[inds[0]][list_sites[k]]['lon']
        summary_stats_assim[list_sites[k]]['alt']            = assim_dict[inds[0]][list_sites[k]]['alt']
        summary_stats_assim[list_sites[k]]['NbPts']          = len(inds)
        
        #summary_stats_assim[list_sites[k]]['PSMslope']       = assim_dict[inds[0]][list_sites[k]]['PSMslope']
        #summary_stats_assim[list_sites[k]]['PSMintercept']   = assim_dict[inds[0]][list_sites[k]]['PSMintercept']
        #summary_stats_assim[list_sites[k]]['PSMcorrel']      = assim_dict[inds[0]][list_sites[k]]['PSMcorrel']
        summary_stats_assim[list_sites[k]]['PSMinfo']      = assim_dict[inds[0]][list_sites[k]]['PSMinfo']

        # These contain data for the "grand ensemble" (i.e. ensemble of realizations) for "kth" site
        nbpts      = [assim_dict[j][list_sites[k]]['NbEvalPts'] for j in inds]
        me         = [assim_dict[j][list_sites[k]]['EnsMean_MeanError'] for j in inds]
        rmse       = [assim_dict[j][list_sites[k]]['EnsMean_RMSE'] for j in inds]
        corr       = [assim_dict[j][list_sites[k]]['EnsMean_Corr'] for j in inds]
        ce         = [assim_dict[j][list_sites[k]]['EnsMean_CE'] for j in inds]
        corr_prior = [assim_dict[j][list_sites[k]]['PriorEnsMean_Corr'] for j in inds]
        ce_prior   = [assim_dict[j][list_sites[k]]['PriorEnsMean_CE'] for j in inds]
        DAensCalib = [np.mean(assim_dict[j][list_sites[k]]['ts_DAensCalib']) for j in inds]
        PriorEnsCalib = [np.mean(assim_dict[j][list_sites[k]]['ts_PriorEnsCalib']) for j in inds]


        # Scores on grand ensemble
        summary_stats_assim[list_sites[k]]['GrandEnsME']        = me
        summary_stats_assim[list_sites[k]]['GrandEnsRMSE']      = rmse
        summary_stats_assim[list_sites[k]]['GrandEnsCorr']      = corr
        summary_stats_assim[list_sites[k]]['GrandEnsCE']        = ce
        summary_stats_assim[list_sites[k]]['GrandEnsCalib']     = DAensCalib
        # prior
        summary_stats_assim[list_sites[k]]['PriorGrandEnsCorr'] = corr_prior
        summary_stats_assim[list_sites[k]]['PriorGrandEnsCE']   = ce_prior
        summary_stats_assim[list_sites[k]]['PriorGrandEnsCalib']= PriorEnsCalib

        # Summary across grand ensemble
        summary_stats_assim[list_sites[k]]['MeanME']          = np.mean(me)
        summary_stats_assim[list_sites[k]]['SpreadME']        = np.std(me)
        summary_stats_assim[list_sites[k]]['MeanRMSE']        = np.mean(rmse)
        summary_stats_assim[list_sites[k]]['SpreadRMSE']      = np.std(rmse)
        summary_stats_assim[list_sites[k]]['MeanCorr']        = np.mean(corr)
        summary_stats_assim[list_sites[k]]['SpreadCorr']      = np.std(corr)
        summary_stats_assim[list_sites[k]]['MeanCE']          = np.mean(ce)
        summary_stats_assim[list_sites[k]]['SpreadCE']        = np.std(ce)
        # prior
        summary_stats_assim[list_sites[k]]['PriorMeanCorr']   = np.nanmean(corr_prior)
        summary_stats_assim[list_sites[k]]['PriorSpreadCorr'] = np.nanstd(corr_prior)
        summary_stats_assim[list_sites[k]]['PriorMeanCE']     = np.mean(ce_prior)
        summary_stats_assim[list_sites[k]]['PriorSpreadCE']   = np.std(ce_prior)

        # for time series
        summary_stats_assim[list_sites[k]]['ts_years']       = assim_dict[inds[0]][list_sites[k]]['ts_years']
        summary_stats_assim[list_sites[k]]['ts_ProxyValues'] = assim_dict[inds[0]][list_sites[k]]['ts_ProxyValues']
        ts_recon = [assim_dict[j][list_sites[k]]['ts_EnsMean'] for j in inds]
        summary_stats_assim[list_sites[k]]['ts_MeanRecon']   = np.mean(ts_recon,axis=0)
        summary_stats_assim[list_sites[k]]['ts_SpreadRecon'] = np.std(ts_recon,axis=0)        
        # ens. calibration
        R = [assim_dict[inds[0]][list_sites[k]]['PSMinfo']['R']]
        ensVar = np.mean(np.var(ts_recon,axis=0,ddof=1)) # !!! variance of ens. means in grand ensemble (realizations, not DA ensemble) !!!
        mse = np.mean(np.square(rmse))
        calib = mse/(ensVar+R)
        summary_stats_assim[list_sites[k]]['EnsCalib'] = calib[0]
        

    # Dump data to pickle file
    outfile = open('%s/reconstruction_eval_assim_proxy_summary.pckl' % (outdir),'w')
    cPickle.dump(summary_stats_assim,outfile)
    outfile.close()

    verif_time = time() - begin_time
    print '======================================================='
    print 'Verification completed in '+ str(verif_time/3600.0)+' hours'
    print '======================================================='











    

    # ============================================
    # Plotting the verification summary stats data
    # ============================================
    if make_plots:

        print 'Now producing plots ...'

        figdir = outdir+'/figs'        
        if not os.path.isdir(figdir):
            os.system('mkdir %s' % figdir)
        
        vtype = {'verif':'Non-assimilated proxies', 'assim': 'Assimilated proxies'}
        for v in vtype.keys():
            
            # pick right dict and associate to "workdict"
            dname = 'summary_stats_'+v
            workdict = eval(dname)
            sitetag = workdict.keys()

            site_status = vtype[v].split()[0]

            # ===============================================================================================
            # 1) Histogram of (recon, proxy) CORRELATION, CE across grand ensemble for all and per proxy type
            # ===============================================================================================

            proxy_types = list(set([item[0] for item in sitetag]))

            # --------------
            # All types
            # --------------
            # --- Correlation ---
            tmp = [workdict[k]['GrandEnsCorr'] for k in sitetag if k[0] in proxy_types]
            stat = [item for sublist in tmp for item in sublist] # flatten list of lists
            nbdata = len(stat)
            mean_stat = np.mean(stat)
            std_stat = np.std(stat)

            ptitle = '%s, %s' % ('All proxy types',vtype[v])
            n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=True)
            plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
            plt.title(ptitle)
            plt.xlabel("Correlation")
            plt.ylabel("Probability density")
            xmin,xmax,ymin,ymax = plt.axis()
            plt.axis((-1,1,ymin,ymax))

            # Annotate plot
            xmin,xmax,ymin,ymax = plt.axis()
            ypos = ymax-0.05*(ymax-ymin)
            xpos = xmin+0.025*(xmax-xmin)
            plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
            ypos = ypos-0.05*(ymax-ymin)
            plt.text(xpos,ypos,'Mean= %.2f' %mean_stat,fontsize=11,fontweight='bold')
            ypos = ypos-0.05*(ymax-ymin)
            plt.text(xpos,ypos,'Std-dev= %.2f' %std_stat,fontsize=11,fontweight='bold')
            
            plt.savefig('%s/summary_%s_stats_hist_corr_All.png' % (figdir,v),bbox_inches='tight')
            plt.close()

            # --- CE ---
            tmp = [workdict[k]['GrandEnsCE'] for k in sitetag if k[0] in proxy_types]
            stat = [item for sublist in tmp for item in sublist] # flatten list of lists
            nbdata = len(stat)
            mean_stat = np.mean(stat)
            std_stat = np.std(stat)
            
            ptitle = '%s, %s' % ('All proxy types', vtype[v])
            n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=True)
            plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
            plt.title(ptitle)
            plt.xlabel("Coefficient of efficiency")
            plt.ylabel("Probability density")
            xmin,xmax,ymin,ymax = plt.axis()
            plt.axis((-1,1,ymin,ymax))
            
            # Annotate plot
            xmin,xmax,ymin,ymax = plt.axis()
            ypos = ymax-0.05*(ymax-ymin)
            xpos = xmin+0.025*(xmax-xmin)
            plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
            ypos = ypos-0.05*(ymax-ymin)
            plt.text(xpos,ypos,'Mean= %.2f' %mean_stat,fontsize=11,fontweight='bold')
            ypos = ypos-0.05*(ymax-ymin)
            plt.text(xpos,ypos,'Std-dev= %.2f' %std_stat,fontsize=11,fontweight='bold')
            
            plt.savefig('%s/summary_%s_stats_hist_ce_All.png' % (figdir,v),bbox_inches='tight')
            plt.close()

            # --------------
            # Per proxy type
            # --------------
            # loop over proxy types
            for p in proxy_types:

                # --- Correlation ---
                tmp = [workdict[k]['GrandEnsCorr'] for k in sitetag if k[0] == p]
                stat = [item for sublist in tmp for item in sublist] # flatten list of lists
                nbdata = len(stat)
                mean_stat = np.mean(stat)
                std_stat = np.std(stat)

                ptitle = '%s, %s' % (p,vtype[v])
                n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=True)
                plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
                plt.title(ptitle)
                plt.xlabel("Correlation")
                plt.ylabel("Probability density")
                xmin,xmax,ymin,ymax = plt.axis()
                plt.axis((-1,1,ymin,ymax))

                # Annotate plot
                xmin,xmax,ymin,ymax = plt.axis()
                ypos = ymax-0.05*(ymax-ymin)
                xpos = xmin+0.025*(xmax-xmin)
                plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
                ypos = ypos-0.05*(ymax-ymin)
                plt.text(xpos,ypos,'Mean= %.2f' %mean_stat,fontsize=11,fontweight='bold')
                ypos = ypos-0.05*(ymax-ymin)
                plt.text(xpos,ypos,'Std-dev= %.2f' %std_stat,fontsize=11,fontweight='bold')

                pfname = p.replace (" ", "_")
                plt.savefig('%s/summary_%s_stats_hist_corr_%s.png' % (figdir,v,pfname),bbox_inches='tight')
                plt.close()

                # --- CE ---
                tmp = [workdict[k]['GrandEnsCE'] for k in sitetag if k[0] == p]
                stat = [item for sublist in tmp for item in sublist] # flatten list of lists
                nbdata = len(stat)
                mean_stat = np.mean(stat)
                std_stat = np.std(stat)

                ptitle = '%s, %s' % (p,vtype[v])
                n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=True)
                plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
                plt.title(ptitle)
                plt.xlabel("Coefficient of efficiency")
                plt.ylabel("Probability density")
                xmin,xmax,ymin,ymax = plt.axis()
                plt.axis((-1,1,ymin,ymax))

                # Annotate plot
                xmin,xmax,ymin,ymax = plt.axis()
                ypos = ymax-0.05*(ymax-ymin)
                xpos = xmin+0.025*(xmax-xmin)
                plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
                ypos = ypos-0.05*(ymax-ymin)
                plt.text(xpos,ypos,'Mean= %.2f' %mean_stat,fontsize=11,fontweight='bold')
                ypos = ypos-0.05*(ymax-ymin)
                plt.text(xpos,ypos,'Std-dev= %.2f' %std_stat,fontsize=11,fontweight='bold')
            
                pfname = p.replace (" ", "_")
                plt.savefig('%s/summary_%s_stats_hist_ce_%s.png' % (figdir,v,pfname),bbox_inches='tight')
                plt.close()


            # ===================================================================================
            # 2) Histogram of reconstruction DA ensemble calibration ratio across grand ensemble
            # ===================================================================================
            if v == 'assim':
                tmp = [workdict[k]['GrandEnsCalib'] for k in sitetag]
                stat = [item for sublist in tmp for item in sublist] # flatten list of lists
                mean_stat = np.mean(stat)
                std_stat = np.std(stat)

                ptitle = '%s, DA ensemble calibration' % (vtype[v])
                n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=True)
                plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
                plt.title(ptitle)
                plt.xlabel("Ensemble calibration ratio")
                plt.ylabel("Probability density")
                xmin,xmax,ymin,ymax = plt.axis()
                plt.axis((xmin,xmax,ymin,ymax))
                plt.axis((0,5,ymin,ymax))
                plt.plot([1,1],[ymin,ymax],'r--')

                # Annotate with summary stats
                xmin,xmax,ymin,ymax = plt.axis()
                ypos = ymax-0.05*(ymax-ymin)
                xpos = xmax-0.40*(xmax-xmin)
                plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
                ypos = ypos-0.05*(ymax-ymin)
                plt.text(xpos,ypos,'Mean= %.2f' %mean_stat,fontsize=11,fontweight='bold')
                ypos = ypos-0.05*(ymax-ymin)
                plt.text(xpos,ypos,'Std-dev= %.2f' %std_stat,fontsize=11,fontweight='bold')

                plt.savefig('%s/summary_%s_stats_hist_DAensCal_GrandEns.png' % (figdir,v),bbox_inches='tight')
                plt.close()


            # ===============================================================================
            # 3) Maps with proxy sites plotted on with dots colored according to sample size
            # ===============================================================================

            verif_metric = 'Sample size'

            mapcolor = plt.cm.hot_r
            
            itermax = roundup(Nbiter)
            fmin = 0.0; fmax = itermax
            fval = np.linspace(fmin, fmax, 100);  fvalc = np.linspace(0, 1, 101);           
            scaled_colors = mapcolor(fvalc)
            cmap, norm = from_levels_and_colors(levels=fval, colors=scaled_colors, extend='both')
            cbarticks=np.linspace(fmin,fmax,11)

            fig = plt.figure()
            ax  = fig.add_axes([0.1,0.1,0.8,0.8])
            m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=40.            # GLOBAL
            
            water = '#9DD4F0'
            continents = '#888888'
            m.drawmapboundary(fill_color=water)
            m.drawcoastlines(); m.drawcountries()
            m.fillcontinents(color=continents,lake_color=water)
            m.drawparallels(np.arange(-80.,81.,latres))
            m.drawmeridians(np.arange(-180.,181.,lonres))

            # loop over proxy sites
            l = []
            proxy_types = []
            for sitetag in workdict.keys():
                sitetype = sitetag[0]
                sitename = sitetag[1]
                sitemarker = proxy_verif[sitetype]

                lat = workdict[sitetag]['lat']
                lon = workdict[sitetag]['lon']
                x, y = m(lon,lat)
                if sitetype not in proxy_types:
                    proxy_types.append(sitetype)
                    l.append(m.scatter(x,y,35,c='white',marker=sitemarker,edgecolor='black',linewidth='1'))
                Gplt = m.scatter(x,y,35,c=workdict[sitetag]['NbPts'],marker=sitemarker,edgecolor='black',linewidth='1',zorder=4,cmap=cmap,norm=norm)

            cbar = m.colorbar(Gplt,location='right',pad="2%",size="2%",ticks=cbarticks,extend='neither')
            cbar.outline.set_linewidth(1.0)
            cbar.set_label('%s' % verif_metric,size=11,weight='bold')
            cbar.ax.tick_params(labelsize=10)
            plt.title(vtype[v],fontweight='bold')
            plt.legend(l,proxy_types,
                       scatterpoints=1,
                       loc='lower center', bbox_to_anchor=(0.5, -0.30),
                       ncol=3,
                       fontsize=9)

            plt.savefig('%s/map_recon_proxy_%s_stats_SampleSize.png' % (figdir,v),bbox_inches='tight')
            plt.close()


            # ===============================================================================
            # 4) Maps with proxy sites plotted on with dots colored according to correlation
            # ===============================================================================

            verif_metric = 'Correlation'

            #mapcolor = plt.cm.bwr
            mapcolor = plt.cm.seismic
            cbarfmt = '%4.1f'

            fmin = -1.0; fmax = 1.0
            fval = np.linspace(fmin, fmax, 100);  fvalc = np.linspace(0, fmax, 101);           
            scaled_colors = mapcolor(fvalc)
            cmap, norm = from_levels_and_colors(levels=fval, colors=scaled_colors, extend='both')
            cbarticks=np.linspace(fmin,fmax,11)

            fig = plt.figure()
            ax  = fig.add_axes([0.1,0.1,0.8,0.8])
            m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=40.            # GLOBAL
            
            water = '#9DD4F0'
            continents = '#888888'
            m.drawmapboundary(fill_color=water)
            m.drawcoastlines(); m.drawcountries()
            m.fillcontinents(color=continents,lake_color=water)
            m.drawparallels(np.arange(-80.,81.,latres))
            m.drawmeridians(np.arange(-180.,181.,lonres))

            # loop over proxy sites
            l = []
            proxy_types = []
            for sitetag in workdict.keys():
                sitetype = sitetag[0]
                sitename = sitetag[1]
                sitemarker = proxy_verif[sitetype]

                lat = workdict[sitetag]['lat']
                lon = workdict[sitetag]['lon']
                x, y = m(lon,lat)
                if sitetype not in proxy_types:
                    proxy_types.append(sitetype)
                    l.append(m.scatter(x,y,35,c='white',marker=sitemarker,edgecolor='black',linewidth='1'))
                Gplt = m.scatter(x,y,35,c=workdict[sitetag]['MeanCorr'],marker=sitemarker,edgecolor='black',linewidth='1',zorder=4,cmap=cmap,norm=norm)

            cbar = m.colorbar(Gplt,location='right',pad="2%",size="2%",ticks=cbarticks,format=cbarfmt,extend='both')
            cbar.outline.set_linewidth(1.0)
            cbar.set_label('%s' % verif_metric,size=11,weight='bold')
            cbar.ax.tick_params(labelsize=10)
            plt.title(vtype[v],fontweight='bold')
            plt.legend(l,proxy_types,
                       scatterpoints=1,
                       loc='lower center', bbox_to_anchor=(0.5, -0.30),
                       ncol=3,
                       fontsize=9)

            plt.savefig('%s/map_recon_proxy_%s_stats_corr.png' % (figdir,v),bbox_inches='tight')
            plt.savefig('map_recon_proxy_%s_stats_corr.pdf' % (v),bbox_inches='tight', dpi=300, format='pdf')
            plt.close()


            # ===============================================================================
            # 5) Maps with proxy sites plotted on with dots colored according to CE
            # ===============================================================================

            verif_metric = 'Coefficient of efficiency'

            #mapcolor = plt.cm.bwr
            mapcolor = plt.cm.seismic
            cbarfmt = '%4.1f'

            fmin = -1.0; fmax = 1.0
            fval = np.linspace(fmin, fmax, 100);  fvalc = np.linspace(0, fmax, 101);           
            scaled_colors = mapcolor(fvalc)
            cmap, norm = from_levels_and_colors(levels=fval, colors=scaled_colors,extend='both')
            cbarticks=np.linspace(fmin,fmax,11)

            fig = plt.figure()
            ax  = fig.add_axes([0.1,0.1,0.8,0.8])
            m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=40.
            
            water = '#9DD4F0'
            continents = '#888888'
            m.drawmapboundary(fill_color=water)
            m.drawcoastlines(); m.drawcountries()
            m.fillcontinents(color=continents,lake_color=water)
            m.drawparallels(np.arange(-80.,81.,latres))
            m.drawmeridians(np.arange(-180.,181.,lonres))

            # loop over proxy sites
            l = []
            proxy_types = []
            for sitetag in workdict.keys():
                sitetype = sitetag[0]
                sitename = sitetag[1]
                sitemarker = proxy_verif[sitetype]

                lat = workdict[sitetag]['lat']
                lon = workdict[sitetag]['lon']
                x, y = m(lon,lat)
                if sitetype not in proxy_types:
                    proxy_types.append(sitetype)
                    l.append(m.scatter(x,y,35,c='white',marker=sitemarker,edgecolor='black',linewidth='1'))
                Gplt = m.scatter(x,y,35,c=workdict[sitetag]['MeanCE'],marker=sitemarker,edgecolor='black',linewidth='1',zorder=4,cmap=cmap,norm=norm)

            cbar = m.colorbar(Gplt,location='right',pad="2%",size="2%",ticks=cbarticks,format=cbarfmt,extend='both')
            cbar.outline.set_linewidth(1.0)
            cbar.set_label('%s' % verif_metric,size=11,weight='bold')
            cbar.ax.tick_params(labelsize=10)
            plt.title(vtype[v],fontweight='bold')
            plt.legend(l,proxy_types,
                       scatterpoints=1,
                       loc='lower center', bbox_to_anchor=(0.5, -0.30),
                       ncol=3,
                       fontsize=9)
            
            plt.savefig('%s/map_recon_proxy_%s_stats_ce.png' % (figdir,v),bbox_inches='tight')
            plt.savefig('map_recon_proxy_%s_stats_ce.pdf' % (v),bbox_inches='tight', dpi=300, format='pdf')
            plt.close()


            if make_plots_individual_sites:
            
                # ===============================================================================
                # 6) Time series: comparison of proxy values and reconstruction-estimated proxies 
                #    for every site used in verification
                #
                # 7) Scatter plots of proxy vs reconstruction (ensemble-mean)
                #
                # 8) Histogram of correlation in grand ensemble for each site
                #
                # 9) Histogram of CE in grand ensemble for each site
                # ===============================================================================

                # loop over proxy sites
                for sitetag in workdict.keys():
                    sitetype = sitetag[0].replace (" ", "_")
                    sitename = sitetag[1]

                    x  = workdict[sitetag]['ts_years']
                    yp = workdict[sitetag]['ts_ProxyValues']
                    yr = workdict[sitetag]['ts_MeanRecon']
                    yrlow = workdict[sitetag]['ts_MeanRecon'] - workdict[sitetag]['ts_SpreadRecon']
                    yrupp = workdict[sitetag]['ts_MeanRecon'] + workdict[sitetag]['ts_SpreadRecon']

                    # -----------
                    # Time series
                    # -----------
                    p1 = plt.plot(x,yp,'-r',linewidth=2, label='Proxy',alpha=0.7)
                    p2 = plt.plot(x,yr,'-b',linewidth=2, label='Reconstruction')
                    plt.fill_between(x, yrlow, yrupp,alpha=0.4,linewidth=0.0)

                    plt.title("Proxy vs reconstruction: %s" % str(sitetag))
                    plt.xlabel("Years")
                    plt.ylabel("Proxy obs./estimate")
                    xmin,xmax,ymin,ymax = plt.axis()
                    plt.legend( loc='lower right', numpoints = 1, fontsize=11 )

                    # keep ymin and ymax
                    ts_pmin = ymin; ts_pmax = ymax

                    # Annotate with summary stats
                    xmin,xmax,ymin,ymax = plt.axis()
                    ypos = ymax-0.05*(ymax-ymin)
                    xpos = xmin+0.025*(xmax-xmin)
                    plt.text(xpos,ypos,'status: %s' %site_status,fontsize=11,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    plt.text(xpos,ypos,'ME = %s' %"{:.4f}".format(workdict[sitetag]['MeanME']),fontsize=11,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    plt.text(xpos,ypos,'RMSE = %s' %"{:.4f}".format(workdict[sitetag]['MeanRMSE']),fontsize=11,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    plt.text(xpos,ypos,'Corr = %s' %"{:.4f}".format(workdict[sitetag]['MeanCorr']),fontsize=11,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    plt.text(xpos,ypos,'CE = %s' %"{:.4f}".format(workdict[sitetag]['MeanCE']),fontsize=11,fontweight='bold')
                    
                    plt.savefig('%s/site_ts_recon_vs_proxy_%s_%s_%s.png' % (figdir, v, sitetype, sitename),bbox_inches='tight')
                    plt.close()

                    # ------------
                    # Scatter plot
                    # ------------
                    minproxy = np.min(yp); maxproxy = np.max(yp)
                    minrecon = np.min(yr); maxrecon = np.max(yr)
                    vmin = np.min([minproxy,minrecon])
                    vmax = np.max([maxproxy,maxrecon])

                    plt.plot(yr,yp,'o',markersize=8,markerfacecolor='#5CB8E6',markeredgecolor='black',markeredgewidth=1)
                    plt.title("Proxy vs reconstruction: %s" % str(sitetag))
                    plt.xlabel("Proxy estimates from reconstruction")
                    plt.ylabel("Proxy values")
                    plt.axis((vmin,vmax,vmin,vmax))
                    xmin,xmax,ymin,ymax = plt.axis()
                    # one-one line
                    plt.plot([vmin,vmax],[vmin,vmax],'r--')

                    # Annotate with summary stats
                    xmin,xmax,ymin,ymax = plt.axis()
                    ypos = ymax-0.05*(ymax-ymin)
                    xpos = xmin+0.025*(xmax-xmin)
                    plt.text(xpos,ypos,'status: %s' %site_status,fontsize=11,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    plt.text(xpos,ypos,'ME = %s' %"{:.4f}".format(workdict[sitetag]['MeanME']),fontsize=11,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    plt.text(xpos,ypos,'RMSE = %s' %"{:.4f}".format(workdict[sitetag]['MeanRMSE']),fontsize=11,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    plt.text(xpos,ypos,'Corr = %s' %"{:.4f}".format(workdict[sitetag]['MeanCorr']),fontsize=11,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    plt.text(xpos,ypos,'CE = %s' %"{:.4f}".format(workdict[sitetag]['MeanCE']),fontsize=11,fontweight='bold')
                    
                    plt.savefig('%s/site_scatter_recon_vs_proxy_%s_%s_%s.png' % (figdir, v, sitetype, sitename),bbox_inches='tight')
                    plt.close()

                    sampleSize = workdict[sitetag]['NbPts']
                    if sampleSize > 1:

                        # ---------------------------
                        # Mean Error (bias) histogram
                        # ---------------------------
                        stat = workdict[sitetag]['GrandEnsME']
                        nbMCiter = len(stat)
                        mean_stat = np.mean(stat)
                        stddev_stat = np.std(stat)

                        n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=True)
                        plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
                        plt.title("Bias: %s" % str(sitetag))
                        plt.xlabel("Mean error")
                        plt.ylabel("Probability density")
                        xmin,xmax,ymin,ymax = plt.axis()
                        #xrg = np.maximum(np.abs(ts_pmin),np.abs(ts_pmax)) # some measure of proxy range
                        xrg = np.std(yp) # some measure of proxy range
                        plt.axis((-xrg,xrg,ymin,ymax))
                        plt.plot([0,0],[ymin,ymax],'r--')
                
                        # Annotate with summary stats
                        xmin,xmax,ymin,ymax = plt.axis()
                        ypos = ymax-0.05*(ymax-ymin)
                        xpos = xmin+0.025*(xmax-xmin)
                        plt.text(xpos,ypos,'status: %s' %site_status,fontsize=11,fontweight='bold')
                        ypos = ypos-0.05*(ymax-ymin)
                        plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
                        ypos = ypos-0.05*(ymax-ymin)
                        plt.text(xpos,ypos,'Mean: %.2f' %mean_stat,fontsize=11,fontweight='bold')
                        ypos = ypos-0.05*(ymax-ymin)
                        plt.text(xpos,ypos,'Std-dev: %.2f' %stddev_stat,fontsize=11,fontweight='bold')

                        plt.savefig('%s/site_hist_bias_%s_%s_%s.png' % (figdir,v,sitetype, sitename),bbox_inches='tight')
                        plt.close()


                        # ----------------------
                        # Correlation histogram
                        # ---------------------
                        stat = workdict[sitetag]['GrandEnsCorr']
                        nbMCiter = len(stat)
                        mean_stat = np.mean(stat)
                        stddev_stat = np.std(stat)

                        n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=True)
                        plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
                        plt.title("Correlation: %s" % str(sitetag))
                        plt.xlabel("Correlation")
                        plt.ylabel("Probability density")
                        xmin,xmax,ymin,ymax = plt.axis()
                        plt.axis((xmin,xmax,ymin,ymax))
                        plt.axis((-1,1,ymin,ymax))
                        plt.plot([0,0],[ymin,ymax],'r--')

                        # Annotate with summary stats
                        xmin,xmax,ymin,ymax = plt.axis()
                        ypos = ymax-0.05*(ymax-ymin)
                        xpos = xmin+0.025*(xmax-xmin)
                        plt.text(xpos,ypos,'status: %s' %site_status,fontsize=11,fontweight='bold')
                        ypos = ypos-0.05*(ymax-ymin)
                        plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
                        ypos = ypos-0.05*(ymax-ymin)
                        plt.text(xpos,ypos,'Mean: %.2f' %mean_stat,fontsize=11,fontweight='bold')
                        ypos = ypos-0.05*(ymax-ymin)
                        plt.text(xpos,ypos,'Std-dev: %.2f' %stddev_stat,fontsize=11,fontweight='bold')

                        plt.savefig('%s/site_hist_corr_%s_%s_%s.png' % (figdir,v,sitetype, sitename),bbox_inches='tight')
                        plt.close()

                        # ---------------------
                        # CE histogram
                        # ---------------------
                        stat = workdict[sitetag]['GrandEnsCE']
                        nbMCiter = len(stat)
                        mean_stat = np.mean(stat)
                        stddev_stat = np.std(stat)

                        n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=True)
                        plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
                        plt.title("Coefficient of efficiency: %s" % str(sitetag))
                        plt.xlabel("Coefficient of efficiency")
                        plt.ylabel("Probability density")
                        xmin,xmax,ymin,ymax = plt.axis()
                        plt.axis((xmin,xmax,ymin,ymax))
                        plt.axis((-1,1,ymin,ymax))
                        plt.plot([0,0],[ymin,ymax],'r--')

                        # Annotate with summary stats
                        xmin,xmax,ymin,ymax = plt.axis()
                        ypos = ymax-0.05*(ymax-ymin)
                        xpos = xmin+0.025*(xmax-xmin)
                        plt.text(xpos,ypos,'status: %s' %site_status,fontsize=11,fontweight='bold')
                        ypos = ypos-0.05*(ymax-ymin)
                        plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
                        ypos = ypos-0.05*(ymax-ymin)
                        plt.text(xpos,ypos,'Mean: %.2f' %mean_stat,fontsize=11,fontweight='bold')
                        ypos = ypos-0.05*(ymax-ymin)
                        plt.text(xpos,ypos,'Std-dev: %.2f' %stddev_stat,fontsize=11,fontweight='bold')

                        plt.savefig('%s/site_hist_ce_%s_%s_%s.png' % (figdir,v,sitetype,sitename),bbox_inches='tight')
                        plt.close()

                
                        # ------------------------------------
                        # Ensemble calibration ratio histogram
                        # ------------------------------------
                        if v == 'assim':                
                            stat = workdict[sitetag]['GrandEnsCalib']
                            mean_stat = np.mean(stat)
                            stddev_stat = np.std(stat)

                            n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=True)
                            plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
                            plt.title("DA ensemble calibration: %s" % str(sitetag))
                            plt.xlabel("Ensemble calibration ratio")
                            plt.ylabel("Probability density")
                            xmin,xmax,ymin,ymax = plt.axis()
                            plt.axis((xmin,xmax,ymin,ymax))
                            plt.axis((0,5,ymin,ymax))
                            plt.plot([0,0],[ymin,ymax],'r--')

                            # Annotate with summary stats
                            xmin,xmax,ymin,ymax = plt.axis()
                            ypos = ymax-0.05*(ymax-ymin)
                            xpos = xmax-0.40*(xmax-xmin)
                            plt.text(xpos,ypos,'status: %s' %site_status,fontsize=11,fontweight='bold')
                            ypos = ypos-0.05*(ymax-ymin)
                            plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
                            ypos = ypos-0.05*(ymax-ymin)
                            plt.text(xpos,ypos,'Mean: %.2f' %mean_stat,fontsize=11,fontweight='bold')
                            ypos = ypos-0.05*(ymax-ymin)
                            plt.text(xpos,ypos,'Std-dev: %.2f' %stddev_stat,fontsize=11,fontweight='bold')

                            plt.savefig('%s/site_hist_enscal_%s_%s_%s.png' % (figdir,v,sitetype,sitename),bbox_inches='tight')
                            plt.close()

                    else:
                        print 'Sample size too small to plot histograms for proxy site:', sitetag

    end_time = time() - begin_time
    print '======================================================='
    print 'All completed in '+ str(end_time/3600.0)+' hours'
    print '======================================================='



# =============================================================================

if __name__ == '__main__':
    main()
