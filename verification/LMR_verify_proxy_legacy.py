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
import os, sys
import numpy as np
import pickle
import timeit
import pandas as pd
from time import time
from os.path import join
from copy import deepcopy

# LMR specific imports
sys.path.append('../')
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

    # -- production recons for paper 1
    #nexp = 'production_gis_ccsm4_pagesall_0.75/'
    #nexp = 'production_mlost_ccsm4_pagesall_0.75/'
    #nexp = 'production_cru_ccsm4_pagesall_0.75/'
    #
    nexp = 'test'

    
    # lmr_path: where all the data is located ... model (prior), analyses (GISTEMP, HAdCRUT...) and proxies.
    #lmr_path = '/home/disk/ice4/nobackup/hakim/lmr'
    #lmr_path = '/home/chaos2/wperkins/data/LMR'
    lmr_path = '/home/disk/kalman3/rtardif/LMR'

    # inclusive
    #verif_period = [0, 1879]
    verif_period = [1880, 2000]

    #iter_range = [0, 0]
    iter_range = [0, 100]

    # Input directory, where to find the reconstruction data
    #datadir_input  = '/home/disk/kalman3/hakim/LMR/'
    #datadir_input  = '/home/disk/kalman2/wperkins/LMR_output/archive' # production recons
    #datadir_input  = '/home/disk/kalman3/rtardif/LMR/output'
    datadir_input  = '/home/disk/ekman4/rtardif/LMR/output'
    
    # Output directory, where the verification results & figs will be dumped.
    datadir_output = datadir_input # if want to keep things tidy
    #datadir_output = '/home/disk/ekman4/rtardif/LMR/output/verification_production_runs'

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

    use_from = ['pages']
    #use_from = ['NCDC']
    proxy_frac = 1.0

    # type of proxy timeseries to return: 'anom' for anomalies
    # (temporal mean removed) or asis' to keep unchanged
    proxy_timeseries_kind = 'asis'
    #
    proxy_availability_filter = False
    proxy_availability_fraction = 1.0
    
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

        # DO NOT CHANGE FORMAT BELOW

        proxy_order = [
            'Tree ring_Width',
            'Tree ring_Density',
            'Ice core_d18O',
            'Ice core_d2H',
            'Ice core_Accumulation',
            'Coral_d18O',
            'Coral_Luminescence',
            'Lake sediment_All',
            'Marine sediment_All',
            'Speleothem_All'
        ]

        # Assignment of psm type per proxy type
        # Choices are: 'linear', 'linear_TorP', 'bilinear', 'h_interp'
        #  The linear PSM can be used on *all* proxies.
        #  The linear_TorP and bilinear w.r.t. temperature or/and moisture
        #  PSMs are aimed at *tree ring* proxies in particular
        #  The h_interp forward model is to be used for isotope proxies when
        #  the prior is taken from an isotope-enabled GCM model output. 
        proxy_psm_type = {
            'Tree ring_Width'      : 'linear',
            'Tree ring_Density'    : 'linear',
            'Ice core_d18O'        : 'linear',
            'Ice core_d2H'         : 'linear',
            'Ice core_Accumulation': 'linear',
            'Coral_d18O'           : 'linear',
            'Coral_Luminescence'   : 'linear',
            'Lake sediment_All'    : 'linear',
            'Marine sediment_All'  : 'linear',
            'Speleothem_All'       : 'linear',
            }
        
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
            self.proxy_timeseries_kind = v_proxies.proxy_timeseries_kind
            self.proxy_order = list(self.proxy_order)
            self.proxy_psm_type = deepcopy(self.proxy_psm_type)
            self.proxy_assim2 = deepcopy(self.proxy_assim2)
            self.proxy_blacklist = list(self.proxy_blacklist)
            self.proxy_availability_filter = v_proxies.proxy_availability_filter
            self.proxy_availability_fraction = v_proxies.proxy_availability_fraction
            
            # Create mapping for Proxy Type/Measurement Type to type names above
            self.proxy_type_mapping = {}
            for type, measurements in self.proxy_assim2.items():
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

        #dbversion = 'v0.0.0'
        #dbversion = 'v0.1.0'
        dbversion = 'v0.2.0'
        
        datadir_proxy = None
        datafile_proxy = 'NCDC_%s_Proxies.df.pckl' %(dbversion)
        metafile_proxy = 'NCDC_%s_Metadata.df.pckl' %(dbversion)
        dataformat_proxy = 'DF'

        # This is not activated with NCDC data yet...
        regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
                   'North America', 'South America']

        proxy_resolution = [1.0]
        
        # Limit proxies to those included in the following databases
        database_filter = []
        #database_filter = ['PAGES2kv2']
        #database_filter = ['LMR','PAGES2kv2']

        # A blacklist on proxy records, to prevent processing of chronologies known to be duplicates.
        proxy_blacklist = []

        
        # DO NOT CHANGE FORMAT BELOW
        proxy_order = [
        #    'Tree Rings_WidthPages',
            'Tree Rings_WidthPages2',
            'Tree Rings_WidthBreit',
            'Tree Rings_WoodDensity',
            'Tree Rings_Isotopes',
            'Corals and Sclerosponges_d18O',
            'Corals and Sclerosponges_SrCa',
            'Corals and Sclerosponges_Rates',
            'Ice Cores_d18O',
            'Ice Cores_dD',
            'Ice Cores_Accumulation',
            'Ice Cores_MeltFeature',
            'Lake Cores_Varve',
            'Lake Cores_BioMarkers',
            'Lake Cores_GeoChem',
            'Lake Cores_Misc',
            'Marine Cores_d18O',
#            'Speleothems_d18O',
            'Bivalve_d18O',
            ]

        # Assignment of psm type per proxy type
        # Choices are: 'linear', 'linear_TorP', 'bilinear', 'h_interp'
        #  The linear PSM can be used on *all* proxies.
        #  The linear_TorP and bilinear w.r.t. temperature or/and moisture
        #  PSMs are aimed at *tree ring* proxies in particular
        #  The h_interp forward model is to be used for isotope proxies when
        #  the prior is taken from an isotope-enabled GCM output. 
        proxy_psm_type = {
            'Corals and Sclerosponges_d18O' : 'linear',
            'Corals and Sclerosponges_SrCa' : 'linear',
            'Corals and Sclerosponges_Rates': 'linear',
            'Ice Cores_d18O'                : 'linear',
            'Ice Cores_dD'                  : 'linear',
            'Ice Cores_Accumulation'        : 'linear',
            'Ice Cores_MeltFeature'         : 'linear',
            'Lake Cores_Varve'              : 'linear',
            'Lake Cores_BioMarkers'         : 'linear',
            'Lake Cores_GeoChem'            : 'linear',
            'Lake Cores_Misc'               : 'linear',
            'Marine Cores_d18O'             : 'linear',
            'Tree Rings_WidthBreit'         : 'linear',
            'Tree Rings_WidthPages2'        : 'linear',
#            'Tree Rings_WidthPages'         : 'linear',
            'Tree Rings_WoodDensity'        : 'linear',
            'Tree Rings_Isotopes'           : 'linear',
            'Speleothems_d18O'              : 'linear',
            'Bivalve_d18O'                  : 'linear',            
        }
        
        proxy_assim2 = {
            'Corals and Sclerosponges_d18O' : ['d18O','delta18O','d18o','d18O_stk','d18O_int','d18O_norm',
                                               'd18o_avg','d18o_ave','dO18','d18O_4'],
            'Corals and Sclerosponges_SrCa' : ['Sr/Ca','Sr_Ca','Sr/Ca_norm','Sr/Ca_anom','Sr/Ca_int'],
            'Corals and Sclerosponges_Rates': ['ext','calc','calcification','calcification rate', 'composite'],
            'Ice Cores_d18O'                : ['d18O','delta18O','delta18o','d18o','dO18',
                                               'd18o_int','d18O_int',
                                               'd18O_norm','d18o_norm',
                                               'd18O_anom'],
            'Ice Cores_dD'                  : ['deltaD','delD','dD'],
            'Ice Cores_Accumulation'        : ['accum','accumu'],
            'Ice Cores_MeltFeature'         : ['MFP','melt'],
            'Lake Cores_Varve'              : ['varve', 'varve_thickness', 'varve thickness','thickness'],
            'Lake Cores_BioMarkers'         : ['Uk37', 'TEX86', 'tex86'],
            'Lake Cores_GeoChem'            : ['Sr/Ca', 'Mg/Ca','Cl_cont'],
            'Lake Cores_Misc'               : ['RABD660_670','X_radiograph_dark_layer','massacum'],
            'Marine Cores_d18O'             : ['d18O'],
            'Speleothems_d18O'              : ['d18O'],
            'Bivalve_d18O'                  : ['d18O'],
            'Tree Rings_WidthBreit'         : ['trsgi_breit'],
            'Tree Rings_WidthPages2'        : ['trsgi'], 
            'Tree Rings_WidthPages'         : ['TRW',
                                               'ERW',
                                               'LRW'],
            'Tree Rings_WoodDensity'        : ['max_d',
                                               'min_d',
                                               'early_d',
                                               'earl_d',
                                               'late_d',
                                               'density',
                                               'MXD'],
            'Tree Rings_Isotopes'           : ['d18O'],
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
            self.proxy_timeseries_kind = v_proxies.proxy_timeseries_kind
            self.proxy_order = list(self.proxy_order)
            self.proxy_psm_type = deepcopy(self.proxy_psm_type)
            self.proxy_assim2 = deepcopy(self.proxy_assim2)
            self.database_filter = list(self.database_filter)
            self.proxy_blacklist = list(self.proxy_blacklist)
            self.proxy_availability_filter = v_proxies.proxy_availability_filter
            self.proxy_availability_fraction = v_proxies.proxy_availability_fraction
            
            self.proxy_type_mapping = {}
            for ptype, measurements in self.proxy_assim2.items():
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
    avgPeriod: str
        Indicates use of PSMs calibrated on annual or seasonal data: allowed tags are 'annual' or 'season'
    """

    # Keep this as annual, as it is assumed that LMR output is annual
    avgPeriod = 'annual'
    
    # Mapping of calibration sources w/ climate variable
    # To be modified only if a new calibration source is added. 
    all_calib_sources = {'temperature': ['GISTEMP', 'MLOST', 'HadCRUT', 'BerkeleyEarth'], 'moisture': ['GPCC','DaiPDSI']}

    
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
        #datafile_calib = 'GPCC_precip.mon.flux.1x1.v6.nc'
        # or
        #datatag_calib_P = 'DaiPDSI'
        #datafile_calib_P = 'Dai_pdsi.mon.mean.selfcalibrated_185001-201412.nc'

        
        pre_calib_datafile = None

        psm_r_crit = 0.0


        def __init__(self):
            self.datatag_calib = self.datatag_calib
            self.datafile_calib = self.datafile_calib
            self.psm_r_crit = self.psm_r_crit
            self.avgPeriod = v_psm.avgPeriod
            
            if self.datadir_calib is None:
                self.datadir_calib = join(v_core.lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = self.datadir_calib

            if self.pre_calib_datafile is None:
                if '-'.join(v_proxies.use_from) == 'NCDC':
                    dbversion = v_proxies._ncdc.dbversion
                    filename = ('PSMs_'+'-'.join(v_proxies.use_from) +
                                '_' + dbversion +
                                '_' + self.avgPeriod +
                                '_' + self.datatag_calib+'.pckl')
                else:
                    filename = ('PSMs_' + '-'.join(v_proxies.use_from) +
                                '_' + self.datatag_calib+'.pckl')
                self.pre_calib_datafile = join(v_core.lmr_path,
                                               'PSM',
                                               filename)
            else:
                self.pre_calib_datafile = self.pre_calib_datafile


            # association of calibration source and state variable needed to calculate Ye's
            if self.datatag_calib in v_psm.all_calib_sources['temperature']:
                self.psm_required_variables = {'tas_sfc_Amon': 'anom'}

            elif self.datatag_calib in v_psm.all_calib_sources['moisture']:
                if self.datatag_calib == 'GPCC':
                    self.psm_required_variables = {'pr_sfc_Amon':'anom'}
                elif self.datatag_calib == 'DaiPDSI':
                    self.psm_required_variables = {'scpdsi_sfc_Amon': 'anom'}
                else:
                    raise KeyError('Unrecognized moisture calibration source.'
                                   ' State variable not identified for Ye calculation.')
            else:
                raise KeyError('Unrecognized calibration source.'
                               ' State variable not identified for Ye calculation.')
    
    
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
            self.avgPeriod = v_psm.avgPeriod
            
            if self.datadir_calib is None:
                self.datadir_calib = join(v_core.lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = self.datadir_calib

            if self.pre_calib_datafile_T is None:
                if '-'.join(v_proxies.use_from) == 'NCDC':
                    dbversion = v_proxies._ncdc.dbversion
                    filename_t = ('PSMs_' + '-'.join(v_proxies.use_from) +
                                  '_' + dbversion +
                                  '_' + self.avgPeriod +
                                  '_' + self.datatag_calib_T + '.pckl')
                else:
                    filename_t = ('PSMs_' + '-'.join(v_proxies.use_from) +
                                  '_' + self.datatag_calib_T + '.pckl')
                self.pre_calib_datafile_T = join(v_core.lmr_path,
                                                 'PSM',
                                                 filename_t)
            else:
                self.pre_calib_datafile_T = self.pre_calib_datafile_T

            if self.pre_calib_datafile_P is None:
                if '-'.join(v_proxies.use_from) == 'NCDC':
                    dbversion = v_proxies._ncdc.dbversion
                    filename_p = ('PSMs_' + '-'.join(v_proxies.use_from) +
                                  '_' + dbversion +
                                  '_' + self.avgPeriod +
                                  '_' + self.datatag_calib_P + '.pckl')
                else:
                    filename_p = ('PSMs_' + '-'.join(v_proxies.use_from) +
                                  '_' + self.datatag_calib_P + '.pckl')
                self.pre_calib_datafile_P = join(v_core.lmr_path,
                                                 'PSM',
                                                 filename_p)
            else:
                self.pre_calib_datafile_P = self.pre_calib_datafile_P


            # association of calibration sources and state variables needed to calculate Ye's
            required_variables = {'tas_sfc_Amon': 'anom'} # start with temperature

            # now check for moisture & add variable to list
            if self.datatag_calib_P == 'GPCC':
                    required_variables['pr_sfc_Amon'] = 'anom'
            elif self.datatag_calib_P == 'DaiPDSI':
                    required_variables['scpdsi_sfc_Amon'] = 'anom'
            else:
                raise KeyError('Unrecognized moisture calibration source.'
                               ' State variable not identified for Ye calculation.')

            self.psm_required_variables = required_variables
            
                
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
            self.avgPeriod = v_psm.avgPeriod
            
            if self.datadir_calib is None:
                self.datadir_calib = join(v_core.lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = self.datadir_calib


            if self.pre_calib_datafile is None:
                if '-'.join(v_proxies.use_from) == 'NCDC':
                    dbversion = v_proxies._ncdc.dbversion
                    filename = ('PSMs_'+'-'.join(v_proxies.use_from) +
                                '_' + dbversion +
                                '_' + self.avgPeriod +
                                '_' + self.datatag_calib_T +
                                '_' + self.datatag_calib_P +'.pckl')
                else:
                    filename = ('PSMs_'+'-'.join(v_proxies.use_from) +
                                '_' + self.datatag_calib_T +
                                '_' + self.datatag_calib_P +'.pckl')
                self.pre_calib_datafile = join(v_core.lmr_path,
                                               'PSM',
                                               filename)
            else:
                self.pre_calib_datafile = self.pre_calib_datafile

            # association of calibration sources and state variables needed to calculate Ye's
            required_variables = {'tas_sfc_Amon': 'anom'} # start with temperature

            # now check for moisture & add variable to list
            if self.datatag_calib_P == 'GPCC':
                    required_variables['pr_sfc_Amon'] = 'anom'
            elif self.datatag_calib_P == 'DaiPDSI':
                    required_variables['scpdsi_sfc_Amon'] = 'anom'
            else:
                raise KeyError('Unrecognized moisture calibration source.'
                               ' State variable not identified for Ye calculation.')

            self.psm_required_variables = required_variables


    class _h_interp(object):
        """
        Parameters for the horizontal interpolator PSM.

        Attributes
        ----------
        radius_influence : real
            Distance-scale used the calculation of exponentially-decaying
            weights in interpolator (in km)
        datadir_obsError: str
            Absolute path to obs. error variance data
        filename_obsError: str
            Filename for obs. error variance data
        dataformat_obsError: str
            String indicating the format of the file containing obs. error
            variance data
            Note: note currently used by code. For info purpose only.
        datafile_obsError: str
            Absolute path/filename of obs. error variance data
        """

        ##** BEGIN User Parameters **##

        # Interpolation parameter:
        # Set to 'None' if want Ye = value at nearest grid point to proxy
        # location
        # Set to a non-zero float if want Ye = weighted-average of surrounding
        # gridpoints
        # radius_influence = None
        radius_influence = 50. # distance-scale in km
        
        ##** END User Parameters **##

        def __init__(self):
            self.radius_influence = self.radius_influence
            # File with R values not required in context of this program.
            self.datafile_obsError = None
                
            # define state variable needed to calculate Ye's
            # only d18O for now ...
            # psm requirements depend on settings in proxies class 
            proxy_kind = v_proxies.proxy_timeseries_kind
            if v_proxies.proxy_timeseries_kind == 'asis':
                psm_var_kind = 'full'
            elif v_proxies.proxy_timeseries_kind == 'anom':
                psm_var_kind = 'anom'
            else:
                raise ValueError('Unrecognized proxy_timeseries_kind value in proxies class.'
                                 ' Unable to assign kind to psm_required_variables'
                                 ' in h_interp psm class.')                
            self.psm_required_variables = {'d18O_sfc_Amon': psm_var_kind}


    # Initialize subclasses with all attributes
    def __init__(self, **kwargs):
        self.linear = self._linear(**kwargs)
        self.linear_TorP = self._linear_TorP(**kwargs)
        self.bilinear = self._bilinear(**kwargs)
        self.h_interp = self._h_interp(**kwargs)

# =============================================================================
# END:  set user parameters here
# =============================================================================
class config(object):
    def __init__(self,core,proxies,psm):
        self.core = core()
        self.proxies = proxies()
        self.psm = psm()

Cfg = config(v_core,v_proxies,v_psm)


# =============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Main code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# =============================================================================
def main():


    #
    # TODO: re-define "verif" proxies as those in the database (all proxy objects given by "load_all") that are NOT in the assimilated set
    # 
    
    verbose = 1

    # Keep = True for now ...
    complement_set = True
    
    begin_time = time()

    nexp =  Cfg.core.nexp
    proxy_database = Cfg.proxies.use_from[0]
    verif_period = Cfg.core.verif_period

    print('Experiment          :', nexp)
    print('Proxies             :', proxy_database)
    print('Verif. period       :', verif_period)

    dimtime = (verif_period[1] - verif_period[0]) + 1
    
    if proxy_database == 'pages':
        print('Proxy data location :', Cfg.proxies.pages.datadir_proxy)
    elif proxy_database == 'NCDC':
        print('Proxy data location :', Cfg.proxies.ncdc.datadir_proxy)
    else:
        print('ERROR in specification of proxy database. Exiting!')
        exit(1)

    # get proxy class
    proxy_class = LMR_proxy_pandas_rework.get_proxy_class(proxy_database)
    # load proxy data
    beginload = timeit.default_timer()
    print('Loading proxy & associated psm objects ...') 
    proxy_ids_by_grp, proxy_objects = proxy_class.load_all(Cfg,
                                                           verif_period,
                                                           None)

    master_proxy_list = []
    for proxy_idx, Y in enumerate(proxy_objects): master_proxy_list.append((Y.type,Y.id))
    
    print('--------------------------------------------------------------------')
    print('Uploaded proxies : counts per proxy type:')
    # count the total number of proxies
    total_proxy_count = len(proxy_objects)
    for pkey, plist in proxy_ids_by_grp.items():
        print('%45s : %5d' % (pkey, len(plist)))
    print('%45s : %5d' % ('TOTAL', total_proxy_count))
    print('--------------------------------------------------------------------')
    endload = timeit.default_timer()
    print('Loading completed in:', (endload-beginload)/60.0 , ' mins')
    print(' ')

    #for pobj_idx, pobj in enumerate(proxy_objects): print pobj.id, pobj.psm_obj.__dict__
    
    
    # Given the set of proxies & PSMs selected, define set of reconstruction
    # state variables that will be have to be uploaded to calculate Ye values
    if proxy_database == 'NCDC':
        proxy_cfg = Cfg.proxies.ncdc
    elif proxy_database == 'pages':
        proxy_cfg = Cfg.proxies.pages
    else:
        print('ERROR in specification of proxy database.')
        raise SystemExit()

    # proxy types activated in configuration
    proxy_types = proxy_cfg.proxy_order 
    # associated psm's
    psm_keys = list(set([proxy_cfg.proxy_psm_type[p] for p in proxy_types]))

    # Forming list of required state variables
    psmclasses = dict([(name,cls)  for name, cls in list(Cfg.psm.__dict__.items())])
    psm_required_variables = []
    calib_sources = []
    for psm_type in psm_keys:
        #print psm_type, ':', psmclasses[psm_type].psm_required_variables
        psm_required_variables.extend(psmclasses[psm_type].psm_required_variables)

        if psm_type == 'linear':
            calib_sources.append(Cfg.psm.linear.datatag_calib)
        elif psm_type == 'linear_TorP':
            calib_sources.append(Cfg.psm.linear_TorP.datatag_calib_T)
            calib_sources.append(Cfg.psm.linear_TorP.datatag_calib_P)
        elif psm_type == 'bilinear':
            calib_sources.append(Cfg.psm.bilinear.datatag_calib_T)
            calib_sources.append(Cfg.psm.bilinear.datatag_calib_P)
        else:
            pass
            
    # keep unique values
    psm_required_variables = list(set(psm_required_variables))
    calib_sources= list(set(calib_sources))

    psm_str = '-'.join(psm_keys + calib_sources)

    
    # ==========================================================================
    # Loop over the Monte-Carlo reconstructions
    # ==========================================================================

    datadir_input = Cfg.core.datadir_input
    datadir_output = Cfg.core.datadir_output

    verif_listdict = []
    assim_listdict = []

    Nbiter = Cfg.core.iter_range[1]
    MCiters = np.arange(Cfg.core.iter_range[0], Cfg.core.iter_range[1]+1)
    for iter in MCiters:

        # Experiment data directory
        workdir = datadir_input+'/'+nexp+'/r'+str(iter)

        print(' Working on: ' + workdir)

        # Check availability of files containing required variables
        for var in psm_required_variables:
            datafile = workdir+'/ensemble_mean_'+var+'.npz'
            if not os.path.isfile(datafile):
                raise SystemExit
        

        # ======================================================
        # Information on assimilated and non-assimilated proxies
        # ======================================================

        # List of assimilated proxies
        # ---------------------------
        loaded_proxies = np.load(workdir+'/'+'assimilated_proxies.npy')
        nb_loaded_proxies = len(loaded_proxies)

        # Filter to match proxy sites in set of calibrated PSMs
        indp = [k for k in range(nb_loaded_proxies) if (list(loaded_proxies[k].keys())[0],loaded_proxies[k][list(loaded_proxies[k].keys())[0]][0]) in master_proxy_list]
        assim_proxies = loaded_proxies[indp]
        nb_assim_proxies = len(assim_proxies)

        # in list form
        assim_proxies_list = [(list(assim_proxies[k].keys())[0],assim_proxies[k][list(assim_proxies[k].keys())[0]][0]) for k in range(nb_assim_proxies)]

        print('------------------------------------------------')
        print('Number of proxy sites in assimilated set :',  nb_assim_proxies)


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
            indp = [k for k in range(nb_loaded_proxies) if (list(loaded_proxies[k].keys())[0],loaded_proxies[k][list(loaded_proxies[k].keys())[0]][0]) in master_proxy_list]
            verif_proxies = loaded_proxies[indp]
            nb_verif_proxies = len(verif_proxies)
            # in list form
            verif_proxies_list = [(list(verif_proxies[k].keys())[0],verif_proxies[k][list(verif_proxies[k].keys())[0]][0]) for k in range(nb_verif_proxies)]
        
        print('Number of proxy sites in verification set:',  nb_verif_proxies)
        print('------------------------------------------------')

        
        # ============================
        # Load the reconstruction data
        # ============================

        # -------------------------------------------------------
        # Load the full ensemble Ye's updated during assimilation
        # => appended state vector method
        # -------------------------------------------------------
        fnameYe = workdir+'/'+'analysis_Ye.pckl'
        try:
            assimYe.clear()
        except:
            pass
        if os.path.isfile(fnameYe):
            infile  = open(fnameYe,'r')
            assimYe = pickle.load(infile)
            infile.close()

        
        # ------------------------------------------------------
        # Load in the prior ensemble data
        # ------------------------------------------------------
        file_prior = workdir+'/Xb_one.npz'
        Xprior_statevector = np.load(file_prior)
        Xb_state_info = Xprior_statevector['state_info'].item()
        available_state_variables = list(Xb_state_info.keys())
        Xb_one = Xprior_statevector['Xb_one']
        Xb_coords = Xprior_statevector['Xb_one_coords']

        # prepare array to contain "state vector"
        print(Xb_state_info)
        totlength = 0
        for var in psm_required_variables:
            position = Xb_state_info[var]['pos']
            totlength = totlength + ((position[1] - position[0]) + 1)
        Xrecon_statevector = np.zeros(shape=[totlength,dimtime])
        Xrecon_coords = np.empty(shape=[totlength,2])
        
        # -------------------------------------------------------
        # Load in the ensemble-mean reconstruction data
        # -------------------------------------------------------
        # extract required data from data files
        nb_state_var = len(psm_required_variables)
        Xrecon_state_info = {}

        Nx = 0
        for state_var in psm_required_variables:

            # Check for presence of file containing reconstruction of state variable
            filein = workdir+'/ensemble_mean_'+state_var+'.npz'

            Xrecon = np.load(filein)
            recon_data = Xrecon['xam']

            if Nx == 0: # 1st time in loop over variables                
                recon_times = np.asarray(Xrecon['years'],dtype='int32')
                recon_lat = Xrecon['lat']
                recon_lon = Xrecon['lon']
                lat =  recon_lat[:,0]
                lon =  recon_lon[0,:]
                indverif, = np.where((recon_times>=verif_period[0]) & (recon_times<=verif_period[1]))

            Xrecon_state_info[state_var] = {}
            Xrecon_state_info[state_var]['spacedims'] = (lat.shape[0], lon.shape[0])
            Xrecon_state_info[state_var]['spacecoords'] = ('lat','lon') # 2D lat/lon field assumed!
            ndimtot = lat.shape[0]*lon.shape[0]

            indstart = Nx
            indend   = Nx+(ndimtot)
            Xrecon_state_info[state_var]['pos'] = (indstart,indend-1)
            for i in range(dimtime):
                Xrecon_statevector[indstart:indend,i] = recon_data[indverif[i],:,:].flatten()

            Xrecon_coords[indstart:indend,0] = recon_lat.flatten()
            Xrecon_coords[indstart:indend,1] = recon_lon.flatten()

            Nx = Nx + (ndimtot)

        
        # ----------------------------------------------------------
        # loop over proxy objects and calculate proxy estimates (Ye)
        # from reconstruction and from prior
        # ----------------------------------------------------------

        verif_times = np.arange(verif_period[0],verif_period[1]+1)
        
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
                ye_recon =  pobj.psm(Xrecon_statevector,Xrecon_state_info,Xrecon_coords)

                # prior (ensemble mean) Ye 
                Xb_tmp = np.mean(Xb_one,axis=1)
                # Broadcast over time/ensemble dimension (axis=1)
                Xb_ensmean = np.repeat(Xb_tmp[:,None],dimtime,axis=1)
                ye_prior = pobj.psm(Xb_ensmean,Xb_state_info,Xb_coords) # HERE

                # prior (full ensemble) Ye
                ye_prior_tmp = pobj.psm(Xb_one,Xb_state_info,Xb_coords) # HERE
                tmp = np.repeat(ye_prior_tmp[:,None],dimtime,axis=1)
                ye_prior_ens = tmp.T
                [_,Nens] = ye_prior_ens.shape
                
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
                ye_data = np.c_[verif_times.T,ye_recon.T,ye_prior.T,ye_prior_ens]
                df = pd.DataFrame(ye_data)
                df.columns = headers

                # Add proxy data
                frame = pd.DataFrame({'time':pobj.time, 'y': pobj.values})
                df = df.merge(frame, how='outer', on='time')

                # ensure all df entries are floats: if not, some calculations choke
                df = df.astype(np.float)

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
                    print('================================================')
                    print('Site:', sitetag)
                    print('status:', pstatus)
                    print('Number of verification points  :', obcount)            
                    print('Mean of proxy values           :', np.mean(df['y'][indok]))
                    print('Mean ensemble-mean             :', np.mean(df['Ye_recon'][indok]))
                    print('Mean ensemble-mean error       :', np.mean(df['Ye_recon_error'][indok]))
                    print('Ensemble-mean RMSE             :', rmsef(df['Ye_recon'][indok],df['y'][indok]))
                    print('Correlation                    :', np.corrcoef(df['y'][indok],df['Ye_recon'][indok])[0,1])
                    print('CE                             :', coefficient_efficiency(df['y'][indok],df['Ye_recon'][indok]))
                    print('Mean ensemble-mean(prior)      :', np.mean(df['Ye_prior'][indok]))
                    print('Mean ensemble-mean error(prior):', np.mean(df['Ye_prior_error'][indok]))
                    print('Ensemble-mean RMSE(prior)      :', rmsef(df['Ye_prior'][indok],df['y'][indok]))
                    corr = np.corrcoef(df['y'][indok],df['Ye_prior'][indok])[0,1]
                    if not np.isfinite(corr): corr = 0.0
                    print('Correlation (prior)            :', corr)
                    print('CE (prior)                     :', coefficient_efficiency(df['y'][indok],df['Ye_prior'][indok]))
                    print('================================================')            

                
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
                    print('pstatus undefined...')

        

        verif_listdict.append(evald_verif)
        assim_listdict.append(evald_assim)


    # check total nb. of sites in verif. set
    nb_tot_verif = sum([len(verif_listdict[k]) for k in range(len(verif_listdict))])

    
    # ==========================================================================
    # End of loop on MC iterations => Now calculate summary statistics ---------
    # ==========================================================================

    outdir = datadir_output+'/'+nexp+'/verifProxy_PSM_'+psm_str+'_'+str(verif_period[0])+'to'+str(verif_period[1])
    if not os.path.isdir(outdir):
        os.system('mkdir %s' % outdir)
    
    # -----------------------
    # With *** verif_dict ***
    # -----------------------

    if nb_tot_verif > 0:
    
        if Cfg.core.write_full_verif_dict:
            # Dump dictionary to pickle files
            outfile = open('%s/reconstruction_eval_verif_proxy_full.pckl' % (outdir),'w')
            pickle.dump(verif_listdict,outfile)
            outfile.close()

        # For each site :
        # List of sites in the verif dictionary
        list_tmp = []
        for i in range(len(verif_listdict)):
            for j in range(len(list(verif_listdict[i].keys()))):
                list_tmp.append(list(verif_listdict[i].keys())[j])
        list_sites = list(set(list_tmp)) # filter to unique elements
    
        summary_stats_verif = {}
        for k in range(len(list_sites)):
            # indices in verif_listdict where this site is present
            inds  = [j for j in range(len(verif_listdict)) if list_sites[k] in list(verif_listdict[j].keys())]

        
            summary_stats_verif[list_sites[k]] = {}
            summary_stats_verif[list_sites[k]]['lat']            = verif_listdict[inds[0]][list_sites[k]]['lat']
            summary_stats_verif[list_sites[k]]['lon']            = verif_listdict[inds[0]][list_sites[k]]['lon']
            summary_stats_verif[list_sites[k]]['alt']            = verif_listdict[inds[0]][list_sites[k]]['alt']
            summary_stats_verif[list_sites[k]]['NbPts']          = len(inds)
        
            #summary_stats_verif[list_sites[k]]['PSMslope']       = verif_listdict[inds[0]][list_sites[k]]['PSMslope']
            #summary_stats_verif[list_sites[k]]['PSMintercept']   = verif_listdict[inds[0]][list_sites[k]]['PSMintercept']
            #summary_stats_verif[list_sites[k]]['PSMcorrel']      = verif_listdict[inds[0]][list_sites[k]]['PSMcorrel']
            summary_stats_verif[list_sites[k]]['PSMinfo']      = verif_listdict[inds[0]][list_sites[k]]['PSMinfo']
        
        
            # These contain data for the "grand ensemble" (i.e. ensemble of realizations) for "kth" site
            nbpts = [verif_listdict[j][list_sites[k]]['NbEvalPts'] for j in inds]
            me    = [verif_listdict[j][list_sites[k]]['EnsMean_MeanError'] for j in inds]
            rmse  = [verif_listdict[j][list_sites[k]]['EnsMean_RMSE'] for j in inds]
            corr  = [verif_listdict[j][list_sites[k]]['EnsMean_Corr'] for j in inds]
            ce    = [verif_listdict[j][list_sites[k]]['EnsMean_CE'] for j in inds]
            corr_prior  = [verif_listdict[j][list_sites[k]]['PriorEnsMean_Corr'] for j in inds]
            ce_prior    = [verif_listdict[j][list_sites[k]]['PriorEnsMean_CE'] for j in inds]

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
            summary_stats_verif[list_sites[k]]['ts_years']       = verif_listdict[inds[0]][list_sites[k]]['ts_years']
            summary_stats_verif[list_sites[k]]['ts_ProxyValues'] = verif_listdict[inds[0]][list_sites[k]]['ts_ProxyValues']
            ts_recon = [verif_listdict[j][list_sites[k]]['ts_EnsMean'] for j in inds]
            summary_stats_verif[list_sites[k]]['ts_MeanRecon']   = np.mean(ts_recon,axis=0)
            summary_stats_verif[list_sites[k]]['ts_SpreadRecon'] = np.std(ts_recon,axis=0)
            # ens. calibration
            R = [verif_listdict[inds[0]][list_sites[k]]['PSMinfo']['R']]
            ensVar = np.mean(np.var(ts_recon,axis=0,ddof=1)) # !!! variance in grand ensemble (realizations, not DA ensemble) !!! 
            mse = np.mean(np.square(rmse))
            calib = mse/(ensVar+R)
            summary_stats_verif[list_sites[k]]['EnsCalib'] = calib[0]
            ## without R
            #calib = mse/(ensVar)
            #summary_stats_verif[list_sites[k]]['EnsCalib'] = calib

        # Dump data to pickle file
        outfile = open('%s/reconstruction_eval_verif_proxy_summary.pckl' % (outdir),'w')
        pickle.dump(summary_stats_verif,outfile)
        outfile.close()

        
    # -----------------------
    # With *** assim_dict ***
    # -----------------------
    # Dump dictionary to pickle files
    if Cfg.core.write_full_verif_dict:
        outfile = open('%s/reconstruction_eval_assim_proxy_full.pckl' % (outdir),'w')
        pickle.dump(assim_listdict,outfile)
        outfile.close()

    # For each site :    
    # List of sites in the assim dictionary
    list_tmp = []
    for i in range(len(assim_listdict)):
        for j in range(len(list(assim_listdict[i].keys()))):
            list_tmp.append(list(assim_listdict[i].keys())[j])
    list_sites = list(set(list_tmp)) # filter to unique elements

    summary_stats_assim = {}
    for k in range(len(list_sites)):
        # indices in assim_listdict where this site is present
        inds  = [j for j in range(len(assim_listdict)) if list_sites[k] in list(assim_listdict[j].keys())]

        summary_stats_assim[list_sites[k]] = {}
        summary_stats_assim[list_sites[k]]['lat']            = assim_listdict[inds[0]][list_sites[k]]['lat']
        summary_stats_assim[list_sites[k]]['lon']            = assim_listdict[inds[0]][list_sites[k]]['lon']
        summary_stats_assim[list_sites[k]]['alt']            = assim_listdict[inds[0]][list_sites[k]]['alt']
        summary_stats_assim[list_sites[k]]['NbPts']          = len(inds)
        
        #summary_stats_assim[list_sites[k]]['PSMslope']       = assim_listdict[inds[0]][list_sites[k]]['PSMslope']
        #summary_stats_assim[list_sites[k]]['PSMintercept']   = assim_listdict[inds[0]][list_sites[k]]['PSMintercept']
        #summary_stats_assim[list_sites[k]]['PSMcorrel']      = assim_listdict[inds[0]][list_sites[k]]['PSMcorrel']
        summary_stats_assim[list_sites[k]]['PSMinfo']      = assim_listdict[inds[0]][list_sites[k]]['PSMinfo']

        # These contain data for the "grand ensemble" (i.e. ensemble of realizations) for "kth" site
        nbpts      = [assim_listdict[j][list_sites[k]]['NbEvalPts'] for j in inds]
        me         = [assim_listdict[j][list_sites[k]]['EnsMean_MeanError'] for j in inds]
        rmse       = [assim_listdict[j][list_sites[k]]['EnsMean_RMSE'] for j in inds]
        corr       = [assim_listdict[j][list_sites[k]]['EnsMean_Corr'] for j in inds]
        ce         = [assim_listdict[j][list_sites[k]]['EnsMean_CE'] for j in inds]
        corr_prior = [assim_listdict[j][list_sites[k]]['PriorEnsMean_Corr'] for j in inds]
        ce_prior   = [assim_listdict[j][list_sites[k]]['PriorEnsMean_CE'] for j in inds]
        DAensCalib = [np.mean(assim_listdict[j][list_sites[k]]['ts_DAensCalib']) for j in inds]
        PriorEnsCalib = [np.mean(assim_listdict[j][list_sites[k]]['ts_PriorEnsCalib']) for j in inds]


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
        summary_stats_assim[list_sites[k]]['ts_years']       = assim_listdict[inds[0]][list_sites[k]]['ts_years']
        summary_stats_assim[list_sites[k]]['ts_ProxyValues'] = assim_listdict[inds[0]][list_sites[k]]['ts_ProxyValues']
        ts_recon = [assim_listdict[j][list_sites[k]]['ts_EnsMean'] for j in inds]
        summary_stats_assim[list_sites[k]]['ts_MeanRecon']   = np.mean(ts_recon,axis=0)
        summary_stats_assim[list_sites[k]]['ts_SpreadRecon'] = np.std(ts_recon,axis=0)        
        # ens. calibration
        R = [assim_listdict[inds[0]][list_sites[k]]['PSMinfo']['R']]
        ensVar = np.mean(np.var(ts_recon,axis=0,ddof=1)) # !!! variance of ens. means in grand ensemble (realizations, not DA ensemble) !!!
        mse = np.mean(np.square(rmse))
        calib = mse/(ensVar+R)
        summary_stats_assim[list_sites[k]]['EnsCalib'] = calib[0]
        

    # Dump data to pickle file
    outfile = open('%s/reconstruction_eval_assim_proxy_summary.pckl' % (outdir),'w')
    pickle.dump(summary_stats_assim,outfile)
    outfile.close()

    verif_time = time() - begin_time
    print('=======================================================')
    print('Verification completed in '+ str(verif_time/3600.0)+' hours')
    print('=======================================================')


# =============================================================================

if __name__ == '__main__':
    main()
