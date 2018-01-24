"""
 Module: LMR_PSMbuild.py
 
   Stand-alone tool building linear forward models (Proxy System Models) relating surface
   temperature and/or moisture to various proxy measurements, through linear regression 
   between proxy chronologies and historical gridded surface analyses.
   This updated version uses the Pandas DataFrame version of the proxy database and
   metadata, and can therefore be used on the PAGES2kS1 and NCDC pandas-formatted 
   proxy datafiles.
 
 Originator : Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
                            | January 2016

 Revisions: 
 - Included definitions related to calibration of linear PSMs for proxy records in the
   NCDC database. 
   [R. Tardif, U. of Washington, Spring 2016]
 - Included the GPCC precipitation historical dataset as a possible PSM calibration source.
   [R. Tardif, U. of Washington, Spring 2016]
 - Included the Dai PSDI historical dataset as a possible PSM calibration source.
   [R. Tardif, U. of Washington, Spring 2016]
 - Added definitions of parameters related to the calibration of bilinear (temperature/
   precipitation or moisture) PSMs. 
   [R. Tardif, U. of Washington, June 2016]
 - Added definitions related to the calibration of PSMs on the basis of a proxy record 
   seasonality metadata. 
   [ R. Tardif, Univ. of Washington, July 2016 ]
 - Adjustment to specification of psm type to calibrate for compatibility with modified 
   classes handling use of different psms per proxy types.
   [ R. Tardif, Univ. of Washington, August 2016 ]
 - Added filters on proxy records based on conditions of data availability.
   [ R. Tardif, Univ. of Washington, October 2016 ]
 - Code modifications for more efficient calibration. Upload of calibration data 
   is now done once up front and passed to psm calibration functions for use on all 
   proxy chronologies. The original code was structured in a way that the upload 
   of the calibration data was performed every time a psm for a proxy chronology
   was to be calibrated. 
   [ R. Tardif, Univ. of Washington, December 2016 ]
 - Added functionalities to objectively determine the seasonality of proxy chronologies 
   based on the quality of the fit to calibration data. 
   [ R. Tardif, Univ. of Washington, December 2016 ]
 - Adjustments to proxy types considered for new merged (PAGES2kv2 and NCDC)
   proxy datasets.   
   [ R. Tardif, Univ. of Washington, May 2017 ]
 - Renamed the proxy databases to less-confusing convention. 
   'pages' renamed as 'PAGES2kv1' and 'NCDC' renamed as 'LMRdb'
   [ R. Tardif, Univ. of Washington, Sept 2017 ]

"""
import os
import numpy as np
import pickle    
import datetime
from time import time
from os.path import join
from copy import deepcopy

import LMR_proxy_pandas_rework
import LMR_calibrate

import matplotlib.pyplot as plt


psm_info = \
"""
Forward model built using a linear PSM calibrated against historical observation-based product(s) of 2m air temperature and/or precipitation/moisture.
"""

# =========================================================================================
# START:  set user parameters here
# =========================================================================================

class v_core(object):
    """
    High-level parameters for the PSM builder

    Attributes
    ----------
    lmr_path: str
        Absolute path to central directory where the data (analyses (GISTEMP, HadCRUT...) and proxies) 
        are located
    calib_period: tuple(int)
        Time period considered for the calibration
    psm_type: str
        Indicates the type of PSM to calibrate. For now, allows 'linear' or 'bilinear'
    """

    ##** BEGIN User Parameters **##
    
    # lmr_path: where all the data is located ... model (prior), analyses (GISTEMP, HadCRUT...) and proxies.
    # lmr_path = '/home/katabatic/wperkins/data/LMR'
    # lmr_path = '/home/disk/kalman3/rtardif/LMR'
    lmr_path = '/home/disk/kalman3/rtardif/LMRpy3'    

    calib_period = (1850, 2015)

    # PSM type to calibrate: 'linear' or 'bilinear'
    psm_type = 'linear'
    #psm_type = 'bilinear'

    # Boolean to indicate whether upload of existing PSM data is to be performed. Keep False here. 
    load_psmobj = False 
    
    ##** END User Parameters **##
        
    def __init__(self):
        self.lmr_path = self.lmr_path
        self.calib_period = self.calib_period
        self.psm_type = self.psm_type
        try:
            self.load_psmobj = self.load_psmobj
        except:
            pass
            
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


    ##** BEGIN User Parameters **##

    # Which proxy database to use ?
    # use_from = ['PAGES2kv1']
    use_from = ['LMRdb']

    ##** END User Parameters **##
    
    proxy_frac = 1.0 # this needs to remain = 1.0 if all possible proxies are to be considered for calibration

    # Filtering proxy records on conditions of data availability during
    # the reconstruction period. 
    # - Filtrering disabled if proxy_availability_filter = False.
    # - If proxy_availability_filter = True, only records with
    #   oldest and youngest data outside or at edges of the recon. period
    #   are considered for assimilation.
    # - Testing for record completeness through the application of a threshold
    #   on data availability fraction (proxy_availability_fraction parameter).
    #   Records with a fraction of available data (ratio of valid data over
    #   the maximum data expected within the reconstruction period) below the
    #   user-defined threshold are omitted. 
    #   Set this threshold to 0.0 if you do not want this threshold applied.
    #   Set this threshold to 1.0 to prevent assimilation of records with
    #   any missing data within the reconstruction period. 
    proxy_availability_filter = False
    proxy_availability_fraction = 0.0
    
    
    # ---------------------
    # for PAGES2kv1 proxies
    # ---------------------
    class _PAGES2kv1(object):
        """
        Parameters for PAGES2kv1 Proxy class

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
        datafile_proxy = 'Pages2kv1_Proxies.df.pckl'
        metafile_proxy = 'Pages2kv1_Metadata.df.pckl'
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

        # Specify, per proxy type, whether proxy seasonality is to be objectively determined or
        # metadata contained in the proxy data files is to be used in the psm calibration.
        # Lists indicating which seasons are to be considered are also specified here.
        # Note: annual = [1,2,3,4,5,6,7,8,9,10,11,12]
        #       JJA    = [6,7,8]
        #       JJASON = [6,7,8,9,10,11]
        #       DJF    = [-12,1,2]
        #       DJFMAM = [-12,1,2,3,4,5]
        proxy_psm_seasonality = {
            'Tree ring_Width'      : {'flag':True,
                                      'seasons': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[6,7,8,9,10,11],[-12,1,2],[-12,1,2,3,4,5]]}, 
            'Tree ring_Density'    : {'flag':True,
                                      'seasons': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[6,7,8,9,10,11],[-12,1,2],[-12,1,2,3,4,5]]}, 
            'Ice core_d18O'        : {'flag':False,
                                      'seasons_T': [],
                                      'seasons_M': []}, 
            'Ice core_d2H'         : {'flag':False,
                                      'seasons_T': [],
                                      'seasons_M': []}, 
            'Ice core_Accumulation': {'flag':False,
                                      'seasons_T': [],
                                      'seasons_M': []}, 
            'Coral_d18O'           : {'flag':False,
                                      'seasons_T': [],
                                      'seasons_M': []}, 
            'Coral_Luminescence'   : {'flag':False,
                                      'seasons_T': [],
                                      'seasons_M': []}, 
            'Lake sediment_All'    : {'flag':False,
                                      'seasons_T': [],
                                      'seasons_M': []}, 
            'Marine sediment_All'  : {'flag':False,
                                      'seasons_T': [],
                                      'seasons_M': []}, 
            'Speleothem_All'       : {'flag':False,
                                      'seasons_T': [],
                                      'seasons_M': []}, 
            }
        
        
        # A blacklist on proxy records, to prevent assimilation of chronologies
        # known to be duplicates.
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
            self.proxy_psm_seasonality = deepcopy(self.proxy_psm_seasonality)
            self.proxy_blacklist = list(self.proxy_blacklist)
            self.proxy_availability_filter = v_proxies.proxy_availability_filter
            self.proxy_availability_fraction = v_proxies.proxy_availability_fraction

            self.proxy_psm_type = {}
            for p in self.proxy_order: self.proxy_psm_type[p] = v_core.psm_type
            
            # Create mapping for Proxy Type/Measurement Type to type names above
            self.proxy_type_mapping = {}
            for ptype, measurements in self.proxy_assim2.items():
                # Fetch proxy type name that occurs before underscore
                type_name = ptype.split('_', 1)[0]
                for measure in measurements:
                    self.proxy_type_mapping[(type_name, measure)] = ptype

            self.simple_filters = {'PAGES 2k Region': self.regions,
                                   'Resolution (yr)': self.proxy_resolution}


    # -----------------
    # for LMRdb proxies
    # -----------------
    class _LMRdb(object):
        """
        Parameters for LMRdb proxy class

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

        ##** BEGIN User Parameters **##

        # db version:
        #  v0.0.0: initial collection of NCDC-templated proxies, including PAGES2k2013 trees
        #  v0.1.0: updated collection of NCDC-templated proxies, without PAGES2k2013 trees
        #          but with an early version of the PAGES2k2017 (phase2) proxies converted
        #          in NCDC-templated text files.
        #  v0.2.0: merge of v0.1.0 NCDC proxies (w/o the NCDC-templated PAGES2k phase2) with
        #          published version (2.0.0) of the PAGES2k2017 proxies contained in a pickle
        #          file exported directly from the LiPD database. 
        #dbversion = 'v0.0.0'
        #dbversion = 'v0.1.0'
        dbversion = 'v0.2.0' 
        
        datadir_proxy = None
        datafile_proxy = 'LMRdb_%s_Proxies.df.pckl' %(dbversion)
        metafile_proxy = 'LMRdb_%s_Metadata.df.pckl' %(dbversion)
        dataformat_proxy = 'DF'

        regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
                   'North America', 'South America']

        proxy_resolution = [1.0]

        proxy_timeseries_kind = 'asis' # 'anom' for anomalies (temporal mean removed) or 'asis' to keep unchanged

        ##** END User Parameters **##

        
        # Limit proxies to those included in the following databases
        database_filter = []

        # A blacklist on proxy records, to prevent processing of specific chronologies.
        proxy_blacklist = []

        # DO NOT CHANGE FORMAT BELOW
        proxy_order = [
#old        'Tree Rings_WidthPages',
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
            'Marine Cores_tex86',
            'Marine Cores_uk37',
            'Bivalve_d18O',
            'Speleothems_d18O',
            ]

        proxy_assim2 = {
            'Bivalve_d18O'                  : ['d18O'],
            'Corals and Sclerosponges_d18O' : ['d18O','delta18O','d18o','d18O_stk','d18O_int','d18O_norm',
                                               'd18o_avg','d18o_ave','dO18','d18O_4'],
            'Corals and Sclerosponges_Rates': ['ext','calc','calcification','calcification rate','composite'],
            'Corals and Sclerosponges_SrCa' : ['Sr/Ca','Sr_Ca','Sr/Ca_norm','Sr/Ca_anom','Sr/Ca_int'],\
            'Corals and Sclerosponges_d14C' : ['d14C','d14c','ac_d14c'],
            'Corals and Sclerosponges_d13C' : ['d13C','d13c','d13c_ave','d13c_ann_ave','d13C_int'],
            'Corals and Sclerosponges_Sr'   : ['Sr'],
            'Corals and Sclerosponges_BaCa' : ['Ba/Ca'],
            'Corals and Sclerosponges_CdCa' : ['Cd/Ca'],
            'Corals and Sclerosponges_MgCa' : ['Mg/Ca'],
            'Corals and Sclerosponges_UCa'  : ['U/Ca','U/Ca_anom'],
            'Corals and Sclerosponges_Pb'   : ['Pb'],
            'Ice Cores_d18O'                : ['d18O','delta18O','delta18o','d18o','dO18',
                                               'd18o_int','d18O_int',
                                               'd18O_norm','d18o_norm',
                                               'd18O_anom'],
            'Ice Cores_dD'                  : ['deltaD','delD','dD'],
            'Ice Cores_Accumulation'        : ['accum','accumu'],
            'Ice Cores_MeltFeature'         : ['MFP','melt'],
            'Lake Cores_Varve'              : ['varve', 'varve_thickness', 'varve thickness','thickness'],
            'Lake Cores_BioMarkers'         : ['Uk37', 'TEX86', 'tex86'],
            'Lake Cores_GeoChem'            : ['Sr/Ca', 'Mg/Ca', 'Cl_cont'],
            'Lake Cores_Misc'               : ['RABD660_670','X_radiograph_dark_layer','massacum'],
            'Marine Cores_d18O'             : ['d18O'],
            'Marine Cores_tex86'            : ['tex86'],
            'Marine Cores_uk37'             : ['uk37','UK37'],
            'Speleothems_d18O'              : ['d18O'],
            'Speleothems_d13C'              : ['d13C'],
            'Tree Rings_WidthBreit'         : ['trsgi_breit'],
            'Tree Rings_WidthPages2'        : ['trsgi'], 
#old         'Tree Rings_WidthPages'         : ['TRW',
#old                                            'ERW',
#old                                            'LRW'],
            'Tree Rings_WoodDensity'        : ['max_d',
                                               'min_d',
                                               'early_d',
                                               'earl_d',
                                               'late_d',
                                               'density',
                                               'MXD'],
            'Tree Rings_Isotopes'           : ['d18O'],
            }


        # Specify, per proxy type, whether proxy seasonality is to be objectively determined or
        # metadata contained in the proxy data files is to be used in the psm calibration.
        # Lists indicating which seasons are to be considered are also specified here.
        # Note: annual = [1,2,3,4,5,6,7,8,9,10,11,12]
        #       JJA    = [6,7,8]
        #       JJASON = [6,7,8,9,10,11]
        #       DJF    = [-12,1,2]
        #       DJFMAM = [-12,1,2,3,4,5]
        proxy_psm_seasonality = {
            'Bivalve_d18O'                 : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Corals and Sclerosponges_d18O' : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Corals and Sclerosponges_SrCa' : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Corals and Sclerosponges_Rates': {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Ice Cores_d18O'                : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Ice Cores_dD'                  : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Ice Cores_Accumulation'        : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Ice Cores_MeltFeature'         : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Lake Cores_Varve'              : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Lake Cores_BioMarkers'         : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Lake Cores_GeoChem'            : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Lake Cores_Misc'               : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Marine Cores_d18O'             : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Marine Cores_tex86'            : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Marine Cores_uk37'             : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []},
            'Tree Rings_WidthBreit'         : {'flag':True,
                                               'seasons_T': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
                                               'seasons_M': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]]},
            'Tree Rings_WidthPages2'        : {'flag':True,
                                               'seasons_T': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
                                               'seasons_M': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]]},
#            'Tree Rings_WidthPages'         : {'flag':True,
#                                               'seasons_T': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
#                                               'seasons_M': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]]},
            'Tree Rings_WoodDensity'        : {'flag':True,
                                               'seasons_T': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
                                               'seasons_M': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]]},
            'Tree Rings_Isotopes'           : {'flag':True,
                                               'seasons_T': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
                                               'seasons_M': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]]},
            'Speleothems_d18O'              : {'flag':False,
                                               'seasons_T': [],
                                               'seasons_M': []}
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

            self.dbversion = self.dbversion
            self.dataformat_proxy = self.dataformat_proxy
            self.regions = list(self.regions)
            self.proxy_resolution = self.proxy_resolution
            self.proxy_timeseries_kind = self.proxy_timeseries_kind
            self.proxy_order = list(self.proxy_order)
            self.proxy_assim2 = deepcopy(self.proxy_assim2)
            self.proxy_psm_seasonality = deepcopy(self.proxy_psm_seasonality)
            self.database_filter = list(self.database_filter)
            self.proxy_blacklist = list(self.proxy_blacklist)
            self.proxy_availability_filter = v_proxies.proxy_availability_filter
            self.proxy_availability_fraction = v_proxies.proxy_availability_fraction
            
            self.proxy_psm_type = {}
            for p in self.proxy_order: self.proxy_psm_type[p] = v_core.psm_type

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
        self.PAGES2kv1 = self._PAGES2kv1(**kwargs)
        self.LMRdb = self._LMRdb(**kwargs)



class v_psm(object):
    """
    Parameters for PSM classes

    Attributes
    ----------
    avgPeriod: str
        PSM calibrated on annual or seasonal data: allowed tags are 'annual' or 'season'
    """

    ##** BEGIN User Parameters **##

    # ...
    load_precalib = False
    
    # PSM calibrated on annual or seasonal data: allowed tags are 'annual' or 'season'
    avgPeriod = 'annual'
    # avgPeriod = 'season'

    # Boolean flag indicating whether PSMs are to be calibrated using objectively-derived
    # proxy seasonality instead of using the "seasonality" metadata included in the data
    # files.
    # Activated only if avgPeriod = 'season' above.
    # If set to True, refer back to the appropriate proxy class above
    # (proxy_psm_seasonality dict.) to set which proxy type(s) and associated seasons
    # will be considered. 
    test_proxy_seasonality = False
    
    ##** END User Parameters **##

    if avgPeriod == 'season':
        if test_proxy_seasonality:
            avgPeriod = avgPeriod+'PSM'
        else:
            avgPeriod = avgPeriod+'META'
    
    
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

        ##** BEGIN User Parameters **##
        
        # Choice between:
        #datatag_calib = 'MLOST'
        #datafile_calib = 'MLOST_air.mon.anom_V3.5.4.nc'
        # or
        datatag_calib = 'GISTEMP'
        datafile_calib = 'gistemp1200_ERSSTv4.nc'
        # or
        #datatag_calib = 'HadCRUT'
        #datafile_calib = 'HadCRUT.4.4.0.0.median.nc'
        # or 
        #datatag_calib = 'BerkeleyEarth'
        #datafile_calib = 'Land_and_Ocean_LatLong1.nc'
        # or 
        #datatag_calib = 'GPCC'
        #datafile_calib = 'GPCC_precip.mon.flux.1x1.v6.nc'  # Precipitation flux (kg m2 s-1)
        # or
        #datatag_calib = 'DaiPDSI'
        #datafile_calib = 'Dai_pdsi.mon.mean.selfcalibrated_185001-201412.nc'
        # or
        #datatag_calib = 'SPEI'
        #datafile_calib = 'spei_monthly_v2.4_190001-201412.nc'
                
        pre_calib_datafile = None
        
        psm_r_crit = 0.0

        ##** END User Parameters **##
        
        
        def __init__(self):
            self.datatag_calib = self.datatag_calib
            self.datafile_calib = self.datafile_calib
            self.psm_r_crit = self.psm_r_crit
            self.avgPeriod = v_psm.avgPeriod
            
            if '-'.join(v_proxies.use_from) == 'PAGES2kv1' and 'season' in self.avgPeriod:
                print('ERROR: Trying to use seasonality information with the PAGES2kv1 proxy records.')
                print('       No seasonality metadata provided in that dataset. Exiting!')
                print('       Change avgPeriod to "annual" in your configuration.')
                raise SystemExit()
            
            if self.datadir_calib is None:
                self.datadir_calib = join(v_core.lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = self.datadir_calib

            if self.pre_calib_datafile is None:
                if '-'.join(v_proxies.use_from) == 'LMRdb':
                    dbversion = v_proxies._LMRdb.dbversion
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

        ##** BEGIN User Parameters **##
        
        datadir_calib = None

        # calibration w.r.t. temperature
        # -----------------------------
        # Choice between:
        #
        datatag_calib_T = 'GISTEMP'
        datafile_calib_T = 'gistemp1200_ERSSTv4.nc'
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
        # or
        #datatag_calib_P = 'SPEI'
        #datafile_calib_P = 'spei_monthly_v2.4_190001-201412.nc'

        pre_calib_datafile = None
        psm_r_crit = 0.0

        
        ##** END User Parameters **##
        
        def __init__(self):
            self.datatag_calib_T = self.datatag_calib_T
            self.datafile_calib_T = self.datafile_calib_T
            self.datatag_calib_P = self.datatag_calib_P
            self.datafile_calib_P = self.datafile_calib_P
            self.psm_r_crit = self.psm_r_crit
            self.avgPeriod = v_psm.avgPeriod

            if '-'.join(v_proxies.use_from) == 'PAGES2kv1' and 'season' in self.avgPeriod:
                print('ERROR: Trying to use seasonality information with the PAGES2kv1 proxy records.')
                print('       No seasonality metadata provided in that dataset. Exiting!')
                print('       Change avgPeriod to "annual" in your configuration.')
                raise SystemExit()
            
            if self.datadir_calib is None:
                self.datadir_calib = join(v_core.lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = self.datadir_calib


            if self.pre_calib_datafile is None:
                if '-'.join(v_proxies.use_from) == 'LMRdb':
                    dbversion = v_proxies._LMRdb.dbversion
                    filename = ('PSMs_' + '-'.join(v_proxies.use_from) +
                                '_' + dbversion +
                                '_' + self.avgPeriod +
                                '_'+self.datatag_calib_T +
                                '_'+self.datatag_calib_P +'.pckl')
                else:
                    filename = ('PSMs_' + '-'.join(v_proxies.use_from) +
                                '_'+self.datatag_calib_T +
                                '_'+self.datatag_calib_P +'.pckl')

                self.pre_calib_datafile = join(v_core.lmr_path,
                                               'PSM',
                                               filename)
            else:
                self.pre_calib_datafile = self.pre_calib_datafile
    

    
    # Initialize subclasses with all attributes
    def __init__(self, **kwargs):
        self.linear = self._linear(**kwargs)
        self.bilinear = self._bilinear(**kwargs)

# =========================================================================================
# END:  set user parameters here
# =========================================================================================

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

    begin_time = time()

    proxy_database = Cfg.proxies.use_from[0]
    psm_type = Cfg.core.psm_type

    print('Proxies             :', proxy_database)
    print('PSM type            :', psm_type)
    print('Calib. period       :', Cfg.core.calib_period)
    
    if proxy_database == 'PAGES2kv1':
        print('Proxy data location :', Cfg.proxies.PAGES2kv1.datadir_proxy)
        proxy_psm_seasonality =  Cfg.proxies.PAGES2kv1.proxy_psm_seasonality
    elif proxy_database == 'LMRdb':
        print('Proxy data location :', Cfg.proxies.LMRdb.datadir_proxy)
        proxy_psm_seasonality =  Cfg.proxies.LMRdb.proxy_psm_seasonality
    else:
        raise SystemExit('ERROR in specification of proxy database. Exiting!')
        
    # psm type
    if psm_type == 'linear':
        datatag_calib = Cfg.psm.linear.datatag_calib
        print('Calibration source  :', datatag_calib)
        psm_file = Cfg.psm.linear.pre_calib_datafile
        calib_avgPeriod = Cfg.psm.linear.avgPeriod

        # load calibration data
        C = LMR_calibrate.calibration_assignment(datatag_calib)
        C.datadir_calib = Cfg.psm.linear.datadir_calib
        C.read_calibration()

    elif psm_type == 'bilinear':
        datatag_calib_T = Cfg.psm.bilinear.datatag_calib_T
        datatag_calib_P = Cfg.psm.bilinear.datatag_calib_P
        print('Calibration sources :', datatag_calib_T, '+', datatag_calib_P)
        psm_file = Cfg.psm.bilinear.pre_calib_datafile
        calib_avgPeriod = Cfg.psm.bilinear.avgPeriod

        # load calibration data: two calibration objects, temperature and precipitation/moisture
        C_T = LMR_calibrate.calibration_assignment(datatag_calib_T)
        C_T.datadir_calib = Cfg.psm.bilinear.datadir_calib
        C_T.read_calibration()
        C_P = LMR_calibrate.calibration_assignment(datatag_calib_P)
        C_P.datadir_calib = Cfg.psm.bilinear.datadir_calib
        C_P.read_calibration()

    else:
        raise SystemExit('ERROR: problem with the specified type of psm. Exiting!')
    
    print('PSM calibration/parameters file:', psm_file)

    
    # corresponding file containing complete diagnostics
    psm_file_diag = psm_file.replace('.pckl', '_diag.pckl')
    
    # Check if psm_file already exists, archive it with current date/time if it exists
    # and replace by new file
    if os.path.isfile(psm_file):        
        nowstr = datetime.datetime.now().strftime("%Y%m%d:%H%M")
        os.system('mv %s %s_%s.pckl' %(psm_file,psm_file.rstrip('.pckl'),nowstr) )
        if os.path.isfile(psm_file_diag):
            os.system('mv %s %s_%s.pckl' %(psm_file_diag,psm_file_diag.rstrip('.pckl'),nowstr) )
        
    prox_manager = LMR_proxy_pandas_rework.ProxyManager(Cfg, Cfg.core.calib_period)
    type_site_calib = prox_manager.assim_ids_by_group

    print('--------------------------------------------------------------------')
    print('Total proxies available: counts per proxy type:')
    # count the total number of proxies
    total_proxy_count = len(prox_manager.ind_assim)
    for pkey, plist in sorted(type_site_calib.items()):
        print('%45s : %5d' % (pkey, len(plist)))
    print('--------------------------------------------------------------------')
    print('%45s : %5d' % ('TOTAL', total_proxy_count))
    print('--------------------------------------------------------------------')

    
    # Loop over proxies
    psm_dict = {}
    psm_dict_diag = {}
    for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
        sitetag = (Y.type,Y.id)

        print(' ')
        print(sitetag)

        
        # -----------------------------------------------------
        # Prep: defining seasons to be tested, depending on the
        # chosen configuration 
        # -----------------------------------------------------
        if calib_avgPeriod == 'annual':
            # override any proxy seasonality metadata with calendar year
            seasons = [[1,2,3,4,5,6,7,8,9,10,11,12]]
            if psm_type == 'bilinear':
                seasons_T = seasons[:]
                seasons_M = seasons[:]

        elif 'season' in calib_avgPeriod:
            # try to determine seasonality objectively ?
            if Cfg.psm.test_proxy_seasonality and proxy_psm_seasonality[Y.type]['flag']:

                # psm type is linear or bilinear ?
                if psm_type == 'linear':
                    # if linear, calibrating against temperature or moisture ?
                    if datatag_calib == 'DaiPDSI' or datatag_calib == 'GPCC':
                        seasons = proxy_psm_seasonality[Y.type]['seasons_M'][:]
                    else:
                        seasons = proxy_psm_seasonality[Y.type]['seasons_T'][:]

                    # If not part of list already, insert entry from metadata at beginning of list
                    if Y.seasonality not in seasons:
                        seasons.insert(0, Y.seasonality)

                    
                elif psm_type == 'bilinear':
                    seasons_T = proxy_psm_seasonality[Y.type]['seasons_T'][:]
                    seasons_M = proxy_psm_seasonality[Y.type]['seasons_M'][:]

                    # insert entry from metadata at beginning of list, if not part of list already
                    if Y.seasonality not in seasons_T: seasons_T.insert(0, Y.seasonality)
                    if Y.seasonality not in seasons_M: seasons_M.insert(0, Y.seasonality)
                    
            else:
                # revert back to proxy metadata
                seasons = [Y.seasonality]
                if psm_type == 'bilinear':
                    seasons_T = seasons[:]
                    seasons_M = seasons[:]

        else: 
            raise SystemExit('Error in choice of seasonality. Exiting!')

        # ---------------------------------------------------------
        # Calculating the regressions and associated statistics for
        # all the seasons to be tested
        # ---------------------------------------------------------
        if psm_type == 'linear':
            # --------------
            # --- linear ---
            # --------------
            nbseasons = len(seasons)
            defaultnb = 1.0e10 # arbitrary large number
            metric = np.zeros(nbseasons);
            metric[:] = defaultnb
            i = 0
            test_psm_obj_dict = {}

            # Loop over seasonality iterable
            for s in seasons:
                Y.seasonality = s # re-assign seasonality to proxy object
            
                # Create a psm object
                psm_obj = Y.get_psm_obj(Cfg,Y.type)

                try:
                    # Calibrate the statistical forward model (psm)
                    test_psm_obj = psm_obj(Cfg, Y, calib_obj=C)
                    
                    print('=>', "{:2d}".format(i),
                           "{:45s}".format(str(s)),
                           "{:12.4f}".format(test_psm_obj.slope),
                           "{:12.4f}".format(test_psm_obj.intercept),
                           "{:12.4f}".format(test_psm_obj.corr),
                           "{:12.4f}".format(test_psm_obj.R),
                           '(', "{:10.5f}".format(test_psm_obj.R2adj), ')')
                    
                                        
                    # BIC used as the selection criterion
                    #metric[i] = test_psm_obj.BIC
                    # Adjusted R-squared used as the selection criterion
                    metric[i] = test_psm_obj.R2adj
                    
                    test_psm_obj_dict[str(s)] =  test_psm_obj
                
                except ValueError as e:
                    print(e)
                    print('Test on seasonnality %s could not be completed.' %
                          str(s))


                i += 1
            
        elif psm_type == 'bilinear':
            # ----------------
            # --- bilinear ---
            # ----------------
            nbseasons = len(seasons_T) * len(seasons_M)
            defaultnb = 1.0e10 # arbitrary large number
            metric = np.zeros(nbseasons);
            metric[:] = defaultnb
            seasons = np.empty(shape=[nbseasons],dtype=object)
            i = 0
            test_psm_obj_dict = {}

            # Loop over seasonality iterables
            for sT in seasons_T:
                Y.seasonality_T = sT
                for sM in seasons_M:
                    Y.seasonality_P = sM
                    
                    # Create a psm object
                    psm_obj = Y.get_psm_obj(Cfg,Y.type)

                    try:
                        # Calibrate the statistical forward model (psm)
                        test_psm_obj = psm_obj(Cfg, Y, calib_obj_T=C_T, calib_obj_P=C_P)

                        print('=>', "{:2d}".format(i),
                              "{:40s}".format(str(sT)),
                              "{:40s}".format(str(sM)),
                              "{:12.4f}".format(test_psm_obj.slope_temperature),
                              "{:12.4f}".format(test_psm_obj.slope_moisture),
                              "{:12.4f}".format(test_psm_obj.intercept),
                              "{:12.4f}".format(test_psm_obj.corr),
                              "{:12.4f}".format(test_psm_obj.R))
                
                        # BIC used as the selection criterion
                        #metric[i] = test_psm_obj.BIC
                        # Adjusted R-squared used as the selection criterion
                        metric[i] = test_psm_obj.R2adj

                        # Associated pair of seasonalities (as tuple of lists)
                        # and psm object
                        seasons[i] = (sT,sM)
                        test_psm_obj_dict[str((sT,sM))] =  test_psm_obj
                                                
                    except:
                        print('Test on seasonnality pair %s could not be completed.' %(str(sT)+':'+str(sM)))

                    
                    i += 1


        # if calculations could not be completed, just move on to next
        # proxy record
        if np.all(metric==defaultnb):
            print('Calibration could not be completed...Skipping proxy record.')
            continue


        # Select the psm object corresponding to the season (linear) or
        # pair of seasons (bilinear) that provide the best fit 
        # -------------------------------------------------------------
        # Select the "seasonal" model (psm)
        # criterion: min of metric if BIC, max if adjusted R-squared
        indmin = np.argmin(metric)
        indmax = np.argmax(metric)        
        #select_psm_obj = test_psm_obj_dict[str(seasons[indmin])]
        #Y.seasonality = seasons[indmin] # a list if linear psm, a tuple of lists if bilinear
        select_psm_obj = test_psm_obj_dict[str(seasons[indmax])]
        Y.seasonality = seasons[indmax] # a list if linear psm, a tuple of lists if bilinear

        Y.psm_obj = select_psm_obj
        Y.psm = Y.psm_obj.psm

        
        # Load proxy object in dictionary
        # -------------------------------
        # Site info
        psm_dict[sitetag] = {}
        psm_dict[sitetag]['lat']   = Y.lat
        psm_dict[sitetag]['lon']   = Y.lon
        psm_dict[sitetag]['elev']  = Y.elev

        # selected PSM info into dictionary
        psm_dict[sitetag]['Seasonality']  = Y.seasonality
        psm_dict[sitetag]['NbCalPts']     = Y.psm_obj.NbPts
        psm_dict[sitetag]['PSMintercept'] = Y.psm_obj.intercept
        psm_dict[sitetag]['PSMcorrel']    = Y.psm_obj.corr
        psm_dict[sitetag]['PSMmse']       = Y.psm_obj.R

        # diagnostic information
        psm_dict_diag[sitetag] = {}
        
        if psm_type == 'linear':
            psm_dict[sitetag]['calib']        = datatag_calib
            psm_dict[sitetag]['PSMslope']     = Y.psm_obj.slope
            psm_dict[sitetag]['PSMintercept'] = Y.psm_obj.intercept
            psm_dict[sitetag]['fitBIC']       = Y.psm_obj.BIC
            psm_dict[sitetag]['fitR2adj']     = Y.psm_obj.R2adj
            
            # diagnostic information
            # ----------------------
            # copy main psm attributes
            psm_dict_diag[sitetag] = deepcopy(psm_dict[sitetag])
            # add diagnostics
            psm_dict_diag[sitetag]['calib_time'] = Y.psm_obj.calib_time
            psm_dict_diag[sitetag]['calib_refer_values'] = Y.psm_obj.calib_refer_values
            psm_dict_diag[sitetag]['calib_proxy_values'] = Y.psm_obj.calib_proxy_values
            psm_dict_diag[sitetag]['calib_fit_values'] = Y.psm_obj.calib_proxy_fit

        elif psm_type == 'bilinear':
            psm_dict[sitetag]['calib_temperature']    = datatag_calib_T
            psm_dict[sitetag]['calib_moisture']       = datatag_calib_P
            psm_dict[sitetag]['PSMslope_temperature'] = Y.psm_obj.slope_temperature
            psm_dict[sitetag]['PSMslope_moisture']    = Y.psm_obj.slope_moisture
            psm_dict[sitetag]['PSMintercept']         = Y.psm_obj.intercept
            psm_dict[sitetag]['fitBIC']               = Y.psm_obj.BIC
            psm_dict[sitetag]['fitR2adj']             = Y.psm_obj.R2adj            
            
            # diagnostic information
            # ----------------------
            # copy main psm attributes
            psm_dict_diag[sitetag] = deepcopy(psm_dict[sitetag])
            # add diagnostics
            psm_dict_diag[sitetag]['calib_time'] = Y.psm_obj.calib_time
            psm_dict_diag[sitetag]['calib_temperature_refer_values'] = Y.psm_obj.calib_temperature_refer_values
            psm_dict_diag[sitetag]['calib_moisture_refer_values'] = Y.psm_obj.calib_moisture_refer_values
            psm_dict_diag[sitetag]['calib_proxy_values'] = Y.psm_obj.calib_proxy_values
            psm_dict_diag[sitetag]['calib_fit_values'] = Y.psm_obj.calib_proxy_fit
            
        else:
            raise SystemExit('ERROR: problem with the type of psm!')


    # Summary of calibrated proxy sites
    # ---------------------------------
    calibrated_sites = list(psm_dict.keys())
    calibrated_types = list(set([item[0] for item in calibrated_sites]))

    print('--------------------------------------------------------------------')
    print('Calibrated proxies : counts per proxy type:')
    # count the total number of proxies
    total_proxy_count = len(calibrated_sites)

    for ptype in sorted(calibrated_types):
        plist= [item[1] for item in calibrated_sites if item[0] == ptype]
        print('%45s : %5d' % (ptype, len(plist)))
    print('--------------------------------------------------------------------')
    print('%45s : %5d' % ('TOTAL', total_proxy_count))
    print('--------------------------------------------------------------------')

    
    # Dump dictionaries to pickle files
    outfile = open('%s' % (psm_file),'wb')
    # using protocol 2 for more efficient storing
    pickle.dump(psm_dict,outfile,protocol=2)
    pickle.dump(psm_info,outfile,protocol=2)
    outfile.close()

    outfile_diag = open('%s' % (psm_file_diag),'wb')
    # using protocol 2 for more efficient storing
    pickle.dump(psm_dict_diag,outfile_diag,protocol=2)
    pickle.dump(psm_info,outfile_diag,protocol=2)
    outfile_diag.close()

    
    end_time = time() - begin_time
    print('=========================================================')
    print('PSM calibration completed in '+ str(end_time/60.0)+' mins')
    print('=========================================================')

# =============================================================================

if __name__ == '__main__':
    main()
