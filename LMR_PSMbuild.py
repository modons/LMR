"""
 Module: LMR_PSMbuild.py
 
   Stand-alone tool building linear forward models (Proxy System Models) relating surface
   temperature to various proxy measurements, through linear regression between proxy 
   chronologies and historical gridded surface temperature analyses.
   This updated version uses the Pandas DataFrame version of the proxy database and
   metadata, and can therefore be used on the PAGES2kS1 and NCDC pandas-formatted 
   proxy datafiles.
 
 Originator : Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
                            | January 2016

 Revisions: 
 - Included definitions related to calibration of linear PSMs for proxy records in the
   NCDC database. [R. Tardif, U. of Washington, Spring 2016]
 - Included the GPCC precipitation historical dataset as a possible PSM calibration source.
   [R. Tardif, U. of Washington, Spring 2016]
 - Included the Dai PSDI historical dataset as a possible PSM calibration source.
   [R. Tardif, U. of Washington, Spring 2016]
 - Added definitions of parameters related to the calibration of bilinear (temperature/
   precipitation or moisture) PSMs. [R. Tardif, U. of Washington, June 2016]
 - Added definitions related to the calibration of PSMs on the basis of a proxy record 
   seasonality metadata. [ R. Tardif, Univ. of Washington, July 2016 ]
 - Adjustment to specification of psm type to calibrate for compatibility with modified 
   classes handling use of different psms per proxy types.
   [ R. Tardif, Univ. of Washington, August 2016 ]
"""
import os
import numpy as np
import cPickle    
import datetime
from time import time
from os.path import join
from copy import deepcopy

import LMR_proxy_pandas_rework

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
    #lmr_path = '/home/chaos2/wperkins/data/LMR'
    #lmr_path = '/home/disk/kalman3/rtardif/LMR'
    lmr_path = '/home/scec-00/lmr/erbm/LMR'

    calib_period = (1850, 2010)

    # PSM type to calibrate: 'linear' or 'bilinear'
    psm_type = 'linear'
    #psm_type = 'bilinear'

    ##** END User Parameters **##
        
    def __init__(self):
        self.lmr_path = self.lmr_path
        self.calib_period = self.calib_period
        self.psm_type = self.psm_type
    
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
    #use_from = ['pages']
    use_from = ['NCDC']

    ##** END User Parameters **##
    
    proxy_frac = 1.0 # this needs to remain = 1.0 if all possible proxies are to be considered for calibration

    # -------------------
    # for PAGES2k proxies
    # -------------------
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
            self.proxy_blacklist = list(self.proxy_blacklist)

            self.proxy_psm_type = {}
            for p in self.proxy_order: self.proxy_psm_type[p] = v_core.psm_type
            
            # Create mapping for Proxy Type/Measurement Type to type names above
            self.proxy_type_mapping = {}
            for ptype, measurements in self.proxy_assim2.iteritems():
                # Fetch proxy type name that occurs before underscore
                type_name = ptype.split('_', 1)[0]
                for measure in measurements:
                    self.proxy_type_mapping[(type_name, measure)] = ptype

            self.simple_filters = {'PAGES 2k Region': self.regions,
                                   'Resolution (yr)': self.proxy_resolution}


    # ----------------
    # for NCDC proxies
    # ----------------
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
        
        #dbversion = 'v0.0.0'
        #dbversion = 'v0.1.0'
        dbversion = 'Pages2kv2'  # This uses the Pages2kv2 database, without any NCDC records!
        
        datadir_proxy = None
        datafile_proxy = 'NCDC_%s_Proxies.df.pckl' %(dbversion)
        metafile_proxy = 'NCDC_%s_Metadata.df.pckl' %(dbversion)
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
            'Tree Rings_WoodDensity',
            'Tree Rings_WidthPages',
            'Tree Rings_WidthPages2',
            'Tree Rings_WidthBreit',
            'Tree Rings_Isotopes',
            'Tree Rings_Temperature',
            'Corals and Sclerosponges_d18O',
            'Corals and Sclerosponges_SrCa',
            'Corals and Sclerosponges_Rates',
            'Corals and Sclerosponges_Composite',
            'Corals and Sclerosponges_Temperature',
            'Ice Cores_d18O',
            'Ice Cores_dD',
            'Ice Cores_Accumulation',
            'Ice Cores_MeltFeature',
            'Lake Cores_Varve',
            'Lake Cores_BioMarkers',
            'Lake Cores_GeoChem',
            'Lake Cores_Misc',
            'Lake Cores_Temperature',
            'Marine Cores_d18O',
            'Marine Cores_Temperature',
            'Speleothems_d18O',
            'bivalve_d18O',
            'borehole_Temperature',
            'documents_Temperature',
            'hybrid_Temperature',
            ]

        proxy_assim2 = {
            'Corals and Sclerosponges_d18O' : ['d18O','delta18O','d18o','d18O_stk','d18O_int','d18O_norm',
                                               'd18o_avg','d18o_ave','dO18','d18O_4'],
            'Corals and Sclerosponges_d14C' : ['d14C','d14c','ac_d14c'],
            'Corals and Sclerosponges_d13C' : ['d13C','d13c','d13c_ave','d13c_ann_ave','d13C_int'],
            'Corals and Sclerosponges_SrCa' : ['Sr/Ca','Sr_Ca','Sr/Ca_norm','Sr/Ca_anom','Sr/Ca_int'],
            'Corals and Sclerosponges_Sr'   : ['Sr'],
            'Corals and Sclerosponges_BaCa' : ['Ba/Ca'],
            'Corals and Sclerosponges_CdCa' : ['Cd/Ca'],
            'Corals and Sclerosponges_MgCa' : ['Mg/Ca'],
            'Corals and Sclerosponges_UCa'  : ['U/Ca','U/Ca_anom'],
            'Corals and Sclerosponges_Pb'   : ['Pb'],
            'Corals and Sclerosponges_Rates': ['ext','calc','calcification'],
            'Corals and Sclerosponges_Composite': ['composite'],
            'Corals and Sclerosponges_Temperature': ['temperature'],
            'Ice Cores_d18O'                : ['d18O','delta18O','delta18o','d18o','dO18',
                                               'd18o_int','d18O_int',
                                               'd18O_norm','d18o_norm',
                                               'd18O_anom'],
            'Ice Cores_dD'                  : ['deltaD','delD','dD'],
            'Ice Cores_Accumulation'        : ['accum','accumu'],
            'Ice Cores_MeltFeature'         : ['MFP','melt'],
            'Lake Cores_Varve'              : ['varve', 'varve_thickness', 'varve thickness','thickness'],
            'Lake Cores_BioMarkers'         : ['Uk37', 'TEX86'], 
            'Lake Cores_GeoChem'            : ['Sr/Ca', 'Mg/Ca', 'Cl_cont'],
            'Lake Cores_Misc'               : ['RABD660_670','X_radiograph_dark_layer','massacum'],
            'Lake Cores_Temperature'        : ['temperature'],
            'Marine Cores_d18O'             : ['d18O'],
            'Marine Cores_Temperature'      : ['temperature'],
            'Speleothems_d18O'              : ['d18O'],
            'Speleothems_d13C'              : ['d13C'],
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
            'Tree Rings_Temperature'        : ['temperature'],
            'bivalve_d18O'                  : ['d18O'],
            'borehole_Temperature'          : ['temperature'],
            'documents_Temperature'         : ['temperature'],
            'hybrid_Temperature'            : ['temperature'],
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
            self.database_filter = list(self.database_filter)
            self.proxy_blacklist = list(self.proxy_blacklist)

            self.proxy_psm_type = {}
            for p in self.proxy_order: self.proxy_psm_type[p] = v_core.psm_type

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
    avgPeriod: str
        PSM calibrated on annual or seasonal data: allowed tags are 'annual' or 'season'
    """

    ##** BEGIN User Parameters **##

    # PSM calibrated on annual or seasonal data: allowed tags are 'annual' or 'season'
    avgPeriod = 'annual'
    #avgPeriod = 'season'
    
    ##** END User Parameters **##

    
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
        datafile_calib = 'gistemp1200_ERSST.nc'
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
        #datatag_calib_P = 'DaiPDSI'
        #datafile_calib_P = 'Dai_pdsi.mon.mean.selfcalibrated_185001-201412.nc'

        
        pre_calib_datafile = None

        psm_r_crit = 0.0

        ##** END User Parameters **##
        
        
        def __init__(self):
            self.datatag_calib = self.datatag_calib
            self.datafile_calib = self.datafile_calib
            self.psm_r_crit = self.psm_r_crit
            self.avgPeriod = v_psm.avgPeriod
            
            if '-'.join(v_proxies.use_from) == 'pages' and self.avgPeriod == 'season':
                print 'ERROR: Trying to use seasonality information with the PAGES1 proxy records.'
                print '       No seasonality metadata provided in that dataset. Exiting!'
                print '       Change avgPeriod to "annual" in your configuration.'
                raise SystemExit()
            
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

        ##** END User Parameters **##
        
        def __init__(self):
            self.datatag_calib_T = self.datatag_calib_T
            self.datafile_calib_T = self.datafile_calib_T
            self.datatag_calib_P = self.datatag_calib_P
            self.datafile_calib_P = self.datafile_calib_P
            self.psm_r_crit = self.psm_r_crit
            self.avgPeriod = v_psm.avgPeriod

            if '-'.join(v_proxies.use_from) == 'pages' and self.avgPeriod == 'season':
                print 'ERROR: Trying to use seasonality information with the PAGES1 proxy records.'
                print '       No seasonality metadata provided in that dataset. Exiting!'
                print '       Change avgPeriod to "annual" in your configuration.'
                raise SystemExit()
            
            if self.datadir_calib is None:
                self.datadir_calib = join(v_core.lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = self.datadir_calib


            if self.pre_calib_datafile is None:
                if '-'.join(v_proxies.use_from) == 'NCDC':
                    dbversion = v_proxies._ncdc.dbversion
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

    print 'Proxies             :', proxy_database
    print 'PSM type            :', psm_type
    print 'Calib. period       :', Cfg.core.calib_period

    if proxy_database == 'pages':
        print 'Proxy data location :', Cfg.proxies.pages.datadir_proxy
    elif proxy_database == 'NCDC':
        print 'Proxy data location :', Cfg.proxies.ncdc.datadir_proxy
    else:
        print 'ERROR in specification of proxy database. Exiting!'
        raise SystemExit()
        
    # psm type
    if psm_type == 'linear':
        print 'Calibration source  :', Cfg.psm.linear.datatag_calib
        datatag_calib = Cfg.psm.linear.datatag_calib
        psm_file = Cfg.psm.linear.pre_calib_datafile
        calib_avgPeriod = Cfg.psm.linear.avgPeriod
    elif psm_type == 'bilinear':
        print 'Calibration sources :', Cfg.psm.bilinear.datatag_calib_T, '+', Cfg.psm.bilinear.datatag_calib_P
        datatag_calib_T = Cfg.psm.bilinear.datatag_calib_T
        datatag_calib_P = Cfg.psm.bilinear.datatag_calib_P
        psm_file = Cfg.psm.bilinear.pre_calib_datafile
        calib_avgPeriod = Cfg.psm.bilinear.avgPeriod
    else:
        print 'ERROR: problem with the type of psm!'
        exit(1)
    
    print 'PSM calibration/parameters file:', psm_file
    
    # Check if psm_file already exists, archive it with current date/time if it exists
    # and replace by new file
    if os.path.isfile(psm_file):        
        nowstr = datetime.datetime.now().strftime("%Y%m%d:%H%M")
        os.system('mv %s %s_%s.pckl' %(psm_file,psm_file.rstrip('.pckl'),nowstr) )

    prox_manager = LMR_proxy_pandas_rework.ProxyManager(Cfg, Cfg.core.calib_period)
    type_site_calib = prox_manager.assim_ids_by_group

    print '--------------------------------------------------------------------'
    print 'Calibrated proxies : counts per proxy type:'
    # count the total number of proxies
    total_proxy_count = len(prox_manager.ind_assim)
    for pkey, plist in sorted(type_site_calib.iteritems()):
        print('%45s : %5d' % (pkey, len(plist)))
    print '--------------------------------------------------------------------'
    print('%45s : %5d' % ('TOTAL', total_proxy_count))
    print '--------------------------------------------------------------------'
    
    psm_dict = {}
    for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
        sitetag = (Y.type,Y.id)

        # Load proxy object in proxy dictionary
        # -------------------------------------
        # Site info
        psm_dict[sitetag] = {}
        psm_dict[sitetag]['lat']   = Y.psm_obj.lat
        psm_dict[sitetag]['lon']   = Y.psm_obj.lon
        psm_dict[sitetag]['elev']  = Y.psm_obj.elev        
        # PSM info
        if calib_avgPeriod == 'annual':
            # override proxy record metadata of seasonality w/ annual tag
            psm_dict[sitetag]['Seasonality'] = [1,2,3,4,5,6,7,8,9,10,11,12]
        else:
            psm_dict[sitetag]['Seasonality']  = Y.seasonality
        psm_dict[sitetag]['NbCalPts']     = Y.psm_obj.NbPts
        psm_dict[sitetag]['PSMintercept'] = Y.psm_obj.intercept
        psm_dict[sitetag]['PSMcorrel']    = Y.psm_obj.corr
        psm_dict[sitetag]['PSMmse']       = Y.psm_obj.R
        if psm_type == 'linear':
            psm_dict[sitetag]['calib']        = datatag_calib
            psm_dict[sitetag]['PSMslope']     = Y.psm_obj.slope
            psm_dict[sitetag]['PSMintercept'] = Y.psm_obj.intercept
        elif psm_type == 'bilinear':
            psm_dict[sitetag]['calib_temperature']    = datatag_calib_T
            psm_dict[sitetag]['calib_moisture']       = datatag_calib_P
            psm_dict[sitetag]['PSMslope_temperature'] = Y.psm_obj.slope_temperature
            psm_dict[sitetag]['PSMslope_moisture']    = Y.psm_obj.slope_moisture
        else:
            print 'ERROR: problem with the type of psm!'
            exit(1)

    # Dump dictionary to pickle file
    outfile = open('%s' % (psm_file),'w')
    cPickle.dump(psm_dict,outfile)
    cPickle.dump(psm_info,outfile)
    outfile.close()

    end_time = time() - begin_time
    print '========================================================='
    print 'PSM calibration completed in '+ str(end_time/60.0)+' mins'
    print '========================================================='

# =============================================================================

if __name__ == '__main__':
    main()
