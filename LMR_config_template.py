"""
Class based config module to help with passing information to LMR modules for
paleoclimate reconstruction experiments.

NOTE:  All general user parameters that should be edited are displayed
       between the following sections:

       ##** BEGIN User Parameters **##

       parameters, etc.

       ##** END User Parameters **##

Adapted from LMR_exp_NAMELIST by AndreP

Revisions:
 - Introduction of definitions related to use of newly developed NCDC proxy
   database.
   [ R. Tardif, Univ. of Washington, January 2016 ]
 - Added functionality restricting assimilated proxy records to those belonging
   to specific databases (e.g. PAGES1, PAGES2, LMR) (only for NCDC proxies).
   [ R. Tardif, Univ. of Washington, February 2016 ]
 - Introduction of "blacklist" to prevent the assimilation of specific proxy
   records as defined (through a python list) by the user. Applicable for both
   NCDC and Pages proxy sets.
   [ R. Tardif, Univ. of Washington, February 2016 ]
 - Added boolean allowing the user to indicate whether the prior is to be
   detrended or not.
   [ R. Tardif, Univ. of Washington, February 2016 ]
 - Added definitions associated with a new psm class (linear_TorP) allowing the
   use of temperature-calibrated OR precipitation-calibrated linear PSMs.
   [ R. Tardif, Univ. of Washington, March 2016 ]
 - Added definitions associated with a new psm class (h_interp) for use of
   isotope-enabled GCM data as prior: Ye values are taken as the prior isotope
   field either at the nearest grid pt. or as the weighted-average of values at
   grid points surrounding the isotope proxy site assimilated.
   [ R. Tardif, Univ. of Washington, June 2016 ]
 - Added definitions associated with a new psm class (bilinear) for bivariate
   linear regressions w/ temperature AND precipitation/PSDI as independent
   variables.
   [ R. Tardif, Univ. of Washington, June 2016 ]
 - Added initialization features to all configuration classes and sub_classes
   The new usage should now grab an instance of Config and use that object.
   This instance variable copies most values and generates some intermediate
   values used by the reconstruction process.  This helps the configuration
   stay consistent if one is altering values on the fly.
   [ A. Perkins, Univ. of Washington, June 2016 ]
 - Use of PSMs calibrated on the basis of a proxy record seasonality metadata
   can now be activated (see avgPeriod parameter in the "psm" class)
   [ R. Tardif, Univ. of Washington, July 2016 ]
 - PSM classes can now be specified per proxy type. 
   See proxy_psm_type dictionaries in the "proxies" class below.
   [ R. Tardif, Univ. of Washington, August 2016 ]
 - Addition of filters for selecting the set of proxy records available for
   assimilation based on data availability over reconstruction period.
   [ R. Tardif, Univ. of Washington, October 2016 ]
 - Added features associated with the use of low-resolution marine
   proxies (uk37 from marine cores) and production of reconstructions at 
   lower temporal resolutions (i.e. other than annual).
   [ R. Tardif, Univ. of Washington, Jan-Feb 2017 ]
 - Added flexibility to regridding capabilities: added option to by-pass 
   the regridding, added new option using simple distance-weighted averaging
   and replaced the hared-coded truncation resolution (T42) of spatial fields 
   by a user-specified value. This value applies to both the simple
   interpolation and the original spherical harmonic-based regridding.
   [ R. Tardif, Univ. of Washington, March 2017 ]
 - Added functionalities associated with the use of the simplified 
   config.yml configuration file. 
   [ A. Perkins & R. Tardif, Univ. of Washington, April 2017 ]
 - Added a boolean flag to activate/deactivate output to analysis_Ye.pckl file.
   [G. Hakim, Univ. of Washington, August 2017] 
 - Added parameter allowing a user to define the reference period w.r.t. which
   anomalies in climate variable are calculated. 
   [ R. Tardif, Univ. of Washington, March 2018 ]
 - Added option to regrid the reanalysis at the archiving stage.
   [ R. Tardif, Univ. of Washington, April 2018 ]
 - Clearer more flexible options to save ensemble information other than the mean
   (i.e. full ensemble, ensemble variance, percentiles or subset of members)
   [ R. Tardif, Univ. of Washington, April 2018 ]
 - Added feature of multiprocessing
   [ F. Zhu, Univ. of Southern California, July 2018 ]
"""

from os.path import join
from copy import deepcopy
import yaml
#import matlab.engine


# If true, uses only LMR_config.  No yaml loading
LEGACY_CONFIG = False

# Absolute path to LMR source code directory
#SRC_DIR = '/home/disk/ekman/rtardif/codes/LMR/pyLMR'
SRC_DIR = '/home/disk/ice4/hakim/gitwork/LMR'

# Control logging output. (0 = none; 1 = most important; 2 = many; 3 = a lot;
#   >=4 all)
LOG_LEVEL = 4

# Class for distinction of configuration classes
class ConfigGroup(object):

    def __init__(self, **kwargs):
        if kwargs:
            update_config_class_yaml(kwargs, self)


class _YamlStorage(object):
    """
    Generic object for loading in dictionaries from yaml files, 
    and a convenience function for returning loaded information.
    """

    def __init__(self, filename):
        print('Loading information from {}'.format(filename))
        try:
            f = open(join(SRC_DIR, filename), 'r')
            self.data = yaml.load(f)
        except IOError as e:
            raise SystemExit(('Could not load {} file when initializing the '
                              ' in LMR_config.py. File is required for locating'
                              ' dataset specific files. Please ensure '
                              'LMR_Config.SRC_DIR is correctly set. '
                              'Exiting...').format(filename))

    def get_info(self, tag):
        return dict(self.data[tag])


class _DatasetDescriptors(_YamlStorage):
    """
    Loads and stores the datasets.yml file and return dictionaries of file
    specifications for each dataset. Information used by psm, prior,
    and forecaster configuration classes.
    """

    def __init__(self):
        super(_DatasetDescriptors, self).__init__('datasets.yml')


class _GridDefinitions(_YamlStorage):
    """
    Loads and stores the grid_def.yml file and returns dictionaries of 
    information necessary to construct a given grid in ESMpy regridding.
    """

    def __init__(self):
        super(_GridDefinitions, self).__init__('grid_def.yml')

# Load dataset information on configuration import
_DataInfo = _DatasetDescriptors()
_GridDef = _GridDefinitions()


class constants:

    class file_types:
        netcdf = 'NCD'
        ascii = 'ASC'
        numpy = 'NPY'
        numpy_zip = 'NPZ'
        dataframe = 'DF'

class wrapper(ConfigGroup):
    """
    Parameters for reconstruction realization manager LMR_wrapper.

    Attributes
    ----------
    multi_seed: list(int), None
        List of RNG seeds to be used during a reconstruction for each
        realization.  This overrides the 'core.seed' parameter.
    iter_range: tuple(int)
        Range of Monte-Carlo iterations to perform
    param_search: dict{str: Iterable} or None
        Names of configuration parameters to iterate over when performing a
        reconstruction experiment

    Example
    -------
    The param_search dictionary should use the configuration object attribute
    syntax as the key to reference which parameters to iterate over. The
    value should be some iterable object of values to be covered in the
    parameter search::
        param_search = {'core.hybrid_a': (0.6, 0.7, 0.8),
                        'core.inf_factor': (1.1, 1.2, 1.3)}
    """

    ##** BEGIN User Parameters **##

    iter_range = (0, 0)
    param_search = None
    multi_seed = None

    ##** END User Parameters **##

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        if self.multi_seed is not None:
            self.multi_seed = list(self.multi_seed)
        self.iter_range = self.iter_range
        self.param_search = deepcopy(self.param_search)

class core(ConfigGroup):
    """
    High-level parameters of LMR_driver_callable.

    Notes
    -----
    curr_iter attribute is created during initialization

    Attributes
    ----------
    nexp: str
        Name of reconstruction experiment
    lmr_path: str
        Absolute path for the experiment
    online_reconstruction: bool
        Perform reconstruction with (True) or without (False) cycling
    clean_start: bool
        Delete existing files in output directory (otherwise they will be used
        as the prior!)
    use_precalc_ye: bool
        Use pre-existing files for the psm Ye values.  If the file does not
        exist and the required state variables are missing the reconstruction
        will quit.
    recon_period: tuple(int)
        Time period for reconstruction
    nens: int
        Ensemble size
    nthreads: int
        number of threads
    loc_rad: float
        Localization radius for DA (in km)
    inflation_fact : float
        Covariance inflation factor
    seed: int, None
        RNG seed.  Passed to all random function calls. (e.g. prior and proxy
        record sampling)  Overridden by wrapper.multi_seed.
    datadir_output: str
        Absolute path to working directory output for LMR
    archive_dir: str
        Absolute path to LMR reconstruction archive directory
    write_posterior_Ye: bool
        Flag to indicate whether the analysis_Ye.pckl is to be generated 
        or not (large file containing full information on the posterior 
        proxy estimates (assimilated proxy records).

    """

    ##** BEGIN User Parameters **##

    nexp = 'test'

    #lmr_path = '/home/disk/ice4/nobackup/hakim/lmr'
    #lmr_path = '/home/chaos2/wperkins/data/LMR'
    lmr_path = '/home/disk/kalman3/rtardif/LMR'

    online_reconstruction = False
    clean_start = True
    use_precalc_ye = True

    # TODO: More pythonic to make last time a non-inclusive edge
    #recon_period = (-20000, 2000)
    #recon_period = (-150000, 2000)
    recon_period = (1850, 2000)
    
    # time interval of reconstructed fields (in years). Note: should be of type integer
    recon_timescale = 1    # annual
    #recon_timescale = 100
    #recon_timescale = 250
    #recon_timescale = 500
    
    nens = 100
    nthreads = 8

    seed = None

    loc_rad = None

    inflation_fact = None

    # Reference period w.r.t. which anomalies are to be defined.
    anom_reference_period = (1951,1980)
    
    #datadir_output = '/home/disk/ice4/hakim/svnwork/lmr/trunk/data'
    #datadir_output = '/home/chaos2/wperkins/data/LMR/output/working'
    datadir_output  = '/home/disk/kalman3/rtardif/LMR/output/wrk'
    #datadir_output  = '/home/disk/kalman3/rtardif/LMR/output/wrk'
    
    #archive_dir = '/home/disk/kalman3/hakim/LMR/'
    #archive_dir = '/home/chaos2/wperkins/data/LMR/output/testing'
    archive_dir = '/home/disk/kalman3/rtardif/LMR/output'
    #archive_dir = '/home/disk/ekman4/rtardif/LMR/output'

    # Whether or not to produce the analysis_Ye.pckl file
    write_posterior_Ye = False

    # Ensemble archiving options: ens_full, ens_variance, ens_percentiles, ens_subsample
    save_archive = 'ens_variance'
    # if save_archive = 'ens_percentiles', which percentiles to caclulate and save
    save_archive_percentiles = (5,95)
    # if save_archive = 'ens_subsample', number of members to archive
    save_archive_ens_subsample = 10
    
    # Possibly regrid the generated reanalysis fields to archive files.
    # Options: None (no regridding) or 'esmpy'
    archive_regrid_method = None
    archive_esmpy_interp_method = 'bilinear'
    archive_esmpy_regrid_to = 't42'

    
    ##** END User Parameters **##

    def __init__(self, curr_iter=None, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        # some checks
        if type(self.recon_timescale) != 'int': self.recon_timescale = int(self.recon_timescale)
        
        self.nexp = self.nexp
        self.lmr_path = self.lmr_path
        self.online_reconstruction = self.online_reconstruction
        self.clean_start = self.clean_start
        self.use_precalc_ye = self.use_precalc_ye
        self.recon_period = self.recon_period
        self.nens = self.nens
        self.loc_rad = self.loc_rad
        self.inflation_fact = self.inflation_fact
        self.seed = self.seed
        self.datadir_output = self.datadir_output
        self.archive_dir = self.archive_dir
        self.write_posterior_Ye = self.write_posterior_Ye
        self.anom_reference_period = self.anom_reference_period

        
        self.save_archive = self.save_archive
        self.save_archive_percentiles = self.save_archive_percentiles
        self.save_archive_ens_subsample = self.save_archive_ens_subsample

        
        if self.archive_regrid_method is not None:
            if self.archive_regrid_method == 'esmpy':
                self.archive_esmpy_interp_method = self.archive_esmpy_interp_method
                self.archive_esmpy_grid_def = _GridDef.get_info(self.archive_esmpy_regrid_to)
            else:
                raise ValueError('Unrecognized option for regridding to archive files!'
                                 ' Only None or esmpy are allowed.')

        if curr_iter is None:
            self.curr_iter = wrapper.iter_range[0]
        else:
            self.curr_iter = curr_iter

class proxies(ConfigGroup):
    """
    Parameters for proxy data

    Attributes
    ----------
    use_from: list(str)
        A list of keys for proxy classes to load from.  Keys available are
        stored in LMR_proxy_pandas_rework.
    proxy_frac: float
        Fraction of available proxy data (sites) to assimilate
    proxy_timeseries_kind: string
        Type of proxy timeseries to use. 'anom' for animalies or 'asis'
        to keep records as included in the database. 
    proxy_availability_filter: boolean
        True/False flag indicating whether filtering of proxy records
        according to data availability over reconstruction period is
        to be performed. If True, only proxies with data covering the
        reconstruction period are retained for assimilation. 
        Condition on record completeness is controlled with the next 
        config. parameter (see just below).
    proxy_availability_fraction: float
        Minimum threshold on the fraction of available proxy annual data 
        over the reconstruction period. i.e. control on the fraction of 
        available data that a recors must have in order to be assimilated. 
    """

    ##** BEGIN User Parameters **##

    # =============================
    # Which proxy database to use ?
    # =============================
    use_from = ['PAGES2kv1']
    #use_from = ['LMRdb']
    #use_from = ['NCDCdtda']

    proxy_frac = 1.0
    #proxy_frac = 0.75

    # type of proxy timeseries to return: 'anom' for anomalies
    # (temporal mean removed) or asis' to keep unchanged
    proxy_timeseries_kind = 'asis'

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
    proxy_availability_fraction = 1.0
    
    ##** END User Parameters **##

    # -----------------
    # PAGES2kv1 proxies
    # -----------------
    class PAGES2kv1(ConfigGroup):
        """
        Parameters for PAGES2kv1Proxy class

        Notes
        -----
        proxy_type_mappings and simple_filters are creating during instance
        creation.

        Attributes
        ----------
        datadir_proxy: str
            Absolute path to proxy data *or* None if using default lmr_path
        datafile_proxy: str
            proxy records filename
        metafile_proxy: str
            proxy metadata filename
        dataformat_proxy: str
            File format of the proxy data files
        regions: list(str)
            List of proxy data regions (data keys) to use.
        proxy_resolution: list(float)
            List of proxy time resolutions to use
        proxy_order: list(str):
            Proxy types to be assimilated and 
            order of assimilation.
        proxy_psm_type: dict{str:str}
            Association between proxy type and psm type.
        proxy_assim2: dict{ str: list(str)}
            Maps proxy type and measurement to our proxy type keys.
            Uses dictionary with structure {<<proxy type>>: [.. list of measurement
            tags ..] where "proxy type" is written as
            "<<archive type>>_<<measurement type>>"
        simple_filters: dict{'str': Iterable}
            List mapping Pages2k metadata sheet columns to a list of values
            to filter by.
        proxy_blacklist: list(str)
            A list of proxy ids to prevent from being used in the reconstruction
        proxy_type_mapping: dict{(str,str): str}
            Maps proxy type and measurement to our proxy type keys.
            (e.g. {('Tree ring', 'TRW'): 'Tree ring_Width'} )
        """

        ##** BEGIN User Parameters **##

        datadir_proxy = None
        datafile_proxy = 'Pages2kv1_Proxies.df.pckl'
        metafile_proxy = 'Pages2kv1_Metadata.df.pckl'
        dataformat_proxy = 'DF'

        regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
                   'North America', 'South America']

        proxy_resolution = [1.0]

        # DO NOT CHANGE *FORMAT* BELOW

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

        # A blacklist on proxy records, to prevent assimilation of chronologies
        # known to be duplicates
        proxy_blacklist = []

        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(self.__class__, self).__init__(**kwargs)
            if lmr_path is None:
                lmr_path = core.lmr_path

            if self.datadir_proxy is None:
                self.datadir_proxy = join(lmr_path, 'data', 'proxies')
            else:
                self.datadir_proxy = self.datadir_proxy

            self.datafile_proxy = join(self.datadir_proxy,
                                       self.datafile_proxy)
            self.metafile_proxy = join(self.datadir_proxy,
                                       self.metafile_proxy)

            self.dataformat_proxy = self.dataformat_proxy
            self.regions = list(self.regions)
            self.proxy_resolution = list(self.proxy_resolution)
            self.proxy_timeseries_kind = proxies.proxy_timeseries_kind
            self.proxy_order = list(self.proxy_order)
            self.proxy_psm_type = deepcopy(self.proxy_psm_type)
            self.proxy_assim2 = deepcopy(self.proxy_assim2)
            self.proxy_blacklist = list(self.proxy_blacklist)
            self.proxy_availability_filter = proxies.proxy_availability_filter
            self.proxy_availability_fraction = proxies.proxy_availability_fraction
            
            # Create mapping for Proxy Type/Measurement Type to type names above
            self.proxy_type_mapping = {}
            for ptype, measurements in self.proxy_assim2.items():
                # Fetch proxy type name that occurs before underscore
                type_name = ptype.split('_', 1)[0]
                for measure in measurements:
                    self.proxy_type_mapping[(type_name, measure)] = ptype

            self.simple_filters = {'PAGES 2k Region': self.regions,
                                   'Resolution (yr)': self.proxy_resolution}

    # -------------
    # LMRdb proxies
    # -------------
    class LMRdb(ConfigGroup):
        """
        Parameters for LMRdb proxy class

        Notes
        -----
        proxy_type_mappings and simple_filters are creating during instance
        creation.

        Attributes
        ----------
        datadir_proxy: str
            Absolute path to proxy data *or* None if using default lmr_path
        datafile_proxy: str
            proxy records filename
        metafile_proxy: str
            proxy metadata filename
        dataformat_proxy: str
            File format of the proxy data
        regions: list(str)
            List of proxy data regions (data keys) to use.
        proxy_resolution: list(float)
            List of proxy time resolutions to use
        database_filter: list(str)
            List of databases from which to limit the selection of proxies.
            Use [] (empty list) if no restriction, or ['db_name1', db_name2'] to
            limit to proxies contained in "db_name1" OR "db_name2".
            Possible choices are: 'PAGES1', 'PAGES2', 'LMR_FM'
        proxy_order: list(str):
            Order of assimilation by proxy type key
        proxy_assim2: dict{ str: list(str)}
            Proxy types to be assimilated.
            Uses dictionary with structure {<<proxy type>>: [.. list of
            measuremant tags ..] where "proxy type" is written as
            "<<archive type>>_<<measurement type>>"
        proxy_type_mapping: dict{(str,str): str}
            Maps proxy type and measurement to our proxy type keys.
            (e.g. {('Tree ring', 'TRW'): 'Tree ring_Width'} )
        proxy_psm_type: dict{str:str}
            Association between proxy type and psm type.
        simple_filters: dict{'str': Iterable}
            List mapping proxy metadata sheet columns to a list of values
            to filter by.
        """

        ##** BEGIN User Parameters **##

        dbversion = 'v0.4.0'
        
        datadir_proxy = None
        datafile_proxy = 'LMRdb_{}_Proxies.df.pckl'
        metafile_proxy = 'LMRdb_{}_Metadata.df.pckl'
        dataformat_proxy = 'DF'

        # This is not activated with LMRdb data yet...
        regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
                   'North America', 'South America']

        proxy_resolution = [1.0]

        # Limit proxies to those included in the following list of databases
        # Note: Empty list = no restriction
        #       If list has more than one element, only records contained in ALL
        #       databases listed will be retained. Possibilities are:
        #       database_filter = ['PAGES2']                   # for db v0.1.0
        #       database_filter = ['LMR','Breits','PAGES2']    # for db v0.1.0
        #       database_filter = ['LMR','Breits','PAGES2kv2'] # for db v0.2.0
        database_filter = []
        
        # DO NOT CHANGE *FORMAT* BELOW
        proxy_order = [
#old            'Tree Rings_WidthPages',
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
            'Bivalve_d18O',
            'Speleothems_d18O',
            ]

        # Assignment of psm type per proxy type
        # Choices are: 'linear', 'linear_TorP', 'bilinear', 'h_interp'
        #  The linear PSM can be used on *all* proxies.
        #  The linear_TorP and bilinear w.r.t. temperature or/and moisture
        #  PSMs are aimed at *tree ring* proxies in particular
        #  The h_interp forward model is to be used for isotope proxies when
        #  the prior is taken from an isotope-enabled GCM output. 
        proxy_psm_type = {
            'Bivalve_d18O'                  : 'linear',
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
#old            'Tree Rings_WidthPages'         : 'linear',
            'Tree Rings_WoodDensity'        : 'linear',
            'Tree Rings_Isotopes'           : 'linear',
            'Speleothems_d18O'              : 'linear',
        }
         
        proxy_assim2 = {
            'Bivalve_d18O'                  : ['d18O'],
            'Corals and Sclerosponges_d18O' : ['d18O', 'delta18O', 'd18o',
                                               'd18O_stk', 'd18O_int',
                                               'd18O_norm', 'd18o_avg',
                                               'd18o_ave', 'dO18',
                                               'd18O_4'],
            'Corals and Sclerosponges_SrCa' : ['Sr/Ca', 'Sr_Ca', 'Sr/Ca_norm',
                                               'Sr/Ca_anom', 'Sr/Ca_int'],
            'Corals and Sclerosponges_Rates': ['ext','calc','calcification','calcification rate',
                                               'composite'],
            'Ice Cores_d18O'                : ['d18O', 'delta18O', 'delta18o',
                                               'd18o', 'd18o_int', 'd18O_int',
                                               'd18O_norm', 'd18o_norm', 'dO18',
                                               'd18O_anom'],
            'Ice Cores_dD'                  : ['deltaD', 'delD', 'dD'],
            'Ice Cores_Accumulation'        : ['accum', 'accumu'],
            'Ice Cores_MeltFeature'         : ['MFP','melt'],
            'Lake Cores_Varve'              : ['varve', 'varve_thickness',
                                               'varve thickness', 'thickness'],
            'Lake Cores_BioMarkers'         : ['Uk37', 'TEX86','tex86'],
            'Lake Cores_GeoChem'            : ['Sr/Ca', 'Mg/Ca', 'Cl_cont'],
            'Lake Cores_Misc'               : ['RABD660_670','X_radiograph_dark_layer','massacum'],
            'Marine Cores_d18O'             : ['d18O'],
            'Tree Rings_WidthBreit'         : ['trsgi_breit'],
            'Tree Rings_WidthPages2'        : ['trsgi'],
#old            'Tree Rings_WidthPages'         : ['TRW',
#old                                              'ERW',
#old                                              'LRW'],
            'Tree Rings_WoodDensity'        : ['max_d',
                                               'min_d',
                                               'early_d',
                                               'earl_d',
                                               'density',
                                               'late_d',
                                               'MXD'],
            'Tree Rings_Isotopes'           : ['d18O'],
            'Speleothems_d18O'              : ['d18O'],
        }

        # A blacklist on proxy records, to prevent assimilation of specific
        # chronologies known to be duplicates or to have errors.
        #proxy_blacklist = ['00aust01a', '06cook02a', '06cook03a', '08vene01a',
        #                   '09japa01a', '10guad01a', '99aust01a', '99fpol01a',
        #                   '72Devo01',  '72Devo05'] # for db v0.1.0
        proxy_blacklist = []
        
        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(self.__class__, self).__init__(**kwargs)
            if lmr_path is None:
                lmr_path = core.lmr_path

            if self.datadir_proxy is None:
                self.datadir_proxy = join(lmr_path, 'data', 'proxies')
            else:
                self.datadir_proxy = self.datadir_proxy

            self.datafile_proxy = self.datafile_proxy.format(self.dbversion)
            self.metafile_proxy = self.metafile_proxy.format(self.dbversion)
                
            self.datafile_proxy = join(self.datadir_proxy,
                                       self.datafile_proxy)
            self.metafile_proxy = join(self.datadir_proxy,
                                       self.metafile_proxy)

            self.dataformat_proxy = self.dataformat_proxy
            self.regions = list(self.regions)
            self.proxy_resolution = list(self.proxy_resolution)
            self.proxy_timeseries_kind = proxies.proxy_timeseries_kind
            self.proxy_order = list(self.proxy_order)
            self.proxy_psm_type = deepcopy(self.proxy_psm_type)
            self.proxy_assim2 = deepcopy(self.proxy_assim2)
            self.database_filter = list(self.database_filter)
            self.proxy_blacklist = list(self.proxy_blacklist)
            self.proxy_availability_filter = proxies.proxy_availability_filter
            self.proxy_availability_fraction = proxies.proxy_availability_fraction
            
            self.proxy_type_mapping = {}
            for ptype, measurements in self.proxy_assim2.items():
                # Fetch proxy type name that occurs before underscore
                type_name = ptype.split('_', 1)[0]
                for measure in measurements:
                    self.proxy_type_mapping[(type_name, measure)] = ptype

            self.simple_filters = {'Resolution (yr)': self.proxy_resolution}

            

    # --------------------------------------------------------
    # proxies specific to Deep Times Data Assimilation project
    # --------------------------------------------------------
    class NCDCdtda(ConfigGroup):
        """
        Parameters for NCDCdtda proxy class

        Notes
        -----
        proxy_type_mappings and simple_filters are creating during instance
        creation.

        Attributes
        ----------
        datadir_proxy: str
            Absolute path to proxy data *or* None if using default lmr_path
        datafile_proxy: str
            proxy records filename
        metafile_proxy: str
            proxy metadata filename
        dataformat_proxy: str
            File format of the proxy data
        regions: list(str)
            List of proxy data regions (data keys) to use.
        proxy_resolution: list(float or tuple)
            List of proxy time resolutions to use. 
            If tuple, indicates a *range* of resolutions. 
        database_filter: list(str)
            List of databases from which to limit the selection of proxies.
            Use [] (empty list) if no restriction, or ['db_name1', db_name2'] to
            limit to proxies contained in "db_name1" OR "db_name2".
            Possible choices are: 'PAGES1', 'PAGES2', 'LMR'
        proxy_order: list(str):
            Order of assimilation by proxy type key
        proxy_assim2: dict{ str: list(str)}
            Proxy types to be assimilated.
            Uses dictionary with structure {<<proxy type>>: [.. list of
            measuremant tags ..] where "proxy type" is written as
            "<<archive type>>_<<measurement type>>"
        proxy_type_mapping: dict{(str,str): str}
            Maps proxy type and measurement to our proxy type keys.
            (e.g. {('Tree ring', 'TRW'): 'Tree ring_Width'} )
        proxy_psm_type: dict{str:str}
            Association between proxy type and psm type.
        simple_filters: dict{'str': Iterable}
            List mapping proxy metadata sheet columns to a list of values
            to filter by.
        """

        ##** BEGIN User Parameters **##

        #dbversion = 'v0.0.0'
        dbversion = 'v0.0.1'
        
        datadir_proxy = None
        datafile_proxy = 'DTDA_{}_Proxies.df.pckl'
        metafile_proxy = 'DTDA_{}_Metadata.df.pckl'
        dataformat_proxy = 'DF'

        # This is not activated with yet...
        regions = []

        # Restrict uploaded proxy records to those within specified
        # range of temporal resolutions. 
        proxy_resolution = [(0.,5000.)]

        # Limit proxies to those included in the following list of databases
        # Note: Empty list = no restriction
        #       If list has more than one element, only records contained in ALL
        #       databases listed will be retained.
        database_filter = []

        # DO NOT CHANGE *FORMAT* BELOW

        proxy_order = [
        #    'Marine Cores_uk37',
            'Marine sediments_uk37',
            ]

        # Assignment of psm type per proxy type
        # Choices are: 'linear', 'h_interp', 'bayesreg_uk37', 'bayesreg_tex86'
        #  The linear PSM can be used on *all* proxies.
        #  The h_interp forward model is to be used for isotope proxies when
        #  the prior is taken from an isotope-enabled GCM output. 
        proxy_psm_type = {
            'Marine Cores_uk37'             : 'bayesreg_uk37',
            'Marine sediments_uk37'         : 'bayesreg_uk37',
        }
        
        proxy_assim2 = {
            'Marine Cores_uk37'             : ['uk37', 'UK37'],
            'Marine sediments_uk37'         : ['UK37'],
        }

        # A blacklist on proxy records, to prevent assimilation of specific
        # chronologies known to be duplicates.
        proxy_blacklist = []
        
        
        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(self.__class__, self).__init__(**kwargs)
            if lmr_path is None:
                lmr_path = core.lmr_path

            if self.datadir_proxy is None:
                self.datadir_proxy = join(lmr_path, 'data', 'proxies')
            else:
                self.datadir_proxy = self.datadir_proxy

            self.datafile_proxy = self.datafile_proxy.format(self.dbversion)
            self.metafile_proxy = self.metafile_proxy.format(self.dbversion)
            
            self.datafile_proxy = join(self.datadir_proxy,
                                       self.datafile_proxy)
            self.metafile_proxy = join(self.datadir_proxy,
                                       self.metafile_proxy)

            self.dataformat_proxy = self.dataformat_proxy
            self.regions = list(self.regions)
            self.proxy_resolution = list(self.proxy_resolution)
            self.proxy_timeseries_kind = proxies.proxy_timeseries_kind
            self.proxy_order = list(self.proxy_order)
            self.proxy_psm_type = deepcopy(self.proxy_psm_type)
            self.proxy_assim2 = deepcopy(self.proxy_assim2)
            self.database_filter = list(self.database_filter)
            self.proxy_blacklist = list(self.proxy_blacklist)
            self.proxy_availability_filter = proxies.proxy_availability_filter
            self.proxy_availability_fraction = proxies.proxy_availability_fraction
            
            self.proxy_type_mapping = {}
            for ptype, measurements in self.proxy_assim2.items():
                # Fetch proxy type name that occurs before underscore
                type_name = ptype.split('_', 1)[0]
                for measure in measurements:
                    self.proxy_type_mapping[(type_name, measure)] = ptype

            self.simple_filters = {'Resolution (yr)': self.proxy_resolution}

    
    # Initialize subclasses with all attributes
    def __init__(self, lmr_path=None, seed=None, **kwargs):
        self.PAGES2kv1 = self.PAGES2kv1(lmr_path=lmr_path, **kwargs.pop('PAGES2kv1', {}))
        self.LMRdb = self.LMRdb(lmr_path=lmr_path, **kwargs.pop('LMRdb', {}))
        self.NCDCdtda = self.NCDCdtda(lmr_path=lmr_path, **kwargs.pop('NCDCdtda', {}))

        super(self.__class__, self).__init__(**kwargs)
        
        self.use_from = list(self.use_from)
        self.proxy_frac = self.proxy_frac
        if seed is None:
            seed = core.seed
        self.seed = seed



class psm(ConfigGroup):
    """
    Parameters for PSM classes

    Attributes
    ----------
    avgPeriod: str
        Indicates use of PSMs calibrated on annual or seasonal data: 
        allowed tags are 'annual' or 'season'
    """

    ##** BEGIN User Parameters **##

    # Period over which data is used to establish a statistical relationship
    # between proxy and instrumental data (statistical PSMs only)
    calib_period = (1850,2015)
    
    # Averaging period for the PSM: 'annual' or 'season'
    avgPeriod = 'annual'
    #avgPeriod = 'season'

    # If avgPeriod = 'season', use seasonality from proxy metadata or objectively derived on the basis of psm calibration?
    season_source = 'proxy_metadata'
    #season_source = 'psm_calib'
    
    # Mapping of calibration sources w/ climate variable
    # To be modified only if a new calibration source is added. 
    all_calib_sources = {'temperature': ['GISTEMP', 'MLOST', 'NOAAGlobalTemp', 'HadCRUT', 'BerkeleyEarth'], 'moisture': ['GPCC','DaiPDSI']}
    
    ##** END User Parameters **##

    
    class linear(ConfigGroup):
        """
        Parameters for the linear fit PSM.

        Attributes
        ----------
        datatag_calib: str
            Source key of calibration data for PSM
        datadir_calib: str
            Absolute path to calibration data *or* None if using default
            lmr_path
        datafile_calib: str
            Filename for calibration data
        dataformat_calib: str
            Data storage type for calibration data
        pre_calib_datafile: str
            Absolute path to precalibrated Linear PSM data *or* None if using
            default LMR path
        varname_calib: str
            Variable name to use from the calibration dataset
        psm_r_crit: float
            Usage threshold for correlation of linear PSM
        """

        ##** BEGIN User Parameters **##

        datatag_calib = 'GISTEMP'

        pre_calib_datafile = None

        psm_r_crit = 0.0


        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            self.datatag_calib = self.datatag_calib

            dataset_descr = _DataInfo.get_info(self.datatag_calib)
            self.datainfo_calib = dataset_descr['info']
            self.datadir_calib = dataset_descr['datadir']
            self.datafile_calib = dataset_descr['datafile']
            self.dataformat_calib = dataset_descr['dataformat']

            self.psm_r_crit = self.psm_r_crit

            if '-'.join(proxies.use_from) == 'PAGES2kv1' and 'season' in psm.avgPeriod:
                print('ERROR: Trying to use seasonality information with the PAGES2kv1 proxy records.')
                print('       No seasonality metadata provided in that dataset. Exiting!')
                print('       Change avgPeriod to "annual" in your configuration.')
                raise SystemExit()

            if lmr_path is None:
                lmr_path = core.lmr_path

            try:
                if psm.avgPeriod == 'annual':
                    self.avgPeriod = psm.avgPeriod
                elif psm.avgPeriod == 'season' and psm.season_source:
                    if psm.season_source == 'proxy_metadata':
                        self.avgPeriod = psm.avgPeriod+'META'
                    elif psm.season_source == 'psm_calib':
                        self.avgPeriod = psm.avgPeriod+'PSM'
                    else:
                        print('ERROR: unrecognized psm.season_source attribute!')
                        raise SystemExit()
                else:
                    self.avgPeriod = psm.avgPeriod
            except:
                self.avgPeriod = psm.avgPeriod


            if self.datadir_calib is None:
                self.datadir_calib = join(lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = self.datadir_calib

            if self.pre_calib_datafile is None:
                if '-'.join(proxies.use_from) == 'LMRdb':
                    dbversion = proxies.LMRdb.dbversion
                    filename = ('PSMs_' + '-'.join(proxies.use_from) +
                                '_' + dbversion +
                                '_' + self.avgPeriod +
                                '_' + self.datatag_calib +
                                '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                '.pckl')
                else:
                    filename = ('PSMs_' + '-'.join(proxies.use_from) +
                                '_' + self.datatag_calib +
                                '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                '.pckl')
                self.pre_calib_datafile = join(lmr_path,
                                               'PSM',
                                               filename)
            else:
                self.pre_calib_datafile = self.pre_calib_datafile

            # association of calibration source and state variable needed to calculate Ye's
            if self.datatag_calib in psm.all_calib_sources['temperature']:
                self.psm_required_variables = {'tas_sfc_Amon': 'anom'}

            elif self.datatag_calib in psm.all_calib_sources['moisture']:
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

                
    class linear_TorP(ConfigGroup):                
        """
        Parameters for the linear fit PSM, calibrated against 
        temperature OR moisture.

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

        ##** BEGIN User Parameters **##

        
        # linear PSM w.r.t. temperature
        # -----------------------------
        # Choice between:
        # ---------------
        datatag_calib_T = 'GISTEMP'
        
        # linear PSM w.r.t. precipitation/moisture
        # ----------------------------------------
        datatag_calib_P = 'GPCC'

        pre_calib_datafile_T = None
        pre_calib_datafile_P = None

        psm_r_crit = 0.0

        
        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            self.datatag_calib_T = self.datatag_calib_T
            dataset_descr_T = _DataInfo.get_info(self.datatag_calib_T)
            self.datainfo_calib_T = dataset_descr_T['info']
            self.datadir_calib_T = dataset_descr_T['datadir']
            self.datafile_calib_T = dataset_descr_T['datafile']
            self.dataformat_calib_T = dataset_descr_T['dataformat']

            
            self.datatag_calib_P = self.datatag_calib_P
            dataset_descr_P = _DataInfo.get_info(self.datatag_calib_P)
            self.datainfo_calib_P = dataset_descr_P['info']
            self.datadir_calib_P = dataset_descr_P['datadir']
            self.datafile_calib_P = dataset_descr_P['datafile']
            self.dataformat_calib_P = dataset_descr_P['dataformat']

            self.psm_r_crit = self.psm_r_crit
            
            if '-'.join(proxies.use_from) == 'PAGES2kv1' and 'season' in psm.avgPeriod:
                print('ERROR: Trying to use seasonality information with the PAGES2kv1 proxy records.')
                print('       No seasonality metadata provided in that dataset. Exiting!')
                print('       Change avgPeriod to "annual" in your configuration.')
                raise SystemExit()

            try:
                if psm.avgPeriod == 'annual':
                    self.avgPeriod = psm.avgPeriod
                elif psm.avgPeriod == 'season' and psm.season_source:
                    if psm.season_source == 'proxy_metadata':
                        self.avgPeriod = psm.avgPeriod+'META'
                    elif psm.season_source == 'psm_calib':
                        self.avgPeriod = psm.avgPeriod+'PSM'
                    else:
                        print('ERROR: unrecognized psm.season_source attribute!')
                        raise SystemExit()
                else:
                    self.avgPeriod = psm.avgPeriod
            except:
                self.avgPeriod = psm.avgPeriod


            if lmr_path is None:
                lmr_path = core.lmr_path

            if self.datadir_calib_T is None:
                self.datadir_calib_T = join(lmr_path, 'data', 'analyses')
            if self.datadir_calib_P is None:
                self.datadir_calib_P = join(lmr_path, 'data', 'analyses')
                
            if self.pre_calib_datafile_T is None:
                if '-'.join(proxies.use_from) == 'LMRdb':
                    dbversion = proxies.LMRdb.dbversion
                    filename_t = ('PSMs_' + '-'.join(proxies.use_from) +
                                  '_' + dbversion +
                                  '_' + self.avgPeriod +
                                  '_' + self.datatag_calib_T +
                                  '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                  '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                  '.pckl')
                else:
                    filename_t = ('PSMs_' + '-'.join(proxies.use_from) +
                                  '_' + self.datatag_calib_T +
                                  '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                  '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                  '.pckl')
                self.pre_calib_datafile_T = join(lmr_path,
                                                 'PSM',
                                                 filename_t)
            else:
                self.pre_calib_datafile_T = self.pre_calib_datafile_T


                
            if self.pre_calib_datafile_P is None:
                if '-'.join(proxies.use_from) == 'LMRdb':
                    dbversion = proxies.LMRdb.dbversion
                    filename_p = ('PSMs_' + '-'.join(proxies.use_from) +
                                  '_' + dbversion +
                                  '_' + self.avgPeriod +
                                  '_' + self.datatag_calib_P +
                                  '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                  '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                  '.pckl')
                else:
                    filename_p = ('PSMs_' + '-'.join(proxies.use_from) +
                                  '_' + self.datatag_calib_P +
                                  '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                  '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                  '.pckl')
                self.pre_calib_datafile_P = join(lmr_path,
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

                
    class bilinear(ConfigGroup):
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

        # calibration source for  temperature
        # -----------------------------------
        datatag_calib_T = 'GISTEMP'
        
        # calibration source for precipitation/moisture
        # ---------------------------------------------
        datatag_calib_P = 'GPCC'

        pre_calib_datafile = None
        psm_r_crit = 0.0

        
        ##** END User Parameters **##

        def __init__(self, lmr_path=None, **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            self.datatag_calib_T = self.datatag_calib_T
            dataset_descr_T = _DataInfo.get_info(self.datatag_calib_T)
            self.datainfo_calib_T = dataset_descr_T['info']
            self.datadir_calib_T = dataset_descr_T['datadir']
            self.datafile_calib_T = dataset_descr_T['datafile']
            self.dataformat_calib_T = dataset_descr_T['dataformat']

            self.datatag_calib_P = self.datatag_calib_P
            dataset_descr_P = _DataInfo.get_info(self.datatag_calib_P)
            self.datainfo_calib_P = dataset_descr_P['info']
            self.datadir_calib_P = dataset_descr_P['datadir']
            self.datafile_calib_P = dataset_descr_P['datafile']
            self.dataformat_calib_P = dataset_descr_P['dataformat']

            self.psm_r_crit = self.psm_r_crit

            if '-'.join(proxies.use_from) == 'PAGES2kv1' and 'season' in psm.avgPeriod:
                print('ERROR: Trying to use seasonality information with the PAGES2kv1 proxy records.')
                print('       No seasonality metadata provided in that dataset. Exiting!')
                print('       Change avgPeriod to "annual" in your configuration.')
                raise SystemExit()

            try:
                if psm.avgPeriod == 'annual':
                    self.avgPeriod = psm.avgPeriod
                elif psm.avgPeriod == 'season' and psm.season_source:
                    if psm.season_source == 'proxy_metadata':
                        self.avgPeriod = psm.avgPeriod+'META'
                    elif psm.season_source == 'psm_calib':
                        self.avgPeriod = psm.avgPeriod+'PSM'
                    else:
                        print('ERROR: unrecognized psm.season_source attribute!')
                        raise SystemExit()
                else:
                    self.avgPeriod = psm.avgPeriod
            except:
                self.avgPeriod = psm.avgPeriod

            
            if lmr_path is None:
                lmr_path = core.lmr_path
            
            if self.datadir_calib_T is None:
                self.datadir_calib_T = join(lmr_path, 'data', 'analyses')
            if self.datadir_calib_P is None:
                self.datadir_calib_P = join(lmr_path, 'data', 'analyses')
                
            if self.pre_calib_datafile is None:
                if '-'.join(proxies.use_from) == 'LMRdb':
                    dbversion = proxies.LMRdb.dbversion
                    filename = ('PSMs_'+'-'.join(proxies.use_from) +
                                '_' + dbversion +
                                '_' + self.avgPeriod +
                                '_' + self.datatag_calib_T +
                                '_' + self.datatag_calib_P +
                                '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                '.pckl')
                else:
                    filename = ('PSMs_'+'-'.join(proxies.use_from) +
                                '_' + self.datatag_calib_T +
                                '_' + self.datatag_calib_P +
                                '_ref' + str(core.anom_reference_period[0]) + '-' + str(core.anom_reference_period[1]) +
                                '_cal' + str(psm.calib_period[0]) + '-' + str(psm.calib_period[1]) +
                                '.pckl')
                self.pre_calib_datafile = join(lmr_path,
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

        
    class h_interp(ConfigGroup):
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
        
        datadir_obsError = './'
        filename_obsError = 'R.txt'
        dataformat_obsError = 'TXT'

        datafile_obsError = None

        ##** END User Parameters **##

        def __init__(self, **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            self.radius_influence = self.radius_influence
            self.datadir_obsError = self.datadir_obsError
            self.filename_obsError = self.filename_obsError
            self.dataformat_obsError = self.dataformat_obsError 

            if self.datafile_obsError is None:
                self.datafile_obsError = join(self.datadir_obsError,
                                              self.filename_obsError)
            else:
                self.datafile_obsError = self.datafile_obsError

            # define state variable needed to calculate Ye's
            # only d18O for now ...

            # psm requirements depend on settings in proxies class 
            proxy_kind = proxies.proxy_timeseries_kind
            if proxies.proxy_timeseries_kind == 'asis':
                psm_var_kind = 'full'
            elif proxies.proxy_timeseries_kind == 'anom':
                psm_var_kind = 'anom'
            else:
                raise ValueError('Unrecognized proxy_timeseries_kind value in proxies class.'
                                 ' Unable to assign kind to psm_required_variables'
                                 ' in h_interp psm class.')
            self.psm_required_variables = {'d18O_sfc_Amon': psm_var_kind}


    class bayesreg_uk37(ConfigGroup):
        """
        Parameters for the Bayesian regression PSM for uk37 proxies.

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
        
        datadir_BayesRegressionData = None
        filename_BayesRegressionData = None
        
        dataformat_BayesRegressionData = 'MAT' # a .mat Matlab file

        
        ##** END User Parameters **##

        def __init__(self, **kwargs):
            super(self.__class__, self).__init__(**kwargs)

            if self.datadir_BayesRegressionData is None:
                self.datadir_BayesRegressionData = join(core.lmr_path, 'PSM')
            else:
                self.datadir_BayesRegressionData = self.datadir_BayesRegressionData
            
            if self.filename_BayesRegressionData is None:
                self.filename_BayesRegressionData = 'PSM_bayes_posterior_UK37.mat'
            else:
                self.filename_BayesRegressionData = self.filename_BayesRegressionData

            self.datafile_BayesRegressionData = join(self.datadir_BayesRegressionData,
                                              self.filename_BayesRegressionData)
    
            # define state variable needed to calculate Ye's
            self.psm_required_variables = {'tos_sfc_Odec': 'full'}

            # matlab engine as python object, needed for calculations
            # of the forward model.
            #self.MatlabEng = matlab.engine.start_matlab('-nojvm')

            self.datadir_BayesRegressionData = self.datadir_BayesRegressionData
            self.datafile_BayesRegressionData = self.datafile_BayesRegressionData
            self.dataformat_BayesRegressionData = self.dataformat_BayesRegressionData


    # Initialize subclasses with all attributes 
    def __init__(self, lmr_path=None, **kwargs):
        self.linear = self.linear(lmr_path=lmr_path, **kwargs.pop('linear', {}))
        self.linear_TorP = self.linear_TorP(lmr_path=lmr_path,
                                            **kwargs.pop('linear_TorP', {}))
        self.bilinear = self.bilinear(lmr_path=lmr_path,
                                      **kwargs.pop('bilinear', {}))
        self.h_interp = self.h_interp(**kwargs.pop('h_interp', {}))
        self.bayesreg_uk37 = self.bayesreg_uk37(**kwargs.pop('bayesreg_uk37', {}))

        super(self.__class__, self).__init__(**kwargs)
        self.avgPeriod = self.avgPeriod
        self.all_calib_sources = deepcopy(self.all_calib_sources)


class prior(ConfigGroup):
    """
    Parameters for the ensemble DA prior

    Attributes
    ----------
    prior_source: str
        Source of prior data
    datadir_prior: str
        Absolute path to prior data *or* None if using default LMR path
    datafile_prior: str
        Name of prior file to use
    dataformat_prior: str
        Datatype of prior container ('NCD' for netCDF, 'TXT' for ascii files).
        Note: Currently not used. 
    state_variables: dict.
       Dict. of the form {'var1': 'kind1', 'var2':'kind2', etc.} where 'var1', 'var2', etc.
       (keys of the dict) are the names of the state variables to be included in the state
       vector and 'kind1', 'kind2' etc. are the associated "kind" for each state variable
       indicating whether anomalies ('anom') or full field ('full') are desired. 
    detrend: bool
        Indicates whether to detrend the prior or not. Applies to ALL state variables.
    avgInterval: dict OR list(int) 
        dict of the form {'type':value} where 'type' indicates the type of averaging
        ('annual' or 'multiyear'). 
        If type = 'annual', the corresponding value is a 
        list of integers indficsting the months of the year over which the averaging
        is the be performed (ex. [6,7,8] for JJA).
        If type = 'multiyear', the list is composed of a single integer indicating
        the length of the averaging period, in number of years 
        (ex. [100] for prior returned as 100-yr averages).
        -OR-
        List of integers indicating the months over which to average the annual prior.
        (as 'annual' above).
    regrid_method: str
        String indicating the method used to regrid the prior to lower spatial resolution.
        Allowed options are: 
        1) None : Regridding NOT performed. 
        2) 'spherical_harmonics' : Original regridding using pyspharm library.
        3) 'simple': Regridding through simple inverse distance averaging of surrounding grid points.
        4) 'esmpy': Regridding using the ESMpy package. Includes bilinear and
           higher-order patch fit regridding.
    regrid_resolution: int
        Integer representing the triangular truncation of the lower resolution grid (e.g. 42 for T42).
        Not used for 'esmpy' regrid_method.
    esmpy_interp_method: str
        Which ESMpy regridding method to use.  Currently supports bilinear or 
        higher-oder patch fit interpolation regridding.
    esmpy_regrid_to: str
        A grid defined in grid_def.yml to use as the regridding target.  
        Currently supports 't42' and 'reg_4x5deg'.
    state_variables_info: dict
        Defines which variables represent temperature or moisture.
        Should be modified only if a new temperature or moisture state variable is added. 
    """

    ##** BEGIN User Parameters **##

    # Prior data directory & model source
    prior_source = 'ccsm4_last_millenium'

    # dict defining variables to be included in state vector (keys)
    # and associated "kind", i.e. as anomalies ('anom') or full field ('full')
    state_variables = {
        'tas_sfc_Amon'              : 'anom',
    #    'tos_sfc_Omon'              : 'anom',
    #    'pr_sfc_Amon'               : 'anom',
    #    'scpdsi_sfc_Amon'           : 'anom',
    #    'psl_sfc_Amon'              : 'anom',
    #    'zg_500hPa_Amon'            : 'anom',
    #    'wap_500hPa_Amon'           : 'anom',
    #    'wap_700hPa_Amon'           : 'anom',
    #    'wap_850hPa_Amon'           : 'anom',
    #    'wap_1000hPa_Amon'          : 'anom',
    #    'AMOCindex_Omon'            : 'anom',
    #    'AMOC26Nmax_Omon'           : 'anom',
    #    'AMOC26N1000m_Omon'         : 'anom',
    #    'AMOC45N1000m_Omon'         : 'anom',
    #    'ohcAtlanticNH_0-700m_Omon' : 'anom',
    #    'ohcAtlanticSH_0-700m_Omon' : 'anom',
    #    'ohcPacificNH_0-700m_Omon'  : 'anom',
    #    'ohcPacificSH_0-700m_Omon'  : 'anom',
    #    'ohcIndian_0-700m_Omon'     : 'anom',
    #    'ohcSouthern_0-700m_Omon'   : 'anom',
    #    'ohcArctic_0-700m_Omon'     : 'anom',
    #    'ohc_0-700m_Omon'           : 'anom',
    #    'sos_sfc_Omon'              : 'anom',        
    #    'd18O_sfc_Amon'             : 'full',
    #    'tas_sfc_Adec'              : 'full',
    #    'psl_sfc_Adec'              : 'full',
    #    'tos_sfc_Odec'              : 'full',
    #    'sos_sfc_Odec'              : 'full',
        }


    
    
    ## The reference period (in year CE) for calculation of anomalies
    ## ** Valid for prior ccsm3_trace21ka only for now. Use None for all others **
    ## Options: None or tuple indicating the reference period
    #anom_reference = None



    
    # boolean : detrend prior?
    # by default, considers the entire length of the simulation
    detrend = False

    # Method for regridding to lower resolution grid.
    # Possible methods:
    # 1) None : No regridding performed. 
    # 2) 'spherical_harmonics': through spherical harmonics using pyspharm library.
    #    Note: Does *NOT* handle fields with missing/masked values (e.g. ocean variables)
    # 3) 'simple': through simple interpolation using distance-weighted averaging.
    #    Note: fields with missing/masked values (e.g. ocean variables) allowed.
    # 4) 'esmpy': regridding facilitated by ESMpy package, includes blinear,
    #    and higher order patch interpolation
    regrid_method = 'simple'
    # resolution of truncated grid, based on triangular truncation (e.g., use 42 for T42))
    # note: this option applies only to 'simple' or 'spherical_harmonics' options
    regrid_resolution = 42
    # for 'esmpy' option
    esmpy_interp_method = 'bilinear'
    esmpy_regrid_to = 't42'
    
    # Dict. defining which variables represent temperature or moisture.
    # To be modified only if a new temperature or moisture state variable is added. 
    state_variables_info = {'temperature': ['tas_sfc_Amon'],
                            'moisture': ['pr_sfc_Amon','scpdsi_sfc_Amon']}

    
    ##** END User Parameters **##

    
    def __init__(self, lmr_path=None, seed=None, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        self.prior_source = self.prior_source

        dataset_descr = _DataInfo.get_info(self.prior_source)
        self.datainfo_prior = dataset_descr['info']
        self.datadir_prior = dataset_descr['datadir']
        self.datafile_prior = dataset_descr['datafile']
        self.dataformat_prior = dataset_descr['dataformat']

        self.state_variables = deepcopy(self.state_variables)
        self.state_variables_info = deepcopy(self.state_variables_info)
        self.detrend = self.detrend
        self.regrid_method = self.regrid_method

        # check if "anom" has been selected for any state variable
        # and set the anom_reference attribute accordingly
        if any(self.state_variables[item] for item in self.state_variables.keys()):
            self.anom_reference = core.anom_reference_period
        else:
            self.anom_reference = None
        
        if seed is None:
            seed = core.seed
        self.seed = seed

        if lmr_path is None:
            lmr_path = core.lmr_path

        if self.datadir_prior is None:
            self.datadir_prior = join(lmr_path, 'data', 'model',
                                      self.prior_source)

        if core.recon_timescale == 1:
            self.avgInterval = {'annual': [1,2,3,4,5,6,7,8,9,10,11,12]} # annual (calendar) as default
        elif core.recon_timescale > 1:
            # new format for multiyear DTDA reconstructions:
            self.avgInterval = {'multiyear': [core.recon_timescale]}
        else:
            print('ERROR in config.: unrecognized core.recon_timescale!')
            raise SystemExit()
        
        if self.regrid_method != 'esmpy':
            self.regrid_resolution = int(self.regrid_resolution)
        elif self.regrid_method == 'esmpy':
            self.regrid_resolution = None
            self.esmpy_interp_method = self.esmpy_interp_method
            self.esmpy_grid_def = _GridDef.get_info(self.esmpy_regrid_to)
        else:
            self.regrid_resolution = None

        
        # Is variable requested in list of those specified as available?
        var_mismat = [varname for varname in self.state_variables
                      if varname not in self.datainfo_prior['available_vars']]
        if var_mismat:
            raise SystemExit(('Could not find requested variable(s) {} in the '
                              'list of available variables for the {} '
                              'dataset').format(var_mismat,
                                                self.prior_source))


            
class Config(ConfigGroup):
    """
    An instanceable container for all the configuration objects.
    """
    
    def __init__(self, **kwargs):
        self.LEGACY_CONFIG = LEGACY_CONFIG
        self.SRC_DIR = SRC_DIR
        self.LOG_LEVEL = LOG_LEVEL

        self.wrapper = wrapper(**kwargs.pop('wrapper', {}))
        self.core = core(**kwargs.pop('core', {}))
        lmr_path = self.core.lmr_path
        seed = self.core.seed
        self.proxies = proxies(lmr_path=lmr_path,
                               seed=seed,
                               **kwargs.pop('proxies', {}))
        self.psm = psm(lmr_path=lmr_path, **kwargs.pop('psm', {}))
        self.prior = prior(lmr_path=lmr_path,
                           seed=seed,
                           **kwargs.pop('prior', {}))


def is_config_class(obj):
    """
    Tests whether the input object is an instance of ConfigGroup

    Parameters
    ----------
    obj: object
        Object for testing
    """
    try:
        if isinstance(obj, ConfigGroup):
            return True
        else:
            return issubclass(obj, ConfigGroup)
    except TypeError:
        return False


def update_config_class_yaml(yaml_dict, cfg_module):
    """
    Updates a configuration object using a dictionary (typically from a yaml
    file) that follows the naming convention and nesting of these
    configuration classes.

    Parameters
    ----------
    yaml_dict: dict
        The dictionary of values to update in the current configuration input
    cfg_module: ConfigGroup like
        The configuration object to be updated by yaml_dict

    Returns
    -------
    dict
        Returns a dictionary of all unused parameters from the update process

    Examples
    --------
    If cfg_module is an imported LMR_config as cfg then the following
    dictionary could be used to update a core and linear psm attribute.
    yaml_dict = {'core': {'lmr_path': '/new/path/to/LMR_files'},
                 'psm': {'linear': {'datatag_calib': 'GISTEMP'}}}

    These are the types of dictionaries that result from a yaml.load function.

    Warnings
    --------
    This function is meant to be run on imported configuration classes not
    their instances.  If you'd only like to update the attributes of an
    instance then please use keyword arguments during initialization.
    """

    for attr_name in list(yaml_dict.keys()):
        try:
            curr_cfg_obj = getattr(cfg_module, attr_name)
            cfg_attr = yaml_dict.pop(attr_name)

            if is_config_class(curr_cfg_obj):
                result = update_config_class_yaml(cfg_attr, curr_cfg_obj)

                if result:
                    yaml_dict[attr_name] = result

            else:
                setattr(cfg_module, attr_name, cfg_attr)
        except (AttributeError, KeyError) as e:
            print(e)

    return yaml_dict


if __name__ == "__main__":
    kwargs = {'wrapper': {'multi_seed': [1, 2, 3]}, 'psm': {'linear': {'datatag_calib': 'BE'}}}
    tmp = Config(**kwargs)
    pass
