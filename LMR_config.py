"""
Class based config module to help with passing information to LMR modules for
paleoclimate reconstruction experiments.

Adapted from LMR_exp_NAMELIST by AndreP

Revisions:
 - Introduction of definitions related to use of newly developed NCDC proxy
   database.
 - Added functionality restricting assimilated proxy records to those belonging
   to specific databases (e.g. PAGES1, PAGES2, LMR) (only for NCDC proxies).
 - Introduction of "blacklist" to prevent the assimilation of specific proxy
   records as defined (through a python list) by the user. Applicable for both
   NCDC and Pages proxy sets.
 - Added boolean allowing the user to indicate whether the prior is to be
   detrended or not.
 - Added definitions associated with a new psm class (linear_TorP) allowing the
   use of temperature-calibrated OR precipitation-calibrated linear PSMs.
   [ R. Tardif, Univ. of Washington, Spring 2016 ]

"""

from os.path import join
from copy import deepcopy

class wrapper(object):
    """
    Parameters for reconstruction realization manager LMR_wrapper.

    Attributes
    ----------
    multi_seed: list(int), None
        List of RNG seeds to be used during a reconstruction for each
        realization.  This overrides the 'core.seed' parameter.
    iter_range: tuple(int)
        Range of Monte-Carlo iterations to perform
    """

    multi_seed = None
    iter_range = (0, 0)

    def __init__(self):
        self.multi_seed = self.multi_seed
        self.iter_range = self.iter_range

class core(object):
    """
    High-level parameters of reconstruction experiment

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
    recon_period: tuple(int)
        Time period for reconstruction
    nens: int
        Ensemble size
    loc_rad: float
        Localization radius for DA (in km)
    seed: int, None
        RNG seed.  Passed to all random function calls. (e.g. Prior and proxy
        record sampling)  Overridden by wrapper.multi_seed.
    datadir_output: str
        Absolute path to working directory output for LMR
    archive_dir: str
        Absolute path to LMR reconstruction archive directory
    """

    nexp = 'testdev_precalc_integ_use_precalc_pr_req_not_in_statevar'

    lmr_path = '/home/chaos2/wperkins/data/LMR'
    # lmr_path = '/home/disk/kalman3/rtardif/LMR'
    online_reconstruction = False
    clean_start = True
    use_precalc_ye = True
    # TODO: More pythonic to make last time a non-inclusive edge
    recon_period = (1900, 1960)
    nens = 10
    seed = 0
    loc_rad = None

    datadir_output = '/home/chaos2/wperkins/data/LMR/output/working'
    # datadir_output  = '/home/disk/kalman3/rtardif/LMR/output/wrk'
    #datadir_output  = '/home/disk/ekman/rtardif/nobackup/LMR/output'

    archive_dir = '/home/chaos2/wperkins/data/LMR/output/testing'
    # archive_dir = '/home/disk/kalman3/rtardif/LMR/output'
    # archive_dir = '/home/disk/kalman3/hakim/LMR/'

    def __init__(self, curr_iter=None):
        self.nexp = self.nexp
        self.lmr_path = self.lmr_path
        self.online_reconstruction = self.online_reconstruction
        self.clean_start = self.clean_start
        self.use_precalc_ye = self.use_precalc_ye
        self.recon_period = self.recon_period
        self.nens = self.nens
        self.loc_rad = self.loc_rad
        self.seed = self.seed
        self.datadir_output = self.datadir_output
        self.archive_dir = self.archive_dir

        if curr_iter is None:
            self.curr_iter = wrapper.iter_range[0]
        else:
            self.curr_iter = curr_iter

class proxies(object):
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

    # =============================
    # Which proxy database to use ?
    # =============================
    # use_from = ['pages']
    use_from = ['NCDC']

    # proxy_frac = 1.0
    proxy_frac = 0.75

    # ---------------
    # PAGES2k proxies
    # ---------------
    class _pages(object):
        """
        Parameters for PagesProxy class

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

        def __init__(self, datadir_proxy=None):
            if datadir_proxy is None:
                self.datadir_proxy = join(core.lmr_path, 'data', 'proxies')
            else:
                self.datadir_proxy = datadir_proxy

            self.datafile_proxy = join(self.datadir_proxy,
                                       self.datafile_proxy)
            self.metafile_proxy = join(self.datadir_proxy,
                                       self.metafile_proxy)

            self.dataformat_proxy = self.dataformat_proxy
            self.regions = list(self.regions)
            self.proxy_resolution = self.proxy_resolution
            self.proxy_order = list(self.proxy_order)
            self.proxy_assim2 = deepcopy(self.proxy_assim2)
            self.proxy_blacklist = list(self.proxy_blacklist)

            # Create mapping for Proxy Type/Measurement Type to type names above
            self.proxy_type_mapping = {}
            for ptype, measurements in self.proxy_assim2.iteritems():
                # Fetch proxy type name that occurs before underscore
                type_name = ptype.split('_', 1)[0]
                for measure in measurements:
                    self.proxy_type_mapping[(type_name, measure)] = ptype

            self.simple_filters = {'PAGES 2k Region': self.regions,
                                   'Resolution (yr)': self.proxy_resolution}

    # ---------------
    # NCDC proxies
    # ---------------
    class _ncdc(object):
        """
        Parameters for NCDCProxy class

        Attributes
        ----------
        datadir_proxy: str
            Absolute path to proxy data *or* None if using default lmf_path
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
        simple_filters: dict{'str': Iterable}
            List mapping proxy metadata sheet columns to a list of values
            to filter by.
        """
        datafile_proxy = 'NCDC_Proxies.df.pckl'
        metafile_proxy = 'NCDC_Metadata.df.pckl'
        dataformat_proxy = 'DF'

        regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
                   'North America', 'South America']

        proxy_resolution = [1.0]

        # Limit proxies to those included in the following databases
        #database_filter = ['PAGES1']
        database_filter = []

        # DO NOT CHANGE FORMAT BELOW
        proxy_order = [
#            'Tree Rings_All',
            'Tree Rings_WoodDensity',
            'Tree Rings_WidthPages',
#            'Tree Rings_WidthBreit',
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
            'Speleothems_d18O',
#            'Speleothems_d13C'
            ]

        proxy_assim2 = {
            'Corals and Sclerosponges_d18O': ['d18O', 'delta18O', 'd18o',
                                              'd18O_stk', 'd18O_int',
                                              'd18O_norm', 'd18o_avg',
                                              'd18o_ave', 'dO18',
                                              'd18O_4'],
            'Corals and Sclerosponges_d14C': ['d14C', 'd14c', 'ac_d14c'],
            'Corals and Sclerosponges_d13C': ['d13C', 'd13c', 'd13c_ave',
                                              'd13c_ann_ave', 'd13C_int'],
            'Corals and Sclerosponges_SrCa': ['Sr/Ca', 'Sr/Ca_norm',
                                              'Sr/Ca_anom', 'Sr/Ca_int'],
            'Corals and Sclerosponges_Sr'  : ['Sr'],
            'Corals and Sclerosponges_BaCa': ['Ba/Ca'],
            'Corals and Sclerosponges_CdCa': ['Cd/Ca'],
            'Corals and Sclerosponges_MgCa': ['Mg/Ca'],
            'Corals and Sclerosponges_UCa' : ['U/Ca', 'U/Ca_anom'],
            'Corals and Sclerosponges_Pb'  : ['Pb'],
            'Ice Cores_d18O'               : ['d18O', 'delta18O', 'delta18o',
                                              'd18o', 'd18o_int', 'd18O_int',
                                              'd18O_norm', 'd18o_norm', 'dO18',
                                              'd18O_anom'],
            'Ice Cores_dD'                 : ['deltaD', 'delD'],
            'Ice Cores_Accumulation'       : ['accum', 'accumu'],
            'Ice Cores_MeltFeature'        : ['MFP'],
            'Lake Cores_Varve'             : ['varve', 'varve_thickness',
                                              'varve thickness'],
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

        # A blacklist on proxy records, to prevent assimilation of chronologies
        # known to be duplicates.
        # proxy_blacklist = []
        proxy_blacklist = ['00aust01a', '06cook02a', '06cook03a', '08vene01a',
                           '09japa01a', '10guad01a', '99aust01a', '99fpol01a']

        def __init__(self, datadir_proxy=None):
            if datadir_proxy is None:
                self.datadir_proxy = join(core.lmr_path, 'data', 'proxies')
            else:
                self.datadir_proxy = datadir_proxy

            self.datafile_proxy = join(self.datadir_proxy,
                                       self.datafile_proxy)
            self.metafile_proxy = join(self.datadir_proxy,
                                       self.metafile_proxy)

            self.dataformat_proxy = self.dataformat_proxy
            self.regions = list(self.regions)
            self.proxy_resolution = self.proxy_resolution
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
        self.seed = core.seed
        self.pages = self._pages(**kwargs)
        self.ncdc = self._ncdc(**kwargs)


class psm(object):
    """
    Parameters for PSM classes

    Attributes
    ----------
    use_psm: dict{str: str}
        Maps proxy class key to psm class key.  Used to determine which psm
        is associated with what Proxy type. This mapping can be extended to
        make more intricate proxy - psm relationships.
    """

    # use_psm = {'pages': 'linear', 'NCDC': 'linear'}
    use_psm = {'pages': 'linear', 'NCDC': 'linear_TorP'}

    class _linear(object):
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
            Absolute path to precalibrated Linear PSM data
        psm_r_crit: float
            Usage threshold for correlation of linear PSM
        """
        datatag_calib = 'GISTEMP'
        datafile_calib = 'gistemp1200_ERSST.nc'
        # or
        # datatag_calib = 'MLOST'
        # datafile_calib = 'MLOST_air.mon.anom_V3.5.4.nc'
        # or
        # datatag_calib = 'HadCRUT'
        # datafile_calib = 'HadCRUT.4.4.0.0.median.nc'
        # or
        # datatag_calib = 'BerkeleyEarth'
        # datafile_calib = 'Land_and_Ocean_LatLong1.nc'
        # or
        # datatag_calib = 'GPCC'
        # datafile_calib = 'GPCC_precip.mon.total.1x1.v6.nc'
        # or
        # datatag_calib = 'GPCC'
        # datafile_calib = 'GPCC_precip.mon.flux.1x1.v6.nc'

        dataformat_calib = 'NCD'

        psm_r_crit = 0.0

        def __init__(self, datadir_calib=None, pre_calib_datafile=None):
            self.datatag_calib = self.datatag_calib
            self.datafile_calib = self.datafile_calib
            self.dataformat_calib = self.dataformat_calib
            self.psm_r_crit = self.psm_r_crit

            if datadir_calib is None:
                self.datadir_calib = join(core.lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = datadir_calib

            if pre_calib_datafile is None:
                filename = 'PSMs_'+self.datatag_calib+'.pckl'
                self.pre_calib_datafile = join(core.lmr_path,
                                               'PSM',
                                               filename)
            else:
                self.pre_calib_datafile = pre_calib_datafile

    class _linear_TorP(_linear):
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
        # linear PSM w.r.t. temperature
        # -----------------------------
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
        # linear PSM w.r.t. precipitation
        # ------------------------------
        # datatag_calib_P = 'GPCC'
        # datafile_calib_P = 'GPCC_precip.mon.total.1x1.v6.nc'
        # or
        datatag_calib_P = 'GPCC'
        datafile_calib_P = 'GPCC_precip.mon.flux.1x1.v6.nc'

        dataformat_calib = 'NCD'

        psm_r_crit = 0.0

        def __init__(self, datadir_calib=None, pre_calib_datafile_T=None,
                     pre_calib_datafile_P=None):
            self.datatag_calib_T = self.datatag_calib_T
            self.datafile_calib_T = self.datafile_calib_T
            self.datatag_calib_P = self.datatag_calib_P
            self.datafile_calib_P = self.datafile_calib_P
            self.dataformat_calib = self.dataformat_calib
            self.psm_r_crit = self.psm_r_crit

            if datadir_calib is None:
                self.datadir_calib = join(core.lmr_path, 'data', 'analyses')
            else:
                self.datadir_calib = datadir_calib

            if pre_calib_datafile_T is None:
                filename_t = 'PSMs_' + '-'.join(proxies.use_from) + '_' + \
                             self.datatag_calib_T + '.pckl'
                self.pre_calib_datafile_T = join(core.lmr_path,
                                                 'PSM',
                                                 filename_t)
            else:
                self.pre_calib_datafile_T = pre_calib_datafile_T

            if pre_calib_datafile_P is None:
                filename_p = 'PSMs_' + '-'.join(proxies.use_from) + '_' + \
                             self.datatag_calib_P + '.pckl'
                self.pre_calib_datafile_P = join(core.lmr_path,
                                                 'PSM',
                                                 filename_p)
            else:
                self.pre_calib_datafile_P = pre_calib_datafile_P

    def __init__(self, **kwargs):
        self.use_psm = self.use_psm
        self.linear = self._linear(**kwargs)

        # TODO: had to take kwargs out, need to figure out a way to structure
        self.linear_TorP = self._linear_TorP()


class prior(object):
    """
    Parameters for the ensDA prior

    Attributes
    ----------
    prior_source: str
        Source of prior data
    datadir_prior: str
        Absolute path to prior data
    datafile_prior: str
        Name of prior file to use
    dataformat_prior: str
        Datatype of prior container
    psm_required_variables: list(str)
        List of variables used to calculate ye values.
    state_variables: list(str)
        List of variables to use in the state vector for the prior.
    """
    # Prior data directory & model source
    prior_source = 'ccsm4_last_millenium'
    datafile_prior = '[vardef_template]_CCSM4_past1000_085001-185012.nc'

    # prior_source     = 'ccsm4_preindustrial_control'
    # datafile_prior   = '[vardef_template]_CCSM4_piControl_080001-130012.nc
    #
    # prior_source     = 'gfdl-cm3_preindustrial_control'
    # datafile_prior   = '[vardef_template]_GFDL-CM3_piControl_000101-050012.nc'
    #
    # prior_source     = 'mpi-esm-p_last_millenium'
    # datafile_prior   = '[vardef_template]_MPI-ESM-P_past1000_085001-185012.nc'
    #
    # prior_source     = '20cr'
    # datafile_prior   = '[vardef_template]_20CR_185101-201112.nc'
    #
    # prior_source     = 'era20c'
    # datafile_prior   = '[vardef_template]_ERA20C_190001-201012.nc'
    #
    # prior_source     = 'era20cm'
    # datafile_prior   = '[vardef_template]_ERA20CM_190001-201012.nc'

    dataformat_prior = 'NCD'
    psm_required_variables = ['tas_sfc_Amon', 'pr_sfc_Amon']
    state_variables = ['tas_sfc_Amon']
    # state_variables = ['tas_sfc_Amon', 'zg_500hPa_Amon']
    # state_variables = ['tas_sfc_Amon', 'zg_500hPa_Amon', 'AMOCindex_Omon']
    # state_variables = ['tas_sfc_Amon', 'zg_500hPa_Amon',
    #                    'AMOCindex_Omon', 'AMOC26Nmax_Omon',
    #                    'AMOC26N1000m_Omon', 'AMOC45N1000m_Omon',
    #                    'ohcAtlanticNH_0-700m_Omon',
    #                    'ohcAtlanticSH_0-700m_Omon',
    #                    'ohcPacificNH_0-700m_Omon', 'ohcPacificSH_0-700m_Omon',
    #                    'ohcIndian_0-700m_Omon', 'ohcSouthern_0-700m_Omon',
    #                    'ohcArctic_0-700m_Omon']
    # state_variables = ['tas_sfc_Amon', 'pr_sfc_Amon']

    # boolean : detrend prior?
    # by default, considers the entire length of the simulation
    detrend = False

    def __init__(self, datadir_prior=None):
        self.prior_source = self.prior_source
        self.datafile_prior = self.datafile_prior
        self.dataformat_prior = self.dataformat_prior
        self.state_variables = list(self.state_variables)
        self.psm_required_variables = self.psm_required_variables
        self.detrend = self.detrend
        self.seed = core.seed

        if datadir_prior is None:
            self.datadir_prior = join(core.lmr_path, 'data', 'model',
                                      self.prior_source)
        else:
            self.datadir_prior = datadir_prior


class Config(object):

    def __init__(self):
        self.wrapper = wrapper()
        self.core = core()
        self.proxies = proxies()
        self.psm = psm()
        self.prior = prior()

