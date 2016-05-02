"""
Class based config module to help with passing information to LMR modules for
reconstruction experiments.

Adapted from LMR_exp_NAMELIST by AndreP
"""

from os.path import join

class wrapper(object):
    """
    Parameters for reconstruction realization manager LMR_wrapper.

    Attributes
    ----------
    multi_seed: list(int), None
        List of RNG seeds to be used during a reconstruction for each
        realization.  This overrides the 'core.seed' parameter.
    iter_range: list(int)
        Range of Monte-Carlo iterations to perform
    """

    multi_seed = None
    iter_range = [0, 0]

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
    recon_period: list(int)
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
    nexp = 'testdev_ye_precalc_include_tas_statevar_falseflag'
    lmr_path = '/home/chaos2/wperkins/data/LMR'
    online_reconstruction = False
    clean_start = True
    use_precalc_ye = False
    # TODO: More pythonic to make last time a non-inclusive edge
    recon_period = [1900, 1960]
    nens = 10
    seed = 42
    loc_rad = None

    datadir_output = '/home/chaos2/wperkins/data/LMR/output/working'
    #datadir_output  = '/home/disk/kalman3/rtardif/LMR/output/wrk'
    #datadir_output  = '/home/disk/ekman/rtardif/nobackup/LMR/output'
    #datadir_output  = '/home/disk/ice4/hakim/svnwork/python-lib/trunk/src/ipython_notebooks/data'

    archive_dir = '/home/chaos2/wperkins/data/LMR/output/testing'
    #archive_dir = '/home/disk/kalman3/rtardif/LMR/output'
    #archive_dir = '/home/disk/kalman3/hakim/LMR/'

    def __init__(self):
        self.nexp = self.nexp
        self.lmr_path = self.lmr_path
        self.online_reconstruction = self.online_reconstruction
        self.clean_start = self.clean_start
        self.use_precalc_ye = self.use_precalc_ye
        self.recon_period = self.recon_period
        self.nens = self.nens
        self.curr_iter = wrapper.iter_range[0]
        self.loc_rad = self.loc_rad
        self.seed = self.seed
        self.datadir_output = self.datadir_output
        self.archive_dir = self.archive_dir

class proxies(object):
    """
    Parameters for proxy data

    Attributes
    ----------
    use_from: list(str)
        A list of keys for proxy classes to load from.  Keys available are
        stored in LMR_proxy2.
    proxy_frac: float
        Fraction of available proxy data (sites) to assimilate
    """

    use_from = ['pages']
    proxy_frac = 0.75

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

        def __init__(self, datadir_proxy=None):
            if datadir_proxy is None:
                self.datadir_proxy = join(core.lmr_path, 'data', 'proxies')
            else:
                self.datadir_proxy = datadir_proxy

            self.datafile_proxy = join(self.datadir_proxy,
                                       self.datafile_proxy)
            self.metafile_proxy = join(self.datadir_proxy,
                                       self.metafile_proxy)\

            self.dataformat_proxy = self.dataformat_proxy
            self.regions = self.regions
            self.proxy_resolution = self.proxy_resolution
            self.proxy_order = self.proxy_order
            self.proxy_assim2 = self.proxy_assim2

            # Create mapping for Proxy Type/Measurement Type to type names above
            self.proxy_type_mapping = {}
            for ptype, measurements in self.proxy_assim2.iteritems():
                # Fetch proxy type name that occurs before underscore
                type_name = ptype.split('_', 1)[0]
                for measure in measurements:
                    self.proxy_type_mapping[(type_name, measure)] = ptype

            self.simple_filters = {'PAGES 2k Region': self.regions,
                                   'Resolution (yr)': self.proxy_resolution}

    # Initialize subclasses with all attributes
    def __init__(self, **kwargs):
        self.use_from = self.use_from
        self.proxy_frac = self.proxy_frac
        self.seed = core.seed
        self.pages = self._pages(**kwargs)

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

    use_psm = {'pages': 'linear'}

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
        dataformat_calib = 'NCD'
        #datatag_calib = 'MLOST'
        #datafile_calib = 'MLOST_air.mon.anom_V3.5.4.nc'

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

    def __init__(self, **kwargs):
        self.use_psm = self.use_psm
        self.linear = self._linear(**kwargs)



class prior:
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
    datafile_prior   = '[vardef_template]_CCSM4_past1000_085001-185012.nc'

    #prior_source     = 'ccsm4_preindustrial_control'
    #datafile_prior   = '[vardef_template]_CCSM4_piControl_080001-130012.nc

    #prior_source     = 'gfdl-cm3_preindustrial_control'
    #datafile_prior   = '[vardef_template]_GFDL-CM3_piControl_000101-050012.nc'

    #prior_source     = 'mpi-esm-p_last_millenium'
    #datafile_prior   = '[vardef_template]_MPI-ESM-P_past1000_085001-185012.nc'

    # prior_source     = '20cr'
    # datafile_prior   = '[vardef_template]_20CR_185101-201112.nc'

    #prior_source     = 'era20c'
    #datafile_prior   = '[vardef_template]_ERA20C_190001-201212.nc'

    dataformat_prior = 'NCD'
    psm_required_variables = ['tas_sfc_Amon']
    # state_variables = ['tas_sfc_Amon']
    state_variables = ['tas_sfc_Amon', 'zg_500hPa_Amon']
    #state_variables = ['tas_sfc_Amon', 'zg_500hPa_Amon', 'AMOCindex_Omon']
    # state_variables = ['tas_sfc_Amon', 'zg_500hPa_Amon', 'AMOCindex_Omon',
    #                    'ohcAtlanticNH_0-700m_Omon', 'ohcAtlanticSH_0-700m_Omon',
    #                    'ohcPacificNH_0-700m_Omon', 'ohcPacificSH_0-700m_Omon',
    #                    'ohcIndian_0-700m_Omon', 'ohcSouthern_0-700m_Omon']

    def __init__(self, datadir_prior=None):
        self.prior_source = self.prior_source
        self.datafile_prior = self.datafile_prior
        self.dataformat_prior = self.dataformat_prior
        self.state_variables = self.state_variables
        self.psm_required_variables = self.psm_required_variables
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

