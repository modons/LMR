"""
Class based config module to help with passing information to LMR modules for
reconstruction experiments.

Adapted from LMR_exp_NAMELIST by AndreP
"""

from os.path import join
import random


class constants:

    class file_types:
        netcdf = 'NCD'
        ascii = 'ASC'
        numpy = 'NPY'
        numpy_zip = 'NPZ'
        dataframe = 'DF'

    calib = {}
    calib['GISTEMP'] = {'fname': 'gistemp1200_ERSST.nc',
                       'varname': 'tempanomaly',
                       'type': 'NCD'}

    calib['HadCRUT'] = {'fname': 'HadCRUT.4.3.0.0.median.nc',
                        'varname': 'Tsfc',
                        'type': 'NCD'}

    calib['BerkeleyEarth'] = {'fname': 'Land_and_Ocean_LatLong1.nc',
                              'varname': 'Tsfc',
                              'type': 'NCD'}

    calib['MLOST'] = {'fname': 'MLOST_air.mon.anom.V3.5.4.nc',
                      'varname': 'Tsfc',
                      'type': 'NCD'}

    calib['NOAA'] = {'fname': 'er-ghcn-sst.nc',
                     'varname': 'Tsfc',
                     'type': 'NCD'}

    prior = {}
    prior['ccsm4_last_millenium'] = \
        {'fname': 'tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc',
         'type': 'NCD',
         'state_vars': ['tas_sfc_Amon']}

    prior['mpi-esm-p_last_millenium'] = \
        {'fname': '[vardef_template]_MPI-ESM-P_past1000_085001-185012.nc',
         'type': 'NCD',
         'state_vars': ['tas_sfc_Amon']}

class core:
    """
    High-level parameters of reconstruction experiment

    Attributes
    ----------
    nexp: str
        Name of reconstruction experiment
    lmr_path: str
        Absolute path for the experiment
    clean_start: bool
        Delete existing files in output directory (otherwise they will be used
        as the prior!)
    recon_period: list(int)
        Time period for reconstruction
    nens: int
        Ensemble size
    iter_range: list(int)
        Number of Monte-Carlo iterations to perform
    loc_rad: float
        Localization radius for DA (in km)
    datadir_output: str
        Absolute path to working directory output for LMR
    archive_dir: str
        Absolute path to LMR reconstruction archive directory
    """
    nexp = 'testdev_addlim_parameter_search'
    lmr_path = '/home/chaos2/wperkins/data/LMR'
    online_reconstruction = True
    clean_start = True
    ignore_pre_avg_file = False
    overwrite_pre_avg_file = False
    # TODO: More pythonic to make last time a non-inclusive edge
    recon_period = [1850, 2000]
    nens = 100
    seed = 25
    iter_range = [0, 2]
    curr_iter = iter_range[0]
    loc_rad = None
    assimilation_time_res = [1.0]  # in yrs
    # maps year shift (in years) to resolution
    res_yr_shift = {0.5: 0.25, 1.0: 0.0}

    # Forecasting Hybrid Update
    hybrid_update = True
    hybrid_update &= online_reconstruction
    hybrid_a = 0.5


    # TODO: add rules for shift?
    # If shifting on smaller time scales than smallest time chunk it becomes
    # the base resolution
    sub_base_res = assimilation_time_res[0]
    for res, shift in res_yr_shift.iteritems():
        if (res in assimilation_time_res and
           shift < sub_base_res and
           shift != 0.0):
            sub_base_res = shift

    datadir_output = '/home/chaos2/wperkins/data/LMR/output/working'
    #datadir_output  = '/home/disk/kalman3/rtardif/LMR/output/wrk'
    #datadir_output  = '/home/disk/ekman/rtardif/nobackup/LMR/output'
    #datadir_output  = '/home/disk/ice4/hakim/svnwork/python-lib/trunk/src/ipython_notebooks/data'

    archive_dir = '/home/chaos2/wperkins/data/LMR/output/archive'
    #archive_dir = '/home/disk/kalman3/rtardif/LMR/output'
    #archive_dir = '/home/disk/kalman3/hakim/LMR/'

class proxies:
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

    class pages:
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

        datadir_proxy = join(core.lmr_path, 'data', 'proxies')
        # Pages 0.5yr resolution
        # datafile_proxy = join(datadir_proxy,
        #                       'Pages2k_Proxies_0pt5res.df.pckl')
        # metafile_proxy = join(datadir_proxy,
        #                       'Pages2k_Metadata_0pt5res.df.pckl')

        # Pages 1.0 yr res only
        datafile_proxy = join(datadir_proxy,
                              'Pages2k_Proxies.df.pckl')
        metafile_proxy = join(datadir_proxy,
                              'Pages2k_Metadata.df.pckl')
        dataformat_proxy = 'DF'

        regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
                   'North America', 'South America']
        proxy_resolution = core.assimilation_time_res

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

        # Create mapping for Proxy Type/Measurement Type to type names above
        proxy_type_mapping = {}
        for type, measurements in proxy_assim2.iteritems():
            # Fetch proxy type name that occurs before underscore
            type_name = type.split('_', 1)[0]
            for measure in measurements:
                proxy_type_mapping[(type_name, measure)] = type

        simple_filters = {'PAGES 2k Region': regions,
                          'Resolution (yr)': proxy_resolution}


class psm:
    """
    Parameters for PSM classes

    Attributes
    ----------
    use_psm: dict{str: str}
        Maps proxy class key to psm class key.  Used to determine which psm
        is associated with what Proxy type.
    """

    use_psm = {'pages': 'linear'}

    class linear:
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
        datatag_calib = 'GISTEMP'
        sub_base_res = core.sub_base_res
        datadir_calib = join(core.lmr_path, 'data', 'analyses', datatag_calib)
        datafile_calib = constants.calib[datatag_calib]['fname']
        varname_calib = constants.calib[datatag_calib]['varname']
        dataformat_calib = constants.calib[datatag_calib]['type']

        ignore_pre_avg_file = core.ignore_pre_avg_file
        overwrite_pre_avg_file = core.overwrite_pre_avg_file

        # pre_calib_datafile = join(core.lmr_path,
        #                           'PSM',
        #                           'PSMs_' + datatag_calib +
        #                           '_0pt5_1pt0_res.pckl')
        pre_calib_datafile = join(core.lmr_path,
                                  'PSM', 'test_psms',
                                  'PSMs_' + datatag_calib +
                                  '_1pt0res_1.00datfrac')
        psm_r_crit = 0.2
        min_data_req_frac = 1.0  # 0.0 no data required, 1.0 all data required


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
    state_variables: list(str)
        List of variables to use in the state vector for the prior
    """
    # Prior data directory & model source
    prior_source = 'ccsm4_last_millenium'
    #prior_source = 'mpi-esm-p_last_millenium'

    datadir_prior = join(core.lmr_path, 'data', 'model', prior_source)
    datafile_prior   = constants.prior[prior_source]['fname']
    dataformat_prior = constants.prior[prior_source]['type']
    state_variables = constants.prior[prior_source]['state_vars']
    truncate_state = True


class forecaster:
    """
    Parameters for the online DA forecasting method.
    """

    # Which forecaster class to use
    use_forecaster = 'lim'

    class LIM:
        """
        calib_filename: Filename for LIM calibration data.  Should be netcdf
                        file or an HDF5 file from the DataTools.netcdf_to_hdf5_
                        container.
        calib_varname: Variable name to grab from calib_filename
        fcast_times: A list of lead times (in years) to forecast
        """
        calib_filename = '/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc'
        calib_varname = 'air'
        dataformat = 'NCD'
        fcast_times = [1]
        wsize = 12
        fcast_num_pcs = 15
        detrend = True

        eig_adjust = 1
