"""
Class based config module to help with passing information to LMR modules for
reconstruction experiments.

Adapted from LMR_exp_NAMELIST by AndreP
"""


# =============================================================================
# Section 1: High-level parameters of reconstruction experiment
# =============================================================================

class core:
    """
    High-level parameters of reconstruction experiment
    """
    # Name of reconstruction experiment
    nexp = 'testdev_1000_75pct'

    # set the absolute path the experiment
    LMRpath = '/home/chaos2/wperkins/data/LMR'

    # set clean_start to True to delete existing files in the outpout directory
    # (otherwise they will be used as the prior!)
    clean_start = True

    # Reconstruction period (years)
    recon_period = [1000, 2000]

    # Ensemble size
    Nens = 100

    # Fraction of available proxy data (sites) to assimilate
    # (=1.0 for all, 0.5 for half etc.)
    proxy_frac = 0.75

    # Number of Monte-Carlo iterations
    iter_range = [0,10]

    # Localization radius for DA (in km)
    locRad = None
    #locRad = 2000.0
    #locRad = 10000.0

# =============================================================================
# Section 2: PROXIES
# =============================================================================

class proxies:
    # Proxy data directory & file
    datadir_proxy    = core.LMRpath+'/proxies'
    datafile_proxy   = 'Pages2k_db_metadata.df'
    dataformat_proxy = 'DF'

    regions = ['Antarctica', 'Arctic', 'Asia', 'Australasia', 'Europe',
               'North America', 'South America']

    # Proxy temporal resolution (in yrs)
    # proxy_resolution = [1.0,5.0]
    proxy_resolution = [1.0]

    # Define proxies to be assimilated
    # Use dictionary with structure: {<<proxy type>>:[... list of all measurement tags ...]}
    # where "proxy type" is written as "<<archive type>>_<<measurement type>>"
    # DO NOT CHANGE FORMAT BELOW
    proxy_assim = {
        '01:Tree ring_Width': ['Ring width','Tree ring width','Total ring width','TRW'],
        '02:Tree ring_Density': ['Maximum density','Minimum density','Earlywood density','Latewood density','MXD'],
        '03:Ice core_d18O': ['d18O'],
        '04:Ice core_d2H': ['d2H'],
        '05:Ice core_Accumulation':['Accumulation'],
        '06:Coral_d18O': ['d18O'],
        '07:Coral_Luminescence':['Luminescence'],
        '08:Lake sediment_All':['Varve thickness','Thickness','Mass accumulation rate','Particle-size distribution','Organic matter','X-ray density'],
        '09:Marine sediment_All':['Mg/Ca'],
        '10:Speleothem_All':['Lamina thickness'],
        }

    dat_filters = {'PAGES 2k Region': regions,
                   'Resolution (yr)': proxy_resolution,
                   'Proxy measurement': proxy_assim}

    # Source of calibration data (for PSM)
    datatag_calib = 'GISTEMP'
    #datatag_calib = 'HadCRUT'
    #datatag_calib = 'BerkeleyEarth'
    datadir_calib = core.LMRpath+'/analyses'

    # Threshold correlation of linear PSM
    PSM_r_crit = 0.2

# =============================================================================
# Section 3: MODEL (PRIOR)
# =============================================================================

class prior:
    # Prior data directory & model source
    prior_source = 'ccsm4_last_millenium'
    datadir_prior = '/home/chaos2/wperkins/data/'
    datafile_prior = 'tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc'
    dataformat_prior = 'NCD'

    # Define variables in state vector (will be updated by assimilation)
    state_variables = ['Tsfc']
    #state_variables = ['Tsfc', 'h500']

# =============================================================================
# Section 4: OUTPUT
# =============================================================================

# Output
#datadir_output  = '/home/disk/kalman3/rtardif/LMR/output'
#datadir_output  = '/home/disk/ekman/rtardif/nobackup/LMR/output'
datadir_output = '/home/chaos2/wperkins/data/LMR/output'
