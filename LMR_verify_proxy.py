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
           - Addded the use of NCDC proxy database as proxy data input.
             [R. Tardif, U. of Washington, March 2016]
"""
import os
import numpy as np
import cPickle    
from time import time
from os.path import join
# LMR specific imports
import LMR_proxy_pandas_rework
import LMR_calibrate
import LMR_prior
from LMR_utils import haversine, coefficient_efficiency

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.basemap import Basemap

# =========================================================================================
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

# =========================================================================================
# START:  set user parameters here
# =========================================================================================

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

class v_core:
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
    nexp = 't2_2k_CCSM4_LastMillenium_ens100_cGISTEMP_NCDCproxiesPagesTrees_pf0.75'
    #nexp = 't2_2k_CCSM4_LastMillenium_ens100_cGISTEMP_NCDCproxiesBreitTrees_pf0.75'

    # lmr_path: where all the data is located ... model (prior), analyses (GISTEMP, HAdCRUT...) and proxies.
    #lmr_path = '/home/chaos2/wperkins/data/LMR'
    lmr_path = '/home/disk/kalman3/rtardif/LMR'
    #verif_period = [0, 2000]
    #verif_period = [0, 1849]
    #verif_period = [1700,1849]
    verif_period = [0, 1879]
    #verif_period = [1880, 2000]

    iter_range = [0, 100]

    # Input directory, where to find the reconstruction data
    datadir_input  = '/home/disk/kalman3/rtardif/LMR/output'
    #datadir_input  = '/home/disk/kalman3/hakim/LMR/'
    #datadir_input  = '/home/disk/kalman2/wperkins/LMR_output/archive' # production recons

    # Output directory, where the verification results & figs will be dumped.
    datadir_output = datadir_input # if want to keep things tidy
    #datadir_output = '/home/disk/ekman/rtardif/kalman3/LMR/output/verification_production_runs'

    # Whether to write the full eval python dictionary to output directory ("summary" dict. has most of what we want)
    write_full_verif_dict = False

class v_proxies:
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

        datadir_proxy = join(v_core.lmr_path, 'data', 'proxies')
        datafile_proxy = join(datadir_proxy,
                              'Pages2k_Proxies.df.pckl')
        metafile_proxy = join(datadir_proxy,
                              'Pages2k_Metadata.df.pckl')
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

        # Create mapping for Proxy Type/Measurement Type to type names above
        proxy_type_mapping = {}
        for type, measurements in proxy_assim2.iteritems():
            # Fetch proxy type name that occurs before underscore
            type_name = type.split('_', 1)[0]
            for measure in measurements:
                proxy_type_mapping[(type_name, measure)] = type

        simple_filters = {'PAGES 2k Region': regions,
                          'Resolution (yr)': proxy_resolution}

        # A blacklist on proxy records, to prevent assimilation of chronologies known to be duplicates
        proxy_blacklist = []

    # ---------------
    # NCDC proxies
    # ---------------
    class NCDC:
        """
        Parameters for NCDCProxy class

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

        datadir_proxy = join(v_core.lmr_path, 'data', 'proxies')
        datafile_proxy = join(datadir_proxy,
                              'NCDC_Proxies.df.pckl')
        metafile_proxy = join(datadir_proxy,
                              'NCDC_Metadata.df.pckl')
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
#            'Corals and Sclerosponges_d14C': ['d14C','d14c','ac_d14c'],
#            'Corals and Sclerosponges_d13C': ['d13C','d13c','d13c_ave','d13c_ann_ave','d13C_int'],
            'Corals and Sclerosponges_SrCa': ['Sr/Ca','Sr/Ca_norm','Sr/Ca_anom','Sr/Ca_int'],
#            'Corals and Sclerosponges_Sr'  : ['Sr'],
#            'Corals and Sclerosponges_BaCa': ['Ba/Ca'],
#            'Corals and Sclerosponges_CdCa': ['Cd/Ca'],
#            'Corals and Sclerosponges_MgCa': ['Mg/Ca'],
#            'Corals and Sclerosponges_UCa' : ['U/Ca','U/Ca_anom'],
#            'Corals and Sclerosponges_Pb'  : ['Pb'],
            'Ice Cores_d18O'               : ['d18O','delta18O','delta18o','d18o','d18o_int','d18O_int','d18O_norm',
                                              'd18o_norm','dO18','d18O_anom'],
            'Ice Cores_dD'                 : ['deltaD','delD'],
            'Ice Cores_Accumulation'       : ['accum','accumu'],
            'Ice Cores_MeltFeature'        : ['MFP'],
            'Lake Cores_Varve'             : ['varve', 'varve_thickness', 'varve thickness'],
#            'Speleothems_d18O'             : ['d18O'],
#            'Speleothems_d13C'             : ['d13C'],
#            'Tree Rings_All'               : ['clim_signal'],
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

        # Create mapping for Proxy Type/Measurement Type to type names above
        proxy_type_mapping = {}
        for type, measurements in proxy_assim2.iteritems():
            # Fetch proxy type name that occurs before underscore
            type_name = type.split('_', 1)[0]
            for measure in measurements:
                proxy_type_mapping[(type_name, measure)] = type

        #simple_filters = {'NCDC Region': regions,
        #                  'Resolution (yr)': proxy_resolution}
        simple_filters = {'Resolution (yr)': proxy_resolution}

        # A blacklist on proxy records, to prevent assimilation of chronologies known to be duplicates.
        # An empty list will 
        proxy_blacklist = []
        #proxy_blacklist = ['00aust01a', '06cook02a', '06cook03a', '09japa01a', '10guad01a', '99aust01a', '99fpol01a']


class v_psm:
    """
    Parameters for PSM classes

    Attributes
    ----------
    use_psm: dict{str: str}
        Maps proxy class key to psm class key.  Used to determine which psm
        is associated with what Proxy type.
    """

    use_psm = {'pages': 'linear', 'NCDC': 'linear'}

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

        datadir_calib = join(v_core.lmr_path, 'data', 'analyses')
        dataformat_calib = 'NCD'
        pre_calib_datafile = join(v_core.lmr_path,
                                  'PSM',
                                  'PSMs_'+'-'.join(v_proxies.use_from)+'_' + datatag_calib + '.pckl')

        psm_r_crit = 0.0


# =========================================================================================
# END:  set user parameters here
# =========================================================================================
class config:
    def __init__(self,core,proxies,psm):
        self.core = core
        self.proxies = proxies
        self.psm = psm

Cfg = config(v_core, v_proxies,v_psm)


#==========================================================================================
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def recon_proxy_eval_stats(sites_eval,mode,proxy_dict,Xrecon,Xprior,recondir,verif_period):
#==========================================================================================
# 
# 
#   Input : 
#         - sites_eval    : Dictionary containing list of sites (proxy chronologies) per
#                           proxy type
#         - mode          : "assim" or "verif" to distinguish if we are working with
#                           assimilated or withheld proxies
#         - proxy_dict    : Dictionary containing proxy objects (proxy & PSM data)
#         - Xrecon        : Info/data of reconstruction (ensemble-mean)
#         - Xprior        : Prior data array (full ensemble)
#         - recondir      : Directory where the reconstruction data is located
#         - verif_period  : Period over which verification is performed. Format: [start,end]
#
#   Output: ... list of dictionaries ...
# 
#==========================================================================================
    
    # Output dictionary
    evald = {}

    # Build list of proxy types in verification set
    proxy_types = []
    for k in range(len(sites_eval)):
        key = sites_eval[k].keys()
        proxy_types.append(key[0])
    proxy_types_eval = list(set(proxy_types))

    proxy_list = [item for item in sites_eval] # a list of dictionaries

    # For information only
    print '-----------------------------------------------------------------------'
    print 'Sites to be processed: '
    totalsites = 0
    for proxy_key in proxy_types_eval:
        tmplist = [d.keys()[0] for d in proxy_list]
        nbsites = len([item for item in tmplist if item == proxy_key])
        print('%45s : %5d' % (proxy_key,nbsites))
        totalsites = totalsites + nbsites
    print('%45s : %5d' %('TOTAL', totalsites))
    print '-----------------------------------------------------------------------'
    print ' '

    # Reconstruction
    recon_times = Xrecon['years']
    recon_times = map(float,recon_times)
    recon_lat = Xrecon['lat']
    recon_lon = Xrecon['lon']
    recon_tas =  Xrecon['xam']

    # if "assim" mode, load full ensemble of Ye's corresponding to assimilated proxies
    if mode == 'assim':
        fnameYe = recondir+'/'+'analysis_Ye.pckl'
        infile  = open(fnameYe,'r')
        assimYe = cPickle.load(infile)
        infile.close()

    # Loop over proxy sites in eval. set
    sitecount = 0
    for proxy in proxy_list:

        sitecount = sitecount + 1

        ptype = proxy.keys()[0]
        psite = proxy[ptype][0]
        sitetag = (ptype, psite)
        Yeval = proxy_dict[sitetag]
        
        print 'Site:', Yeval.type, ':', psite, '=> nb', sitecount, 'out of', totalsites, '(',(np.float(sitecount)/np.float(totalsites))*100,'% )'
        print ' latitude, longitude: ' + str(Yeval.lat), str(Yeval.lon)

        Ntime = len(Yeval.time)
        if Ntime < 10: # if no significant nb of obs uploaded/retained, move to next proxy site
            continue

        evald[sitetag] = {}
        evald[sitetag]['lat'] = Yeval.lat
        evald[sitetag]['lon'] = Yeval.lon
        evald[sitetag]['alt'] = 0.0 # unknown variable in PAGES2k set
        
        # Set up arrays
        Xrecon_error       = np.zeros(shape=[Ntime])
        truth              = np.zeros(shape=[Ntime]) 
        Ye_recon_EnsMean   = np.zeros(shape=[Ntime]) 
        Ye_recon_EnsSpread = np.zeros(shape=[Ntime]) 

        Ye_prior_EnsMean   = np.zeros(shape=[Ntime]) 
        [_,_,Nens] = Xprior.shape
        Ye_prior_FullEns   = np.zeros(shape=[Ntime,Nens]) 

        # closest lat/lon in recon grid to proxy site
        a = abs( recon_lat-Yeval.lat ) + abs( recon_lon-Yeval.lon )
        i,j = np.unravel_index(a.argmin(), a.shape)
        dist = haversine(Yeval.lon,Yeval.lat,recon_lon[i,j],recon_lat[i,j])
        #print Yeval.lat, Yeval.lon, i, j, recon_lat[i,j], recon_lon[i,j], dist

        # Loop over time in proxy record ----
        obcount = 0
        for t in Yeval.time:
            truth[obcount] = Yeval.values[t]

            # array time index for recon_values
            indt_recon = recon_times.index(t)

            # calculate Ye (reconstruction & prior)
            Ye_recon_EnsMean[obcount] = Yeval.psm_obj.slope*recon_tas[indt_recon,i,j] + Yeval.psm_obj.intercept
            #print obcount, t, truth[obcount], Ye_recon_EnsMean[obcount]

            Xprior_EnsMean = np.mean(Xprior, axis=2) # Prior ensemble mean
            Ye_prior_EnsMean[obcount] = Yeval.psm_obj.slope*Xprior_EnsMean[i,j] + Yeval.psm_obj.intercept
            Ye_prior_FullEns[obcount,:] = Yeval.psm_obj.slope*Xprior[i,j,:] + Yeval.psm_obj.intercept

            # Ensemble-mean reconstruction error
            Xrecon_error[obcount] = (Ye_recon_EnsMean[obcount] - truth[obcount])

            obcount = obcount + 1


        # -----------------------------------

        if obcount > 0:
            print '================================================'
            print 'Site:', Yeval.type, ':', psite
            print 'Number of verification points:', obcount            
            print 'Mean of proxy values         :', np.mean(truth)
            print 'Mean ensemble-mean           :', np.mean(Ye_recon_EnsMean)
            print 'Mean ensemble-mean error     :', np.mean(Ye_recon_EnsMean-truth)
            print 'Ensemble-mean RMSE           :', rmse(Ye_recon_EnsMean,truth)
            print 'Correlation                  :', np.corrcoef(truth,Ye_recon_EnsMean)[0,1]
            print 'CE                           :', coefficient_efficiency(truth,Ye_recon_EnsMean)
            print 'Correlation (prior)          :', np.corrcoef(truth,Ye_prior_EnsMean)[0,1]
            print 'CE (prior)                   :', coefficient_efficiency(truth,Ye_prior_EnsMean)
            print '================================================'            

            # Fill dictionary with data generated for evaluation of reconstruction
            # PSM info
            evald[sitetag]['PSMslope']          = Yeval.psm_obj.slope
            evald[sitetag]['PSMintercept']      = Yeval.psm_obj.intercept
            evald[sitetag]['PSMcorrel']         = Yeval.psm_obj.corr
            evald[sitetag]['PSMmse']            = Yeval.psm_obj.R
            # Verif. data
            evald[sitetag]['NbEvalPts']         = obcount
            evald[sitetag]['EnsMean_MeanError'] = np.mean(Ye_recon_EnsMean-truth)
            evald[sitetag]['EnsMean_RMSE']      = rmse(Ye_recon_EnsMean,truth)
            evald[sitetag]['EnsMean_Corr']      = np.corrcoef(truth,Ye_recon_EnsMean)[0,1]
            evald[sitetag]['EnsMean_CE']        = coefficient_efficiency(truth,Ye_recon_EnsMean)
            evald[sitetag]['PriorEnsMean_Corr'] = np.corrcoef(truth,Ye_prior_EnsMean)[0,1]
            evald[sitetag]['PriorEnsMean_CE']   = coefficient_efficiency(truth,Ye_prior_EnsMean)
            evald[sitetag]['ts_years']          = list(Yeval.time)
            evald[sitetag]['ts_ProxyValues']    = truth
            evald[sitetag]['ts_EnsMean']        = Ye_recon_EnsMean

            if mode == 'assim':
                R = assimYe[sitetag]['R']
                [_,Nens] = assimYe[sitetag]['HXa'].shape
                YeFullEns = np.zeros(shape=[Ntime,Nens])
                YeFullEns_error = np.zeros(shape=[Ntime,Nens])
                YePriorFullEns_error = np.zeros(shape=[Ntime,Nens])

                Ye_time = assimYe[sitetag]['years']
                obcount = 0
                for t in Yeval.time:
                    truth[obcount] = Yeval.values[t]
                    # array time index for recon_values
                    indt_recon = np.where(Ye_time==t)
                    YeFullEns[obcount,:] = assimYe[sitetag]['HXa'][indt_recon]
                    YeFullEns_error[obcount,:] = YeFullEns[obcount,:]-truth[obcount,None] # i.e. full ensemble innovations
                    YePriorFullEns_error[obcount,:] = Ye_prior_FullEns[obcount,:]-truth[obcount,None]
                    obcount = obcount + 1
                mse = np.mean(np.square(YeFullEns_error),axis=1)
                varYe = np.var(YeFullEns,axis=1,ddof=1)
                # time series of DA ensemble calibration ratio
                evald[sitetag]['ts_DAensCalib']   =  mse/(varYe+R)

                # Prior
                mse = np.mean(np.square(YePriorFullEns_error),axis=1)
                varYe = np.var(Ye_prior_FullEns,axis=1,ddof=1)
                # time series of prior ensemble calibration ratio
                evald[sitetag]['ts_PriorEnsCalib']   =  mse/(varYe+R)


    return evald


# =============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Main code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# =============================================================================
def main():

    begin_time = time()

    print Cfg.proxies.pages.datadir_proxy
    print Cfg.core.verif_period
    #verif_times = list(range(Cfg.core.verif_period[0],Cfg.core.verif_period[1]+1))

    # Make sure proxy_frac is set to 1.0 to read in all proxies
    Cfg.proxies.proxy_frac = 1.0


    prox_manager = LMR_proxy_pandas_rework.ProxyManager(Cfg, Cfg.core.verif_period)
    type_site_assim = prox_manager.assim_ids_by_group

    print '--------------------------------------------------------------------'
    print 'Uploaded proxies : counts per proxy type:'
    # count the total number of proxies
    total_proxy_count = len(prox_manager.ind_assim)
    for pkey, plist in type_site_assim.iteritems():
        print('%45s : %5d' % (pkey, len(plist)))
    print('%45s : %5d' % ('TOTAL', total_proxy_count))
    print '--------------------------------------------------------------------'
    master_proxy_list = []
    proxy_dict = {}
    for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
        sitetag = (Y.type,Y.id)
        master_proxy_list.append(sitetag)
        # Load proxy object in proxy dictionary
        proxy_dict[sitetag] = Y

    load_time = time() - begin_time
    print '======================================================='
    print 'Loading completed in '+ str(load_time/60.0)+' mins'
    print '======================================================='


    # ==========================================================================
    # Loop over the Monte-Carlo reconstructions
    # ==========================================================================
    nexp = Cfg.core.nexp
    datadir_input = Cfg.core.datadir_input
    verif_period = Cfg.core.verif_period
    datatag_calib = Cfg.psm.linear.datatag_calib
    Nbiter = Cfg.core.iter_range[1]
    datadir_output = Cfg.core.datadir_output

    verif_dict = []
    assim_dict = []

    MCiters = np.arange(Cfg.core.iter_range[0], Cfg.core.iter_range[1]+1)
    for iter in MCiters:

        # Experiment data directory
        workdir = datadir_input+'/'+nexp+'/r'+str(iter)
        print workdir

        print '============================================================'
        print 'Working on: ' + nexp + ' : ' + '/r' + str(iter)
        print '============================================================'

        # Check for presence of file containing reconstruction of **surface temperature**
        filein = workdir+'/ensemble_mean_tas_sfc_Amon.npz'
        if not os.path.isfile(filein):
            print 'ERROR in specification of reconstruction data'
            print 'File ', filein, ' does not exist! - Exiting!'
            exit(1)

        # =============================================
        # Load in the ensemble-mean reconstruction data
        # =============================================
        Xrecon = np.load(filein)

        # ======================
        # Load in the prior data
        # ======================
        file_prior = workdir+'/Xb_one.npz'
        Xprior_statevector = np.load(file_prior)
        # extract sfc temperature from state vector
        state_info = Xprior_statevector['state_info'].item()  
        posbeg = state_info['tas_sfc_Amon']['pos'][0]
        posend = state_info['tas_sfc_Amon']['pos'][1]
        Xb_one = Xprior_statevector['Xb_one']
        
        nlat = state_info['tas_sfc_Amon']['spacedims'][0]
        nlon = state_info['tas_sfc_Amon']['spacedims'][1]
        tas_prior = Xb_one[posbeg:posend+1,:]
        [_,Nens] = tas_prior.shape
        Xprior = tas_prior.reshape(nlat,nlon,Nens)
        
        # ============================================================================
        # Proxy site-based statistics on reconstruction fit to non-assimilated proxies 
        # ============================================================================

        # List of non-assimilated proxies
        loaded_proxies = np.load(workdir+'/'+'nonassimilated_proxies.npy')
        nb_loaded_proxies = len(loaded_proxies)
        
        # Filter to match proxy sites in set of calibrated PSMs
        indp = [k for k in range(nb_loaded_proxies) if (loaded_proxies[k].keys()[0],loaded_proxies[k][loaded_proxies[k].keys()[0]][0]) in master_proxy_list]
        verif_proxies = loaded_proxies[indp]
        nb_verif_proxies = len(verif_proxies)

        print '------------------------------------------------'
        print 'Number of proxy sites in verification set:',  nb_verif_proxies
        print '------------------------------------------------'

        # Build list of proxy types in verification set
        verif_types = []
        for k in range(nb_verif_proxies):
            key = verif_proxies[k].keys()
            verif_types.append(key[0])
        verif_types_list = list(set(verif_types))

        # Calculate reconstruction error statistics & output in "verif_dict" dictionary
        out_dict = recon_proxy_eval_stats(verif_proxies,'verif',proxy_dict,Xrecon,Xprior,workdir,verif_period)
        verif_dict.append(out_dict)

        # ==========================================================================
        # Proxy site-based statistics on reconstruction fit to assimilated proxies 
        # ==========================================================================

        # List of assimilated proxies
        loaded_proxies = np.load(workdir+'/'+'assimilated_proxies.npy')
        nb_loaded_proxies = len(loaded_proxies)

        # Filter to match proxy sites in set of calibrated PSMs
        indp = [k for k in range(nb_loaded_proxies) if (loaded_proxies[k].keys()[0],loaded_proxies[k][loaded_proxies[k].keys()[0]][0]) in master_proxy_list]
        assim_proxies = loaded_proxies[indp]
        nb_assim_proxies = len(assim_proxies)

        print '------------------------------------------------'
        print 'Number of proxy sites in assimilated set:',  nb_assim_proxies
        print '------------------------------------------------'

        # Calculate reconstruction error statistics & output in "assim_dict" dictionary
        out_dict = recon_proxy_eval_stats(assim_proxies,'assim',proxy_dict,Xrecon,Xprior,workdir,verif_period)
        assim_dict.append(out_dict)

    # ==========================================================================
    # End of loop on iterations => Now calculate summary statistics ------------
    # ==========================================================================

    outdir = datadir_output+'/'+nexp+'/verifProxy_PSMcalib'+datatag_calib+'_'+str(verif_period[0])+'to'+str(verif_period[1])
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
        summary_stats_verif[list_sites[k]]['PSMslope']       = verif_dict[inds[0]][list_sites[k]]['PSMslope']
        summary_stats_verif[list_sites[k]]['PSMintercept']   = verif_dict[inds[0]][list_sites[k]]['PSMintercept']
        summary_stats_verif[list_sites[k]]['PSMcorrel']      = verif_dict[inds[0]][list_sites[k]]['PSMcorrel']
        summary_stats_verif[list_sites[k]]['NbPts']          = len(inds)
        
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
        R = [verif_dict[inds[0]][list_sites[k]]['PSMmse']]
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
        summary_stats_assim[list_sites[k]]['PSMslope']       = assim_dict[inds[0]][list_sites[k]]['PSMslope']
        summary_stats_assim[list_sites[k]]['PSMintercept']   = assim_dict[inds[0]][list_sites[k]]['PSMintercept']
        summary_stats_assim[list_sites[k]]['PSMcorrel']      = assim_dict[inds[0]][list_sites[k]]['PSMcorrel']
        summary_stats_assim[list_sites[k]]['NbPts']          = len(inds)

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
        R = [assim_dict[inds[0]][list_sites[k]]['PSMmse']]
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
    print 'Verification completed in '+ str(verif_time/60.0)+' mins'
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
    print 'All completed in '+ str(end_time/60.0)+' mins'
    print '======================================================='



# =============================================================================

if __name__ == '__main__':
    main()
