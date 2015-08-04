
#==========================================================================================
# Contains functions used to evaluate the LMR paleoclimate reconstructions on the basis of
# proxy data set aside for verification (non-assimilated).
# Error statistics of the reconstructions are evaluated using an independent set of proxy 
# chronologies. 
# 
# Originators: Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
#                            | March 2015
#
# Revisions:
#           - From original LMR_diagnostics_proxy code, adapted to new reconstruction 
#             output (ensemble mean only) and use of pre-built PSMs (R. Tardif, July 2015)
#
#========================================================================================== 

import os
import numpy as np
import numpy as np
import cPickle    
from time import time

import LMR_proxy
import LMR_calibrate
import LMR_prior
from load_proxy_data import read_proxy_metadata_S1csv as read_proxy_metadata
from LMR_utils import haversine, coefficient_efficiency

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# =========================================================================================
# START:  set user parameters here
# =========================================================================================

# ------------------------------
# Section 1: Plotting parameters
# ------------------------------

make_plots = True

# set the default size of the figure in inches. ['figure.figsize'] = width, height;  
plt.rcParams['figure.figsize'] = 9, 7  # that's default image size for this interactive session
plt.rcParams['axes.linewidth'] = 2.0 #set the value globally
plt.rcParams['font.weight'] = 'bold' #set the font weight globally
#plt.rc('text', usetex=True)
plt.rc('text', usetex=False)

# -------------------------------------------------------------
# Section 2: High-level parameters of reconstruction experiment
# -------------------------------------------------------------

# Name of reconstruction experiment to verify
nexp = 'ReconMultiState_ens100_allAnnualProxyTypes_pf0.5'

# Run diagnostics over this range of Monte-Carlo reconstructions
iter_range = [0,100]

# Reconstruction period (years)
recon_period = [1850,2000]

# set the absolute path the experiment (could make this cwd with some os coding)
LMRpath = '/home/disk/kalman3/rtardif/LMR'

# Input directory, where to find the reconstruction data
datadir_input  = '/home/disk/kalman3/rtardif/LMR/output'

# ------------------
# Section 3: Proxies
# ------------------

# Proxy data directory & file
datadir_proxy    = LMRpath+'/data/proxies';
datafile_proxy   = 'Pages2k_DatabaseS1-All-proxy-records.xlsx';

# Define proxies types to be used in verification
proxy_verif = [\
    'Tree ring_Width',\
    'Tree ring_Density',\
    'Ice core_d18O',\
    'Ice core_d2H',\
    'Ice core_Accumulation',\
    'Coral_d18O',\
    'Coral_Luminescence',\
    'Lake sediment_All',\
    'Marine sediment_All',\
    'Speleothem_All',\
    ]

# Regions where proxy sites are located (only for PAGES2K dataset) 
regions = ['Antarctica','Arctic','Asia','Australasia','Europe','North America','South America']

# Proxy temporal resolution (in yrs)
proxy_resolution = [1.0]

# Source of calibration data (for PSM)
datatag_calib = 'GISTEMP'
#datatag_calib = 'HadCRUT'
#datatag_calib = 'BerkeleyEarth'

datadir_calib = LMRpath+'/data/analyses';

# Threshold correlation of linear PSM 
PSM_r_crit = 0.2

# =========================================================================================
# END:  set user parameters here
# =========================================================================================

#==========================================================================================
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def recon_proxy_eval_stats(sites_eval,proxy_dict,Xrecon,psm_data,recondir):
#==========================================================================================
# 
# 
#   Input : 
#         - sites_eval    : Dictionary containing list of sites (proxy chronologies) per
#                           proxy type
#         - proxy_dict    : Dictionary containing proxy objects (proxy data)
#         - Xrecon        : Info/data of reconstruction
#         - psm_data      : Calibrated PSM parameters for each proxy site
#         - recondir      : Directory where the reconstruction data is located
#
#   Output: ...
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


    # Loop over proxy sites in eval. set
    sitecount = 0
    for proxy in proxy_list:

        sitecount = sitecount + 1

        ptype = proxy.keys()[0]
        psite = proxy[ptype][0]
        sitetag = (ptype, psite)
        Yeval = proxy_dict[sitetag]
        
        print 'Site:', Yeval.proxy_type, ':', psite, '=> nb', sitecount, 'out of', totalsites, '(',(np.float(sitecount)/np.float(totalsites))*100,'% )'
        print ' latitude, longitude, altitude: ' + str(Yeval.lat), str(Yeval.lon), str(Yeval.alt)

        if Yeval.nobs == 0: # if no obs uploaded, move to next proxy site
                continue

        evald[sitetag] = {}
        evald[sitetag]['lat'] = Yeval.lat
        evald[sitetag]['lon'] = Yeval.lon
        evald[sitetag]['alt'] = Yeval.alt
        
        # indices of proxy ob that overlap with recon. period
        indices = [j for j, t in enumerate(Yeval.time) if t in recon_times]
        Ntime = len(indices)
        
        if Ntime == 0: # if no obs uploaded, move to next proxy site
            continue

        # Set up arrays
        Xrecon_error       = np.zeros(shape=[Ntime])
        truth              = np.zeros(shape=[Ntime]) 
        Ye_recon_EnsMean   = np.zeros(shape=[Ntime]) 
        Ye_recon_EnsSpread = np.zeros(shape=[Ntime]) 


        # Load in corresponding psm data
        # does a pre-built PSM exist for this site?
        if sitetag in psm_data.keys():
            psm_params = psm_data[sitetag]
        else:
            print 'Cannot find pre-built PSM for site:', sitetag
            exit(1) # just exit for now ...

        #print sitetag, psm_params, psm_params['PSMslope'], psm_params['PSMintercept']

        # closest lat/lon in recon grid to proxy site
        a = abs( recon_lat-Yeval.lat ) + abs( recon_lon-Yeval.lon )
        i,j = np.unravel_index(a.argmin(), a.shape)
        dist = haversine(Yeval.lon,Yeval.lat,recon_lon[i,j],recon_lat[i,j])
        #print Yeval.lat, Yeval.lon, i, j, recon_lat[i,j], recon_lon[i,j], dist

        # Loop over time in proxy record
        obcount = 0
        for t in [Yeval.time[k] for k in indices]:

            indt = Yeval.time.index(t)
            truth[obcount] = Yeval.value[indt]

            # array time index for recon_values
            indt_recon = recon_times.index(t)

            # calculate Ye
            Ye_recon_EnsMean[obcount] = psm_params['PSMslope']*recon_tas[indt_recon,i,j] + psm_params['PSMintercept']
            #print obcount, t, truth[obcount], Ye_recon_EnsMean[obcount]

            # Ensemble-mean reconstruction error
            Xrecon_error[obcount] = (Ye_recon_EnsMean[obcount] - truth[obcount])

            obcount = obcount + 1


        if obcount > 0:

            print '================================================'
            print 'Site:', Yeval.proxy_type, ':', psite
            print 'Number of verification points:', obcount            
            print 'Mean of proxy values         :', np.mean(truth)
            print 'Mean ensemble-mean           :', np.mean(Ye_recon_EnsMean)
            print 'Mean ensemble-mean error     :', np.mean(Ye_recon_EnsMean-truth)
            print 'Ensemble-mean RMSE           :', rmse(Ye_recon_EnsMean,truth)
            print 'Correlation                  :', np.corrcoef(truth,Ye_recon_EnsMean)[0,1]
            print 'CE                           :', coefficient_efficiency(truth,Ye_recon_EnsMean)
            print '================================================'            

            # Fill dictionary with data generated for evaluation of reconstruction
            # PSM info
            evald[sitetag]['PSMslope']          = psm_params['PSMslope']
            evald[sitetag]['PSMintercept']      = psm_params['PSMintercept']
            evald[sitetag]['PSMcorrel']         = psm_params['PSMcorrel']
            evald[sitetag]['PSMmse']            = psm_params['PSMmse']
            # Verif. data
            evald[sitetag]['NbEvalPts']         = obcount
            evald[sitetag]['EnsMean_MeanError'] = np.mean(Ye_recon_EnsMean-truth)
            evald[sitetag]['EnsMean_RMSE']      = rmse(Ye_recon_EnsMean,truth)
            evald[sitetag]['EnsMean_Corr']      = np.corrcoef(truth,Ye_recon_EnsMean)[0,1]
            evald[sitetag]['EnsMean_CE']        = coefficient_efficiency(truth,Ye_recon_EnsMean)
            evald[sitetag]['ts_years']          = [Yeval.time[k] for k in indices]
            evald[sitetag]['ts_ProxyValues']    = truth
            evald[sitetag]['ts_EnsMean']        = Ye_recon_EnsMean

    return evald


# =============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Main code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# =============================================================================
def main():

    begin_time = time()

    # Load pre-calibrated PSM parameters 
    fnamePSM = LMRpath+'/PSM/PSMs_'+datatag_calib+'.pckl'
    infile   = open(fnamePSM,'rb')
    psm_data = cPickle.load(infile)
    infile.close()

    # Build list of proxy types/sites to be processed
    keys = psm_data.keys()
    if PSM_r_crit:
        # Master list of proxies, filtered to keep sites that match the r_crit criteria (PSM r >= 0.2)
        master_proxy_list = [item for item in keys if abs(psm_data[item]['PSMcorrel']) >= PSM_r_crit and item[0] in proxy_verif]
    else:
        # Master list of all available proxies (included in PSM file)
        master_proxy_list = [item for item in keys if item[0] in proxy_verif]
    print 'Nb of proxies to upload:', len(master_proxy_list)
    
    # Build list of proxy types in verification set
    master_proxy_types = list(set([item[0] for item in master_proxy_list]))

    # For information only
    print '-----------------------------------------------------------------------'
    print 'Proxy sites per proxy type: '
    totalsites = 0
    for proxy_key in master_proxy_types:
        nbsites = len([item for item in master_proxy_list if item[0] == proxy_key])
        print('%45s : %5d' % (proxy_key,nbsites))
        totalsites = totalsites + nbsites
    print('%45s : %5d' %('TOTAL', totalsites))
    print '-----------------------------------------------------------------------'
    print ' '

    # Read in all available proxies
    proxy_dict = {}
    sitecount = 0
    for proxy in master_proxy_list:
        ptype = proxy[0]
        psite = proxy[1]
        sitetag = (ptype, psite)
        sitecount = sitecount + 1
        print 'Loading data for:', sitetag, '=> nb', sitecount, 'out of', totalsites, '(',(np.float(sitecount)/np.float(totalsites))*100,'% )'

        Y = LMR_proxy.proxy_assignment(ptype)
        # add namelist attributes to the proxy object
        Y.proxy_datadir  = datadir_proxy
        Y.proxy_datafile = datafile_proxy
        Y.proxy_region   = regions
        # read the data
        Y.read_proxy(psite)
        
        # Load proxy object in proxy dictionary
        proxy_dict[sitetag] = Y


    load_time = time() - begin_time
    print '======================================================='
    print 'Loading completed in '+ str(load_time/60.0)+' mins'
    print '======================================================='


    # ==========================================================================
    # Loop over the Monte-Carlo reconstructions
    # ==========================================================================
    verif_dict = []
    assim_dict = []

    MCiters = np.arange(iter_range[0], iter_range[1]+1)
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

        # Calculate reconstruction error statistics & output in "assim_dict" dictionary
        out_dict = recon_proxy_eval_stats(verif_proxies,proxy_dict,Xrecon,psm_data,workdir)
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
        out_dict = recon_proxy_eval_stats(assim_proxies,proxy_dict,Xrecon,psm_data,workdir)
        assim_dict.append(out_dict)

    # ==========================================================================
    # End of loop on iterations => Now calculate summary statistics
    # ==========================================================================

    outdir = datadir_input+'/'+nexp+'/verifProxy_PSMcalib'+datatag_calib
    if not os.path.isdir(outdir):
        os.system('mkdir %s' % outdir)
    
    # -----------------------
    # With *** verif_dict ***
    # -----------------------
    ## Dump dictionary to pickle files
    #outfile = open('%s/reconstruction_eval_verif_proxy_full.pckl' % (outdir),'w')
    #cPickle.dump(verif_dict,outfile)
    #outfile.close()

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

        nbpts = [verif_dict[j][list_sites[k]]['NbEvalPts'] for j in inds]
        me    = [verif_dict[j][list_sites[k]]['EnsMean_MeanError'] for j in inds]
        rmse  = [verif_dict[j][list_sites[k]]['EnsMean_RMSE'] for j in inds]
        corr  = [verif_dict[j][list_sites[k]]['EnsMean_Corr'] for j in inds]
        ce    = [verif_dict[j][list_sites[k]]['EnsMean_CE'] for j in inds]

        summary_stats_verif[list_sites[k]] = {}        
        summary_stats_verif[list_sites[k]]['lat']            = verif_dict[inds[0]][list_sites[k]]['lat']
        summary_stats_verif[list_sites[k]]['lon']            = verif_dict[inds[0]][list_sites[k]]['lon']
        summary_stats_verif[list_sites[k]]['alt']            = verif_dict[inds[0]][list_sites[k]]['alt']
        summary_stats_verif[list_sites[k]]['NbPts']          = len(inds)
        summary_stats_verif[list_sites[k]]['MeanME']         = np.mean(me)
        summary_stats_verif[list_sites[k]]['SpreadME']       = np.std(me)
        summary_stats_verif[list_sites[k]]['MeanRMSE']       = np.mean(rmse)
        summary_stats_verif[list_sites[k]]['SpreadRMSE']     = np.std(rmse)
        summary_stats_verif[list_sites[k]]['MeanCorr']       = np.mean(corr)
        summary_stats_verif[list_sites[k]]['SpreadCorr']     = np.std(corr)
        summary_stats_verif[list_sites[k]]['MeanCE']         = np.mean(ce)
        summary_stats_verif[list_sites[k]]['SpreadCE']       = np.std(ce)
        # for time series
        summary_stats_verif[list_sites[k]]['ts_years']       = verif_dict[inds[0]][list_sites[k]]['ts_years']
        summary_stats_verif[list_sites[k]]['ts_ProxyValues'] = verif_dict[inds[0]][list_sites[k]]['ts_ProxyValues']
        ts_recon = [verif_dict[j][list_sites[k]]['ts_EnsMean'] for j in inds]
        summary_stats_verif[list_sites[k]]['ts_MeanRecon']   = np.mean(ts_recon,axis=0)
        summary_stats_verif[list_sites[k]]['ts_SpreadRecon'] = np.std(ts_recon,axis=0)
        # Ens. calibration
        R = [verif_dict[inds[0]][list_sites[k]]['PSMmse']]
        ensVar = np.mean(np.var(ts_recon,axis=0,ddof=1))
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
    ## Dump dictionary to pickle files
    #outfile = open('%s/reconstruction_eval_assim_proxy_full.pckl' % (outdir),'w')
    #cPickle.dump(assim_dict,outfile)
    #outfile.close()

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

        nbpts = [assim_dict[j][list_sites[k]]['NbEvalPts'] for j in inds]
        me    = [assim_dict[j][list_sites[k]]['EnsMean_MeanError'] for j in inds]
        rmse  = [assim_dict[j][list_sites[k]]['EnsMean_RMSE'] for j in inds]
        corr  = [assim_dict[j][list_sites[k]]['EnsMean_Corr'] for j in inds]
        ce    = [assim_dict[j][list_sites[k]]['EnsMean_CE'] for j in inds]

        summary_stats_assim[list_sites[k]] = {}
        summary_stats_assim[list_sites[k]]['lat']            = assim_dict[inds[0]][list_sites[k]]['lat']
        summary_stats_assim[list_sites[k]]['lon']            = assim_dict[inds[0]][list_sites[k]]['lon']
        summary_stats_assim[list_sites[k]]['alt']            = assim_dict[inds[0]][list_sites[k]]['alt']
        summary_stats_assim[list_sites[k]]['NbPts']          = len(inds)
        summary_stats_assim[list_sites[k]]['MeanME']         = np.mean(me)
        summary_stats_assim[list_sites[k]]['SpreadME']       = np.std(me)
        summary_stats_assim[list_sites[k]]['MeanRMSE']       = np.mean(rmse)
        summary_stats_assim[list_sites[k]]['SpreadRMSE']     = np.std(rmse)
        summary_stats_assim[list_sites[k]]['MeanCorr']       = np.mean(corr)
        summary_stats_assim[list_sites[k]]['SpreadCorr']     = np.std(corr)
        summary_stats_assim[list_sites[k]]['MeanCE']         = np.mean(ce)
        summary_stats_assim[list_sites[k]]['SpreadCE']       = np.std(ce)
        # for time series
        summary_stats_assim[list_sites[k]]['ts_years']       = assim_dict[inds[0]][list_sites[k]]['ts_years']
        summary_stats_assim[list_sites[k]]['ts_ProxyValues'] = assim_dict[inds[0]][list_sites[k]]['ts_ProxyValues']
        ts_recon = [assim_dict[j][list_sites[k]]['ts_EnsMean'] for j in inds]
        summary_stats_assim[list_sites[k]]['ts_MeanRecon']   = np.mean(ts_recon,axis=0)
        summary_stats_assim[list_sites[k]]['ts_SpreadRecon'] = np.std(ts_recon,axis=0)        
        # Ens. calibration
        R = [assim_dict[inds[0]][list_sites[k]]['PSMmse']]
        ensVar = np.mean(np.var(ts_recon,axis=0,ddof=1))
        mse = np.mean(np.square(rmse))
        calib = mse/(ensVar+R)
        summary_stats_assim[list_sites[k]]['EnsCalib'] = calib[0]
        ## without R
        #calib = mse/(ensVar)
        #summary_stats_assim[list_sites[k]]['EnsCalib'] = calib
        

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
            
            dname = 'summary_stats_'+v
            workdict = eval(dname)
            sitetag = workdict.keys()

            site_status = vtype[v].split()[0]

            # ========================================================
            # 1) Histogram of (recon, proxy) CORRELATION
            # ========================================================
            stat = [workdict[k]['MeanCorr'] for k in sitetag]
            nbsites = len(stat)
            mean_stat = np.mean(stat)

            ptitle = '%s (Nb. sites= %d , Mean Corr=%.2f)' % (vtype[v],nbsites,mean_stat)
            n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=False)
            plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
            plt.title(ptitle)
            plt.xlabel("Correlation")
            plt.ylabel("Count")
            xmin,xmax,ymin,ymax = plt.axis()
            plt.axis((-1,1,ymin,ymax))

            # Annotate with summary stats
            xmin,xmax,ymin,ymax = plt.axis()
            ypos = ymax-0.05*(ymax-ymin)
            xpos = xmin+0.025*(xmax-xmin)
            plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
            
            plt.savefig('%s/recon_proxy_%s_stats_hist_corr.png' % (figdir,v),bbox_inches='tight')
            plt.close()


            # ========================================================
            # 2) Histogram of (recon, proxy) coeff. of efficiency (CE)
            # ========================================================
            stat = [workdict[k]['MeanCE'] for k in sitetag]
            nbsites = len(stat)
            mean_stat = np.mean(stat)

            ptitle = '%s (Nb. sites= %d , Mean CE=%.2f)' % (vtype[v],nbsites,mean_stat)
            n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=False)
            plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
            plt.title(ptitle)
            plt.xlabel("CE")
            plt.ylabel("Count")
            xmin,xmax,ymin,ymax = plt.axis()
            plt.axis((-1,1,ymin,ymax))

            # Annotate with summary stats
            xmin,xmax,ymin,ymax = plt.axis()
            ypos = ymax-0.05*(ymax-ymin)
            xpos = xmin+0.025*(xmax-xmin)
            plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')
            
            plt.savefig('%s/recon_proxy_%s_stats_hist_ce.png' % (figdir,v),bbox_inches='tight')
            plt.close()


            # =========================================================
            # 3) Histogram of reconstruction ensemble calibration ratio
            # =========================================================
            stat = [workdict[k]['EnsCalib'] for k in sitetag]
            nbsites = len(stat)
            mean_stat = np.mean(stat)

            ptitle = '%s (Nb. sites= %d , Mean ens. calib ratio=%.2f)' % (vtype[v],nbsites,mean_stat)
            n, bins, patches = plt.hist(stat, histtype='stepfilled',normed=False)
            plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
            plt.title(ptitle)
            plt.xlabel("Ensemble calibration ratio")
            plt.ylabel("Count")
            xmin,xmax,ymin,ymax = plt.axis()
            plt.axis((xmin,xmax,ymin,ymax))
            plt.axis((0,5,ymin,ymax))
            #plt.axis((0,200,ymin,ymax)) # without R
            plt.plot([1,1],[ymin,ymax],'r--')

            # Annotate with summary stats
            xmin,xmax,ymin,ymax = plt.axis()
            ypos = ymax-0.05*(ymax-ymin)
            xpos = xmax-0.40*(xmax-xmin)
            plt.text(xpos,ypos,'PSM calib: %s' %datatag_calib,fontsize=11,fontweight='bold')

            plt.savefig('%s/recon_proxy_%s_stats_hist_EnsCal.png' % (figdir,v),bbox_inches='tight')
            plt.close()


            # ===============================================================================
            # 4) Time series: comparison of proxy values and reconstruction-estimated proxies 
            #    for every site used in verification
            #
            # 5) Scatter plots of proxy vs reconstruction (ensemble-mean)
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

                plt.savefig('%s/ts_recon_vs_proxy_%s_%s_%s.png' % (figdir, v, sitetype, sitename),bbox_inches='tight')
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

                plt.savefig('%s/scatter_recon_vs_proxy_%s_%s_%s.png' % (figdir, v, sitetype, sitename),bbox_inches='tight')
                plt.close()


    end_time = time() - begin_time
    print '======================================================='
    print 'All completed in '+ str(end_time/60.0)+' mins'
    print '======================================================='



# =============================================================================

if __name__ == '__main__':
    main()
