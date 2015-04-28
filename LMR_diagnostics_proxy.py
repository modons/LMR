
#==========================================================================================
# Contains functions used to evaluate the LMR paleoclimate reconstructions. 
# Error statistics of the reconstructions are evaluated using an independent set of proxy 
# chronologies. 
# 
# Originators: Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
#                            | March 2015
#  
#========================================================================================== 

# Import parameters for the verification
from LMR_verif_NAMELIST import *

#==========================================================================================
def rmse(predictions, targets):
    import numpy as np
    return np.sqrt(((predictions - targets) ** 2).mean())

#==========================================================================================
def CE(predictions, targets):
    import numpy as np
    return 1.0 - ( (np.sum((targets-predictions)**2))/(np.sum((targets-np.mean(targets))**2)) )


def recon_proxy_eval_stats(sites_eval,recon_period,recondir,calib,prior,datadir_proxy,datafile_proxy,regions):
#==========================================================================================
# 
# 
#   Input : 
#         - sites_eval    : Dictionary containing list of sites (proxy chronologies) per
#                           proxy type
#         - recon_period  : Years defining the reconstruction period. Ex.: [1800, 2000]
#         - recondir      : Directory where the reconstruction data is located
#         - calib         : Calibration object
#         - prior         : Prior object
#         - datadir_proxy : 
#         - datadir_file  :
#         - regions       : 
#
#   Output: None
# 
#==========================================================================================

    import LMR_proxy
    import LMR_utils
    import numpy as np

    # Output dictionary
    evald = {}

    proxy_types_eval = sites_eval.keys()

    print '-----------------------------------'
    print 'Sites to be processed:             '
    totalsites = 0
    for proxy_key in proxy_types_eval:
        print proxy_key, ':', len(sites_eval[proxy_key]), 'sites'
        totalsites = totalsites + len(sites_eval[proxy_key])

    print '-----------------------------------'
    print 'Total:', totalsites
    print ' '

    sitecount = 0
    # Loop over proxy types
    for proxy_key in proxy_types_eval:
        # Strip the assim. order info & keep only proxy type
        #proxy = proxy_key.split(':', 1)[1]
        proxy = proxy_key

        # Loop over proxy sites set aside for evaluation 
        for site in sites_eval[proxy_key]:

            sitecount = sitecount + 1

            Yeval = LMR_proxy.proxy_assignment(proxy)
            # add namelist attributes to the proxy object
            Yeval.proxy_datadir  = datadir_proxy
            Yeval.proxy_datafile = datafile_proxy
            Yeval.proxy_region   = regions

            print 'Site:', Yeval.proxy_type, ':', site, '=> nb', sitecount, 'out of', totalsites, '(',(np.float(sitecount)/np.float(totalsites))*100,'% )'
            # Read data for current proxy type/chronology
            Yeval.read_proxy(site)
            print ' latitude, longitude: ' + str(Yeval.lat), str(Yeval.lon)

            if Yeval.nobs == 0: # if no obs uploaded, move to next proxy site
                continue

            sitetag = (proxy, site)
            evald[sitetag] = {}
            evald[sitetag]['lat'] = Yeval.lat
            evald[sitetag]['lon'] = Yeval.lon
            evald[sitetag]['alt'] = Yeval.alt

            # indices of proxy ob that overlap with recon. period
            indices = [j for j, t in enumerate(Yeval.time) if t >= recon_period[0] and t <= recon_period[1]]
            Ntime = len(indices)

            if Ntime == 0: # if no obs uploaded, move to next proxy site
                continue

            Xrecon_error = np.zeros(shape=[Ntime,prior.Nens]) 
            truth = np.zeros(shape=[Ntime]) 
            Ye_recon_EnsMean   = np.zeros(shape=[Ntime]) 
            Ye_recon_EnsSpread = np.zeros(shape=[Ntime]) 

            # Loop over time in proxy record
            obcount = 0
            for t in [Yeval.time[k] for k in indices]:

                indt = Yeval.time.index(t)
                truth[obcount] = Yeval.value[indt]

                ypad = LMR_utils.year_fix(t)
                #print 'Processing year:', ypad
                filen = recondir + '/' + 'year' + ypad
                Xrecon = np.load(filen+'.npy')

                # Proxy extimates from posterior (reconstruction)
                Ye_recon = Yeval.psm(calib,Xrecon,prior.lat,prior.lon)

                # Reasons to abandon this proxy record due to calibration problems
                if Ye_recon is None:
                    print 'PSM could not be built for this proxy chronology...too little overlap for calibration'
                    break
                elif abs(Yeval.corr) < PSM_r_crit:
                    print 'PSM could not be built for this proxy chronology...calibration correlation below threshold'
                    break

                # ensemble mean Ye
                Ye_recon_EnsMean[obcount] = np.mean(Ye_recon,dtype=np.float64)
                # ensemble spread Ye
                Ye_recon_EnsSpread[obcount] = np.std(Ye_recon,dtype=np.float64)
                # Reconstruction error (full ensemble)
                Xrecon_error[obcount,:] = (Ye_recon - truth[obcount])

                obcount = obcount + 1

            if obcount > 0:

                print '================================================'
                print 'Site:', Yeval.proxy_type, ':', site
                print 'Number of verification points:', obcount            
                print 'Mean of proxy values         :', np.mean(truth)
                print 'Mean ensemble-mean           :', np.mean(Ye_recon_EnsMean)
                print 'Mean ensemble-mean error     :', np.mean(Ye_recon_EnsMean-truth)
                print 'Ensemble-mean RMSE           :', rmse(Ye_recon_EnsMean,truth)
                print 'Correlation                  :', np.corrcoef(truth,Ye_recon_EnsMean)[0,1]
                print 'CE                           :', CE(Ye_recon_EnsMean,truth)
                print '================================================'            

                # Fill dictionary with data generated for evaluation of reconstruction
                # PSM info
                evald[sitetag]['PSMslope']          = Yeval.slope
                evald[sitetag]['PSMintercept']      = Yeval.intercept
                evald[sitetag]['PSMcorrel']         = Yeval.corr
                evald[sitetag]['PSMmse']            = Yeval.R
                # Verif. data
                evald[sitetag]['NbEvalPts']         = obcount
                evald[sitetag]['EnsMean_MeanError'] = np.mean(Ye_recon_EnsMean-truth)
                evald[sitetag]['EnsMean_RMSE']      = rmse(Ye_recon_EnsMean,truth)
                evald[sitetag]['EnsMean_Corr']      = np.corrcoef(truth,Ye_recon_EnsMean)[0,1]
                evald[sitetag]['EnsMean_CE']        = CE(Ye_recon_EnsMean,truth)            
                evald[sitetag]['ts_years']          = [Yeval.time[k] for k in indices]
                evald[sitetag]['ts_ProxyValues']    = truth
                evald[sitetag]['ts_EnsMean']        = Ye_recon_EnsMean
                evald[sitetag]['ts_EnsSpread']      = Ye_recon_EnsSpread

    return evald


# =============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Main code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# =============================================================================
def main():

    from time import time
    import numpy as np
    import cPickle    

    import LMR_calibrate
    import LMR_proxy
    import LMR_prior

    from load_proxy_data import read_proxy_metadata_S1csv as read_proxy_metadata
    
    # Loop over the Monte-Carlo reconstructions
    MCiters = np.arange(iter_range[0], iter_range[1]+1)
    for iter in MCiters:

        # Experiment data directory
        workdir = datadir_input+'/'+nexp+'/r'+str(iter)
        print workdir

        print '============================================================'
        print 'Working on: ' + nexp + ' : ' + '/r' + str(iter)
        print '============================================================'

        begin_time = time()

        # ===========================
        # Creating calibration object
        # ===========================
        # Assign calibration object according to "datatag_calib" (from namelist)
        C = LMR_calibrate.calibration_assignment(datatag_calib)
        # the path to the calibration directory is specified in the namelist file; bind it here
        C.datadir_calib = datadir_calib;
        # read the data
        C.read_calibration()
    
        # ==============
        # Get prior info
        # ==============
        # Assign empty prior object
        X = LMR_prior.prior_assignment('generic')

        # Load data from Xb_one.npz tp get recon grid info (gridpt lat/lon)
        Xtmp = np.load(workdir+'/'+'Xb_one.npz')
        nlat = Xtmp['nlat']; nlon = Xtmp['nlon']
        Xb_one = Xtmp['Xb_one']
        # Bind to prior object
        X.lat  = Xtmp['lat']
        X.lon  = Xtmp['lon']
        X.Nens = Xb_one.shape[1]

        # ===========================
        # List of assimilated proxies
        # ===========================
    
        # Read in the file of assimilated proxies for experiment
        assim_proxies = np.load(workdir+'/'+'assimilated_proxies.npy')

        assim_types = []
        for k in xrange(len(assim_proxies)):
            key = assim_proxies[k].keys()
            assim_types.append(key[0])
        assim_types_list = list(set(assim_types))

        sites_assim = {}
        for t in assim_types_list:
            sites_assim[t] = []
            for k in xrange(len(assim_proxies)):
                key = assim_proxies[k].keys()
                if key[0] == t:
                    sites_assim[t].append(assim_proxies[k][key[0]][0])

        sites_assim_list = []
        tmp = [sites_assim[t] for t in assim_types_list]
        print '=>', tmp
        for k in xrange(len(tmp)):
            sites_assim_list.extend(tmp[k])

        print 'Sites assimilated:', len(sites_assim_list), sites_assim

        # ==========================================================================
        # Proxy site-based statistics on reconstruction fit to assimilated proxies 
        # ==========================================================================
        # Calculate reconstruction error statistics & output in "assim_dict" dictionary
        assim_dict = recon_proxy_eval_stats(sites_assim,recon_period,workdir,C,X,datadir_proxy,datafile_proxy,regions)

        # Dump dictionary to pickle file
        outfile = open('%s/reconstruction_assim_diag.pckl' % (workdir),'w')
        cPickle.dump(assim_dict,outfile)
        outfile.close()

        # ==========================================
        # List of proxies available for verification
        # ==========================================

        # Read in the master proxy metadata file to list all sites matching parameters set in NAMELIST 
        all_proxies = read_proxy_metadata(datadir_proxy,datafile_proxy,regions,proxy_resolution,proxy_verif)

        # Determine set of proxies that can be used for verification (have not been assimilated)
        proxy_types_verif = all_proxies.keys()

        # Start with all possible proxy sites
        sites_verif = all_proxies

        for t in proxy_types_verif:
            for s in sites_verif[t]:
                if s in sites_assim_list:
                    # Remove from list if site assimilated
                    sites_verif[t].remove(s)

        print 'Sites for verification:', len(sites_assim_list), sites_verif

        # ==========================================================================
        # Proxy site-based statistics on reconstruction fit to non-assimilated 
        # (verification) proxies 
        # ==========================================================================
        # Calculate reconstruction error statistics & output in "verif_dict" dictionary
        verif_dict = recon_proxy_eval_stats(sites_verif,recon_period,workdir,C,X,datadir_proxy,datafile_proxy,regions)

        # Dump dictionary to pickle file
        outfile = open('%s/reconstruction_verif_diag.pckl' % (workdir),'w')
        cPickle.dump(verif_dict,outfile)
        outfile.close()

        end_time = time() - begin_time
        print '======================================================='
        print 'Verification completed in '+ str(end_time/60.0)+' mins'
        print '======================================================='


# =============================================================================

if __name__ == '__main__':
    main()
