
#==========================================================================================
# Contains functions used to evaluate the LMR paleoclimate reconstructions. 
# Error statistics of the reconstructions are evaluated using an independent set of proxy 
# chronologies. 
# 
# Originators: Greg Hakim    | Dept. of Atmospheric Sciences, Univ. of Washington
#              Robert Tardif | March 2015
#  
#========================================================================================== 

def rmse(predictions, targets):
    import numpy as np
    return np.sqrt(((predictions - targets) ** 2).mean())


def CE(predictions, targets):
    import numpy as np
    return 1.0 - ( (np.sum((targets-predictions)**2))/(np.sum((targets-np.mean(targets))**2)) )


def recon_eval_stats(sites_eval,recon_period,recondir,calib,prior,datadir_proxy,datafile_proxy,regions):
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
    import cPickle
    import numpy as np

    # Output dictionary
    evald = {}

    proxy_types_eval = sites_eval.keys()

    # Loop over proxy types
    for proxy_key in proxy_types_eval:
        # Strip the assim. order info & keep only proxy type
        proxy = proxy_key.split(':', 1)[1]

        # Loop over proxy sites set aside for evaluation 
        for site in sites_eval[proxy_key]:

            Yeval = LMR_proxy.proxy_assignment(proxy)
            # add namelist attributes to the proxy object
            Yeval.proxy_datadir  = datadir_proxy
            Yeval.proxy_datafile = datafile_proxy
            Yeval.proxy_region   = regions

            print 'Site:', Yeval.proxy_type, ':', site
            # Read data for current proxy type/chronology
            Yeval.read_proxy(site)
            print ' latitude, longitude: ' + str(Yeval.lat), str(Yeval.lon)

            if Yeval.nobs == 0: # if no obs uploaded, move to next proxy type
                continue

            sitetag = (proxy, site)
            evald[sitetag] = {}
            evald[sitetag]['lat'] = Yeval.lat
            evald[sitetag]['lon'] = Yeval.lon
            evald[sitetag]['alt'] = Yeval.alt

            indices = [j for j, t in enumerate(Yeval.time) if t >= recon_period[0] and t <= recon_period[1]]
            Ntime = len(indices)

            Xrecon_error = np.zeros(shape=[Ntime,prior.Nens]) 
            truth = np.zeros(shape=[Ntime]) 
            Ye_recon_EnsMean   = np.zeros(shape=[Ntime]) 
            Ye_recon_EnsSpread = np.zeros(shape=[Ntime]) 

            # Loop over time in proxy record
            tcount = 0
            for t in [Yeval.time[k] for k in indices]:

                indt = Yeval.time.index(t)
                truth[tcount] = Yeval.value[indt]

                ypad = LMR_utils.year_fix(t)
                #print 'Processing year:', ypad
                filen = recondir + '/' + 'year' + ypad
                Xrecon = np.load(filen+'.npy')

                # Proxy extimates from posterior (reconstruction)
                Ye_recon = Yeval.psm(calib,Xrecon,prior.lat,prior.lon)

                # ensemble mean Ye
                Ye_recon_EnsMean[tcount] = np.mean(Ye_recon,dtype=np.float64)
                # ensemble spread Ye
                Ye_recon_EnsSpread[tcount] = np.std(Ye_recon,dtype=np.float64)
                # Reconstruction error (full ensemble)
                Xrecon_error[tcount,:] = (Ye_recon - truth[tcount])

                tcount = tcount + 1

            print '================================================'
            print 'Site:', Yeval.proxy_type, ':', site
            print 'Number of verification points:', tcount            
            print 'Mean of proxy values         :', np.mean(truth)
            print 'Mean ensemble-mean           :', np.mean(Ye_recon_EnsMean)
            print 'Mean ensemble-mean error     :', np.mean(Ye_recon_EnsMean-truth)
            print 'Ensemble-mean RMSE           :', rmse(Ye_recon_EnsMean,truth)
            print 'Correlation                  :', np.corrcoef(truth,Ye_recon_EnsMean)[0,1]
            print 'CE                           :', CE(Ye_recon_EnsMean,truth)
            print '================================================'            

            # Fill dictionary with data generated for evaluation of reconstruction
            evald[sitetag]['NbEvalPts']         = tcount
            evald[sitetag]['EnsMean_MeanError'] = np.mean(Ye_recon_EnsMean-truth)
            evald[sitetag]['EnsMean_RMSE']      = rmse(Ye_recon_EnsMean,truth)
            evald[sitetag]['EnsMean_Corr']      = np.corrcoef(truth,Ye_recon_EnsMean)[0,1]
            evald[sitetag]['EnsMean_CE']        = CE(Ye_recon_EnsMean,truth)            
            evald[sitetag]['ts_years']          = [Yeval.time[k] for k in indices]
            evald[sitetag]['ts_ProxyValues']    = truth
            evald[sitetag]['ts_EnsMean']        = Ye_recon_EnsMean
            evald[sitetag]['ts_EnsSpread']      = Ye_recon_EnsSpread

    # Dump dictionary to pickle file
    outfile = open('%s/reconstruction_eval.pckl' % (recondir),'w')
    cPickle.dump(evald,outfile)
    outfile.close()



