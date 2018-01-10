"""
Module: LMR_verify_proxy.py

Purpose: Updated version of the proxy verification module. 
         This performs an evaluation of LMR paleoclimate reconstructions by comparing
         proxy values estimated from the reanalysis and actual proxy records that were
         withheld from the reanalysis. Statistics are also compiled on assimilated
         proxies for comparison.
         New: This version uses proxy estimates (Ye) from withheld records that have
              been updated as part of the update procedure and included in the
              analysis_Ye.pckl file, as with the assimilated records.

Originator: Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
                          | March 2015

Revisions:
           - From original LMR_diagnostics_proxy code, adapted to new reconstruction 
             output (ensemble mean only) and use of pre-built PSMs.
             [R. Tardif, U. of Washington, July 2015]
           - Adapted to the Pandas DataFrames version of the proxy database.
             [R. Tardif, U. of Washington, December 2015]
           - Addded the use of NCDC proxy database as possible proxy data input.
             [R. Tardif, U. of Washington, March 2016]
           - Adjustments made to code to reflect recent changes to how config classes
             are defined and instanciated + general code re-org. for improving speed.
             [R. Tardif, U. of Washington, July 2016]
           - General re-org. of code for input of estimated proxy values from 
             withheld records now stored in analysis_Ye.pckl file. 
             [R. Tardif, U. of Washington, Sept 2017]
"""
import os, sys
import numpy as np
import pickle
import pandas as pd
import yaml
import glob
from time import time
from os.path import join, isfile

# LMR specific imports
sys.path.append('../')
import LMR_config
import LMR_proxy_pandas_rework
from LMR_utils import coefficient_efficiency, rmsef

# ------------------------------------------------------------ #
# -------------- Begin: user-defined parameters -------------- #

verbose = 1

# Input directory, where to find the reconstruction experiment data
#datadir_input  = '/home/disk/kalman2/wperkins/LMR_output/archive' # production recons
#datadir_input  = '/home/disk/kalman3/hakim/LMR/'
datadir_input  = '/home/disk/kalman3/rtardif/LMR/output'
#datadir_input  = '/home/disk/ekman4/rtardif/LMR/output'

# Name of experiment
nexp = 'test'

# perform verification using all recon. MC realizations ( MCset = None )
# or over a custom selection ( MCset = (begin,end) )
# ex. MCset = (0,0)    -> only the first MC run
#     MCset = (0,10)   -> the first 11 MC runs (from 0 to 10 inclusively)
#     MCset = (80,100) -> the 80th to 100th MC runs (21 realizations)
MCset = None
#MCset = (0,0)

# period of reconstruction over which to perform the evaluation
# (inclusive)
#verif_period = (0, 1879)
verif_period = (1880, 2000)
#verif_period = (1985, 2000)

# Output directory, where the verification results & figs will be dumped.
datadir_output = datadir_input # if want to keep things tidy
#datadir_output = '/home/disk/ekman4/rtardif/LMR/output/verification_production_runs'

write_full_verif_dict = True

# --------------  End: user-defined parameters  -------------- #
# ------------------------------------------------------------ #


begin_time = time()

print(' ')
print('Experiment           : %s' %nexp)
print('Verif. period        : %s' %str(verif_period))
print('MC iterations        : %s' %str(MCset))


# get a listing of the iteration directories
workdir = join(datadir_input,nexp)

dirs = glob.glob(workdir+"/r*")
ntotiters = len(dirs)

if MCset:
    dirset = dirs[MCset[0]:MCset[1]+1]
else:
    dirset = dirs
niters = len(dirset)


# load proxy data
# ---------------
# query config found in first MC directory
# check for experiment config information in input directory 

yaml_file = join(dirset[0], 'config.yml')
cfgpy_file = join(dirset[0], 'LMR_config.py')
if isfile(yaml_file):
    # .yml file?
    print('Loading configuration: {}'.format(yaml_file))

    f = open(yaml_file, 'r')
    yml_dict = yaml.load(f)
    update_result = LMR_config.update_config_class_yaml(yml_dict,
                                                        LMR_config)
elif isfile(cfgpy_file):
    # LMR_config.py file?
    print('Loading configuration: {}'.format(cfgpy_file))
    file_name = cfgpy_file.split('/')[-1]
    module_name = file_name.rstrip('.py')
    path = list(sys.path)
    sys.path.insert(0,dir)
    LMR_config = __import__(module_name)
else:    
    raise SystemExit('Could not locate experiment configuration information. Exiting!')

cfg = LMR_config.Config()

proxy_database = cfg.proxies.use_from[0]
print('Proxy database       : %s' %proxy_database)

if proxy_database == 'PAGES2kv1':
    proxy_cfg = cfg.proxies.PAGES2kv1
elif proxy_database == 'LMRdb':
    proxy_cfg = cfg.proxies.LMRdb
elif proxy_database == 'NCDCdtda':
    proxy_cfg = cfg.proxies.ncdcdtda
else:
    raise SystemExit('ERROR in specification of proxy database.')

proxy_types_psm = proxy_cfg.proxy_psm_type
if cfg.psm.avgPeriod == 'annual':
    psm_seasonality = cfg.psm.avgPeriod
else:
    psm_seasonality = '_'.join([cfg.psm.avgPeriod,cfg.psm.season_source])

proxy_class = LMR_proxy_pandas_rework.get_proxy_class(proxy_database)
print('loading proxy data...') 

# ensure all proxies are considered, given the applied filters 
cfg.proxies.proxy_frac = 1.0
#proxy_objects = proxy_class.load_all_annual_no_filtering(cfg) # to load all records irrespective of config filters
ids_by_grp, proxy_objects = proxy_class.load_all(cfg,verif_period,None)


print('\ntotal number of iterations in directory: ' + str(ntotiters))
print 'number of iterations considered        : ' + str(niters)


# ---------------------------------------------
# Loop over the Monte-Carlo reconstructions ---
# ---------------------------------------------

verif_listdict = []
assim_listdict = []
iter = 0
for dir in dirset:

    if verbose > 0:
        print('================================================')
        print('\nMC iteration: %s' %dir)
    
    # Load the Ye data and parse between assimilated and withheld records
    filn = join(dir,'analysis_Ye.pckl')
    infile = open(filn,'rb')
    Ye_data_MC = pickle.load(infile)
    infile.close()
    prx_sites = list(Ye_data_MC.keys())

    # Load the prior Ye ensemble data
    file_prior = join(dir,'Xb_one.npz')

    Xprior_statevector = np.load(file_prior)
    Xb_state_info = Xprior_statevector['state_info'].item()
    # extract augmented state
    Xb_one_aug = Xprior_statevector['Xb_one_aug']
    stateDim = Xprior_statevector['stateDim']    
    # extract the Ye values from the state vector
    Ye_prior = Xb_one_aug[stateDim:,:]
    
    #print Xb_one_aug.shape, Ye_prior.shape, len(prx_sites)

    
    # ---------------------
    # Loop over proxies ---
    # ---------------------
    dict_assim = {}
    dict_verif = {}
    pcount_tot = 0
    for p in prx_sites:

        # check whether info on 'status' of records is included in the input file
        try:
            pstatus = Ye_data_MC[p]['status']
        except KeyError as e:
            # assume assimilated (as in original file structure)
            pstatus = 'assimilated'

        [ntimes, nens] = Ye_data_MC[p]['HXa'].shape

        R = Ye_data_MC[p]['R'] # obs error variance

        # Keep only the entries within the verification period
        Ye_years = Ye_data_MC[p]['years'][(Ye_data_MC[p]['years'] >= verif_period[0]) &
                                          (Ye_data_MC[p]['years'] <= verif_period[1])]

        # check if proxy data available over verification interval
        indp = [i for i, pobj in enumerate(proxy_objects) if pobj.id == p[1]]

        if len(Ye_years) > 0 and indp:
            pobj = proxy_objects[indp[0]]
            Ye_HXa = Ye_data_MC[p]['HXa'][(Ye_data_MC[p]['years'] >= verif_period[0]) &
                                          (Ye_data_MC[p]['years'] <= verif_period[1])]
        else:
            continue # move to next record
            
                
        # Use pandas DataFrame to store proxy & Ye data side-by-side
        headers = ['time'] + ['Ye_recon_%s' %str(k+1) for k in range(nens)] + ['Ye_recon_ensMean']

        # Merge times and Ye values from recon (full sensemble & ensemble-mean) in single array
        # & convert to dataframe
        ye_data = np.c_[Ye_years.T,Ye_HXa, np.mean(Ye_HXa, axis=1)]
        df = pd.DataFrame(ye_data)
        df.columns = headers
        
        # Add proxy data        
        frame = pd.DataFrame({'time':pobj.time, 'y': pobj.values})
        df = df.merge(frame, how='outer', on='time')

        # ensure all df entries are floats: if not, some calculations choke
        df = df.astype(np.float)

        # define dataframe indexing
        col0 = df.columns[0]
        df.set_index(col0, drop=True, inplace=True)
        df.index.name = 'time'
        df.sort_index(inplace=True)
        
        df_error = df.iloc[:,:-1].apply(lambda x: x - df.y, axis=0).add_suffix('_error')
        obcount = df_error['Ye_recon_ensMean_error'].count()
                                                     
        if obcount < 50:
            # too few points for verification, move to next record
            continue

        pcount_tot += 1


        indok = df['y'].notnull()

        
        # prior info for proxy record
        # ---------------------------
        p_idx = Ye_data_MC[p]['augStateIndex']
        dimtime = Ye_years.shape
        # Broadcast over time dimension
        Ye_prior_ens = np.repeat(Ye_prior[p_idx][None,:], dimtime,axis=0)

        # Use pandas DataFrame to store proxy & Ye data side-by-side
        headers = ['time'] + ['Ye_prior_%s' %str(k+1) for k in range(nens)] + ['Ye_prior_ensMean']

        # Merge times and Ye values from recon (full sensemble & ensemble-mean) in single array
        # & convert to dataframe
        ye_data = np.c_[Ye_years.T,Ye_prior_ens, np.mean(Ye_prior_ens, axis=1)]
        dfp = pd.DataFrame(ye_data)
        dfp.columns = headers
        
        # Add proxy data        
        frame = pd.DataFrame({'time':pobj.time, 'y': pobj.values})
        dfp = dfp.merge(frame, how='outer', on='time')

        # ensure all df entries are floats: if not, some calculations choke
        dfp = dfp.astype(np.float)

        # define dataframe indexing
        col0 = dfp.columns[0]
        dfp.set_index(col0, drop=True, inplace=True)
        dfp.index.name = 'time'
        dfp.sort_index(inplace=True)
        
        dfp_error = dfp.iloc[:,:-1].apply(lambda x: x - dfp.y, axis=0).add_suffix('_error')

        
        # Ensemble calibration ratios
        # Reconstruction
        mse = np.square(rmsef(df['Ye_recon_ensMean'][indok],df['y'][indok]))
        varYe = np.var(Ye_HXa,axis=1,ddof=1)
        recon_calib_ratio = np.mean(mse/(varYe+R))
        # Prior
        mse = np.square(rmsef(dfp['Ye_prior_ensMean'][indok],dfp['y'][indok]))
        varYe = np.var(Ye_prior_ens,axis=1,ddof=1)
        prior_calib_ratio = np.mean(mse/(varYe+R))

        
        if verbose > 0:
            print('================================================')
            print('Site:', p)
            print('status:', pstatus)
            print('Number of verification points    :', obcount)            
            print('Mean of proxy values             :', np.mean(df['y'][indok]))
            print('Mean ensemble-mean               :', np.mean(df['Ye_recon_ensMean'][indok]))
            print('Mean ensemble-mean error         :', np.mean(df_error['Ye_recon_ensMean_error'][indok]))
            print('Ensemble-mean RMSE               :', rmsef(df['Ye_recon_ensMean'][indok],df['y'][indok]))
            print('Ensemble-mean Correlation        :', np.corrcoef(df['y'][indok],df['Ye_recon_ensMean'][indok])[0,1])
            print('Ensemble-mean CE                 :', coefficient_efficiency(df['y'][indok],df['Ye_recon_ensMean'][indok]))
            print('Ensemble calibration ratio       :', recon_calib_ratio)
            print('..................................')
            print('Mean ensemble-mean(prior)        :', np.mean(dfp['Ye_prior_ensMean'][indok]))
            print('Mean ensemble-mean error(prior)  :', np.mean(dfp_error['Ye_prior_ensMean_error'][indok]))
            print('Ensemble-mean RMSE(prior)        :', rmsef(dfp['Ye_prior_ensMean'][indok],dfp['y'][indok]))
            corr = np.corrcoef(dfp['y'][indok],dfp['Ye_prior_ensMean'][indok])[0,1]
            if not np.isfinite(corr): corr = 0.0
            print('Ensemble-mean Correlation(prior) :', corr)
            print('Ensemble-mean CE(prior)          :', coefficient_efficiency(dfp['y'][indok],dfp['Ye_prior_ensMean'][indok]))
            print('Ensemble calibration ratio(prior):', prior_calib_ratio)
            print('================================================')


        # Fill dictionaries with verification statistics for this MC iteration
        if pstatus == 'assimilated':
            dict_assim[p] = {}
            dict_assim[p]['MCiter'] = iter
            # site info
            dict_assim[p]['lat'] = pobj.lat
            dict_assim[p]['lon'] = pobj.lon
            dict_assim[p]['alt'] = pobj.elev
            # PSM info
            dict_assim[p]['PSMinfo'] = pobj.psm_obj.__dict__
            # verif stats:
            #  Reconstruction
            dict_assim[p]['NbEvalPts'] = obcount
            dict_assim[p]['EnsMean_MeanError'] = np.mean(df_error['Ye_recon_ensMean_error'][indok])
            dict_assim[p]['EnsMean_RMSE']      = rmsef(df['Ye_recon_ensMean'][indok],df['y'][indok])
            dict_assim[p]['EnsMean_Corr']      = np.corrcoef(df['y'][indok],df['Ye_recon_ensMean'][indok])[0,1]
            dict_assim[p]['EnsMean_CE']        = coefficient_efficiency(df['y'][indok],df['Ye_recon_ensMean'][indok])
            dict_assim[p]['EnsCalRatio']       = recon_calib_ratio
            #  Prior
            dict_assim[p]['PriorEnsMean_MeanError'] = np.mean(dfp_error['Ye_prior_ensMean_error'][indok])
            dict_assim[p]['PriorEnsMean_RMSE']      = rmsef(dfp['Ye_prior_ensMean'][indok],dfp['y'][indok])
            corr = np.corrcoef(dfp['y'][indok],dfp['Ye_prior_ensMean'][indok])[0,1]
            if not np.isfinite(corr): corr = 0.0
            dict_assim[p]['PriorEnsMean_Corr']      = corr
            dict_assim[p]['PriorEnsMean_CE']        = coefficient_efficiency(dfp['y'][indok],dfp['Ye_prior_ensMean'][indok])
            dict_assim[p]['PriorEnsCalRatio']       = prior_calib_ratio
            # time series (for plotting later)
            dict_assim[p]['ts_years'] = df.index[indok].values
            dict_assim[p]['ts_ProxyValues'] = df['y'][indok].values
            dict_assim[p]['ts_EnsMean'] = df['Ye_recon_ensMean'][indok].values
            dict_assim[p]['ts_PriorEnsMean'] = dfp['Ye_prior_ensMean'][indok].values
            
        elif pstatus == 'withheld':
            dict_verif[p] = {}
            dict_verif[p]['MCiter'] = iter
            # site info
            dict_verif[p]['lat'] = pobj.lat
            dict_verif[p]['lon'] = pobj.lon
            dict_verif[p]['alt'] = pobj.elev
            # PSM info
            dict_verif[p]['PSMinfo'] = pobj.psm_obj.__dict__
            # verif stats:
            #  Reconstruction
            dict_verif[p]['NbEvalPts'] = obcount
            dict_verif[p]['EnsMean_MeanError'] = np.mean(df_error['Ye_recon_ensMean_error'][indok])
            dict_verif[p]['EnsMean_RMSE']      = rmsef(df['Ye_recon_ensMean'][indok],df['y'][indok])
            dict_verif[p]['EnsMean_Corr']      = np.corrcoef(df['y'][indok],df['Ye_recon_ensMean'][indok])[0,1]
            dict_verif[p]['EnsMean_CE']        = coefficient_efficiency(df['y'][indok],df['Ye_recon_ensMean'][indok])
            dict_verif[p]['EnsCalRatio']       = recon_calib_ratio
            #  Prior
            dict_verif[p]['PriorEnsMean_MeanError'] = np.mean(dfp_error['Ye_prior_ensMean_error'][indok])
            dict_verif[p]['PriorEnsMean_RMSE']      = rmsef(dfp['Ye_prior_ensMean'][indok],dfp['y'][indok])
            corr = np.corrcoef(dfp['y'][indok],dfp['Ye_prior_ensMean'][indok])[0,1]
            if not np.isfinite(corr): corr = 0.0
            dict_verif[p]['PriorEnsMean_Corr']      = corr
            dict_verif[p]['PriorEnsMean_CE']        = coefficient_efficiency(dfp['y'][indok],dfp['Ye_prior_ensMean'][indok])
            dict_verif[p]['PriorEnsCalRatio']       = prior_calib_ratio
            # time series (for plotting later)
            dict_verif[p]['ts_years'] = df.index[indok].values
            dict_verif[p]['ts_ProxyValues'] = df['y'][indok].values
            dict_verif[p]['ts_EnsMean'] = df['Ye_recon_ensMean'][indok].values
            dict_verif[p]['ts_PriorEnsMean'] = dfp['Ye_prior_ensMean'][indok].values            
        else:
            print('proxy status undefined...')
                        
    verif_listdict.append(dict_verif)
    assim_listdict.append(dict_assim)

    iter +=1

        
# --------------------------------------------------------------------
# End of loop on MC iterations => Now calculate summary statistics ---
# --------------------------------------------------------------------

fname = '_'.join(['verifProxy',str(verif_period[0])+'to'+str(verif_period[1])])
outdir = join(datadir_output,nexp,fname)
if not os.path.isdir(outdir):
    os.system('mkdir %s' % outdir)

# check total nb. of sites in verif. and assim sets
nb_tot_verif = sum([len(verif_listdict[k]) for k in range(len(verif_listdict))])
nb_tot_assim = sum([len(assim_listdict[k]) for k in range(len(assim_listdict))])    

# --------------------
# - withheld proxies -
# --------------------
if nb_tot_verif > 0:
    if write_full_verif_dict:
        # Dump dictionary to pickle files
        outfile = open('%s/reconstruction_eval_withheld_proxy_full.pckl' % (outdir),'w')
        pickle.dump(verif_listdict,outfile,protocol=2)
        outfile.close()


    # List of sites in the verif dictionary
    list_tmp = []
    for i in range(len(verif_listdict)):
        for j in range(len(list(verif_listdict[i].keys()))):
            list_tmp.append(list(verif_listdict[i].keys())[j])
    list_sites = list(set(list_tmp)) # filter to unique elements

    summary_stats_verif = {}
    for k in range(len(list_sites)):
        # indices in verif_listdict where this site is present
        inds  = [j for j in range(len(verif_listdict)) if list_sites[k] in list(verif_listdict[j].keys())]

        summary_stats_verif[list_sites[k]] = {}
        summary_stats_verif[list_sites[k]]['lat']          = verif_listdict[inds[0]][list_sites[k]]['lat']
        summary_stats_verif[list_sites[k]]['lon']          = verif_listdict[inds[0]][list_sites[k]]['lon']
        summary_stats_verif[list_sites[k]]['alt']          = verif_listdict[inds[0]][list_sites[k]]['alt']
        summary_stats_verif[list_sites[k]]['NbMCiters']    = len(inds)        
        summary_stats_verif[list_sites[k]]['PSMinfo']      = verif_listdict[inds[0]][list_sites[k]]['PSMinfo']
        
        # These contain data over the "MC ensemble" (i.e. ensemble of realizations) for "kth" site
        nbpts          = [verif_listdict[j][list_sites[k]]['NbEvalPts'] for j in inds]
        me             = [verif_listdict[j][list_sites[k]]['EnsMean_MeanError'] for j in inds]
        rmse           = [verif_listdict[j][list_sites[k]]['EnsMean_RMSE'] for j in inds]
        corr           = [verif_listdict[j][list_sites[k]]['EnsMean_Corr'] for j in inds]
        ce             = [verif_listdict[j][list_sites[k]]['EnsMean_CE'] for j in inds]
        calratio       = [verif_listdict[j][list_sites[k]]['EnsCalRatio'] for j in inds]        
        corr_prior     = [verif_listdict[j][list_sites[k]]['PriorEnsMean_Corr'] for j in inds]
        ce_prior       = [verif_listdict[j][list_sites[k]]['PriorEnsMean_CE'] for j in inds]
        calratio_prior = [verif_listdict[j][list_sites[k]]['PriorEnsCalRatio'] for j in inds]
        
        # Scores for every element in MC ensemble
        summary_stats_verif[list_sites[k]]['MCensME']        = me
        summary_stats_verif[list_sites[k]]['MCensRMSE']      = rmse
        summary_stats_verif[list_sites[k]]['MCensCorr']      = corr
        summary_stats_verif[list_sites[k]]['MCensCE']        = ce
        summary_stats_verif[list_sites[k]]['MCensCalRatio']  = calratio 
        # prior
        summary_stats_verif[list_sites[k]]['PriorMCensCorr']      = corr_prior
        summary_stats_verif[list_sites[k]]['PriorMCensCE']        = ce_prior
        summary_stats_verif[list_sites[k]]['PriorMCensCalRatio']  = calratio_prior
        
        # Summary stats (mean,std) across MC ensemble
        summary_stats_verif[list_sites[k]]['MeanME']          = np.mean(me)
        summary_stats_verif[list_sites[k]]['SpreadME']        = np.std(me)
        summary_stats_verif[list_sites[k]]['MeanRMSE']        = np.mean(rmse)
        summary_stats_verif[list_sites[k]]['SpreadRMSE']      = np.std(rmse)
        summary_stats_verif[list_sites[k]]['MeanCorr']        = np.mean(corr)
        summary_stats_verif[list_sites[k]]['SpreadCorr']      = np.std(corr)
        summary_stats_verif[list_sites[k]]['MeanCE']          = np.mean(ce)
        summary_stats_verif[list_sites[k]]['SpreadCE']        = np.std(ce)
        summary_stats_verif[list_sites[k]]['MeanCalRatio']    = np.mean(calratio)
        summary_stats_verif[list_sites[k]]['SpreadCalRatio']  = np.std(calratio)
        # prior
        summary_stats_verif[list_sites[k]]['PriorMeanCorr']       = np.nanmean(corr_prior)
        summary_stats_verif[list_sites[k]]['PriorSpreadCorr']     = np.nanstd(corr_prior)
        summary_stats_verif[list_sites[k]]['PriorMeanCE']         = np.mean(ce_prior)
        summary_stats_verif[list_sites[k]]['PriorSpreadCE']       = np.std(ce_prior)
        summary_stats_verif[list_sites[k]]['PriorMeanCalRatio']   = np.mean(calratio_prior)
        summary_stats_verif[list_sites[k]]['PriorSpreadCalRatio'] = np.std(calratio_prior)

        # for time series
        summary_stats_verif[list_sites[k]]['ts_years']       = verif_listdict[inds[0]][list_sites[k]]['ts_years']
        summary_stats_verif[list_sites[k]]['ts_ProxyValues'] = verif_listdict[inds[0]][list_sites[k]]['ts_ProxyValues']
        ts_recon = [verif_listdict[j][list_sites[k]]['ts_EnsMean'] for j in inds]
        summary_stats_verif[list_sites[k]]['ts_MeanRecon']   = np.mean(ts_recon,axis=0)
        summary_stats_verif[list_sites[k]]['ts_SpreadRecon'] = np.std(ts_recon,axis=0)
        ts_prior = [verif_listdict[j][list_sites[k]]['ts_PriorEnsMean'] for j in inds]
        summary_stats_verif[list_sites[k]]['ts_MeanPrior']   = np.mean(ts_recon,axis=0)
        summary_stats_verif[list_sites[k]]['ts_SpreadPrior'] = np.std(ts_recon,axis=0)

        
    # Dump data to pickle file
    outfile = open('%s/reconstruction_eval_withheld_proxy_summary.pckl' % (outdir),'w')
    pickle.dump(summary_stats_verif,outfile,protocol=2)
    outfile.close()



# -----------------------
# - assimilated proxies -
# -----------------------
if nb_tot_assim > 0:
    if write_full_verif_dict:
        # Dump dictionary to pickle files
        outfile = open('%s/reconstruction_eval_assimilated_proxy_full.pckl' % (outdir),'w')
        pickle.dump(assim_listdict,outfile,protocol=2)
        outfile.close()


    # List of sites in the assim dictionary
    list_tmp = []
    for i in range(len(assim_listdict)):
        for j in range(len(list(assim_listdict[i].keys()))):
            list_tmp.append(list(assim_listdict[i].keys())[j])
    list_sites = list(set(list_tmp)) # filter to unique elements

    summary_stats_assim = {}
    for k in range(len(list_sites)):
        # indices in assim_listdict where this site is present
        inds  = [j for j in range(len(assim_listdict)) if list_sites[k] in list(assim_listdict[j].keys())]

        summary_stats_assim[list_sites[k]] = {}
        summary_stats_assim[list_sites[k]]['lat']          = assim_listdict[inds[0]][list_sites[k]]['lat']
        summary_stats_assim[list_sites[k]]['lon']          = assim_listdict[inds[0]][list_sites[k]]['lon']
        summary_stats_assim[list_sites[k]]['alt']          = assim_listdict[inds[0]][list_sites[k]]['alt']
        summary_stats_assim[list_sites[k]]['NbMCiters']    = len(inds)
        summary_stats_assim[list_sites[k]]['PSMinfo']      = assim_listdict[inds[0]][list_sites[k]]['PSMinfo']
        
        # These contain data over the "MC ensemble" (i.e. ensemble of realizations) for "kth" site
        nbpts          = [assim_listdict[j][list_sites[k]]['NbEvalPts'] for j in inds]
        me             = [assim_listdict[j][list_sites[k]]['EnsMean_MeanError'] for j in inds]
        rmse           = [assim_listdict[j][list_sites[k]]['EnsMean_RMSE'] for j in inds]
        corr           = [assim_listdict[j][list_sites[k]]['EnsMean_Corr'] for j in inds]
        ce             = [assim_listdict[j][list_sites[k]]['EnsMean_CE'] for j in inds]
        calratio       = [assim_listdict[j][list_sites[k]]['EnsCalRatio'] for j in inds]        
        corr_prior     = [assim_listdict[j][list_sites[k]]['PriorEnsMean_Corr'] for j in inds]
        ce_prior       = [assim_listdict[j][list_sites[k]]['PriorEnsMean_CE'] for j in inds]
        calratio_prior = [assim_listdict[j][list_sites[k]]['PriorEnsCalRatio'] for j in inds]
        
        # Scores for every element in MC ensemble
        summary_stats_assim[list_sites[k]]['MCensME']        = me
        summary_stats_assim[list_sites[k]]['MCensRMSE']      = rmse
        summary_stats_assim[list_sites[k]]['MCensCorr']      = corr
        summary_stats_assim[list_sites[k]]['MCensCE']        = ce
        summary_stats_assim[list_sites[k]]['MCensCalRatio']  = calratio 
        # prior
        summary_stats_assim[list_sites[k]]['PriorMCensCorr']      = corr_prior
        summary_stats_assim[list_sites[k]]['PriorMCensCE']        = ce_prior
        summary_stats_assim[list_sites[k]]['PriorMCensCalRatio']  = calratio_prior
        
        # Summary stats (mean,std) across MC ensemble
        summary_stats_assim[list_sites[k]]['MeanME']          = np.mean(me)
        summary_stats_assim[list_sites[k]]['SpreadME']        = np.std(me)
        summary_stats_assim[list_sites[k]]['MeanRMSE']        = np.mean(rmse)
        summary_stats_assim[list_sites[k]]['SpreadRMSE']      = np.std(rmse)
        summary_stats_assim[list_sites[k]]['MeanCorr']        = np.mean(corr)
        summary_stats_assim[list_sites[k]]['SpreadCorr']      = np.std(corr)
        summary_stats_assim[list_sites[k]]['MeanCE']          = np.mean(ce)
        summary_stats_assim[list_sites[k]]['SpreadCE']        = np.std(ce)
        summary_stats_assim[list_sites[k]]['MeanCalRatio']    = np.mean(calratio)
        summary_stats_assim[list_sites[k]]['SpreadCalRatio']  = np.std(calratio)
        # prior
        summary_stats_assim[list_sites[k]]['PriorMeanCorr']       = np.nanmean(corr_prior)
        summary_stats_assim[list_sites[k]]['PriorSpreadCorr']     = np.nanstd(corr_prior)
        summary_stats_assim[list_sites[k]]['PriorMeanCE']         = np.mean(ce_prior)
        summary_stats_assim[list_sites[k]]['PriorSpreadCE']       = np.std(ce_prior)
        summary_stats_assim[list_sites[k]]['PriorMeanCalRatio']   = np.mean(calratio_prior)
        summary_stats_assim[list_sites[k]]['PriorSpreadCalRatio'] = np.std(calratio_prior)

        # for time series
        summary_stats_assim[list_sites[k]]['ts_years']       = assim_listdict[inds[0]][list_sites[k]]['ts_years']
        summary_stats_assim[list_sites[k]]['ts_ProxyValues'] = assim_listdict[inds[0]][list_sites[k]]['ts_ProxyValues']
        ts_recon = [assim_listdict[j][list_sites[k]]['ts_EnsMean'] for j in inds]
        summary_stats_assim[list_sites[k]]['ts_MeanRecon']   = np.mean(ts_recon,axis=0)
        summary_stats_assim[list_sites[k]]['ts_SpreadRecon'] = np.std(ts_recon,axis=0)
        ts_prior = [assim_listdict[j][list_sites[k]]['ts_PriorEnsMean'] for j in inds]
        summary_stats_assim[list_sites[k]]['ts_MeanPrior']   = np.mean(ts_recon,axis=0)
        summary_stats_assim[list_sites[k]]['ts_SpreadPrior'] = np.std(ts_recon,axis=0)

        
    # Dump data to pickle file
    outfile = open('%s/reconstruction_eval_assimilated_proxy_summary.pckl' % (outdir),'w')
    pickle.dump(summary_stats_assim,outfile,protocol=2)
    outfile.close()

    
verif_time = time() - begin_time
print('=======================================================')
print('Verification completed in '+ str(verif_time/3600.0)+' hours')
print('=======================================================')
