"""
Module: compare2recons.py

Purpose: Plotting differences between two LMR paleoclimate reanalyses.

Originator: Robert Tardif - Univ. of Washington, Dept. of Atmospheric Sciences
            August 2017

Revisions: 


"""
import sys
import os
import glob
import re
import pickle
import numpy as np
from scipy.interpolate import griddata

from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors


sys.path.append('../')
from LMR_plot_support import truncate_colormap
from LMR_utils import global_hemispheric_means

mapcolor = truncate_colormap(plt.cm.jet,0.15,1.0)


# ------------------------------------------------
# --- Begin section of user-defined parameters ---

#datadir = '/home/disk/kalman2/wperkins/LMR_output/archive'
#datadir = '/home/disk/katabatic2/wperkins/LMR_output/testing'
#datadir = '/home/disk/kalman3/hakim/LMR'
datadir = '/home/disk/kalman3/rtardif/LMR/output'
#datadir = '/home/disk/ekman4/rtardif/LMR/output'


# Experiments to compare. Format: [test, reference]
# -------------------------------------------------

exps = ['test2', 'test']

# --
iter_range = [0,10]
#iter_range = [0,100]
# --
year_range = [1800,2000]
#year_range = [0,2000]
#year_range = [-25000,2000]
#year_range = [-115000,2000]

# options of which figures to produce
make_gmt_plot  = True   # time series of gmt, nhmt and shmt
make_map_plots = False  # maps for every recon output within year_range

# for maps:
show_assimilated_proxies = True
make_movie = False


# ==== for GMT timeseries plot:
# -- anomalies --
pltymin = -1.5; pltymax = 1.5; pltymindiff = -0.5; pltymaxdiff = 0.5; ylabel = 'Temperature anomaly (K)'
#pltymin = -6.0; pltymax = 6.0; ylabel = 'Temperature anomaly (K)'
# -- full field --
#pltymin = 276.; pltymax = 290.; ylabel = 'Temperature (K)'

#infile = 'gmt'
infile = 'gmt_ensemble'


# ==== for map plots:
var_to_plot = 'tas_sfc_Amon'
#var_to_plot = 'psl_sfc_Amon'
#var_to_plot = 'wap_850hPa_Amon'
#var_to_plot = 'wap_700hPa_Amon'
#var_to_plot = 'tos_sfc_Omon'
#var_to_plot = 'ohc_0-700m_Omon'
#var_to_plot = 'sos_sfc_Omon'
#var_to_plot = 'hfy_depthavg_Omon'
# --
#var_to_plot = 'tas_sfc_Adec'
#var_to_plot = 'psl_sfc_Adec'
#var_to_plot = 'tos_sfc_Odec'
#var_to_plot = 'sos_sfc_Odec'


mapmin = -2.; mapmax = +2.; mapint = 0.5; cmap = plt.cm.bwr; cbarfmt = '%4.1f'          # T anomalies
#mapmin = -6.; mapmax = +6.; mapint = 2.; cmap = plt.cm.bwr; cbarfmt = '%4.0f'           # T anomalies(2)
#mapmin = -.04; mapmax = +.04; mapint = .01; cmap = plt.cm.bwr; cbarfmt = '%4.2f'        # wap anomalies
#mapmin = -2.e9; mapmax = +2.e9; mapint = 1.e9; cmap = plt.cm.bwr; cbarfmt = '%4.0e'     # OHC anomalies 
#mapmin = -.5; mapmax = +.5; mapint = 0.1; cmap = plt.cm.bwr; cbarfmt = '%4.1f'          # S anomalies
#mapmin = -1.e14; mapmax = +1.e14; mapint = 0.5e15; cmap = plt.cm.bwr; cbarfmt = '%4.0e' # hfy test
# --
#mapmin = 270.; mapmax = 300.; mapint = 2.; cmap = mapcolor; cbarfmt = '%4.0f'         # T full field
#mapmin = 20.; mapmax = 40.; mapint = 5.; cmap = mapcolor; cbarfmt = '%4.0f'           # S full field
#mapmin = 98000.; mapmax = 103000.; mapint = 1000.; cmap = mapcolor; cbarfmt = '%4.0f' # MSLP full field


# ---- End section of user-defined parameters ----
# ------------------------------------------------

bckgcolor = 'gray'


exp_test = exps[0]
exp_refe = exps[1]

expdir_test = datadir + '/'+exp_test
expdir_refe = datadir + '/'+exp_refe

# check if the experiment directories exist
if not os.path.isdir(expdir_test):
    raise SystemExit('Experiment directory for test experiment is not found!'
                     ' Please verify your settings in this verification module.')
if not os.path.isdir(expdir_refe):
    raise SystemExit('Experiment directory for reference experiment is not found!'
                     ' Please verify your settings in this verification module.')

# Figures will be generated in the "test experiment" directory 
figdir = expdir_test+'/CompareFigs'
if not os.path.isdir(figdir):
    os.chdir(expdir_test)
    os.system('mkdir CompareFigs')


if make_gmt_plot:
    
    # ======================================================
    # 1) Time series of global mean temperature
    # ======================================================

    list_iters_test = []
    list_iters_refe = []
    count_test = 0
    count_refe = 0
    gmt_present = False
    iters = np.arange(iter_range[0], iter_range[1]+1)
    for iter in iters:
        dirname_test = expdir_test+'/r'+str(iter)
        dirname_refe = expdir_refe+'/r'+str(iter)
        print(iter, dirname_test)
        list_iters_test.append(dirname_test)
        list_iters_refe.append(dirname_refe)
        # check presence of gmt file 
        if os.path.exists(dirname_test+'/'+infile+'.npz'): count_test +=1
        if os.path.exists(dirname_refe+'/'+infile+'.npz'): count_refe +=1
    nbiters = len(list_iters_refe)
    if (count_test == nbiters) & (count_refe == nbiters): gmt_present = True


    if gmt_present:
        # get array dimensions
        gmt_data_test     = np.load(list_iters_test[0]+'/'+infile+'.npz')
        recon_times_test  = gmt_data_test['recon_times']
        gmt_data_refe     = np.load(list_iters_refe[0]+'/'+infile+'.npz')
        recon_times_refe  = gmt_data_refe['recon_times']
        
        if infile == 'gmt':
            recon_gmt_data_test          = gmt_data_test['gmt_save']
            [nbproxy_test, nbtimes_test] = recon_gmt_data_test.shape
            recon_gmt_data_refe          = gmt_data_refe['gmt_save']
            [nbproxy_refe, nbtimes_refe] = recon_gmt_data_refe.shape
            nens = 1
            file_to_read = 'gmt_save'
        elif infile == 'gmt_ensemble':
            recon_gmt_data_test  = gmt_data_test['gmt_ensemble']
            [nbtimes_test, nens_test] = recon_gmt_data_test.shape
            recon_gmt_data_refe  = gmt_data_refe['gmt_ensemble']
            [nbtimes_refe, nens_refe] = recon_gmt_data_refe.shape            
            nbproxy = 0
            file_to_read = 'gmt_ensemble'
        else:
            SystemExit('Error in infile! Exiting!')

            
        # Declare arrays
        recon_test_years  = np.zeros([nbiters,nbtimes_test])
        recon_test_gmt    = np.zeros([nbiters,nens_test,nbtimes_test])
        recon_test_nhmt   = np.zeros([nbiters,nens_test,nbtimes_test])
        recon_test_shmt   = np.zeros([nbiters,nens_test,nbtimes_test])
        prior_test_gmt    = np.zeros([nbiters,nens_test,nbtimes_test])
        prior_test_nhmt   = np.zeros([nbiters,nens_test,nbtimes_test])
        prior_test_shmt   = np.zeros([nbiters,nens_test,nbtimes_test])

        recon_refe_years  = np.zeros([nbiters,nbtimes_refe])
        recon_refe_gmt    = np.zeros([nbiters,nens_refe,nbtimes_refe])
        recon_refe_nhmt   = np.zeros([nbiters,nens_refe,nbtimes_refe])
        recon_refe_shmt   = np.zeros([nbiters,nens_refe,nbtimes_refe])
        prior_refe_gmt    = np.zeros([nbiters,nens_refe,nbtimes_refe])
        prior_refe_nhmt   = np.zeros([nbiters,nens_refe,nbtimes_refe])
        prior_refe_shmt   = np.zeros([nbiters,nens_refe,nbtimes_refe])
        
        # init. with nan's
        recon_test_gmt[:]  = np.nan
        recon_test_nhmt[:] = np.nan
        recon_test_shmt[:] = np.nan
        prior_test_gmt[:] = np.nan
        prior_test_nhmt[:] = np.nan
        prior_test_shmt[:] = np.nan

        recon_refe_gmt[:]  = np.nan
        recon_refe_nhmt[:] = np.nan
        recon_refe_shmt[:] = np.nan
        prior_refe_gmt[:] = np.nan
        prior_refe_nhmt[:] = np.nan
        prior_refe_shmt[:] = np.nan

        
        # Read-in the data : loop over MC iters
        citer = 0
        for d in list_iters_test: 
            # File of global mean values
            fname = d+'/'+infile+'.npz'
            gmt_data = np.load(fname)
            recon_test_years[citer,:] = gmt_data['recon_times']

            if infile == 'gmt':
                recon_gmt_data = gmt_data[file_to_read]
                [nbproxy_test, nbtimes_test] = recon_gmt_data.shape
                # Final reconstruction
                recon_test_gmt[citer,0,:] = recon_gmt_data[nbproxy-1]

            elif infile == 'gmt_ensemble':
                # Full ensemble reconstruction
                # Global mean
                recon_test_gmt_data = gmt_data[file_to_read]
                recon_test_gmt[citer,:,:] = recon_test_gmt_data.T # flip time/nens dims
                # NH mean
                recon_test_data = gmt_data['nhmt_ensemble']
                recon_test_nhmt[citer,:,:] = recon_test_data.T # flip time/nens dims
                # SH mean
                recon_test_data = gmt_data['shmt_ensemble']
                recon_test_shmt[citer,:,:] = recon_test_data.T # flip time/nens dims
                
            else:
                print('Unrecognized option for infile. Exiting.')
                SystemExit(1)

            # load prior data ---
            file_prior = d + '/Xb_one.npz'
            Xprior_statevector = np.load(file_prior)
            Xb_one = Xprior_statevector['Xb_one']
            # extract variable (sfc temperature) from state vector
            state_info = Xprior_statevector['state_info'].item()
            vars = list(state_info.keys())
            indvar = [j for j, k in enumerate(vars) if 'tas' in k]
            if indvar:
                # surface air temp is in the state vector?
                var_to_extract = vars[indvar[0]]
                posbeg = state_info[var_to_extract]['pos'][0]
                posend = state_info[var_to_extract]['pos'][1]
                tas_prior = Xb_one[posbeg:posend+1,:]
                Xb_one_coords = Xprior_statevector['Xb_one_coords']
                tas_coords =  Xb_one_coords[posbeg:posend+1,:]
                nlat, nlon = state_info[var_to_extract]['spacedims']    
                lat_lalo = tas_coords[:, 0].reshape(nlat, nlon)
                nstate, nens = tas_prior.shape
                tas_lalo = tas_prior.transpose().reshape(nens, nlat, nlon)
                # here, gmt,nhmt and shmt contain the prior ensemble: dims = [nens] 
                [gmt,nhmt,shmt] = global_hemispheric_means(tas_lalo, lat_lalo[:, 0])

                prior_test_gmt[citer,:,:]  = np.repeat(gmt[:,np.newaxis],nbtimes_test,1)
                prior_test_nhmt[citer,:,:] = np.repeat(nhmt[:,np.newaxis],nbtimes_test,1)
                prior_test_shmt[citer,:,:] = np.repeat(shmt[:,np.newaxis],nbtimes_test,1)                
                
            citer = citer + 1

        # reference recon ---
        citer = 0
        for d in list_iters_refe: 
            # File of global mean values
            fname = d+'/'+infile+'.npz'
            gmt_data = np.load(fname)
            recon_refe_years[citer,:] = gmt_data['recon_times']

            if infile == 'gmt':
                recon_gmt_data = gmt_data[file_to_read]
                [nbproxy_refe, nbtimes_refe] = recon_gmt_data.shape
                # Final reconstruction
                recon_refe_gmt[citer,0,:] = recon_gmt_data[nbproxy-1]

            elif infile == 'gmt_ensemble':
                # Full ensemble reconstruction
                # Global mean
                recon_refe_gmt_data = gmt_data[file_to_read]
                recon_refe_gmt[citer,:,:] = recon_refe_gmt_data.T # flip time/nens dims
                # NH mean
                recon_refe_data = gmt_data['nhmt_ensemble']
                recon_refe_nhmt[citer,:,:] = recon_refe_data.T # flip time/nens dims
                # SH mean
                recon_refe_data = gmt_data['shmt_ensemble']
                recon_refe_shmt[citer,:,:] = recon_refe_data.T # flip time/nens dims
                
            else:
                print('Unrecognized option for infile. Exiting.')
                SystemExit(1)

            # load prior data ---
            file_prior = d + '/Xb_one.npz'
            Xprior_statevector = np.load(file_prior)
            Xb_one = Xprior_statevector['Xb_one']
            # extract variable (sfc temperature) from state vector
            state_info = Xprior_statevector['state_info'].item()
            vars = list(state_info.keys())
            indvar = [j for j, k in enumerate(vars) if 'tas' in k]
            if indvar:
                # surface air temp is in the state vector?
                var_to_extract = vars[indvar[0]]
                posbeg = state_info[var_to_extract]['pos'][0]
                posend = state_info[var_to_extract]['pos'][1]
                tas_prior = Xb_one[posbeg:posend+1,:]
                Xb_one_coords = Xprior_statevector['Xb_one_coords']
                tas_coords =  Xb_one_coords[posbeg:posend+1,:]
                nlat, nlon = state_info[var_to_extract]['spacedims']    
                lat_lalo = tas_coords[:, 0].reshape(nlat, nlon)
                nstate, nens = tas_prior.shape
                tas_lalo = tas_prior.transpose().reshape(nens, nlat, nlon)
                # here, gmt,nhmt and shmt contain the prior ensemble: dims = [nens] 
                [gmt,nhmt,shmt] = global_hemispheric_means(tas_lalo, lat_lalo[:, 0])

                prior_refe_gmt[citer,:,:]  = np.repeat(gmt[:,np.newaxis],nbtimes_refe,1)
                prior_refe_nhmt[citer,:,:] = np.repeat(nhmt[:,np.newaxis],nbtimes_refe,1)
                prior_refe_shmt[citer,:,:] = np.repeat(shmt[:,np.newaxis],nbtimes_refe,1)                
                
            citer = citer + 1

            
        if nbiters > 1:
            # Reshaping arrays for easier calculation of stats over the "grand" ensemble (MC iters + DA ensemble members)
            gmpp_test = prior_test_gmt.transpose(2,0,1).reshape(nbtimes_test,-1)
            gmpr_test = recon_test_gmt.transpose(2,0,1).reshape(nbtimes_test,-1)
            nhmpp_test = prior_test_nhmt.transpose(2,0,1).reshape(nbtimes_test,-1)
            nhmpr_test = recon_test_nhmt.transpose(2,0,1).reshape(nbtimes_test,-1)
            shmpp_test = prior_test_shmt.transpose(2,0,1).reshape(nbtimes_test,-1)
            shmpr_test = recon_test_shmt.transpose(2,0,1).reshape(nbtimes_test,-1)

            gmpp_refe = prior_refe_gmt.transpose(2,0,1).reshape(nbtimes_refe,-1)
            gmpr_refe = recon_refe_gmt.transpose(2,0,1).reshape(nbtimes_refe,-1)
            nhmpp_refe = prior_refe_nhmt.transpose(2,0,1).reshape(nbtimes_refe,-1)
            nhmpr_refe = recon_refe_nhmt.transpose(2,0,1).reshape(nbtimes_refe,-1)
            shmpp_refe = prior_refe_shmt.transpose(2,0,1).reshape(nbtimes_refe,-1)
            shmpr_refe = recon_refe_shmt.transpose(2,0,1).reshape(nbtimes_refe,-1)

            
        else:
            gmpp_test = np.squeeze(prior_test_gmt).transpose()
            gmpr_test = np.squeeze(recon_test_gmt).transpose()
            nhmpp_test = np.squeeze(prior_test_nhmt).transpose()
            nhmpr_test = np.squeeze(recon_test_nhmt).transpose()
            shmpp_test = np.squeeze(prior_test_shmt).transpose()
            shmpr_test = np.squeeze(recon_test_shmt).transpose()

            gmpp_refe = np.squeeze(prior_refe_gmt).transpose()
            gmpr_refe = np.squeeze(recon_refe_gmt).transpose()
            nhmpp_refe = np.squeeze(prior_refe_nhmt).transpose()
            nhmpr_refe = np.squeeze(recon_refe_nhmt).transpose()
            shmpp_refe = np.squeeze(prior_refe_shmt).transpose()
            shmpr_refe = np.squeeze(recon_refe_shmt).transpose()

        
        # Priors
        # test
        prior_test_gmt_ensmean    = np.mean(gmpp_test,axis=1)
        prior_test_gmt_ensmin     = np.amin(gmpp_test,axis=1)
        prior_test_gmt_ensmax     = np.amax(gmpp_test,axis=1)
        prior_test_gmt_enssprd    = np.std(gmpp_test,axis=1)
        prior_test_gmt_enslowperc = np.percentile(gmpp_test,5,axis=1)
        prior_test_gmt_ensuppperc = np.percentile(gmpp_test,95,axis=1)

        prior_test_nhmt_ensmean    = np.mean(nhmpp_test,axis=1)
        prior_test_nhmt_ensmin     = np.amin(nhmpp_test,axis=1)
        prior_test_nhmt_ensmax     = np.amax(nhmpp_test,axis=1)
        prior_test_nhmt_enssprd    = np.std(nhmpp_test,axis=1)
        prior_test_nhmt_enslowperc = np.percentile(nhmpp_test,5,axis=1)
        prior_test_nhmt_ensuppperc = np.percentile(nhmpp_test,95,axis=1)

        prior_test_shmt_ensmean    = np.mean(shmpp_test,axis=1)
        prior_test_shmt_ensmin     = np.amin(shmpp_test,axis=1)
        prior_test_shmt_ensmax     = np.amax(shmpp_test,axis=1)
        prior_test_shmt_enssprd    = np.std(shmpp_test,axis=1)
        prior_test_shmt_enslowperc = np.percentile(shmpp_test,5,axis=1)
        prior_test_shmt_ensuppperc = np.percentile(shmpp_test,95,axis=1)

        # refe
        prior_refe_gmt_ensmean    = np.mean(gmpp_refe,axis=1)
        prior_refe_gmt_ensmin     = np.amin(gmpp_refe,axis=1)
        prior_refe_gmt_ensmax     = np.amax(gmpp_refe,axis=1)
        prior_refe_gmt_enssprd    = np.std(gmpp_refe,axis=1)
        prior_refe_gmt_enslowperc = np.percentile(gmpp_refe,5,axis=1)
        prior_refe_gmt_ensuppperc = np.percentile(gmpp_refe,95,axis=1)

        prior_refe_nhmt_ensmean    = np.mean(nhmpp_refe,axis=1)
        prior_refe_nhmt_ensmin     = np.amin(nhmpp_refe,axis=1)
        prior_refe_nhmt_ensmax     = np.amax(nhmpp_refe,axis=1)
        prior_refe_nhmt_enssprd    = np.std(nhmpp_refe,axis=1)
        prior_refe_nhmt_enslowperc = np.percentile(nhmpp_refe,5,axis=1)
        prior_refe_nhmt_ensuppperc = np.percentile(nhmpp_refe,95,axis=1)

        prior_refe_shmt_ensmean    = np.mean(shmpp_refe,axis=1)
        prior_refe_shmt_ensmin     = np.amin(shmpp_refe,axis=1)
        prior_refe_shmt_ensmax     = np.amax(shmpp_refe,axis=1)
        prior_refe_shmt_enssprd    = np.std(shmpp_refe,axis=1)
        prior_refe_shmt_enslowperc = np.percentile(shmpp_refe,5,axis=1)
        prior_refe_shmt_ensuppperc = np.percentile(shmpp_refe,95,axis=1)

        
        # Posteriors
        # test
        recon_test_gmt_ensmean    = np.mean(gmpr_test,axis=1)
        recon_test_gmt_ensmin     = np.amin(gmpr_test,axis=1)
        recon_test_gmt_ensmax     = np.amax(gmpr_test,axis=1)
        recon_test_gmt_enssprd    = np.std(gmpr_test,axis=1)
        recon_test_gmt_enslowperc = np.percentile(gmpr_test,5,axis=1)
        recon_test_gmt_ensuppperc = np.percentile(gmpr_test,95,axis=1)

        recon_test_nhmt_ensmean    = np.mean(nhmpr_test,axis=1)
        recon_test_nhmt_ensmin     = np.amin(nhmpr_test,axis=1)
        recon_test_nhmt_ensmax     = np.amax(nhmpr_test,axis=1)
        recon_test_nhmt_enssprd    = np.std(nhmpr_test,axis=1)
        recon_test_nhmt_enslowperc = np.percentile(nhmpr_test,5,axis=1)
        recon_test_nhmt_ensuppperc = np.percentile(nhmpr_test,95,axis=1)

        recon_test_shmt_ensmean    = np.mean(shmpr_test,axis=1)
        recon_test_shmt_ensmin     = np.amin(shmpr_test,axis=1)
        recon_test_shmt_ensmax     = np.amax(shmpr_test,axis=1)
        recon_test_shmt_enssprd    = np.std(shmpr_test,axis=1)
        recon_test_shmt_enslowperc = np.percentile(shmpr_test,5,axis=1)
        recon_test_shmt_ensuppperc = np.percentile(shmpr_test,95,axis=1)
        
        # refe
        recon_refe_gmt_ensmean    = np.mean(gmpr_refe,axis=1)
        recon_refe_gmt_ensmin     = np.amin(gmpr_refe,axis=1)
        recon_refe_gmt_ensmax     = np.amax(gmpr_refe,axis=1)
        recon_refe_gmt_enssprd    = np.std(gmpr_refe,axis=1)
        recon_refe_gmt_enslowperc = np.percentile(gmpr_refe,5,axis=1)
        recon_refe_gmt_ensuppperc = np.percentile(gmpr_refe,95,axis=1)

        recon_refe_nhmt_ensmean    = np.mean(nhmpr_refe,axis=1)
        recon_refe_nhmt_ensmin     = np.amin(nhmpr_refe,axis=1)
        recon_refe_nhmt_ensmax     = np.amax(nhmpr_refe,axis=1)
        recon_refe_nhmt_enssprd    = np.std(nhmpr_refe,axis=1)
        recon_refe_nhmt_enslowperc = np.percentile(nhmpr_refe,5,axis=1)
        recon_refe_nhmt_ensuppperc = np.percentile(nhmpr_refe,95,axis=1)

        recon_refe_shmt_ensmean    = np.mean(shmpr_refe,axis=1)
        recon_refe_shmt_ensmin     = np.amin(shmpr_refe,axis=1)
        recon_refe_shmt_ensmax     = np.amax(shmpr_refe,axis=1)
        recon_refe_shmt_enssprd    = np.std(shmpr_refe,axis=1)
        recon_refe_shmt_enslowperc = np.percentile(shmpr_refe,5,axis=1)
        recon_refe_shmt_ensuppperc = np.percentile(shmpr_refe,95,axis=1)
                
        
        # => plot +/- 5-95 percentiles among the various realizations
        # test
        recon_test_gmt_low = recon_test_gmt_enslowperc
        recon_test_gmt_upp = recon_test_gmt_ensuppperc
        prior_test_gmt_low = prior_test_gmt_enslowperc
        prior_test_gmt_upp = prior_test_gmt_ensuppperc

        recon_test_nhmt_low = recon_test_nhmt_enslowperc
        recon_test_nhmt_upp = recon_test_nhmt_ensuppperc
        prior_test_nhmt_low = prior_test_nhmt_enslowperc
        prior_test_nhmt_upp = prior_test_nhmt_ensuppperc

        recon_test_shmt_low = recon_test_shmt_enslowperc
        recon_test_shmt_upp = recon_test_shmt_ensuppperc
        prior_test_shmt_low = prior_test_shmt_enslowperc
        prior_test_shmt_upp = prior_test_shmt_ensuppperc

        # refe
        recon_refe_gmt_low = recon_refe_gmt_enslowperc
        recon_refe_gmt_upp = recon_refe_gmt_ensuppperc
        prior_refe_gmt_low = prior_refe_gmt_enslowperc
        prior_refe_gmt_upp = prior_refe_gmt_ensuppperc

        recon_refe_nhmt_low = recon_refe_nhmt_enslowperc
        recon_refe_nhmt_upp = recon_refe_nhmt_ensuppperc
        prior_refe_nhmt_low = prior_refe_nhmt_enslowperc
        prior_refe_nhmt_upp = prior_refe_nhmt_ensuppperc

        recon_refe_shmt_low = recon_refe_shmt_enslowperc
        recon_refe_shmt_upp = recon_refe_shmt_ensuppperc
        prior_refe_shmt_low = prior_refe_shmt_enslowperc
        prior_refe_shmt_upp = prior_refe_shmt_ensuppperc

        
        # -------------------------------------
        # Calculate differences (ensemble mean)
        # -------------------------------------
        recon_diff_years = recon_refe_years 

        recon_diff_gmt_ensmean = recon_test_gmt_ensmean - recon_refe_gmt_ensmean
        prior_diff_gmt_ensmean = prior_test_gmt_ensmean - prior_refe_gmt_ensmean        

        recon_diff_nhmt_ensmean = recon_test_nhmt_ensmean - recon_refe_nhmt_ensmean
        prior_diff_nhmt_ensmean = prior_test_nhmt_ensmean - prior_refe_nhmt_ensmean        

        recon_diff_shmt_ensmean = recon_test_shmt_ensmean - recon_refe_shmt_ensmean
        prior_diff_shmt_ensmean = prior_test_shmt_ensmean - prior_refe_shmt_ensmean        
        

        
        # -----------------------------------------------
        # Plotting time series of global mean temperature
        # -----------------------------------------------

        plt.rcParams['font.weight'] = 'bold'    # set the font weight globally

        fig = plt.figure(figsize=[8,10])
        #fig = plt.figure()

        # test
        fig.add_subplot(3,1,1)
        p1 = plt.plot(recon_test_years[0,:],recon_test_gmt_ensmean,'-b',linewidth=2, label='Posterior')
        plt.fill_between(recon_test_years[0,:], recon_test_gmt_low, recon_test_gmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_test_years[0,:],prior_test_gmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        plt.fill_between(recon_test_years[0,:], prior_test_gmt_low, prior_test_gmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        #plt.suptitle('Global mean temperature', fontsize=12,fontweight='bold')
        plt.title('Global mean temperature\n%s'%exp_test, fontsize=12)
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')
        plt.axis((year_range[0],year_range[1],pltymin,pltymax))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        # refe
        fig.add_subplot(3,1,2)
        p1 = plt.plot(recon_refe_years[0,:],recon_refe_gmt_ensmean,'-b',linewidth=2, label='Posterior')
        plt.fill_between(recon_refe_years[0,:], recon_refe_gmt_low, recon_refe_gmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_refe_years[0,:],prior_refe_gmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        plt.fill_between(recon_refe_years[0,:], prior_refe_gmt_low, prior_refe_gmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        #plt.suptitle(exp_refe, fontsize=12)
        plt.title(exp_refe, fontsize=12)
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')
        plt.axis((year_range[0],year_range[1],pltymin,pltymax))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        # diff
        fig.add_subplot(3,1,3)
        p1 = plt.plot(recon_diff_years[0,:],recon_diff_gmt_ensmean,'-b',linewidth=2, label='Posterior')
        #plt.fill_between(recon_diff_years[0,:], recon_diff_gmt_low, recon_diff_gmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_diff_years[0,:],prior_diff_gmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        #plt.fill_between(recon_diff_years[0,:], prior_diff_gmt_low, prior_diff_gmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        #plt.suptitle('Global mean temperature', fontsize=12,,fontweight='bold')
        plt.title('Difference', fontsize=12)
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')

        plt.axis((year_range[0],year_range[1],pltymindiff,pltymaxdiff))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        fig.tight_layout()
        plt.savefig('%s/%s_vs_%s_GMT_%sto%syrs.png' % (figdir,exp_test,exp_refe,str(year_range[0]),str(year_range[1])),bbox_inches='tight')
        plt.close()
        #plt.show()

        
        # -------------------------------------------
        # Plotting time series of NH mean temperature
        # -------------------------------------------

        fig = plt.figure(figsize=[8,10])
        #fig = plt.figure()

        # test
        fig.add_subplot(3,1,1)
        p1 = plt.plot(recon_test_years[0,:],recon_test_nhmt_ensmean,'-b',linewidth=2, label='Posterior')
        plt.fill_between(recon_test_years[0,:], recon_test_nhmt_low, recon_test_nhmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_test_years[0,:],prior_test_nhmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        plt.fill_between(recon_test_years[0,:], prior_test_nhmt_low, prior_test_nhmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        #plt.suptitle('Global mean temperature', fontsize=12,fontweight='bold')
        plt.title('NH mean temperature\n%s'%exp_test, fontsize=12)
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')
        plt.axis((year_range[0],year_range[1],pltymin,pltymax))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        # refe
        fig.add_subplot(3,1,2)
        p1 = plt.plot(recon_refe_years[0,:],recon_refe_nhmt_ensmean,'-b',linewidth=2, label='Posterior')
        plt.fill_between(recon_refe_years[0,:], recon_refe_nhmt_low, recon_refe_nhmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_refe_years[0,:],prior_refe_nhmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        plt.fill_between(recon_refe_years[0,:], prior_refe_nhmt_low, prior_refe_nhmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        #plt.suptitle(exp_refe, fontsize=12)
        plt.title(exp_refe, fontsize=12)
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')
        plt.axis((year_range[0],year_range[1],pltymin,pltymax))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        # diff
        fig.add_subplot(3,1,3)
        p1 = plt.plot(recon_diff_years[0,:],recon_diff_nhmt_ensmean,'-b',linewidth=2, label='Posterior')
        #plt.fill_between(recon_diff_years[0,:], recon_diff_nhmt_low, recon_diff_nhmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_diff_years[0,:],prior_diff_nhmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        #plt.fill_between(recon_diff_years[0,:], prior_diff_nhmt_low, prior_diff_nhmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        #plt.suptitle('Global mean temperature', fontsize=12,,fontweight='bold')
        plt.title('Difference', fontsize=12)
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')

        plt.axis((year_range[0],year_range[1],pltymindiff,pltymaxdiff))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        fig.tight_layout()
        plt.savefig('%s/%s_vs_%s_NHMT_%sto%syrs.png' % (figdir,exp_test,exp_refe,str(year_range[0]),str(year_range[1])),bbox_inches='tight')
        plt.close()
        #plt.show()


        
        # -------------------------------------------
        # Plotting time series of SH mean temperature
        # -------------------------------------------

        fig = plt.figure(figsize=[8,10])
        #fig = plt.figure()

        # test
        fig.add_subplot(3,1,1)
        p1 = plt.plot(recon_test_years[0,:],recon_test_shmt_ensmean,'-b',linewidth=2, label='Posterior')
        plt.fill_between(recon_test_years[0,:], recon_test_shmt_low, recon_test_shmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_test_years[0,:],prior_test_shmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        plt.fill_between(recon_test_years[0,:], prior_test_shmt_low, prior_test_shmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        #plt.suptitle('Global mean temperature', fontsize=12,fontweight='bold')
        plt.title('SH mean temperature\n%s'%exp_test, fontsize=12)
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')
        plt.axis((year_range[0],year_range[1],pltymin,pltymax))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        # refe
        fig.add_subplot(3,1,2)
        p1 = plt.plot(recon_refe_years[0,:],recon_refe_shmt_ensmean,'-b',linewidth=2, label='Posterior')
        plt.fill_between(recon_refe_years[0,:], recon_refe_shmt_low, recon_refe_shmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_refe_years[0,:],prior_refe_shmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        plt.fill_between(recon_refe_years[0,:], prior_refe_shmt_low, prior_refe_shmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        #plt.suptitle(exp_refe, fontsize=12)
        plt.title(exp_refe, fontsize=12)
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')
        plt.axis((year_range[0],year_range[1],pltymin,pltymax))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        # diff
        fig.add_subplot(3,1,3)
        p1 = plt.plot(recon_diff_years[0,:],recon_diff_shmt_ensmean,'-b',linewidth=2, label='Posterior')
        #plt.fill_between(recon_diff_years[0,:], recon_diff_shmt_low, recon_diff_shmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_diff_years[0,:],prior_diff_shmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        #plt.fill_between(recon_diff_years[0,:], prior_diff_shmt_low, prior_diff_shmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        #plt.suptitle('Global mean temperature', fontsize=12,,fontweight='bold')
        plt.title('Difference', fontsize=12)
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')

        plt.axis((year_range[0],year_range[1],pltymindiff,pltymaxdiff))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        fig.tight_layout()
        plt.savefig('%s/%s_vs_%s_SHMT_%sto%syrs.png' % (figdir,exp_test,exp_refe,str(year_range[0]),str(year_range[1])),bbox_inches='tight')
        plt.close()
        #plt.show()
        

if make_map_plots:    

    # ======================================================
    # Plots of reconstructed spatial fields
    # ======================================================

    # get a listing of the iteration directories
    # and restrict to those selected in iter_range

    #dirs = glob.glob(expdir+"/r*")
    #mcdir = [item.split('/')[-1] for item in dirs]
    #niters = len(mcdir)


    dirs = [item.split('/')[-1] for item in glob.glob(expdir_test+"/r*")]
    mcdir_test = dirs[iter_range[0]:iter_range[1]+1]
    niters_test = len(mcdir_test)

    print('mcdir (test):' + str(mcdir_test))
    print('niters (test) = ' + str(niters_test))

    dirs = [item.split('/')[-1] for item in glob.glob(expdir_refe+"/r*")]
    mcdir_refe = dirs[iter_range[0]:iter_range[1]+1]
    niters_refe = len(mcdir_refe)

    print('mcdir (refe):' + str(mcdir_refe))
    print('niters (refe) = ' + str(niters_refe))

    
    # for info on assimilated proxies
    assimprox_test = {}
    assimprox_refe = {}

    # read ensemble mean data (test recon)
    print('\n reading LMR ensemble-mean data (test recon)...\n')

    first = True
    k = -1
    for dir in mcdir_test:
        k = k + 1
        # Posterior (reconstruction)
        ensfiln = expdir_test + '/' + dir + '/ensemble_mean_'+var_to_plot+'.npz'
        npzfile = np.load(ensfiln)
        print(npzfile.files)
        tmp = npzfile['xam']
        print('shape of tmp: ' + str(np.shape(tmp)))

        # load prior data
        file_prior = expdir_test + '/' + dir + '/Xb_one.npz'
        Xprior_statevector = np.load(file_prior)
        Xb_one = Xprior_statevector['Xb_one']
        # extract variable (sfc temperature) from state vector
        state_info = Xprior_statevector['state_info'].item()
        posbeg = state_info[var_to_plot]['pos'][0]
        posend = state_info[var_to_plot]['pos'][1]
        tas_prior = Xb_one[posbeg:posend+1,:]

        if first:
            first = False

            years = npzfile['years']
            nyrs =  len(years)

            lat = npzfile['lat']
            lon = npzfile['lon']
            # 1D arrays or already in 2D arrays?
            if len(lat.shape) == 1: 
                nlat = npzfile['nlat']
                nlon = npzfile['nlon']
                lat2 = np.reshape(lat,(nlat,nlon))
                lon2 = np.reshape(lon,(nlat,nlon))
            else:
                lat2 = lat
                lon2 = lon

            xam_test = np.zeros([nyrs,np.shape(tmp)[1],np.shape(tmp)[2]])
            xam_all_test = np.zeros([niters_test,nyrs,np.shape(tmp)[1],np.shape(tmp)[2]])
            # prior
            [_,Nens] = tas_prior.shape
            nlatp = state_info[var_to_plot]['spacedims'][0]
            nlonp = state_info[var_to_plot]['spacedims'][1]
            xbm_all_test = np.zeros([niters_test,nyrs,nlatp,nlonp])

        xam_test = xam_test + tmp
        xam_all_test[k,:,:,:] = tmp

        # prior ensemble mean of MC iteration "k"
        tmpp = np.mean(tas_prior,axis=1)
        xbm_all_test[k,:,:,:] = tmpp.reshape(nlatp,nlonp)


        # info on assimilated proxies ---
        assimproxfiln = expdir_test + '/' + dir + '/assimilated_proxies.npy'

        # check existence of file
        if show_assimilated_proxies and os.path.exists(assimproxfiln):    
            assimproxiter = np.load(assimproxfiln)
            nbassimprox, = assimproxiter.shape
            for i in range(nbassimprox):
                ptype = list(assimproxiter[i].keys())[0]
                psite = assimproxiter[i][ptype][0]
                plat  = assimproxiter[i][ptype][1]
                plon  = assimproxiter[i][ptype][2]
                yrs  = assimproxiter[i][ptype][3]

                ptag = (ptype,psite)

                if ptag not in assimprox_test.keys():
                    assimprox_test[ptag] = {}
                    assimprox_test[ptag]['lat']   = plat
                    assimprox_test[ptag]['lon']   = plon
                    assimprox_test[ptag]['years'] = yrs.astype('int')
                    assimprox_test[ptag]['iters'] = [k]
                else:
                    assimprox_test[ptag]['iters'].append(k)


    # Prior sample mean over all MC iterations
    xbm_test = xbm_all_test.mean(0)
    xbm_var_test = xbm_all_test.var(0)

    # Posterior
    #  this is the sample mean computed with low-memory accumulation
    xam_test = xam_test/len(mcdir_test)
    #  this is the sample mean computed with numpy on all data
    xam_check_test = xam_all_test.mean(0)
    #  check..
    max_err = np.max(np.max(np.max(xam_check_test - xam_test)))
    if max_err > 1e-4:
        print('max error = ' + str(max_err))
        raise Exception('sample mean does not match what is in the ensemble files!')

    # sample variance
    xam_test_var = xam_all_test.var(0)
    print(np.shape(xam_test_var))

    print(' shape of the ensemble array: ' + str(np.shape(xam_all_test)) +'\n')
    print(' shape of the ensemble-mean array: ' + str(np.shape(xam_test)) +'\n')
    print(' shape of the ensemble-mean prior array: ' + str(np.shape(xbm_test)) +'\n')

    lmr_lat_range = (lat2[0,0],lat2[-1,0])
    lmr_lon_range = (lon2[0,0],lon2[0,-1])
    print('LMR grid info:')
    print(' lats=', lmr_lat_range)
    print(' lons=', lmr_lon_range)

    recon_times = years.astype(np.float)

    

    # ...



    # read ensemble mean data (refe recon)
    print('\n reading LMR ensemble-mean data (refe recon)...\n')

    first = True
    k = -1
    for dir in mcdir_refe:
        k = k + 1
        # Posterior (reconstruction)
        ensfiln = expdir_refe + '/' + dir + '/ensemble_mean_'+var_to_plot+'.npz'
        npzfile = np.load(ensfiln)
        print(npzfile.files)
        tmp = npzfile['xam']
        print('shape of tmp: ' + str(np.shape(tmp)))

        # load prior data
        file_prior = expdir_refe + '/' + dir + '/Xb_one.npz'
        Xprior_statevector = np.load(file_prior)
        Xb_one = Xprior_statevector['Xb_one']
        # extract variable (sfc temperature) from state vector
        state_info = Xprior_statevector['state_info'].item()
        posbeg = state_info[var_to_plot]['pos'][0]
        posend = state_info[var_to_plot]['pos'][1]
        tas_prior = Xb_one[posbeg:posend+1,:]

        if first:
            first = False

            years = npzfile['years']
            nyrs =  len(years)

            lat = npzfile['lat']
            lon = npzfile['lon']
            # 1D arrays or already in 2D arrays?
            if len(lat.shape) == 1: 
                nlat = npzfile['nlat']
                nlon = npzfile['nlon']
                lat2 = np.reshape(lat,(nlat,nlon))
                lon2 = np.reshape(lon,(nlat,nlon))
            else:
                lat2 = lat
                lon2 = lon

            xam_refe = np.zeros([nyrs,np.shape(tmp)[1],np.shape(tmp)[2]])
            xam_all_refe = np.zeros([niters_refe,nyrs,np.shape(tmp)[1],np.shape(tmp)[2]])
            # prior
            [_,Nens] = tas_prior.shape
            nlatp = state_info[var_to_plot]['spacedims'][0]
            nlonp = state_info[var_to_plot]['spacedims'][1]
            xbm_all_refe = np.zeros([niters_refe,nyrs,nlatp,nlonp])

        xam_refe = xam_refe + tmp
        xam_all_refe[k,:,:,:] = tmp

        # prior ensemble mean of MC iteration "k"
        tmpp = np.mean(tas_prior,axis=1)
        xbm_all_refe[k,:,:,:] = tmpp.reshape(nlatp,nlonp)


        # info on assimilated proxies ---
        assimproxfiln = expdir_refe + '/' + dir + '/assimilated_proxies.npy'

        # check existence of file
        if show_assimilated_proxies and os.path.exists(assimproxfiln):    
            assimproxiter = np.load(assimproxfiln)
            nbassimprox, = assimproxiter.shape
            for i in range(nbassimprox):
                ptype = list(assimproxiter[i].keys())[0]
                psite = assimproxiter[i][ptype][0]
                plat  = assimproxiter[i][ptype][1]
                plon  = assimproxiter[i][ptype][2]
                yrs  = assimproxiter[i][ptype][3]

                ptag = (ptype,psite)

                if ptag not in assimprox_refe.keys():
                    assimprox_refe[ptag] = {}
                    assimprox_refe[ptag]['lat']   = plat
                    assimprox_refe[ptag]['lon']   = plon
                    assimprox_refe[ptag]['years'] = yrs.astype('int')
                    assimprox_refe[ptag]['iters'] = [k]
                else:
                    assimprox_refe[ptag]['iters'].append(k)


    # Prior sample mean over all MC iterations
    xbm_refe = xbm_all_refe.mean(0)
    xbm_var_refe = xbm_all_refe.var(0)

    # Posterior
    #  this is the sample mean computed with low-memory accumulation
    xam_refe = xam_refe/len(mcdir_refe)
    #  this is the sample mean computed with numpy on all data
    xam_check_refe = xam_all_refe.mean(0)
    #  check..
    max_err = np.max(np.max(np.max(xam_check_refe - xam_refe)))
    if max_err > 1e-4:
        print('max error = ' + str(max_err))
        raise Exception('sample mean does not match what is in the ensemble files!')

    # sample variance
    xam_refe_var = xam_all_refe.var(0)
    print(np.shape(xam_refe_var))

    print(' shape of the ensemble array: ' + str(np.shape(xam_all_refe)) +'\n')
    print(' shape of the ensemble-mean array: ' + str(np.shape(xam_refe)) +'\n')
    print(' shape of the ensemble-mean prior array: ' + str(np.shape(xbm_refe)) +'\n')

    lmr_lat_range = (lat2[0,0],lat2[-1,0])
    lmr_lon_range = (lon2[0,0],lon2[0,-1])
    print('LMR grid info:')
    print(' lats=', lmr_lat_range)
    print(' lons=', lmr_lon_range)

    recon_times = years.astype(np.float)

    

    # ----------------------------------
    # Plotting -------------------------
    # ----------------------------------

    recon_interval = np.diff(recon_times)[0]
    proxsites_test = list(assimprox_test.keys())
    proxsites_refe = list(assimprox_refe.keys())
    
    # loop over recon_times within user specified "year_range"
    ntimes, = recon_times.shape
    inds = np.where((recon_times>=year_range[0]) & (recon_times<=year_range[1]))
    inds_in_range = [it for i, it in np.ndenumerate(inds)]

    countit = 1
    for it in inds_in_range:

        year = int(recon_times[it])    
        print(' plotting:', year)


        # assimilated proxies
        ndots_test = 0
        if proxsites_test:
            time_range = (year-recon_interval/2., year+recon_interval/2.)
            lats = []
            lons = []
            for s in proxsites_test:
                inds, = np.where((assimprox_test[s]['years']>=time_range[0]) & (assimprox_test[s]['years']<=time_range[1]))
                if len(inds) > 0:
                    lats.append(assimprox_test[s]['lat'])
                    lons.append(assimprox_test[s]['lon'])
            plats_test = np.asarray(lats)
            plons_test = np.asarray(lons)
            ndots_test, = plats_test.shape


        ndots_refe = 0
        if proxsites_refe:
            time_range = (year-recon_interval/2., year+recon_interval/2.)
            lats = []
            lons = []
            for s in proxsites_refe:
                inds, = np.where((assimprox_refe[s]['years']>=time_range[0]) & (assimprox_refe[s]['years']<=time_range[1]))
                if len(inds) > 0:
                    lats.append(assimprox_refe[s]['lat'])
                    lons.append(assimprox_refe[s]['lon'])
            plats_refe = np.asarray(lats)
            plons_refe = np.asarray(lons)
            ndots_refe, = plats_refe.shape


            
        Xam2D_test = xam_test[it,:,:]
        Xam2D_test = np.ma.masked_invalid(Xam2D_test)
        nlat,nlon = Xam2D_test.shape

        Xam2D_refe = xam_refe[it,:,:]
        Xam2D_refe = np.ma.masked_invalid(Xam2D_refe)
        #nlat,nlon = Xam2D_refe.shape

        
        if np.unique(lat2).shape[0] == nlat and np.unique(lon2).shape[0] == nlon :
            # Regular lat/lon grid
            plotlat = lat2
            plotlon = lon2
            plotdata_test = Xam2D_test
            plotdata_refe = Xam2D_refe            
        else:
            # Irregular grid: simple regrid to regular lat-lon grid for plotting        
            longrid = np.linspace(0.,360.,nlon)
            latgrid = np.linspace(-90.,90.,nlat)
            datagrid = np.zeros((nlat,nlon))
            datagrid[:] = np.nan
            plotlon, plotlat = np.meshgrid(longrid, latgrid)

            inpoints = np.zeros(shape=[nlat*nlon, 2])
            inpoints[:,0] = lon2.flatten()
            inpoints[:,1] = lat2.flatten()

            values_rg_test = Xam2D_test.reshape((nlat*nlon))
            datagrid_test = griddata(inpoints,values_rg_test,(plotlon,plotlat),method='nearest',fill_value=np.nan) # nearest or linear
            plotdata_test = np.ma.masked_invalid(datagrid_test)

            values_rg_refe = Xam2D_refe.reshape((nlat*nlon))
            datagrid_refe = griddata(inpoints,values_rg_refe,(plotlon,plotlat),method='nearest',fill_value=np.nan) # nearest or linear
            plotdata_refe = np.ma.masked_invalid(datagrid_refe)

            
        # Generating the map...
        fig = plt.figure(figsize=[7,9])

        # test recon
        # ----------        
        ax = fig.add_subplot(3,1,1)

        m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 30.; lonres=30. 
        cbnds = [mapmin,mapint,mapmax];
        nlevs = 101
        cints = np.linspace(mapmin, mapmax, nlevs, endpoint=True)
        cs = m.contourf(plotlon,plotlat,plotdata_test,cints,cmap=plt.get_cmap(cmap),vmin=mapmin,vmax=mapmax,extend='both',latlon=True)
        cbarticks = np.linspace(cbnds[0],cbnds[2],num=int((cbnds[2]-cbnds[0])/cbnds[1])+1)
        cbar = m.colorbar(cs,location='right',pad="5%",ticks=cbarticks, extend='both',format=cbarfmt)
        m.drawmapboundary(fill_color = bckgcolor)
        m.drawcoastlines(); m.drawcountries()
        m.drawparallels(np.arange(-80.,81.,latres))
        m.drawmeridians(np.arange(-180.,181.,lonres))
        # Make sure continents appear filled-in for ocean fields
        if 'Omon' in var_to_plot or 'Odec' in var_to_plot: 
            m.fillcontinents(color=bckgcolor)

        plt.title(exp_test,fontsize=10)
        
        # dots marking sites of assimilated proxies
        if ndots_test > 0:
            x, y = m(plons_test,plats_test)
            #dotcolor = '#e6e9ef'
            dotcolor = '#42f4b3'
            m.scatter(x,y,10,marker='o',color=dotcolor,edgecolor='black',linewidth='.5',zorder=4)


        # refe recon
        # ----------
        ax = fig.add_subplot(3,1,2)

        m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 30.; lonres=30. 
        cbnds = [mapmin,mapint,mapmax];
        nlevs = 101
        cints = np.linspace(mapmin, mapmax, nlevs, endpoint=True)
        cs = m.contourf(plotlon,plotlat,plotdata_refe,cints,cmap=plt.get_cmap(cmap),vmin=mapmin,vmax=mapmax,extend='both',latlon=True)
        cbarticks = np.linspace(cbnds[0],cbnds[2],num=int((cbnds[2]-cbnds[0])/cbnds[1])+1)
        cbar = m.colorbar(cs,location='right',pad="5%",ticks=cbarticks, extend='both',format=cbarfmt)
        m.drawmapboundary(fill_color = bckgcolor)
        m.drawcoastlines(); m.drawcountries()
        m.drawparallels(np.arange(-80.,81.,latres))
        m.drawmeridians(np.arange(-180.,181.,lonres))
        # Make sure continents appear filled-in for ocean fields
        if 'Omon' in var_to_plot or 'Odec' in var_to_plot: 
            m.fillcontinents(color=bckgcolor)

        plt.title(exp_refe,fontsize=10)
            
        # dots marking sites of assimilated proxies
        if ndots_refe > 0:
            x, y = m(plons_refe,plats_refe)
            #dotcolor = '#e6e9ef'
            dotcolor = '#42f4b3'
            m.scatter(x,y,10,marker='o',color=dotcolor,edgecolor='black',linewidth='.5',zorder=4)


        # difference
        # ----------
        ax = fig.add_subplot(3,1,3)

        m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 30.; lonres=30. 
        cbnds = [mapmin,mapint,mapmax];
        nlevs = 101
        cints = np.linspace(mapmin, mapmax, nlevs, endpoint=True)
        cs = m.contourf(plotlon,plotlat,(plotdata_test-plotdata_refe),cints,cmap=plt.get_cmap(cmap),vmin=mapmin,vmax=mapmax,extend='both',latlon=True)
        cbarticks = np.linspace(cbnds[0],cbnds[2],num=int((cbnds[2]-cbnds[0])/cbnds[1])+1)
        cbar = m.colorbar(cs,location='right',pad="5%",ticks=cbarticks, extend='both',format=cbarfmt)
        m.drawmapboundary(fill_color = bckgcolor)
        m.drawcoastlines(); m.drawcountries()
        m.drawparallels(np.arange(-80.,81.,latres))
        m.drawmeridians(np.arange(-180.,181.,lonres))

        # Make sure continents appear filled-in for ocean fields
        if 'Omon' in var_to_plot or 'Odec' in var_to_plot: 
            m.fillcontinents(color=bckgcolor)

        plt.title('Difference between reconstructions',fontsize=12)
        

        fig.tight_layout()
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.93, wspace=0.5, hspace=0.5)
        plt.suptitle(var_to_plot+', Year:'+str(year),fontsize=12,fontweight='bold')        

        plt.savefig('%s/%s_vs_%s_%s_%syr.png' % (figdir,exp_test,exp_refe,var_to_plot,year),bbox_inches='tight')
        if make_movie:
                plt.savefig('%s/fig_%s.png' % (figdir,str("{:06d}".format(countit))),bbox_inches='tight')
                # to make it look like a pause at end of animation
                if it == inds_in_range[-1]:
                    nbextraframes = 5
                    for i in range(nbextraframes):
                        plt.savefig('%s/fig_%s.png' % (figdir,str("{:06d}".format(countit+i+1))),bbox_inches='tight')
        plt.close()
        
        countit += 1


    
    if make_movie:
        # create the animation
        # check if old files are there, if yes, remove
        fname = '%s/%s_%s_anim_%sto%s' %(figdir,exp,var_to_plot,str(year_range[0]),str(year_range[1]))    
        if os.path.exists(fname+'.gif'):
            os.system('rm -f %s.gif' %fname)
        if os.path.exists(fname+'.mp4'):
            os.system('rm -f %s.mp4' %fname)

        os.system('convert -delay 50 -loop 100 %s/fig_*.png %s.gif' %(figdir,fname))
        os.system('ffmpeg -r 3 -i %s/fig_%s.png %s.mp4' %(figdir,'%06d', fname))

        # clean up temporary files
        os.system('rm -f %s/fig_*.png' %(figdir))

