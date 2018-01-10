"""
Module: LMR_plot_output.py

Purpose: Plotting results from the LMR paleoclimate reanalysis.

Originator: Robert Tardif - Univ. of Washington, Dept. of Atmospheric Sciences
            February 2017

Revisions: 
            - Added option to make 2d maps of selected variable for individual years and create animation 
              of the reconstruction.
              [R. Tardif, U. of Washington - May 2017]
            - Now plots the time series of NH and SH mean temperatures. 
              [R. Tardif, U. of Washington - June 2017]

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

datadir = '/Users/hakim/data/LMR/archive'
#datadir = '/home/disk/kalman2/wperkins/LMR_output/archive'
#datadir = '/home/disk/kalman3/rtardif/LMR/output'
#datadir = '/home/disk/ekman4/rtardif/LMR/output'


#exp = 'production_mlost_ccsm4_pagesall_0.75'
#exp = 'production_mlost_era20c_pagesall_0.75'
#exp = 'production_mlost_era20cm_pagesall_0.75'
#exp = 'production_gis_ccsm4_pagesall_0.75'
# --
exp = 'test'


#year_range = [1800,2000]
year_range = [0,2000]
#year_range = [-25000,2000]
#year_range = [-115000,2000]

# --
#iter_range = [0,0]
iter_range = [0,100]

# options of which figures to produce
make_gmt_plot  = True
make_map_plots = False

# for maps (if make_map_plots is set to True):
show_assimilated_proxies = True
make_movie = False


# ==== for GMT timeseries plot:
# -- anomalies --
pltymin = -1.5; pltymax = 1.5; ylabel = 'Temperature anomaly (K)'
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

expdir = datadir + '/'+exp

# check if the experiment directory exists
if not os.path.isdir(expdir):
    print ('Experiment directory is not found! Please verify'
           ' your settings in this verification module.')
    raise SystemExit()


figdir = expdir+'/VerifFigs'
if not os.path.isdir(figdir):
    os.chdir(expdir)
    os.system('mkdir VerifFigs')


if make_gmt_plot:
    
    # ======================================================
    # 1) Time series of global mean temperature
    # ======================================================

    list_iters = []
    count = 0
    gmt_present = False
    iters = np.arange(iter_range[0], iter_range[1]+1)
    for iter in iters:
        dirname = expdir+'/r'+str(iter)
        print(iter, dirname)
        list_iters.append(dirname)
        # check presence of gmt file 
        if os.path.exists(dirname+'/'+infile+'.npz'): count +=1
    nbiters = len(list_iters)
    if count == nbiters: gmt_present = True


    if gmt_present:
        # get array dimensions
        gmt_data     = np.load(list_iters[0]+'/'+infile+'.npz')
        recon_times  = gmt_data['recon_times']

        if infile == 'gmt':
            recon_gmt_data     = gmt_data['gmt_save']
            [nbproxy, nbtimes] = recon_gmt_data.shape
            nens = 1
            file_to_read = 'gmt_save'
        elif infile == 'gmt_ensemble':
            recon_gmt_data     = gmt_data['gmt_ensemble']
            [nbtimes, nens] = recon_gmt_data.shape
            nbproxy = 0
            file_to_read = 'gmt_ensemble'
        else:
            print('Error in infile! Exiting!')
            SystemExit(1)


        # Declare arrays
        recon_years  = np.zeros([nbiters,nbtimes])
        recon_gmt    = np.zeros([nbiters,nens,nbtimes])
        recon_nhmt   = np.zeros([nbiters,nens,nbtimes])
        recon_shmt   = np.zeros([nbiters,nens,nbtimes])
        prior_gmt    = np.zeros([nbiters,nens,nbtimes])
        prior_nhmt   = np.zeros([nbiters,nens,nbtimes])
        prior_shmt   = np.zeros([nbiters,nens,nbtimes])
        # init. with nan's
        recon_gmt[:]  = np.nan
        recon_nhmt[:] = np.nan
        recon_shmt[:] = np.nan
        prior_gmt[:] = np.nan
        prior_nhmt[:] = np.nan
        prior_shmt[:] = np.nan
        
        # Read-in the data : loop over MC iters
        citer = 0
        for d in list_iters:

            # File of global mean values
            fname = d+'/'+infile+'.npz'
            gmt_data = np.load(fname)

            recon_years[citer,:] = gmt_data['recon_times']

            if infile == 'gmt':
                recon_gmt_data = gmt_data[file_to_read]
                [nbproxy, nbtimes] = recon_gmt_data.shape
                # Final reconstruction
                recon_gmt[citer,0,:] = recon_gmt_data[nbproxy-1]

            elif infile == 'gmt_ensemble':
                # Full ensemble reconstruction
                # Global mean
                recon_gmt_data = gmt_data[file_to_read]
                recon_gmt[citer,:,:] = recon_gmt_data.T # flip time/nens dims
                # NH mean
                recon_data = gmt_data['nhmt_ensemble']
                recon_nhmt[citer,:,:] = recon_data.T # flip time/nens dims
                # SH mean
                recon_data = gmt_data['shmt_ensemble']
                recon_shmt[citer,:,:] = recon_data.T # flip time/nens dims
                
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

                prior_gmt[citer,:,:]  = np.repeat(gmt[:,np.newaxis],nbtimes,1)
                prior_nhmt[citer,:,:] = np.repeat(nhmt[:,np.newaxis],nbtimes,1)
                prior_shmt[citer,:,:] = np.repeat(shmt[:,np.newaxis],nbtimes,1)                
                
            citer = citer + 1


        if nbiters > 1:
            # Reshaping arrays for easier calculation of stats over the "grand" ensemble (MC iters + DA ensemble members)
            gmpp = prior_gmt.transpose(2,0,1).reshape(nbtimes,-1)
            gmpr = recon_gmt.transpose(2,0,1).reshape(nbtimes,-1)

            nhmpp = prior_nhmt.transpose(2,0,1).reshape(nbtimes,-1)
            nhmpr = recon_nhmt.transpose(2,0,1).reshape(nbtimes,-1)

            shmpp = prior_shmt.transpose(2,0,1).reshape(nbtimes,-1)
            shmpr = recon_shmt.transpose(2,0,1).reshape(nbtimes,-1)

        else:
            gmpp = np.squeeze(prior_gmt).transpose()
            gmpr = np.squeeze(recon_gmt).transpose()

            nhmpp = np.squeeze(prior_nhmt).transpose()
            nhmpr = np.squeeze(recon_nhmt).transpose()

            shmpp = np.squeeze(prior_shmt).transpose()
            shmpr = np.squeeze(recon_shmt).transpose()


        # Prior
        prior_gmt_ensmean    = np.mean(gmpp,axis=1)
        prior_gmt_ensmin     = np.amin(gmpp,axis=1)
        prior_gmt_ensmax     = np.amax(gmpp,axis=1)
        prior_gmt_enssprd    = np.std(gmpp,axis=1)
        prior_gmt_enslowperc = np.percentile(gmpp,5,axis=1)
        prior_gmt_ensuppperc = np.percentile(gmpp,95,axis=1)

        prior_nhmt_ensmean    = np.mean(nhmpp,axis=1)
        prior_nhmt_ensmin     = np.amin(nhmpp,axis=1)
        prior_nhmt_ensmax     = np.amax(nhmpp,axis=1)
        prior_nhmt_enssprd    = np.std(nhmpp,axis=1)
        prior_nhmt_enslowperc = np.percentile(nhmpp,5,axis=1)
        prior_nhmt_ensuppperc = np.percentile(nhmpp,95,axis=1)

        prior_shmt_ensmean    = np.mean(shmpp,axis=1)
        prior_shmt_ensmin     = np.amin(shmpp,axis=1)
        prior_shmt_ensmax     = np.amax(shmpp,axis=1)
        prior_shmt_enssprd    = np.std(shmpp,axis=1)
        prior_shmt_enslowperc = np.percentile(shmpp,5,axis=1)
        prior_shmt_ensuppperc = np.percentile(shmpp,95,axis=1)
        
        # Posterior
        recon_gmt_ensmean    = np.mean(gmpr,axis=1)
        recon_gmt_ensmin     = np.amin(gmpr,axis=1)
        recon_gmt_ensmax     = np.amax(gmpr,axis=1)
        recon_gmt_enssprd    = np.std(gmpr,axis=1)
        recon_gmt_enslowperc = np.percentile(gmpr,5,axis=1)
        recon_gmt_ensuppperc = np.percentile(gmpr,95,axis=1)

        recon_nhmt_ensmean    = np.mean(nhmpr,axis=1)
        recon_nhmt_ensmin     = np.amin(nhmpr,axis=1)
        recon_nhmt_ensmax     = np.amax(nhmpr,axis=1)
        recon_nhmt_enssprd    = np.std(nhmpr,axis=1)
        recon_nhmt_enslowperc = np.percentile(nhmpr,5,axis=1)
        recon_nhmt_ensuppperc = np.percentile(nhmpr,95,axis=1)

        recon_shmt_ensmean    = np.mean(shmpr,axis=1)
        recon_shmt_ensmin     = np.amin(shmpr,axis=1)
        recon_shmt_ensmax     = np.amax(shmpr,axis=1)
        recon_shmt_enssprd    = np.std(shmpr,axis=1)
        recon_shmt_enslowperc = np.percentile(shmpr,5,axis=1)
        recon_shmt_ensuppperc = np.percentile(shmpr,95,axis=1)
        
        
        # => plot +/- spread in the various realizations
        #recon_gmt_upp = recon_gmt_ensmean + recon_gmt_enssprd
        #recon_gmt_low = recon_gmt_ensmean - recon_gmt_enssprd
        #prior_gmt_upp = prior_gmt_ensmean + prior_gmt_enssprd
        #prior_gmt_low = prior_gmt_ensmean - prior_gmt_enssprd
        #
        #recon_nhmt_upp = recon_nhmt_ensmean + recon_nhmt_enssprd
        #recon_nhmt_low = recon_nhmt_ensmean - recon_nhmt_enssprd
        #prior_nhmt_upp = prior_nhmt_ensmean + prior_nhmt_enssprd
        #prior_nhmt_low = prior_nhmt_ensmean - prior_nhmt_enssprd
        #
        #recon_shmt_upp = recon_shmt_ensmean + recon_shmt_enssprd
        #recon_shmt_low = recon_shmt_ensmean - recon_shmt_enssprd
        #prior_shmt_upp = prior_shmt_ensmean + prior_shmt_enssprd
        #prior_shmt_low = prior_shmt_ensmean - prior_shmt_enssprd        

        # => plot +/- min-max among the various realizations
        #recon_gmt_upp = recon_gmt_ensmax
        #recon_gmt_low = recon_gmt_ensmin
        #prior_gmt_upp = prior_gmt_ensmax
        #prior_gmt_low = prior_gmt_ensmin
        #
        #recon_nhmt_upp = recon_nhmt_ensmax
        #recon_nhmt_low = recon_nhmt_ensmin
        #prior_nhmt_upp = prior_nhmt_ensmax
        #prior_nhmt_low = prior_nhmt_ensmin
        #
        #recon_shmt_upp = recon_shmt_ensmax
        #recon_shmt_low = recon_shmt_ensmin
        #prior_shmt_upp = prior_shmt_ensmax
        #prior_shmt_low = prior_shmt_ensmin
        
        # => plot +/- 5-95 percentiles among the various realizations
        recon_gmt_low = recon_gmt_enslowperc
        recon_gmt_upp = recon_gmt_ensuppperc
        prior_gmt_low = prior_gmt_enslowperc
        prior_gmt_upp = prior_gmt_ensuppperc

        recon_nhmt_low = recon_nhmt_enslowperc
        recon_nhmt_upp = recon_nhmt_ensuppperc
        prior_nhmt_low = prior_nhmt_enslowperc
        prior_nhmt_upp = prior_nhmt_ensuppperc

        recon_shmt_low = recon_shmt_enslowperc
        recon_shmt_upp = recon_shmt_ensuppperc
        prior_shmt_low = prior_shmt_enslowperc
        prior_shmt_upp = prior_shmt_ensuppperc
        
        
        # -----------------------------------------------
        # Plotting time series of global mean temperature
        # -----------------------------------------------

        plt.rcParams['font.weight'] = 'bold'    # set the font weight globally

        #fig = plt.figure(figsize=[10,6])
        fig = plt.figure()

        p1 = plt.plot(recon_years[0,:],recon_gmt_ensmean,'-b',linewidth=2, label='Posterior')
        plt.fill_between(recon_years[0,:], recon_gmt_low, recon_gmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_years[0,:],prior_gmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        plt.fill_between(recon_years[0,:], prior_gmt_low, prior_gmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        plt.suptitle(exp, fontsize=12)
        plt.title('Global mean temperature', fontsize=12,fontweight='bold')
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')
        plt.axis((year_range[0],year_range[1],pltymin,pltymax))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        plt.savefig('%s/%s_GMT_%sto%syrs.png' % (figdir,exp,str(year_range[0]),str(year_range[1])),bbox_inches='tight')
        plt.close()
        #plt.show()

        # -------------------------------------------
        # Plotting time series of NH mean temperature
        # -------------------------------------------

        #fig = plt.figure(figsize=[10,6])
        fig = plt.figure()

        p1 = plt.plot(recon_years[0,:],recon_nhmt_ensmean,'-b',linewidth=2, label='Posterior')
        plt.fill_between(recon_years[0,:], recon_nhmt_low, recon_nhmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_years[0,:],prior_nhmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        plt.fill_between(recon_years[0,:], prior_nhmt_low, prior_nhmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        plt.suptitle(exp, fontsize=12)
        plt.title('NH mean temperature', fontsize=12,fontweight='bold')
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')
        plt.axis((year_range[0],year_range[1],pltymin,pltymax))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        plt.savefig('%s/%s_NHMT_%sto%syrs.png' % (figdir,exp,str(year_range[0]),str(year_range[1])),bbox_inches='tight')
        plt.close()
        #plt.show()

        # -------------------------------------------
        # Plotting time series of SH mean temperature
        # -------------------------------------------

        #fig = plt.figure(figsize=[10,6])
        fig = plt.figure()

        p1 = plt.plot(recon_years[0,:],recon_shmt_ensmean,'-b',linewidth=2, label='Posterior')
        plt.fill_between(recon_years[0,:], recon_shmt_low, recon_shmt_upp,facecolor='blue',alpha=0.2,linewidth=0.0)
        xmin,xmax,ymin,ymax = plt.axis()
        p2 = plt.plot(recon_years[0,:],prior_shmt_ensmean,'-',color='black',linewidth=2,label='Prior')
        plt.fill_between(recon_years[0,:], prior_shmt_low, prior_shmt_upp,facecolor='black',alpha=0.2,linewidth=0.0)

        p0 = plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
        plt.suptitle(exp, fontsize=12)
        plt.title('SH mean temperature', fontsize=12,fontweight='bold')
        plt.xlabel('Year (BC/AD)',fontsize=12,fontweight='bold')
        plt.ylabel(ylabel,fontsize=12,fontweight='bold')
        plt.axis((year_range[0],year_range[1],pltymin,pltymax))
        plt.legend( loc='lower right', numpoints = 1,fontsize=12)

        plt.savefig('%s/%s_SHMT_%sto%syrs.png' % (figdir,exp,str(year_range[0]),str(year_range[1])),bbox_inches='tight')
        plt.close()
        #plt.show()
        

if make_map_plots:    

    # ======================================================
    # Plots of reconstructed spatial fields
    # ======================================================

    # get a listing of the iteration directories
    dirs = glob.glob(expdir+"/r*")

    mcdir = [item.split('/')[-1] for item in dirs]
    niters = len(mcdir)

    print('mcdir:' + str(mcdir))
    print('niters = ' + str(niters))

    # for info on assimilated proxies
    assimprox = {}

    # read ensemble mean data
    print('\n reading LMR ensemble-mean data...\n')

    first = True
    k = -1
    for dir in mcdir:
        k = k + 1
        # Posterior (reconstruction)
        ensfiln = expdir + '/' + dir + '/ensemble_mean_'+var_to_plot+'.npz'
        npzfile = np.load(ensfiln)
        print(npzfile.files)
        tmp = npzfile['xam']
        print('shape of tmp: ' + str(np.shape(tmp)))

        # load prior data
        file_prior = expdir + '/' + dir + '/Xb_one.npz'
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

            xam = np.zeros([nyrs,np.shape(tmp)[1],np.shape(tmp)[2]])
            xam_all = np.zeros([niters,nyrs,np.shape(tmp)[1],np.shape(tmp)[2]])
            # prior
            [_,Nens] = tas_prior.shape
            nlatp = state_info[var_to_plot]['spacedims'][0]
            nlonp = state_info[var_to_plot]['spacedims'][1]
            xbm_all = np.zeros([niters,nyrs,nlatp,nlonp])

        xam = xam + tmp
        xam_all[k,:,:,:] = tmp

        # prior ensemble mean of MC iteration "k"
        tmpp = np.mean(tas_prior,axis=1)
        xbm_all[k,:,:,:] = tmpp.reshape(nlatp,nlonp)


        # info on assimilated proxies ---
        assimproxfiln = expdir + '/' + dir + '/assimilated_proxies.npy'

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

                if ptag not in assimprox.keys():
                    assimprox[ptag] = {}
                    assimprox[ptag]['lat']   = plat
                    assimprox[ptag]['lon']   = plon
                    assimprox[ptag]['years'] = yrs.astype('int')
                    assimprox[ptag]['iters'] = [k]
                else:
                    assimprox[ptag]['iters'].append(k)


    # Prior sample mean over all MC iterations
    xbm = xbm_all.mean(0)
    xbm_var = xbm_all.var(0)

    # Posterior
    #  this is the sample mean computed with low-memory accumulation
    xam = xam/len(mcdir)
    #  this is the sample mean computed with numpy on all data
    xam_check = xam_all.mean(0)
    #  check..
    max_err = np.max(np.max(np.max(xam_check - xam)))
    if max_err > 1e-4:
        print('max error = ' + str(max_err))
        raise Exception('sample mean does not match what is in the ensemble files!')

    # sample variance
    xam_var = xam_all.var(0)
    print(np.shape(xam_var))

    print(' shape of the ensemble array: ' + str(np.shape(xam_all)) +'\n')
    print(' shape of the ensemble-mean array: ' + str(np.shape(xam)) +'\n')
    print(' shape of the ensemble-mean prior array: ' + str(np.shape(xbm)) +'\n')

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
    proxsites = list(assimprox.keys())

    # loop over recon_times within user specified "year_range"
    ntimes, = recon_times.shape
    inds = np.where((recon_times>=year_range[0]) & (recon_times<=year_range[1]))
    inds_in_range = [it for i, it in np.ndenumerate(inds)]

    countit = 1
    for it in inds_in_range:

        year = int(recon_times[it])    
        print(' plotting:', year)


        # assimilated proxies
        ndots = 0
        if proxsites:
            time_range = (year-recon_interval/2., year+recon_interval/2.)
            lats = []
            lons = []
            for s in proxsites:
                inds, = np.where((assimprox[s]['years']>=time_range[0]) & (assimprox[s]['years']<=time_range[1]))
                if len(inds) > 0:
                    lats.append(assimprox[s]['lat'])
                    lons.append(assimprox[s]['lon'])
            plats = np.asarray(lats)
            plons = np.asarray(lons)
            ndots, = plats.shape


        Xam2D = xam[it,:,:]
        Xam2D = np.ma.masked_invalid(Xam2D)
        nlat,nlon = Xam2D.shape

        if np.unique(lat2).shape[0] == nlat and np.unique(lon2).shape[0] == nlon :
            # Regular lat/lon grid
            plotlat = lat2
            plotlon = lon2
            plotdata = Xam2D
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
            values_rg = Xam2D.reshape((nlat*nlon))
            datagrid = griddata(inpoints,values_rg,(plotlon,plotlat),method='nearest',fill_value=np.nan) # nearest or linear
            plotdata = np.ma.masked_invalid(datagrid)


        # Generating the map...
        fig = plt.figure()
        ax  = fig.add_axes([0.1,0.1,0.8,0.8])
        m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=30. 

        cbnds = [mapmin,mapint,mapmax];
        nlevs = 101
        cints = np.linspace(mapmin, mapmax, nlevs, endpoint=True)
        #cs = m.pcolormesh(plotlon,plotlat,plotdata,shading='flat',vmin=mapmin,vmax=mapmax,cmap=cmap,latlon=True)
        cs = m.contourf(plotlon,plotlat,plotdata,cints,cmap=plt.get_cmap(cmap),vmin=mapmin,vmax=mapmax,extend='both',latlon=True)

        cbarticks = np.linspace(cbnds[0],cbnds[2],num=int((cbnds[2]-cbnds[0])/cbnds[1])+1)
        cbar = m.colorbar(cs,location='bottom',pad="5%",ticks=cbarticks, extend='both',format=cbarfmt)

        m.drawmapboundary(fill_color = bckgcolor)
        m.drawcoastlines(); m.drawcountries()
        m.drawparallels(np.arange(-80.,81.,latres))
        m.drawmeridians(np.arange(-180.,181.,lonres))
        plt.title(var_to_plot+', Year:'+str(year),fontsize=14,fontweight='bold')
        # Make sure continents appear filled-in for ocean fields
        if 'Omon' in var_to_plot or 'Odec' in var_to_plot: 
            m.fillcontinents(color=bckgcolor)

        # dots marking sites of assimilated proxies
        if ndots > 0:
            x, y = m(plons,plats)
            dotcolor = '#e6e9ef'
            m.scatter(x,y,10,marker='o',color=dotcolor,edgecolor='black',linewidth='.5',zorder=4)
            #m.scatter(x,y,25,marker='o',facecolor='none',edgecolor='black',linewidth='1',zorder=4,alpha=0.5)



                    
        plt.savefig('%s/%s_%s_%syr.png' % (figdir,exp,var_to_plot,year),bbox_inches='tight')
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

