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

def read_gridded_data(expdir,dirset,var):

        # read ensemble mean data
        print('\n reading LMR ensemble-mean data...\n')

        # check if variable is in the reanalysis data
        indir = expdir + '/' + dirset[0]
        infile = 'ensemble_mean_'+var+'.npz'
        filename = os.path.join(indir, infile)
        
        if not os.path.isfile(filename):
            print('Variable %s not in the available set of reanalysis variables. Skipping.' %var)
             
        first = True
        k = -1
        for dir in dirset:
            k = k + 1
            # Posterior (reconstruction)
            ensfiln = expdir + '/' + dir + '/ensemble_mean_'+var+'.npz'
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
            posbeg = state_info[var]['pos'][0]
            posend = state_info[var]['pos'][1]
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
                nlatp = state_info[var]['spacedims'][0]
                nlonp = state_info[var]['spacedims'][1]
                xbm_all = np.zeros([niters,nyrs,nlatp,nlonp])

            xam = xam + tmp
            xam_all[k,:,:,:] = tmp

            # prior ensemble mean of MC iteration "k"
            tmpp = np.mean(tas_prior,axis=1)
            xbm_all[k,:,:,:] = tmpp.reshape(nlatp,nlonp)

        # Prior sample mean over all MC iterations
        xbm = xbm_all.mean(0)
        xbm_var = xbm_all.var(0)

        # Posterior
        #  this is the sample mean computed with low-memory accumulation
        xam = xam/niters
        #  this is the sample mean computed with numpy on all data
        xam_check = xam_all.mean(0)
        #  check..
        max_err = np.max(np.max(np.max(xam_check - xam)))
        if max_err > 1e-4:
            print('max error = ' + str(max_err))
            raise Exception('WARNING: sample mean does not match what is in the ensemble files!')

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

        return xam,lat,lon,recon_times

sys.path.append('../')
from LMR_plot_support import truncate_colormap
from LMR_utils import global_hemispheric_means, natural_sort, PAGES2K_regional_means

mapcolor = truncate_colormap(plt.cm.jet,0.15,1.0)


# ------------------------------------------------
# --- Begin section of user-defined parameters ---

#datadir = '/Users/hakim/data/LMR/archive'
#datadir = '/home/disk/kalman2/wperkins/LMR_output/archive'
#datadir = '/home/disk/kalman3/rtardif/LMR/output'
#datadir = '/home/disk/ekman4/rtardif/LMR/output'
datadir = '/Users/hakim/data/LMR_python3/archive/'

#exp = 'production_mlost_ccsm4_pagesall_0.75'
#exp = 'production_mlost_era20c_pagesall_0.75'
#exp = 'production_mlost_era20cm_pagesall_0.75'
#exp = 'production_gis_ccsm4_pagesall_0.75'
exp = 'dadt_full_prior'

# --
#exp = 'test'
# --

var = 'tas_sfc_Adec'

year_range = [0,2000]
#year_range = [1850,2000]
#year_range = [-25000,2000]
#year_range = [-115000,2000]

# --
# MC realizations to consider.
#  All available : MCset = None
#  or over a custom selection ( MCset = (begin,end) )
#  ex. MCset = (0,0)    -> only the first MC run
#      MCset = (0,10)   -> the first 11 MC runs (from 0 to 10 inclusively)
#      MCset = (80,100) -> the 80th to 100th MC runs (21 realizations)
MCset = None
#MCset = (0,10)
# --

expdir = datadir + '/'+exp

print('experiment directory...',expdir)

# check if the experiment directory exists
if not os.path.isdir(expdir):
    print ('Experiment directory is not found! Please verify'
           ' your settings in this verification module.')
    raise SystemExit()

# Check if directory where figures will be generated exists
figdir = expdir+'/VerifFigs'
if not os.path.isdir(figdir):
    os.chdir(expdir)
    os.system('mkdir VerifFigs')

# get a listing of all available MC iteration directories
dirs = glob.glob(expdir+"/r*")
mcdir = [item.split('/')[-1] for item in dirs]

# Make sure list is properly sorted
mcdir = natural_sort(mcdir)

# Keep those specified through MCset
if MCset:
    targetlist = ['r'+str(item) for item in range(MCset[0],MCset[1]+1)]
    dirset = [item for item in mcdir if item in targetlist]
    if len(dirset) != len(targetlist):
        print('*** Problem with MCset: Not all specified iterations are available. Exiting!')
        raise SystemExit()
else:
    dirset = mcdir

niters = len(dirset)
print('dirset:', dirset)
print('niters = ', niters)

xam,lat,lon,time = read_gridded_data(expdir,dirset,var)

print('shape of xam...',xam.shape)
print('shape of lat:',lat.shape)
print('shape of lon:',lon.shape)

#aaa = global_hemispheric_means(xam,)
rm =PAGES2K_regional_means(xam,lat,lon)
print('rm shape...',rm.shape)

plt.plot(time,rm[7,:])
plt.savefig('greenland_dadt.png',dpi=300)
