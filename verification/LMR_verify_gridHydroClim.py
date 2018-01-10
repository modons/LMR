""" 
Module: LMR_verify_gridHydroClim.py

Purpose: Generates spatial verification statistics of (some) of LMR gridded hydroclimate variables 
         (if available in reconstructed data) against related gridded historical instrumental datasets.
         The current code handles two hydroclimate variables:
         - scpdsi (self-calibrated PDSI) w/ verification against DaiPDSI dataset
         - pr (precipitation (flux in kg m-2 s-1) w/ verification against GPCC dataset

Originator: Robert Tardif, U. of Washington, December 2017

Revisions: 

"""
import matplotlib
# need to do this when running remotely, and to suppress figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# need to do this backend when running remotely or to suppress figures interactively

# generic imports
import numpy as np
import glob, os, sys
from datetime import datetime, timedelta

import mpl_toolkits.basemap as bm
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import from_levels_and_colors
from matplotlib import rc
from spharm import Spharmt, getspecindx, regrid
import pickle
import warnings

# LMR specific imports
sys.path.append('../')
from LMR_utils import global_hemispheric_means, assimilated_proxies, coefficient_efficiency
from load_gridded_data import read_gridded_data_DaiPDSI
from load_gridded_data import read_gridded_data_GPCC
from LMR_plot_support import *

# change default value of latlon kwarg to True.
bm.latlon_default = True


warnings.filterwarnings('ignore')

##################################
# START:  set user parameters here
##################################

# option to suppress figures
#iplot = False
iplot = True

# centered time mean (nya must be odd! 3 = 3 yr mean; 5 = 5 year mean; etc 0 = none)
nya = 0

# option to print figures
fsave = True
#fsave = False

# save statistics file
stat_save = True

# set paths, the filename for plots, and global plotting preferences

# file specification
#
# current datasets
#
# ---
#nexp = 'production_gis_ccsm4_pagesall_0.75'
#nexp = 'production_mlost_ccsm4_pagesall_0.75'
#nexp = 'production_cru_ccsm4_pagesall_0.75'
#nexp = 'production_mlost_era20c_pagesall_0.75'
#nexp = 'production_mlost_era20cm_pagesall_0.75'
# ---
nexp = 'test'
# ---

# perform verification using all recon. MC realizations ( MCset = None )
# or over a custom selection ( MCset = (begin,end) )
# ex. MCset = (0,0)    -> only the first MC run
#     MCset = (0,10)   -> the first 11 MC runs (from 0 to 10 inclusively)
#     MCset = (80,100) -> the 80th to 100th MC runs (21 realizations)
MCset = None
#MCset = (0,0)

# override datadir
#datadir_output = './data/'
#datadir_output = '/home/disk/kalman3/hakim/LMR'
#datadir_output = '/home/katabatic2/wperkins/LMR_output/testing'
datadir_output = '/home/disk/kalman3/rtardif/LMR/output'
#datadir_output = '/home/disk/kalman3/hakim/LMR'
#datadir_output = '/home/disk/kalman2/wperkins/LMR_output/archive'

# Directory where verification data can be found
datadir_verif = '/home/disk/kalman3/rtardif/LMR/data/analyses'

# Verification datafiles
datafile_verif_PDSI   = 'Dai_pdsi.mon.mean.selfcalibrated_185001-201412.nc'
datafile_verif_GPCC   = 'GPCC_precip.mon.flux.1x1.v6.nc'

# threshold for fraction of valid data in calculation of verif. stats
valid_frac = 0.5

# time range for verification (in years CE)
trange = [1880,2000] #works for nya = 0
#trange = [1885,1995] #works for nya = 5
#trange = [1890,1990] #works for nya = 10
#trange = [1900,2000]

# reference period over which mean is calculated & subtracted 
# from all other datasets (in years CE)
#ref_period = [1951,1980] # ref. period for GIS & BE
#ref_period = [1961,1990] # ref. period for CRU & MLOST
ref_period = [1900, 1999] # 20th century

# set the default size of the figure in inches. ['figure.figsize'] = width, height;  
# aspect ratio appears preserved on smallest of the two
plt.rcParams['figure.figsize'] = 10, 10 # that's default image size for this interactive session
plt.rcParams['axes.linewidth'] = 2.0    # set the value globally
plt.rcParams['font.weight'] = 'bold'    # set the font weight globally
plt.rcParams['font.size'] = 11          # set the font size globally
#plt.rc('text', usetex=True)
plt.rc('text', usetex=False)

# number of contours for plots
nlevs = 21
# plot alpha transparency
alpha = 0.5

##################################
# END:  set user parameters here
##################################

# variables---this script is scpdsi and precip. only!
varPDSI = 'scpdsi_sfc_Amon'; varPDSI_label = 'PDSI'
varPRCP = 'pr_sfc_Amon'; varPRCP_label = 'PRCP'


workdir = datadir_output + '/' + nexp
print('working directory = %s' %workdir)

print('\n getting file system information...\n')

# get number of mc realizations from directory count
# RT: modified way to determine list of directories with mc realizations
# get a listing of the iteration directories
dirs = glob.glob(workdir+"/r*")

# selecting  the MC iterations to keep
if MCset:
    dirset = dirs[MCset[0]:MCset[1]+1]
else:
    dirset = dirs

mcdir = [item.split('/')[-1] for item in sorted(dirset)]
niters = len(mcdir)

print('mcdir: %s' % str(mcdir))
print('niters = %s' % str(niters))

# read ensemble mean data
print('\n reading LMR ensemble-mean data...\n')

# check if all requested variables are available
availablePDSI = False
availablePRCP = False

# Load prior data in first MC directory
file_prior = workdir + '/' + mcdir[0] + '/Xb_one.npz'
Xprior_statevector = np.load(file_prior)
Xb_one = Xprior_statevector['Xb_one']
# extract variable (sfc temperature) from state vector
state_info = Xprior_statevector['state_info'].item()
print(state_info)
variables_available = list(state_info.keys())
if varPDSI in variables_available: availablePDSI = True
if varPRCP in variables_available: availablePRCP = True

first = True
k = -1
for dir in mcdir:
    k = k + 1
    
    # Load prior data
    file_prior = workdir + '/' + dir + '/Xb_one.npz'
    Xprior_statevector = np.load(file_prior)
    Xb_one = Xprior_statevector['Xb_one']
    # extract variable (sfc temperature) from state vector
    state_info = Xprior_statevector['state_info'].item()


    # Posterior (reconstruction)
    # --------------------------    
    if availablePDSI:
        ensfiln = workdir + '/' + dir + '/ensemble_mean_'+varPDSI+'.npz'
        npzfile1 = np.load(ensfiln)
        #print  npzfile1.files
        dvarPDSI = npzfile1['xam']
        print('dir: %s :: shape of dvarPDSI: %s' % (dir,str(np.shape(dvarPDSI))))
        # prior
        posbeg = state_info[varPDSI]['pos'][0]
        posend = state_info[varPDSI]['pos'][1]
        varPDSI_prior = Xb_one[posbeg:posend+1,:]


    if availablePRCP:
        ensfiln = workdir + '/' + dir + '/ensemble_mean_'+varPRCP+'.npz'
        npzfile2 = np.load(ensfiln)
        #print  npzfile2.files
        dvarPRCP = npzfile2['xam']
        print('dir: %s :: shape of dvarPRCP: %s' % (dir,str(np.shape(dvarPRCP))))
        # prior
        posbeg = state_info[varPRCP]['pos'][0]
        posend = state_info[varPRCP]['pos'][1]
        varPRCP_prior = Xb_one[posbeg:posend+1,:]

    
    if first:
        first = False
        if availablePRCP:
            npzfile = npzfile2
        elif availablePDSI:
            npzfile = npzfile1
        else:
            raise SystemExit('Neither target variables are present in this reconstruction dataset. Exiting!')

        lat = npzfile['lat']
        lon = npzfile['lon']
        nlat = npzfile['nlat']
        nlon = npzfile['nlon']
        lat2 = np.reshape(lat,(nlat,nlon))
        lon2 = np.reshape(lon,(nlat,nlon))

        recon_times = npzfile['years']
        LMR_time = np.array(list(map(int,recon_times)))
        years = recon_times
        nyrs =  len(years)

        # initialize data arrays
        if availablePDSI:
            xamPDSI = np.zeros([nyrs,np.shape(dvarPDSI)[1],np.shape(dvarPDSI)[2]])
            xamPDSI_all = np.zeros([niters,nyrs,np.shape(dvarPDSI)[1],np.shape(dvarPDSI)[2]])
            # prior
            [_,Nens] = varPDSI_prior.shape
            nlatp = state_info[varPDSI]['spacedims'][0]
            nlonp = state_info[varPDSI]['spacedims'][1]
            xbmPDSI_all = np.zeros([niters,nyrs,nlatp,nlonp])
            
        if availablePRCP:
            xamPRCP = np.zeros([nyrs,np.shape(dvarPRCP)[1],np.shape(dvarPRCP)[2]])
            xamPRCP_all = np.zeros([niters,nyrs,np.shape(dvarPRCP)[1],np.shape(dvarPRCP)[2]])
            # prior
            [_,Nens] = varPRCP_prior.shape
            nlatp = state_info[varPRCP]['spacedims'][0]
            nlonp = state_info[varPRCP]['spacedims'][1]
            xbmPRCP_all = np.zeros([niters,nyrs,nlatp,nlonp])        


    if availablePDSI:
        xamPDSI = xamPDSI + dvarPDSI
        xamPDSI_all[k,:,:,:] = dvarPDSI
        # prior ensemble mean of MC iteration "k"
        tmpp = np.mean(varPDSI_prior,axis=1)
        xbmPDSI_all[k,:,:,:] = tmpp.reshape(nlatp,nlonp)

    if availablePRCP:
        xamPRCP = xamPRCP + dvarPRCP
        xamPRCP_all[k,:,:,:] = dvarPRCP
        # prior ensemble mean of MC iteration "k"
        tmpp = np.mean(varPRCP_prior,axis=1)
        xbmPRCP_all[k,:,:,:] = tmpp.reshape(nlatp,nlonp)

# Data over all MC iterations
if availablePDSI:
    # Prior sample mean over all MC iterations
    xbmPDSI = xbmPDSI_all.mean(0)
    # Posterior: this is the sample mean computed with low-memory accumulation
    xamPDSI = xamPDSI/len(mcdir)
    #  this is the sample mean computed with numpy on all data
    xamPDSI_check = xamPDSI_all.mean(0)
    #  check..
    max_err = np.max(np.max(np.max(xamPDSI_check - xamPDSI)))
    if max_err > 1e-4:
        print('varPDSI max error = ' + str(max_err))
        raise Exception('sample mean does not match what is in the ensemble files!')
    # sample variance
    xamPDSI_var = xamPDSI_all.var(0)

    print(('Variable PDSI: %s' %varPDSI_label))
    print(np.shape(xamPDSI_var))
    print('\n shape of the ensemble array: ' + str(np.shape(xamPDSI_all)) +'\n')
    print('\n shape of the ensemble-mean array: ' + str(np.shape(xamPDSI)) +'\n')
    print('\n shape of the ensemble-mean prior array: ' + str(np.shape(xbmPDSI)) +'\n')

    
if availablePRCP:
    # Prior sample mean over all MC iterations
    xbmPRCP = xbmPRCP_all.mean(0)
    # Posterior: this is the sample mean computed with low-memory accumulation
    xamPRCP = xamPRCP/len(mcdir)
    #  this is the sample mean computed with numpy on all data
    xamPRCP_check = xamPRCP_all.mean(0)
    #  check ..
    max_err = np.max(np.max(np.max(xamPRCP_check - xamPRCP)))
    if max_err > 1e-4:
        print('varPRCP max error = ' + str(max_err))
        raise Exception('sample mean does not match what is in the ensemble files!')
    # sample variance
    xamPRCP_var = xamPRCP_all.var(0)    

    print(('Variable PRCP: %s' %varPRCP_label))
    print(np.shape(xamPRCP_var))
    print('\n shape of the ensemble array: ' + str(np.shape(xamPRCP_all)) +'\n')
    print('\n shape of the ensemble-mean array: ' + str(np.shape(xamPRCP)) +'\n')
    print('\n shape of the ensemble-mean prior array: ' + str(np.shape(xbmPRCP)) +'\n')

    
lmr_lat_range = (lat2[0,0],lat2[-1,0])
lmr_lon_range = (lon2[0,0],lon2[0,-1])
print('LMR grid info:')
print(' lats=%s' % str(lmr_lat_range))
print(' lons=%s' % str(lmr_lon_range))

# ===========================================================================================================
# BEGIN: load verification data
# ===========================================================================================================

print('\nloading verification data...\n')

# Define month sequence for the calendar year 
# (argument needed in upload of reanalysis data)
annual = list(range(1,13))


# DaiPDSI product
# ---------------
calib_vars = ['pdsi']

[dtime,Dai_lat,Dai_lon,DaiPDSI] = read_gridded_data_DaiPDSI(datadir_verif,datafile_verif_PDSI,calib_vars,out_anomalies=True,outfreq='annual')
Dai_time = np.array([d.year for d in dtime])
nlat_Dai = len(Dai_lat)
nlon_Dai = len(Dai_lon)
lon2d_Dai, lat2d_Dai = np.meshgrid(Dai_lon, Dai_lat)
Dai_lat_range = (lat2d_Dai[0,0],lat2d_Dai[-1,0])
Dai_lon_range = (lon2d_Dai[0,0],lon2d_Dai[0,-1])
print('DaiPDSI grid info:')
print(' lats=%s' % str(Dai_lat_range))
print(' lons=%s' % str(Dai_lon_range))

# DaiPDSI longitudes are off by 180 degrees
print(' Shifting longitudes by 180 degrees')
lat2d_Dai = np.roll(lat2d_Dai,shift=nlon_Dai/2,axis=1)
lon2d_Dai = np.roll(lon2d_Dai,shift=nlon_Dai/2,axis=1)
DaiPDSI = np.roll(DaiPDSI,shift=nlon_Dai/2,axis=2)

# PDSI is land-based data: use most recent data in array to estimate the fraction of land pts
# (assumed that valid data is present on all land pts at that time) vs. total nb. of pts.
nbokpts = np.sum(np.isfinite(DaiPDSI[-1,:,:]))
nbtotpts = nlat_Dai*nlon_Dai
fracpts = nbokpts/nbtotpts
# adjust valid_frac
valid_frac1 = valid_frac*fracpts


# GPCC product
# ---------------

calib_vars = ['precip']

[dtime,gpcc_lat,gpcc_lon,GPCC] = read_gridded_data_GPCC(datadir_verif,datafile_verif_GPCC,calib_vars,out_anomalies=True,outfreq='annual')
GPCC_time = np.array([d.year for d in dtime])
nlat_GPCC = len(gpcc_lat)
nlon_GPCC = len(gpcc_lon)
lon2d_GPCC, lat2d_GPCC = np.meshgrid(gpcc_lon, gpcc_lat)
GPCC_lat_range = (lat2d_GPCC[0,0],lat2d_GPCC[-1,0])
GPCC_lon_range = (lon2d_GPCC[0,0],lon2d_GPCC[0,-1])
print('GPCC grid info:')
print(' lats=%s' % str(GPCC_lat_range))
print(' lons=%s' % str(GPCC_lon_range))

# GPCC latitudes are upside-down
print(' Flipping latitudes')
lat2d_GPCC  = np.flipud(lat2d_GPCC )
GPCC = GPCC[:,::-1,:]

# GPCC is land-based data: use most recent data in array to estimate the fraction of land pts
# (assumed that valid data is present on all land pts at that time) vs. total nb. of pts.
nbokpts = np.sum(np.isfinite(GPCC[-1,:,:]))
nbtotpts = nlat_GPCC*nlon_GPCC
fracpts = nbokpts/nbtotpts
# adjust valid_frac
valid_frac2 = valid_frac*fracpts


# ===========================================================================================================
# END: load verification data
# ===========================================================================================================

# adjust so that all anomaly data pertain to the mean over a user-defined reference period (e.g. 20th century)
stime = ref_period[0]
etime = ref_period[1]

# LMR
smatch, ematch = find_date_indices(LMR_time,stime,etime)

if availablePDSI:
    LMR_pdsi_anomaly = xamPDSI # PDSI
    LMR_pdsi_anomaly = LMR_pdsi_anomaly - np.nanmean(LMR_pdsi_anomaly[smatch:ematch,:,:],axis=0)

    # DaiPDSI
    smatch, ematch = find_date_indices(Dai_time,stime,etime)
    DaiPDSI = DaiPDSI - np.nanmean(DaiPDSI[smatch:ematch,:,:],axis=0)

if availablePRCP:
    LMR_prcp_anomaly = xamPRCP # Precip
    LMR_prcp_anomaly = LMR_prcp_anomaly - np.nanmean(LMR_prcp_anomaly[smatch:ematch,:,:],axis=0)

    #  GPCC
    smatch, ematch = find_date_indices(GPCC_time,stime,etime)
    GPCC = GPCC - np.nanmean(GPCC[smatch:ematch,:,:],axis=0)


print('\n regridding LMR data to grids of verification data...\n')

iplot_loc= False
#iplot_loc= True

# create instance of the spherical harmonics object for each grid
specob_lmr   = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')
specob_dai   = Spharmt(nlon_Dai,nlat_Dai,gridtype='regular',legfunc='computed')
specob_gpcc  = Spharmt(nlon_GPCC,nlat_GPCC,gridtype='regular',legfunc='computed')

# loop over years of interest and transform...specify trange at top of file
iw = 0
if nya > 0:
    iw = (nya-1)/2

cyears = list(range(trange[0],trange[1]))

# time series for combination of data products
lmr_dai_csave   = np.zeros([len(cyears)])
lmr_gpcc_csave  = np.zeros([len(cyears)])

# for full 2d grids
# -----------------
# obs products
dai_allyears   = np.zeros([len(cyears),nlat_Dai,nlon_Dai])
gpcc_allyears  = np.zeros([len(cyears),nlat_GPCC,nlon_GPCC])

# for lmr projected over the various grids
lmr_on_dai_allyears   = np.zeros([len(cyears),nlat_Dai,nlon_Dai])
lmr_on_gpcc_allyears  = np.zeros([len(cyears),nlat_GPCC,nlon_GPCC])

# for prior projected over the various grids
xbm_on_dai_allyears   = np.zeros([len(cyears),nlat_Dai,nlon_Dai])
xbm_on_gpcc_allyears  = np.zeros([len(cyears),nlat_GPCC,nlon_GPCC])



# for zonal means
# ---------------
# obs products
dai_zm   = np.zeros([len(cyears),nlat_Dai])
gpcc_zm  = np.zeros([len(cyears),nlat_GPCC])

# for lmr projected over the various grids
lmr_on_dai_zm   = np.zeros([len(cyears),nlat_Dai])
lmr_on_gpcc_zm  = np.zeros([len(cyears),nlat_GPCC])

# for prior projected over the various grids
xbm_on_dai_zm   = np.zeros([len(cyears),nlat_Dai])
xbm_on_gpcc_zm  = np.zeros([len(cyears),nlat_GPCC])


# Loop over years defining the verification set
k = -1
for yr in cyears:
    k = k + 1
    LMR_smatch, LMR_ematch     = find_date_indices(LMR_time,yr-iw,yr+iw+1)
    Dai_smatch, Dai_ematch     = find_date_indices(Dai_time,yr-iw,yr+iw+1)
    GPCC_smatch, GPCC_ematch   = find_date_indices(GPCC_time,yr-iw,yr+iw+1)
    
    print('------------------------------------------------------------------------')
    print('working on year... %s' % str(yr))
    print('working on year... %5s LMR index = %5s : LMR year = %5s' % (str(yr),str(LMR_smatch),str(LMR_time[LMR_smatch])))
    
    # DaiPDSI
    if Dai_smatch and Dai_ematch:
        dai_verif = np.mean(DaiPDSI[Dai_smatch:Dai_ematch,:,:],0)
    else:
        dai_verif = np.zeros(shape=[nlat_Dai,nlon_Dai])
        dai_verif.fill(np.nan)

    # GPCC
    if GPCC_smatch and GPCC_ematch:
        gpcc_verif = np.mean(GPCC[GPCC_smatch:GPCC_ematch,:,:],0)
    else:
        gpcc_verif = np.zeros(shape=[nlat_GPCC,nlon_GPCC])
        gpcc_verif.fill(np.nan)
        
    
    if iplot_loc:
        fig = plt.figure()
        vmin = -5.0; vmax = 5.0
        cbarfmt = '%4.1f'
        nticks = 6 # number of ticks on the colorbar
        ax = fig.add_subplot(1,1,1)
        LMR_plotter(dai_verif,lat2d_Dai,lon2d_Dai,'bwr',nlevs,vmin=vmin,vmax=vmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('DaiPDSI : '+str(yr))
        plt.clim(vmin,vmax)

        fig.tight_layout()
        plt.savefig('DaiPDSI_%s.png' %str(yr))
        plt.close()

        # ---
        fig = plt.figure()
        vmin = -1.0e-4; vmax = 1.0e-4 # kg m-2 s-1
        cbarfmt = '%7.5f'
        nticks = 6 # number of ticks on the colorbar
        ax = fig.add_subplot(1,1,1)
        LMR_plotter(gpcc_verif,lat2d_GPCC,lon2d_GPCC,'bwr',nlevs,vmin=vmin,vmax=vmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('GPCC : '+str(yr))
        plt.clim(vmin,vmax)

        fig.tight_layout()
        plt.savefig('GPCC_%s.png' %str(yr))
        plt.close()

        

    if availablePDSI:
        # LMR anomaly for current period
        pdata_pdsi_lmr = np.mean(LMR_pdsi_anomaly[LMR_smatch:LMR_ematch,:,:],0)
        # prior
        pdata_pdsi_xbm = np.mean(xbmPDSI[LMR_smatch:LMR_ematch,:,:],0)

        # regrid LMR on the various verification grids
        lmr_on_dai   = regrid(specob_lmr, specob_dai,   pdata_pdsi_lmr, ntrunc=None, smooth=None)
        # prior
        xbm_on_dai   = regrid(specob_lmr, specob_dai,   pdata_pdsi_xbm, ntrunc=None, smooth=None)

        # save the full grids
        dai_allyears[k,:,:]          = dai_verif
        lmr_on_dai_allyears[k,:,:]   = lmr_on_dai
        # prior
        xbm_on_dai_allyears[k,:,:]   = xbm_on_dai

        # -------------------------    
        # DaiPDSI
        fracok    = np.sum(np.isfinite(dai_verif),axis=1,dtype=np.float16)/float(nlon_Dai)
        boolok    = np.where(fracok >= valid_frac1)
        boolnotok = np.where(fracok < valid_frac1)
        for i in boolok:
            dai_zm[k,i] = np.nanmean(dai_verif[i,:],axis=1)
        dai_zm[k,boolnotok]  = np.NAN
        lmr_on_dai_zm[k,:]   = np.mean(lmr_on_dai,1)
        xbm_on_dai_zm[k,:]   = np.mean(xbm_on_dai,1)

        # anomaly correlation
        # -------------------    
        # prepare arrays
        dai_vec          = np.reshape(dai_verif,(1,nlat_Dai*nlon_Dai))
        lmr_on_dai_vec   = np.reshape(lmr_on_dai,(1,nlat_Dai*nlon_Dai))

        # compute correlations, taking into account the missing data in obs. products
        # ---------------------------------------------------------------------------
        # lmr <-> dai
        indok = np.isfinite(dai_vec); nbok = np.sum(indok); nball = dai_vec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac1:
            lmr_dai_csave[k] = np.corrcoef(lmr_on_dai_vec[indok],dai_vec[indok])[0,1]
        else:
            lmr_dai_csave[k] = np.nan
        print('  lmr-dai correlation     : %s' % str(lmr_dai_csave[k]))





        
        
    if availablePRCP:
        # LMR anomaly for current period
        pdata_prcp_lmr = np.mean(LMR_prcp_anomaly[LMR_smatch:LMR_ematch,:,:],0)
        # prior
        pdata_prcp_xbm = np.mean(xbmPRCP[LMR_smatch:LMR_ematch,:,:],0)

        # regrid LMR on the various verification grids
        lmr_on_gpcc   = regrid(specob_lmr, specob_gpcc,   pdata_prcp_lmr, ntrunc=None, smooth=None)
        # prior
        xbm_on_gpcc   = regrid(specob_lmr, specob_gpcc,   pdata_prcp_xbm, ntrunc=None, smooth=None)

        # save the full grids
        gpcc_allyears[k,:,:]         = gpcc_verif
        lmr_on_gpcc_allyears[k,:,:]  = lmr_on_gpcc
        # prior
        xbm_on_gpcc_allyears[k,:,:]  = xbm_on_gpcc

        # compute zonal-mean values    
        # GPCC
        fracok    = np.sum(np.isfinite(gpcc_verif),axis=1,dtype=np.float16)/float(nlon_GPCC)
        boolok    = np.where(fracok >= valid_frac2)
        boolnotok = np.where(fracok < valid_frac2)
        for i in boolok:
            gpcc_zm[k,i] = np.nanmean(gpcc_verif[i,:],axis=1)
        gpcc_zm[k,boolnotok]  = np.NAN
        lmr_on_gpcc_zm[k,:]   = np.mean(lmr_on_gpcc,1)
        xbm_on_gpcc_zm[k,:]   = np.mean(xbm_on_gpcc,1)
    
        # anomaly correlation
        gpcc_vec         = np.reshape(gpcc_verif,(1,nlat_GPCC*nlon_GPCC))
        lmr_on_gpcc_vec  = np.reshape(lmr_on_gpcc,(1,nlat_GPCC*nlon_GPCC))

        # compute correlations, taking into account the missing data in obs. products
        # ---------------------------------------------------------------------------    
        # lmr <-> gpcc
        indok = np.isfinite(gpcc_vec); nbok = np.sum(indok); nball = gpcc_vec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac2:
            lmr_gpcc_csave[k] = np.corrcoef(lmr_on_gpcc_vec[indok],gpcc_vec[indok])[0,1]
        else:
            lmr_gpcc_csave[k] = np.nan
        print('  lmr-gpcc correlation    : %s' % str(lmr_gpcc_csave[k]))


# ===================================================================================
# plots for anomaly correlation statistics
# ===================================================================================
# number of bins in the histograms
nbins = 15
corr_range = [-0.6,0.8]
bins = np.linspace(corr_range[0],corr_range[1],nbins)

fig = plt.figure()


k = 0
if availablePRCP:
    # LMR compared to GPCC
    ax = fig.add_subplot(3,2,k+1)
    ax.plot(cyears,lmr_gpcc_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('LMR - GPCC')
    ax.set_xlim(trange[0],trange[-1])
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    ax = fig.add_subplot(3,2,k+2)
    ax.hist(lmr_gpcc_csave[~np.isnan(lmr_gpcc_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
    ax.set_title('LMR - GPCC')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    if ymax < 10:
        ymax = 10
    elif ymax >= 10 and ymax < 20:
        ymax = 20
    ax.set_ylim(ymin,ymax)
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lmr_gpcc_csave)),fontsize=11,fontweight='bold')

    k = k + 2

if availablePDSI:
    # LMR compared to DaiPDSI
    ax = fig.add_subplot(3,2,k+1)
    ax.plot(cyears,lmr_dai_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('LMR - DaiPDSI')
    ax.set_xlim(trange[0],trange[-1])
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    ax = fig.add_subplot(3,2,k+2)
    ax.hist(lmr_dai_csave[~np.isnan(lmr_dai_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
    ax.set_title('LMR - DaiPDSI')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    if ymax < 10:
        ymax = 10
    elif ymax >= 10 and ymax < 20:
        ymax = 20
    ax.set_ylim(ymin,ymax)
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lmr_dai_csave)),fontsize=11,fontweight='bold')
    

fig.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.93, wspace=0.5, hspace=0.5)
fig.suptitle('Hydroclimate variables anomaly correlation',fontweight='bold') 
if fsave:
    print('saving to .png')
    plt.savefig(nexp+'_verify_grid_HYDRO_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'.png')
    #plt.savefig(nexp+'_verify_grid_HYDRO_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')
    

    
# ===================================================================================
# BEGIN bias, r and CE calculations
# ===================================================================================

# correlation and CE at each (lat,lon) point
# ------------------------------------------


if availablePDSI:
    # LMR-DaiPDSI
    lmr_on_dai_err   = lmr_on_dai_allyears - dai_allyears
    # prior
    xbm_on_dai_err   = xbm_on_dai_allyears - dai_allyears
    bias_lmr_dai     = np.zeros([nlat_Dai,nlon_Dai])
    # prior
    bias_xbm_dai     = np.zeros([nlat_Dai,nlon_Dai])
    r_lmr_dai        = np.zeros([nlat_Dai,nlon_Dai])
    ce_lmr_dai       = np.zeros([nlat_Dai,nlon_Dai])
    # prior
    r_xbm_dai        = np.zeros([nlat_Dai,nlon_Dai])
    ce_xbm_dai       = np.zeros([nlat_Dai,nlon_Dai])
    ce_lmr_dai_unbiased   = np.zeros([nlat_Dai,nlon_Dai])

    ce_lmr_dai = coefficient_efficiency(dai_allyears,lmr_on_dai_allyears,valid_frac1)
    ce_xbm_dai = coefficient_efficiency(dai_allyears,xbm_on_dai_allyears,valid_frac1)
    for la in range(nlat_Dai):
        for lo in range(nlon_Dai):
            indok = np.isfinite(dai_allyears[:,la,lo])
            nbok = np.sum(indok)
            nball = lmr_on_dai_allyears[:,la,lo].shape[0]
            ratio = float(nbok)/float(nball)
            if ratio > valid_frac1:
                r_lmr_dai[la,lo] = np.corrcoef(lmr_on_dai_allyears[indok,la,lo],dai_allyears[indok,la,lo])[0,1]
                bias_lmr_dai[la,lo] = np.mean(lmr_on_dai_err[indok,la,lo],axis=0)
                numer_unbiased = np.sum(np.power(lmr_on_dai_err[indok,la,lo]-bias_lmr_dai[la,lo],2),axis=0)
                denom = np.sum(np.power(dai_allyears[indok,la,lo]-np.mean(dai_allyears[indok,la,lo],axis=0),2),axis=0)
                ce_lmr_dai_unbiased[la,lo] = 1. - (numer_unbiased/denom)
                #r_xbm_dai[la,lo] = np.corrcoef(xbm_on_dai_allyears[indok,la,lo],dai_allyears[indok,la,lo])[0,1]
                bias_xbm_dai[la,lo] = np.mean(xbm_on_dai_err[indok,la,lo],axis=0)
            else:
                r_lmr_dai[la,lo]  = np.nan
                bias_lmr_dai[la,lo] = np.nan
                ce_lmr_dai_unbiased[la,lo] = np.nan
                r_xbm_dai[la,lo]  = np.nan
                bias_xbm_dai[la,lo] = np.nan

    # median
    print('')
    lat_Dai = np.squeeze(lat2d_Dai[:,0])
    indlat = np.where((lat_Dai[:] > -60.0) & (lat_Dai[:] < 60.0))
    lmr_dai_rmedian = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_dai)) ))
    print('lmr-dai all-grid median r    : %s' % str(lmr_dai_rmedian))
    lmr_dai_rmedian60 = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_dai[indlat,:])) ))
    print('lmr-dai 60S-60N median r     : %s' % str(lmr_dai_rmedian60))
    lmr_dai_cemedian = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_dai)) ))
    print('lmr-dai all-grid median ce   : %s' % str(lmr_dai_cemedian))
    lmr_dai_cemedian60 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_dai[indlat,:])) ))
    print('lmr-dai 60S-60N median ce    : %s' % str(lmr_dai_cemedian60))
    lmr_dai_biasmedian = str(float('%.2f' % np.nanmedian(bias_lmr_dai) ))
    # mean
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lmr_dai,lat_Dai)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lmr_dai,lat_Dai)
    [biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_lmr_dai,lat_Dai)
    lmr_dai_rmean_global    = str(float('%.2f' %rmean_global[0]))
    lmr_dai_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
    lmr_dai_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
    lmr_dai_cemean_global   = str(float('%.2f' %cemean_global[0]))
    lmr_dai_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
    lmr_dai_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
    lmr_dai_biasmean_global = str(float('%.2f' %biasmean_global[0]))
    lmr_dai_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
    lmr_dai_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))
    # prior
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_xbm_dai,lat_Dai)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_xbm_dai,lat_Dai)
    [biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_xbm_dai,lat_Dai)
    xbm_dai_rmean_global    = str(float('%.2f' %rmean_global[0]))
    xbm_dai_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
    xbm_dai_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
    xbm_dai_cemean_global   = str(float('%.2f' %cemean_global[0]))
    xbm_dai_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
    xbm_dai_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
    xbm_dai_biasmean_global = str(float('%.2f' %biasmean_global[0]))
    xbm_dai_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
    xbm_dai_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))

    
    # zonal mean verification statistics
    # ----------------------------------
    # LMR-DaiPDSI
    r_lmr_dai_zm     = np.zeros([nlat_Dai])
    ce_lmr_dai_zm    = np.zeros([nlat_Dai])
    bias_lmr_dai_zm  = np.zeros([nlat_Dai])
    lmr_dai_err_zm = lmr_on_dai_zm - dai_zm
    ce_lmr_dai_zm = coefficient_efficiency(dai_zm,lmr_on_dai_zm,valid_frac1)
    # prior
    r_xbm_dai_zm     = np.zeros([nlat_Dai])
    ce_xbm_dai_zm    = np.zeros([nlat_Dai])
    bias_xbm_dai_zm  = np.zeros([nlat_Dai])
    xbm_dai_err_zm = xbm_on_dai_zm - dai_zm
    ce_xbm_dai_zm = coefficient_efficiency(dai_zm,xbm_on_dai_zm,valid_frac1)
    for la in range(nlat_Dai):
        indok = np.isfinite(dai_zm[:,la])
        nbok = np.sum(indok)
        nball = len(cyears)
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_lmr_dai_zm[la] = np.corrcoef(lmr_on_dai_zm[indok,la],dai_zm[indok,la])[0,1]
            bias_lmr_dai_zm[la] = np.mean(lmr_dai_err_zm[indok,la],axis=0)
            r_xbm_dai_zm[la] = np.corrcoef(xbm_on_dai_zm[indok,la],dai_zm[indok,la])[0,1]
            bias_xbm_dai_zm[la] = np.mean(xbm_dai_err_zm[indok,la],axis=0)
        else:
            r_lmr_dai_zm[la]  = np.nan
            bias_lmr_dai_zm[la] = np.nan
            r_xbm_dai_zm[la]  = np.nan
            bias_xbm_dai_zm[la] = np.nan

    
if availablePRCP:
    # LMR-GPCC
    lmr_on_gpcc_err  = lmr_on_gpcc_allyears - gpcc_allyears
    # prior
    xbm_on_gpcc_err  = xbm_on_gpcc_allyears - gpcc_allyears
    bias_lmr_gpcc    = np.zeros([nlat_GPCC,nlon_GPCC])
    # prior
    bias_xbm_gpcc    = np.zeros([nlat_GPCC,nlon_GPCC])
    r_lmr_gpcc       = np.zeros([nlat_GPCC,nlon_GPCC])
    ce_lmr_gpcc      = np.zeros([nlat_GPCC,nlon_GPCC])
    # prior
    r_xbm_gpcc       = np.zeros([nlat_GPCC,nlon_GPCC])
    ce_xbm_gpcc      = np.zeros([nlat_GPCC,nlon_GPCC])
    ce_lmr_gpcc_unbiased  = np.zeros([nlat_GPCC,nlon_GPCC])

    ce_lmr_gpcc = coefficient_efficiency(gpcc_allyears,lmr_on_gpcc_allyears,valid_frac2)
    ce_xbm_gpcc = coefficient_efficiency(gpcc_allyears,xbm_on_gpcc_allyears,valid_frac2)
    for la in range(nlat_GPCC):
        for lo in range(nlon_GPCC):
            indok = np.isfinite(gpcc_allyears[:,la,lo])
            nbok = np.sum(indok)
            nball = lmr_on_gpcc_allyears[:,la,lo].shape[0]
            ratio = float(nbok)/float(nball)
            if ratio > valid_frac2:
                r_lmr_gpcc[la,lo] = np.corrcoef(lmr_on_gpcc_allyears[indok,la,lo],gpcc_allyears[indok,la,lo])[0,1]
                bias_lmr_gpcc[la,lo] = np.mean(lmr_on_gpcc_err[indok,la,lo],axis=0)
                numer_unbiased = np.sum(np.power(lmr_on_gpcc_err[indok,la,lo]-bias_lmr_gpcc[la,lo],2),axis=0)
                denom = np.sum(np.power(gpcc_allyears[indok,la,lo]-np.mean(gpcc_allyears[indok,la,lo],axis=0),2),axis=0)
                ce_lmr_gpcc_unbiased[la,lo] = 1. - (numer_unbiased/denom)
                bias_xbm_gpcc[la,lo] = np.mean(xbm_on_gpcc_err[indok,la,lo],axis=0)
            else:
                r_lmr_gpcc[la,lo]  = np.nan
                bias_lmr_gpcc[la,lo] = np.nan
                ce_lmr_gpcc_unbiased[la,lo] = np.nan
                r_xbm_gpcc[la,lo]  = np.nan
                bias_xbm_gpcc[la,lo] = np.nan

    # median
    print('')
    lat_GPCC = np.squeeze(lat2d_GPCC[:,0])
    indlat = np.where((lat_GPCC[:] > -60.0) & (lat_GPCC[:] < 60.0))
    lmr_gpcc_rmedian = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_gpcc)) ))
    print('lmr-gpcc all-grid median r    : %s' % str(lmr_gpcc_rmedian))
    lmr_gpcc_rmedian60 = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_gpcc[indlat,:])) ))
    print('lmr-gpcc 60S-60N median r     : %s' % str(lmr_gpcc_rmedian60))
    lmr_gpcc_cemedian = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_gpcc)) ))
    print('lmr-gpcc all-grid median ce   : %s' % str(lmr_gpcc_cemedian))
    lmr_gpcc_cemedian60 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_gpcc[indlat,:])) ))
    print('lmr-gpcc 60S-60N median ce    : %s' % str(lmr_gpcc_cemedian60))
    lmr_gpcc_biasmedian = str(float('%.2f' % np.nanmedian(bias_lmr_gpcc) ))
    # mean
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lmr_gpcc,lat_GPCC)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lmr_gpcc,lat_GPCC)
    [biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_lmr_gpcc,lat_GPCC)
    lmr_gpcc_rmean_global    = str(float('%.2f' %rmean_global[0]))
    lmr_gpcc_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
    lmr_gpcc_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
    lmr_gpcc_cemean_global   = str(float('%.2f' %cemean_global[0]))
    lmr_gpcc_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
    lmr_gpcc_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
    lmr_gpcc_biasmean_global = str(float('%.2f' %biasmean_global[0]))
    lmr_gpcc_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
    lmr_gpcc_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))
    # prior
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_xbm_gpcc,lat_GPCC)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_xbm_gpcc,lat_GPCC)
    [biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_xbm_gpcc,lat_GPCC)
    xbm_gpcc_rmean_global    = str(float('%.2f' %rmean_global[0]))
    xbm_gpcc_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
    xbm_gpcc_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
    xbm_gpcc_cemean_global   = str(float('%.2f' %cemean_global[0]))
    xbm_gpcc_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
    xbm_gpcc_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
    xbm_gpcc_biasmean_global = str(float('%.2f' %biasmean_global[0]))
    xbm_gpcc_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
    xbm_gpcc_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))

    # zonal mean verification statistics
    # ----------------------------------
    # LMR-GPCC
    r_lmr_gpcc_zm    = np.zeros([nlat_GPCC])
    ce_lmr_gpcc_zm   = np.zeros([nlat_GPCC])
    bias_lmr_gpcc_zm = np.zeros([nlat_GPCC])
    lmr_gpcc_err_zm  = lmr_on_gpcc_zm - gpcc_zm
    ce_lmr_gpcc_zm   = coefficient_efficiency(gpcc_zm,lmr_on_gpcc_zm,valid_frac2)
    # prior
    r_xbm_gpcc_zm    = np.zeros([nlat_GPCC])
    ce_xbm_gpcc_zm   = np.zeros([nlat_GPCC])
    bias_xbm_gpcc_zm = np.zeros([nlat_GPCC])
    xbm_gpcc_err_zm  = xbm_on_gpcc_zm - gpcc_zm
    ce_xbm_gpcc_zm   = coefficient_efficiency(gpcc_zm,xbm_on_gpcc_zm,valid_frac2)
    for la in range(nlat_GPCC):
        indok = np.isfinite(gpcc_zm[:,la])
        nbok = np.sum(indok)
        nball = len(cyears)
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac2:
            r_lmr_gpcc_zm[la] = np.corrcoef(lmr_on_gpcc_zm[indok,la],gpcc_zm[indok,la])[0,1]
            bias_lmr_gpcc_zm[la] = np.mean(lmr_gpcc_err_zm[indok,la],axis=0)
            r_xbm_gpcc_zm[la] = np.corrcoef(xbm_on_gpcc_zm[indok,la],gpcc_zm[indok,la])[0,1]
            bias_xbm_gpcc_zm[la] = np.mean(xbm_gpcc_err_zm[indok,la],axis=0)
        else:
            r_lmr_gpcc_zm[la]  = np.nan
            bias_lmr_gpcc_zm[la] = np.nan
            r_xbm_gpcc_zm[la]  = np.nan
            bias_xbm_gpcc_zm[la] = np.nan
        

        
# plot zonal mean statistics
# --------------------------
major_ticks = np.arange(-90, 91, 30)
fig = plt.figure()

# Correlation
ax = fig.add_subplot(1,2,1)    
legendItems = []
if availablePDSI:
    daileg,   = ax.plot(r_lmr_dai_zm,  lat_Dai,   'blue',  linestyle='-',lw=2,label='DaiPDSI')
    legendItems.append(daileg)
if availablePRCP:
    gpccleg,  = ax.plot(r_lmr_gpcc_zm, lat_GPCC,  'red', linestyle='-',lw=2,label='GPCC')
    legendItems.append(gpccleg)
ax.plot([0,0],[-90,90],'k:')
ax.set_yticks(major_ticks)
plt.ylim([-90,90])
plt.xlim([-1,1])
plt.ylabel('latitude',fontweight='bold')
plt.xlabel('Correlation',fontweight='bold')
ax.legend(handles=legendItems,handlelength=1.5,ncol=1,fontsize=11,loc='upper left',frameon=False)
# CE
ax = fig.add_subplot(1,2,2)    
if availablePDSI: ax.plot(ce_lmr_dai_zm,  lat_Dai,   'blue',  linestyle='-',lw=2)
if availablePRCP: ax.plot(ce_lmr_gpcc_zm, lat_GPCC,  'red', linestyle='-',lw=2)
ax.plot([0,0],[-90,90],'k:')
ax.set_yticks([])                                                       
plt.ylim([-90,90])
plt.xlim([-1.0,1.0])
plt.xlabel('Coefficient of efficiency',fontweight='bold')
plt.suptitle('LMR zonal-mean verification - Hydroclimate',fontweight='bold')

fig.tight_layout(pad = 2.0)
if fsave:
    print('saving to .png')
    plt.savefig(nexp+'_verify_grid_HYDRO_r_ce_zonal_mean_'+str(trange[0])+'-'+str(trange[1])+'.png')
    #plt.savefig(nexp+'_verify_grid_HYDRO_r_ce_zonal_mean_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')
    plt.close()









    
    
# ---------
# bias maps
# ---------

nticks = 4 # number of ticks on the colorbar

if iplot:
    fig = plt.figure()

    if availablePRCP:
        bmin = -1.0e-5
        bmax = 1.0e-5
        cbarfmt = '%7.5f'    
        ax = fig.add_subplot(3,2,1)    
        LMR_plotter(bias_lmr_gpcc,lat2d_GPCC,lon2d_GPCC,'bwr',nlevs,vmin=bmin,vmax=bmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - GPCC bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_gpcc_biasmean_global))
        plt.clim(bmin,bmax)
        ax.title.set_position([.5, 1.05])
    
    if availablePDSI:
        bmin = -1.0
        bmax = 1.0
        cbarfmt = '%4.1f'
        ax = fig.add_subplot(3,2,2)    
        LMR_plotter(bias_lmr_dai,lat2d_Dai,lon2d_Dai,'bwr',nlevs,vmin=bmin,vmax=bmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - DaiPDSI bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_dai_biasmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_HYDRO_bias_'+str(trange[0])+'-'+str(trange[1])+'.png')


# -------------
# r and ce maps
# -------------

cbarfmt = '%4.1f'
nticks = 4 # number of ticks on the colorbar
if iplot:

    fig = plt.figure()

    k = 0

    if availablePRCP:
        # LMR - GPCC
        ax = fig.add_subplot(3,2,k+1)    
        LMR_plotter(r_lmr_gpcc,lat2d_GPCC,lon2d_GPCC,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - GPCC r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_gpcc_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(3,2,k+2)    
        LMR_plotter(ce_lmr_gpcc,lat2d_GPCC,lon2d_GPCC,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - GPCC CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_gpcc_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])
    
        k = k + 2

    if availablePDSI:
        # LMR - DaiPDSI
        ax = fig.add_subplot(3,2,k+1)    
        LMR_plotter(r_lmr_dai,lat2d_Dai,lon2d_Dai,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - DaiPDSI r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_dai_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(3,2,k+2)    
        LMR_plotter(ce_lmr_dai,lat2d_Dai,lon2d_Dai,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - DaiPDSI CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_dai_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_HYDRO_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.png')
        #plt.savefig(nexp+'_verify_grid_HYDRO_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')

        
    # Prior 
    # -----
    fig = plt.figure()
    k = 0

    if availablePRCP:
        # LMR - GPCC
        ax = fig.add_subplot(3,2,k+1)    
        LMR_plotter(r_xbm_gpcc,lat2d_GPCC,lon2d_GPCC,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('Prior - GPCC r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_gpcc_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(3,2,k+2)    
        LMR_plotter(ce_xbm_gpcc,lat2d_GPCC,lon2d_GPCC,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('Prior - GPCC CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_gpcc_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        k = k + 2


    if availablePDSI:
        # LMR - DaiPDSI
        ax = fig.add_subplot(3,2,k+1)    
        LMR_plotter(r_xbm_dai,lat2d_Dai,lon2d_Dai,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('Prior - DaiPDSI r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_dai_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(3,2,k+2)    
        LMR_plotter(ce_xbm_dai,lat2d_Dai,lon2d_Dai,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('Prior - DaiPDSI CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_dai_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])
        
    
    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_HYDRO_r_ce_Prior_'+str(trange[0])+'-'+str(trange[1])+'.png')
        #plt.savefig(nexp+'_verify_grid_HYDRO_r_ce_Prior_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')



    # Comparisons of CE (full) & CE (unbiased errors)    
    fig = plt.figure()
    k = 0

    if availablePRCP:
        # LMR - GPCC
        lmr_gpcc_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_gpcc[:,:])) ))
        lmr_gpcc_cemean2 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_gpcc[:,:]-ce_lmr_gpcc_unbiased[:,:])) ))

        ax = fig.add_subplot(3,2,k+1)    
        LMR_plotter(ce_lmr_gpcc,lat2d_GPCC,lon2d_GPCC,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - GPCC CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_gpcc_cemean))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(3,2,k+2)    
        LMR_plotter(ce_lmr_gpcc-ce_lmr_gpcc_unbiased,lat2d_GPCC,lon2d_GPCC,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - GPCC CE bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_gpcc_cemean2))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        k = k + 2

    if availablePDSI:
        # LMR - DaiPDSI
        lmr_dai_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_dai[:,:])) ))
        lmr_dai_cemean2 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_dai[:,:]-ce_lmr_dai_unbiased[:,:])) ))
    
        ax = fig.add_subplot(3,2,k+1)    
        LMR_plotter(ce_lmr_dai,lat2d_Dai,lon2d_Dai,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - DaiPDSI CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_dai_cemean))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(3,2,k+2)    
        LMR_plotter(ce_lmr_dai-ce_lmr_dai_unbiased,lat2d_Dai,lon2d_Dai,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - DaiPDSI CE bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_dai_cemean2))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])
        
        
    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_HYDRO_ce_vsBias_'+str(trange[0])+'-'+str(trange[1])+'.png')


