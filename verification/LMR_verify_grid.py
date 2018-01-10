""" 
Module: LMR_verify_grid.py

Purpose: Generates spatial verification statistics of LMR gridded air temperature
         against various gridded historical instrumental temperature datasets
         and reanalyses.  

Originator: Greg Hakim, U. of Washington, November 2015

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
from netCDF4 import Dataset
import mpl_toolkits.basemap as bm
from matplotlib import ticker
from spharm import Spharmt, getspecindx, regrid
import pickle
import warnings

# LMR specific imports
sys.path.append('../')
from LMR_utils import global_hemispheric_means, assimilated_proxies, coefficient_efficiency
from load_gridded_data import read_gridded_data_GISTEMP
from load_gridded_data import read_gridded_data_HadCRUT
from load_gridded_data import read_gridded_data_BerkeleyEarth
from load_gridded_data import read_gridded_data_MLOST
from load_gridded_data import read_gridded_data_CMIP5_model

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
#nexp = 'test'
#nexp = 'pages2_loc20000'
#nexp = 'pages2_loc12000'
#nexp = 'pages2_loc15000'
#nexp = 'pages2_loc10000'
#nexp = 'pages2_loc1000'
#nexp = 'pages2_loc5000'
#nexp = 'pages2_noloc'
#nexp = 'pages2_loc12000_breit'
#nexp = 'pages2_loc12000_pages2k2_seasonal_TorP'
#nexp = 'pages2_loc12000_pages2k2_seasonal_TorP_nens500'
#nexp = 'pages2_loc12000_pages2k2_seasonal_TorP_nens200'
#nexp = 'pages2_loc15000_pages2k2_seasonal_TorP_nens200'
#nexp = 'pages2_loc15000_pages2k2_seasonal_TorP_nens200_inflate1.5'
#nexp = 'pages2_noloc_nens200'
#nexp = 'pages2_loc20000_pages2k2_seasonal_TorP_nens200'
#nexp = 'pages2_loc25000_pages2k2_seasonal_TorP_nens200'
#nexp = 'pages2_loc20000_seasonal_bilinear_nens200'
#nexp = 'pages2_loc25000_seasonal_bilinear_nens200'
#nexp = 'pages2_loc25000_seasonal_bilinear_nens200_75pct'
#nexp = 'pages2_loc25000_seasonal_bilinear_nens200_meta'
nexp = 'pages2_loc25000_seasonal_bilinear_nens200'

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
#datadir_output = '/home/disk/kalman3/rtardif/LMR/output'
datadir_output = '/home/disk/kalman3/hakim/LMR'
#datadir_output = '/home/disk/kalman2/wperkins/LMR_output/archive'

# Directory where reanalysis data can be found
datadir_reanl = '/home/disk/kalman3/rtardif/LMR/data/model'

# threshold for fraction of valid data in calculation of verif. stats
valid_frac = 0.5

# number of contours for plots
nlevs = 21

# plot alpha transparency
alpha = 0.5

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

##################################
# END:  set user parameters here
##################################

# variable---this script is temperature only!
var = 'tas_sfc_Amon'

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

mcdir = [item.split('/')[-1] for item in dirset]
niters = len(mcdir)

print('mcdir: %s' % str(mcdir))
print('niters = %s' % str(niters))

# read ensemble mean data
print('\n reading LMR ensemble-mean data...\n')

first = True
k = -1
for dir in mcdir:
    k = k + 1
    # Posterior (reconstruction)
    #ensfiln = workdir + '/' + dir + '/ensemble_mean.npz'
    ensfiln = workdir + '/' + dir + '/ensemble_mean_'+var+'.npz'
    npzfile = np.load(ensfiln)
    print(npzfile.files)
    tmp = npzfile['xam']
    print('shape of tmp: %s' % str(np.shape(tmp)))

    # load prior data
    file_prior = workdir + '/' + dir + '/Xb_one.npz'
    Xprior_statevector = np.load(file_prior)
    Xb_one = Xprior_statevector['Xb_one']
    # extract variable (sfc temperature) from state vector
    state_info = Xprior_statevector['state_info'].item()
    print(state_info)
    posbeg = state_info[var]['pos'][0]
    posend = state_info[var]['pos'][1]
    tas_prior = Xb_one[posbeg:posend+1,:]
    
    if first:
        first = False
        recon_times = npzfile['years']
        LMR_time = np.array(list(map(int,recon_times)))
        lat = npzfile['lat']
        lon = npzfile['lon']
        nlat = npzfile['nlat']
        nlon = npzfile['nlon']
        lat2 = np.reshape(lat,(nlat,nlon))
        lon2 = np.reshape(lon,(nlat,nlon))
        years = npzfile['years']
        nyrs =  len(years)
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

print('\n shape of the ensemble array: ' + str(np.shape(xam_all)) +'\n')
print('\n shape of the ensemble-mean array: ' + str(np.shape(xam)) +'\n')
print('\n shape of the ensemble-mean prior array: ' + str(np.shape(xbm)) +'\n')

lmr_lat_range = (lat2[0,0],lat2[-1,0])
lmr_lon_range = (lon2[0,0],lon2[0,-1])
print('LMR grid info:')
print((' lats=%s' % str(lmr_lat_range)))
print((' lons=%s' % str(lmr_lon_range)))


# ===========================================================================================================
# BEGIN: load verification data (GISTEMP, MLOST, HadCRU, BE, 20CR and ERA20C) 
# ===========================================================================================================

print('\nloading verification data...\n')

# ===================
# Reanalysis products
# ===================

# Define month sequence for the calendar year 
# (argument needed in upload of reanalysis data)
annual = list(range(1,13))

# load ERA20C reanalysis -------------------------------------------------------
datadir = datadir_reanl+'/era20c'
datafile = 'tas_sfc_Amon_ERA20C_190001-201012.nc'
vardict = {'tas_sfc_Amon': 'anom'}
vardef = list(vardict.keys())[0]

dd = read_gridded_data_CMIP5_model(datadir,datafile,vardict,outtimeavg=annual)
rtime = dd[vardef]['years']
ERA20C_time = np.array([d.year for d in rtime])
lats = dd[vardef]['lat']
lons = dd[vardef]['lon']
latshape = lats.shape
lonshape = lons.shape
if len(latshape) == 2 & len(lonshape) == 2:
    # stored in 2D arrays
    lat_ERA20C = np.unique(lats)
    lon_ERA20C = np.unique(lons)
    nlat_ERA20C, = lat_ERA20C.shape
    nlon_ERA20C, = lon_ERA20C.shape
else:
    # stored in 1D arrays
    lon_ERA20C = lons
    lat_ERA20C = lats
    nlat_ERA20C = len(lat_ERA20C)
    nlon_ERA20C = len(lon_ERA20C)
lon2d_ERA20C, lat2d_ERA20C = np.meshgrid(lon_ERA20C, lat_ERA20C)

ERA20C = dd[vardef]['value'] + dd[vardef]['climo'] # Full field
#ERA20C = dd[vardef]['value']                      # Anomalies

# compute and remove the mean over 1951-1980 reference period as w/ GIS & BE
smatch, ematch = find_date_indices(ERA20C_time,1951,1980)
ref_mean_era = np.mean(ERA20C[smatch:ematch,:,:],axis=0)
ERA20C = ERA20C - ref_mean_era

# load 20th century reanalysis (TCR) reanalysis --------------------------------
datadir = datadir_reanl+'/20cr'
datafile = 'tas_sfc_Amon_20CR_185101-201112.nc'
vardict = {'tas_sfc_Amon': 'anom'}
vardef = list(vardict.keys())[0]

dd = read_gridded_data_CMIP5_model(datadir,datafile,vardict,outtimeavg=annual)
rtime = dd[vardef]['years']
TCR_time = np.array([d.year for d in rtime])
lats = dd[vardef]['lat']
lons = dd[vardef]['lon']
latshape = lats.shape
lonshape = lons.shape
if len(latshape) == 2 & len(lonshape) == 2:
    # stored in 2D arrays
    lat_TCR = np.unique(lats)
    lon_TCR = np.unique(lons)
    nlat_TCR, = lat_TCR.shape
    nlon_TCR, = lon_TCR.shape
else:
    # stored in 1D arrays
    lon_TCR = lons
    lat_TCR = lats
    nlat_TCR = len(lat_TCR)
    nlon_TCR = len(lon_TCR)
lon2d_TCR, lat2d_TCR = np.meshgrid(lon_TCR, lat_TCR)

TCR = dd[vardef]['value'] + dd[vardef]['climo'] # Full field
#TCR = dd[vardef]['value']                      # Anomalies

# compute and remove the mean over 1951-1980 reference period as w/ GIS & BE
smatch, ematch = find_date_indices(TCR_time,1951,1980)
ref_mean_tcr = np.mean(TCR[smatch:ematch,:,:],axis=0)
TCR = TCR - ref_mean_tcr

# ==================
# Obs-based products
# ==================

#datadir_calib = '../data/'
datadir_calib = '/home/disk/kalman3/rtardif/LMR/data/analyses'

# load GISTEMP -----------------------------------------------------------------
# Note: Anomalies w.r.t. 1951-1980 mean
datafile_calib   = 'gistemp1200_ERSSTv4.nc'
calib_vars = ['Tsfc']
[gtime,GIS_lat,GIS_lon,GIS_anomaly] = read_gridded_data_GISTEMP(datadir_calib,datafile_calib,calib_vars,outfreq='annual')
GIS_time = np.array([d.year for d in gtime])
nlat_GIS = len(GIS_lat)
nlon_GIS = len(GIS_lon)
lon2d_GIS, lat2d_GIS = np.meshgrid(GIS_lon, GIS_lat)
gis_lat_range = (lat2d_GIS[0,0],lat2d_GIS[-1,0])
gis_lon_range = (lon2d_GIS[0,0],lon2d_GIS[0,-1])
print('GIS grid info:')
print(' lats=%s' % str(gis_lat_range))
print(' lons=%s' % str(gis_lon_range))
# GIS longitudes are off by 180 degrees
print(' Shifting longitudes by 180 degrees')
lat2d_GIS = np.roll(lat2d_GIS,shift=nlon_GIS/2,axis=1)
lon2d_GIS = np.roll(lon2d_GIS,shift=nlon_GIS/2,axis=1)
GIS_anomaly = np.roll(GIS_anomaly,shift=nlon_GIS/2,axis=2)

# load HadCRUT -----------------------------------------------------------------
# Note: Anomalies w.r.t. 1961-1990 mean
datafile_calib   = 'HadCRUT.4.3.0.0.median.nc'
calib_vars = ['Tsfc']
[ctime,CRU_lat,CRU_lon,CRU_anomaly] = read_gridded_data_HadCRUT(datadir_calib,datafile_calib,calib_vars,outfreq='annual')
CRU_time = np.array([d.year for d in ctime])
nlat_CRU = len(CRU_lat)
nlon_CRU = len(CRU_lon)
lon2d_CRU, lat2d_CRU = np.meshgrid(CRU_lon, CRU_lat)
cru_lat_range = (lat2d_CRU[0,0],lat2d_CRU[-1,0])
cru_lon_range = (lon2d_CRU[0,0],lon2d_CRU[0,-1])
print('CRU grid info:')
print(' lats=%s' % str(cru_lat_range))
print(' lons=%s' % str(cru_lon_range))
# CRU longitudes are off by 180 degrees
print(' Shifting longitudes by 180 degrees')
lat2d_CRU = np.roll(lat2d_CRU,shift=nlon_CRU/2,axis=1)
lon2d_CRU = np.roll(lon2d_CRU,shift=nlon_CRU/2,axis=1)
CRU_anomaly = np.roll(CRU_anomaly,shift=nlon_CRU/2,axis=2)

# load BerkeleyEarth -----------------------------------------------------------
# Note: Anomalies w.r.t. 1951-1980 mean
datafile_calib   = 'Land_and_Ocean_LatLong1.nc'
calib_vars = ['Tsfc']
[btime,BE_lat,BE_lon,BE_anomaly] = read_gridded_data_BerkeleyEarth(datadir_calib,datafile_calib,calib_vars,outfreq='annual')
BE_time = np.array([d.year for d in btime])
nlat_BE = len(BE_lat)
nlon_BE = len(BE_lon)
lon2d_BE, lat2d_BE = np.meshgrid(BE_lon, BE_lat)
be_lat_range = (lat2d_BE[0,0],lat2d_BE[-1,0])
be_lon_range = (lon2d_BE[0,0],lon2d_BE[0,-1])
print('BE grid info:')
print(' lats=%s' % str(be_lat_range))
print(' lons=%s' % str(be_lon_range))
# BE longitudes are off by 180 degrees
print(' Shifting longitudes by 180 degrees')
lat2d_BE = np.roll(lat2d_BE,shift=nlon_BE/2,axis=1)
lon2d_BE = np.roll(lon2d_BE,shift=nlon_BE/2,axis=1)
BE_anomaly = np.roll(BE_anomaly,shift=nlon_BE/2,axis=2)

# load MLOST -------------------------------------------------------------------
# Note: Anomalies w.r.t. 1961-1990 mean
datafile_calib   = 'MLOST_air.mon.anom_V3.5.4.nc'
calib_vars = ['Tsfc']
[mtime,MLOST_lat,MLOST_lon,MLOST_anomaly] = read_gridded_data_MLOST(datadir_calib,datafile_calib,calib_vars,outfreq='annual')
MLOST_time = np.array([d.year for d in mtime])
nlat_MLOST = len(MLOST_lat)
nlon_MLOST = len(MLOST_lon)
lon2d_MLOST, lat2d_MLOST = np.meshgrid(MLOST_lon, MLOST_lat)
mlost_lat_range = (lat2d_MLOST[0,0],lat2d_MLOST[-1,0])
mlost_lon_range = (lon2d_MLOST[0,0],lon2d_MLOST[0,-1])
print('MLOST grid info:')
print(' lats=%s' % str(mlost_lat_range))
print(' lons=%s' % str(mlost_lon_range))


# ===========================================================================================================
# END: load verification data (GISTEMP, MLOST, HadCRU, BE, and 20CR) 
# ===========================================================================================================


# adjust so that all anomaly data pertain to the mean over a user-defined reference period (e.g. 20th century)
stime = ref_period[0]
etime = ref_period[1]

# LMR
LMR_anomaly = xam
smatch, ematch = find_date_indices(LMR_time,stime,etime)
LMR_anomaly = LMR_anomaly - np.nanmean(LMR_anomaly[smatch:ematch,:,:],axis=0)

# TCR
smatch, ematch = find_date_indices(TCR_time,stime,etime)
TCR = TCR - np.nanmean(TCR[smatch:ematch,:,:],axis=0)
# ERA
smatch, ematch = find_date_indices(ERA20C_time,stime,etime)
ERA20C = ERA20C - np.nanmean(ERA20C[smatch:ematch,:,:],axis=0)

# GIS
smatch, ematch = find_date_indices(GIS_time,stime,etime)
GIS_anomaly = GIS_anomaly - np.nanmean(GIS_anomaly[smatch:ematch,:,:],axis=0)
# BE
smatch, ematch = find_date_indices(BE_time,stime,etime)
BE_anomaly = BE_anomaly - np.nanmean(BE_anomaly[smatch:ematch,:,:],axis=0)
# CRU
smatch, ematch = find_date_indices(CRU_time,stime,etime)
CRU_anomaly = CRU_anomaly - np.nanmean(CRU_anomaly[smatch:ematch,:,:],axis=0)
# MLOST
smatch, ematch = find_date_indices(MLOST_time,stime,etime)
MLOST_anomaly = MLOST_anomaly - np.nanmean(MLOST_anomaly[smatch:ematch,:,:],axis=0)


print('\n regridding LMR data to grids of verification data...\n')

iplot_loc= False
#iplot_loc= True

# create instance of the spherical harmonics object for each grid
specob_lmr   = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')
specob_gis   = Spharmt(nlon_GIS,nlat_GIS,gridtype='regular',legfunc='computed')
specob_be    = Spharmt(nlon_BE,nlat_BE,gridtype='regular',legfunc='computed')
specob_cru   = Spharmt(nlon_CRU,nlat_CRU,gridtype='regular',legfunc='computed')
specob_mlost = Spharmt(nlon_MLOST,nlat_MLOST,gridtype='regular',legfunc='computed')
specob_tcr   = Spharmt(nlon_TCR,nlat_TCR,gridtype='regular',legfunc='computed')
specob_era   = Spharmt(nlon_ERA20C,nlat_ERA20C,gridtype='regular',legfunc='computed')


# loop over years of interest and transform...specify trange at top of file

iw = 0
if nya > 0:
    iw = (nya-1)/2

cyears = list(range(trange[0],trange[1]))

# time series for combination of data products
lmr_tcr_csave   = np.zeros([len(cyears)])
lmr_era_csave   = np.zeros([len(cyears)])
lmr_gis_csave   = np.zeros([len(cyears)])
lmr_be_csave    = np.zeros([len(cyears)])
lmr_cru_csave   = np.zeros([len(cyears)])
lmr_mlost_csave = np.zeros([len(cyears)])

tcr_gis_csave = np.zeros([len(cyears)])
era_gis_csave = np.zeros([len(cyears)])


# for full 2d grids
# -----------------
# obs products
tcr_allyears   = np.zeros([len(cyears),nlat_TCR,nlon_TCR])
era_allyears   = np.zeros([len(cyears),nlat_ERA20C,nlon_ERA20C])
gis_allyears   = np.zeros([len(cyears),nlat_GIS,nlon_GIS])
be_allyears    = np.zeros([len(cyears),nlat_BE,nlon_BE])
cru_allyears   = np.zeros([len(cyears),nlat_CRU,nlon_CRU])
mlost_allyears = np.zeros([len(cyears),nlat_MLOST,nlon_MLOST])

# for ERA20C (reference) projected over the various grids
era_on_tcr_allyears   = np.zeros([len(cyears),nlat_TCR,nlon_TCR])
era_on_gis_allyears   = np.zeros([len(cyears),nlat_GIS,nlon_GIS])
era_on_be_allyears    = np.zeros([len(cyears),nlat_BE,nlon_BE])
era_on_cru_allyears   = np.zeros([len(cyears),nlat_CRU,nlon_CRU])
era_on_mlost_allyears = np.zeros([len(cyears),nlat_MLOST,nlon_MLOST])

tcr_on_gis_allyears = np.zeros([len(cyears),nlat_GIS,nlon_GIS])

# for lmr projected over the various grids
lmr_on_tcr_allyears   = np.zeros([len(cyears),nlat_TCR,nlon_TCR])
lmr_on_era_allyears   = np.zeros([len(cyears),nlat_ERA20C,nlon_ERA20C])
lmr_on_gis_allyears   = np.zeros([len(cyears),nlat_GIS,nlon_GIS])
lmr_on_be_allyears    = np.zeros([len(cyears),nlat_BE,nlon_BE])
lmr_on_cru_allyears   = np.zeros([len(cyears),nlat_CRU,nlon_CRU])
lmr_on_mlost_allyears = np.zeros([len(cyears),nlat_MLOST,nlon_MLOST])

# for prior projected over the various grids
xbm_on_tcr_allyears   = np.zeros([len(cyears),nlat_TCR,nlon_TCR])
xbm_on_era_allyears   = np.zeros([len(cyears),nlat_ERA20C,nlon_ERA20C])
xbm_on_gis_allyears   = np.zeros([len(cyears),nlat_GIS,nlon_GIS])
xbm_on_be_allyears    = np.zeros([len(cyears),nlat_BE,nlon_BE])
xbm_on_cru_allyears   = np.zeros([len(cyears),nlat_CRU,nlon_CRU])
xbm_on_mlost_allyears = np.zeros([len(cyears),nlat_MLOST,nlon_MLOST])


# for zonal means
# ---------------
# obs products
tcr_zm   = np.zeros([len(cyears),nlat_TCR])
era_zm   = np.zeros([len(cyears),nlat_ERA20C])
gis_zm   = np.zeros([len(cyears),nlat_GIS])
be_zm    = np.zeros([len(cyears),nlat_BE])
cru_zm   = np.zeros([len(cyears),nlat_CRU])
mlost_zm = np.zeros([len(cyears),nlat_MLOST])

# for lmr projected over the various grids
lmr_on_tcr_zm   = np.zeros([len(cyears),nlat_TCR])
lmr_on_era_zm   = np.zeros([len(cyears),nlat_ERA20C])
lmr_on_gis_zm   = np.zeros([len(cyears),nlat_GIS])
lmr_on_be_zm    = np.zeros([len(cyears),nlat_BE])
lmr_on_cru_zm   = np.zeros([len(cyears),nlat_CRU])
lmr_on_mlost_zm = np.zeros([len(cyears),nlat_MLOST])

tcr_on_gis_zm = np.zeros([len(cyears),nlat_GIS])
era_on_gis_zm = np.zeros([len(cyears),nlat_GIS])

# for prior projected over the various grids
xbm_on_tcr_zm   = np.zeros([len(cyears),nlat_TCR])
xbm_on_era_zm   = np.zeros([len(cyears),nlat_ERA20C])
xbm_on_gis_zm   = np.zeros([len(cyears),nlat_GIS])
xbm_on_be_zm    = np.zeros([len(cyears),nlat_BE])
xbm_on_cru_zm   = np.zeros([len(cyears),nlat_CRU])
xbm_on_mlost_zm = np.zeros([len(cyears),nlat_MLOST])


# Loop over years defining the verification set
k = -1
for yr in cyears:
    k = k + 1
    LMR_smatch, LMR_ematch     = find_date_indices(LMR_time,yr-iw,yr+iw+1)
    TCR_smatch, TCR_ematch     = find_date_indices(TCR_time,yr-iw,yr+iw+1)
    ERA_smatch, ERA_ematch     = find_date_indices(ERA20C_time,yr-iw,yr+iw+1)
    GIS_smatch, GIS_ematch     = find_date_indices(GIS_time,yr-iw,yr+iw+1)
    BE_smatch, BE_ematch       = find_date_indices(BE_time,yr-iw,yr+iw+1)
    CRU_smatch, CRU_ematch     = find_date_indices(CRU_time,yr-iw,yr+iw+1)
    MLOST_smatch, MLOST_ematch = find_date_indices(MLOST_time,yr-iw,yr+iw+1)

    print('------------------------------------------------------------------------')
    print('working on year... %s' % str(yr))
    print('working on year... %5s LMR index = %5s : LMR year = %5s' % (str(yr),str(LMR_smatch),str(LMR_time[LMR_smatch])))

    
    # TCR
    if TCR_smatch and TCR_ematch:
        tcr_verif = np.mean(TCR[TCR_smatch:TCR_ematch,:,:],0)
    else:
        tcr_verif = np.zeros(shape=[nlat_TCR,nlon_TCR])
        tcr_verif.fill(np.nan)
    # ERA
    if ERA_smatch and ERA_ematch:
        era_verif = np.mean(ERA20C[ERA_smatch:ERA_ematch,:,:],0)
    else:
        era_verif = np.zeros(shape=[nlat_ERA20C,nlon_ERA20C])
        era_verif.fill(np.nan)
    # GIS
    if GIS_smatch and GIS_ematch:
        gis_verif = np.mean(GIS_anomaly[GIS_smatch:GIS_ematch,:,:],0)
    else:
        gis_verif = np.zeros(shape=[nlat_GIS,nlon_GIS])
        gis_verif.fill(np.nan)
    # BE
    if BE_smatch and BE_ematch:
        be_verif = np.mean(BE_anomaly[BE_smatch:BE_ematch,:,:],0)
    else:
        be_verif = np.zeros(shape=[nlat_BE,nlon_BE])
        be_verif.fill(np.nan)
    # CRU
    if CRU_smatch and CRU_ematch:
        cru_verif = np.mean(CRU_anomaly[CRU_smatch:CRU_ematch,:,:],0)
    else:
        cru_verif = np.zeros(shape=[nlat_CRU,nlon_CRU])
        cru_verif.fill(np.nan)
    # MLOST
    if MLOST_smatch and MLOST_ematch:
        mlost_verif = np.mean(MLOST_anomaly[MLOST_smatch:MLOST_ematch,:,:],0)
    else:
        mlost_verif = np.zeros(shape=[nlat_MLOST,nlon_MLOST])
        mlost_verif.fill(np.nan)

    if iplot_loc:
        fig = plt.figure()
        vmin = -3.0; vmax = 3.0
        cbarfmt = '%4.1f'
        nticks = 6 # number of ticks on the colorbar

        ax = fig.add_subplot(3,2,1)
        LMR_plotter(gis_verif,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=vmin,vmax=vmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('GIS : '+str(yr))
        plt.clim(vmin,vmax)
        ax = fig.add_subplot(3,2,2)
        LMR_plotter(be_verif,lat2d_BE,lon2d_BE,'bwr',nlevs,vmin=vmin,vmax=vmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('BE : '+str(yr))
        plt.clim(vmin,vmax)
        ax = fig.add_subplot(3,2,3)
        LMR_plotter(cru_verif,lat2d_CRU,lon2d_CRU,'bwr',nlevs,vmin=vmin,vmax=vmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('CRU : '+str(yr))
        plt.clim(vmin,vmax)
        ax = fig.add_subplot(3,2,4)
        LMR_plotter(mlost_verif,lat2d_MLOST,lon2d_MLOST,'bwr',nlevs,vmin=vmin,vmax=vmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('MLOST : '+str(yr))
        plt.clim(vmin,vmax)
        ax = fig.add_subplot(3,2,5)
        LMR_plotter(tcr_verif,lat2d_TCR,lon2d_TCR,'bwr',nlevs,vmin=vmin,vmax=vmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('20CR : '+str(yr))
        plt.clim(vmin,vmax)
        ax = fig.add_subplot(3,2,6)
        LMR_plotter(era_verif,lat2d_ERA20C,lon2d_ERA20C,'bwr',nlevs,vmin=vmin,vmax=vmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('ERA20C : '+str(yr))
        plt.clim(vmin,vmax)

        fig.tight_layout()
        plt.savefig('gis_be_cru_mlost_tcr_era_%s.png' %str(yr))
        plt.close()



    # LMR anomaly for current period
    pdata_lmr = np.mean(LMR_anomaly[LMR_smatch:LMR_ematch,:,:],0)
    # prior
    pdata_xbm = np.mean(xbm[LMR_smatch:LMR_ematch,:,:],0)

    # regrid LMR on the various verification grids
    lmr_on_gis   = regrid(specob_lmr, specob_gis,   pdata_lmr, ntrunc=None, smooth=None)
    lmr_on_be    = regrid(specob_lmr, specob_be,    pdata_lmr, ntrunc=None, smooth=None)
    lmr_on_cru   = regrid(specob_lmr, specob_cru,   pdata_lmr, ntrunc=None, smooth=None)
    lmr_on_mlost = regrid(specob_lmr, specob_mlost, pdata_lmr, ntrunc=None, smooth=None)
    lmr_on_tcr   = regrid(specob_lmr, specob_tcr,   pdata_lmr, ntrunc=None, smooth=None)
    lmr_on_era   = regrid(specob_lmr, specob_era,   pdata_lmr, ntrunc=None, smooth=None)
    # prior
    xbm_on_gis   = regrid(specob_lmr, specob_gis,   pdata_xbm, ntrunc=None, smooth=None)
    xbm_on_be    = regrid(specob_lmr, specob_be,    pdata_xbm, ntrunc=None, smooth=None)
    xbm_on_cru   = regrid(specob_lmr, specob_cru,   pdata_xbm, ntrunc=None, smooth=None)
    xbm_on_mlost = regrid(specob_lmr, specob_mlost, pdata_xbm, ntrunc=None, smooth=None)
    xbm_on_tcr   = regrid(specob_lmr, specob_tcr,   pdata_xbm, ntrunc=None, smooth=None)
    xbm_on_era   = regrid(specob_lmr, specob_era,   pdata_xbm, ntrunc=None, smooth=None)

    # regrid TCR & ERA reanalyses on GIS grid (for reference)
    if np.isnan(tcr_verif).all():
        tcr_on_gis = np.zeros(shape=[nlat_GIS,nlon_GIS])
        tcr_on_gis.fill(np.nan)
    else:
        tcr_on_gis   = regrid(specob_tcr, specob_gis, tcr_verif, ntrunc=None, smooth=None)

    if np.isnan(era_verif).all():
        era_on_gis = np.zeros(shape=[nlat_GIS,nlon_GIS])
        era_on_gis.fill(np.nan)
    else:
        era_on_gis   = regrid(specob_era, specob_gis, era_verif, ntrunc=None, smooth=None)

    # save the full grids
    era_allyears[k,:,:]   = era_verif
    tcr_allyears[k,:,:]   = tcr_verif
    gis_allyears[k,:,:]   = gis_verif
    be_allyears[k,:,:]    = be_verif
    cru_allyears[k,:,:]   = cru_verif
    mlost_allyears[k,:,:] = mlost_verif

    lmr_on_gis_allyears[k,:,:]   = lmr_on_gis
    lmr_on_be_allyears[k,:,:]    = lmr_on_be
    lmr_on_cru_allyears[k,:,:]   = lmr_on_cru
    lmr_on_mlost_allyears[k,:,:] = lmr_on_mlost
    lmr_on_tcr_allyears[k,:,:]   = lmr_on_tcr
    lmr_on_era_allyears[k,:,:]   = lmr_on_era

    tcr_on_gis_allyears[k,:,:]   = tcr_on_gis
    era_on_gis_allyears[k,:,:]   = era_on_gis

    # prior
    xbm_on_gis_allyears[k,:,:]   = xbm_on_gis
    xbm_on_be_allyears[k,:,:]    = xbm_on_be
    xbm_on_cru_allyears[k,:,:]   = xbm_on_cru
    xbm_on_mlost_allyears[k,:,:] = xbm_on_mlost
    xbm_on_tcr_allyears[k,:,:]   = xbm_on_tcr
    xbm_on_era_allyears[k,:,:]   = xbm_on_era


    # 
    # compute zonal-mean values
    #

    # ERA
    fracok    = np.sum(np.isfinite(era_verif),axis=1,dtype=np.float16)/float(nlon_ERA20C)
    boolok    = np.where(fracok >= valid_frac)
    boolnotok = np.where(fracok < valid_frac)
    for i in boolok:
        era_zm[k,i] = np.nanmean(era_verif[i,:],axis=1)
    era_zm[k,boolnotok]  = np.NAN
    lmr_on_era_zm[k,:]   = np.mean(lmr_on_era,1)
    xbm_on_era_zm[k,:]   = np.mean(xbm_on_era,1)

    # TCR
    fracok    = np.sum(np.isfinite(tcr_verif),axis=1,dtype=np.float16)/float(nlon_TCR)
    boolok    = np.where(fracok >= valid_frac)
    boolnotok = np.where(fracok < valid_frac)
    for i in boolok:
        tcr_zm[k,i] = np.nanmean(tcr_verif[i,:],axis=1)
    tcr_zm[k,boolnotok]  = np.NAN
    lmr_on_tcr_zm[k,:]   = np.mean(lmr_on_tcr,1)
    xbm_on_tcr_zm[k,:]   = np.mean(xbm_on_tcr,1)

    # GIS
    fracok    = np.sum(np.isfinite(gis_verif),axis=1,dtype=np.float16)/float(nlon_GIS)
    boolok    = np.where(fracok >= valid_frac)
    boolnotok = np.where(fracok < valid_frac)
    for i in boolok:
        gis_zm[k,i] = np.nanmean(gis_verif[i,:],axis=1)
    gis_zm[k,boolnotok]  = np.NAN
    lmr_on_gis_zm[k,:]   = np.mean(lmr_on_gis,axis=1)
    tcr_on_gis_zm[k,:]   = np.mean(tcr_on_gis,axis=1)
    xbm_on_gis_zm[k,:]   = np.mean(xbm_on_gis,axis=1)

    # BE
    fracok    = np.sum(np.isfinite(be_verif),axis=1,dtype=np.float16)/float(nlon_BE)
    boolok    = np.where(fracok >= valid_frac)
    boolnotok = np.where(fracok < valid_frac)
    for i in boolok:
        be_zm[k,i] = np.nanmean(be_verif[i,:],axis=1)
    be_zm[k,boolnotok]  = np.NAN
    lmr_on_be_zm[k,:]   = np.mean(lmr_on_be,1)
    xbm_on_be_zm[k,:]   = np.mean(xbm_on_be,1)

    # CRU
    fracok    = np.sum(np.isfinite(cru_verif),axis=1,dtype=np.float16)/float(nlon_CRU)
    boolok    = np.where(fracok >= valid_frac)
    boolnotok = np.where(fracok < valid_frac)
    for i in boolok:
        cru_zm[k,i] = np.nanmean(cru_verif[i,:],axis=1)
    cru_zm[k,boolnotok]  = np.NAN
    lmr_on_cru_zm[k,:]   = np.mean(lmr_on_cru,1)
    xbm_on_cru_zm[k,:]   = np.mean(xbm_on_cru,1)

    # MLOST
    fracok    = np.sum(np.isfinite(mlost_verif),axis=1,dtype=np.float16)/float(nlon_MLOST)
    boolok    = np.where(fracok >= valid_frac)
    boolnotok = np.where(fracok < valid_frac)
    for i in boolok:
        mlost_zm[k,i] = np.nanmean(mlost_verif[i,:],axis=1)
    mlost_zm[k,boolnotok]  = np.NAN
    lmr_on_mlost_zm[k,:] = np.mean(lmr_on_mlost,1)
    xbm_on_mlost_zm[k,:] = np.mean(xbm_on_mlost,1)


    # ===============================================================================
    # anomaly correlation
    # ===============================================================================

    # --------------
    # prepare arrays
    # --------------
    tcr_vec   = np.reshape(tcr_verif,(1,nlat_TCR*nlon_TCR))
    era_vec   = np.reshape(era_verif,(1,nlat_ERA20C*nlon_ERA20C))
    gis_vec   = np.reshape(gis_verif,(1,nlat_GIS*nlon_GIS))
    be_vec    = np.reshape(be_verif,(1,nlat_BE*nlon_BE))
    cru_vec   = np.reshape(cru_verif,(1,nlat_CRU*nlon_CRU))
    mlost_vec = np.reshape(mlost_verif,(1,nlat_MLOST*nlon_MLOST))

    lmr_on_tcr_vec   = np.reshape(lmr_on_tcr,(1,nlat_TCR*nlon_TCR))
    lmr_on_era_vec   = np.reshape(lmr_on_era,(1,nlat_ERA20C*nlon_ERA20C))
    lmr_on_gis_vec   = np.reshape(lmr_on_gis,(1,nlat_GIS*nlon_GIS))
    lmr_on_be_vec    = np.reshape(lmr_on_be,(1,nlat_BE*nlon_BE))
    lmr_on_cru_vec   = np.reshape(lmr_on_cru,(1,nlat_CRU*nlon_CRU))
    lmr_on_mlost_vec = np.reshape(lmr_on_mlost,(1,nlat_MLOST*nlon_MLOST))

    tcr_on_gis_vec   = np.reshape(tcr_on_gis,(1,nlat_GIS*nlon_GIS))
    era_on_gis_vec   = np.reshape(era_on_gis,(1,nlat_GIS*nlon_GIS))

    # ---------------------------------------------------------------------------
    # compute correlations, taking into account the missing data in obs. products
    # ---------------------------------------------------------------------------
    # lmr <-> tcr
    indok = np.isfinite(tcr_vec); nbok = np.sum(indok); nball = tcr_vec.shape[1]
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        lmr_tcr_csave[k] = np.corrcoef(lmr_on_tcr_vec[indok],tcr_vec[indok])[0,1]
    else:
        lmr_tcr_csave[k] = np.nan
    print('  lmr-tcr correlation     : %s' % str(lmr_tcr_csave[k]))

    # lmr <-> era
    indok = np.isfinite(era_vec); nbok = np.sum(indok); nball = era_vec.shape[1]
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        lmr_era_csave[k] = np.corrcoef(lmr_on_era_vec[indok],era_vec[indok])[0,1]
    else:
        lmr_era_csave[k] = np.nan
    print('  lmr-era correlation     : %s' % str(lmr_era_csave[k]))
    
    # lmr <-> gis
    indok = np.isfinite(gis_vec); nbok = np.sum(indok); nball = gis_vec.shape[1]
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        lmr_gis_csave[k] = np.corrcoef(lmr_on_gis_vec[indok],gis_vec[indok])[0,1]
    else:
        lmr_gis_csave[k] = np.nan
    print('  lmr-gis correlation     : %s' % str(lmr_gis_csave[k]))
    
    # lmr <-> be
    indok = np.isfinite(be_vec); nbok = np.sum(indok); nball = be_vec.shape[1]
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        lmr_be_csave[k] = np.corrcoef(lmr_on_be_vec[indok],be_vec[indok])[0,1]
    else:
        lmr_be_csave[k] = np.nan
    print('  lmr-be correlation      : %s' % str(lmr_be_csave[k]))
    
    # lmr <-> cru
    indok = np.isfinite(cru_vec); nbok = np.sum(indok); nball = cru_vec.shape[1]
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        lmr_cru_csave[k] = np.corrcoef(lmr_on_cru_vec[indok],cru_vec[indok])[0,1]
    else:
        lmr_cru_csave[k] = np.nan
    print('  lmr-cru correlation     : %s' % str(lmr_cru_csave[k]))
    
    # lmr <-> mlost
    indok = np.isfinite(mlost_vec); nbok = np.sum(indok); nball = mlost_vec.shape[1]
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        lmr_mlost_csave[k] = np.corrcoef(lmr_on_mlost_vec[indok],mlost_vec[indok])[0,1]
    else:
        lmr_mlost_csave[k] = np.nan
    print('  lmr-mlost correlation   : %s' % str(lmr_mlost_csave[k]))
    
    # tcr <-> gis
    if ~np.isnan(tcr_on_gis_vec).all():
        indok = np.isfinite(gis_vec); nbok = np.sum(indok); nball = gis_vec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            tcr_gis_csave[k] = np.corrcoef(tcr_on_gis_vec[indok],gis_vec[indok])[0,1]
        else:
            tcr_gis_csave[k] = np.nan
    else:
        tcr_gis_csave[k] = np.nan
    print('  tcr-gis correlation     : %s' % str(tcr_gis_csave[k]))
    
    # era <-> gis
    if ~np.isnan(era_on_gis_vec).all():
        indok = np.isfinite(gis_vec); nbok = np.sum(indok); nball = gis_vec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            era_gis_csave[k] = np.corrcoef(era_on_gis_vec[indok],gis_vec[indok])[0,1]
        else:
            era_gis_csave[k] = np.nan
    else:
        era_gis_csave[k] = np.nan
    print('  era-gis correlation     : %s' % str(era_gis_csave[k]))
    

# ===================================================================================
# plots for anomaly correlation statistics
# ===================================================================================
# number of bins in the histograms
nbins = 15
corr_range = [-0.6,0.8]
#corr_range = [-0.4,1.0]
bins = np.linspace(corr_range[0],corr_range[1],nbins)

# LMR compared to TCR, ERA, GIS, BE, CRU and MLOST
#fig = plt.figure(figsize=(10,12))
fig = plt.figure()

# TCR
ax = fig.add_subplot(6,2,1)
ax.plot(cyears,lmr_tcr_csave,lw=2)
ax.plot([trange[0],trange[-1]],[0,0],'k:')
ax.set_title('LMR - 20CR-V2')
ax.set_xlim(trange[0],trange[-1])
ax.set_ylim(corr_range[0],corr_range[-1])
ax.set_ylabel('Correlation',fontweight='bold')
ax = fig.add_subplot(6,2,2)
ax.hist(lmr_tcr_csave[~np.isnan(lmr_tcr_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
ax.set_title('LMR - 20CR-V2')
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
ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lmr_tcr_csave)),fontsize=11,fontweight='bold')

# ERA
ax = fig.add_subplot(6,2,3)
ax.plot(cyears,lmr_era_csave,lw=2)
ax.plot([trange[0],trange[-1]],[0,0],'k:')
ax.set_title('LMR - ERA-20C')
ax.set_xlim(trange[0],trange[-1])
ax.set_ylim(corr_range[0],corr_range[-1])
ax.set_ylabel('Correlation',fontweight='bold')
ax = fig.add_subplot(6,2,4)
ax.hist(lmr_era_csave[~np.isnan(lmr_era_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
ax.set_title('LMR - ERA-20C')
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
ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lmr_era_csave)),fontsize=11,fontweight='bold')

# GIS
ax = fig.add_subplot(6,2,5)
ax.plot(cyears,lmr_gis_csave,lw=2)
ax.plot([trange[0],trange[-1]],[0,0],'k:')
ax.set_title('LMR - GISTEMP')
ax.set_xlim(trange[0],trange[-1])
ax.set_ylim(corr_range[0],corr_range[-1])
ax.set_ylabel('Correlation',fontweight='bold')
ax = fig.add_subplot(6,2,6)
indok = np.isfinite(lmr_gis_csave)
if np.sum(indok) > 0:
    ax.hist(lmr_gis_csave[indok],bins=bins,histtype='stepfilled',alpha=0.25,
            range=(bins.min(), bins.max()))
ax.set_title('LMR - GISTEMP')
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
ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lmr_gis_csave)),fontsize=11,fontweight='bold')
print('LMR-GIS mean anomaly correlation: ' + str (np.nanmean(lmr_gis_csave)))

# BE
ax = fig.add_subplot(6,2,7)
ax.plot(cyears,lmr_be_csave,lw=2)
ax.set_title('LMR - BE')
ax.plot([trange[0],trange[-1]],[0,0],'k:')
ax.set_xlim(trange[0],trange[-1])
ax.set_ylim(corr_range[0],corr_range[-1])
ax.set_ylabel('Correlation',fontweight='bold')
ax = fig.add_subplot(6,2,8)
indok = np.isfinite(lmr_be_csave)
if np.sum(indok) > 0:
    ax.hist(lmr_be_csave[indok],bins=bins,histtype='stepfilled',alpha=0.25,
            range=(bins.min(), bins.max()))
ax.set_title('LMR - BE')
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
ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lmr_be_csave)),fontsize=11,fontweight='bold')

# CRU
ax = fig.add_subplot(6,2,9)
ax.plot(cyears,lmr_cru_csave,lw=2)
ax.set_title('LMR - HadCRUT4')
ax.plot([trange[0],trange[-1]],[0,0],'k:')
ax.set_xlim(trange[0],trange[-1])
ax.set_ylim(corr_range[0],corr_range[-1])
ax.set_ylabel('Correlation',fontweight='bold')
ax = fig.add_subplot(6,2,10)
indok = np.isfinite(lmr_cru_csave)
if np.sum(indok) > 0:
    ax.hist(lmr_cru_csave[indok],bins=bins,histtype='stepfilled',alpha=0.25,
        range=(bins.min(), bins.max()))
ax.set_title('LMR - HadCRUT4')
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
ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lmr_cru_csave)),fontsize=11,fontweight='bold')

# MLOST
ax = fig.add_subplot(6,2,11)
ax.plot(cyears,lmr_mlost_csave,lw=2)
ax.set_title('LMR - MLOST')
ax.plot([trange[0],trange[-1]],[0,0],'k:')
ax.set_xlim(trange[0],trange[-1])
ax.set_ylim(corr_range[0],corr_range[-1])
ax.set_ylabel('Correlation',fontweight='bold')
ax.set_xlabel('Year CE',fontweight='bold')
ax = fig.add_subplot(6,2,12)
indok = np.isfinite(lmr_mlost_csave)
if np.sum(indok) > 0:
    ax.hist(lmr_mlost_csave[indok],bins=bins,histtype='stepfilled',alpha=0.25,
        range=(bins.min(), bins.max()))
ax.set_title('LMR - MLOST')
ax.set_xlim(corr_range[0],corr_range[-1])
ax.set_ylabel('Counts',fontweight='bold')
ax.set_xlabel('Correlation',fontweight='bold')
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
if ymax < 10:
    ymax = 10
elif ymax >= 10 and ymax < 20:
    ymax = 20
ax.set_ylim(ymin,ymax)
ypos = ymax-0.15*(ymax-ymin)
xpos = xmin+0.025*(xmax-xmin)
ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lmr_mlost_csave)),fontsize=11,fontweight='bold')

fig.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.93, wspace=0.5, hspace=0.5)
fig.suptitle('Surface air temperature anomaly correlation',fontweight='bold') 
if fsave:
    print('saving to .png')
    plt.savefig(nexp+'_verify_grid_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'.png')
    plt.savefig(nexp+'_verify_grid_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')


    # =================================================================================================================================
    # For paper 1 :
    
    # in paper:
    fig = plt.figure()
    # TCR
    ax = fig.add_subplot(2,2,1)
    ax.plot(cyears,lmr_tcr_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('LMR - 20CR-V2')
    ax.set_xlim(trange[0],trange[-1])
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    ax = fig.add_subplot(2,2,2)
    ax.hist(lmr_tcr_csave,bins=bins,histtype='stepfilled',alpha=0.25,
        range=(bins.min(), bins.max()))
    ax.set_title('LMR - 20CR-V2')
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
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lmr_tcr_csave)),fontsize=11,fontweight='bold')

    # GIS
    ax = fig.add_subplot(2,2,3)
    ax.plot(cyears,lmr_gis_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('LMR - GISTEMP')
    ax.set_xlim(trange[0],trange[-1])
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    ax.set_xlabel('Year CE',fontweight='bold')    
    ax = fig.add_subplot(2,2,4)
    indok = np.isfinite(lmr_gis_csave)
    if np.sum(indok) > 0:
        ax.hist(lmr_gis_csave[indok],bins=bins,histtype='stepfilled',alpha=0.25,
                range=(bins.min(), bins.max()))
    ax.set_title('LMR - GISTEMP')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    ax.set_xlabel('Correlation',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    if ymax < 10:
        ymax = 10
    elif ymax >= 10 and ymax < 20:
        ymax = 20
    ax.set_ylim(ymin,ymax)
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lmr_gis_csave)),fontsize=11,fontweight='bold')


    fig.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.45, right=0.95, top=0.93, wspace=0.5, hspace=0.5)
    fig.suptitle('Surface air temperature anomaly correlation',fontweight='bold') 

    print('saving to .png')
    plt.savefig(nexp+'_verify_grid_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'_paper.png')
    plt.savefig(nexp+'_verify_grid_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'_paper.pdf',bbox_inches='tight', dpi=300, format='pdf')

    # =================================================================================================================================



# TCR compared to GIS (reference)
fig = plt.figure()
ax = fig.add_subplot(2,2,1)
ax.plot(cyears,tcr_gis_csave,lw=2)
ax.set_title('20CR-V2 - GISTEMP')
ax.plot([trange[0],trange[-1]],[0,0],'k:')
ax.set_xlim(trange[0],trange[-1])
ax.set_ylim(corr_range[0],corr_range[-1])
ax.set_ylabel('Correlation',fontweight='bold')
ax = fig.add_subplot(2,2,2)
ax.hist(tcr_gis_csave[~np.isnan(tcr_gis_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
ax.set_title('20CR-V2 - GISTEMP')
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
ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(tcr_gis_csave)),fontsize=11,fontweight='bold')

# ERA compared to GIS (reference)
ax = fig.add_subplot(2,2,3)
ax.plot(cyears,era_gis_csave,lw=2)
ax.set_title('ERA-20C - GISTEMP')
ax.plot([trange[0],trange[-1]],[0,0],'k:')
ax.set_xlim(trange[0],trange[-1])
ax.set_ylim(corr_range[0],corr_range[-1])
ax.set_ylabel('Correlation',fontweight='bold')
ax.set_xlabel('Year CE',fontweight='bold')
ax = fig.add_subplot(2,2,4)
ax.hist(era_gis_csave[~np.isnan(era_gis_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
ax.set_title('ERA-20C - GISTEMP')
ax.set_xlim(corr_range[0],corr_range[-1])
ax.set_ylabel('Counts',fontweight='bold')
ax.set_xlabel('Correlation',fontweight='bold')
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
if ymax < 10:
    ymax = 10
elif ymax >= 10 and ymax < 20:
    ymax = 20
ax.set_ylim(ymin,ymax)
ypos = ymax-0.15*(ymax-ymin)
xpos = xmin+0.025*(xmax-xmin)
ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(era_gis_csave)),fontsize=11,fontweight='bold')

#fig.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.55, right=0.95, top=0.93, wspace=0.5, hspace=0.5)
fig.suptitle('Surface air temperature anomaly correlation',fontweight='bold') 
if fsave:
    print('saving to .png')
    plt.savefig(nexp+'_verify_grid_anomaly_correlation_reference_'+str(trange[0])+'-'+str(trange[1])+'.png')
    plt.savefig(nexp+'_verify_grid_anomaly_correlation_reference_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')

# ===================================================================================
# BEGIN bias, r and CE calculations
# ===================================================================================

# correlation and CE at each (lat,lon) point

lmr_on_tcr_err   = lmr_on_tcr_allyears - tcr_allyears
lmr_on_era_err   = lmr_on_era_allyears - era_allyears
lmr_on_gis_err   = lmr_on_gis_allyears - gis_allyears
lmr_on_be_err    = lmr_on_be_allyears  - be_allyears
lmr_on_cru_err   = lmr_on_cru_allyears - cru_allyears
lmr_on_mlost_err = lmr_on_mlost_allyears - mlost_allyears
tcr_on_gis_err   = tcr_on_gis_allyears - gis_allyears
era_on_gis_err   = era_on_gis_allyears - gis_allyears
# prior
xbm_on_tcr_err   = xbm_on_tcr_allyears - tcr_allyears
xbm_on_era_err   = xbm_on_era_allyears - era_allyears
xbm_on_gis_err   = xbm_on_gis_allyears - gis_allyears
xbm_on_be_err    = xbm_on_be_allyears  - be_allyears
xbm_on_cru_err   = xbm_on_cru_allyears - cru_allyears
xbm_on_mlost_err = xbm_on_mlost_allyears - mlost_allyears

bias_lmr_tcr   = np.zeros([nlat_TCR,nlon_TCR])
bias_lmr_era   = np.zeros([nlat_ERA20C,nlon_ERA20C])
bias_lmr_gis   = np.zeros([nlat_GIS,nlon_GIS])
bias_lmr_be    = np.zeros([nlat_BE,nlon_BE])
bias_lmr_cru   = np.zeros([nlat_CRU,nlon_CRU])
bias_lmr_mlost = np.zeros([nlat_MLOST,nlon_MLOST])
bias_tcr_gis   = np.zeros([nlat_GIS,nlon_GIS])
bias_era_gis   = np.zeros([nlat_GIS,nlon_GIS])
# prior
bias_xbm_tcr   = np.zeros([nlat_TCR,nlon_TCR])
bias_xbm_era   = np.zeros([nlat_ERA20C,nlon_ERA20C])
bias_xbm_gis   = np.zeros([nlat_GIS,nlon_GIS])
bias_xbm_be    = np.zeros([nlat_BE,nlon_BE])
bias_xbm_cru   = np.zeros([nlat_CRU,nlon_CRU])
bias_xbm_mlost = np.zeros([nlat_MLOST,nlon_MLOST])

r_lmr_tcr    = np.zeros([nlat_TCR,nlon_TCR])
ce_lmr_tcr   = np.zeros([nlat_TCR,nlon_TCR])
r_lmr_era    = np.zeros([nlat_ERA20C,nlon_ERA20C])
ce_lmr_era   = np.zeros([nlat_ERA20C,nlon_ERA20C])
r_lmr_gis    = np.zeros([nlat_GIS,nlon_GIS])
ce_lmr_gis   = np.zeros([nlat_GIS,nlon_GIS])
r_lmr_be     = np.zeros([nlat_BE,nlon_BE])
ce_lmr_be    = np.zeros([nlat_BE,nlon_BE])
r_lmr_cru    = np.zeros([nlat_CRU,nlon_CRU])
ce_lmr_cru   = np.zeros([nlat_CRU,nlon_CRU])
r_lmr_mlost  = np.zeros([nlat_MLOST,nlon_MLOST])
ce_lmr_mlost = np.zeros([nlat_MLOST,nlon_MLOST])
r_tcr_gis    = np.zeros([nlat_GIS,nlon_GIS])
ce_tcr_gis   = np.zeros([nlat_GIS,nlon_GIS])
r_era_gis    = np.zeros([nlat_GIS,nlon_GIS])
ce_era_gis   = np.zeros([nlat_GIS,nlon_GIS])
# prior
r_xbm_tcr    = np.zeros([nlat_TCR,nlon_TCR])
ce_xbm_tcr   = np.zeros([nlat_TCR,nlon_TCR])
r_xbm_era    = np.zeros([nlat_ERA20C,nlon_ERA20C])
ce_xbm_era   = np.zeros([nlat_ERA20C,nlon_ERA20C])
r_xbm_gis    = np.zeros([nlat_GIS,nlon_GIS])
ce_xbm_gis   = np.zeros([nlat_GIS,nlon_GIS])
r_xbm_be     = np.zeros([nlat_BE,nlon_BE])
ce_xbm_be    = np.zeros([nlat_BE,nlon_BE])
r_xbm_cru    = np.zeros([nlat_CRU,nlon_CRU])
ce_xbm_cru   = np.zeros([nlat_CRU,nlon_CRU])
r_xbm_mlost  = np.zeros([nlat_MLOST,nlon_MLOST])
ce_xbm_mlost = np.zeros([nlat_MLOST,nlon_MLOST])

ce_lmr_tcr_unbiased   = np.zeros([nlat_TCR,nlon_TCR])
ce_lmr_era_unbiased   = np.zeros([nlat_ERA20C,nlon_ERA20C])
ce_lmr_gis_unbiased   = np.zeros([nlat_GIS,nlon_GIS])
ce_lmr_be_unbiased    = np.zeros([nlat_BE,nlon_BE])
ce_lmr_cru_unbiased   = np.zeros([nlat_CRU,nlon_CRU])
ce_lmr_mlost_unbiased = np.zeros([nlat_MLOST,nlon_MLOST])
ce_tcr_gis_unbiased   = np.zeros([nlat_GIS,nlon_GIS])
ce_era_gis_unbiased   = np.zeros([nlat_GIS,nlon_GIS])


# LMR-TCR
ce_lmr_tcr = coefficient_efficiency(tcr_allyears,lmr_on_tcr_allyears,valid_frac)
ce_xbm_tcr = coefficient_efficiency(tcr_allyears,xbm_on_tcr_allyears,valid_frac)
for la in range(nlat_TCR):
    for lo in range(nlon_TCR):
        indok = np.isfinite(tcr_allyears[:,la,lo])
        nbok = np.sum(indok)
        nball = lmr_on_tcr_allyears[:,la,lo].shape[0]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_lmr_tcr[la,lo] = np.corrcoef(lmr_on_tcr_allyears[indok,la,lo],tcr_allyears[indok,la,lo])[0,1]
            bias_lmr_tcr[la,lo] = np.mean(lmr_on_tcr_err[indok,la,lo],axis=0)
            numer_unbiased = np.sum(np.power(lmr_on_tcr_err[indok,la,lo]-bias_lmr_tcr[la,lo],2),axis=0)
            denom = np.sum(np.power(tcr_allyears[indok,la,lo]-np.mean(tcr_allyears[indok,la,lo],axis=0),2),axis=0)
            ce_lmr_tcr_unbiased[la,lo] = 1. - (numer_unbiased/denom)
            #r_xbm_tcr[la,lo] = np.corrcoef(xbm_on_tcr_allyears[indok,la,lo],tcr_allyears[indok,la,lo])[0,1]
            bias_xbm_tcr[la,lo] = np.mean(xbm_on_tcr_err[indok,la,lo],axis=0)
        else:
            r_lmr_tcr[la,lo]  = np.nan
            bias_lmr_tcr[la,lo] = np.nan
            ce_lmr_tcr_unbiased[la,lo] = np.nan
            r_xbm_tcr[la,lo]  = np.nan
            bias_xbm_tcr[la,lo] = np.nan

# LMR-ERA
ce_lmr_era = coefficient_efficiency(era_allyears,lmr_on_era_allyears,valid_frac)
ce_xbm_era = coefficient_efficiency(era_allyears,xbm_on_era_allyears,valid_frac)
for la in range(nlat_ERA20C):
    for lo in range(nlon_ERA20C):
        indok = np.isfinite(era_allyears[:,la,lo])
        nbok = np.sum(indok)
        nball = lmr_on_era_allyears[:,la,lo].shape[0]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_lmr_era[la,lo] = np.corrcoef(lmr_on_era_allyears[indok,la,lo],era_allyears[indok,la,lo])[0,1]
            bias_lmr_era[la,lo] = np.mean(lmr_on_era_err[indok,la,lo],axis=0)
            numer_unbiased = np.sum(np.power(lmr_on_era_err[indok,la,lo]-bias_lmr_era[la,lo],2),axis=0)
            denom = np.sum(np.power(era_allyears[indok,la,lo]-np.mean(era_allyears[indok,la,lo],axis=0),2),axis=0)
            ce_lmr_era_unbiased[la,lo] = 1. - (numer_unbiased/denom)
            #r_xbm_era[la,lo] = np.corrcoef(xbm_on_era_allyears[indok,la,lo],era_allyears[indok,la,lo])[0,1]
            bias_xbm_era[la,lo] = np.mean(xbm_on_era_err[indok,la,lo],axis=0)
        else:
            r_lmr_era[la,lo]  = np.nan
            bias_lmr_era[la,lo] = np.nan
            ce_lmr_era_unbiased[la,lo] = np.nan
            r_xbm_era[la,lo]  = np.nan
            bias_xbm_era[la,lo] = np.nan

# LMR-GIS
ce_lmr_gis = coefficient_efficiency(gis_allyears,lmr_on_gis_allyears,valid_frac)
ce_xbm_gis = coefficient_efficiency(gis_allyears,xbm_on_gis_allyears,valid_frac)
for la in range(nlat_GIS):
    for lo in range(nlon_GIS):
        indok = np.isfinite(gis_allyears[:,la,lo])
        nbok = np.sum(indok)
        nball = lmr_on_gis_allyears[:,la,lo].shape[0]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_lmr_gis[la,lo] = np.corrcoef(lmr_on_gis_allyears[indok,la,lo],gis_allyears[indok,la,lo])[0,1]
            bias_lmr_gis[la,lo] = np.mean(lmr_on_gis_err[indok,la,lo],axis=0)
            numer_unbiased = np.sum(np.power(lmr_on_gis_err[indok,la,lo]-bias_lmr_gis[la,lo],2),axis=0)
            denom = np.sum(np.power(gis_allyears[indok,la,lo]-np.mean(gis_allyears[indok,la,lo],axis=0),2),axis=0)
            ce_lmr_gis_unbiased[la,lo] = 1. - (numer_unbiased/denom)
            #r_xbm_gis[la,lo] = np.corrcoef(xbm_on_gis_allyears[indok,la,lo],gis_allyears[indok,la,lo])[0,1]
            bias_xbm_gis[la,lo] = np.mean(xbm_on_gis_err[indok,la,lo],axis=0)
        else:
            r_lmr_gis[la,lo]  = np.nan
            bias_lmr_gis[la,lo] = np.nan
            ce_lmr_gis_unbiased[la,lo] = np.nan
            r_xbm_gis[la,lo]  = np.nan
            bias_xbm_gis[la,lo] = np.nan

# LMR-BE
ce_lmr_be = coefficient_efficiency(be_allyears,lmr_on_be_allyears,valid_frac)
ce_xbm_be = coefficient_efficiency(be_allyears,xbm_on_be_allyears,valid_frac)
for la in range(nlat_BE):
    for lo in range(nlon_BE):
        indok = np.isfinite(be_allyears[:,la,lo])
        nbok = np.sum(indok)
        nball = lmr_on_be_allyears[:,la,lo].shape[0]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_lmr_be[la,lo] = np.corrcoef(lmr_on_be_allyears[indok,la,lo],be_allyears[indok,la,lo])[0,1]
            bias_lmr_be[la,lo] = np.mean(lmr_on_be_err[indok,la,lo],axis=0)
            numer_unbiased = np.sum(np.power(lmr_on_be_err[indok,la,lo]-bias_lmr_be[la,lo],2),axis=0)
            denom = np.sum(np.power(be_allyears[indok,la,lo]-np.mean(be_allyears[indok,la,lo],axis=0),2),axis=0)
            ce_lmr_be_unbiased[la,lo] = 1. - (numer_unbiased/denom)
            #r_xbm_be[la,lo] = np.corrcoef(xbm_on_be_allyears[indok,la,lo],be_allyears[indok,la,lo])[0,1]
            bias_xbm_be[la,lo] = np.mean(xbm_on_be_err[indok,la,lo],axis=0)
        else:
            r_lmr_be[la,lo]  = np.nan
            bias_lmr_be[la,lo] = np.nan
            ce_lmr_be_unbiased[la,lo] = np.nan
            r_xbm_be[la,lo]  = np.nan
            bias_xbm_be[la,lo] = np.nan

# LMR-CRU
ce_lmr_cru = coefficient_efficiency(cru_allyears,lmr_on_cru_allyears,valid_frac)
ce_xbm_cru = coefficient_efficiency(cru_allyears,xbm_on_cru_allyears,valid_frac)
for la in range(nlat_CRU):
    for lo in range(nlon_CRU):
        indok = np.isfinite(cru_allyears[:,la,lo])
        nbok = np.sum(indok)
        nball = lmr_on_cru_allyears[:,la,lo].shape[0]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_lmr_cru[la,lo] = np.corrcoef(lmr_on_cru_allyears[indok,la,lo],cru_allyears[indok,la,lo])[0,1]
            bias_lmr_cru[la,lo] = np.mean(lmr_on_cru_err[indok,la,lo],axis=0)
            numer_unbiased = np.sum(np.power(lmr_on_cru_err[indok,la,lo]-bias_lmr_cru[la,lo],2),axis=0)
            denom = np.sum(np.power(cru_allyears[indok,la,lo]-np.mean(cru_allyears[indok,la,lo],axis=0),2),axis=0)
            ce_lmr_cru_unbiased[la,lo] = 1. - (numer_unbiased/denom)
            #r_xbm_cru[la,lo] = np.corrcoef(xbm_on_cru_allyears[indok,la,lo],cru_allyears[indok,la,lo])[0,1]
            bias_xbm_cru[la,lo] = np.mean(xbm_on_cru_err[indok,la,lo],axis=0)
        else:
            r_lmr_cru[la,lo]  = np.nan
            bias_lmr_cru[la,lo] = np.nan
            ce_lmr_cru_unbiased[la,lo] = np.nan
            r_xbm_cru[la,lo]  = np.nan
            bias_xbm_cru[la,lo] = np.nan

# LMR-MLOST
ce_lmr_mlost = coefficient_efficiency(mlost_allyears,lmr_on_mlost_allyears,valid_frac)
ce_xbm_mlost = coefficient_efficiency(mlost_allyears,xbm_on_mlost_allyears,valid_frac)
for la in range(nlat_MLOST):
    for lo in range(nlon_MLOST):
        indok = np.isfinite(mlost_allyears[:,la,lo])
        nbok = np.sum(indok)
        nball = lmr_on_mlost_allyears[:,la,lo].shape[0]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_lmr_mlost[la,lo] = np.corrcoef(lmr_on_mlost_allyears[indok,la,lo],mlost_allyears[indok,la,lo])[0,1]
            bias_lmr_mlost[la,lo] = np.mean(lmr_on_mlost_err[indok,la,lo],axis=0)
            numer_unbiased = np.sum(np.power(lmr_on_mlost_err[indok,la,lo]-bias_lmr_mlost[la,lo],2),axis=0)
            denom = np.sum(np.power(mlost_allyears[indok,la,lo]-np.mean(mlost_allyears[indok,la,lo],axis=0),2),axis=0)
            ce_lmr_mlost_unbiased[la,lo] = 1. - (numer_unbiased/denom)
            #r_xbmlost[la,lo] = np.corrcoef(xbm_on_mlost_allyears[indok,la,lo],mlost_allyears[indok,la,lo])[0,1]
            bias_xbm_mlost[la,lo] = np.mean(xbm_on_mlost_err[indok,la,lo],axis=0)
        else:
            r_lmr_mlost[la,lo]  = np.nan
            bias_lmr_mlost[la,lo] = np.nan
            ce_lmr_mlost_unbiased[la,lo] = np.nan
            r_xbm_mlost[la,lo]  = np.nan
            bias_xbm_mlost[la,lo] = np.nan

# TCR-GIS
ce_tcr_gis = coefficient_efficiency(gis_allyears,tcr_on_gis_allyears,valid_frac)
for la in range(nlat_GIS):
    for lo in range(nlon_GIS):
        indok1 = np.isfinite(gis_allyears[:,la,lo])
        indok2 = np.isfinite(tcr_on_gis_allyears[:,la,lo]) 
        indok = np.logical_and(indok1,indok2)
        nbok = np.sum(indok)
        nball = tcr_on_gis_allyears[:,la,lo].shape[0]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_tcr_gis[la,lo] = np.corrcoef(tcr_on_gis_allyears[indok,la,lo],gis_allyears[indok,la,lo])[0,1]
            bias_tcr_gis[la,lo] = np.mean(tcr_on_gis_err[indok,la,lo],axis=0)
            numer_unbiased = np.sum(np.power(tcr_on_gis_err[indok,la,lo]-bias_tcr_gis[la,lo],2),axis=0)
            denom = np.sum(np.power(gis_allyears[indok1,la,lo]-np.mean(gis_allyears[indok1,la,lo],axis=0),2),axis=0)
            ce_tcr_gis_unbiased[la,lo] = 1. - (numer_unbiased/denom)
        else:
            r_tcr_gis[la,lo]  = np.nan
            bias_tcr_gis[la,lo] = np.nan
            ce_tcr_gis_unbiased[la,lo] = np.nan

# ERA-GIS
ce_era_gis = coefficient_efficiency(gis_allyears,era_on_gis_allyears,valid_frac)
for la in range(nlat_GIS):
    for lo in range(nlon_GIS):
        indok1 = np.isfinite(gis_allyears[:,la,lo])
        indok2 = np.isfinite(era_on_gis_allyears[:,la,lo]) 
        indok = np.logical_and(indok1,indok2)
        nbok = np.sum(indok)
        nball = era_on_gis_allyears[:,la,lo].shape[0]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_era_gis[la,lo] = np.corrcoef(era_on_gis_allyears[indok,la,lo],gis_allyears[indok,la,lo])[0,1]
            bias_era_gis[la,lo] = np.mean(era_on_gis_err[indok,la,lo],axis=0)
            numer_unbiased = np.sum(np.power(era_on_gis_err[indok,la,lo]-bias_era_gis[la,lo],2),axis=0)
            denom = np.sum(np.power(gis_allyears[indok1,la,lo]-np.mean(gis_allyears[indok1,la,lo],axis=0),2),axis=0)
            ce_era_gis_unbiased[la,lo] = 1. - (numer_unbiased/denom)
        else:
            r_era_gis[la,lo]  = np.nan
            bias_era_gis[la,lo] = np.nan
            ce_era_gis_unbiased[la,lo] = np.nan

# -------
# LMR-TCR
# -------
# median
print('')
lat_TCR = np.squeeze(lat2d_TCR[:,0])
indlat = np.where((lat_TCR[:] > -60.0) & (lat_TCR[:] < 60.0))
lmr_tcr_rmedian = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_tcr)) ))
print('lmr-tcr all-grid median r    : %s' % str(lmr_tcr_rmedian))
lmr_tcr_rmedian60 = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_tcr[indlat,:])) ))
print('lmr-tcr 60S-60N median r     : %s' % str(lmr_tcr_rmedian60))
lmr_tcr_cemedian = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_tcr)) ))
print('lmr-tcr all-grid median ce   : %s' % str(lmr_tcr_cemedian))
lmr_tcr_cemedian60 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_tcr[indlat,:])) ))
print('lmr-tcr 60S-60N median ce    : %s' % str(lmr_tcr_cemedian60))
lmr_tcr_biasmedian = str(float('%.2f' % np.nanmedian(bias_lmr_tcr) ))
# mean
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lmr_tcr,lat_TCR)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lmr_tcr,lat_TCR)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_lmr_tcr,lat_TCR)
lmr_tcr_rmean_global    = str(float('%.2f' %rmean_global[0]))
lmr_tcr_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
lmr_tcr_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
lmr_tcr_cemean_global   = str(float('%.2f' %cemean_global[0]))
lmr_tcr_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
lmr_tcr_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
lmr_tcr_biasmean_global = str(float('%.2f' %biasmean_global[0]))
lmr_tcr_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
lmr_tcr_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))
# prior
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_xbm_tcr,lat_TCR)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_xbm_tcr,lat_TCR)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_xbm_tcr,lat_TCR)
xbm_tcr_rmean_global    = str(float('%.2f' %rmean_global[0]))
xbm_tcr_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
xbm_tcr_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
xbm_tcr_cemean_global   = str(float('%.2f' %cemean_global[0]))
xbm_tcr_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
xbm_tcr_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
xbm_tcr_biasmean_global = str(float('%.2f' %biasmean_global[0]))
xbm_tcr_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
xbm_tcr_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))

# -------
# LMR-ERA
# -------
# median
print('')
lat_ERA = np.squeeze(lat2d_ERA20C[:,0])
indlat = np.where((lat_ERA20C[:] > -60.0) & (lat_ERA20C[:] < 60.0))
lmr_era_rmean = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_era)) ))
print('lmr-era all-grid median r    : %s' % str(lmr_era_rmean))
lmr_era_rmean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_era[indlat,:])) ))
print('lmr-era 60S-60N median r     : %s' % str(lmr_era_rmean60))
lmr_era_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_era)) ))
print('lmr-era all-grid median ce   : %s' % str(lmr_era_cemean))
lmr_era_cemean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_era[indlat,:])) ))
print('lmr-era 60S-60N median ce    : %s' % str(lmr_era_cemean60))
lmr_era_biasmean = str(float('%.2f' % np.nanmedian(bias_lmr_era) ))
# mean
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lmr_era,lat_ERA20C)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lmr_era,lat_ERA20C)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_lmr_era,lat_ERA20C)
lmr_era_rmean_global    = str(float('%.2f' %rmean_global[0]))
lmr_era_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
lmr_era_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
lmr_era_cemean_global   = str(float('%.2f' %cemean_global[0]))
lmr_era_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
lmr_era_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
lmr_era_biasmean_global = str(float('%.2f' %biasmean_global[0]))
lmr_era_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
lmr_era_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))
# prior
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_xbm_era,lat_ERA20C)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_xbm_era,lat_ERA20C)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_xbm_era,lat_ERA20C)
xbm_era_rmean_global    = str(float('%.2f' %rmean_global[0]))
xbm_era_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
xbm_era_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
xbm_era_cemean_global   = str(float('%.2f' %cemean_global[0]))
xbm_era_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
xbm_era_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
xbm_era_biasmean_global = str(float('%.2f' %biasmean_global[0]))
xbm_era_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
xbm_era_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))

# -------
# LMR_GIS
# -------
# median
print('')
lat_GIS = np.squeeze(lat2d_GIS[:,0])
indlat = np.where((lat_GIS[:] > -60.0) & (lat_GIS[:] < 60.0))
lmr_gis_rmean = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_gis)) ))
print('lmr-gis all-grid median r    : %s' % str(lmr_gis_rmean))
lmr_gis_rmean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_gis[indlat,:])) ))
print('lmr-gis 60S-60N median r     : %s' % str(lmr_gis_rmean60))
lmr_gis_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_gis)) ))
print('lmr-gis all-grid median ce   : %s' % str(lmr_gis_cemean))
lmr_gis_cemean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_gis[indlat,:])) ))
print('lmr-gis 60S-60N median ce    : %s' % str(lmr_gis_cemean60))
lmr_gis_biasmean = str(float('%.2f' % np.nanmedian(bias_lmr_gis[~np.isnan(bias_lmr_gis)]) ))
# mean
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lmr_gis,lat_GIS)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lmr_gis,lat_GIS)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_lmr_gis,lat_GIS)
lmr_gis_rmean_global    = str(float('%.2f' %rmean_global[0]))
lmr_gis_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
lmr_gis_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
lmr_gis_cemean_global   = str(float('%.2f' %cemean_global[0]))
lmr_gis_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
lmr_gis_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
lmr_gis_biasmean_global = str(float('%.2f' %biasmean_global[0]))
lmr_gis_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
lmr_gis_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))
# prior
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_xbm_gis,lat_GIS)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_xbm_gis,lat_GIS)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_xbm_gis,lat_GIS)
xbm_gis_rmean_global    = str(float('%.2f' %rmean_global[0]))
xbm_gis_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
xbm_gis_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
xbm_gis_cemean_global   = str(float('%.2f' %cemean_global[0]))
xbm_gis_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
xbm_gis_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
xbm_gis_biasmean_global = str(float('%.2f' %biasmean_global[0]))
xbm_gis_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
xbm_gis_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))

print('LMR-GIS globa-mean CE:' + str(lmr_gis_cemean_global))

# ------
# LMR_BE
# ------
# median
print('')
lat_BE = np.squeeze(lat2d_BE[:,0])
indlat = np.where((lat_BE[:] > -60.0) & (lat_BE[:] < 60.0))
lmr_be_rmean = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_be)) ))
print('lmr-be all-grid median r     : %s' % str(lmr_be_rmean))
lmr_be_rmean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_be[indlat,:])) ))
print('lmr-be 60S-60N median r      : %s' % str(lmr_be_rmean60))
lmr_be_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_be)) ))
print('lmr-be all-grid median ce    : %s' % str(lmr_be_cemean))
lmr_be_cemean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_be[indlat,:])) ))
print('lmr-be 60S-60N median ce     : %s' % str(lmr_be_cemean60))
lmr_be_biasmean = str(float('%.2f' % np.nanmedian(bias_lmr_be[~np.isnan(bias_lmr_be)]) ))
# mean
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lmr_be,lat_BE)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lmr_be,lat_BE)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_lmr_be,lat_BE)
lmr_be_rmean_global    = str(float('%.2f' %rmean_global[0]))
lmr_be_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
lmr_be_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
lmr_be_cemean_global   = str(float('%.2f' %cemean_global[0]))
lmr_be_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
lmr_be_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
lmr_be_biasmean_global = str(float('%.2f' %biasmean_global[0]))
lmr_be_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
lmr_be_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))
# prior
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_xbm_be,lat_BE)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_xbm_be,lat_BE)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_xbm_be,lat_BE)
xbm_be_rmean_global    = str(float('%.2f' %rmean_global[0]))
xbm_be_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
xbm_be_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
xbm_be_cemean_global   = str(float('%.2f' %cemean_global[0]))
xbm_be_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
xbm_be_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
xbm_be_biasmean_global = str(float('%.2f' %biasmean_global[0]))
xbm_be_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
xbm_be_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))

# -------
# LMR_CRU
# -------
# median
print('')
lat_CRU = np.squeeze(lat2d_CRU[:,0])
indlat = np.where((lat_CRU[:] > -60.0) & (lat_CRU[:] < 60.0))
lmr_cru_rmean = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_cru)) ))
print('lmr-cru all-grid median r    : %s' % str(lmr_cru_rmean))
lmr_cru_rmean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_cru[indlat,:])) ))
print('lmr-cru 60S-60N median r     : %s' % str(lmr_cru_rmean60))
lmr_cru_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_cru)) ))
print('lmr-cru all-grid median ce   : %s' % str(lmr_cru_cemean))
lmr_cru_cemean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_cru[indlat,:])) ))
print('lmr-cru 60S-60N median ce    : %s' % str(lmr_cru_cemean60))
lmr_cru_biasmean = str(float('%.2f' % np.nanmedian(bias_lmr_cru[~np.isnan(bias_lmr_cru)]) ))
# mean
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lmr_cru,lat_CRU)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lmr_cru,lat_CRU)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_lmr_cru,lat_CRU)
lmr_cru_rmean_global    = str(float('%.2f' %rmean_global[0]))
lmr_cru_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
lmr_cru_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
lmr_cru_cemean_global   = str(float('%.2f' %cemean_global[0]))
lmr_cru_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
lmr_cru_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
lmr_cru_biasmean_global = str(float('%.2f' %biasmean_global[0]))
lmr_cru_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
lmr_cru_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))
# prior
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_xbm_cru,lat_CRU)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_xbm_cru,lat_CRU)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_xbm_cru,lat_CRU)
xbm_cru_rmean_global    = str(float('%.2f' %rmean_global[0]))
xbm_cru_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
xbm_cru_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
xbm_cru_cemean_global   = str(float('%.2f' %cemean_global[0]))
xbm_cru_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
xbm_cru_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
xbm_cru_biasmean_global = str(float('%.2f' %biasmean_global[0]))
xbm_cru_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
xbm_cru_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))

# ---------
# LMR_MLOST
# ---------
# median
print('')
lat_MLOST = np.squeeze(lat2d_MLOST[:,0])
indlat = np.where((lat_MLOST[:] > -60.0) & (lat_MLOST[:] < 60.0))
lmr_mlost_rmean = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_mlost)) ))
print('lmr-mlost all-grid median r  : %s' % str(lmr_mlost_rmean))
lmr_mlost_rmean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(r_lmr_mlost[indlat,:])) ))
print('lmr-mlost 60S-60N median r   : %s' % str(lmr_mlost_rmean60))
lmr_mlost_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_mlost)) ))
print('lmr-mlost all-grid median ce : %s' % str(lmr_mlost_cemean))
lmr_mlost_cemean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_mlost[indlat,:])) ))
print('lmr-mlost 60S-60N median ce  : %s' % str(lmr_mlost_cemean60))
lmr_mlost_biasmean = str(float('%.2f' % np.nanmedian(bias_lmr_mlost[~np.isnan(bias_lmr_mlost)]) ))
# mean
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lmr_mlost,lat_MLOST)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lmr_mlost,lat_MLOST)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_lmr_mlost,lat_MLOST)
lmr_mlost_rmean_global    = str(float('%.2f' %rmean_global[0]))
lmr_mlost_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
lmr_mlost_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
lmr_mlost_cemean_global   = str(float('%.2f' %cemean_global[0]))
lmr_mlost_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
lmr_mlost_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
lmr_mlost_biasmean_global = str(float('%.2f' %biasmean_global[0]))
lmr_mlost_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
lmr_mlost_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))
# prior
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_xbm_mlost,lat_MLOST)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_xbm_mlost,lat_MLOST)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_xbm_mlost,lat_MLOST)
xbm_mlost_rmean_global    = str(float('%.2f' %rmean_global[0]))
xbm_mlost_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
xbm_mlost_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
xbm_mlost_cemean_global   = str(float('%.2f' %cemean_global[0]))
xbm_mlost_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
xbm_mlost_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
xbm_mlost_biasmean_global = str(float('%.2f' %biasmean_global[0]))
xbm_mlost_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
xbm_mlost_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))

# -------
# TCR_GIS
# -------
# median
print('')
lat_GIS = np.squeeze(lat2d_GIS[:,0])
indlat = np.where((lat_GIS[:] > -60.0) & (lat_GIS[:] < 60.0))
tcr_gis_rmean = str(float('%.2f' % np.nanmedian(np.nanmedian(r_tcr_gis)) ))
print('tcr-gis all-grid median r    : %s' % str(tcr_gis_rmean))
tcr_gis_rmean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(r_tcr_gis[indlat,:])) ))
print('tcr-gis 60S-60N median r     : %s' % str(tcr_gis_rmean60))
tcr_gis_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_tcr_gis)) ))
print('tcr-gis all-grid median ce   : %s' % str(tcr_gis_cemean))
tcr_gis_cemean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_tcr_gis[indlat,:])) ))
print('tcr-gis 60S-60N median ce    : %s' % str(tcr_gis_cemean60))
tcr_gis_biasmean = str(float('%.2f' % np.nanmedian(bias_tcr_gis[~np.isnan(bias_tcr_gis)]) ))
# mean
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_tcr_gis,lat_GIS)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_tcr_gis,lat_GIS)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_tcr_gis,lat_GIS)
tcr_gis_rmean_global    = str(float('%.2f' %rmean_global[0]))
tcr_gis_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
tcr_gis_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
tcr_gis_cemean_global   = str(float('%.2f' %cemean_global[0]))
tcr_gis_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
tcr_gis_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
tcr_gis_biasmean_global = str(float('%.2f' %biasmean_global[0]))
tcr_gis_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
tcr_gis_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))

# -------
# ERA_GIS
# -------
# median
print('')
lat_GIS = np.squeeze(lat2d_GIS[:,0])
indlat = np.where((lat_GIS[:] > -60.0) & (lat_GIS[:] < 60.0))
era_gis_rmean = str(float('%.2f' % np.nanmedian(np.nanmedian(r_era_gis)) ))
print('era-gis all-grid median r    : %s' % str(era_gis_rmean))
era_gis_rmean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(r_era_gis[indlat,:])) ))
print('era-gis 60S-60N median r     : %s' % str(era_gis_rmean60))
era_gis_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_era_gis)) ))
print('era-gis all-grid median ce   : %s' % str(era_gis_cemean))
era_gis_cemean60 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_era_gis[indlat,:])) ))
print('era-gis 60S-60N median ce    : %s' % str(era_gis_cemean60))
era_gis_biasmean = str(float('%.2f' % np.nanmedian(bias_era_gis[~np.isnan(bias_era_gis)]) ))

print('')

# mean
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_era_gis,lat_GIS)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_era_gis,lat_GIS)
[biasmean_global,biasmean_nh,biasmean_sh] = global_hemispheric_means(bias_era_gis,lat_GIS)
era_gis_rmean_global    = str(float('%.2f' %rmean_global[0]))
era_gis_rmean_nh        = str(float('%.2f' %rmean_nh[0]))
era_gis_rmean_sh        = str(float('%.2f' %rmean_sh[0]))
era_gis_cemean_global   = str(float('%.2f' %cemean_global[0]))
era_gis_cemean_nh       = str(float('%.2f' %cemean_nh[0]))
era_gis_cemean_sh       = str(float('%.2f' %cemean_sh[0]))
era_gis_biasmean_global = str(float('%.2f' %biasmean_global[0]))
era_gis_biasmean_nh     = str(float('%.2f' %biasmean_nh[0]))
era_gis_biasmean_sh     = str(float('%.2f' %biasmean_sh[0]))


# zonal mean verification statistics

# LMR-TCR
r_lmr_tcr_zm     = np.zeros([nlat_TCR])
ce_lmr_tcr_zm    = np.zeros([nlat_TCR])
bias_lmr_tcr_zm  = np.zeros([nlat_TCR])
lmr_tcr_err_zm = lmr_on_tcr_zm - tcr_zm
ce_lmr_tcr_zm = coefficient_efficiency(tcr_zm,lmr_on_tcr_zm,valid_frac)
# prior
r_xbm_tcr_zm     = np.zeros([nlat_TCR])
ce_xbm_tcr_zm    = np.zeros([nlat_TCR])
bias_xbm_tcr_zm  = np.zeros([nlat_TCR])
xbm_tcr_err_zm = xbm_on_tcr_zm - tcr_zm
ce_xbm_tcr_zm = coefficient_efficiency(tcr_zm,xbm_on_tcr_zm,valid_frac)
for la in range(nlat_TCR):
    indok = np.isfinite(tcr_zm[:,la])
    nbok = np.sum(indok)
    nball = len(cyears)
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        r_lmr_tcr_zm[la] = np.corrcoef(lmr_on_tcr_zm[indok,la],tcr_zm[indok,la])[0,1]
        bias_lmr_tcr_zm[la] = np.mean(lmr_tcr_err_zm[indok,la],axis=0)
        r_xbm_tcr_zm[la] = np.corrcoef(xbm_on_tcr_zm[indok,la],tcr_zm[indok,la])[0,1]
        bias_xbm_tcr_zm[la] = np.mean(xbm_tcr_err_zm[indok,la],axis=0)
    else:
        r_lmr_tcr_zm[la]  = np.nan
        bias_lmr_tcr_zm[la] = np.nan
        r_xbm_tcr_zm[la]  = np.nan
        bias_xbm_tcr_zm[la] = np.nan

# LMR-ERA
r_lmr_era_zm     = np.zeros([nlat_ERA20C])
ce_lmr_era_zm    = np.zeros([nlat_ERA20C])
bias_lmr_era_zm  = np.zeros([nlat_ERA20C])
lmr_era_err_zm = lmr_on_era_zm - era_zm
ce_lmr_era_zm = coefficient_efficiency(era_zm,lmr_on_era_zm,valid_frac)
# prior
r_xbm_era_zm     = np.zeros([nlat_ERA20C])
ce_xbm_era_zm    = np.zeros([nlat_ERA20C])
bias_xbm_era_zm  = np.zeros([nlat_ERA20C])
xbm_era_err_zm = xbm_on_era_zm - era_zm
ce_xbm_era_zm = coefficient_efficiency(era_zm,xbm_on_era_zm,valid_frac)
for la in range(nlat_ERA20C):
    indok = np.isfinite(era_zm[:,la])
    nbok = np.sum(indok)
    nball = len(cyears)
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        r_lmr_era_zm[la] = np.corrcoef(lmr_on_era_zm[indok,la],era_zm[indok,la])[0,1]
        bias_lmr_era_zm[la] = np.mean(lmr_era_err_zm[indok,la],axis=0)
        r_xbm_era_zm[la] = np.corrcoef(xbm_on_era_zm[indok,la],era_zm[indok,la])[0,1]
        bias_xbm_era_zm[la] = np.mean(xbm_era_err_zm[indok,la],axis=0)
    else:
        r_lmr_era_zm[la]  = np.nan
        bias_lmr_era_zm[la] = np.nan
        r_xbm_era_zm[la]  = np.nan
        bias_xbm_era_zm[la] = np.nan

# LMR-GIS
r_lmr_gis_zm   = np.zeros([nlat_GIS])
ce_lmr_gis_zm  = np.zeros([nlat_GIS])
bias_lmr_gis_zm  = np.zeros([nlat_GIS])
lmr_gis_err_zm = lmr_on_gis_zm - gis_zm
ce_lmr_gis_zm = coefficient_efficiency(gis_zm,lmr_on_gis_zm,valid_frac)
# prior
r_xbm_gis_zm   = np.zeros([nlat_GIS])
ce_xbm_gis_zm  = np.zeros([nlat_GIS])
bias_xbm_gis_zm  = np.zeros([nlat_GIS])
xbm_gis_err_zm = xbm_on_gis_zm - gis_zm
ce_xbm_gis_zm = coefficient_efficiency(gis_zm,xbm_on_gis_zm,valid_frac)
for la in range(nlat_GIS):
    indok = np.isfinite(gis_zm[:,la])
    nbok = np.sum(indok)
    nball = len(cyears)
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        r_lmr_gis_zm[la] = np.corrcoef(lmr_on_gis_zm[indok,la],gis_zm[indok,la])[0,1]
        bias_lmr_gis_zm[la] = np.mean(lmr_gis_err_zm[indok,la],axis=0)
        r_xbm_gis_zm[la] = np.corrcoef(xbm_on_gis_zm[indok,la],gis_zm[indok,la])[0,1]
        bias_xbm_gis_zm[la] = np.mean(xbm_gis_err_zm[indok,la],axis=0)
    else:
        r_lmr_gis_zm[la]  = np.nan
        bias_lmr_gis_zm[la] = np.nan
        r_xbm_gis_zm[la]  = np.nan
        bias_xbm_gis_zm[la] = np.nan

# LMR-BE
r_lmr_be_zm   = np.zeros([nlat_BE])
ce_lmr_be_zm  = np.zeros([nlat_BE])
bias_lmr_be_zm  = np.zeros([nlat_BE])
lmr_be_err_zm = lmr_on_be_zm - be_zm
ce_lmr_be_zm = coefficient_efficiency(be_zm,lmr_on_be_zm,valid_frac)
# prior
r_xbm_be_zm   = np.zeros([nlat_BE])
ce_xbm_be_zm  = np.zeros([nlat_BE])
bias_xbm_be_zm  = np.zeros([nlat_BE])
xbm_be_err_zm = xbm_on_be_zm - be_zm
ce_xbm_be_zm = coefficient_efficiency(be_zm,xbm_on_be_zm,valid_frac)
for la in range(nlat_BE):
    indok = np.isfinite(be_zm[:,la])
    nbok = np.sum(indok)
    nball = len(cyears)
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        r_lmr_be_zm[la] = np.corrcoef(lmr_on_be_zm[indok,la],be_zm[indok,la])[0,1]
        bias_lmr_be_zm[la] = np.mean(lmr_be_err_zm[indok,la],axis=0)
        r_xbm_be_zm[la] = np.corrcoef(xbm_on_be_zm[indok,la],be_zm[indok,la])[0,1]
        bias_xbm_be_zm[la] = np.mean(xbm_be_err_zm[indok,la],axis=0)
    else:
        r_lmr_be_zm[la]  = np.nan
        bias_lmr_be_zm[la] = np.nan
        r_xbm_be_zm[la]  = np.nan
        bias_xbm_be_zm[la] = np.nan

# LMR-CRU
r_lmr_cru_zm   = np.zeros([nlat_CRU])
ce_lmr_cru_zm  = np.zeros([nlat_CRU])
bias_lmr_cru_zm  = np.zeros([nlat_CRU])
lmr_cru_err_zm = lmr_on_cru_zm - cru_zm
ce_lmr_cru_zm = coefficient_efficiency(cru_zm,lmr_on_cru_zm,valid_frac)
# prior
r_xbm_cru_zm   = np.zeros([nlat_CRU])
ce_xbm_cru_zm  = np.zeros([nlat_CRU])
bias_xbm_cru_zm  = np.zeros([nlat_CRU])
xbm_cru_err_zm = xbm_on_cru_zm - cru_zm
ce_xbm_cru_zm = coefficient_efficiency(cru_zm,xbm_on_cru_zm,valid_frac)
for la in range(nlat_CRU):
    indok = np.isfinite(cru_zm[:,la])
    nbok = np.sum(indok)
    nball = len(cyears)
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        r_lmr_cru_zm[la] = np.corrcoef(lmr_on_cru_zm[indok,la],cru_zm[indok,la])[0,1]
        bias_lmr_cru_zm[la] = np.mean(lmr_cru_err_zm[indok,la],axis=0)
        r_xbm_cru_zm[la] = np.corrcoef(xbm_on_cru_zm[indok,la],cru_zm[indok,la])[0,1]
        bias_xbm_cru_zm[la] = np.mean(xbm_cru_err_zm[indok,la],axis=0)
    else:
        r_lmr_cru_zm[la]  = np.nan
        bias_lmr_cru_zm[la] = np.nan
        r_xbm_cru_zm[la]  = np.nan
        bias_xbm_cru_zm[la] = np.nan

# LMR-MLOST
r_lmr_mlost_zm   = np.zeros([nlat_MLOST])
ce_lmr_mlost_zm  = np.zeros([nlat_MLOST])
bias_lmr_mlost_zm  = np.zeros([nlat_MLOST])
lmr_mlost_err_zm = lmr_on_mlost_zm - mlost_zm
ce_lmr_mlost_zm = coefficient_efficiency(mlost_zm,lmr_on_mlost_zm,valid_frac)
# prior
r_xbm_mlost_zm   = np.zeros([nlat_MLOST])
ce_xbm_mlost_zm  = np.zeros([nlat_MLOST])
bias_xbm_mlost_zm  = np.zeros([nlat_MLOST])
xbm_mlost_err_zm = xbm_on_mlost_zm - mlost_zm
ce_xbm_mlost_zm = coefficient_efficiency(mlost_zm,xbm_on_mlost_zm,valid_frac)
for la in range(nlat_MLOST):
    indok = np.isfinite(mlost_zm[:,la])
    nbok = np.sum(indok)
    nball = len(cyears)
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        r_lmr_mlost_zm[la] = np.corrcoef(lmr_on_mlost_zm[indok,la],mlost_zm[indok,la])[0,1]
        bias_lmr_mlost_zm[la] = np.mean(lmr_mlost_err_zm[indok,la],axis=0)
        r_xbm_mlost_zm[la] = np.corrcoef(xbm_on_mlost_zm[indok,la],mlost_zm[indok,la])[0,1]
        bias_xbm_mlost_zm[la] = np.mean(xbm_mlost_err_zm[indok,la],axis=0)
    else:
        r_lmr_mlost_zm[la]  = np.nan
        bias_lmr_mlost_zm[la] = np.nan
        r_xbm_mlost_zm[la]  = np.nan
        bias_xbm_mlost_zm[la] = np.nan

#
# plot zonal mean statistics
#
major_ticks = np.arange(-90, 91, 30)
fig = plt.figure()
# Bias
#ax = fig.add_subplot(1,3,1)    
#tcrleg,   = ax.plot(bias_lmr_tcr_zm,  lat_TCR,  'black',         linestyle='-',lw=2,label='TCR')
#gisleg,   = ax.plot(bias_lmr_gis_zm,  lat_GIS,  'red',           linestyle='-',lw=2,label='GIS')
#beleg,    = ax.plot(bias_lmr_be_zm,   lat_BE,   'steelblue',     linestyle='-',lw=2,label='BE')
#cruleg,   = ax.plot(bias_lmr_cru_zm,  lat_CRU,  'mediumseagreen',linestyle='-',lw=2,label='CRU')
#mlostleg, = ax.plot(bias_lmr_mlost_zm,lat_MLOST,'darkorange',    linestyle='-',lw=2,label='MLOST')
#ax.plot([0,0],[-90,90],'k:')
#ax.set_yticks(major_ticks)                                                       
#plt.ylim([-90,90])
#plt.xlim([-1,1])
#plt.ylabel('latitude',fontweight='bold')
#plt.xlabel('bias',fontweight='bold')
#ax.legend(handles=[tcrleg,gisleg,beleg,cruleg,mlostleg],handlelength=1.5,ncol=1,fontsize=11,loc='upper left',frameon=False)

# Correlation
ax = fig.add_subplot(1,2,1)    
tcrleg,   = ax.plot(r_lmr_tcr_zm,  lat_TCR,   'black',         linestyle='-',lw=2,label='20CR-V2')
eraleg,   = ax.plot(r_lmr_era_zm,  lat_ERA20C,'blue',          linestyle='-',lw=2,label='ERA-20C')
gisleg,   = ax.plot(r_lmr_gis_zm,  lat_GIS,   'red',           linestyle='-',lw=2,label='GISTEMP')
beleg,    = ax.plot(r_lmr_be_zm,   lat_BE,    'steelblue',     linestyle='-',lw=2,label='BE')
cruleg,   = ax.plot(r_lmr_cru_zm,  lat_CRU,   'mediumseagreen',linestyle='-',lw=2,label='HadCRUT4')
mlostleg, = ax.plot(r_lmr_mlost_zm,lat_MLOST, 'darkorange',    linestyle='-',lw=2,label='MLOST')
ax.plot([0,0],[-90,90],'k:')
ax.set_yticks(major_ticks)                                                       
#ax.set_yticks([]) 
plt.ylim([-90,90])
plt.xlim([-1,1])
plt.ylabel('latitude',fontweight='bold')
plt.xlabel('Correlation',fontweight='bold')
ax.legend(handles=[tcrleg,eraleg,gisleg,beleg,cruleg,mlostleg],handlelength=1.5,ncol=1,fontsize=11,loc='upper left',frameon=False)
# CE
ax = fig.add_subplot(1,2,2)    
ax.plot(ce_lmr_tcr_zm,  lat_TCR,   'black',         linestyle='-',lw=2)
ax.plot(ce_lmr_era_zm,  lat_ERA20C,'blue',          linestyle='-',lw=2)
ax.plot(ce_lmr_gis_zm,  lat_GIS,   'red',           linestyle='-',lw=2)
ax.plot(ce_lmr_be_zm,   lat_BE,    'steelblue',     linestyle='-',lw=2)
ax.plot(ce_lmr_cru_zm,  lat_CRU,   'mediumseagreen',linestyle='-',lw=2)
ax.plot(ce_lmr_mlost_zm,lat_MLOST, 'darkorange',    linestyle='-',lw=2)
ax.plot([0,0],[-90,90],'k:')
ax.set_yticks([])                                                       
plt.ylim([-90,90])
plt.xlim([-1.0,1.0])
plt.xlabel('Coefficient of efficiency',fontweight='bold')
plt.suptitle('LMR zonal-mean verification - surface air temperature',fontweight='bold')
fig.tight_layout(pad = 2.0)
if fsave:
    print('saving to .png')
    plt.savefig(nexp+'_verify_grid_r_ce_zonal_mean_'+str(trange[0])+'-'+str(trange[1])+'.png')
    plt.savefig(nexp+'_verify_grid_r_ce_zonal_mean_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')
    plt.close()

# ---------
# bias maps
# ---------

cbarfmt = '%4.1f'
nticks = 4 # number of ticks on the colorbar
bmin = -1.0
bmax = 1.0
if iplot:
    fig = plt.figure()
    ax = fig.add_subplot(3,2,1)    
    LMR_plotter(bias_lmr_tcr,lat2d_TCR,lon2d_TCR,'bwr',nlevs,vmin=bmin,vmax=bmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - 20CR-V2 T bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_tcr_biasmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(3,2,2)    
    LMR_plotter(bias_lmr_era,lat2d_ERA20C,lon2d_ERA20C,'bwr',nlevs,vmin=bmin,vmax=bmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - ERA-20C T bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_era_biasmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(3,2,3)    
    LMR_plotter(bias_lmr_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=bmin,vmax=bmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - GISTEMP T bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_gis_biasmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(3,2,4)    
    LMR_plotter(bias_lmr_be,lat2d_BE,lon2d_BE,'bwr',nlevs,vmin=bmin,vmax=bmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - BE T bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_be_biasmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(3,2,5)    
    LMR_plotter(bias_lmr_mlost,lat2d_MLOST,lon2d_MLOST,'bwr',nlevs,vmin=bmin,vmax=bmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - MLOST T bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_mlost_biasmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(3,2,6)    
    LMR_plotter(bias_lmr_cru,lat2d_CRU,lon2d_CRU,'bwr',nlevs,vmin=bmin,vmax=bmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - HadCRUT4 T bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_cru_biasmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_bias_'+str(trange[0])+'-'+str(trange[1])+'.png')


    # reference (TCR vs GIS & ERA vs GIS)
    fig = plt.figure()
    ax = fig.add_subplot(3,2,1)    
    LMR_plotter(bias_tcr_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=bmin,vmax=bmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('20CR-V2 - GISTEMP T bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(tcr_gis_biasmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(3,2,2)    
    LMR_plotter(bias_era_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=bmin,vmax=bmax,extend='both',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('ERA-20C - GISTEMP T bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(era_gis_biasmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_bias_reference_'+str(trange[0])+'-'+str(trange[1])+'.png')



# -------------
# r and ce maps
# -------------

cbarfmt = '%4.1f'
nticks = 4 # number of ticks on the colorbar
if iplot:

    # LMR reconstruction
    fig = plt.figure()
    ax = fig.add_subplot(5,2,1)    
    LMR_plotter(r_lmr_tcr,lat2d_TCR,lon2d_TCR,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - 20CR-V2 T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_tcr_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,2)    
    LMR_plotter(ce_lmr_tcr,lat2d_TCR,lon2d_TCR,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - 20CR-V2 T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_tcr_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,3)    
    LMR_plotter(r_lmr_era,lat2d_ERA20C,lon2d_ERA20C,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - ERA-20C T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_era_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,4)    
    LMR_plotter(ce_lmr_era,lat2d_ERA20C,lon2d_ERA20C,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - ERA-20C T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_era_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,5)    
    LMR_plotter(r_lmr_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - GISTEMP T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_gis_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,6)    
    LMR_plotter(ce_lmr_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - GISTEMP T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_gis_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,7)    
    LMR_plotter(r_lmr_be,lat2d_BE,lon2d_BE,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - BE T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_be_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,8)
    LMR_plotter(ce_lmr_be,lat2d_BE,lon2d_BE,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - BE T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_be_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,9)
    LMR_plotter(r_lmr_mlost,lat2d_MLOST,lon2d_MLOST,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - MLOST T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_mlost_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,10)    
    LMR_plotter(ce_lmr_mlost,lat2d_MLOST,lon2d_MLOST,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - MLOST T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_mlost_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_grid_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')

    # =================================================================================================================================
    # For paper 1 :

    # in paper :
    fig = plt.figure()
    ax = fig.add_subplot(4,2,1)    
    LMR_plotter(r_lmr_tcr,lat2d_TCR,lon2d_TCR,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - 20CR-V2 T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_tcr_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,2)    
    LMR_plotter(ce_lmr_tcr,lat2d_TCR,lon2d_TCR,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - 20CR-V2 T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_tcr_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,3)    
    LMR_plotter(r_lmr_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - GISTEMP T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_gis_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,4)    
    LMR_plotter(ce_lmr_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - GISTEMP T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_gis_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_r_ce_'+str(trange[0])+'-'+str(trange[1])+'_paper.png')
        plt.savefig(nexp+'_verify_grid_r_ce_'+str(trange[0])+'-'+str(trange[1])+'_paper.pdf',bbox_inches='tight', dpi=300, format='pdf')


    # in supp. material:
    fig = plt.figure()

    ax = fig.add_subplot(4,2,1)    
    LMR_plotter(r_lmr_era,lat2d_ERA20C,lon2d_ERA20C,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - ERA-20C T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_era_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,2)    
    LMR_plotter(ce_lmr_era,lat2d_ERA20C,lon2d_ERA20C,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - ERA-20C T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_era_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,3)    
    LMR_plotter(r_lmr_be,lat2d_BE,lon2d_BE,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - BE T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_be_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,4)
    LMR_plotter(ce_lmr_be,lat2d_BE,lon2d_BE,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - BE T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_be_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,5)
    LMR_plotter(r_lmr_mlost,lat2d_MLOST,lon2d_MLOST,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - MLOST T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_mlost_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,6)    
    LMR_plotter(ce_lmr_mlost,lat2d_MLOST,lon2d_MLOST,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - MLOST T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_mlost_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,7)
    LMR_plotter(r_lmr_cru,lat2d_CRU,lon2d_CRU,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - HadCRUT4 T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_cru_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,8)    
    LMR_plotter(ce_lmr_cru,lat2d_CRU,lon2d_CRU,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - HadCRUT4 T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lmr_cru_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_r_ce_'+str(trange[0])+'-'+str(trange[1])+'_supp.png')
        plt.savefig(nexp+'_verify_grid_r_ce_'+str(trange[0])+'-'+str(trange[1])+'_supp.pdf',bbox_inches='tight', dpi=300, format='pdf')

    # =================================================================================================================================


    # TCR vs GIS for reference
    fig = plt.figure()
    ax = fig.add_subplot(4,2,1)    
    LMR_plotter(r_tcr_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('20CR-V2 - GISTEMP T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(tcr_gis_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,2)    
    LMR_plotter(ce_tcr_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('20CR-V2 - GISTEMP T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(tcr_gis_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    # ERA vs GIS for reference
    ax = fig.add_subplot(4,2,3)    
    LMR_plotter(r_era_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('ERA-20C - GISTEMP T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(era_gis_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,4)    
    LMR_plotter(ce_era_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('ERA-20C - GISTEMP T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(era_gis_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])
  
    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_r_ce_reference_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_grid_r_ce_reference_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')
        plt.close()


    # Prior 
    fig = plt.figure()
    ax = fig.add_subplot(5,2,1)    
    LMR_plotter(r_xbm_tcr,lat2d_TCR,lon2d_TCR,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('Prior - 20CR-V2 T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_tcr_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,2)    
    LMR_plotter(ce_xbm_tcr,lat2d_TCR,lon2d_TCR,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('Prior - 20CR-V2 T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_tcr_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,3)    
    LMR_plotter(r_xbm_era,lat2d_ERA20C,lon2d_ERA20C,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('Prior - ERA-20C T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_era_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,4)    
    LMR_plotter(ce_xbm_era,lat2d_ERA20C,lon2d_ERA20C,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('Prior - ERA-20C T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_era_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,5)    
    LMR_plotter(r_xbm_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('Prior - GISTEMP T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_gis_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,6)    
    LMR_plotter(ce_xbm_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('Prior - GISTEMP T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_gis_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,7)    
    LMR_plotter(r_xbm_be,lat2d_BE,lon2d_BE,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('Prior - BE T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_be_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,8)
    LMR_plotter(ce_xbm_be,lat2d_BE,lon2d_BE,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('Prior - BE T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_be_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,9)
    LMR_plotter(r_xbm_mlost,lat2d_MLOST,lon2d_MLOST,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('Prior - MLOST T r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_mlost_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(5,2,10)    
    LMR_plotter(ce_xbm_mlost,lat2d_MLOST,lon2d_MLOST,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('Prior - MLOST T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(xbm_mlost_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_r_ce_Prior_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_grid_r_ce_Prior_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')


    # Comparisons of CE (full) & CE (unbiased errors)

    lmr_tcr_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_tcr[:,:])) ))
    lmr_tcr_cemean2 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_tcr[:,:]-ce_lmr_tcr_unbiased[:,:])) ))

    lmr_gis_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_gis[:,:])) ))
    lmr_gis_cemean2 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_gis[:,:]-ce_lmr_gis_unbiased[:,:])) ))

    lmr_be_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_be[:,:])) ))
    lmr_be_cemean2 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_be[:,:]-ce_lmr_be_unbiased[:,:])) ))

    lmr_mlost_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_mlost[:,:])) ))
    lmr_mlost_cemean2 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_lmr_mlost[:,:]-ce_lmr_mlost_unbiased[:,:])) ))

    fig = plt.figure()
    ax = fig.add_subplot(4,2,1)    
    LMR_plotter(ce_lmr_tcr,lat2d_TCR,lon2d_TCR,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - 20CR-V2 T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_tcr_cemean))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,2)    
    LMR_plotter(ce_lmr_tcr-ce_lmr_tcr_unbiased,lat2d_TCR,lon2d_TCR,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - 20CR-V2 T CE bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_tcr_cemean2))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,3)    
    LMR_plotter(ce_lmr_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - GISTEMP T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_gis_cemean))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,4)    
    LMR_plotter(ce_lmr_gis-ce_lmr_gis_unbiased,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - GISTEMP T CE bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_gis_cemean2))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,5)    
    LMR_plotter(ce_lmr_be,lat2d_BE,lon2d_BE,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - BE T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_be_cemean))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,6)    
    LMR_plotter(ce_lmr_be-ce_lmr_be_unbiased,lat2d_BE,lon2d_BE,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - BE T CE bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_be_cemean2))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,7)    
    LMR_plotter(ce_lmr_mlost,lat2d_MLOST,lon2d_MLOST,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - MLOST T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_mlost_cemean))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,8)    
    LMR_plotter(ce_lmr_mlost-ce_lmr_mlost_unbiased,lat2d_MLOST,lon2d_MLOST,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('LMR - MLOST T CE bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lmr_mlost_cemean2))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_ce_vsBias_'+str(trange[0])+'-'+str(trange[1])+'.png')


    # Reference
    tcr_gis_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_tcr_gis[:,:])) ))
    tcr_gis_cemean2 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_tcr_gis[:,:]-ce_tcr_gis_unbiased[:,:])) ))

    era_gis_cemean = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_era_gis[:,:])) ))
    era_gis_cemean2 = str(float('%.2f' % np.nanmedian(np.nanmedian(ce_era_gis[:,:]-ce_era_gis_unbiased[:,:])) ))

    fig = plt.figure()
    ax = fig.add_subplot(4,2,1)    
    LMR_plotter(ce_tcr_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('20CR-V2 - GISTEMP T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(tcr_gis_cemean))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,2)    
    LMR_plotter(ce_tcr_gis-ce_tcr_gis_unbiased,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('20CR-V2 - GISTEMP T CE bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(tcr_gis_cemean2))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,3)    
    LMR_plotter(ce_era_gis,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('ERA-20C - GISTEMP T CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(era_gis_cemean))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,4)    
    LMR_plotter(ce_era_gis-ce_era_gis_unbiased,lat2d_GIS,lon2d_GIS,'bwr',nlevs,vmin=-1,vmax=1,extend='min',backg='lightgrey',cbarfmt=cbarfmt,nticks=nticks)
    plt.title('ERA-20C - GISTEMP T CE bias '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(era_gis_cemean2))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    fig.tight_layout()
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_ce_vsBias_reference_'+str(trange[0])+'-'+str(trange[1])+'.png')

if iplot:
    plt.show(block=True)

# ensemble calibration

# Need to project TCR on LMR grid
nyears,nlat,nlon = np.shape(xam_var)
nyears_tcr,nlat_tcr,nlon_tcr = tcr_allyears.shape
tcr_on_lmr = np.zeros(shape=[len(cyears),nlat,nlon])
lmr_err_vs_tcr = np.zeros(shape=[len(cyears),nlat,nlon])
be_on_lmr = np.zeros(shape=[len(cyears),nlat,nlon])
lmr_err_vs_be = np.zeros(shape=[len(cyears),nlat,nlon])

k = -1
for yr in cyears:
    k = k + 1
    tcr_on_lmr[k,:,:] = regrid(specob_tcr,specob_lmr,tcr_allyears[k,:,:], ntrunc=None, smooth=None)
    LMR_smatch, LMR_ematch = find_date_indices(LMR_time,yr,yr+1)
    pdata_lmr = np.mean(xam[LMR_smatch:LMR_ematch,:,:],0)
    lmr_err_vs_tcr[k,:,:] =  pdata_lmr - tcr_on_lmr[k,:,:]


print(np.shape(lmr_err_vs_tcr))
print(np.shape(xam_var))
LMR_smatch, LMR_ematch = find_date_indices(LMR_time,trange[0],trange[1])
svar = xam_var[LMR_smatch:LMR_ematch,:,:]
calib_tcr = lmr_err_vs_tcr.var(0)/svar.mean(0)
print(calib_tcr[0:-1,:].mean())


# create the plot
mapcolor_calib = truncate_colormap(plt.cm.YlOrBr,0.0,0.8)
fig = plt.figure()
cb = LMR_plotter(calib_tcr,lat2,lon2,mapcolor_calib,11,0,10,extend='max',nticks=10)
#cb.set_ticks(range(11))
# overlay stations!
plt.title('Ratio of ensemble-mean error variance to mean ensemble variance \n Surface air temperature')
if fsave:
    print('saving to .png')
    plt.savefig(nexp+'_verify_grid_ensemble_calibration_'+str(trange[0])+'-'+str(trange[1])+'.png')


# in loop over lat,lon, add a call to the rank histogram function; need to move up the function def

# NEW look at trends over specified time periods as a function of latitude

# zonal means of the original LMR data

# new---time-mean spatial anomaly correlation
lmr_tcr_mac = str(float('%.2f' %np.nanmean(lmr_tcr_csave)))
lmr_era_mac = str(float('%.2f' %np.nanmean(lmr_era_csave)))
lmr_gis_mac = str(float('%.2f' %np.nanmean(lmr_gis_csave)))
lmr_be_mac = str(float('%.2f' %np.nanmean(lmr_be_csave)))
lmr_cru_mac = str(float('%.2f' %np.nanmean(lmr_cru_csave)))
lmr_mlost_mac = str(float('%.2f' %np.nanmean(lmr_mlost_csave)))

# new---store data in pickle file for analysis
#
# need to store anomaly correlation data too
#
if stat_save:
    grid_verification_stats = {}
    stat_vars = ['trange','ref_period',
                 'lmr_tcr_rmean_global','lmr_tcr_rmean_nh','lmr_tcr_rmean_sh','lmr_tcr_cemean_global','lmr_tcr_cemean_nh','lmr_tcr_cemean_sh',
                 'lmr_era_rmean_global','lmr_era_rmean_nh','lmr_era_rmean_sh','lmr_era_cemean_global','lmr_era_cemean_nh','lmr_era_cemean_sh',
                 'lmr_gis_rmean_global','lmr_gis_rmean_nh','lmr_gis_rmean_sh','lmr_gis_cemean_global','lmr_gis_cemean_nh','lmr_gis_cemean_sh',
                 'lmr_be_rmean_global','lmr_be_rmean_nh','lmr_be_rmean_sh','lmr_be_cemean_global','lmr_be_cemean_nh','lmr_be_cemean_sh',
                 'lmr_cru_rmean_global','lmr_cru_rmean_nh','lmr_cru_rmean_sh','lmr_cru_cemean_global','lmr_cru_cemean_nh','lmr_cru_cemean_sh',
                 'lmr_mlost_rmean_global','lmr_mlost_rmean_nh','lmr_mlost_rmean_sh','lmr_mlost_cemean_global','lmr_mlost_cemean_nh','lmr_mlost_cemean_sh',
                 'xbm_tcr_rmean_global','xbm_tcr_rmean_nh','xbm_tcr_rmean_sh','xbm_tcr_cemean_global','xbm_tcr_cemean_nh','xbm_tcr_cemean_sh',
                 'xbm_era_rmean_global','xbm_era_rmean_nh','xbm_era_rmean_sh','xbm_era_cemean_global','xbm_era_cemean_nh','xbm_era_cemean_sh',
                 'xbm_gis_rmean_global','xbm_gis_rmean_nh','xbm_gis_rmean_sh','xbm_gis_cemean_global','xbm_gis_cemean_nh','xbm_gis_cemean_sh',
                 'xbm_be_rmean_global','xbm_be_rmean_nh','xbm_be_rmean_sh','xbm_be_cemean_global','xbm_be_cemean_nh','xbm_be_cemean_sh',
                 'xbm_cru_rmean_global','xbm_cru_rmean_nh','xbm_cru_rmean_sh','xbm_cru_cemean_global','xbm_cru_cemean_nh','xbm_cru_cemean_sh',
                 'xbm_mlost_rmean_global','xbm_mlost_rmean_nh','xbm_mlost_rmean_sh','xbm_mlost_cemean_global','xbm_mlost_cemean_nh','xbm_mlost_cemean_sh',
                 'lmr_tcr_mac','lmr_era_mac','lmr_gis_mac','lmr_be_mac','lmr_cru_mac','lmr_mlost_mac']

    stat_metadata = {'trange':"starting and ending year of verification time period",
                     'ref_period':"starting and ending year of reference time period (mean removed)",
                     'lmr_tcr_rmean_global':"LMR_TCR correlation global mean",
                     'lmr_tcr_rmean_nh':"LMR_TCR correlation Northern Hemisphere mean",
                     'lmr_tcr_rmean_sh':"LMR_TCR correlation Southern Hemisphere mean",
                     'lmr_tcr_cemean_global':"LMR_TCR coefficient of efficiency global mean",
                     'lmr_tcr_cemean_nh':"LMR_TCR coefficient of efficiency Northern Hemisphere mean",
                     'lmr_tcr_cemean_sh':"LMR_TCR coefficient of efficiency Southern Hemisphere mean",
                     'lmr_era_rmean_global':"LMR_ERA correlation global mean",
                     'lmr_era_rmean_nh':"LMR_ERA correlation Northern Hemisphere mean",
                     'lmr_era_rmean_sh':"LMR_ERA correlation Southern Hemisphere mean",
                     'lmr_era_cemean_global':"LMR_ERA coefficient of efficiency global mean",
                     'lmr_era_cemean_nh':"LMR_ERA coefficient of efficiency Northern Hemisphere mean",
                     'lmr_era_cemean_sh':"LMR_ERA coefficient of efficiency Southern Hemisphere mean",
                     'lmr_gis_rmean_global':"LMR_GIS correlation global mean",
                     'lmr_gis_rmean_nh':"LMR_GIS correlation Northern Hemisphere mean",
                     'lmr_gis_rmean_sh':"LMR_GIS correlation Southern Hemisphere mean",
                     'lmr_gis_cemean_global':"LMR_GIS coefficient of efficiency global mean",
                     'lmr_gis_cemean_nh':"LMR_GIS coefficient of efficiency Northern Hemisphere mean",
                     'lmr_gis_cemean_sh':"LMR_GIS coefficient of efficiency Southern Hemisphere mean",
                     'lmr_be_rmean_global':"LMR_BE correlation global mean",
                     'lmr_be_rmean_nh':"LMR_BE correlation Northern Hemisphere mean",
                     'lmr_be_rmean_sh':"LMR_BE correlation Southern Hemisphere mean",
                     'lmr_be_cemean_global':"LMR_BE coefficient of efficiency global mean",
                     'lmr_be_cemean_nh':"LMR_BE coefficient of efficiency Northern Hemisphere mean",
                     'lmr_be_cemean_sh':"LMR_BE coefficient of efficiency Southern Hemisphere mean",
                     'lmr_cru_rmean_global':"LMR_CRU correlation global mean",
                     'lmr_cru_rmean_nh':"LMR_CRU correlation Northern Hemisphere mean",
                     'lmr_cru_rmean_sh':"LMR_CRU correlation Southern Hemisphere mean",
                     'lmr_cru_cemean_global':"LMR_CRU coefficient of efficiency global mean",
                     'lmr_cru_cemean_nh':"LMR_CRU coefficient of efficiency Northern Hemisphere mean",
                     'lmr_cru_cemean_sh':"LMR_CRU coefficient of efficiency Southern Hemisphere mean",
                     'lmr_mlost_rmean_global':"LMR_MLOST correlation global mean",
                     'lmr_mlost_rmean_nh':"LMR_MLOST correlation Northern Hemisphere mean",
                     'lmr_mlost_rmean_sh':"LMR_MLOST correlation Southern Hemisphere mean",
                     'lmr_mlost_cemean_global':"LMR_MLOST coefficient of efficiency global mean",
                     'lmr_mlost_cemean_nh':"LMR_MLOST coefficient of efficiency Northern Hemisphere mean",
                     'lmr_mlost_cemean_sh':"LMR_MLOST coefficient of efficiency Southern Hemisphere mean",
                     'xbm_tcr_rmean_global':"XBM_TCR correlation global mean",
                     'xbm_tcr_rmean_nh':"XBM_TCR correlation Northern Hemisphere mean",
                     'xbm_tcr_rmean_sh':"XBM_TCR correlation Southern Hemisphere mean",
                     'xbm_tcr_cemean_global':"XBM_TCR coefficient of efficiency global mean",
                     'xbm_tcr_cemean_nh':"XBM_TCR coefficient of efficiency Northern Hemisphere mean",
                     'xbm_tcr_cemean_sh':"XBM_TCR coefficient of efficiency Southern Hemisphere mean",
                     'xbm_era_rmean_global':"XBM_ERA correlation global mean",
                     'xbm_era_rmean_nh':"XBM_ERA correlation Northern Hemisphere mean",
                     'xbm_era_rmean_sh':"XBM_ERA correlation Southern Hemisphere mean",
                     'xbm_era_cemean_global':"XBM_ERA coefficient of efficiency global mean",
                     'xbm_era_cemean_nh':"XBM_ERA coefficient of efficiency Northern Hemisphere mean",
                     'xbm_era_cemean_sh':"XBM_ERA coefficient of efficiency Southern Hemisphere mean",
                     'xbm_gis_rmean_global':"XBM_GIS correlation global mean",
                     'xbm_gis_rmean_nh':"XBM_GIS correlation Northern Hemisphere mean",
                     'xbm_gis_rmean_sh':"XBM_GIS correlation Southern Hemisphere mean",
                     'xbm_gis_cemean_global':"XBM_GIS coefficient of efficiency global mean",
                     'xbm_gis_cemean_nh':"XBM_GIS coefficient of efficiency Northern Hemisphere mean",
                     'xbm_gis_cemean_sh':"XBM_GIS coefficient of efficiency Southern Hemisphere mean",
                     'xbm_be_rmean_global':"XBM_BE correlation global mean",
                     'xbm_be_rmean_nh':"XBM_BE correlation Northern Hemisphere mean",
                     'xbm_be_rmean_sh':"XBM_BE correlation Southern Hemisphere mean",
                     'xbm_be_cemean_global':"XBM_BE coefficient of efficiency global mean",
                     'xbm_be_cemean_nh':"XBM_BE coefficient of efficiency Northern Hemisphere mean",
                     'xbm_be_cemean_sh':"XBM_BE coefficient of efficiency Southern Hemisphere mean",
                     'xbm_cru_rmean_global':"XBM_CRU correlation global mean",
                     'xbm_cru_rmean_nh':"XBM_CRU correlation Northern Hemisphere mean",
                     'xbm_cru_rmean_sh':"XBM_CRU correlation Southern Hemisphere mean",
                     'xbm_cru_cemean_global':"XBM_CRU coefficient of efficiency global mean",
                     'xbm_cru_cemean_nh':"XBM_CRU coefficient of efficiency Northern Hemisphere mean",
                     'xbm_cru_cemean_sh':"XBM_CRU coefficient of efficiency Southern Hemisphere mean",
                     'xbm_mlost_rmean_global':"XBM_MLOST correlation global mean",
                     'xbm_mlost_rmean_nh':"XBM_MLOST correlation Northern Hemisphere mean",
                     'xbm_mlost_rmean_sh':"XBM_MLOST correlation Southern Hemisphere mean",
                     'xbm_mlost_cemean_global':"XBM_MLOST coefficient of efficiency global mean",
                     'xbm_mlost_cemean_nh':"XBM_MLOST coefficient of efficiency Northern Hemisphere mean",
                     'xbm_mlost_cemean_sh':"XBM_MLOST coefficient of efficiency Southern Hemisphere mean",
                     'lmr_tcr_mac':"LMR_TCR time-mean spatial anomaly correlation",
                     'lmr_era_mac':"LMR_ERA time-mean spatial anomaly correlation",
                     'lmr_gis_mac':"LMR_GIS time-mean spatial anomaly correlation",
                     'lmr_be_mac':"LMR_BE time-mean spatial anomaly correlation",
                     'lmr_cru_mac':"LMR_CRU time-mean spatial anomaly correlation",
                     'lmr_mlost_mac':"LMR_MLOST time-mean spatial anomaly correlation",
                     'stat_metadata':'metadata'
                     }

    for var in stat_vars:
        grid_verification_stats[var] = locals()[var]

    grid_verification_stats['stat_metadata'] = stat_metadata

    # dump the dictionary to a pickle file
    spfile = nexp+'_'+str(niters)+'_iters_grid_verification.pckl'
    print('writing statistics to pickle file: ' + spfile)
    outfile = open(spfile,'w')
    pickle.dump(grid_verification_stats,outfile)
