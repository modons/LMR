""" 
Module: LMR_verify_GM.py

Purpose: Generates verification statistics of LMR global-mean 2m air temperature
         against various gridded historical instrumental temperature datsasets 
         and reanalyses.  
         Note: started from LMR_plots.py r-86

Originator: Greg Hakim, U. of Washington, November 2015

Revisions: 
           21 July 2017: add consensus to detrended verification (GJH)
                         to do: make functions to do the repetetive actions
"""

import matplotlib
import sys
import csv
import glob, os, fnmatch
import numpy as np
import mpl_toolkits.basemap as bm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from scipy import stats
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pickle
import warnings

import pandas as pd

# LMR specific imports
sys.path.append('../')
from LMR_utils import global_hemispheric_means, assimilated_proxies, coefficient_efficiency, rank_histogram
from load_gridded_data import read_gridded_data_GISTEMP
from load_gridded_data import read_gridded_data_HadCRUT
from load_gridded_data import read_gridded_data_BerkeleyEarth
from load_gridded_data import read_gridded_data_MLOST
from load_gridded_data import read_gridded_data_CMIP5_model
from LMR_plot_support import find_date_indices, moving_average

# =============================================================================
def truncate_colormap(cmap, minval=0.0,maxval=1.0,n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name,a=minval,b=maxval),
        cmap(np.linspace(minval,maxval,n)))
    return new_cmap
# =============================================================================


warnings.filterwarnings('ignore')

##################################
# START:  set user parameters here
##################################

# define the verification time interval
stime = 1880
etime = 2000

# define the running time mean 
#nsyrs = 31 # 31-> 31-year running mean--nsyrs must be odd!
nsyrs = 5 # 5-> 5-year running mean--nsyrs must be odd!

# option to create figures
#iplot = False
iplot = True

# Open interactive windows of figures
interactive = True

if interactive:
    plt.ion()
else:
    # need to do this when running remotely, and to suppress figures
    matplotlib.use('agg')
    matplotlib.pyplot.switch_backend('agg')
# option to save figures to a file
# fsave = True
fsave = False

# save statistics file
stat_save = False

# file specification
#
# current datasets
# ---
# controls, published:
#nexp = 'production_gis_ccsm4_pagesall_0.75'
#nexp = 'production_mlost_ccsm4_pagesall_0.75'
#nexp = 'production_cru_ccsm4_pagesall_0.75'
#nexp = 'production_mlost_era20c_pagesall_0.75'
#nexp = 'production_mlost_era20cm_pagesall_0.75'
# ---
#nexp = 'test'
#nexp = 'pages2_noloc'
#nexp = 'pages2_loc10000'
#nexp = 'pages2_loc1000'
#nexp = 'pages2_loc5000'
#nexp = 'pages2_loc12000'
#nexp = 'pages2_loc15000'
#nexp = 'pages2_loc20000'
#nexp = 'pages2_loc12000_breit_seasonal_MetaTandP'
#nexp = 'pages2_loc12000_breit_seasonal_TorP'
#nexp = 'pages2_loc12000_pages2k2_seasonal_TorP'
#nexp = 'pages2_loc12000_pages2k2_seasonal_TorP_nens200'
#nexp = 'pages2_loc12000_pages2k2_seasonal_TorP_nens500'
#nexp = 'pages2_loc15000_pages2k2_seasonal_TorP_nens200'
#nexp = 'pages2_loc15000_pages2k2_seasonal_TorP_nens200_inflate1.25'
#nexp = 'pages2_loc15000_pages2k2_seasonal_TorP_nens200_inflate1.5'
#nexp = 'pages2_noloc_nens200'
#nexp = 'pages2_loc20000_pages2k2_seasonal_TorP_nens200'
#nexp = 'pages2_loc25000_pages2k2_seasonal_TorP_nens200'
#nexp = 'pages2_loc20000_seasonal_bilinear_nens200'
#nexp = 'pages2_loc25000_seasonal_bilinear_nens200'
#nexp = 'pages2_loc25000_seasonal_bilinear_nens200_75pct'
#nexp = 'pages2_loc25000_seasonal_bilinear_nens200_meta'
#nexp = 'pages2_noloc_seasonal_bilinear_nens1000'
#nexp = 'pages2_loc25000_seasonal_bilinear_nens1000'
nexp = 'test_instrumental_recon_py3'

# perform verification using all recon. MC realizations ( MCset = None )
# or over a custom selection ( MCset = (begin,end) )
# ex. MCset = (0,0)    -> only the first MC run
#     MCset = (0,10)   -> the first 11 MC runs (from 0 to 10 inclusively)
#     MCset = (80,100) -> the 80th to 100th MC runs (21 realizations)
MCset = None
#MCset = (0,0)

# specify directories for LMR data
#datadir_output = './data/'
#datadir_output = '/home/disk/kalman3/hakim/LMR'
#datadir_output = '/home/disk/kalman3/rtardif/LMR/output'
# datadir_output = '/home/disk/kalman3/hakim/LMR'
datadir_output = '/home/disk/katabatic3/wperkins/LMR_output/testing'
#datadir_output = '/home/disk/kalman3/rtardif/LMR/output'


# Directory where historical griddded data products can be found
datadir_calib = '/home/disk/kalman3/rtardif/LMR/data/analyses'

# Directory where reanalysis data can be found
datadir_reanl = '/home/disk/kalman3/rtardif/LMR/data/model'

# plotting preferences
nlevs = 30 # number of contours
alpha = 0.5 # alpha transpareny

# time limit for plot axis in years CE
xl = [stime,etime]

# this sets the default size of the figure in inches. ['figure.figsize'] = width, height;  
# aspect ratio appears preserved on smallest of the two
plt.rcParams['figure.figsize'] = 10, 10  # that's default image size for this interactive session
plt.rcParams['axes.linewidth'] = 2.0 #set the value globally
plt.rcParams['font.weight'] = 'bold' #set the font weight globally
plt.rc('text', usetex=False)
#plt.rc('text', usetex=True)

##################################
# END:  set user parameters here
##################################

print('--------------------------------------------------')
print('verification of global-mean 2m air temperature')
print('--------------------------------------------------')

workdir = datadir_output + '/' + nexp

# get directory and information for later use

print('--------------------------------------------------')
print('working directory: %s' % workdir)
print('--------------------------------------------------')

# get a listing of the iteration directories
dirs = glob.glob(workdir+"/r*")

# query file for assimilated proxy information (for now, ONLY IN THE r0 directory!)

ptypes,nrecords = assimilated_proxies(workdir+'/r0/')

print('--------------------------------------------------')
print('Assimilated proxies by type:')
for pt in sorted(ptypes.keys()):
    print('%40s : %s' % (pt, str(ptypes[pt])))
print('%40s : %s' % ('Total',str(nrecords)))
print('--------------------------------------------------')

# ==========================================
# load GISTEMP, HadCRU, BerkeleyEarth, MLOST
# ==========================================

# load GISTEMP
datafile_calib   = 'gistemp1200_ERSSTv4.nc'
calib_vars = ['Tsfc']
[gtime,GIS_lat,GIS_lon,GIS_anomaly] = read_gridded_data_GISTEMP(datadir_calib,datafile_calib,calib_vars,outfreq='annual')
GIS_time = np.array([d.year for d in gtime])
nlat_GIS = len(GIS_lat)
nlon_GIS = len(GIS_lon)


# load HadCRUT
datafile_calib   = 'HadCRUT.4.3.0.0.median.nc'
calib_vars = ['Tsfc']
[ctime,CRU_lat,CRU_lon,CRU_anomaly] = read_gridded_data_HadCRUT(datadir_calib,datafile_calib,calib_vars,outfreq='annual')
CRU_time = np.array([d.year for d in ctime])

## use GMT time series computed by Hadley Centre instead !!!!!!!!!!!!
#datafile_calib = '/home/disk/ekman/rtardif/kalman3/LMR/data/analyses/HadCRUT/HadCRUT.4.4.0.0.annual_ns_avg.txt'
#data = np.loadtxt(datafile_calib, usecols = (0,1))
#CRU_time = data[:,0].astype(np.int64)
#cru_gm   = data[:,1]

# load BerkeleyEarth
datafile_calib   = 'Land_and_Ocean_LatLong1.nc'
calib_vars = ['Tsfc']
[btime,BE_lat,BE_lon,BE_anomaly] = read_gridded_data_BerkeleyEarth(datadir_calib,datafile_calib,calib_vars,outfreq='annual')
BE_time = np.array([d.year for d in btime])

# load NOAA MLOST
path = datadir_calib + '/NOAA/'
#fname = 'NOAA_MLOST_aravg.ann.land_ocean.90S.90N.v3.5.4.201504.asc'
fname = 'NOAA_MLOST_aravg.ann.land_ocean.90S.90N.v4.0.0.201506.asc'
f = open(path+fname,'r')
dat = csv.reader(f)
mlost_time = []
mlost = []
for row in dat:
    # this is the year
    mlost_time.append(int(row[0].split()[0]))
    # this is the GMT temperature anomaly
    mlost.append(float(row[0].split()[1]))

# convert to numpy arrays
mlost_gm = np.array(mlost)
MLOST_time = np.array(mlost_time)


# ===================
# Reanalysis products
# ===================

# Define month sequence for the calendar year 
# (argument needed in upload of reanalysis data)
annual = list(range(1,13))

# load ECMWF's 20th century reanalysis (ERA20C) reanalysis --------------------------------
datadir = datadir_reanl+'/era20c'
datafile = 'tas_sfc_Amon_ERA20C_190001-201012.nc'
vardict = {'tas_sfc_Amon': 'anom'}
vardef = list(vardict.keys())[0]

dd = read_gridded_data_CMIP5_model(datadir,datafile,vardict,outtimeavg=annual)

rtime = dd[vardef]['years']
ERA20C_time = np.array([d.year for d in rtime])
lat_ERA20C = dd[vardef]['lat'][:,0]
lon_ERA20C = dd[vardef]['lon'][0,:]
nlat_ERA20C = len(lat_ERA20C)
nlon_ERA20C = len(lon_ERA20C)
ERA20C = dd[vardef]['value'] + dd[vardef]['climo'] # Full field
#ERA20C = dd[vardef]['value']                      # Anomalies

# compute and remove the mean over 1951-1980 reference period as w/ GIS & BE
smatch, ematch = find_date_indices(ERA20C_time,1951,1980)
ref_mean_era = np.mean(ERA20C[smatch:ematch,:,:],axis=0)
ERA20C = ERA20C - ref_mean_era

era_gm = np.zeros([len(ERA20C_time)])
era_nhm = np.zeros([len(ERA20C_time)])
era_shm = np.zeros([len(ERA20C_time)])
# Loop over years in dataset
for i in range(0,len(ERA20C_time)): 
    # compute the global & hemispheric mean temperature
    [era_gm[i],
     era_nhm[i],
     era_shm[i]] = global_hemispheric_means(ERA20C[i,:, :], lat_ERA20C)

# load NOAA's 20th century reanalysis (TCR) reanalysis --------------------------------
datadir = datadir_reanl+'/20cr'
datafile = 'tas_sfc_Amon_20CR_185101-201112.nc'
vardict = {'tas_sfc_Amon': 'anom'}
vardef = list(vardict.keys())[0]

dd = read_gridded_data_CMIP5_model(datadir,datafile,vardict,outtimeavg=annual)

rtime = dd[vardef]['years']
TCR_time = np.array([d.year for d in rtime])
lat_TCR = dd[vardef]['lat'][:,0]
lon_TCR = dd[vardef]['lon'][0,:]
nlat_TCR = len(lat_TCR)
nlon_TCR = len(lon_TCR)
TCR = dd[vardef]['value'] + dd[vardef]['climo'] # Full field
#TCR = dd[vardef]['value']                      # Anomalies

# compute and remove the mean over 1951-1980 reference period as w/ GIS & BE
smatch, ematch = find_date_indices(TCR_time,1951,1980)
ref_mean_tcr = np.mean(TCR[smatch:ematch,:,:],axis=0)
TCR = TCR - ref_mean_tcr

tcr_gm = np.zeros([len(TCR_time)])
tcr_nhm = np.zeros([len(TCR_time)])
tcr_shm = np.zeros([len(TCR_time)])
# Loop over years in dataset
for i in range(0,len(TCR_time)): 
    # compute the global & hemispheric mean temperature
    [tcr_gm[i],tcr_nhm[i],tcr_shm[i]] = global_hemispheric_means(TCR[i,:,:],
                                                                 lat_TCR)


#
# read LMR GMT data computed during DA
#

print('--------------------------------------------------')
print('reading LMR GMT data...')
print('--------------------------------------------------')
kk = -1
print('IPLOT = ' + str(iplot))
if iplot:
    fig = plt.figure()

first = True
kk = -1

# selecting  the MC iterations to keep
if MCset:
    dirset = dirs[MCset[0]:MCset[1]+1]
else:
    dirset = dirs
niters = len(dirset)

print('--------------------------------------------------')
print('niters = %s' % str(niters))
print('--------------------------------------------------')

# NEW---"grand ensemble approach"---collect all iterations into a superensemble
first = True
kk = -1
for dir in dirset:
    kk = kk + 1
    gmtpfile =  dir + '/gmt_ensemble.npz'
    npzfile = np.load(gmtpfile)
    npzfile.files
    gmt = npzfile['gmt_ensemble']
    nhmt = npzfile['nhmt_ensemble']
    shmt = npzfile['shmt_ensemble']
    recon_times = npzfile['recon_times']
    print(recon_times)
    print(gmtpfile)
    gmt_shape = np.shape(gmt)
    nhmt_shape = np.shape(nhmt)
    shmt_shape = np.shape(shmt)
    if first:
        gmt_save = np.zeros([gmt_shape[0],gmt_shape[1],niters])
        nhmt_save = np.zeros([nhmt_shape[0],nhmt_shape[1],niters])
        shmt_save = np.zeros([shmt_shape[0],shmt_shape[1],niters])
        first = False
        
    gmt_save[:,:,kk] = gmt
    nhmt_save[:,:,kk] = nhmt
    shmt_save[:,:,kk] = shmt
       
# average and 5-95% range
# 1. global mean
gmse = np.reshape(gmt_save,(gmt_shape[0],gmt_shape[1]*niters))
sagmt = np.mean(gmse,1)
gmt_min = np.percentile(gmse,5,axis=1)
gmt_max = np.percentile(gmse,95,axis=1)
# 2. NH
nhse = np.reshape(nhmt_save,(nhmt_shape[0],nhmt_shape[1]*niters))
sanhmt = np.mean(nhse,1)
nhmt_min = np.percentile(nhse,5,axis=1)
nhmt_max = np.percentile(nhse,95,axis=1)
# 3. SH
shse = np.reshape(shmt_save,(shmt_shape[0],shmt_shape[1]*niters))
sashmt = np.mean(shse,1)
shmt_min = np.percentile(shse,5,axis=1)
shmt_max = np.percentile(shse,95,axis=1)
    
# define for later use
lmr_gm = sagmt
LMR_time = recon_times


# 
# compute GIS, CRU  & BE global mean 
#

[gis_gm,_,_] = global_hemispheric_means(GIS_anomaly,GIS_lat)
[cru_gm,_,_] = global_hemispheric_means(CRU_anomaly,CRU_lat)
[be_gm,_,_]  = global_hemispheric_means(BE_anomaly,BE_lat)


# adjust so that all time series pertain to 20th century mean

# compute and remove the 20th century mean
satime = 1900
eatime = 1999

# LMR
smatch, ematch = find_date_indices(LMR_time,satime,eatime)
lmr_off = np.mean(lmr_gm[smatch:ematch])
lmr_gm = lmr_gm - lmr_off
# fix previously set values
gmt_min = gmt_min - lmr_off
gmt_max = gmt_max - lmr_off

# TCR
smatch, ematch = find_date_indices(TCR_time,satime,eatime)
tcr_gm  = tcr_gm  - np.mean(tcr_gm[smatch:ematch])
tcr_nhm = tcr_nhm - np.mean(tcr_nhm[smatch:ematch])
tcr_shm = tcr_shm - np.mean(tcr_shm[smatch:ematch])
# ERA
smatch, ematch = find_date_indices(ERA20C_time,satime,eatime)
era_gm  = era_gm  - np.mean(era_gm[smatch:ematch])
era_nhm = era_nhm - np.mean(era_nhm[smatch:ematch])
era_shm = era_shm - np.mean(era_shm[smatch:ematch])
# GIS
smatch, ematch = find_date_indices(GIS_time,satime,eatime)
gis_gm = gis_gm - np.mean(gis_gm[smatch:ematch])
# CRU
smatch, ematch = find_date_indices(CRU_time,satime,eatime)
cru_gm = cru_gm - np.mean(cru_gm[smatch:ematch])
# BE
smatch, ematch = find_date_indices(BE_time,satime,eatime)
be_gm = be_gm - np.mean(be_gm[smatch:ematch])
# MLOST
smatch, ematch = find_date_indices(MLOST_time,satime,eatime)
mlost_gm = mlost_gm - np.mean(mlost_gm[smatch:ematch])

# indices for chosen time interval defined by stime and etime
lmr_smatch, lmr_ematch = find_date_indices(LMR_time,stime,etime)
tcr_smatch, tcr_ematch = find_date_indices(TCR_time,stime,etime)
era_smatch, era_ematch = find_date_indices(ERA20C_time,stime,etime)
gis_smatch, gis_ematch = find_date_indices(GIS_time,stime,etime)
cru_smatch, cru_ematch = find_date_indices(CRU_time,stime,etime)
be_smatch, be_ematch = find_date_indices(BE_time,stime,etime)
mlost_smatch, mlost_ematch = find_date_indices(MLOST_time,stime,etime)

# "consensus" global mean: average all non-LMR (obs-based) values
consensus_gmt = np.array([gis_gm[gis_smatch:gis_ematch],cru_gm[cru_smatch:cru_ematch],be_gm[be_smatch:be_ematch],mlost_gm[mlost_smatch:mlost_ematch]])
con_gm = np.mean(consensus_gmt,axis=0)
CON_time = np.arange(stime,etime)
CON_time = np.asarray(CON_time) # fixed 21 July 2017 (GJH)

# write to a file for use by other programs
#filen = 'consensus_gmt.npz'
#np.savez(filen,con_gm=con_gm,CON_time=CON_time)

#
# correlation coefficients & CE over chosen time interval 
#

verif_yrs = np.arange(stime,etime+1,1)


# LMR-TCR
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time, TCR_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, overlap_yrs))
ind_tcr = np.searchsorted(TCR_time, np.intersect1d(TCR_time, overlap_yrs))
lmr_tcr_corr = np.corrcoef(lmr_gm[ind_lmr],tcr_gm[ind_tcr])
lmr_tcr_ce = coefficient_efficiency(tcr_gm[ind_tcr],lmr_gm[ind_lmr])

# LMR-ERA
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time, ERA20C_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, overlap_yrs))
ind_era = np.searchsorted(ERA20C_time, np.intersect1d(ERA20C_time, overlap_yrs))
lmr_era_corr = np.corrcoef(lmr_gm[ind_lmr],era_gm[ind_era])
lmr_era_ce = coefficient_efficiency(era_gm[ind_era],lmr_gm[ind_lmr])

# LMR-GIS
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time, GIS_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, overlap_yrs))
ind_gis = np.searchsorted(GIS_time, np.intersect1d(GIS_time, overlap_yrs))
lmr_gis_corr = np.corrcoef(lmr_gm[ind_lmr],gis_gm[ind_gis])
lmr_gis_ce = coefficient_efficiency(gis_gm[ind_gis],lmr_gm[ind_lmr])

# LMR-CRU
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time, CRU_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, overlap_yrs))
ind_cru = np.searchsorted(CRU_time, np.intersect1d(CRU_time, overlap_yrs))
lmr_cru_corr = np.corrcoef(lmr_gm[ind_lmr],cru_gm[ind_cru])
lmr_cru_ce = coefficient_efficiency(cru_gm[ind_cru],lmr_gm[ind_lmr])

# LMR-BE
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time, BE_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, overlap_yrs))
ind_be  = np.searchsorted(BE_time, np.intersect1d(BE_time, overlap_yrs))
lmr_be_corr = np.corrcoef(lmr_gm[ind_lmr],be_gm[ind_be])
lmr_be_ce = coefficient_efficiency(be_gm[ind_be],lmr_gm[ind_lmr])

# LMR-MLOST
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time, MLOST_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, overlap_yrs))
ind_mlost = np.searchsorted(MLOST_time, np.intersect1d(MLOST_time, overlap_yrs))
lmr_mlost_corr = np.corrcoef(lmr_gm[ind_lmr],mlost_gm[ind_mlost])
lmr_mlost_ce = coefficient_efficiency(mlost_gm[ind_mlost],lmr_gm[ind_lmr])

# LMR-consensus
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time, CON_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, overlap_yrs))
ind_con = np.searchsorted(CON_time, np.intersect1d(CON_time, overlap_yrs))
lmr_con_corr = np.corrcoef(lmr_gm[ind_lmr],con_gm[ind_con])
lmr_con_ce = coefficient_efficiency(con_gm[ind_con],lmr_gm[ind_lmr])

# GIS-TCR
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(GIS_time, TCR_time), verif_yrs)
ind_gis = np.searchsorted(GIS_time, np.intersect1d(GIS_time, overlap_yrs))
ind_tcr = np.searchsorted(TCR_time, np.intersect1d(TCR_time, overlap_yrs))
gis_tcr_corr = np.corrcoef(gis_gm[ind_gis],tcr_gm[ind_tcr])
tcr_gis_ce = coefficient_efficiency(gis_gm[ind_gis],tcr_gm[ind_tcr])

# GIS-ERA
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(GIS_time, ERA20C_time), verif_yrs)
ind_gis = np.searchsorted(GIS_time, np.intersect1d(GIS_time, overlap_yrs))
ind_era = np.searchsorted(ERA20C_time, np.intersect1d(ERA20C_time, overlap_yrs))
gis_era_corr = np.corrcoef(gis_gm[ind_gis],era_gm[ind_era])
era_gis_ce = coefficient_efficiency(gis_gm[ind_gis],era_gm[ind_era])

# GIS-BE
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(GIS_time, BE_time), verif_yrs)
ind_gis = np.searchsorted(GIS_time, np.intersect1d(GIS_time, overlap_yrs))
ind_be  = np.searchsorted(BE_time, np.intersect1d(BE_time, overlap_yrs))
gis_be_corr = np.corrcoef(gis_gm[ind_gis],be_gm[ind_be])
be_gis_ce = coefficient_efficiency(gis_gm[ind_gis],be_gm[ind_be])

# GIS-CRU
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(GIS_time, CRU_time), verif_yrs)
ind_gis = np.searchsorted(GIS_time, np.intersect1d(GIS_time, overlap_yrs))
ind_cru = np.searchsorted(CRU_time, np.intersect1d(CRU_time, overlap_yrs))
gis_cru_corr = np.corrcoef(gis_gm[ind_gis],cru_gm[ind_cru])
cru_gis_ce = coefficient_efficiency(gis_gm[ind_gis],cru_gm[ind_cru])


# LMR
ltc = str(float('%.2f' % lmr_tcr_corr[0,1]))
lec = str(float('%.2f' % lmr_era_corr[0,1]))
lcc = str(float('%.2f' % lmr_cru_corr[0,1]))
lgc = str(float('%.2f' % lmr_gis_corr[0,1]))
lbc = str(float('%.2f' % lmr_be_corr[0,1]))
loc = str(float('%.2f' % lmr_con_corr[0,1]))
lmc = str(float('%.2f' % lmr_mlost_corr[0,1]))
# reference
gtc = str(float('%.2f' % gis_tcr_corr[0,1]))
gec = str(float('%.2f' % gis_era_corr[0,1]))
gcc = str(float('%.2f' % gis_cru_corr[0,1]))
gbc = str(float('%.2f' % gis_be_corr[0,1]))
print('--------------------------------------------------')
print('annual-mean correlations: ')
print('LMR_TCR       correlation: %s' % ltc)
print('LMR_ERA       correlation: %s' % lec)
print('LMR_GIS       correlation: %s' % lgc)
print('LMR_CRU       correlation: %s' % lcc)
print('LMR_BE        correlation: %s' % lbc)
print('LMR_MLOST     correlation: %s' % lmc)
print('GIS_TCR       correlation: %s' % gtc)
print('GIS_ERA       correlation: %s' % gec)
print('GIS_CRU       correlation: %s' % gcc)
print('GIS_BE        correlation: %s' % gbc)
print('LMR_consensus correlation: %s' % loc)
print('--------------------------------------------------')


ltce = str(float('%.2f' % lmr_tcr_ce))
lece = str(float('%.2f' % lmr_era_ce))
lgce = str(float('%.2f' % lmr_gis_ce))
lcce = str(float('%.2f' % lmr_cru_ce))
lbce = str(float('%.2f' % lmr_be_ce))
lmce = str(float('%.2f' % lmr_mlost_ce))
loce = str(float('%.2f' % lmr_con_ce))
tgce = str(float('%.2f' % tcr_gis_ce))
egce = str(float('%.2f' % era_gis_ce))
bgce = str(float('%.2f' % be_gis_ce))
cgce = str(float('%.2f' % cru_gis_ce))
print('--------------------------------------------------')
print('coefficient of efficiency: ')
print('LMR-TCR CE  : %s' % str(ltce))
print('LMR-ERA CE  : %s' % str(lece))
print('LMR-GIS CE  : %s' % str(lgce))
print('LMR-CRU CE  : %s' % str(lcce))
print('LMR-BE CE   : %s' % str(lbce))
print('LMR-MLOST CE: %s' % str(lmce))
print('LMR-CON CE  : %s' % str(loce))
print('TCR-GIS CE  : %s' % str(tgce))
print('ERA-GIS CE  : %s' % str(egce))
print('BE-CRU CE   : %s' % str(bgce))
print('GIS-CRU CE  : %s' % str(cgce))
print('--------------------------------------------------')

#
# spread--error
#
lg_err = lmr_gm[lmr_smatch:lmr_ematch] - gis_gm[gis_smatch:gis_ematch]
svar = gmt_save[:,lmr_smatch:lmr_ematch].var(0,ddof=1)
calib = lg_err.var(0,ddof=1)/svar.mean(0)
print('--------------------------------------------------')
print(('ensemble calibration: %s' % str(calib)))
print('--------------------------------------------------')

# ========================================================
# plots
# ========================================================
if iplot:
    lw = 2
    fig = plt.figure()
    plt.plot(LMR_time,lmr_gm,'k-'        ,linewidth=lw*2,label='LMR')
    plt.plot(GIS_time,gis_gm,'r-'        ,linewidth=lw,label='GISTEMP',alpha=alpha)
    plt.plot(CRU_time,cru_gm,'m-'        ,linewidth=lw,label='HadCRUT4',alpha=alpha)
    plt.plot(BE_time,be_gm,'g-'          ,linewidth=lw,label='BE',alpha=alpha)
    plt.plot(MLOST_time,mlost_gm,'c-'    ,linewidth=lw,label='MLOST',alpha=alpha)
    plt.plot(TCR_time,tcr_gm,'y-'        ,linewidth=lw,label='20CR-V2',alpha=alpha)
    plt.plot(ERA20C_time,era_gm,'b-'     ,linewidth=lw,label='ERA-20C',alpha=alpha)
    plt.plot(CON_time,con_gm,color='lime',linestyle='-',linewidth=lw,label='consensus',alpha=alpha)
    plt.fill_between(recon_times,gmt_min,gmt_max,facecolor='gray',alpha = 0.5,linewidth=0.)
    #plt.plot(LMR_time,lmr_gm,'k-'    ,linewidth=lw*2) # LMR back on top
    xl_loc = [stime,etime]
    yl_loc = [-1.,1.]
    #plt.title('Global mean temperature\n(' + nexp + ')',weight='bold',y=1.025)
    plt.title('Global mean temperature',weight='bold',y=1.025)
    plt.xlabel('Year CE',fontweight='bold')
    plt.ylabel('Temperature anomaly (K)',fontweight='bold')

    plt.xlim(xl_loc)
    plt.ylim(yl_loc)
    txl = xl_loc[0] + (xl_loc[1]-xl_loc[0])*.45
    tyl = yl_loc[0] + (yl_loc[1]-yl_loc[0])*.2
    offset = 0.05

    plt.text(txl,tyl,'(LMR,GISTEMP)  : r= ' + lgc.ljust(5,' ') + ' CE= ' + lgce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,HadCRUT4) : r= ' + lcc.ljust(5,' ') + ' CE= ' + lcce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,BE)       : r= ' + lbc.ljust(5,' ') + ' CE= ' + lbce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,MLOST)    : r= ' + lmc.ljust(5,' ') + ' CE= ' + lmce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,20CR-V2)  : r= ' + ltc.ljust(5,' ') + ' CE= ' + ltce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,ERA-20C)  : r= ' + lec.ljust(5,' ') + ' CE= ' + lece.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,consensus): r= ' + loc.ljust(5,' ') + ' CE= ' + loce.ljust(5,' '), fontsize=14, family='monospace')

    plt.plot(xl_loc,[0,0],color='gray',linestyle=':',lw=2)
    plt.legend(loc=2)

    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_annual.png')
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_annual.pdf',bbox_inches='tight', dpi=300, format='pdf')

#
# time averages
#

LMR_smoothed,LMR_smoothed_years     = moving_average(lmr_gm,recon_times,nsyrs) 
TCR_smoothed,TCR_smoothed_years     = moving_average(tcr_gm,TCR_time,nsyrs) 
ERA_smoothed,ERA_smoothed_years     = moving_average(era_gm,ERA20C_time,nsyrs) 
GIS_smoothed,GIS_smoothed_years     = moving_average(gis_gm,GIS_time,nsyrs) 
CRU_smoothed,CRU_smoothed_years     = moving_average(cru_gm,CRU_time,nsyrs) 
BE_smoothed,BE_smoothed_years       = moving_average(be_gm,BE_time,nsyrs) 
MLOST_smoothed,MLOST_smoothed_years = moving_average(mlost_gm,MLOST_time,nsyrs) 
CON_smoothed,CON_smoothed_years = moving_average(con_gm,CON_time,nsyrs) 

# index offsets to account for averaging
toff = int(nsyrs/2)

verif_yrs = np.arange(stime+toff,etime-toff+1,1)

# LMR-TCR
overlap_yrs = np.intersect1d(np.intersect1d(LMR_smoothed_years, TCR_smoothed_years), verif_yrs)
ind_lmr = np.searchsorted(LMR_smoothed_years, np.intersect1d(LMR_smoothed_years, overlap_yrs))
ind_tcr = np.searchsorted(TCR_smoothed_years, np.intersect1d(TCR_smoothed_years, overlap_yrs))
ls_ts_corr = np.corrcoef(LMR_smoothed[ind_lmr],TCR_smoothed[ind_tcr])
ls_ts_ce = coefficient_efficiency(TCR_smoothed[ind_tcr],LMR_smoothed[ind_lmr])

lstsc  = str(float('%.2f' % ls_ts_corr[0,1]))
lstsce = str(float('%.2f' % ls_ts_ce))

# LMR-ERA
overlap_yrs = np.intersect1d(np.intersect1d(LMR_smoothed_years, ERA_smoothed_years), verif_yrs)
ind_lmr = np.searchsorted(LMR_smoothed_years, np.intersect1d(LMR_smoothed_years, overlap_yrs))
ind_era = np.searchsorted(ERA_smoothed_years, np.intersect1d(ERA_smoothed_years, overlap_yrs))
ls_es_corr = np.corrcoef(LMR_smoothed[ind_lmr],ERA_smoothed[ind_era])
ls_es_ce = coefficient_efficiency(ERA_smoothed[ind_era],LMR_smoothed[ind_lmr])

lsesc  = str(float('%.2f' % ls_es_corr[0,1]))
lsesce = str(float('%.2f' % ls_es_ce))

# LMR-GIS
overlap_yrs = np.intersect1d(np.intersect1d(LMR_smoothed_years, GIS_smoothed_years), verif_yrs)
ind_lmr = np.searchsorted(LMR_smoothed_years, np.intersect1d(LMR_smoothed_years, overlap_yrs))
ind_gis = np.searchsorted(GIS_smoothed_years, np.intersect1d(GIS_smoothed_years, overlap_yrs))
ls_gs_corr = np.corrcoef(LMR_smoothed[ind_lmr],GIS_smoothed[ind_gis])
ls_gs_ce = coefficient_efficiency(GIS_smoothed[ind_gis],LMR_smoothed[ind_lmr])

lsgsc  = str(float('%.2f' % ls_gs_corr[0,1]))
lsgsce = str(float('%.2f' % ls_gs_ce))

# LMR-CRU
overlap_yrs = np.intersect1d(np.intersect1d(LMR_smoothed_years, CRU_smoothed_years), verif_yrs)
ind_lmr = np.searchsorted(LMR_smoothed_years, np.intersect1d(LMR_smoothed_years, overlap_yrs))
ind_cru = np.searchsorted(CRU_smoothed_years, np.intersect1d(CRU_smoothed_years, overlap_yrs))
ls_cs_corr = np.corrcoef(LMR_smoothed[ind_lmr],CRU_smoothed[ind_cru])
ls_cs_ce = coefficient_efficiency(CRU_smoothed[ind_cru],LMR_smoothed[ind_lmr])

lscsc  = str(float('%.2f' % ls_cs_corr[0,1]))
lscsce = str(float('%.2f' % ls_cs_ce))

# LMR-BE
overlap_yrs = np.intersect1d(np.intersect1d(LMR_smoothed_years, BE_smoothed_years), verif_yrs)
ind_lmr = np.searchsorted(LMR_smoothed_years, np.intersect1d(LMR_smoothed_years, overlap_yrs))
ind_be = np.searchsorted(BE_smoothed_years, np.intersect1d(BE_smoothed_years, overlap_yrs))
ls_bs_corr = np.corrcoef(LMR_smoothed[ind_lmr],CRU_smoothed[ind_be])
ls_bs_ce = coefficient_efficiency(BE_smoothed[ind_be],LMR_smoothed[ind_lmr])

lsbsc  = str(float('%.2f' % ls_bs_corr[0,1]))
lsbsce = str(float('%.2f' % ls_bs_ce))

# LMR-MLOST
overlap_yrs = np.intersect1d(np.intersect1d(LMR_smoothed_years, MLOST_smoothed_years), verif_yrs)
ind_lmr = np.searchsorted(LMR_smoothed_years, np.intersect1d(LMR_smoothed_years, overlap_yrs))
ind_mlost = np.searchsorted(MLOST_smoothed_years, np.intersect1d(MLOST_smoothed_years, overlap_yrs))
ls_ms_corr = np.corrcoef(LMR_smoothed[ind_lmr],MLOST_smoothed[ind_mlost])
ls_ms_ce = coefficient_efficiency(MLOST_smoothed[ind_mlost],LMR_smoothed[ind_lmr])

lsmsc  = str(float('%.2f' % ls_ms_corr[0,1]))
lsmsce = str(float('%.2f' % ls_ms_ce))


# LMR-consensus
overlap_yrs = np.intersect1d(np.intersect1d(LMR_smoothed_years, CON_smoothed_years), verif_yrs)
ind_lmr = np.searchsorted(LMR_smoothed_years, np.intersect1d(LMR_smoothed_years, overlap_yrs))
ind_con = np.searchsorted(CON_smoothed_years, np.intersect1d(CON_smoothed_years, overlap_yrs))
ls_con_corr = np.corrcoef(LMR_smoothed[ind_lmr],CON_smoothed[ind_con])
ls_con_ce = coefficient_efficiency(CON_smoothed[ind_con],LMR_smoothed[ind_lmr])
lmr_con_corr = np.corrcoef(lmr_gm[ind_lmr],con_gm[ind_con])
lmr_con_ce = coefficient_efficiency(con_gm[ind_con],lmr_gm[ind_lmr])

lsconsc  = str(float('%.2f' % ls_con_corr[0,1]))
lsconsce = str(float('%.2f' % ls_con_ce))

print('--------------------------------------------------')
print('%s-year-smoothed correlations:' % str(nsyrs))
print('smoothed lmr-gis correlation =   %s' % lsgsc)
print('smoothed lmr-cru correlation =   %s' % lscsc)
print('smoothed lmr-be correlation =    %s' % lsbsc)
print('smoothed lmr-mlost correlation = %s' % lsmsc)
print('smoothed lmr-tcr correlation =   %s' % lstsc)
print('smoothed lmr-era correlation =   %s' % lsesc)
print('smoothed lmr-con correlation =   %s' % lsconsc)
print('--------------------------------------------------')
print('--------------------------------------------------')
print('%s-year-smoothed CE:' % str(nsyrs))
print('smoothed lmr-gis CE =   %s' % lsgsce)
print('smoothed lmr-cru CE =   %s' % lscsce)
print('smoothed lmr-be CE =    %s' % lsbsce)
print('smoothed lmr-mlost CE = %s' % lsmsce)
print('smoothed lmr-tcr CE =   %s' % lstsce)
print('smoothed lmr-era CE =   %s' % lsesce)
print('smoothed lmr-con CE =   %s' % lsconsce)
print('--------------------------------------------------')

if iplot:
    fig = plt.figure()
    #plt.plot(recon_times,lmr_gm,'k-',linewidth=2)
    #plt.fill_between(recon_times,gmt_min,gmt_max,facecolor='gray',alpha=alpha,linewidth=0.)

    # add smoothed lines
    plt.plot(LMR_smoothed_years,LMR_smoothed,'k-'         ,linewidth=4, label='LMR')
    plt.plot(GIS_smoothed_years,GIS_smoothed,'r-'         ,linewidth=4, label='GISTEMP',alpha=alpha)
    plt.plot(CRU_smoothed_years,CRU_smoothed,'m-'         ,linewidth=4, label='HadCRUT4',alpha=alpha)
    plt.plot(BE_smoothed_years,BE_smoothed,'g-'           ,linewidth=4, label='BE',alpha=alpha)
    plt.plot(MLOST_smoothed_years,MLOST_smoothed,'c-'     ,linewidth=4, label='MLOST',alpha=alpha)
    plt.plot(TCR_smoothed_years,TCR_smoothed,'y-'         ,linewidth=4, label='20CR-V2',alpha=alpha)
    plt.plot(ERA_smoothed_years,ERA_smoothed,'b-'         ,linewidth=4, label='ERA-20C',alpha=alpha)
    plt.plot(CON_smoothed_years,CON_smoothed,color='lime' ,linewidth=4, label='consensus',alpha=alpha)
    #plt.title('Global mean temperature range (gray) and ' +str(nsyrs) + '-year moving average\n(' + nexp + ')',weight='bold',y=1.03)
    plt.title('Global mean temperature range (gray) and ' +str(nsyrs) + '-year moving average',weight='bold',y=1.03)
    plt.xlabel('Year CE', fontweight='bold')
    plt.ylabel('Temperature anomaly (K)', fontweight='bold')

    if nsyrs == 5:
        xl_loc = [stime,etime]
        yl_loc = [-1.,1.]
    elif nsyrs == 31:
        xl_loc = [1000,2000]
        yl_loc = [-1.1,0.6] # for comparison with Wikipedia figure
    else:
        xl_loc = [stime,etime]
        yl_loc = [-1,1]
        
    plt.xlim(xl_loc)
    plt.ylim(yl_loc)
    txl = xl_loc[0] + (xl_loc[1]-xl_loc[0])*.4
    tyl = yl_loc[0] + (yl_loc[1]-yl_loc[0])*.2

    plt.text(txl,tyl,'(LMR,GISTEMP)  : r= ' + lsgsc.ljust(5,' ') + ' CE= ' + lsgsce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-0.05
    plt.text(txl,tyl,'(LMR,HadCRUT4) : r= ' + lscsc.ljust(5,' ') + ' CE= ' + lscsce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-0.05
    plt.text(txl,tyl,'(LMR,BE)       : r= ' + lsbsc.ljust(5,' ') + ' CE= ' + lsbsce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-0.05
    plt.text(txl,tyl,'(LMR,MLOST)    : r= ' + lsmsc.ljust(5,' ') + ' CE= ' + lsmsce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-0.05
    plt.text(txl,tyl,'(LMR,20CR-V2)  : r= ' + lstsc.ljust(5,' ') + ' CE= ' + lstsce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-0.05
    plt.text(txl,tyl,'(LMR,ERA-20C)  : r= ' + lsesc.ljust(5,' ') + ' CE= ' + lsesce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-0.05
    plt.text(txl,tyl,'(LMR,consensus): r= ' + lsconsc.ljust(5,' ') + ' CE= ' + lsconsce.ljust(5,' '), fontsize=14, family='monospace')

    plt.plot(xl_loc,[0,0],color='gray',linestyle=':',lw=2)
    plt.legend(loc=2)

    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_'+str(nsyrs)+'yr_smoothed.png')
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_'+str(nsyrs)+'yr_smoothed.pdf',bbox_inches='tight', dpi=300, format='pdf')




# =======================================
# detrend and verify the detrended signal
# =======================================

print('--------------------------------------------------')
print('verification of detrended data')
print('--------------------------------------------------')

verif_yrs = np.arange(stime,etime+1,1)

# for LMR
ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, verif_yrs))
# save copies of the original data for residual estimates later
lmr_gm_copy = np.copy(lmr_gm[ind_lmr])
LMR_time_copy = np.copy(LMR_time[ind_lmr])
xvar = list(range(len(lmr_gm_copy)))
lmr_slope, lmr_intercept, r_value, p_value, std_err = stats.linregress(xvar,lmr_gm_copy)
lmr_trend = lmr_slope*np.squeeze(xvar) + lmr_intercept
lmr_gm_detrend = lmr_gm_copy - lmr_trend


# for GIS
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time_copy, GIS_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time_copy, np.intersect1d(LMR_time_copy, overlap_yrs))
ind_gis = np.searchsorted(GIS_time, np.intersect1d(GIS_time, overlap_yrs))
GIS_time_copy = GIS_time[ind_gis]
gis_gm_copy = np.copy(gis_gm[ind_gis])
xvar = list(range(len(ind_gis)))
gis_slope, gis_intercept, r_value, p_value, std_err = stats.linregress(xvar,gis_gm_copy)
gis_trend = gis_slope*np.squeeze(xvar) + gis_intercept
gis_gm_detrend = gis_gm_copy - gis_trend
# r and ce on full data
full_err  = lmr_gm_copy[ind_lmr] - gis_gm_copy
lmr_gis_corr_full = np.corrcoef(lmr_gm_copy[ind_lmr],gis_gm_copy)
lmr_gis_ce_full = coefficient_efficiency(gis_gm_copy,lmr_gm_copy[ind_lmr])
lgrf =  str(float('%.2f' % lmr_gis_corr_full[0,1]))
lgcf =  str(float('%.2f' % lmr_gis_ce_full))
# r and ce on detrended data
lmr_gis_corr_detrend = np.corrcoef(lmr_gm_detrend[ind_lmr],gis_gm_detrend)
lmr_detrend_err = lmr_gm_detrend[ind_lmr] - gis_gm_detrend
lmr_gis_ce_detrend = coefficient_efficiency(gis_gm_detrend,lmr_gm_detrend[ind_lmr])
lgrd   =  str(float('%.2f' % lmr_gis_corr_detrend[0,1]))
lgcd   =  str(float('%.2f' % lmr_gis_ce_detrend))

# check if the two pieces are correlated (if not, then they sum to the total error)
error_trend = lmr_trend[ind_lmr] - gis_trend
error_detrend = lmr_gm_detrend[ind_lmr] - gis_gm_detrend
check = np.corrcoef(error_trend,error_detrend)
print('correlaton between trend and detrend errors = %s' % str(check[0,1]))
print('check error variances...')
print('trend error: %s' % str(np.var(error_trend)))
print('detrend error: %s' % str(np.var(error_detrend)))
print('detrend error + trend error: %s' % str(np.var(error_trend)+np.var(error_detrend)))
print('full error : %s' % str(np.var(error_trend+error_detrend)))

# for CRU
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time_copy, CRU_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time_copy, np.intersect1d(LMR_time_copy, overlap_yrs))
ind_cru = np.searchsorted(CRU_time, np.intersect1d(CRU_time, overlap_yrs))
CRU_time_copy = CRU_time[ind_cru]
cru_gm_copy = np.copy(cru_gm[ind_cru])
xvar = list(range(len(ind_cru)))
cru_slope, cru_intercept, r_value, p_value, std_err = stats.linregress(xvar,cru_gm_copy)
cru_trend = cru_slope*np.squeeze(xvar) + cru_intercept
cru_gm_detrend = cru_gm_copy - cru_trend
# r and ce on full data
full_err  = lmr_gm_copy[ind_lmr] - cru_gm_copy
lmr_cru_corr_full = np.corrcoef(lmr_gm_copy[ind_lmr],cru_gm_copy)
lmr_cru_ce_full = coefficient_efficiency(cru_gm_copy,lmr_gm_copy[ind_lmr])
lcrf =  str(float('%.2f' % lmr_cru_corr_full[0,1]))
lccf =  str(float('%.2f' % lmr_cru_ce_full))
# r and ce on detrended data
lmr_cru_corr_detrend = np.corrcoef(lmr_gm_detrend[ind_lmr],cru_gm_detrend)
lmr_detrend_err = lmr_gm_detrend[ind_lmr] - cru_gm_detrend
lmr_cru_ce_detrend = coefficient_efficiency(cru_gm_detrend,lmr_gm_detrend[ind_lmr])
lcrd   =  str(float('%.2f' % lmr_cru_corr_detrend[0,1]))
lccd   =  str(float('%.2f' % lmr_cru_ce_detrend))

# for BE
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time_copy, BE_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time_copy, np.intersect1d(LMR_time_copy, overlap_yrs))
ind_be = np.searchsorted(BE_time, np.intersect1d(BE_time, overlap_yrs))
BE_time_copy = BE_time[ind_be]
be_gm_copy = np.copy(be_gm[ind_be])
xvar = list(range(len(ind_be)))
be_slope, be_intercept, r_value, p_value, std_err = stats.linregress(xvar,be_gm_copy)
be_trend = be_slope*np.squeeze(xvar) + be_intercept
be_gm_detrend = be_gm_copy - be_trend
# r and ce on full data
full_err  = lmr_gm_copy[ind_lmr] - be_gm_copy
lmr_be_corr_full = np.corrcoef(lmr_gm_copy[ind_lmr],be_gm_copy)
lmr_be_ce_full = coefficient_efficiency(be_gm_copy,lmr_gm_copy[ind_lmr])
lbrf =  str(float('%.2f' % lmr_be_corr_full[0,1]))
lbcf =  str(float('%.2f' % lmr_be_ce_full))
# r and ce on detrended data
lmr_be_corr_detrend = np.corrcoef(lmr_gm_detrend[ind_lmr],be_gm_detrend)
lmr_detrend_err = lmr_gm_detrend[ind_lmr] - be_gm_detrend
lmr_be_ce_detrend = coefficient_efficiency(be_gm_detrend,lmr_gm_detrend[ind_lmr])
lbrd   =  str(float('%.2f' % lmr_be_corr_detrend[0,1]))
lbcd   =  str(float('%.2f' % lmr_be_ce_detrend))

# for MLOST
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time_copy, MLOST_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time_copy, np.intersect1d(LMR_time_copy, overlap_yrs))
ind_mlost = np.searchsorted(MLOST_time, np.intersect1d(MLOST_time, overlap_yrs))
MLOST_time_copy = MLOST_time[ind_mlost]
mlost_gm_copy = np.copy(mlost_gm[ind_mlost])
xvar = list(range(len(ind_mlost)))
mlost_slope, mlost_intercept, r_value, p_value, std_err = stats.linregress(xvar,mlost_gm_copy)
mlost_trend = mlost_slope*np.squeeze(xvar) + mlost_intercept
mlost_gm_detrend = mlost_gm_copy - mlost_trend
# r and ce on full data
full_err  = lmr_gm_copy[ind_lmr] - mlost_gm_copy
lmr_mlost_corr_full = np.corrcoef(lmr_gm_copy[ind_lmr],mlost_gm_copy)
lmr_mlost_ce_full = coefficient_efficiency(mlost_gm_copy,lmr_gm_copy[ind_lmr])
lmrf =  str(float('%.2f' % lmr_mlost_corr_full[0,1]))
lmcf =  str(float('%.2f' % lmr_mlost_ce_full))
# r and ce on detrended data
lmr_mlost_corr_detrend = np.corrcoef(lmr_gm_detrend[ind_lmr],mlost_gm_detrend)
lmr_detrend_err = lmr_gm_detrend[ind_lmr] - mlost_gm_detrend
lmr_mlost_ce_detrend = coefficient_efficiency(mlost_gm_detrend,lmr_gm_detrend[ind_lmr])
lmrd   =  str(float('%.2f' % lmr_mlost_corr_detrend[0,1]))
lmcd   =  str(float('%.2f' % lmr_mlost_ce_detrend))

# for TCR
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time_copy, TCR_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time_copy, np.intersect1d(LMR_time_copy, overlap_yrs))
ind_tcr = np.searchsorted(TCR_time, np.intersect1d(TCR_time, overlap_yrs))
TCR_time_copy = TCR_time[ind_tcr]
tcr_gm_copy = np.copy(tcr_gm[ind_tcr])
xvar = list(range(len(ind_tcr)))
tcr_slope, tcr_intercept, r_value, p_value, std_err = stats.linregress(xvar,tcr_gm_copy)
tcr_trend = tcr_slope*np.squeeze(xvar) + tcr_intercept
tcr_gm_detrend = tcr_gm_copy - tcr_trend
# r and ce on full data
full_err  = lmr_gm_copy[ind_lmr] - tcr_gm_copy
lmr_tcr_corr_full = np.corrcoef(lmr_gm_copy[ind_lmr],tcr_gm_copy)
lmr_tcr_ce_full = coefficient_efficiency(tcr_gm_copy,lmr_gm_copy[ind_lmr])
ltrf =  str(float('%.2f' % lmr_tcr_corr_full[0,1]))
ltcf =  str(float('%.2f' % lmr_tcr_ce_full))
# r and ce on detrended data
lmr_tcr_corr_detrend = np.corrcoef(lmr_gm_detrend[ind_lmr],tcr_gm_detrend)
lmr_detrend_err = lmr_gm_detrend[ind_lmr] - tcr_gm_detrend
lmr_tcr_ce_detrend = coefficient_efficiency(tcr_gm_detrend,lmr_gm_detrend[ind_lmr])
ltrd   =  str(float('%.2f' % lmr_tcr_corr_detrend[0,1]))
ltcd   =  str(float('%.2f' % lmr_tcr_ce_detrend))

# for ERA
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time_copy, ERA20C_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time_copy, np.intersect1d(LMR_time_copy, overlap_yrs))
ind_era = np.searchsorted(ERA20C_time, np.intersect1d(ERA20C_time, overlap_yrs))
ERA_time_copy = ERA20C_time[ind_era]
era_gm_copy = np.copy(era_gm[ind_era])
xvar = list(range(len(ind_era)))
era_slope, era_intercept, r_value, p_value, std_err = stats.linregress(xvar,era_gm_copy)
era_trend = era_slope*np.squeeze(xvar) + era_intercept
era_gm_detrend = era_gm_copy - era_trend
# r and ce on full data
full_err  = lmr_gm_copy[ind_lmr] - era_gm_copy
lmr_era_corr_full = np.corrcoef(lmr_gm_copy[ind_lmr],era_gm[ind_era])
lmr_era_ce_full = coefficient_efficiency(era_gm_copy,lmr_gm_copy[ind_lmr])
lerf =  str(float('%.2f' % lmr_era_corr_full[0,1]))
lecf =  str(float('%.2f' % lmr_era_ce_full))
# r and ce on detrended data
lmr_era_corr_detrend = np.corrcoef(lmr_gm_detrend[ind_lmr],era_gm_detrend)
lmr_detrend_err = lmr_gm_detrend[ind_lmr] - era_gm_detrend
lmr_era_ce_detrend = coefficient_efficiency(era_gm_detrend,lmr_gm_detrend[ind_lmr])
lerd   =  str(float('%.2f' % lmr_era_corr_detrend[0,1]))
lecd   =  str(float('%.2f' % lmr_era_ce_detrend))

# for CONsensus
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time_copy, CON_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time_copy, np.intersect1d(LMR_time_copy, overlap_yrs))
ind_con = np.searchsorted(CON_time, np.intersect1d(CON_time, overlap_yrs))
CON_time_copy = CON_time[ind_con]
con_gm_copy = np.copy(con_gm[ind_con])
xvar = list(range(len(ind_con)))
con_slope, con_intercept, r_value, p_value, std_err = stats.linregress(xvar,con_gm_copy)
con_trend = con_slope*np.squeeze(xvar) + con_intercept
con_gm_detrend = con_gm_copy - con_trend
# r and ce on full data
full_err  = lmr_gm_copy[ind_lmr] - con_gm_copy
lmr_con_corr_full = np.corrcoef(lmr_gm_copy[ind_lmr],con_gm[ind_con])
lmr_con_ce_full = coefficient_efficiency(con_gm_copy,lmr_gm_copy[ind_lmr])
lconrf =  str(float('%.2f' % lmr_con_corr_full[0,1]))
lconcf =  str(float('%.2f' % lmr_con_ce_full))
# r and ce on detrended data
lmr_con_corr_detrend = np.corrcoef(lmr_gm_detrend[ind_lmr],con_gm_detrend)
lmr_detrend_err = lmr_gm_detrend[ind_lmr] - con_gm_detrend
lmr_con_ce_detrend = coefficient_efficiency(con_gm_detrend,lmr_gm_detrend[ind_lmr])
lconrd   =  str(float('%.2f' % lmr_con_corr_detrend[0,1]))
lconcd   =  str(float('%.2f' % lmr_con_ce_detrend))

# Trends
lmrs   =  str(float('%.2f' % (lmr_slope*100.)))
gs     =  str(float('%.2f' % (gis_slope*100.)))
crus   =  str(float('%.2f' % (cru_slope*100.)))
bes    =  str(float('%.2f' % (be_slope*100.)))
mlosts =  str(float('%.2f' % (mlost_slope*100.)))
tcrs   =  str(float('%.2f' % (tcr_slope*100.)))
eras   =  str(float('%.2f' % (era_slope*100.)))
cons   =  str(float('%.2f' % (con_slope*100.)))

print('r:  %s %s' % (str(lgrf), str(lgrd)))
print('ce: %s %s' % (str(lgcf), str(lgcd)))

print('LMR trend: '+str(lmrs) + ' K/100yrs')
# plots

if iplot:
    lw = 2
    # LMR
    fig = plt.figure()
    #plt.plot(LMR_time_copy,lmr_trend,'k-',lw=lw*2)
    plt.plot(LMR_time_copy,lmr_gm_detrend,'k-',lw=lw*2,label='LMR (trend: '+lmrs+' K/100yrs)')
    # GIS
    #plt.plot(GIS_time_copy,gis_trend,'r-',lw=lw,alpha=alpha)
    plt.plot(GIS_time_copy,gis_gm_detrend,'r-',lw=lw,alpha=alpha,label='GISTEMP (trend: '+gs+' K/100yrs)')
    # CRU
    #plt.plot(CRU_time_copy,cru_trend,'m-',lw=lw,alpha=alpha)
    plt.plot(CRU_time_copy,cru_gm_detrend,'m-',lw=lw,alpha=alpha,label='HadCRUT4 (trend: '+crus+' K/100yrs)')
    # BE
    #plt.plot(BE_time_copy,be_trend,'g-',lw=lw,alpha=alpha)
    plt.plot(BE_time_copy,be_gm_detrend,'g-',lw=lw,alpha=alpha,label='BE (trend: '+bes+' K/100yrs)')
    # MLOST
    #plt.plot(MLOST_time_copy,mlost_trend,'c-',lw=lw,alpha=alpha)
    plt.plot(MLOST_time_copy,mlost_gm_detrend,'c-',lw=lw,alpha=alpha,label='MLOST (trend: '+mlosts+' K/100yrs)')
    # TCR
    #plt.plot(TCR_time_copy,tcr_trend,'y-',lw=lw,alpha=alpha)
    plt.plot(TCR_time_copy,tcr_gm_detrend,'y-',lw=lw,alpha=alpha,label='20CR-V2 (trend: '+tcrs+' K/100yrs)')
    # ERA
    #plt.plot(ERA_time_copy,era_trend,'b-',lw=lw,alpha=alpha)
    plt.plot(ERA_time_copy,era_gm_detrend,'b-',lw=lw,alpha=alpha,label='ERA-20C (trend: '+eras+' K/100yrs)')
    # CONsensus
    #plt.plot(CON_time_copy,con_trend,color='lime',lw=lw,alpha=alpha)
    plt.plot(CON_time_copy,con_gm_detrend,color='lime',lw=lw*2,alpha=alpha,label='consensus (trend: '+cons+' K/100yrs)')
    
    plt.ylim(-1,1)    
    plt.legend(loc=2,fontsize=12)

    # add to figure
    #plt.title('Detrended global mean temperature \n(' + nexp + ')',weight='bold',y=1.03)
    plt.title('Detrended global mean temperature',weight='bold',y=1.03)
    plt.xlabel('Year CE',fontweight='bold')
    plt.ylabel('Temperature anomaly (K)',fontweight='bold')
    xl_loc = [stime,etime]
    yl_loc = [-.6,.7]
    plt.xlim(xl_loc)
    plt.ylim(yl_loc)
    txl = xl_loc[0] + (xl_loc[1]-xl_loc[0])*.005
    tyl = yl_loc[0] + (yl_loc[1]-yl_loc[0])*.15
    
    off = .03
    plt.text(txl,tyl,      '(LMR,GISTEMP) : r full= ' + lgrf.ljust(4,' ') + ' r detrend= ' + lgrd.ljust(4,' ') + ' CE full= ' + lgcf.ljust(5,' ') + ' CE detrend= ' + lgcd.ljust(5,' '), fontsize=12, family='monospace')
    plt.text(txl,tyl-off,  '(LMR,HadCRUT4): r full= ' + lcrf.ljust(4,' ') + ' r detrend= ' + lcrd.ljust(4,' ') + ' CE full= ' + lccf.ljust(5,' ') + ' CE detrend= ' + lccd.ljust(5,' '), fontsize=12, family='monospace')
    plt.text(txl,tyl-2*off,'(LMR,BE)      : r full= ' + lbrf.ljust(4,' ') + ' r detrend= ' + lbrd.ljust(4,' ') + ' CE full= ' + lbcf.ljust(5,' ') + ' CE detrend= ' + lbcd.ljust(5,' '), fontsize=12, family='monospace')
    plt.text(txl,tyl-3*off,'(LMR,MLOST)   : r full= ' + lmrf.ljust(4,' ') + ' r detrend= ' + lmrd.ljust(4,' ') + ' CE full= ' + lmcf.ljust(5,' ') + ' CE detrend= ' + lmcd.ljust(5,' '), fontsize=12, family='monospace')
    plt.text(txl,tyl-4*off,'(LMR,20CR-V2) : r full= ' + ltrf.ljust(4,' ') + ' r detrend= ' + ltrd.ljust(4,' ') + ' CE full= ' + ltcf.ljust(5,' ') + ' CE detrend= ' + ltcd.ljust(5,' '), fontsize=12, family='monospace')
    plt.text(txl,tyl-5*off,'(LMR,ERA-20C) : r full= ' + lerf.ljust(4,' ') + ' r detrend= ' + lerd.ljust(4,' ') + ' CE full= ' + lecf.ljust(5,' ') + ' CE detrend= ' + lecd.ljust(5,' '), fontsize=12, family='monospace')
    plt.text(txl,tyl-6*off,'(LMR,consens.): r full= ' + lconrf.ljust(4,' ') + ' r detrend= ' + lconrd.ljust(4,' ') + ' CE full= ' + lconcf.ljust(5,' ') + ' CE detrend= ' + lconcd.ljust(5,' '), fontsize=12, family='monospace')
    
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_'+'detrended.png')
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_'+'detrended.pdf',bbox_inches='tight', dpi=300, format='pdf')

"""
# rank histograms
# loop over all years; send ensemble and a verification value
print(' ')
print(np.shape(gmt_save))
print(lmr_smatch)
print(len(lmr_gm_copy))
rank = []
for yr in range(len(lmr_gm_copy)):
    rankval = rank_histogram(gmt_save[lmr_smatch+yr:lmr_smatch+yr+1,:,:],gis_gm_copy[yr])
    rank.append(rankval)
    
if iplot:
    fig = plt.figure()
    nbins = 10
    plt.hist(rank,nbins)
    if fsave:
        fname = nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_rank_histogram.png'
        print(fname)
        plt.savefig(fname)
"""

#  Summary "table" figures

# dataset catalog IN ORDER
dset = ['LMR', 'GISTEMP', 'HadCRUT4', '20CR-V2', 'BE', 'MLOST', 'ERA-20C', 'CON']
#dset = ['LMR', 'GISTEMP', 'HadCRUT4', '20CR-V2', 'BE', 'MLOST', 'CON']
ndset = len(dset)

# construct a master array with each dataset in a column in the order of dset
nyrs = (etime - stime)+1
verif_yrs = np.arange(stime,etime+1,1)
ALL_array = np.zeros([nyrs,ndset])

# define padded arrays to handle possible missing data
lmr_gm_pad   = np.zeros(shape=[nyrs])
gis_gm_pad   = np.zeros(shape=[nyrs])
cru_gm_pad   = np.zeros(shape=[nyrs])
be_gm_pad    = np.zeros(shape=[nyrs])
mlost_gm_pad = np.zeros(shape=[nyrs])
tcr_gm_pad   = np.zeros(shape=[nyrs])
era_gm_pad   = np.zeros(shape=[nyrs])
con_gm_pad   = np.zeros(shape=[nyrs])

# fill with missing values (nan)
lmr_gm_pad[:]   = np.nan
gis_gm_pad[:]   = np.nan
cru_gm_pad[:]   = np.nan
be_gm_pad[:]    = np.nan
mlost_gm_pad[:] = np.nan
tcr_gm_pad[:]   = np.nan
era_gm_pad[:]   = np.nan
con_gm_pad[:]   = np.nan


ind_lmr   = np.searchsorted(LMR_time,   np.intersect1d(LMR_time, verif_yrs))
ind_gis   = np.searchsorted(GIS_time,   np.intersect1d(GIS_time, verif_yrs))
ind_cru   = np.searchsorted(CRU_time,   np.intersect1d(CRU_time, verif_yrs))
ind_be    = np.searchsorted(BE_time,    np.intersect1d(BE_time, verif_yrs))
ind_mlost = np.searchsorted(MLOST_time, np.intersect1d(MLOST_time, verif_yrs))
ind_tcr   = np.searchsorted(TCR_time,   np.intersect1d(TCR_time, verif_yrs))
ind_era   = np.searchsorted(ERA20C_time,np.intersect1d(ERA20C_time, verif_yrs))
ind_con   = np.searchsorted(CON_time,   np.intersect1d(CON_time, verif_yrs))


ind_ver = np.searchsorted(verif_yrs,   np.intersect1d(LMR_time[ind_lmr], verif_yrs))
lmr_gm_pad[ind_ver]     = lmr_gm[ind_lmr]
ind_ver = np.searchsorted(verif_yrs,   np.intersect1d(GIS_time[ind_gis], verif_yrs))
gis_gm_pad[ind_ver]     = gis_gm[ind_gis]
ind_ver = np.searchsorted(verif_yrs,   np.intersect1d(CRU_time[ind_cru], verif_yrs))
cru_gm_pad[ind_ver]     = cru_gm[ind_cru]
ind_ver = np.searchsorted(verif_yrs,   np.intersect1d(BE_time[ind_be], verif_yrs))
be_gm_pad[ind_ver]      = be_gm[ind_be]
ind_ver = np.searchsorted(verif_yrs,   np.intersect1d(MLOST_time[ind_mlost], verif_yrs))
mlost_gm_pad[ind_ver]   = mlost_gm[ind_mlost]
ind_ver = np.searchsorted(verif_yrs,   np.intersect1d(TCR_time[ind_tcr], verif_yrs))
tcr_gm_pad[ind_ver]     = tcr_gm[ind_tcr]
ind_ver = np.searchsorted(verif_yrs,   np.intersect1d(ERA20C_time[ind_era], verif_yrs))
era_gm_pad[ind_ver]     = era_gm[ind_era]
ind_ver = np.searchsorted(verif_yrs,   np.intersect1d(CON_time[ind_con], verif_yrs))
con_gm_pad[ind_ver]     = con_gm[ind_con]


k = 0;  ALL_array[:,k] = lmr_gm_pad
k += 1; ALL_array[:,k] = gis_gm_pad
k += 1; ALL_array[:,k] = cru_gm_pad
k += 1; ALL_array[:,k] = tcr_gm_pad
k += 1; ALL_array[:,k] = be_gm_pad
k += 1; ALL_array[:,k] = mlost_gm_pad
k += 1; ALL_array[:,k] = era_gm_pad
k += 1; ALL_array[:,k] = con_gm_pad

#
# correlation coefficients for a chosen time interval 
#

# get ALL_array in a pandas data frame -> pandas is nan-friendly for calculations of correlation
df = pd.DataFrame(ALL_array,columns=dset)
#corr_matrix = df.corr()
corr_matrix = df.corr().as_matrix()


#
# coefficient of efficiency
#

CE_matrix = np.zeros([ndset,ndset])

for i in range(ndset): # verification dataset
    for j in range(ndset): # test dataset that is verified
        ref = ALL_array[:,i]
        test = ALL_array[:,j]
        CE_matrix[i,j] = coefficient_efficiency(ref,test,valid=0.5)


##################################
#----------- plotting starts here:
##################################

plt.figure()

# make sure this matches what is in the plt.table call below
#cmap = plt.cm.Reds
cmap = truncate_colormap(plt.cm.Reds,0.0,0.9)

#cellsize = 0.2 # table cell size
cellsize = 0.19 # table cell size
fontsize = 14

nticks = 11

# cell padding for the row labels; not sure why this is needed, but it is
lpad = ' '
#rpad = lpad + ' '
rpad = lpad
idx = []
for d in dset:
    idx.append(lpad+d+rpad)

#tempvals = np.random.randn(ndset,ndset) # random data for testing
#df = pandas.DataFrame(randn(ndset, ndset), index=idx, columns=dset) # keep this example of how to do it with pandas
#vals = np.around(df.values,2)
# without pandas...
#vals = np.around(tempvals,2)

vals = np.around(corr_matrix,2) # round to two decimal places

# set max and min values for color range
vmax = np.max(np.abs(vals))
vmax = 1.0
vmin = 0.75

# this is here just to get the colorbar; apparently plt.table has no association
img = plt.imshow(vals, cmap=cmap, vmin = vmin, vmax =vmax)
cbar = plt.colorbar(img,shrink=.65, pad = .4)
cbar.ax.tick_params(labelsize=fontsize)
tick_locator = ticker.MaxNLocator(nbins=nticks)
cbar.locator = tick_locator
cbar.ax.yaxis.set_major_locator(ticker.AutoLocator())
cbar.update_ticks()
img.set_visible(False)

# normalize on the range of the colormap so that the cell colors match the colorbar
normal = plt.Normalize(vmin, vmax)
newvals = normal(vals)

# make the table
# using pandas...
#the_table=plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns,loc='center',cellColours=plt.cm.bwr(newvals))
# not using pandas...
# colors...

#the_table=plt.table(cellText=vals, rowLabels=idx, colLabels=dset,loc='center',cellColours=plt.cm.Reds(newvals))
the_table=plt.table(cellText=vals, rowLabels=idx, colLabels=dset,loc='center',cellColours=cmap(newvals))

# no colors...
#the_table=plt.table(cellText=vals, rowLabels=idx, colLabels=dset,loc='center')

# adjust font and cell size
the_table.auto_set_font_size(False)
the_table.set_fontsize(fontsize)
table_props = the_table.properties()
table_cells = table_props['child_artists']
for cell in table_cells: cell.set_height(cellsize)
for cell in table_cells: cell.set_width(cellsize*1.8)

plt.axis('off') # remove the axes that came with imshow
#plt.title('Correlation',fontweight='bold',fontsize=18, y=1.2)

fname =  nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_corr_table'
#plt.savefig(fname+'.png')
plt.savefig(fname+'.pdf',format='pdf',dpi=300,bbox_inches='tight')


#
# CE table
#

plt.figure()

vals = np.around(CE_matrix,2)

# set max and min values for color range
vmax = np.max(np.abs(vals))
vmin = 0.75

# this is here just to get the colorbar; apparently plt.table has no association
img = plt.imshow(vals, cmap=cmap, vmin = vmin, vmax =vmax)
cbar = plt.colorbar(img,shrink=.65, pad = .4)
cbar.ax.tick_params(labelsize=fontsize)
tick_locator = ticker.MaxNLocator(nbins=nticks)
cbar.locator = tick_locator
cbar.ax.yaxis.set_major_locator(ticker.AutoLocator())
cbar.update_ticks()
img.set_visible(False)

# normalize on the range of the colormap so that the cell colors match the colorbar
normal = plt.Normalize(vmin, vmax)
newvals = normal(vals)

# make the table
#the_table=plt.table(cellText=vals, rowLabels=idx, colLabels=dset,loc='center',cellColours=plt.cm.Reds(newvals))
the_table=plt.table(cellText=vals, rowLabels=idx, colLabels=dset,loc='center',cellColours=cmap(newvals))

# adjust font and cell size
the_table.auto_set_font_size(False)
the_table.set_fontsize(fontsize)
table_props = the_table.properties()
table_cells = table_props['child_artists']
for cell in table_cells: cell.set_height(cellsize)
for cell in table_cells: cell.set_width(cellsize*1.8)

plt.axis('off') # remove the axes that came with imshow
#plt.title('Coefficient of efficiency',fontweight='bold', fontsize=18, y=1.2)

fname =  nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_ce_table'
#plt.savefig(fname+'.png')
plt.savefig(fname+'.pdf',format='pdf',dpi=300,bbox_inches='tight')

    
#
# NEW 9/15/16 dictionary for objective verification
#
if stat_save:
    gmt_verification_stats = {}
    stat_vars = ['stime','etime',
                 'ltc','lec','lgc','lcc','lbc','lmc','loc',
                 'ltce','lece','lgce','lcce','lbce','lmce','loce',
                 'lgrd','lgcd', 'lconcd','lconrd','lcrd','lccd','lbrd','lbcd','lmrd','lmcd','ltrd','ltcd','lerd','lecd',
                 'lmrs','gs','crus','bes','mlosts','tcrs','eras']

    stat_metadata = {'stime':"starting year of verification time period",
                     'etime':"ending year of verification time period",
                     'ltc':'LMR_TCR correlation',
                     'lec':'LMR_ERA correlation',
                     'lgc':'LMR_GIS correlation',
                     'lcc':'LMR_CRU correlation',
                     'lbc':'LMR_BE correlation',
                     'lmc':'LMR_MLOST correlation',
                     'loc':'LMR_consensus correlation',
                     'ltce':'LMR_TCR coefficient of efficiency',
                     'lece':'LMR_ERA coefficient of efficiency',
                     'lgce':'LMR_GIS coefficient of efficiency',
                     'lcce':'LMR_CRU coefficient of efficiency',
                     'lbce':'LMR_BE coefficient of efficiency',
                     'lmce':'LMR_MLOST coefficient of efficiency',
                     'loce':'LMR_consensus coefficient of efficiency',
                     'ltrd':'LMR_TCR detrended correlation',
                     'lerd':'LMR_ERA detrended correlation',
                     'lgrd':'LMR_GIS detrended correlation',
                     'lcrd':'LMR_CRU detrended correlation',
                     'lbrd':'LMR_BE detrended correlation',
                     'lmrd':'LMR_MLOST detrended correlation',
                     'lconrd':'LMR_consensus detrended correlation',
                     'ltcd':'LMR_TCR detrended coefficient of efficiency',
                     'lecd':'LMR_ERA detrended coefficient of efficiency',
                     'lgcd':'LMR_GIS detrended coefficient of efficiency',
                     'lccd':'LMR_CRU detrended coefficient of efficiency',
                     'lbcd':'LMR_BE detrended coefficient of efficiency',
                     'lmcd':'LMR_MLOST detrended coefficient of efficiency',
                     'lconcd':'LMR_consensus detrended coefficient of efficiency',
                     'lmrs':'LMR trend (K/100 years)',
                     'gs':'GIS trend (K/100 years)',
                     'crus':'CRU trend (K/100 years)',
                     'bes':'BE trend (K/100 years)',
                     'mlosts':'MLOST trend (K/100 years)',
                     'tcrs':'TCR trend (K/100 years)',
                     'eras':'ERA trend (K/100 years)',
                     'stat_metadata':'metdata'
                     }

    for var in stat_vars:
        gmt_verification_stats[var] = locals()[var]

    gmt_verification_stats['stat_metadata'] = stat_metadata
    # dump the dictionary to a pickle file
    spfile = nexp + '_' + str(niters) + '_iters_gmt_verification.pckl'
    print('writing statistics to pickle file: ' + spfile)
    outfile = open(spfile, 'w')
    pickle.dump(gmt_verification_stats, outfile)

if interactive:
    plt.show(block=True)
