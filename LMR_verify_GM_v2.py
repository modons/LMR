
# verify statistics related to the global-mean 2m air temperature
#
# started from LMR_plots.py r-86

import matplotlib
# need to do this when running remotely, and to suppress figures
#matplotlib.use('Agg')

import csv
import glob, os, fnmatch
import numpy as np
import mpl_toolkits.basemap as bm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from scipy import stats
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pandas as pd
#
from LMR_plot_support import *
from LMR_utils2 import global_hemispheric_means, assimilated_proxies, coefficient_efficiency, rank_histogram
from load_gridded_data import read_gridded_data_GISTEMP
from load_gridded_data import read_gridded_data_HadCRUT
from load_gridded_data import read_gridded_data_BerkeleyEarth
from load_gridded_data import read_gridded_data_CMIP5_model
from LMR_plot_support import *

# =============================================================================
def truncate_colormap(cmap, minval=0.0,maxval=1.0,n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name,a=minval,b=maxval),
        cmap(np.linspace(minval,maxval,n)))
    return new_cmap
# =============================================================================


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

# option to save figures to a file
#fsave = True
fsave = False

# file specification
#
# current datasets
#
#nexp = 'production_gis_ccsm4_pagesall_0.75'
#nexp = 'testdev_paramsearch_noxbblend_a7_d0_100itr'
nexp = 'testdev_gis_ccsm4_posterior_lim'

# specify directories for LMR and calibration data
#datadir_output = '/home/disk/kalman3/rtardif/LMR/output'
#datadir_output = './data/'
#datadir_output = '/home/disk/kalman2/wperkins/LMR_output/archive'
datadir_output = '/home/disk/kalman2/wperkins/LMR_output/testing'

datadir_calib = '/home/disk/kalman3/rtardif/LMR/data/analyses'

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

print '--------------------------------------------------'
print 'verification of global-mean 2m air temperature'
print '--------------------------------------------------'

workdir = datadir_output + '/' + nexp

# get directory and information for later use

print '--------------------------------------------------'
print 'working directory: '+workdir
print '--------------------------------------------------'

# get a listing of the iteration directories
dirs = glob.glob(workdir+"/r*")
# sorted
dirs.sort()


# query file for assimilated proxy information (for now, ONLY IN THE r0 directory!)

ptypes,nrecords = assimilated_proxies(workdir+'/r0/')

print '--------------------------------------------------'
print 'Assimilated proxies by type:'
for pt in ptypes.keys():
    print pt + ': ' + str(ptypes[pt])
                
print 'Total: ' + str(nrecords)
print '--------------------------------------------------'

# ==========================================
# load GISTEMP, HadCRU, BerkeleyEarth, MLOST
# ==========================================

# load GISTEMP
datafile_calib   = 'gistemp1200_ERSST.nc'
calib_vars = ['Tsfc']
[GIS_time,GIS_lat,GIS_lon,GIS_anomaly] = read_gridded_data_GISTEMP(datadir_calib,datafile_calib,calib_vars)
nlat_GIS = len(GIS_lat)
nlon_GIS = len(GIS_lon)

# load HadCRU
datafile_calib   = 'HadCRUT.4.3.0.0.median.nc'
calib_vars = ['Tsfc']
[CRU_time,CRU_lat,CRU_lon,CRU_anomaly] = read_gridded_data_HadCRUT(datadir_calib,datafile_calib,calib_vars)

## use GMT time series computed by Hadley Centre instead !!!!!!!!!!!!
#datafile_calib = '/home/disk/ekman/rtardif/kalman3/LMR/data/analyses/HadCRUT/HadCRUT.4.4.0.0.annual_ns_avg.txt'
#data = np.loadtxt(datafile_calib, usecols = (0,1))
#CRU_time = data[:,0].astype(np.int64)
#cru_gm   = data[:,1]

# load BerkeleyEarth
datafile_calib   = 'Land_and_Ocean_LatLong1.nc'
calib_vars = ['Tsfc']
[BE_time,BE_lat,BE_lon,BE_anomaly] = read_gridded_data_BerkeleyEarth(datadir_calib,datafile_calib,calib_vars)

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

datadir = '/home/disk/kalman3/rtardif/LMR/data/model/era20c'
datafile = 'tas_sfc_Amon_ERA20C_190001-201212.nc'
vardef = 'tas_sfc_Amon'

dd = read_gridded_data_CMIP5_model(datadir,datafile,[vardef])

ERA20C_time = dd[vardef]['years']
lat_ERA20C = dd[vardef]['lat']
lon_ERA20C = dd[vardef]['lon']
nlat_ERA20C = len(lat_ERA20C)
nlon_ERA20C = len(lon_ERA20C)
ERA20C = dd[vardef]['value'] + dd[vardef]['climo'] # Full field (long-term mean NOT REMOVED)
#ERA20C = dd[vardef]['value']                      # Anomalies (long-term mean REMOVED)

# compute and remove the mean over 1951-1980 reference period as w/ GIS & BE
smatch, ematch = find_date_indices(ERA20C_time,1951,1980)
ref_mean_era = np.mean(ERA20C[smatch:ematch,:,:],axis=0)
ERA20C = ERA20C - ref_mean_era

era_gm = np.zeros([len(ERA20C_time)])
era_nhm = np.zeros([len(ERA20C_time)])
era_shm = np.zeros([len(ERA20C_time)])
# Loop over years in dataset
for i in xrange(0,len(ERA20C_time)): 
    # compute the global & hemispheric mean temperature
    [era_gm[i],era_nhm[i],era_shm[i]] = global_hemispheric_means(ERA20C[i,:,:],lat_ERA20C)


# load 20th century reanalysis (TCR) reanalysis --------------------------------
datadir = '/home/disk/kalman3/rtardif/LMR/data/model/20cr'
datafile = 'tas_sfc_Amon_20CR_185101-201112.nc'
vardef = 'tas_sfc_Amon'

dd = read_gridded_data_CMIP5_model(datadir,datafile,[vardef])

TCR_time = dd[vardef]['years']
lat_TCR = dd[vardef]['lat']
lon_TCR = dd[vardef]['lon']
nlat_TCR = len(lat_TCR)
nlon_TCR = len(lon_TCR)
TCR = dd[vardef]['value'] + dd[vardef]['climo'] # Full field (long-term mean NOT REMOVED)
#TCR = dd[vardef]['value']                      # Anomalies (long-term mean REMOVED)

# compute and remove the mean over 1951-1980 reference period as w/ GIS & BE
smatch, ematch = find_date_indices(TCR_time,1951,1980)
ref_mean_tcr = np.mean(TCR[smatch:ematch,:,:],axis=0)
TCR = TCR - ref_mean_tcr

tcr_gm = np.zeros([len(TCR_time)])
tcr_nhm = np.zeros([len(TCR_time)])
tcr_shm = np.zeros([len(TCR_time)])
# Loop over years in dataset
for i in xrange(0,len(TCR_time)): 
    # compute the global & hemispheric mean temperature
    [tcr_gm[i],tcr_nhm[i],tcr_shm[i]] = global_hemispheric_means(TCR[i,:,:],lat_TCR)


#
# read LMR GMT data computed during DA
#

print '--------------------------------------------------'
print 'reading LMR GMT data...'
print '--------------------------------------------------'
kk = -1
if iplot:
    fig = plt.figure()

first = True
kk = -1
# one experiment only

# use all of the directories found from scanning the disk
dirset = dirs
# use a custom selection
#dirset = dirs[0:1]
niters = len(dirset)

print '--------------------------------------------------'
print 'niters = ' + str(niters)
print '--------------------------------------------------'

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
    print gmtpfile
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
CON_time = range(stime,etime)


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
print '--------------------------------------------------'
print 'annual-mean correlations: '
print 'LMR_TCR   correlation: ' + ltc
print 'LMR_ERA   correlation: ' + lec
print 'LMR_GIS   correlation: ' + lgc
print 'LMR_CRU   correlation: ' + lcc
print 'LMR_BE    correlation: ' + lbc
print 'LMR_MLOST correlation: ' + lmc
print 'GIS_TCR   correlation: ' + gtc
print 'GIS_ERA   correlation: ' + gec
print 'GIS_CRU   correlation: ' + gcc
print 'GIS_BE    correlation: ' + gbc
print 'LMR_consensus correlation: ' + loc
print '--------------------------------------------------'


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
print '--------------------------------------------------'
print 'coefficient of efficiency: '
print 'LMR-TCR CE  : ' + str(ltce)
print 'LMR-ERA CE  : ' + str(lece)
print 'LMR-GIS CE  : ' + str(lgce)
print 'LMR-CRU CE  : ' + str(lcce)
print 'LMR-BE CE   : ' + str(lbce)
print 'LMR-MLOST CE: ' + str(lmce)
print 'LMR-CON CE  : ' + str(loce)
print 'TCR-GIS CE  : ' + str(tgce)
print 'ERA-GIS CE  : ' + str(egce)
print 'BE-CRU CE   : ' + str(bgce)
print 'GIS-CRU CE  : ' + str(cgce)
print '--------------------------------------------------'

#
# spread--error
#
lg_err = lmr_gm[lmr_smatch:lmr_ematch] - gis_gm[gis_smatch:gis_ematch]
svar = gmt_save[:,lmr_smatch:lmr_ematch].var(0,ddof=1)
calib = lg_err.var(0,ddof=1)/svar.mean(0)
print '--------------------------------------------------'
print 'ensemble calibration: ' + str(calib)
print '--------------------------------------------------'

# ========================================================
# plots
# ========================================================
if iplot:
    lw = 2
    fig = plt.figure()
    plt.plot(LMR_time,lmr_gm,'k-'    ,linewidth=lw*2,label='LMR')
    plt.plot(GIS_time,gis_gm,'r-'    ,linewidth=lw,label='GIS',alpha=alpha)
    plt.plot(CRU_time,cru_gm,'m-'    ,linewidth=lw,label='CRU',alpha=alpha)
    plt.plot(BE_time,be_gm,'g-'      ,linewidth=lw,label='BE',alpha=alpha)
    plt.plot(MLOST_time,mlost_gm,'c-',linewidth=lw,label='MLOST',alpha=alpha)
    plt.plot(TCR_time,tcr_gm,'y-'    ,linewidth=lw,label='TCR',alpha=alpha)
    plt.plot(ERA20C_time,era_gm,'b-' ,linewidth=lw,label='ERA',alpha=alpha)
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

    plt.text(txl,tyl,'(LMR,GIS)      : r= ' + lgc.ljust(5,' ') + ' CE= ' + lgce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,CRU)      : r= ' + lcc.ljust(5,' ') + ' CE= ' + lcce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,BE)       : r= ' + lbc.ljust(5,' ') + ' CE= ' + lbce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,MLOST)    : r= ' + lmc.ljust(5,' ') + ' CE= ' + lmce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,TCR)      : r= ' + ltc.ljust(5,' ') + ' CE= ' + ltce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,ERA)      : r= ' + lec.ljust(5,' ') + ' CE= ' + lece.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,consensus): r= ' + loc.ljust(5,' ') + ' CE= ' + loce.ljust(5,' '), fontsize=14, family='monospace')

    plt.plot(xl_loc,[0,0],color='gray',linestyle=':',lw=2)
    plt.legend(loc=2)

    if fsave:
        print 'saving to .png'
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_annual.png')
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_annual.pdf',bbox_inches='tight', dpi=300, format='pdf')

    plt.show()
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


# LMR-concensus
overlap_yrs = np.intersect1d(np.intersect1d(LMR_smoothed_years, CON_smoothed_years), verif_yrs)
ind_lmr = np.searchsorted(LMR_smoothed_years, np.intersect1d(LMR_smoothed_years, overlap_yrs))
ind_con = np.searchsorted(CON_smoothed_years, np.intersect1d(CON_smoothed_years, overlap_yrs))
ls_con_corr = np.corrcoef(LMR_smoothed[ind_lmr],CON_smoothed[ind_con])
ls_con_ce = coefficient_efficiency(CON_smoothed[ind_con],LMR_smoothed[ind_lmr])
lmr_con_corr = np.corrcoef(lmr_gm[ind_lmr],con_gm[ind_con])
lmr_con_ce = coefficient_efficiency(con_gm[ind_con],lmr_gm[ind_lmr])

lsconsc  = str(float('%.2f' % ls_con_corr[0,1]))
lsconsce = str(float('%.2f' % ls_con_ce))


print '--------------------------------------------------'
print str(nsyrs)+'-year-smoothed correlations...'
print 'smoothed lmr-gis correlation =   ' + lsgsc
print 'smoothed lmr-cru correlation =   ' + lscsc
print 'smoothed lmr-be correlation =    ' + lsbsc
print 'smoothed lmr-mlost correlation = ' + lsmsc
print 'smoothed lmr-tcr correlation =   ' + lstsc
print 'smoothed lmr-era correlation =   ' + lsesc
print 'smoothed lmr-con correlation =   ' + lsconsc
print '--------------------------------------------------'
print '--------------------------------------------------'
print str(nsyrs)+'-year-smoothed CE...'
print 'smoothed lmr-gis CE =   ' + lsgsce
print 'smoothed lmr-cru CE =   ' + lscsce
print 'smoothed lmr-be CE =    ' + lsbsce
print 'smoothed lmr-mlost CE = ' + lsmsce
print 'smoothed lmr-tcr CE =   ' + lstsce
print 'smoothed lmr-era CE =   ' + lsesce
print 'smoothed lmr-con CE =   ' + lsconsce
print '--------------------------------------------------'


if iplot:
    fig = plt.figure()
    #plt.plot(recon_times,lmr_gm,'k-',linewidth=2)
    plt.fill_between(recon_times,gmt_min,gmt_max,facecolor='gray',alpha=alpha,linewidth=0.)

    # add smoothed lines
    plt.plot(LMR_smoothed_years,LMR_smoothed,'k-'    ,linewidth=4, label='LMR')
    plt.plot(GIS_smoothed_years,GIS_smoothed,'r-'    ,linewidth=4, label='GIS',alpha=alpha)
    plt.plot(CRU_smoothed_years,CRU_smoothed,'m-'    ,linewidth=4, label='CRU',alpha=alpha)
    plt.plot(BE_smoothed_years,BE_smoothed,'g-'      ,linewidth=4, label='BE',alpha=alpha)
    plt.plot(MLOST_smoothed_years,MLOST_smoothed,'c-',linewidth=4, label='MLOST',alpha=alpha)
    plt.plot(TCR_smoothed_years,TCR_smoothed,'y-'    ,linewidth=4, label='TCR',alpha=alpha)
    plt.plot(ERA_smoothed_years,ERA_smoothed,'b-'    ,linewidth=4, label='ERA',alpha=alpha)
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

    plt.text(txl,tyl,'(LMR,GIS)      : r= ' + lsgsc.ljust(5,' ') + ' CE= ' + lsgsce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-0.05
    plt.text(txl,tyl,'(LMR,CRU)      : r= ' + lscsc.ljust(5,' ') + ' CE= ' + lscsce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-0.05
    plt.text(txl,tyl,'(LMR,BE)       : r= ' + lsbsc.ljust(5,' ') + ' CE= ' + lsbsce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-0.05
    plt.text(txl,tyl,'(LMR,MLOST)    : r= ' + lsmsc.ljust(5,' ') + ' CE= ' + lsmsce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-0.05
    plt.text(txl,tyl,'(LMR,TCR)      : r= ' + lstsc.ljust(5,' ') + ' CE= ' + lstsce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-0.05
    plt.text(txl,tyl,'(LMR,ERA)      : r= ' + lsesc.ljust(5,' ') + ' CE= ' + lsesce.ljust(5,' '), fontsize=14, family='monospace')
    #tyl = tyl-0.05
    #plt.text(txl,tyl,'(LMR,consensus): r= ' + lsosc.ljust(5,' ') + ' CE= ' + lsosce.ljust(5,' '), fontsize=14, family='monospace')

    plt.plot(xl_loc,[0,0],color='gray',linestyle=':',lw=2)
    plt.legend(loc=2)

    if fsave:
        print 'saving to .png'
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_'+str(nsyrs)+'yr_smoothed.png')
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_'+str(nsyrs)+'yr_smoothed.pdf',bbox_inches='tight', dpi=300, format='pdf')        

    plt.show()

# =======================================
# detrend and verify the detrended signal
# =======================================

print '--------------------------------------------------'
print 'verification of detrended data'
print '--------------------------------------------------'

verif_yrs = np.arange(stime,etime+1,1)

# for LMR
ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, verif_yrs))
# save copies of the original data for residual estimates later
lmr_gm_copy = np.copy(lmr_gm[ind_lmr])
LMR_time_copy = np.copy(LMR_time[ind_lmr])
xvar = range(len(lmr_gm_copy))
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
xvar = range(len(ind_gis))
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
print 'correlaton between trend and detrend errors = ' + str(check[0,1])
print 'check error variances...'
print 'trend error: ' + str(np.var(error_trend))
print 'detrend error: ' + str(np.var(error_detrend))
print 'detrend error + trend error: ' + str(np.var(error_trend)+np.var(error_detrend))
print 'full error : ' + str(np.var(error_trend+error_detrend))

# for CRU
# overlaping years within verification interval
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time_copy, CRU_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time_copy, np.intersect1d(LMR_time_copy, overlap_yrs))
ind_cru = np.searchsorted(CRU_time, np.intersect1d(CRU_time, overlap_yrs))
CRU_time_copy = CRU_time[ind_cru]
cru_gm_copy = np.copy(cru_gm[ind_cru])
xvar = range(len(ind_cru))
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
xvar = range(len(ind_be))
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
xvar = range(len(ind_mlost))
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
xvar = range(len(ind_tcr))
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
xvar = range(len(ind_era))
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

# Trends
lmrs   =  str(float('%.2f' % (lmr_slope*100.)))
gs     =  str(float('%.2f' % (gis_slope*100.)))
crus   =  str(float('%.2f' % (cru_slope*100.)))
bes    =  str(float('%.2f' % (be_slope*100.)))
mlosts =  str(float('%.2f' % (mlost_slope*100.)))
tcrs   =  str(float('%.2f' % (tcr_slope*100.)))
eras   =  str(float('%.2f' % (era_slope*100.)))

print 'r:  ' + str(lgrf) + ' ' + str(lgrd)
print 'ce: ' + str(lgcf) + ' ' + str(lgcd)

# plots

if iplot:
    lw = 2
    # LMR & GIS
    fig = plt.figure()
    plt.plot(LMR_time_copy,lmr_trend,'k-',lw=lw*2,label='LMR (trend: '+lmrs+' K/100yrs)')
    plt.plot(LMR_time_copy,lmr_gm_detrend,'k-',lw=lw*2)

    plt.plot(GIS_time_copy,gis_trend,'r-',lw=lw,label='GIS  (trend: '+gs+' K/100yrs)',alpha=alpha)
    plt.plot(GIS_time_copy,gis_gm_detrend,'r-',lw=lw,alpha=alpha)

    plt.plot(CRU_time_copy,cru_trend,'m-',lw=lw,label='CRU (trend: '+crus+' K/100yrs)',alpha=alpha)
    plt.plot(CRU_time_copy,cru_gm_detrend,'m-',lw=lw,alpha=alpha)

    plt.plot(BE_time_copy,be_trend,'g-',lw=lw,label='BE   (trend: '+bes+' K/100yrs)',alpha=alpha)
    plt.plot(BE_time_copy,be_gm_detrend,'g-',lw=lw,alpha=alpha)

    plt.plot(MLOST_time_copy,mlost_trend,'c-',lw=lw,label='MLOST (trend: '+mlosts+' K/100yrs)',alpha=alpha)
    plt.plot(MLOST_time_copy,mlost_gm_detrend,'c-',lw=lw,alpha=alpha)

    plt.plot(TCR_time_copy,tcr_trend,'y-',lw=lw,label='TCR (trend: '+tcrs+' K/100yrs)',alpha=alpha)
    plt.plot(TCR_time_copy,tcr_gm_detrend,'y-',lw=lw,alpha=alpha)

    plt.plot(ERA_time_copy,era_trend,'b-',lw=lw,label='ERA (trend: '+eras+' K/100yrs)',alpha=alpha)
    plt.plot(ERA_time_copy,era_gm_detrend,'b-',lw=lw,alpha=alpha)

    plt.ylim(-1,1)    
    plt.legend(loc=2)

    # add to figure
    #plt.title('Detrended global mean temperature \n(' + nexp + ')',weight='bold',y=1.03)
    plt.title('Detrended global mean temperature',weight='bold',y=1.03)
    plt.xlabel('Year CE',fontweight='bold')
    plt.ylabel('Temperature anomaly (K)',fontweight='bold')
    xl_loc = [stime,etime]
    yl_loc = [-.7,.7]
    plt.xlim(xl_loc)
    plt.ylim(yl_loc)
    txl = xl_loc[0] + (xl_loc[1]-xl_loc[0])*.005
    tyl = yl_loc[0] + (yl_loc[1]-yl_loc[0])*.14
    
    off = .03
    plt.text(txl,tyl,      '(LMR,GIS)  : r full= ' + lgrf.ljust(4,' ') + ' r detrend= ' + lgrd.ljust(4,' ') + ' CE full= ' + lgcf.ljust(5,' ') + ' CE detrend= ' + lgcd.ljust(5,' '), fontsize=12, family='monospace')
    plt.text(txl,tyl-off,  '(LMR,CRU)  : r full= ' + lcrf.ljust(4,' ') + ' r detrend= ' + lcrd.ljust(4,' ') + ' CE full= ' + lccf.ljust(5,' ') + ' CE detrend= ' + lccd.ljust(5,' '), fontsize=12, family='monospace')
    plt.text(txl,tyl-2*off,'(LMR,BE)   : r full= ' + lbrf.ljust(4,' ') + ' r detrend= ' + lbrd.ljust(4,' ') + ' CE full= ' + lbcf.ljust(5,' ') + ' CE detrend= ' + lbcd.ljust(5,' '), fontsize=12, family='monospace')
    plt.text(txl,tyl-3*off,'(LMR,MLOST): r full= ' + lmrf.ljust(4,' ') + ' r detrend= ' + lmrd.ljust(4,' ') + ' CE full= ' + lmcf.ljust(5,' ') + ' CE detrend= ' + lmcd.ljust(5,' '), fontsize=12, family='monospace')
    plt.text(txl,tyl-4*off,'(LMR,TCR)  : r full= ' + ltrf.ljust(4,' ') + ' r detrend= ' + ltrd.ljust(4,' ') + ' CE full= ' + ltcf.ljust(5,' ') + ' CE detrend= ' + ltcd.ljust(5,' '), fontsize=12, family='monospace')
    plt.text(txl,tyl-5*off,'(LMR,ERA)  : r full= ' + lerf.ljust(4,' ') + ' r detrend= ' + lerd.ljust(4,' ') + ' CE full= ' + lecf.ljust(5,' ') + ' CE detrend= ' + lecd.ljust(5,' '), fontsize=12, family='monospace')
    
    if fsave:
        print 'saving to .png'
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_'+'detrended.png')
        plt.savefig(nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_'+'detrended.pdf',bbox_inches='tight', dpi=300, format='pdf')

    plt.show()

"""
# rank histograms
# loop over all years; send ensemble and a verification value
print ' '
print np.shape(gmt_save)
print lmr_smatch
print len(lmr_gm_copy)
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
        print fname
        plt.savefig(fname)
"""

#  Summary "table" figures

# dataset catalog IN ORDER
dset = ['LMR', 'GIS', 'CRU', 'BE', 'MLOST', 'TCR','ERA', 'CON']
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

CON_time = np.asarray(CON_time) # not sure why CON_time is a list here ...
ind_ver = np.searchsorted(verif_yrs,   np.intersect1d(CON_time[ind_con], verif_yrs))
con_gm_pad[ind_ver]     = con_gm[ind_con]


k = 0;  ALL_array[:,k] = lmr_gm_pad
k += 1; ALL_array[:,k] = gis_gm_pad
k += 1; ALL_array[:,k] = cru_gm_pad
k += 1; ALL_array[:,k] = be_gm_pad
k += 1; ALL_array[:,k] = mlost_gm_pad
k += 1; ALL_array[:,k] = tcr_gm_pad
k += 1; ALL_array[:,k] = era_gm_pad
k += 1; ALL_array[:,k] = con_gm_pad


#
# correlation coefficients for a chosen time interval 
#

# get ALL_array in a pandas data frame -> pandas is nan-friendly for calculations of correlation
df = pd.DataFrame(ALL_array,columns=dset)
#orr_matrix = df.corr()

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
#cmap = truncate_colormap(plt.cm.gist_heat_r,0.0,0.6)

cellsize = 0.12 # table cell size
fontsize = 12 

nticks = 7

# cell padding for the row labels; not sure why this is needed, but it is
lpad = '     '
rpad = lpad + ' '
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
vmin = 0.7

# this is here just to get the colorbar; apparently plt.table has no association
img = plt.imshow(vals, cmap=cmap, vmin = vmin, vmax =vmax)
cbar = plt.colorbar(shrink=.8)
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
the_table.set_fontsize(fontsize)
table_props = the_table.properties()
table_cells = table_props['child_artists']
for cell in table_cells: cell.set_height(cellsize)
for cell in table_cells: cell.set_width(cellsize)

plt.axis('off') # remove the axes that came with imshow
plt.title('Correlation',fontweight='bold',y=1.05)

fname =  nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_corr_table'

if fsave:
    plt.savefig(fname+'.png')
    plt.savefig(fname+'.pdf',format='pdf',dpi=300,bbox_inches='tight')

plt.show()


#
# CE figure
#

plt.figure()

vals = np.around(CE_matrix,2)

# set max and min values for color range
vmax = np.max(np.abs(vals))
vmin = 0.7

# this is here just to get the colorbar; apparently plt.table has no association
img = plt.imshow(vals, cmap=cmap, vmin = vmin, vmax =vmax)
cbar = plt.colorbar(shrink=.8)
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
the_table.set_fontsize(fontsize)
table_props = the_table.properties()
table_cells = table_props['child_artists']
for cell in table_cells: cell.set_height(cellsize)
for cell in table_cells: cell.set_width(cellsize)

plt.axis('off') # remove the axes that came with imshow
plt.title('Coefficient of efficiency',fontweight='bold',y=1.05)

fname = nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_ce_table'
if fsave:
    plt.savefig(fname+'.png')
    plt.savefig(fname+'.pdf',format='pdf',dpi=300,bbox_inches='tight')

plt.show()
