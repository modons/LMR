#
# copy of LMR_verify_GM.py to loop over the ensemble size experiments and compute essentials stats
#

import matplotlib
# need to do this when running remotely, and to suppress figures
matplotlib.use('Agg')

import csv
import glob, os, fnmatch
import numpy as np
import mpl_toolkits.basemap as bm
import matplotlib.pyplot as plt
from scipy import stats
from netCDF4 import Dataset
from datetime import datetime, timedelta
#
from LMR_plot_support import *
from LMR_utils2 import global_hemispheric_means, assimilated_proxies, coefficient_efficiency, rank_histogram
from load_gridded_data import read_gridded_data_GISTEMP
from load_gridded_data import read_gridded_data_HadCRUT
from load_gridded_data import read_gridded_data_BerkeleyEarth
from LMR_plot_support import *

# this sets the default size of the figure in inches. ['figure.figsize'] = width, height;  
# aspect ratio appears preserved on smallest of the two
plt.rcParams['figure.figsize'] = 10, 10  # that's default image size for this interactive session
plt.rcParams['axes.linewidth'] = 2.0 #set the value globally
plt.rcParams['font.weight'] = 'bold' #set the font weight globally
plt.rc('text', usetex=False)
#plt.rc('text', usetex=True)

def ens_support(dirset,satime,eatime,stime,etime,gis_gm,GIS_time):

    first = True
    kk = -1

    niters = len(dirset)

    for dir in dirset:
        kk = kk + 1
        gmtpfile =  dir + '/gmt.npz'
        npzfile = np.load(gmtpfile)
        npzfile.files
        gmt = npzfile['gmt_save']
        recon_times = npzfile['recon_times']
        apcount = npzfile['apcount']
        gmt_shape = np.shape(gmt)
        if first:
            gmt_save = np.zeros([niters,gmt_shape[1]])
            first = False
            
        gmt_save[kk,:] = gmt[apcount,:]

    # sample mean GMT
    sagmt = np.average(gmt_save,0)
    
    # define for later use
    lmr_gm = sagmt
    LMR_time = recon_times
    
    # 
    # compute GIS & CRU global mean 
    #
    smatch, ematch = find_date_indices(LMR_time,satime,eatime)
    lmr_off = np.mean(lmr_gm[smatch:ematch])
    lmr_gm = lmr_gm - lmr_off

    # adjust so that all time series pertain to 20th century mean
    smatch, ematch = find_date_indices(LMR_time,satime,eatime)
    lmr_off = np.mean(lmr_gm[smatch:ematch])
    lmr_gm = lmr_gm - lmr_off
    
    # indices for chosen time interval defined by stime and etime
    lmr_smatch, lmr_ematch = find_date_indices(LMR_time,stime,etime)
    gis_smatch, gis_ematch = find_date_indices(GIS_time,stime,etime)
    
    #
    # correlation coefficients for a chosen time interval 
    #
    
    lmr_gis_corr = np.corrcoef(lmr_gm[lmr_smatch:lmr_ematch],gis_gm[gis_smatch:gis_ematch])
    lgc = str(float('%.3g' % lmr_gis_corr[0,1]))
    #print '--------------------------------------------------'
    #print 'annual-mean correlations...'
    #print 'LMR_GIS correlation: ' + lgc
    
    #
    # CE
    #
    
    lmr_gis_ce = coefficient_efficiency(gis_gm[gis_smatch:gis_ematch],lmr_gm[lmr_smatch:lmr_ematch])
    lgce = str(float('%.3g' % lmr_gis_ce))
    
    #print '--------------------------------------------------'
    #print 'coefficient of efficiency...'
    #print 'LMR-GIS CE: ' + str(lgce)

    # save data for plotting
    r_save = lmr_gis_corr[0,1]
    ce_save = lmr_gis_ce

    return r_save,ce_save

##################################
# START:  set user parameters here
##################################

# define the verification time interval
stime = 1880
etime = 2000

# compute and remove the 20th century mean
satime = 1900
eatime = 1999

# define the running time mean 
#nsyrs = 31 # 31-> 31-year running mean--nsyrs must be odd!
nsyrs = 5 # 5-> 5-year running mean--nsyrs must be odd!

# option to create figures
#iplot = False
iplot = True

# option to save figures to a file
fsave = True
#fsave = False

# file specification
#
# current datasets
#
#nexp = 'testing_1000_75pct_ens_size_Nens_10'
#nexp = 'testing_1000_75pct_200members'
#nexp = 'testdev_check_1000_75pct'
#nexp = 'ReconDevTest_1000_testing_coral'
#nexp = 'ReconDevTest_1000_testing_icecore'
#nexp = 'testdev_1000_100pct_icecoreonly'
#nexp = 'testdev_1000_100pct_mxdonly'
#nexp = 'testdev_1000_100pct_sedimentonly'
#nexp = 'testdev_detrend4_1000_75pct'
nexp = 'ReconMultiState_MPIESMP_LastMillenium_ens100_allAnnualProxyTypes_pf0.5'

# specify directories for LMR and calibration data
#datadir_output = '/home/disk/kalman3/hakim/LMR/'
datadir_output = '/home/disk/kalman3/rtardif/LMR/output'
#datadir_output = './data/'
#datadir_calib = '../data/'
datadir_calib = '/home/disk/kalman3/rtardif/LMR/data/analyses'

# plotting preferences
nlevs = 30 # number of contours
alpha = 0.5 # alpha transpareny

# time limit for plot axis in years CE
xl = [1000,2000]

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

#
# load GISTEMP, HadCRU, and HadCET
#

# load GISTEMP
datafile_calib   = 'gistemp1200_ERSST.nc'
calib_vars = ['Tsfc']
[GIS_time,GIS_lat,GIS_lon,GIS_anomaly] = read_gridded_data_GISTEMP(datadir_calib,datafile_calib,calib_vars)
nlat_GIS = len(GIS_lat)
nlon_GIS = len(GIS_lon)

# 
# compute GIS & CRU global mean 
#

[gis_gm,gis_nhm,gis_shm] = global_hemispheric_means(GIS_anomaly,GIS_lat)

# adjust so that all time series pertain to 20th century mean
# fix previously set values
smatch, ematch = find_date_indices(GIS_time,satime,eatime)
gis_gm = gis_gm - np.mean(gis_gm[smatch:ematch])
gis_nhm = gis_nhm - np.mean(gis_nhm[smatch:ematch])
gis_shm = gis_shm - np.mean(gis_shm[smatch:ematch])

# the ensemble size experiments
ens_expts = [10,30,50,75,100,200,500]
maxens = len(ens_expts)

# (#iterations,#ensemble size tests)
#maxits = 1 # look at single experiment
maxits = 11 # look at all experiments
r_save = np.zeros([maxits,maxens])
ce_save = np.zeros([maxits,maxens])

# loop over ensemble-size experiments
es = -1
for Nens in ens_expts:
    es = es + 1
    
    if Nens == 200:
        nexp = 'testing_1000_75pct_200members'
    elif Nens == 100:
        nexp = 'testdev_check_1000_75pct'
    else:
        nexp = 'testing_1000_75pct_ens_size_Nens_'+str(Nens)
        
    workdir = datadir_output + '/' + nexp


    print '--------------------------------------------------'
    print 'working directory: '+workdir
    print '--------------------------------------------------'

    # get a listing of the iteration directories and sort
    dirs = glob.glob(workdir+"/r*")
    dirs.sort()

    for k in range(maxits):
        if maxits == 1:
            dirset = dirs[0:11]
        else:
            dirset = dirs[k:k+1]

        r,ce = ens_support(dirset,satime,eatime,stime,etime,gis_gm,GIS_time)
        r_save[k,es] = r
        ce_save[k,es] = ce

rmean = np.mean(r_save,0)
cemean = np.mean(ce_save,0)
rmin = np.min(r_save,0)
rmax = np.max(r_save,0)
cemin = np.min(ce_save,0)
cemax = np.max(ce_save,0)


fig = plt.figure()
if maxits == 1:
    plt.plot(ens_expts,rmean,'ko-')
    plt.plot(ens_expts,cemean,'ro-')
    plt.xlabel('ensemble size',fontsize=12,fontweight='bold')
    plt.ylabel('r (black) / ce (red)',fontsize=12,fontweight='bold')
    plt.title('LMR-GIS verification: r and ce for average over 11 samples')
    plt.ylim([0.25,1])
    plt.xlim([0,550])
    if fsave:
        plt.savefig('ensemble_convergence_11')
else:
    plt.errorbar(ens_expts,rmean,yerr=[rmean-rmin,rmax-rmean],fmt='ko-')
    plt.errorbar(ens_expts,cemean,yerr=[cemean-cemin,cemax-cemean],fmt='ro-')
    plt.title('LMR-GIS verification: r and ce for one iteration (mean and range over 11 samples)')
    plt.xlabel('ensemble size',fontsize=12,fontweight='bold')
    plt.ylabel('r (black) / ce (red)',fontsize=12,fontweight='bold')
    plt.ylim([0.25,1])
    plt.xlim([0,550])
    if fsave:
        plt.savefig('ensemble_convergence_1')

#
# convergence of r and ce in iteration sample size for a single experiment
#

print ' convergence of r and ce in iteration sample size for a single experiment'

# this one has over 100 samples
nexp = 'testdev_check_1000_75pct'
workdir = datadir_output + '/' + nexp

print 'workdir = ' + workdir
dirs = glob.glob(workdir+"/r*")
dirs.sort()

maxsamp = len(dirs)
print 'maxsamp = ' + str(maxsamp)

# (#iterations,#ensemble size tests)
its = [1,2,3,5,10,15,20]
maxits = len(its) 
# number of samples for each iteration value
nsamp = 30
r_save2 = np.zeros([nsamp,maxits])
ce_save2 = np.zeros([nsamp,maxits])

# loop over iterations
es = -1
for it in its:
    es = es + 1

    print 'it = ' + str(it)
    
    # get a listing of the iteration directories and sort
    dirs = glob.glob(workdir+"/r*")
    dirs.sort()

    for k in range(nsamp):
        # make this random once the code works
        dset = np.int_(np.random.rand(it+1)*maxsamp).tolist()
        dirset = list(dirs[i] for i in dset)        
        r,ce = ens_support(dirset,satime,eatime,stime,etime,gis_gm,GIS_time)
        r_save2[k,es] = r
        ce_save2[k,es] = ce

rmean2 = np.mean(r_save2,0)
cemean2 = np.mean(ce_save2,0)
rmin2 = np.min(r_save2,0)
rmax2 = np.max(r_save2,0)
cemin2 = np.min(ce_save2,0)
cemax2 = np.max(ce_save2,0)

fig = plt.figure()
plt.errorbar(its,rmean2,yerr=[rmean2-rmin2,rmax2-rmean2],fmt='ko-')
plt.errorbar(its,cemean2,yerr=[cemean2-cemin2,cemax2-cemean2],fmt='ro-')
plt.xlabel('iteration size',fontsize=12,fontweight='bold')
plt.ylabel('r (black) / ce (red)',fontsize=12,fontweight='bold')
plt.title('LMR-GIS verification: r and ce convergence in iteration ('+ str(nsamp) + ' random draws; Nens=100)')
plt.ylim([0.25,1])
plt.xlim([0,max(its)])
if fsave:
    plt.savefig('iteration_convergence')
    
plt.show()

         
