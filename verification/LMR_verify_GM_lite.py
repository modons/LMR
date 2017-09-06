""" 
Module: LMR_verify_GM_lite.py

Purpose: Use as a platform for future development involving GMT analysis.
         Just reads consensus GMT for CONSENSUS (mean of instrumental datasets) and LMR.

Started from LMR_verify_GM.py commit 45e3a7a2f1d030a489a1e0b6bb24c25632e79387

Originator: Greg Hakim, U. of Washington, August 2017

Revisions: 

"""

def LMR_GM_ce(datadir_output,nexp,MCset=None,stime=1880,etime=2000):
    
    import sys, glob, os, fnmatch
    import numpy as np
    from scipy import stats
    from datetime import datetime, timedelta
    import cPickle
    import warnings
    # LMR specific imports
    sys.path.append('../')
    from LMR_utils import coefficient_efficiency, find_date_indices

    warnings.filterwarnings('ignore')

    print('--------------------------------------------------------------------')
    print('verification of LMR global-mean 2m air temperature against CONSENSUS')
    print('--------------------------------------------------------------------')

    workdir = datadir_output + '/' + nexp

    # get directory and information for later use

    print('--------------------------------------------------')
    print('working on LMR directory: %s' % workdir)
    print('--------------------------------------------------')

    # get a listing of the iteration directories
    dirs = glob.glob(workdir+"/r*")

    # selecting  the MC iterations to keep
    if MCset:
        dirset = dirs[MCset[0]:MCset[1]+1]
    else:
        dirset = dirs
    niters = len(dirset)

    print('--------------------------------------------------')
    print('niters = %s' % str(niters))
    print('--------------------------------------------------')

    # "grand ensemble approach"---collect all LMR iterations into a superensemble
    first = True
    kk = -1
    for dir in dirset:
        kk = kk + 1
        gmtpfile =  dir + '/gmt_ensemble.npz'
        print gmtpfile
        npzfile = np.load(gmtpfile)
        npzfile.files
        gmt = npzfile['gmt_ensemble']
        LMR_time = npzfile['recon_times']
        gmt_shape = np.shape(gmt)
        if first:
            gmt_save = np.zeros([gmt_shape[0],gmt_shape[1],niters])
            first = False

        gmt_save[:,:,kk] = gmt

    gmse = np.reshape(gmt_save,(gmt_shape[0],gmt_shape[1]*niters))
    lmr_gm = np.mean(gmse,1)

    # load consensus GMT info
    filen = 'consensus_gmt.npz'
    npzfile = np.load(filen)
    con_gm = npzfile['con_gm']
    CON_time = npzfile['CON_time']

    # compute and remove the 20th century mean
    satime = 1900
    eatime = 1999

    # LMR
    smatch, ematch = find_date_indices(LMR_time,satime,eatime)
    lmr_off = np.mean(lmr_gm[smatch:ematch])
    lmr_gm = lmr_gm - lmr_off

    # consensus
    smatch, ematch = find_date_indices(CON_time,satime,eatime)
    con_off = np.mean(con_gm[smatch:ematch])
    con_gm = con_gm - con_off

    #-------------------------------------
    # basic stats start here
    #-------------------------------------

    #
    # correlation coefficients & CE over chosen time interval 
    #

    verif_yrs = np.arange(stime,etime+1,1)

    # LMR-consensus
    # overlaping years within verification interval
    overlap_yrs = np.intersect1d(np.intersect1d(LMR_time, CON_time), verif_yrs)
    ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, overlap_yrs))
    ind_con = np.searchsorted(CON_time, np.intersect1d(CON_time, overlap_yrs))
    lmr_con_corr = np.corrcoef(lmr_gm[ind_lmr],con_gm[ind_con])
    lmr_con_ce = coefficient_efficiency(con_gm[ind_con],lmr_gm[ind_lmr])

    loc = str(float('%.3f' % lmr_con_corr[0,1]))
    loce = str(float('%.3f' % lmr_con_ce))

    print('\n')
    print('--------------------------------------------------')
    print('annual-mean correlation: %s' % loc)
    print('coefficient of efficiency: %s' % loce)
    print('--------------------------------------------------')

    ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, verif_yrs))
    lmr_gm_copy = np.copy(lmr_gm[ind_lmr])
    LMR_time_copy = np.copy(LMR_time[ind_lmr])
    xvar = range(len(lmr_gm_copy))
    lmr_slope, lmr_intercept, r_value, p_value, std_err = stats.linregress(xvar,lmr_gm_copy)
    lmr_trend = lmr_slope*np.squeeze(xvar) + lmr_intercept
    lmr_gm_detrend = lmr_gm_copy - lmr_trend

    # for CONsensus
    # overlaping years within verification interval
    overlap_yrs = np.intersect1d(np.intersect1d(LMR_time_copy, CON_time), verif_yrs)
    ind_lmr = np.searchsorted(LMR_time_copy, np.intersect1d(LMR_time_copy, overlap_yrs))
    ind_con = np.searchsorted(CON_time, np.intersect1d(CON_time, overlap_yrs))
    CON_time_copy = CON_time[ind_con]
    con_gm_copy = np.copy(con_gm[ind_con])
    xvar = range(len(ind_con))
    con_slope, con_intercept, r_value, p_value, std_err = stats.linregress(xvar,con_gm_copy)
    con_trend = con_slope*np.squeeze(xvar) + con_intercept
    con_gm_detrend = con_gm_copy - con_trend
    # r and ce on full data
    full_err  = lmr_gm_copy[ind_lmr] - con_gm_copy
    lmr_con_corr_full = np.corrcoef(lmr_gm_copy[ind_lmr],con_gm[ind_con])
    lmr_con_ce_full = coefficient_efficiency(con_gm_copy,lmr_gm_copy[ind_lmr])
    lconrf =  str(float('%.3f' % lmr_con_corr_full[0,1]))
    lconcf =  str(float('%.3f' % lmr_con_ce_full))
    # r and ce on detrended data
    lmr_con_corr_detrend = np.corrcoef(lmr_gm_detrend[ind_lmr],con_gm_detrend)
    lmr_detrend_err = lmr_gm_detrend[ind_lmr] - con_gm_detrend
    lmr_con_ce_detrend = coefficient_efficiency(con_gm_detrend,lmr_gm_detrend[ind_lmr])
    lconrd   =  str(float('%.3f' % lmr_con_corr_detrend[0,1]))
    lconcd   =  str(float('%.3f' % lmr_con_ce_detrend))

    # Trends
    cons   =  str(float('%.3f' % (con_slope*100.)))
    lmrs   =  str(float('%.3f' % (lmr_slope*100.)))

    print('--------------------------------------------------')
    print('detrended annual-mean correlation: %s' % lconrd)
    print('detrended coefficient of efficiency: %s' % lconcd)
    print('--------------------------------------------------')

    print 'LMR trend: '+str(lmrs) + ' K/100yrs'
    print 'CON trend: '+str(cons) + ' K/100yrs'

    return gmt_save,LMR_time,con_gm,CON_time,lmr_gm_detrend,con_gm_detrend,lmr_slope,con_slope

#---------- main ----------------------

import sys
import numpy as np
sys.path.append('../')
from LMR_utils import coefficient_efficiency, find_date_indices

##################################
# START:  set user parameters here
##################################

# specify directories for LMR data
#datadir_output = '/home/disk/kalman3/hakim/LMR'
#datadir_output = '/home/disk/kalman2/wperkins/LMR_output/archive'
#datadir_output = '/home/disk/kalman3/rtardif/LMR/output'
datadir_output = '/Users/hakim/data/LMR/archive/'

# file specification
#nexp = 'pages2_loc25000_seasonal_bilinear_nens50_75pct'
nexp = 'test'
#nexp = 'p2_ccsm4LM_n200_bilin_GISTEMPGPCCseasonPSM_PAGES2kv2_pf0.75_loc25k/'

# perform verification using all recon. MC realizations (MCset = None )
MCset = None
#MCset = (0,0)

# start and end time of verification
stime=1880
etime=2000

##################################
# END:  set user parameters here
##################################

# read GMT data and print some bastic results
gmt_save,LMR_time,con_gm,CON_time,lmr_gm_detrend,con_gm_detrend,lmr_slope,con_slope = \
  LMR_GM_ce(datadir_output,nexp,MCset)

[nyears,nens,niters] = np.shape(gmt_save)
 
# average and 5-95% range
# 1. global mean
gmse = np.reshape(gmt_save,(nyears,nens*niters))
lmr_gm = np.mean(gmse,1)
gmt_min = np.percentile(gmse,5,axis=1)
gmt_max = np.percentile(gmse,95,axis=1)
    
verif_yrs = np.arange(stime,etime+1,1)
overlap_yrs = np.intersect1d(np.intersect1d(LMR_time, CON_time), verif_yrs)
ind_lmr = np.searchsorted(LMR_time, np.intersect1d(LMR_time, overlap_yrs))
ind_con = np.searchsorted(CON_time, np.intersect1d(CON_time, overlap_yrs))

# test:
# 1. convergence in ce for iteration ensemble mean
lmr_samp = np.mean(gmt_save,axis=1)
# 2. convergence in ce for grand ensemble size
#lmr_samp = gmse

nsamp = np.shape(lmr_samp)[1]

# remove 20th century mean from sample

satime = 1900
eatime = 1999
smatch, ematch = find_date_indices(LMR_time,satime,eatime)
lmr_samp_off = np.mean(lmr_samp[smatch:ematch,:],axis=0)
print 'offset: ' + str(lmr_samp_off)
lmr_samp = lmr_samp - lmr_samp_off

print 'lmr sample shape: ' + str(np.shape(lmr_samp))

# number of samples for each batch size
ns = 100

# convergence in chosen sample size
ce_grand = []
ce_min = []
ce_max = []
for k in range(nsamp):
    ce_save = np.zeros(ns)
    for m in range(ns):
        ri = (np.random.rand(k+1)*niters).astype(int)
        # why not tmp = lmr_samp[ind_lmr,ri]? Good question!
        tmp = lmr_samp[ind_lmr,:][:,ri]
        lmr_avg = np.mean(tmp,axis=1)
        
        test_corr = np.corrcoef(lmr_avg,con_gm[ind_con])
        test_ce = coefficient_efficiency(con_gm[ind_con],lmr_avg)
        ce_save[m] = test_ce

    ce_grand.append(np.mean(ce_save))
    ce_max.append(np.max(ce_save))
    ce_min.append(np.min(ce_save))
    
print np.shape(ce_grand)
print np.mean(ce_grand)
print np.min(ce_grand)
print np.max(ce_grand)

# figures
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')
matplotlib.pyplot.switch_backend('agg')

#plt.plot(ce_grand)
xvals = np.arange(nsamp) + 1
ymax = np.array(ce_max)-np.array(ce_grand)
ymin = np.array(ce_grand)-np.array(ce_min)
plt.errorbar(xvals,ce_grand,yerr=[ymin,ymax],ecolor='gray')
plt.savefig('ce_grand.png')
#plt.show()

# add a second plot...
datadir_output = '/Users/hakim/data/LMR/archive/'
#nexp = 'pages2_loc25000_seasonal_bilinear_nens50_75pct'
nexp = 'test2'

gmt_save,LMR_time,con_gm,CON_time,lmr_gm_detrend,con_gm_detrend,lmr_slope,con_slope = \
  LMR_GM_ce(datadir_output,nexp,MCset)
