
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
from scipy import stats
from netCDF4 import Dataset
from datetime import datetime, timedelta
#
from LMR_plot_support import *
from LMR_exp_NAMELIST import *
from LMR_utils import global_mean, assimilated_proxies, coefficient_efficiency, rank_histogram
from load_gridded_data import read_gridded_data_GISTEMP
from load_gridded_data import read_gridded_data_HadCRUT
from load_gridded_data import read_gridded_data_BerkeleyEarth
from LMR_plot_support import *

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
fsave = True

# file specification
#
# current datasets
#
nexp = 'testdev_150yr_75pct'
#nexp = 'testing_1000_75pct_ens_size_Nens_10'
#nexp = 'testing_1000_75pct_200members'
#nexp = 'testdev_check_1000_75pct'
#nexp = 'ReconDevTest_1000_testing_coral'
#nexp = 'ReconDevTest_1000_testing_icecore'
#nexp = 'testdev_1000_100pct_icecoreonly'
#nexp = 'testdev_1000_100pct_mxdonly'
#nexp = 'testdev_1000_100pct_sedimentonly'
#nexp = 'testdev_detrend4_1000_75pct'

# specify directories for LMR and calibration data
datadir_output = '/home/chaos2/wperkins/data/LMR/output/archive'
#datadir_output = './data/'
datadir_calib = '/home/chaos2/wperkins/data/LMR/analyses'

# plotting preferences
nlevs = 30 # number of contours
alpha = 0.5 # alpha transpareny

# time limit for plot axis in years CE
xl = [1880,2000]

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

# In[224]:

# query file for assimilated proxy information (for now, ONLY IN THE r0 directory!)

ptypes,nrecords = assimilated_proxies(workdir+'/r0/')

print '--------------------------------------------------'
print 'Assimilated proxies by type:'
for pt in ptypes.keys():
    print pt + ': ' + str(ptypes[pt])
                
print 'Total: ' + str(nrecords)
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

# load HadCRU
datafile_calib   = 'HadCRUT.4.3.0.0.median.nc'
calib_vars = ['Tsfc']
[CRU_time,CRU_lat,CRU_lon,CRU_anomaly] = read_gridded_data_HadCRUT(datadir_calib,datafile_calib,calib_vars)

# load BerkeleyEarth
datafile_calib   = 'Land_and_Ocean_LatLong1.nc'
calib_vars = ['Tsfc']
[BE_time,BE_lat,BE_lon,BE_anomaly] = read_gridded_data_BerkeleyEarth(datadir_calib,datafile_calib,calib_vars)

# load NOAA MLOST
path = datadir_calib + '/NOAA/'
fname = 'NOAA_MLOST_aravg.ann.land_ocean.90S.90N.v3.5.4.201504.asc'
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

# load 20th century reanalysis
# this is copied from R. Tardif's load_gridded_data.py routine

infile = '/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc'
#infile = './data/500_allproxies_0/air.sig995.mon.mean.nc'

data = Dataset(infile,'r')
lat_20CR   = data.variables['lat'][:]
lon_20CR   = data.variables['lon'][:]
nlat_20CR = len(lat_20CR)
nlon_20CR = len(lon_20CR)

dateref = datetime(1800,1,1,0)
time_yrs = []
# absolute time from the reference
for i in xrange(0,len(data.variables['time'][:])):
    time_yrs.append(dateref + timedelta(hours=int(data.variables['time'][i])))

years_all = []
for i in xrange(0,len(time_yrs)):
    isotime = time_yrs[i].isoformat()
    years_all.append(int(isotime.split("-")[0]))
TCR_time = np.array(list(set(years_all))) # 'set' is used to get unique values in list
TCR_time.sort()  # sort the list

time_yrs  = np.empty(len(TCR_time), dtype=int)
TCR = np.empty([len(TCR_time), len(lat_20CR), len(lon_20CR)], dtype=float)
tcr_gm = np.zeros([len(TCR_time)])

# Loop over years in dataset
for i in xrange(0,len(TCR_time)):        
    # find indices in time array where "years[i]" appear
    ind = [j for j, k in enumerate(years_all) if k == TCR_time[i]]
    time_yrs[i] = TCR_time[i]
    # ---------------------------------------
    # Calculate annual mean from monthly data
    # Note: data has dims [time,lat,lon]
    # ---------------------------------------
    TCR[i,:,:] = np.nanmean(data.variables['air'][ind],axis=0)
    # compute the global mean temperature
    tcr_gm[i] = global_mean(TCR[i,:,:],lat_20CR,lon_20CR)
    
# Remove the temporal mean 
TCR = TCR - np.mean(TCR,axis=0)

# compute and remove the 20th century mean
satime = 1900
eatime = 1999
smatch, ematch = find_date_indices(TCR_time,satime,eatime)
tcr_gm = tcr_gm - np.mean(tcr_gm[smatch:ematch])

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

for dir in dirset:
    kk = kk + 1
    gmtpfile =  dir + '/gmt.npz'
    npzfile = np.load(gmtpfile)
    npzfile.files
    gmt = npzfile['gmt_save']
    recon_times = npzfile['recon_times']
    apcount = npzfile['apcount']
    print gmtpfile
    gmt_shape = np.shape(gmt)
    if first:
        gmt_save = np.zeros([niters,gmt_shape[1]])
        first = False
        
    gmt_save[kk,:] = gmt[apcount,:]
       
    if iplot:
        plt.plot(recon_times,gmt[apcount,:], label=str(kk), alpha=alpha)

# sample mean GMT
sagmt = np.average(gmt_save,0)
gmt_min = np.min(gmt_save,0)
gmt_max = np.max(gmt_save,0)
if iplot:
    fig = plt.figure()
    plt.plot(recon_times,sagmt,'k-',linewidth=2,label='sample mean',alpha=alpha)
    plt.fill_between(recon_times,gmt_min,gmt_max,facecolor='gray',alpha = alpha,linewidth=0.)
    plt.title('sample-mean and range GMT for ' + str(kk+1) + ' samples (exp: ' + nexp + ')')
    yl = [-1,1]
    plt.xlim(xl)
    plt.ylim(yl)
    # this prints the assimilated proxies dictionary on the plot
    txl = xl[0] + (xl[1]-xl[0])*.01
    tyl = yl[0] + (yl[1]-yl[0])*.05
    plt.text(txl,tyl,ptypes,fontsize=7)
    plt.plot(xl,[0,0])
    plt.text(txl,tyl-.05,str(kk+1) + ' samples',fontsize=7)
    if fsave:
        print 'saving to .png'
        plt.savefig(nexp+'_verify_GMT_'+str(xl[0])+'-'+str(xl[1]))
    
# define for later use
lmr_gm = sagmt
LMR_time = recon_times

# 
# compute GIS & CRU global mean 
#

gis_gm = global_mean(GIS_anomaly,GIS_lat,GIS_lon)
cru_gm = global_mean(CRU_anomaly,CRU_lat,CRU_lon)
be_gm = global_mean(BE_anomaly,BE_lat,BE_lon)

# adjust so that all time series pertain to 20th century mean
smatch, ematch = find_date_indices(LMR_time,satime,eatime)
lmr_off = np.mean(lmr_gm[smatch:ematch])
lmr_gm = lmr_gm - lmr_off
# fix previously set values
gmt_min = gmt_min - lmr_off
gmt_max = gmt_max - lmr_off
smatch, ematch = find_date_indices(GIS_time,satime,eatime)
gis_gm = gis_gm - np.mean(gis_gm[smatch:ematch])
smatch, ematch = find_date_indices(CRU_time,satime,eatime)
cru_gm = cru_gm - np.mean(cru_gm[smatch:ematch])
smatch, ematch = find_date_indices(BE_time,satime,eatime)
be_gm = be_gm - np.mean(be_gm[smatch:ematch])
smatch, ematch = find_date_indices(MLOST_time,satime,eatime)
mlost_gm = mlost_gm - np.mean(mlost_gm[smatch:ematch])

# indices for chosen time interval defined by stime and etime
lmr_smatch, lmr_ematch = find_date_indices(LMR_time,stime,etime)
gis_smatch, gis_ematch = find_date_indices(GIS_time,stime,etime)
cru_smatch, cru_ematch = find_date_indices(CRU_time,stime,etime)
tcr_smatch, tcr_ematch = find_date_indices(TCR_time,stime,etime)
be_smatch, be_ematch = find_date_indices(BE_time,stime,etime)
mlost_smatch, mlost_ematch = find_date_indices(MLOST_time,stime,etime)

# "consensus" global mean: average all non-LMR values
consensus_gmt = np.array([gis_gm[gis_smatch:gis_ematch],cru_gm[cru_smatch:cru_ematch],be_gm[be_smatch:be_ematch],mlost_gm[mlost_smatch:mlost_ematch]])
con_gm = np.mean(consensus_gmt,axis=0)
CON_time = range(stime,etime)

#
# correlation coefficients for a chosen time interval 
#

lmr_gis_corr = np.corrcoef(lmr_gm[lmr_smatch:lmr_ematch],gis_gm[gis_smatch:gis_ematch])
lmr_cru_corr = np.corrcoef(lmr_gm[lmr_smatch:lmr_ematch],cru_gm[cru_smatch:cru_ematch])
lmr_tcr_corr = np.corrcoef(lmr_gm[lmr_smatch:lmr_ematch],tcr_gm[tcr_smatch:tcr_ematch])
lmr_be_corr = np.corrcoef(lmr_gm[lmr_smatch:lmr_ematch],be_gm[be_smatch:be_ematch])
lmr_mlost_corr = np.corrcoef(lmr_gm[lmr_smatch:lmr_ematch],mlost_gm[mlost_smatch:mlost_ematch])
lmr_con_corr = np.corrcoef(lmr_gm[lmr_smatch:lmr_ematch],con_gm)
gis_cru_corr = np.corrcoef(gis_gm[gis_smatch:gis_ematch],cru_gm[cru_smatch:cru_ematch])
gis_tcr_corr = np.corrcoef(gis_gm[gis_smatch:gis_ematch],tcr_gm[tcr_smatch:tcr_ematch])
gis_be_corr = np.corrcoef(gis_gm[gis_smatch:gis_ematch],be_gm[be_smatch:be_ematch])

lcc = str(float('%.3g' % lmr_cru_corr[0,1]))
lgc = str(float('%.3g' % lmr_gis_corr[0,1]))
ltc = str(float('%.3g' % lmr_tcr_corr[0,1]))
lbc = str(float('%.3g' % lmr_be_corr[0,1]))
loc = str(float('%.3g' % lmr_con_corr[0,1]))
lmc = str(float('%.3g' % lmr_mlost_corr[0,1]))
gcc = str(float('%.3g' % gis_cru_corr[0,1]))
gtc = str(float('%.3g' % gis_tcr_corr[0,1]))
gbc = str(float('%.3g' % gis_be_corr[0,1]))
print '--------------------------------------------------'
print 'annual-mean correlations...'
print 'LMR_GIS correlation: ' + lgc
print 'LMR_CRU correlation: ' + lcc
print 'LMR_TCR correlation: ' + ltc
print 'LMR_BE  correlation: ' + lbc
print 'LMR_MLOST  correlation: ' + lmc
print 'LMR_CON  correlation: ' + loc
print 'GIS_CRU correlation: ' + gcc
print 'GIS_TCR correlation: ' + gtc
print 'GIS_BE correlation: ' + gbc
print 'LMR_consensus correlation: ' + loc
print '--------------------------------------------------'

#
# CE
#

lmr_gis_ce = coefficient_efficiency(gis_gm[gis_smatch:gis_ematch],lmr_gm[lmr_smatch:lmr_ematch])
lmr_cru_ce = coefficient_efficiency(cru_gm[cru_smatch:cru_ematch],lmr_gm[lmr_smatch:lmr_ematch])
lmr_tcr_ce = coefficient_efficiency(tcr_gm[tcr_smatch:tcr_ematch],lmr_gm[lmr_smatch:lmr_ematch])
lmr_be_ce = coefficient_efficiency(be_gm[be_smatch:be_ematch],lmr_gm[lmr_smatch:lmr_ematch])
lmr_mlost_ce = coefficient_efficiency(mlost_gm[mlost_smatch:mlost_ematch],lmr_gm[lmr_smatch:lmr_ematch])
lmr_con_ce = coefficient_efficiency(con_gm,lmr_gm[lmr_smatch:lmr_ematch])
tcr_gis_ce = coefficient_efficiency(gis_gm[gis_smatch:gis_ematch],tcr_gm[tcr_smatch:tcr_ematch])
cru_gis_ce = coefficient_efficiency(gis_gm[gis_smatch:gis_ematch],cru_gm[cru_smatch:cru_ematch])

lgce = str(float('%.3g' % lmr_gis_ce))
lcce = str(float('%.3g' % lmr_cru_ce))
ltce = str(float('%.3g' % lmr_tcr_ce))
lbce = str(float('%.3g' % lmr_be_ce))
lmce = str(float('%.3g' % lmr_mlost_ce))
loce = str(float('%.3g' % lmr_con_ce))
tgce = str(float('%.3g' % tcr_gis_ce))
cgce = str(float('%.3g' % cru_gis_ce))

print '--------------------------------------------------'
print 'coefficient of efficiency...'
print 'LMR-GIS CE: ' + str(lgce)
print 'LMR-CRU CE: ' + str(lcce)
print 'LMR-TCR CE: ' + str(ltce)
print 'LMR-BE CE: ' + str(lbce)
print 'LMR-CON CE: ' + str(loce)
print 'LMR-MLOST CE: ' + str(lmce)
print 'TCR-GIS CE: ' + str(tgce)
print 'GIS-CRU CE: ' + str(cgce)
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
# compare this with the "year" files from a single experiment
# from one year (1920), I get a GMT variance of 0.018368002408165367
# from this code, I get the MC mean GMT variance: 0.00486763546225
# test spread--error scatterplot
#if iplot:
#    fig = plt.figure()
#    plt.scatter(lg_err*lg_err,svar)
#    plt.show()

# plots
if iplot:
    lw = 1
    fig = plt.figure()
    plt.plot(LMR_time,lmr_gm,'k-',linewidth=2,label='LMR')
    plt.plot(GIS_time,gis_gm,'r-',linewidth=lw,label='GIS',alpha=alpha)
    plt.plot(CRU_time,cru_gm,'m-',linewidth=lw,label='CRU',alpha=alpha)
    plt.plot(TCR_time,tcr_gm,'y-',linewidth=lw,label='TCR',alpha=alpha)
    plt.plot(BE_time,be_gm,'g-',linewidth=lw,label='BE',alpha=alpha)
    plt.plot(MLOST_time,mlost_gm,'c-',linewidth=lw,label='MLOST',alpha=alpha)
    #plt.plot(CON_time,con_gm,'r-',linewidth=lw*2,label='consensus',alpha=alpha)
    #plt.fill_between(recon_times,gmt_min,gmt_max,facecolor='gray',alpha = 0.5,linewidth=0.)

    plt.title('GMT (' + nexp + ')',weight='bold')

    xl_loc = [stime,etime]
    yl_loc = [-.7,.7]
    plt.xlim(xl_loc)
    plt.ylim(yl_loc)
    txl = xl_loc[0] + (xl_loc[1]-xl_loc[0])*.01
    tyl = yl_loc[0] + (yl_loc[1]-yl_loc[0])*.05
    plt.text(txl,tyl,ptypes,fontsize=7)
    plt.text(txl,tyl-0.05,str(kk+1) + ' samples',fontsize=7)
    corrstr = 'correlations: r(lmr,gis)=' + lgc + '   r(lmr,cru)=' + lcc + '   r(lmr,tcr)=' + ltc + '   r(lmr,be)=' + lbc+ '   r(lmr,mlost)=' + lmc + '   r(lmr,con)=' + loc
    plt.text(txl,tyl+0.05,corrstr,fontsize=7)
    cestr = 'CE: ce(lmr,gis)=' + lgce + '   ce(lmr,cru)=' + lcce + '   ce(lmr,tcr)=' + ltce + '   ce(lmr,be)=' + lbce+ '   ce(lmr,mlost)=' + lmce + '   ce(lmr,con)=' + loce
    plt.text(txl,tyl+0.1,cestr,fontsize=7)
    plt.plot(xl_loc,[0,0])
    plt.legend(loc=2)

    if fsave:
        print 'saving to .png'
        plt.savefig(nexp+'_GMT_LMR_GIS_CRU_TCR_BE_MLOST_comparison')

#
# time averages
#

LMR_smoothed,LMR_smoothed_years = moving_average(lmr_gm,recon_times,nsyrs) 
GIS_smoothed,GIS_smoothed_years = moving_average(gis_gm,GIS_time,nsyrs) 
CRU_smoothed,CRU_smoothed_years = moving_average(cru_gm,CRU_time,nsyrs) 
TCR_smoothed,TCR_smoothed_years = moving_average(tcr_gm,TCR_time,nsyrs) 
BE_smoothed,BE_smoothed_years = moving_average(be_gm,BE_time,nsyrs) 
MLOST_smoothed,MLOST_smoothed_years = moving_average(mlost_gm,MLOST_time,nsyrs) 
 
# index offsets to account for averaging
toff = int(nsyrs/2)
ls_smatch, ls_ematch = find_date_indices(LMR_smoothed_years,stime+toff,etime-toff)
gs_smatch, gs_ematch = find_date_indices(GIS_smoothed_years,stime+toff,etime-toff)   
ls_gs_corr = np.corrcoef(LMR_smoothed[ls_smatch:ls_ematch],GIS_smoothed[gs_smatch:gs_ematch])
lscg = str(float('%.2g' % ls_gs_corr[0,1]))
cs_smatch, cs_ematch = find_date_indices(CRU_smoothed_years,stime+toff,etime-toff)   
ls_cs_corr = np.corrcoef(LMR_smoothed[ls_smatch:ls_ematch],CRU_smoothed[cs_smatch:cs_ematch])
lscc = str(float('%.2g' % ls_cs_corr[0,1]))
ts_smatch, ts_ematch = find_date_indices(TCR_smoothed_years,stime+toff,etime-toff)   
ls_ts_corr = np.corrcoef(LMR_smoothed[ls_smatch:ls_ematch],TCR_smoothed[ts_smatch:ts_ematch])
lsct = str(float('%.2g' % ls_ts_corr[0,1]))
bs_smatch, bs_ematch = find_date_indices(BE_smoothed_years,stime+toff,etime-toff)   
ls_bs_corr = np.corrcoef(LMR_smoothed[ls_smatch:ls_ematch],BE_smoothed[bs_smatch:bs_ematch])
lscb = str(float('%.2g' % ls_bs_corr[0,1]))
ms_smatch, ms_ematch = find_date_indices(MLOST_smoothed_years,stime+toff,etime-toff)   
ls_ms_corr = np.corrcoef(LMR_smoothed[ls_smatch:ls_ematch],MLOST_smoothed[ms_smatch:ms_ematch])
lscm = str(float('%.2g' % ls_ms_corr[0,1]))

print '--------------------------------------------------'
print str(nsyrs)+'-year-smoothed correlations...'
print 'smoothed lmr-gis correlation = ' + lscg
print 'smoothed lmr-cru correlation = ' + lscc
print 'smoothed lmr-tcr correlation = ' + lsct
print 'smoothed lmr-be correlation = ' + lscb
print 'smoothed lmr-mlost correlation = ' + lscm
print '--------------------------------------------------'

# smoothed "consensus" global mean: average all non-LMR values
#scon_gm = np.mean([GIS_smoothed[gs_smatch:gs_ematch],CRU_smoothed[cs_smatch:cs_ematch],BE_smoothed[bs_smatch:bs_ematch]],axis=0)
#CON_time = range(stime+toff,etime-toff)
#plt.plot(CON_time,scon_gm,'k-',linewidth=lw*2,label='consensus')

if iplot:
    fig = plt.figure()
    #plt.plot(recon_times,lmr_gm,'k-',linewidth=2)
    plt.fill_between(recon_times,gmt_min,gmt_max,facecolor='gray',alpha=alpha,linewidth=0.)

    # add smoothed lines
    plt.plot(LMR_smoothed_years,LMR_smoothed,'k-',linewidth=4, label='LMR')
    plt.plot(GIS_smoothed_years,GIS_smoothed,'r-',linewidth=4, label='GIS',alpha=alpha)
    plt.plot(CRU_smoothed_years,CRU_smoothed,'m-',linewidth=4, label='CRU',alpha=alpha)
    plt.plot(TCR_smoothed_years,TCR_smoothed,'y-',linewidth=4, label='TCR',alpha=alpha)
    plt.plot(BE_smoothed_years,BE_smoothed,'g-',linewidth=4, label='BE',alpha=alpha)
    plt.plot(MLOST_smoothed_years,MLOST_smoothed,'c-',linewidth=4, label='MLOST',alpha=alpha)
            
    plt.legend(loc=2)

    if nsyrs == 5:
        xl = [stime,etime]
        yl = [-1.,1.]
    elif nsyrs == 31:
        xl = [1000,2000]
        yl = [-1.1,0.6] # for comparison with Wikipedia figure
    else:
        xl = [stime,etime]
        yl = [-1,1]
        
    plt.xlim(xl)
    plt.ylim(yl)
    plt.title('GMT (' + nexp[0:] + ') range (gray) and ' +str(nsyrs) + '-year moving average' ,weight='bold')

    # this prints the assimilated proxies dictionary on the plot
    txl = xl[0] + (xl[1]-xl[0])*.01
    tyl = yl[0] + (yl[1]-yl[0])*.05
    plt.text(txl,tyl,ptypes,fontsize=7)
    plt.text(txl,tyl-.05,str(kk+1) + ' samples',fontsize=7)
    corrstr = 'correlations: r(lmr,gis)=' + lscg + '   r(lmr,cru)=' + lscc + '   r(lmr,tcr)=' + lsct + '   r(lmr,be)=' + lscb + '   r(lmr,mlost)=' + lscm
    plt.text(txl,tyl+0.05,corrstr,fontsize=8)

    plt.plot(xl,[0,0])
    if fsave:
        fname = nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_'+str(nsyrs)+'yr_smoothed'
        print fname
        plt.savefig(fname)
    
#
# detrend and verify the detrended signal
#

print '--------------------------------------------------'
print 'verification for detrended data'
print '--------------------------------------------------'

# indices for chosen time interval defined by stime and etime

lmr_smatch, lmr_ematch = find_date_indices(LMR_time,stime,etime)

# save copies of the original data for residual estimates later
lmr_gm_copy = np.copy(lmr_gm[lmr_smatch:lmr_ematch])
LMR_time_copy = np.copy(LMR_time[lmr_smatch:lmr_ematch])
xvar = range(len(lmr_gm_copy))
lmr_slope, lmr_intercept, r_value, p_value, std_err = stats.linregress(xvar,lmr_gm_copy)
lmr_trend = lmr_slope*np.squeeze(xvar) + lmr_intercept
lmr_gm_detrend = lmr_gm_copy - lmr_trend

# repeat for GIS
gis_smatch, gis_ematch = find_date_indices(GIS_time,stime,etime)
# save copies of the original data for residual estimates later
gis_gm_copy = np.copy(gis_gm[gis_smatch:gis_ematch])
GIS_time_copy = np.copy(GIS_time[gis_smatch:gis_ematch])
xvar = range(len(gis_gm_copy))
gis_slope, gis_intercept, r_value, p_value, std_err = stats.linregress(xvar,gis_gm_copy)
gis_trend = gis_slope*np.squeeze(xvar) + gis_intercept
gis_gm_detrend = gis_gm_copy - gis_trend

# r and ce on full data
lmr_gis_corr_full = np.corrcoef(lmr_gm_copy, gis_gm_copy)
lmr_full_err = lmr_gm_copy - gis_gm_copy
lmr_gis_ce_full = 1. - np.var(lmr_full_err)/np.var(gis_gm_detrend)
lgrf =  str(float('%.3g' % lmr_gis_corr_full[0,1]))
lgcf =  str(float('%.3g' % lmr_gis_ce_full))
# check if the two pieces are correlated (if not, then they sum to the total error)
error_trend = lmr_trend - gis_trend
error_detrend = lmr_gm_detrend - gis_gm_detrend
check = np.corrcoef(error_trend,error_detrend)
print 'correlaton between trend and detrend errors = ' + str(check[0,1])
print 'error variances...'
print 'trend error: ' + str(np.var(error_trend))
print 'detrend error: ' + str(np.var(error_detrend))
print 'detrend error + trend error: ' + str(np.var(error_trend)+np.var(error_detrend))
print 'full error : ' + str(np.var(error_trend+error_detrend))

# r and ce on detrended data
lmr_gis_corr_detrend = np.corrcoef(lmr_gm_detrend,gis_gm_detrend)
lmr_detrend_err = lmr_gm_detrend - gis_gm_detrend
lmr_gis_ce_detrend = 1. - np.var(lmr_detrend_err)/np.var(gis_gm_detrend)
lgrd =  str(float('%.3g' % lmr_gis_corr_detrend[0,1]))
lgcd =  str(float('%.3g' % lmr_gis_ce_detrend))
lmrs =  str(float('%.2g' % (lmr_slope*100.)))
gs =  str(float('%.2g' % (gis_slope*100.)))

print 'r: ' + str(lgrf) + ' ' + str(lgrd)
print 'ce: ' + str(lgcf) + ' ' + str(lgcd)

# plots

"""
# LMR
plt.plot(LMR_time_copy,lmr_gm_copy,'k-')
plt.plot(LMR_time_copy,lmr_trend,'b-')
plt.plot(LMR_time_copy,lmr_gm_detrend,'r-')
plt.ylim(-1,1)
plt.show()

# GIS
plt.plot(GIS_time_copy,gis_gm_copy,'k-')
plt.plot(GIS_time_copy,gis_trend,'b-')
plt.plot(GIS_time_copy,gis_gm_detrend,'r-')
plt.ylim(-1,1)
plt.show()
"""

if iplot:
    # LMR & GIS
    fig = plt.figure()
    plt.plot(GIS_time_copy,gis_trend,'r-')
    plt.plot(GIS_time_copy,gis_gm_detrend,'r-')
    plt.plot(LMR_time_copy,lmr_trend,'k-')
    plt.plot(LMR_time_copy,lmr_gm_detrend,'k-')
    plt.ylim(-1,1)
    
    # add to figure
    plt.title('detrended GMT (' + nexp + ') (red: GIS; black: LMR)',weight='bold')
    xl_loc = [1880,2000]
    yl_loc = [-.7,.7]
    plt.xlim(xl_loc)
    plt.ylim(yl_loc)
    # left side
    #txl = xl_loc[0] + (xl_loc[1]-xl_loc[0])*.05
    #tyl = yl_loc[0] + (yl_loc[1]-yl_loc[0])*.01
    # right side
    txl = xl_loc[1] - (xl_loc[1]-xl_loc[0])*.4
    tyl = yl_loc[0] + (yl_loc[1]-yl_loc[0])*.01
    
    off = .05
    plt.text(txl,tyl+6*off,'lmr trend: '+lmrs+' K/100yrs',fontsize=12)
    plt.text(txl,tyl+5*off,'gis trend: '+gs+' K/100yrs',fontsize=12)
    plt.text(txl,tyl+4*off,'r full: '+str(lgrf),fontsize=12)
    plt.text(txl,tyl+3*off,'r detrend: '+str(lgrd),fontsize=12)
    plt.text(txl,tyl+2*off,'ce full: '+str(lgcf),fontsize=12)
    plt.text(txl,tyl+off,'ce detrend: '+str(lgcd),fontsize=12)
    
    fname =  nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_'+'detrended'
    if fsave:
        plt.savefig(fname)

# rank histograms
# loop over all years; send ensemble and a verification value
print np.shape(gmt_save)
print lmr_smatch
print len(lmr_gm_copy)
rank = []
for yr in range(len(lmr_gm_copy)):
    rankval = rank_histogram(gmt_save[:,lmr_smatch+yr:lmr_smatch+yr+1],gis_gm_copy[yr])
    rank.append(rankval)
    
if iplot:
    fig = plt.figure()
    nbins = 10
    plt.hist(rank,nbins)
    if fsave:
        fname = nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_rank_histogram'
        print fname
        plt.savefig(fname)

# display all figures at the end:
plt.show() 
