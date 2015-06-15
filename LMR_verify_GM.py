#
# verify statistics related to the global-mean 2m air temperature
#
# started from LMR_plots.py r-86

import matplotlib
# need to do this when running remotely
matplotlib.use('Agg')

import csv
from LMR_plot_support import *
from LMR_exp_NAMELIST import *
import numpy as np
import mpl_toolkits.basemap as bm
import matplotlib.pyplot as plt
from LMR_utils import global_mean, assimilated_proxies

#iplot = False
iplot = True

##################################
# START:  set user parameters here
##################################
#
# set paths, the filename for plots, and global plotting preferences
#

# file specification
#nexp = 'testdev_500_allproxies'
#exp = 'testdev_500_trunctest'
#
nexp = 'testdev_1000_75pct'
#
#nexp = 'testdev_1000_75pct_noTRW'
#nexp = 'testdev_1000_75pct_treesonly'
#nexp = 'testdev_1000_75pct_icecoreonly'
#nexp = 'testdev_1000_75pct_coralonly'
#nexp = 'testdev_1000_100pct_coralonly'
#nexp = 'testdev_1000_100pct_icecoreonly'
#nexp = 'testdev_1000_100pct_mxdonly'
#nexp = 'testdev_1000_100pct_sedimentonly'
# new
#nexp = 'testdev_1000_75pct_BE'
#nexp = 'testdev_1000_75pct_BE_noTRW'

# override datadir
datadir_output = '/home/disk/kalman3/hakim/LMR/'
#datadir_output = './data/'

# set some plotting preferences
# number of contours
nlevs = 30

# alpha transpareny
alpha = 0.5

# time range in years CE
xl = [1000,2000]

# this sets the default size of the figure in inches. ['figure.figsize'] = width, height;  
# aspect ratio appears preserved on smallest of the two
plt.rcParams['figure.figsize'] = 10, 10  # that's default image size for this interactive session
plt.rcParams['axes.linewidth'] = 2.0 #set the value globally
plt.rcParams['font.weight'] = 'bold' #set the font weight globally
#plt.rc('text', usetex=True)
plt.rc('text', usetex=False)

# this sets the default size of the figure in inches. ['figure.figsize'] = width, height;  
# aspect ratio appears preserved on smallest of the two
plt.rcParams['figure.figsize'] = 10, 10  # that's default image size for this interactive session
plt.rcParams['axes.linewidth'] = 2.0 #set the value globally
plt.rcParams['font.weight'] = 'bold' #set the font weight globally

# CL True means save figures to .png; False means display figures
CL = True
#CL = False

##################################
# END:  set user parameters here
##################################



workdir = datadir_output + '/' + nexp

# get directory and information for later use

import glob, os, fnmatch
print workdir

# get number of mc realizations from directory count
mcdir = os.listdir(workdir)
print mcdir

#niters = len(mcdir)
# since some files may not be iteration subdirectories; count those that are
niters = 0
for subdir in mcdir:
    if os.path.isdir(workdir+'/'+subdir):
        niters = niters + 1
    
print 'niters = ' + str(niters)

# get a listing of the analysis files
files = glob.glob(workdir+'/'+mcdir[0]+'/year*')
print 'files='+str(files)
years = []
k = -1
for f in files:
    k = k + 1
    i = f.find('year')
    year = f[i+4:i+8]
    years.append(year)

# process the analysis files
nyears = len(files)
print nyears
#print min(years)
#print max(years)


# In[224]:

# query file for assimilated proxy information (for now, ONLY IN THE r0 directory!)

ptypes,nrecords = assimilated_proxies(workdir+'/r0/')
print workdir

print 'Assimilated proxies by type:'
for pt in ptypes.keys():
    print pt + ': ' + str(ptypes[pt])
                
print 'Total: ' + str(nrecords)


# In[225]:

# load GISTEMP, HadCRU, and HadCET

datadir_calib = '../data/'

from load_gridded_data import read_gridded_data_GISTEMP
from load_gridded_data import read_gridded_data_HadCRUT
from load_gridded_data import read_gridded_data_BerkeleyEarth
from netCDF4 import Dataset
from datetime import datetime, timedelta
from LMR_plot_support import *

# load GISTEMP
datafile_calib   = 'gistemp1200_ERSST.nc'
calib_vars = ['Tsfc']
[GIS_time,GIS_lat,GIS_lon,GIS_anomaly] = read_gridded_data_GISTEMP(datadir_calib,datafile_calib,calib_vars)
nlat_GIS = len(GIS_lat)
nlon_GIS = len(GIS_lon)
print GIS_lat[0]

# load HadCRU
datafile_calib   = 'HadCRUT.4.3.0.0.median.nc'
calib_vars = ['Tsfc']
[CRU_time,CRU_lat,CRU_lon,CRU_anomaly] = read_gridded_data_HadCRUT(datadir_calib,datafile_calib,calib_vars)
print CRU_lat[0]

# load BerkeleyEarth
datafile_calib   = 'Land_and_Ocean_LatLong1.nc'
calib_vars = ['Tsfc']
[BE_time,BE_lat,BE_lon,BE_anomaly] = read_gridded_data_BerkeleyEarth(datadir_calib,datafile_calib,calib_vars)

# load NOAA MLOST
path = '/home/disk/ice4/hakim/svnwork/python-lib/trunk/src/ipython_notebooks/data/NOAA_MLOST/'
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

infile = '/home/disk/ice4/hakim/data/20th_century_reanalysis_v2/T_0.995/air.sig995.mon.mean.nc'
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
TCR_time.sort # sort the list

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
stime = 1900
etime = 1999
smatch, ematch = find_date_indices(TCR_time,stime,etime)
tcr_gm = tcr_gm - np.mean(tcr_gm[smatch:ematch])

# load HadCET
#tmp_years,tmp_T = load_HadCET()
#HadCET_years = tmp_years[0:-2] 
#HadCET_T = tmp_T[0:-2]
#HadCET_T_mean = np.mean(HadCET_T)
# option to recenter the temperature field around the time mean (anomalies)
#HadCET_T = HadCET_T - HadCET_T_mean


#
# read LMR GMT data computed during DA
#

# define a sample by the range of experiments (Monte Carlo realizations)
exps = range(niters)

kk = -1
for k in exps:
    kk = kk + 1

    epath = workdir + '/r' + str(k)

    gmtpfile =  epath + '/gmt.npz'
    npzfile = np.load(gmtpfile)
    npzfile.files
    gmt = npzfile['gmt_save']
    recon_times = npzfile['recon_times']
    apcount = npzfile['apcount']
    print gmtpfile
    print apcount
    gmt_shape = np.shape(gmt)
    if kk == 0:
        gmt_save = np.zeros([len(exps),gmt_shape[1]])

    gmt_save[kk,:] = gmt[apcount,:]
       
    if iplot:
        plt.plot(recon_times,gmt[apcount,:], label=str(kk), alpha=alpha)
    
    # get assimilated proxy data too
    #apfile = epath + '/assimilated_proxies.npy'
    #assimilated_proxies = np.load(apfile)

# sample average GMT
sagmt = np.average(gmt_save,0)
gmt_min = np.min(gmt_save,0)
gmt_max = np.max(gmt_save,0)
if iplot:
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
    plot_direction(CL,nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])) # use this to save a .png
    
# define for later use
lmr_gm = sagmt
LMR_time = recon_times

# 
# compute GIS & CRU global mean and plot
#

gis_gm = global_mean(GIS_anomaly,GIS_lat,GIS_lon)
cru_gm = global_mean(CRU_anomaly,CRU_lat,CRU_lon)
be_gm = global_mean(BE_anomaly,BE_lat,BE_lon)

print MLOST_time
print GIS_time

# adjust so that all time series pertain to 20th century mean
stime = 1900; etime = 1999
#stime = 1951; etime = 1980

smatch, ematch = find_date_indices(LMR_time,stime,etime)
lmr_off = np.mean(lmr_gm[smatch:ematch])
lmr_gm = lmr_gm - lmr_off
# fix previously set values
gmt_min = gmt_min - lmr_off
gmt_max = gmt_max - lmr_off
smatch, ematch = find_date_indices(GIS_time,stime,etime)
gis_gm = gis_gm - np.mean(gis_gm[smatch:ematch])
smatch, ematch = find_date_indices(CRU_time,stime,etime)
cru_gm = cru_gm - np.mean(cru_gm[smatch:ematch])
smatch, ematch = find_date_indices(BE_time,stime,etime)
be_gm = be_gm - np.mean(be_gm[smatch:ematch])
smatch, ematch = find_date_indices(MLOST_time,stime,etime)
mlost_gm = mlost_gm - np.mean(mlost_gm[smatch:ematch])

# plots
if iplot:
    lw = 1
    plt.plot(LMR_time,lmr_gm,'k-',linewidth=2,label='LMR')
    plt.plot(GIS_time,gis_gm,'r-',linewidth=lw,label='GIS',alpha=alpha)
    plt.plot(CRU_time,cru_gm,'m-',linewidth=lw,label='CRU',alpha=alpha)
    plt.plot(TCR_time,tcr_gm,'y-',linewidth=lw,label='TCR',alpha=alpha)
    plt.plot(BE_time,be_gm,'g-',linewidth=lw,label='BE',alpha=alpha)
    plt.plot(MLOST_time,mlost_gm,'c-',linewidth=lw,label='MLOST',alpha=alpha)
     #plt.fill_between(recon_times,gmt_min,gmt_max,facecolor='gray',alpha = 0.5,linewidth=0.)
    
    # indices for chosen time interval defined by stime and etime
    stime = 1880
    etime = 2000
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
    plt.plot(CON_time,con_gm,'r-',linewidth=lw*2,label='consensus',alpha=alpha)
            
    # correlation coefficients for a chosen time interval 
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
    print 'LMR_GIS correlation: ' + lgc
    print 'LMR_CRU correlation: ' + lcc
    print 'LMR_TCR correlation: ' + ltc
    print 'LMR_BE  correlation: ' + lbc
    print 'LMR_MLOST  correlation: ' + lmc
    print 'LMR_CON  correlation: ' + loc
    print 'GIS_CRU correlation: ' + gcc
    print 'GIS_TCR correlation: ' + gtc
    print 'GIS_BE correlation: ' + gbc

    # add correlation info to plot title
    #plt.title('sample-mean and range GMT for ' + str(kk+1) + ' samples (exp: ' + nexp[0:-2] + ')')
    #plt.title('GMT (' + nexp + ')' ' r(lmr,gis)=' + lgc + ' r(lmr,cru)=' + lcc + ' r(lmr,tcr)=' + ltc,weight='bold')
    plt.title('GMT (' + nexp + ')',weight='bold')

    xl_loc = [1880,2000]
    yl_loc = [-.7,.7]
    plt.xlim(xl_loc)
    plt.ylim(yl_loc)
    txl = xl_loc[0] + (xl_loc[1]-xl_loc[0])*.01
    tyl = yl_loc[0] + (yl_loc[1]-yl_loc[0])*.05
    plt.text(txl,tyl,ptypes,fontsize=7)
    plt.text(txl,tyl-0.05,str(kk+1) + ' samples',fontsize=7)
    corrstr = 'correlations: r(lmr,gis)=' + lgc + '   r(lmr,cru)=' + lcc + '   r(lmr,tcr)=' + ltc + '   r(lmr,be)=' + lbc+ '   r(lmr,mlost)=' + lmc + '   r(lmr,con)=' + loc
    print corrstr
    plt.text(txl,tyl+0.05,corrstr,fontsize=7)
    plt.plot(xl_loc,[0,0])
    plt.legend(loc=2)

    plot_direction(CL,nexp+'_GMT_LMR_GIS_CRU_TCR_BE_MLOST_comparison') 


if iplot:
    #plt.plot(recon_times,lmr_gm,'k-',linewidth=2)
    plt.fill_between(recon_times,gmt_min,gmt_max,facecolor='gray',alpha=alpha,linewidth=0.)

    # add smoothed lines
    #nsyrs = 31 # 31-> 31-year running mean--nsyrs must be odd!
    nsyrs = 5 # 5-> 5-year running mean--nsyrs must be odd!
    LMR_smoothed,LMR_smoothed_years = moving_average(lmr_gm,recon_times,nsyrs) 
    plt.plot(LMR_smoothed_years,LMR_smoothed,'k-',linewidth=4, label='LMR')
    GIS_smoothed,GIS_smoothed_years = moving_average(gis_gm,GIS_time,nsyrs) 
    plt.plot(GIS_smoothed_years,GIS_smoothed,'r-',linewidth=4, label='GIS',alpha=alpha)
    CRU_smoothed,CRU_smoothed_years = moving_average(cru_gm,CRU_time,nsyrs) 
    plt.plot(CRU_smoothed_years,CRU_smoothed,'m-',linewidth=4, label='CRU',alpha=alpha)
    TCR_smoothed,TCR_smoothed_years = moving_average(tcr_gm,TCR_time,nsyrs) 
    plt.plot(TCR_smoothed_years,TCR_smoothed,'y-',linewidth=4, label='TCR',alpha=alpha)
    BE_smoothed,BE_smoothed_years = moving_average(be_gm,BE_time,nsyrs) 
    plt.plot(BE_smoothed_years,BE_smoothed,'g-',linewidth=4, label='BE',alpha=alpha)
    MLOST_smoothed,MLOST_smoothed_years = moving_average(mlost_gm,MLOST_time,nsyrs) 
    plt.plot(MLOST_smoothed_years,MLOST_smoothed,'c-',linewidth=4, label='MLOST',alpha=alpha)

    # correlation coefficients for a chosen time interval 
    stime = 1890
    etime = 1990
    ls_smatch, ls_ematch = find_date_indices(LMR_smoothed_years,stime,etime)
    gs_smatch, gs_ematch = find_date_indices(GIS_smoothed_years,stime,etime)   
    ls_gs_corr = np.corrcoef(LMR_smoothed[ls_smatch:ls_ematch],GIS_smoothed[gs_smatch:gs_ematch])
    lscg = str(float('%.2g' % ls_gs_corr[0,1]))
    print 'smoothed lmr-gis correlation = ' + lscg
    cs_smatch, cs_ematch = find_date_indices(CRU_smoothed_years,stime,etime)   
    ls_cs_corr = np.corrcoef(LMR_smoothed[ls_smatch:ls_ematch],CRU_smoothed[cs_smatch:cs_ematch])
    lscc = str(float('%.2g' % ls_cs_corr[0,1]))
    print 'smoothed lmr-cru correlation = ' + lscc
    ts_smatch, ts_ematch = find_date_indices(TCR_smoothed_years,stime,etime)   
    ls_ts_corr = np.corrcoef(LMR_smoothed[ls_smatch:ls_ematch],TCR_smoothed[ts_smatch:ts_ematch])
    lsct = str(float('%.2g' % ls_ts_corr[0,1]))
    print 'smoothed lmr-tcr correlation = ' + lsct
    bs_smatch, bs_ematch = find_date_indices(BE_smoothed_years,stime,etime)   
    ls_bs_corr = np.corrcoef(LMR_smoothed[ls_smatch:ls_ematch],BE_smoothed[bs_smatch:bs_ematch])
    lscb = str(float('%.2g' % ls_bs_corr[0,1]))
    print 'smoothed lmr-be correlation = ' + lscb
    ms_smatch, ms_ematch = find_date_indices(MLOST_smoothed_years,stime,etime)   
    ls_ms_corr = np.corrcoef(LMR_smoothed[ls_smatch:ls_ematch],MLOST_smoothed[ms_smatch:ms_ematch])
    lscm = str(float('%.2g' % ls_ms_corr[0,1]))
    print 'smoothed lmr-mlost correlation = ' + lscm

    # smoothed "consensus" global mean: average all non-LMR values
    #scon_gm = np.mean([GIS_smoothed[gs_smatch:gs_ematch],CRU_smoothed[cs_smatch:cs_ematch],BE_smoothed[bs_smatch:bs_ematch]],axis=0)
    #CON_time = range(stime,etime)
    #plt.plot(CON_time,scon_gm,'k-',linewidth=lw*2,label='consensus')

    
    plt.legend(loc=2)
    #xl = [1500,2000]
    #xl = [1000,2000]
    xl = [1880,2000]
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
    fname = nexp+'_GMT_'+str(xl[0])+'-'+str(xl[1])+'_'+str(nsyrs)+'yr_smoothed'
    print fname
    plt.savefig(fname)
    
