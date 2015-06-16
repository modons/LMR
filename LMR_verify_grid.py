#
# verify statistics related to the global-mean 2m air temperature
#
# started from LMR_plots.py r-86
#
# Need to fix anomaly fields so they apply to the same time period!!! Write a function to do this.
#
# CRU data too coarse to include in the analysis
#
# code not yet debugged!

import matplotlib
# need to do this when running remotely
#matplotlib.use('Agg')

# generic imports
import numpy as np
import glob, os
import mpl_toolkits.basemap as bm
import matplotlib.pyplot as plt
from random import sample
from netCDF4 import Dataset
from datetime import datetime, timedelta
from matplotlib import ticker
from spharm import Spharmt, getspecindx, regrid

# LMR specific imports
from LMR_utils import global_mean, assimilated_proxies
from load_gridded_data import read_gridded_data_GISTEMP
from load_gridded_data import read_gridded_data_HadCRUT
from load_gridded_data import read_gridded_data_BerkeleyEarth
from LMR_plot_support import *
from LMR_exp_NAMELIST import *
from LMR_plot_support import *
from load_proxy_data import create_proxy_lists_from_metadata_S1csv as create_proxy_lists_from_metadata
from LMR_exp_NAMELIST import *

# change default value of latlon kwarg to True.
bm.latlon_default = True

##################################
# START:  set user parameters here
##################################

# option to suppress figures
iplot = False
#iplot = True

# centered time mean (nya must be odd! 3 = 3 yr mean; 5 = 5 year mean; etc 0 = none)
nya = 0

# CL True means save figures to .png; False means display figures
CL = True
#CL = False

# set paths, the filename for plots, and global plotting preferences

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

# number of contours for plots
nlevs = 30

# plot alpha transparency
alpha = 0.5

# time range for verification (in years CE)
#trange = [1960,1962]
trange = [1900,1910]

# set the default size of the figure in inches. ['figure.figsize'] = width, height;  
# aspect ratio appears preserved on smallest of the two
plt.rcParams['figure.figsize'] = 10, 10  # that's default image size for this interactive session
plt.rcParams['axes.linewidth'] = 2.0 #set the value globally
plt.rcParams['font.weight'] = 'bold' #set the font weight globally
#plt.rc('text', usetex=True)
plt.rc('text', usetex=False)

##################################
# END:  set user parameters here
##################################

workdir = datadir_output + '/' + nexp
print 'working directory = ' + workdir

print '\n getting file system information...\n'

# get number of mc realizations from directory count
tmp = os.listdir(workdir)
# since some files may not be iteration subdirectories; count those that are
niters = 0
mcdir = []
for subdir in tmp:
    if os.path.isdir(workdir+'/'+subdir):
        niters = niters + 1
        mcdir.append(subdir)
        
print 'mcdir:' + str(mcdir)

# get time period from the GMT file...
gmtpfile =  workdir + '/r0/gmt.npz'
npzfile = np.load(gmtpfile)
npzfile.files
LMR_time = npzfile['recon_times']
print 'time = ' + str(LMR_time)

# get grid information from the prior file...
prior_filn = workdir + '/r0/Xb_one.npz'
npzfile = np.load(prior_filn)
npzfile.files
lat = npzfile['lat']
lon = npzfile['lon']
nlat = npzfile['nlat']
nlon = npzfile['nlon']
lat2 = np.reshape(lat,(nlat,nlon))
lon2 = np.reshape(lon,(nlat,nlon))

# read ensemble mean data
print '\n reading LMR ensemble-mean data...\n'

first = True
for dir in mcdir:
    ensfiln = workdir + '/' + dir + '/ensemble_mean.npz'
    print ensfiln
    npzfile = np.load(ensfiln)
    npzfile.files
    tmp = npzfile['xam']
    if first:
        first = False
        xam = np.zeros(np.shape(tmp))
    xam = xam + tmp

xam = xam/len(mcdir)
print 'shape of the ensemble-mean array: ' + str(np.shape(xam))

#################################################################
# BEGIN: load verification data (GISTEMP, HadCRU, BE, and 20CR) #
#################################################################
print '\nloading verification data...\n'

datadir_calib = '../data/'

# load GISTEMP
print 'loading GISTEMP...'
datafile_calib   = 'gistemp1200_ERSST.nc'
calib_vars = ['Tsfc']
[GIS_time,GIS_lat,GIS_lon,GIS_anomaly] = read_gridded_data_GISTEMP(datadir_calib,datafile_calib,calib_vars)
nlat_GIS = len(GIS_lat)
nlon_GIS = len(GIS_lon)
lon2_GIS, lat2_GIS = np.meshgrid(GIS_lon, GIS_lat)

# load HadCRU
print 'loading HadCRU...'
datafile_calib   = 'HadCRUT.4.3.0.0.median.nc'
calib_vars = ['Tsfc']
[CRU_time,CRU_lat,CRU_lon,CRU_anomaly] = read_gridded_data_HadCRUT(datadir_calib,datafile_calib,calib_vars)

# load BerkeleyEarth
print 'loading BerkeleyEarth...'
datafile_calib   = 'Land_and_Ocean_LatLong1.nc'
calib_vars = ['Tsfc']
[BE_time,BE_lat,BE_lon,BE_anomaly] = read_gridded_data_BerkeleyEarth(datadir_calib,datafile_calib,calib_vars)
nlat_BE = len(BE_lat)
nlon_BE = len(BE_lon)
lon2_BE, lat2_BE = np.meshgrid(BE_lon, BE_lat)

# load 20th century reanalysis (this is copied from R. Tardif's load_gridded_data.py routine)
print 'loading 20th century reanalysis...'
infile = '/home/disk/ice4/hakim/data/20th_century_reanalysis_v2/T_0.995/air.sig995.mon.mean.nc'
#infile = './data/500_allproxies_0/air.sig995.mon.mean.nc'

data = Dataset(infile,'r')
lat_20CR   = data.variables['lat'][:]
lon_20CR   = data.variables['lon'][:]
nlat_20CR = len(lat_20CR)
nlon_20CR = len(lon_20CR)
lon2_TCR, lat2_TCR = np.meshgrid(lon_20CR, lat_20CR)
 
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
print 'TCR shape = ' + str(np.shape(TCR))

# compute and remove the 20th century mean
stime = 1900
etime = 1999
smatch, ematch = find_date_indices(TCR_time,stime,etime)
tcr_gm = tcr_gm - np.mean(tcr_gm[smatch:ematch])

###############################################################
# END: load verification data (GISTEMP, HadCRU, BE, and 20CR) #
###############################################################

print '\n regridding data to a common T42 grid...\n'

# create instance of the spherical harmonics object for each grid
specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')
specob_tcr = Spharmt(nlon_20CR,nlat_20CR,gridtype='regular',legfunc='computed')
specob_gis = Spharmt(nlon_GIS,nlat_GIS,gridtype='regular',legfunc='computed')
specob_be = Spharmt(nlon_BE,nlat_BE,gridtype='regular',legfunc='computed')

# truncate to a lower resolution grid (common:21, 42, 62, 63, 85, 106, 255, 382, 799)
ntrunc_new = 42 # T21
ifix = np.remainder(ntrunc_new,2.0).astype(int)
nlat_new = ntrunc_new + ifix
nlon_new = int(nlat_new*1.5)
# lat, lon grid in the truncated space
dlat = 90./((nlat_new-1)/2.)
dlon = 360./nlon_new
veclat = np.arange(-90.,90.+dlat,dlat)
veclon = np.arange(0.,360.,dlon)
blank = np.zeros([nlat_new,nlon_new])
lat2_new = (veclat + blank.T).T  
lon2_new = (veclon + blank)  

# create instance of the spherical harmonics object for the new grid
specob_new = Spharmt(nlon_new,nlat_new,gridtype='regular',legfunc='computed')

nyrs = trange[1] - trange[0]
lmr_trunc = np.zeros([nyrs,nlat_new,nlon_new])
print 'lmr_trunc shape: ' + str(np.shape(lmr_trunc))

# loop over years of interest and transform...specify trange at top of file

iw = 0
if nya > 0:
    iw = (nya-1)/2

cyears = range(trange[0],trange[1])
lt_csave = np.zeros([len(cyears)])
lg_csave = np.zeros([len(cyears)])
tg_csave = np.zeros([len(cyears)])

lmr_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
tcr_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
gis_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
for yr in cyears:
    LMR_smatch, LMR_ematch = find_date_indices(LMR_time,yr-iw,yr+iw+1)
    TCR_smatch, TCR_ematch = find_date_indices(TCR_time,yr-iw,yr+iw+1)
    GIS_smatch, GIS_ematch = find_date_indices(GIS_time,yr-iw,yr+iw+1)
    BE_smatch, BE_ematch = find_date_indices(BE_time,yr-iw,yr+iw+1)
    print 'working on year...' + str(yr) + ' LMR index = ' + str(LMR_smatch) + ' = LMR year ' + str(LMR_time[LMR_smatch])
    print 'working on year...' + str(yr) + ' TCR index = ' + str(TCR_smatch) + ' = TCR year ' + str(TCR_time[TCR_smatch])
    print 'working on year...' + str(yr) + ' GIS index = ' + str(GIS_smatch) + ' = GIS year ' + str(GIS_time[GIS_smatch])
    print 'working on year...' + str(yr) + ' BE index = ' + str(BE_smatch) + ' = BE year ' + str(BE_time[BE_smatch])

    # LMR
    pdata_lmr = np.mean(xam[LMR_smatch:LMR_ematch,:,:],0)    
    #pdata_lmr = np.squeeze(xam[LMR_smatch,:,:])
    lmr_trunc = regrid(specob_lmr, specob_new, pdata_lmr, ntrunc=nlat_new-1, smooth=None)
    #print 'shape of old LMR data array:' + str(np.shape(pdata_lmr))
    #print 'shape of new LMR data array:' + str(np.shape(lmr_trunc))

    # TCR
    pdata_tcr = np.mean(TCR[TCR_smatch:TCR_ematch,:,:],0)    
    #pdata_tcr = np.squeeze(TCR[TCR_smatch,:,:])
    tcr_trunc = regrid(specob_tcr, specob_new, pdata_tcr, ntrunc=nlat_new-1, smooth=None)
    # TCR latitudes upside down
    tcr_trunc = np.flipud(tcr_trunc)
    #print 'shape of old TCR data array:' + str(np.shape(pdata_tcr))
    #print 'shape of new TCR data array:' + str(np.shape(tcr_trunc))

    # GIS
    pdata_gis = np.mean(GIS_anomaly[GIS_smatch:GIS_ematch,:,:],0)    
    #pdata_gis = np.squeeze(np.nan_to_num(GIS_anomaly[GIS_smatch,:,:]))
    gis_trunc = regrid(specob_gis, specob_new, np.nan_to_num(pdata_gis), ntrunc=nlat_new-1, smooth=None)
    # GIS logitudes are off by 180 degrees
    gis_trunc = np.roll(gis_trunc,shift=nlon_new/2,axis=1)
    #print 'shape of old GIS data array:' + str(np.shape(pdata_gis))
    #print 'shape of new GIS data array:' + str(np.shape(gis_trunc))

    # BE
    pdata_be = np.mean(BE_anomaly[BE_smatch:BE_ematch,:,:],0)    
    #pdata_be = np.squeeze(np.nan_to_num(BE_anomaly[BE_smatch,:,:]))
    be_trunc = regrid(specob_be, specob_new, np.nan_to_num(pdata_be), ntrunc=nlat_new-1, smooth=None)
    # BE logitudes are off by 180 degrees
    be_trunc = np.roll(be_trunc,shift=nlon_new/2,axis=1)
    #print 'shape of old BE data array:' + str(np.shape(pdata_be))
    #print 'shape of new BE data array:' + str(np.shape(be_trunc))

    if iplot:
        ncints = 30
        cmap = 'bwr'
        nticks = 6 # number of ticks on the colorbar
        #set contours based on Berkeley Earth
        maxabs = np.nanmax(np.abs(be_trunc))
        # round the contour interval, and then set limits to fit
        dc = np.round(maxabs*2/ncints,2)
        cl = dc*ncints/2.
        cints = np.linspace(-cl,cl,ncints,endpoint=True)
        
        # compare LMR and TCR and GIS and BE
        fig = plt.figure()
        
        ax = fig.add_subplot(2,2,1)
        m1 = bm.Basemap(projection='robin',lon_0=0)
        # maxabs = np.nanmax(np.abs(lmr_trunc))
        cs = m1.contourf(lon2_new,lat2_new,lmr_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
        m1.drawcoastlines()
        cb = m1.colorbar(cs)
        tick_locator = ticker.MaxNLocator(nbins=nticks)
        cb.locator = tick_locator
        cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        cb.update_ticks()
        ax.set_title('LMR T'+str(ntrunc_new) + ' ' + str(yr))
        
        ax = fig.add_subplot(2,2,2)
        m2 = bm.Basemap(projection='robin',lon_0=0)
        # maxabs = np.nanmax(np.abs(tcr_trunc))
        cs = m2.contourf(lon2_new,lat2_new,tcr_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
        m2.drawcoastlines()
        cb = m1.colorbar(cs)
        tick_locator = ticker.MaxNLocator(nbins=nticks)
        cb.locator = tick_locator
        cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        cb.update_ticks()
        ax.set_title('TCR T'+str(ntrunc_new)+ ' ' + str(yr))
        
        ax = fig.add_subplot(2,2,3)
        m3 = bm.Basemap(projection='robin',lon_0=0)
        # maxabs = np.nanmax(np.abs(gis_trunc))
        cs = m3.contourf(lon2_new,lat2_new,gis_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
        m3.drawcoastlines()
        cb = m1.colorbar(cs)
        tick_locator = ticker.MaxNLocator(nbins=nticks)
        cb.locator = tick_locator
        cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        cb.update_ticks()
        ax.set_title('GIS T'+str(ntrunc_new)+ ' ' + str(yr))
        
        ax = fig.add_subplot(2,2,4)
        m4 = bm.Basemap(projection='robin',lon_0=0)
        # maxabs = np.nanmax(np.abs(be_trunc))
        cs = m2.contourf(lon2_new,lat2_new,be_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
        m4.drawcoastlines()
        cb = m1.colorbar(cs)
        tick_locator = ticker.MaxNLocator(nbins=nticks)
        cb.locator = tick_locator
        cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        cb.update_ticks()
        ax.set_title('BE T'+str(ntrunc_new)+ ' ' + str(yr))
        plt.clim(-maxabs,maxabs)
        
        # get these numbers by adjusting the figure interactively!!!
        plt.subplots_adjust(left=0.05, bottom=0.45, right=0.95, top=0.95, wspace=0.1, hspace=0.0)
        # plt.tight_layout(pad=0.3)
        fig.suptitle('2m air temperature for ' +str(nya) +' year centered average')
        plt.show()
    
    # anomaly correlation
    lmrvec = np.reshape(lmr_trunc,(1,nlat_new*nlon_new))
    tcrvec = np.reshape(tcr_trunc,(1,nlat_new*nlon_new))
    gisvec = np.reshape(gis_trunc,(1,nlat_new*nlon_new))
    bevec = np.reshape(be_trunc,(1,nlat_new*nlon_new))
    lmr_tcr_corr = np.corrcoef(lmrvec,tcrvec)
    print 'lmr-tcr correlation: '+str(lmr_tcr_corr[0,1])
    lmr_gis_corr = np.corrcoef(lmrvec,gisvec)
    print 'lmr-gis correlation: '+ str(lmr_gis_corr[0,1])
    lmr_be_corr = np.corrcoef(lmrvec,bevec)
    print 'lmr-be correlation: '+ str(lmr_be_corr[0,1])
    tcr_gis_corr = np.corrcoef(tcrvec,gisvec)
    print 'gis-tcr correlation: '+ str(tcr_gis_corr[0,1])
    be_gis_corr = np.corrcoef(bevec,gisvec)
    print 'gis-be correlation: '+ str(be_gis_corr[0,1])

    # accumulate anomaly correlations in arrays/lists, then do time series stats (r and CE)

#...working here...

# loop over all years and compute anomaly correlation--USES specob objects from previous block!

# centered time mean (nya must be odd! 3 = 3 yr mean; 5 = 5 year mean; etc 0 = none)
nya = 3
iw = 0
if nya > 0:
    iw = (nya-1)/2

cyears = range(1960,1969-iw)
lt_csave = np.zeros([len(cyears)])
lg_csave = np.zeros([len(cyears)])
tg_csave = np.zeros([len(cyears)])

lmr_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
tcr_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
gis_allyears = np.zeros([len(cyears),nlat_new,nlon_new])

k = -1
for cyear in cyears:
    k = k + 1

    # LMR
    sind, send = find_date_indices(LMR_time,cyear-iw,cyear+iw+1)
    pdata_lmr = np.mean(xam[sind:send,:,:],0)    
    lmr_trunc = regrid(specob_lmr, specob_new, pdata_lmr, ntrunc=nlat_new-1, smooth=None)
    # TCR
    sind, send = find_date_indices(TCR_time,cyear-iw,cyear+iw+1)
    pdata_tcr = np.mean(TCR[sind:send,:,:],0)
    tcr_trunc = regrid(specob_tcr, specob_new, pdata_tcr, ntrunc=nlat_new-1, smooth=None)
    tcr_trunc = np.flipud(tcr_trunc)
    # GIS
    sind, send = find_date_indices(TCR_time,cyear-iw,cyear+iw+1)
    pdata_gis = np.nanmean(GIS_anomaly[sind:send,:,:],0)
    specob_gis = Spharmt(nlon_GIS,nlat_GIS,gridtype='regular',legfunc='computed')
    gis_trunc = regrid(specob_gis, specob_new, np.nan_to_num(pdata_gis), ntrunc=nlat_new-1, smooth=None)
    gis_trunc = np.roll(gis_trunc,shift=nlon_new/2,axis=1)

    lmrvec = np.reshape(lmr_trunc,(1,nlat_new*nlon_new))
    tcrvec = np.reshape(tcr_trunc,(1,nlat_new*nlon_new))
    gisvec = np.reshape(gis_trunc,(1,nlat_new*nlon_new))

    lmr_tcr_corr = np.corrcoef(lmrvec,tcrvec)
    lt_csave[k] = lmr_tcr_corr[0,1]
    lmr_gis_corr = np.corrcoef(lmrvec,gisvec)
    lg_csave[k] = lmr_gis_corr[0,1]
    tcr_gis_corr = np.corrcoef(tcrvec,gisvec)
    tg_csave[k] = tcr_gis_corr[0,1]

    lmr_allyears[k,:,:] = lmr_trunc
    tcr_allyears[k,:,:] = tcr_trunc
    gis_allyears[k,:,:] = gis_trunc
  
    # compare LMR and TCR and GIS and BE
    fig = plt.figure()
    
    ax = fig.add_subplot(2,2,1)
    m1 = bm.Basemap(projection='robin',lon_0=0)
    # maxabs = np.nanmax(np.abs(lmr_trunc))
    cs = m1.contourf(lon2_new,lat2_new,lmr_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
    m1.drawcoastlines()
    cb = m1.colorbar(cs)
    tick_locator = ticker.MaxNLocator(nbins=nticks)
    cb.locator = tick_locator
    cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    cb.update_ticks()
    ax.set_title('LMR T'+str(ntrunc_new) + ' ' + str(cyear))
    
    ax = fig.add_subplot(2,2,2)
    m2 = bm.Basemap(projection='robin',lon_0=0)
    # maxabs = np.nanmax(np.abs(tcr_trunc))
    cs = m2.contourf(lon2_new,lat2_new,tcr_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
    m2.drawcoastlines()
    cb = m1.colorbar(cs)
    tick_locator = ticker.MaxNLocator(nbins=nticks)
    cb.locator = tick_locator
    cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    cb.update_ticks()
    ax.set_title('TCR T'+str(ntrunc_new)+ ' ' + str(cyear))
    
    ax = fig.add_subplot(2,2,3)
    m3 = bm.Basemap(projection='robin',lon_0=0)
    # maxabs = np.nanmax(np.abs(gis_trunc))
    cs = m3.contourf(lon2_new,lat2_new,gis_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
    m3.drawcoastlines()
    cb = m1.colorbar(cs)
    tick_locator = ticker.MaxNLocator(nbins=nticks)
    cb.locator = tick_locator
    cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    cb.update_ticks()
    ax.set_title('GIS T'+str(ntrunc_new)+ ' ' + str(cyear))
    
    ax = fig.add_subplot(2,2,4)
    m4 = bm.Basemap(projection='robin',lon_0=0)
    # maxabs = np.nanmax(np.abs(be_trunc))
    cs = m2.contourf(lon2_new,lat2_new,be_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
    m4.drawcoastlines()
    cb = m1.colorbar(cs)
    tick_locator = ticker.MaxNLocator(nbins=nticks)
    cb.locator = tick_locator
    cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    cb.update_ticks()
    ax.set_title('BE T'+str(ntrunc_new)+ ' ' + str(cyear))
    plt.clim(-maxabs,maxabs)
    
    # get these numbers by adjusting the figure interactively!!!
    plt.subplots_adjust(left=0.05, bottom=0.4, right=0.95, top=0.9, wspace=0.1, hspace=0.0)
    # plt.tight_layout(pad=0.3)
    plt.show()
    
    
if iplot:
    plt.plot(cyears,lt_csave)
    plt.title('LMR-TCR time series of annual-mean anomaly correlation')
    plt.show()
    plt.hist(lt_csave,bins=10)
    plt.title('LMR-TCR distribution of annual-mean anomaly correlation')
    plt.show()
    plt.plot(cyears,lg_csave)
    plt.title('LMR-GIS time series of annual-mean anomaly correlation')
    plt.show()
    #plt.hist(lg_csave,bins=10)
    #plt.title('LMR-GIS distribution of annual-mean anomaly correlation')
    #plt.show()
  

print '\n computing correlation and CE at each point...\n'

#
# BEGIN r and CE
#

# correlation and CE at each (lat,lon) point

lt_err = lmr_allyears - tcr_allyears
lg_err = lmr_allyears - gis_allyears
tg_err = tcr_allyears - gis_allyears

r_lt = np.zeros([nlat_new,nlon_new])
ce_lt = np.zeros([nlat_new,nlon_new])
r_lg = np.zeros([nlat_new,nlon_new])
ce_lg = np.zeros([nlat_new,nlon_new])
r_tg = np.zeros([nlat_new,nlon_new])
ce_tg = np.zeros([nlat_new,nlon_new])
for la in range(nlat_new):
    for lo in range(nlon_new):
        # LMR-TCR
        tstmp = np.corrcoef(lmr_allyears[:,la,lo],tcr_allyears[:,la,lo])
        evar = np.var(lt_err[:,la,lo],ddof=1)
        tvar = np.var(tcr_allyears[:,la,lo],ddof=1)
        r_lt[la,lo] = tstmp[0,1]
        ce_lt[la,lo] = 1. - (evar/tvar)
        # LMR-GIS
        tstmp = np.corrcoef(lmr_allyears[:,la,lo],gis_allyears[:,la,lo])
        evar = np.var(lg_err[:,la,lo],ddof=1)
        tvar = np.var(gis_allyears[:,la,lo],ddof=1)
        r_lg[la,lo] = tstmp[0,1]
        ce_lg[la,lo] = 1. - (evar/tvar)
        # TCR-GIS
        tstmp = np.corrcoef(tcr_allyears[:,la,lo],gis_allyears[:,la,lo])
        evar = np.var(tg_err[:,la,lo],ddof=1)
        tvar = np.var(gis_allyears[:,la,lo],ddof=1)
        r_tg[la,lo] = tstmp[0,1]
        ce_tg[la,lo] = 1. - (evar/tvar)
   
lt_rmean = str(float('%.2g' % np.mean(np.median(r_lt)) ))
print lt_rmean
lt_cemean = str(float('%.2g' % np.mean(np.median(ce_lt)) ))
print lt_cemean
lg_rmean = str(float('%.2g' % np.mean(np.median(r_lg)) ))
print lg_rmean
lg_cemean = str(float('%.2g' % np.mean(np.median(ce_lg)) ))
print lg_cemean
tg_rmean = str(float('%.2g' % np.mean(np.median(r_tg)) ))
print tg_rmean
tg_cemean = str(float('%.2g' % np.mean(np.median(ce_tg)) ))
print tg_cemean

if iplot:
    LMR_plotter(r_lt,lat2_new,lon2_new,'bwr',nlevs)
    plt.title('LMR-TCR T r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lt_rmean))
    plt.show()
    LMR_plot_support.LMR_plotter(ce_lt,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1)
    plt.title('LMR-TCR T CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lt_cemean))
    plt.show()
    LMR_plotter(r_lg,lat2_new,lon2_new,'bwr',nlevs)
    plt.title('LMR-GIS T r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lg_rmean))
    plt.show()
    LMR_plot_support.LMR_plotter(ce_lg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1)
    plt.title('LMR-GIS T CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lg_cemean))
    plt.show()
    LMR_plotter(r_tg,lat2_new,lon2_new,'bwr',nlevs)
    plt.title('TCR-GIS T r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(tg_rmean))
    plt.show()
    LMR_plot_support.LMR_plotter(ce_tg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1)
    plt.title('TCR-GIS T CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(tg_cemean))
    plt.show()


#
# END r and CE
#



# plot time average ensemble and sample mean for a chosen time period

#syear = 1940; eyear = 1950
#syear = 1900; eyear = 1910
#syear = 1930; eyear = 1940
#syear = 1950; eyear = 1960
syear = 1941; eyear = 1942

sind, send = find_date_indices(LMR_time,syear,eyear)
pdata_lmr = np.mean(xam[sind:send,:,:],0)
print np.shape(pdata_lmr)

if iplot:
    LMR_plotter(pdata_lmr,lat2,lon2,'bwr',nlevs)
    plt.title('LMR '+ str(syear)+'-'+str(eyear))
    #plot_direction('True','LMR_S1_50_'+ str(syear)+'-'+str(eyear)) 
    plt.show()

    # compare with TCR
    sind, send = find_date_indices(TCR_time,syear,eyear)
    pdata_tcr = np.mean(TCR[sind:send,:,:],0)
    print np.shape(lat_20CR)
    lonplt, latplt = np.meshgrid(lon_20CR, lat_20CR)
    LMR_plotter(pdata_tcr,latplt,lonplt,'bwr',nlevs)
    plt.title('20th Century Reanalysis ' + str(syear)+'-'+str(eyear))
    #plot_direction('True','TCR_'+ str(syear)+'-'+str(eyear)) 
    #plot_direction(CL)
    
    # compare with GIS (NaNa are a problem here)
    sind, send = find_date_indices(GIS_time,syear,eyear)
    pdata_gis = np.nanmean(GIS_anomaly[sind:send,:,:],0)
    lonplt, latplt = np.meshgrid(GIS_lon, GIS_lat)
    LMR_plotter(pdata_gis,latplt,lonplt,'bwr',nlevs)
    plt.title('GIS ' + str(syear)+'-'+str(eyear))
    #plot_direction(CL)



# In[231]:


# In[233]:



# In[234]:

#
# composite volcanic eruptions
#

# this is the list from D'Ariggo
vyears = [1601,1641,1810,1816,1884]

# this is from Wikipedia (except I added 1809)
#vyears = [1257,1280,1452,1477,1580,1600,1650,1660,1783,1809,1815,1883,1886,1902,1912]

yr_range = 10
vcomp = np.zeros([2*yr_range + 1, len(vyears)])
print np.shape(vcomp)
k = -1
for yr in vyears:
    k = k + 1
    vstart, vend = find_date_indices(LMR_time,yr-yr_range,yr+yr_range)
    vcomp[:,k] = lmr_gm[vstart:vend+1]

mvcomp = np.mean(vcomp,1)
print np.shape(mvcomp)
mvcomp = mvcomp - np.mean(mvcomp[0:5])
vyrplt = np.arange(-yr_range,yr_range+1)
print vyrplt
print np.shape(vyrplt)
#plt.plot(vyrplt,mvcomp,'k-')
#plt.plot([-10,10],[0,0],'gray')
#plt.plot([0,0],[-.2,0.05],'gray')
#plt.xlabel('years relative to eruption',weight='bold')
#plt.ylabel('global mean temperature relative to years -5:-10',weight='bold')
#plot_direction('True','LMR_S1_50_percent_volcanic')
#
# make a version with stretched aspect ratio for comparison in ppt
#
if iplot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(vyrplt,mvcomp,'r-',linewidth=2)
    ax.set_aspect(30)
    ax.plot([-10,10],[0,0],'gray')
    ax.plot([0,0],[-.2,0.05],'gray')
    plt.xlabel('years relative to eruption',weight='bold')
    plt.ylabel('GMT relative to years -5:-10',weight='bold')
    #plot_direction('True','LMR_S1_50_percent_volcanic')


# In[235]:


print workdir

# composite average volcanic eruptions
lag = 2 # lag in years relative to volcanic eruption
xam_vcomp = np.zeros([len(vyears),nlat,nlon])
xam_vcomp_zero = np.zeros([len(vyears),nlat,nlon])
print np.shape(xam_vcomp)
k = -1
for yr in vyears:
    k = k + 1
    vstart, vend = find_date_indices(LMR_time,yr+lag,yr+yr+lag)
    tmp = np.squeeze(xam[vstart:vstart+1,:,:])
    xam_vcomp[k,:,:] = tmp
# relative to eruption year
    vstart, vend = find_date_indices(LMR_time,yr,yr+yr)
    tmp = np.squeeze(xam[vstart:vstart+1,:,:])
    xam_vcomp_zero[k,:,:] = tmp

print np.shape(xam_vcomp)
# sample mean
mxam_vcomp = np.mean(xam_vcomp,0)
# sample mean at zero lag
mxam_vcomp_zero = np.mean(xam_vcomp_zero,0)


# In[236]:

if iplot:
    print np.shape(lat)
    LMR_plotter(mxam_vcomp,lat2,lon2,'bwr',nlevs)
    #plot_direction(CL)
    LMR_plotter(mxam_vcomp-mxam_vcomp_zero,lat2,lon2,'bwr',nlevs)
    #plot_direction(CL)
  
    #plot_direction('True','LMR_S1_50_percent_volcanic_spatial_lag_'+str(lag))


# In[237]:

# revisions to here...


# In[238]:


# load prior data
npzfile = np.load(prior_filn)
npzfile.files
Xbtmp = npzfile['Xb_one']
nlat = npzfile['nlat']
nlon = npzfile['nlon']
nens = np.size(Xbtmp,1)
print 'nlat, nlon, nens: ' + str(nlat) + ' ' + str(nlon) + ' ' + str(nens)
lat = npzfile['lat']
lon = npzfile['lon']
# reshape arrays
lat2 = np.reshape(lat,(nlat,nlon))
lon2 = np.reshape(lon,(nlat,nlon))
Xb = np.reshape(Xbtmp,(nlat,nlon,nens))
print 'shape of Xb: ' + str(np.shape(Xb))
# ensemble mean
xbm = np.mean(Xb,axis=2)

# analysis data
Xatmp = np.load(analysis_filn)
Xa = np.reshape(Xatmp,(nlat,nlon,nens))
print 'shape of Xa: ' + str(np.shape(Xa))
# ensemble mean
xam = np.mean(Xa,axis=2)

# set plotting variables (first, second, and third plots)
#Xbplt = Xb[:,:,1]
#X1plt = xbm; X2plt = xam
X1plt = Xb[:,:,1]; X2plt = Xa[:,:,1]; 

X3plt = X2plt - X1plt


# In[ ]:

# draw random numbers in the range of the calibration data
ncalib = gis_ntimes
print gis_ntimes
ind_rand = sample(range(0,gis_ntimes),ncalib)
print np.max(ind_rand),np.min(ind_rand)


# In[ ]:

#
# test proxy count code for LMR_driver
#

[sites_assim, sites_eval] = create_proxy_lists_from_metadata(datadir_proxy,datafile_proxy,regions,proxy_resolution,proxy_assim,proxy_frac)
#print sites_assim
#print 'Assimilating proxy types/sites:', sites_assim
proxy_types_assim = sites_assim.keys()
sort_types = sorted(proxy_types_assim)
print 'sorted types: ' + str(sort_types)
total_proxy_count = 0
print 'looping over keys...'
for proxy_key in sorted(proxy_types_assim):
    print 'proxy_key:' + proxy_key
    print sites_assim[proxy_key]
    proxy_count = len(sites_assim[proxy_key]) 
    total_proxy_count = total_proxy_count + proxy_count
    print 'number of sites for this proxy: ' + str(proxy_count) 
    print 'running total: ' + str(total_proxy_count)


# In[ ]:

# read 20th century reanalysis data (downloaded especially for this comparison)
#ncfile = '20threanalysis_1965_1000_T.nc'
ncfile = '20threanalysis_1979_1000_T.nc'
nc = Dataset(ncfile, mode='r')
lon_20CR = nc.variables['lon'][:]
lat_20CR = nc.variables['lat'][:]
T_20CR = nc.variables['air'][:]
print np.shape(T_20CR)
nc.close()
#print np.shape(T_20CR)
#print lon_20CR
#print lon2


# In[ ]:

# try regridding a calibration dataset to do anomaly correlation with LMR
# working from http://earthpy.org/interpolation_between_grids_with_basemap.html
#print np.shape(GIS_anomaly)
#print np.shape(GIS_lon)
#print GIS_lon
#
# work on one year to start
#
#gisone = GIS_anomaly[30,:,:] # this is 1910
gisone = GIS_anomaly[85,:,:] # this is 1965
#gisone = GIS_anomaly[14,:,:] # this is 1979
#print np.shape(gisone)
lat = GIS_lat.copy()
lon = GIS_lon.copy()
# GIS lon already 180:-180; just need to reverse order
#for n, l in enumerate(lon1):
#    if l >= 180:
#       lon1[n]=lon1[n]-360. 
#lon = lon1
#print lon
# define the index to cut "left and right" halves of data
cp = 90
#print lon[cp]
lon_1 = lon[0:cp]
lon_2 = lon[cp:]
lon_new = np.hstack((lon_2, lon_1))
gis1 = gisone[:,0:cp]
gis2 = gisone[:,cp:]
gis_new = np.hstack((gis2, gis1))
#print 'lon_new:' + str(lon_new)
#print lat
#print lon2


# In[ ]:

# set NaNs to zero:
gis_no_nans = np.nan_to_num(gis_new)
# map from lon_new,lat TO lon2,lat2 (CCSM4 values)
GIS_on_CCSM = bm.interp(gis_no_nans, lon_new, lat, lon2, lat2, checkbounds=False, masked=False, order=1)
print np.shape(GIS_on_CCSM)
print np.max(GIS_on_CCSM)
print np.min(GIS_on_CCSM)

if iplot:
    # LMR
    LMR_plotter(X2plt,lat2,lon2,'bwr',nlevs,2)
    plt.title('LMR analysis')
    #plot_direction(CL)
    # 20CR
    Tplt = T_20CR[0,:,:]
    lonplt, latplt = np.meshgrid(lon_20CR, lat_20CR)
    lon2_20CR = lonplt
    lat2_20CR = latplt
    LMR_plotter(Tplt,latplt,lonplt,'bwr',nlevs,2)
    plt.title('20th Century Reanalysis')
    #plot_direction(CL)
    # GIS
    LMR_plotter(GIS_on_CCSM,lat2,lon2,'bwr',nlevs,2)
    plt.title('GIS Temp')
    #plot_direction(CL)
    #plot_direction('True','GIS 1965') # use this to save a .png


# In[ ]:




# In[ ]:

print assimilated_proxies[0]
tmp = assimilated_proxies[0].keys()
proxy = tmp[0]
print proxy
site = assimilated_proxies[0][key[0]]
print site
Y = LMR_proxy.proxy_assignment(proxy)
Y.proxy_datadir = datadir_proxy
Y.proxy_datafile = datafile_proxy
Y.proxy_region = regions

Y.read_proxy(site)
print Y.lat
print Y.lon


# In[ ]:

# read the GMT file written by LMR_driver

print workdir
gmtpfile = '/home/disk/ice4/hakim/svnwork/python-lib/trunk/src/ipython_notebooks/data/ReconDevTest_100_testing_callable_0/gmt.npz'
#gmtpfile = workdir + '/gmt.npz'
#gmt = np.load(gmtpfile)
npzfile = np.load(gmtpfile)
npzfile.files
gmt = npzfile['gmt_save']
recon_times = npzfile['recon_times']

print np.shape(gmt)
#print recon_times

if iplot:
    #print lmr_fix
    # plot a curve for each proxy record (lots!)
    plt.plot(recon_times,np.transpose(gmt))
    # plot the curve for the last (need to know the number of assimilated proxies)
    #plt.plot(years,np.transpose(gmt[20,:]))


# In[ ]:

# post-process a gmt file to assess ob impact
#gmt = gmt_save[]
#print np.shape(gmt)
gmtdiff = np.zeros(np.shape(gmt))
for m in range(6):
    gmtdiff[m,:] = gmt[m+1,:]-gmt[m-1,:]
    
gmtmean = np.average(abs(gmtdiff),1)
#print gmtmean
print assimilated_proxies
if iplot:
    plt.plot(np.transpose(gmtdiff))
gmtdcorr = np.corrcoef(gmtdiff[0:6,:],gmtdiff[0:6,:])
print np.shape(gmtdcorr)
#print gmtdcorr


# In[ ]:

# ensemble-mean spatial maps
#workdir = datadir_output + '/' + nexp
#analysis_year = 1809
#analysis_year = 1601
#analysis_year = 1965
#analysis_year = 1910
analysis_year = 1941

exps = range(0)
kk = -1
 
for k in exps:
    kk = kk + 1
    epath = '/home/disk/enkf/hakim/LMR/500_allproxies_' + str(k)
    prior_filn = epath + '/Xb_one.npz'

    # prior data
    npzfile = np.load(prior_filn)
    npzfile.files
    Xbtmp = npzfile['Xb_one']
    nlat = npzfile['nlat']
    nlon = npzfile['nlon']
    nens = np.size(Xbtmp,1)
    lat = npzfile['lat']
    lon = npzfile['lon']
    # reshape arrays
    lat2 = np.reshape(lat,(nlat,nlon))
    lon2 = np.reshape(lon,(nlat,nlon))
    Xb = np.reshape(Xbtmp,(nlat,nlon,nens))
    # ensemble mean
    xbm = np.mean(Xb,axis=2)
    xbm_shape = np.shape(xbm)
    print kk,xbm_shape
    # analysis data
    analysis_filn = epath + '/year' + str(analysis_year) + '.npy'
    Xatmp = np.load(analysis_filn)
    Xa = np.reshape(Xatmp,(nlat,nlon,nens))
    # ensemble mean
    xam = np.mean(Xa,axis=2)

    if kk == 0:
        xbm_save = np.zeros([len(exps),xbm_shape[0],xbm_shape[1]])
        xam_save = np.zeros([len(exps),xbm_shape[0],xbm_shape[1]])
        xbm_save[kk,:,:] = xbm
        xam_save[kk,:,:] = xam
    else:
        xbm_save[kk,:,:] = xbm
        xam_save[kk,:,:] = xam



# In[ ]:

# regrid GIS to LMR for this year

gis_yrmatch = np.min(np.nonzero(GIS_time == analysis_year))
print str(analysis_year) + ' ' + str(gis_yrmatch) + ' ' + str(GIS_time[gis_yrmatch])

#print GIS_time.index(analysis_year)
gisone = GIS_anomaly[gis_yrmatch,:,:] 
lat = GIS_lat.copy()
lon = GIS_lon.copy()
cp = 90
lon_1 = lon[0:cp]
lon_2 = lon[cp:]
lon_new = np.hstack((lon_2, lon_1))
gis1 = gisone[:,0:cp]
gis2 = gisone[:,cp:]
gis_new = np.hstack((gis2, gis1))
gis_no_nans = np.nan_to_num(gis_new)
# map from lon_new,lat TO lon2,lat2 (CCSM4 values)
GIS_on_CCSM = bm.interp(gis_no_nans, lon_new, lat, lon2, lat2, checkbounds=False, masked=False, order=1)

# plots
if iplot:
    LMR_plotter(xasm,lat2,lon2,'bwr',nlevs)
    plt.title('LMR analysis')
    #plot_direction(CL)

    LMR_plotter(GIS_on_CCSM,lat2,lon2,'bwr',nlevs)
    plt.title('GIS analysis')
    #plot_direction(CL)


# In[ ]:

#
# below this point is old code
#


# In[ ]:

# reconstructed global mean temperature

#workdir = datadir_output + '/' + nexp

print 'working directory: ' + workdir
xagm= []
nxagm= []
nxbgm= []
yc = -1
lat_weight = np.cos(np.deg2rad(lat2[:,0]))
# mean can be done in one shot?
#years = range(1500,2001)
#years = range(1000,2001)
#years = range(1900,1995)
years = range(1800,2001)
#years = range(1900,1905)
lmr_years = years
nyrs = len(years)
varfrac = np.zeros([nyrs,nlat,nlon])
for year in years:
    yc = yc + 1
    analysis_filn = workdir + '/year'+str(year)+'.npy'
    print analysis_filn
    Xatmp = np.load(analysis_filn)
    # ensemble mean
    xam = np.mean(Xatmp,axis=1)
    # global mean
    xagm.append(np.mean(xam))
    # more carefull calculation of global mean
    xam_lalo = np.reshape(xam,(nlat,nlon))
    xam_lat = np.mean(xam_lalo,1)
    nxagm.append(np.mean(np.multiply(lat_weight,xam_lat)))
    #add prior as a "noise reference" (if the years are drawn differently!)
    try:
        xbm = Xbtmp[:,year-years[0]]
        xbm_lalo = np.reshape(xbm,(nlat,nlon))
        xbm_lalat = np.mean(xbm_lalo,1)
        nxbgm.append(np.mean(np.multiply(lat_weight,xbm_lat)))
    except:
        pass
    
    Xa = np.reshape(Xatmp,(nlat,nlon,nens))
    tmm = np.mean(Xa[23,45,:])
    xavar = np.var(Xa,axis=2,ddof=1)
    Varplt = np.divide(xavar,xbvar) - 1.
    varfrac[yc,:,:] = Varplt
    
# average variance reduction over all years, a f(lat,lon)
avarfrac = np.mean(varfrac,0)

# plot of spatial pattern
if iplot:
    LMR_plotter(avarfrac,lat2,lon2,'bwr',nlevs)
    plt.title('ensemble variance ratio (analysis/prior - 1.)')
    #plot_direction(CL)

    # plot of temporal pattern, averaged over the globe
    tvarfrac = np.mean(varfrac,axis=(1,2))
    print np.shape(tvarfrac)
    plt.plot(years,tvarfrac)
    plt.show()


# In[ ]:

import cPickle

fnamePSM = LMRpath+'/PSM/PSMs_'+datatag_calib+'.pckl'
print fnamePSM
infile   = open(fnamePSM,'rb')
psm_data = cPickle.load(infile)
infile.close()

psm_r_crit = 0.2

proxy_TypesSites_psm    = psm_data.keys()
proxy_TypesSites_psm_ok = [t for t in proxy_TypesSites_psm if abs(psm_data[t]['PSMcorrel']) > psm_r_crit]

print len(proxy_TypesSites_psm)
print len(proxy_TypesSites_psm_ok)


# In[ ]:

# test truncation loop in LMR_driver_callable...
from spharm import Spharmt, getspecindx, regrid

epath = '/home/disk/kalman3/hakim/LMR/500_allproxies_0' 
prior_filn = epath + '/Xb_one.npz'

print prior_filn

# prior data
npzfile = np.load(prior_filn)
npzfile.files
Xb_one_full = npzfile['Xb_one']
lat = npzfile['lat']
lon = npzfile['lon']
nlat = npzfile['nlat']
nlon = npzfile['nlon']
nens = np.size(Xb_one_full,1)
print 'nlat, nlon, nens: ' + str(nlat) + ' ' + str(nlon) + ' ' + str(nens)
Nens = nens

print np.shape(Xb_one_full)

specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')

# truncate to a lower resolution grid (triangular truncation)
ntrunc_new = 42 # T42
ifix = np.remainder(ntrunc_new,2.0).astype(int)
nlat_new = ntrunc_new + ifix
nlon_new = nlat_new*1.5
specob_new = Spharmt(nlon_new,nlat_new,gridtype='regular',legfunc='computed')
# lat, lon grid in the truncated space
dlat = 90./((nlat_new-1)/2.)
dlon = 360./nlon_new
lat_new = np.arange(-90.,90.+dlat,dlat)
lon_new = np.arange(0.,360.,dlon)

# transform each ensemble member, one at a time
#Xb_lalo = np.zeros([nlat,nlon])
Xb_one = np.zeros([nlat_new*nlon_new,Nens])
for k in range(nens):
    #print 'working on ensemble member...' + str(k)
    Xb_lalo = np.reshape(Xb_one_full[:,k],(nlat,nlon))
    Xbtrunc = regrid(specob_lmr, specob_new, Xb_lalo, ntrunc=nlat_new-1, smooth=None)
    #vectmp = np.reshape(Xbtrunc,nlat_new*nlon_new,1)
    vectmp = Xbtrunc.flatten()
    Xb_one[:,k] = vectmp
    
# plot check
Xbplt = np.reshape(Xb_one_full[:,0],(nlat,nlon))
lat2 = np.reshape(lat,(nlat,nlon))
lon2 = np.reshape(lon,(nlat,nlon))

LMR_plotter(Xbplt,lat2,lon2,'bwr',nlevs)
plt.title('original resolution')
#plot_direction(CL)

Xbtruncplt = np.reshape(Xb_one[:,0],(nlat_new,nlon_new))
print np.shape(Xbtruncplt)
veclat = np.arange(-90.,90.+dlat,dlat)
veclon = np.arange(0.,360.,dlon)
blank = np.zeros([nlat_new,nlon_new])
lat2_new = (veclat + blank.T).T  
lon2_new = (veclon + blank)  
LMR_plotter(Xbtruncplt,lat2_new,lon2_new,'bwr',nlevs)
#LMR_plotter(Xbtrunc,lat2_new,lon2_new,'bwr',nlevs)
plt.title('truncated resolution')
#plot_direction(CL)


# In[ ]:

#check GMT calculation\
print np.shape(xam)
print np.shape(gmt_save)
print np.shape(sagmt)
print np.shape(lat2_new)
# original method:
lat_weight = np.cos(np.deg2rad(lat2_new[:,0]))

#print lat_weight
#print 'sum = ' + str(np.sum(lat_weight))
#print lat2_new[:,0]

years = range(1880,2000)
lmr_years = years
nyrs = len(years)
gmt_w = np.zeros(nyrs)
gmt_uw = np.zeros(nyrs)
gmt_check = np.zeros(nyrs)

#
# new---make a weight matrix for ALL points
#
tmp = np.ones([nlat_new,nlon_new])
W = np.multiply(lat_weight,tmp.T).T
print np.sum(np.sum(W))
print 'weight shape = ' + str(np.shape(W))
print W[:,0]

ic = -1
for year in years:
    ic = ic + 1
    smatch, ematch = find_date_indices(LMR_time,year,year)
    # global mean as an unweighted average
    gdat = np.squeeze(xam[smatch,:,:])
    gmt_uw[ic] = np.mean(gdat)
    gmt_check[ic] = np.sum(np.multiply(W,gdat))/(np.sum(np.sum(W)))
    
#plot
#plt.plot(years,gmt_uw,'b-')
#plt.plot(years,gmt_w,'r-')
plt.plot(years,gmt_check,'r--')
#smatch, ematch = find_date_indices(LMR_time,years[0],years[-1])
#plt.plot(years,sagmt[smatch:ematch+1],'k--')


# In[ ]:

print np.shape(GIS_anomaly)
tmp = GIS_anomaly[0,:]
print np.max(tmp)


# In[ ]:

import LMR_utils 
LMR_utils = reload(LMR_utils)

datafile_calib   = 'gistemp1200_ERSST.nc'
calib_vars = ['Tsfc']
infile = datadir_calib+'/GISTEMP/'+datafile_calib
print infile
data = Dataset(infile,'r')
print data
gtemp = data.variables['tempanomaly']
print np.shape(gtemp)
glat = data.variables['lat']
print np.shape(glat)
glon = data.variables['lon']
print np.shape(glon)

# gtemp is a masked array. copy it to reveal the mask value, the nset to nan. first entry has the mask
gtempc = np.copy(gtemp)
gtempc[gtempc == gtempc[0,0,0]] = np.nan

time_yrs = []
for i in xrange(0,len(data.variables['time'][:])):
    time_yrs.append(dateref + timedelta(days=int(data.variables['time'][i])))
print np.shape(time_yrs)
print time_yrs[-1]

# this is the function called by the LMR code
gisgm2 = LMR_utils.global_mean(gtempc,glat,glon)

# compute here...
lat_weight = np.cos(np.deg2rad(glat))
tmp = np.ones([len(glat),len(glon)])
W = np.multiply(lat_weight,tmp.T).T
print np.sum(np.sum(W))
print 'weight shape = ' + str(np.shape(W))

ic = -1
for k in range(len(time_yrs)):
    ic = ic + 1
    gdat = gtempc[k,:,:]
    gisgm[ic] = np.nansum(np.multiply(W,gdat))/(np.sum(np.sum(W)))

print np.shape(gisgm)
print np.min(gisgm)
print np.max(gisgm)

plt.plot(time_yrs,gisgm)
plt.plot(time_yrs,gisgm2,'r--')
plt.show()
# annual means
nyears = len(time_yrs)/12
gam = np.zeros([nyears])
for yr in range(nyears):
    gam[yr] = np.mean(gisgm[yr*12:(yr+1)*12 -1])

pyears = 1880+np.array(range(nyears))
plt.plot(pyears,gam,'ko')
plt.plot(pyears,gam,'k-')


# In[ ]:

#x = np.ma.array([1, 2, 3], mask=[0, 1, 0])
#x[1] is np.ma.masked
gtemp[0,0,0] is np.ma.masked
#x = np.ma.array(gtemp)
#np.ma.MaskedArray.get_fill_value(x)
#np.ma.MaskedArray.set_fill_value(x,0)
#print gtemp[0,0,0]
#np.ma.MaskedArray.set_fill_value(x)
#x.fillvalue()
#gtemp.get_fill_value()
gtempc = np.copy(gtemp)

gtempc[gtempc == gtempc[0,0,0]] = np.nan
print gtempc[0,:,0]


# In[ ]:

#old code to verify transforms

"""
fig = plt.figure()
m1 = bm.Basemap(projection='robin',lon_0=0)
ax = fig.add_subplot(2,1,1)
maxabs = np.nanmax(np.abs(pdata_lmr))
cs = m1.contourf(lon2,lat2,pdata_lmr,ncints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
m1.drawcoastlines()
m1.colorbar(cs)
ax.set_title('LMR original')
m2 = bm.Basemap(projection='robin',lon_0=0)
ax = fig.add_subplot(2,1,2)
maxabs = np.nanmax(np.abs(lmr_trunc))
cs = m2.contourf(lon2_new,lat2_new,lmr_trunc,ncints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
m2.drawcoastlines()
m2.colorbar(cs)
ax.set_title('LMR T'+str(ntrunc_new))
plt.title('LMR trunc from main')    
plt.show()

# same for TCR
fig = plt.figure()
m1 = bm.Basemap(projection='robin',lon_0=0)
ax = fig.add_subplot(2,1,1)
maxabs = np.nanmax(np.abs(pdata_tcr))
cs = m1.contourf(lon2_TCR,lat2_TCR,pdata_tcr,ncints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
m1.drawcoastlines()
m1.colorbar(cs)
ax.set_title('TCR original')
m2 = bm.Basemap(projection='robin',lon_0=0)
ax = fig.add_subplot(2,1,2)
maxabs = np.nanmax(np.abs(tcr_trunc))
cs = m2.contourf(lon2_new,lat2_new,tcr_trunc,ncints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
m2.drawcoastlines()
m2.colorbar(cs)
ax.set_title('TCR T'+str(ntrunc_new))
plt.show()

# GIS plot
fig = plt.figure()
m1 = bm.Basemap(projection='robin',lon_0=0)
ax = fig.add_subplot(2,1,1)
maxabs = np.nanmax(np.abs(pdata_gis))
cs = m1.contourf(lon2_GIS,lat2_GIS,pdata_gis,ncints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
m1.drawcoastlines()
m1.colorbar(cs)
ax.set_title('GIS original')
m2 = bm.Basemap(projection='robin',lon_0=0)
ax = fig.add_subplot(2,1,2)
maxabs = np.nanmax(np.abs(gis_trunc))
cs = m2.contourf(lon2_new,lat2_new,gis_trunc,ncints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
m2.drawcoastlines()
m2.colorbar(cs)
ax.set_title('GIS T'+str(ntrunc_new))
plt.show()

# generic plotter---set data fields here --THIS DOESN'T work for the first plot
#plot_dat_orig = pdata_gis
#plot_dat_trunc = gis_trunc
#plot_lon = lon2_GIS
#plot_lat = lon2_GIS
plot_dat_orig = pdata_tcr
plot_dat_trunc = tcr_trunc
plot_lon = lon2_TCR
plot_lat = lon2_TCR
# --------------------------------------

fig = plt.figure()
m1 = bm.Basemap(projection='robin',lon_0=0)
ax = fig.add_subplot(2,1,1)
maxabs = np.nanmax(np.abs(plot_dat_orig))
cs = m1.contourf(plot_lon,plot_lat,plot_dat_orig,ncints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
m1.drawcoastlines()
m1.colorbar(cs)
ax.set_title('original grid')
m2 = bm.Basemap(projection='robin',lon_0=0)
ax = fig.add_subplot(2,1,2)
maxabs = np.nanmax(np.abs(plot_dat_trunc))
cs = m2.contourf(lon2_new,lat2_new,plot_dat_trunc,ncints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
m2.drawcoastlines()
m2.colorbar(cs)
ax.set_title('T'+str(ntrunc_new) + ' grid')
plt.show()
"""
