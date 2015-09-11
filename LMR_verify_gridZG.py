
import matplotlib
# need to do this backend when running remotely or to suppress figures interactively
matplotlib.use('Agg')

# generic imports
import numpy as np
import glob, os
from datetime import datetime, timedelta
from netCDF4 import Dataset
import mpl_toolkits.basemap as bm
import matplotlib.pyplot as plt
from matplotlib import ticker
from spharm import Spharmt, getspecindx, regrid
# LMR specific imports
from LMR_utils import global_hemispheric_means, assimilated_proxies
from load_gridded_data import read_gridded_data_GISTEMP
from load_gridded_data import read_gridded_data_HadCRUT
from load_gridded_data import read_gridded_data_BerkeleyEarth
from LMR_plot_support import *
from LMR_exp_NAMELIST import *
from LMR_plot_support import *

# change default value of latlon kwarg to True.
bm.latlon_default = True

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

# variable
var = 'zg_500hPa_Amon'

# set paths, the filename for plots, and global plotting preferences

# file specification
#
# current datasets
#
#nexp = 'testing_1000_75pct_ens_size_Nens_10'
#nexp = 'testdev_150yr_75pct'
#nexp = 'testdev_check_1000_75pct'
#nexp = 'ReconDevTest_1000_testing_coral'
#nexp = 'ReconDevTest_1000_testing_icecore'
#nexp = 'testdev_1000_100pct_icecoreonly'
#nexp = 'testdev_1000_100pct_mxdonly'
#nexp = 'testdev_1000_100pct_sedimentonly'
#nexp = 'p1rl_CCSM4_LastMillenium_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p1rl_CCSM4_PiControl_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p1rl_MPIESMP_LastMillenium_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p1rl_GFDLCM3_PiControl_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p1rl_20CR_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p1rl_ERA20C_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p2rl_CCSM4_LastMillenium_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p2rlrc0_CCSM4_LastMillenium_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
nexp = 'p2rl_CCSM4_LastMillenium_ens100_cMLOST_allAnnualProxyTypes_pf0.75'

# OLD:

#nexp = 'testdev_500_allproxies'
#exp = 'testdev_500_trunctest'
#
#nexp = 'testdev_1000_75pct'
#nexp = 'testdev_1000_75pct_noTRW'
#nexp = 'testdev_1000_75pct_treesonly'
#nexp = 'testdev_1000_75pct_icecoreonly'
#nexp = 'testdev_1000_75pct_coralonly'
#nexp = 'testdev_1000_100pct_coralonly'

#nexp = 'testdev_1000_100pct_mxdonly'
#nexp = 'testdev_1000_100pct_sedimentonly'
# new
#nexp = 'testdev_1000_75pct_BE'
#nexp = 'testdev_1000_75pct_BE_noTRW'

# override datadir
datadir_output = '/home/disk/kalman3/rtardif/LMR/output'
#datadir_output = './data/'

# number of contours for plots
nlevs = 30

# plot alpha transparency
alpha = 0.5

# time range for verification (in years CE)
#trange = [1960,1962]
#trange = [1880,2000] #works for nya = 0
trange = [1900,2000] #works for nya = 0 
#trange = [1885,1995] #works for nya = 5
#trange = [1890,1990] #works for nya = 10

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
# RT: modified way to determine list of directories with mc realizations
# get a listing of the iteration directories
dirs = glob.glob(workdir+"/r*")
# sorted
dirs.sort()
mcdir = [item.split('/')[-1] for item in dirs]
niters = len(mcdir)

print 'mcdir:' + str(mcdir)
print 'niters = ' + str(niters)

# get time period from the GMT file...
gmtpfile =  workdir + '/r0/gmt.npz'
npzfile = np.load(gmtpfile)
npzfile.files
LMR_time = npzfile['recon_times']

# read ensemble mean data
print '\n reading LMR ensemble-mean data...\n'

first = True
k = -1
for dir in mcdir:
    k = k + 1
    ensfiln = workdir + '/' + dir + '/ensemble_mean_'+var+'.npz'
    npzfile = np.load(ensfiln)
    print  npzfile.files
    tmp = npzfile['xam']
    print 'shape of tmp: ' + str(np.shape(tmp))
    if first:
        first = False
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

    xam = xam + tmp
    xam_all[k,:,:,:] = tmp
    
# this is the sample mean computed with low-memory accumulation
xam = xam/len(mcdir)
# this is the sample mean computed with numpy on all data
xam_check = xam_all.mean(0)
# check..
max_err = np.max(np.max(np.max(xam_check - xam)))
if max_err > 1e-4:
    print 'max error = ' + str(max_err)
    raise Exception('sample mean does not match what is in the ensemble files!')

# sample variance
xam_var = xam_all.var(0)
print np.shape(xam_var)

print '\n shape of the ensemble array: ' + str(np.shape(xam_all)) +'\n'
print '\n shape of the ensemble-mean array: ' + str(np.shape(xam)) +'\n'

#################################################################
# BEGIN: load verification data (20CR and ERA20C)               #
#################################################################
print '\nloading verification data...\n'

# load NOAA's 20th century reanalysis
infile = '/home/disk/kalman3/rtardif/LMR/data/model/20cr/zg_500hPa_Amon_20CR_185101-201112.nc'

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
tcr_nhm = np.zeros([len(TCR_time)])
tcr_shm = np.zeros([len(TCR_time)])

# Loop over years in dataset
for i in xrange(0,len(TCR_time)):        
    # find indices in time array where "years[i]" appear
    ind = [j for j, k in enumerate(years_all) if k == TCR_time[i]]
    time_yrs[i] = TCR_time[i]
    # ---------------------------------------
    # Calculate annual mean from monthly data
    # Note: data has dims [time,lat,lon]
    # ---------------------------------------
    TCR[i,:,:] = np.nanmean(data.variables['zg'][ind],axis=0)
    # compute the global mean temperature
    [tcr_gm[i],tcr_nhm[i],tcr_shm[i]] = global_hemispheric_means(TCR[i,:,:],lat_20CR)
    
# Remove the temporal mean 
TCR = TCR - np.mean(TCR,axis=0)
print 'TCR shape = ' + str(np.shape(TCR))

# compute and remove the 20th century mean
stime = 1900
etime = 1999
smatch, ematch = find_date_indices(TCR_time,stime,etime)
tcr_gm = tcr_gm - np.mean(tcr_gm[smatch:ematch])


# load ERA20C reanalysis
infile = '/home/disk/kalman3/rtardif/LMR/data/model/era20c/zg_500hPa_Amon_ERA20C_190001-201212.nc'

data = Dataset(infile,'r')
lat_ERA20C   = data.variables['lat'][:]
lon_ERA20C   = data.variables['lon'][:]
nlat_ERA20C = len(lat_ERA20C)
nlon_ERA20C = len(lon_ERA20C)
lon2_ERA20C, lat2_ERA20C = np.meshgrid(lon_ERA20C, lat_ERA20C)
 
dateref = datetime(1900,1,1,0)
time_yrs = []
# absolute time from the reference
for i in xrange(0,len(data.variables['time'][:])):
    time_yrs.append(dateref + timedelta(hours=int(data.variables['time'][i])))

years_all = []
for i in xrange(0,len(time_yrs)):
    isotime = time_yrs[i].isoformat()
    years_all.append(int(isotime.split("-")[0]))

ERA20C_time = np.array(list(set(years_all))) # 'set' is used to get unique values in list
ERA20C_time.sort # sort the list

time_yrs  = np.empty(len(ERA20C_time), dtype=int)
ERA20C = np.empty([len(ERA20C_time), len(lat_ERA20C), len(lon_ERA20C)], dtype=float)
era20c_gm = np.zeros([len(ERA20C_time)])
era20c_nhm = np.zeros([len(ERA20C_time)])
era20c_shm = np.zeros([len(ERA20C_time)])

# Loop over years in dataset
for i in xrange(0,len(ERA20C_time)):        
    # find indices in time array where "years[i]" appear
    ind = [j for j, k in enumerate(years_all) if k == ERA20C_time[i]]
    time_yrs[i] = ERA20C_time[i]
    # ---------------------------------------
    # Calculate annual mean from monthly data
    # Note: data has dims [time,lat,lon]
    # ---------------------------------------
    ERA20C[i,:,:] = np.nanmean(data.variables['zg'][ind],axis=0)
    # compute the global mean Z500
    [era20c_gm[i],era20c_nhm[i],era20c_shm[i]] = global_hemispheric_means(ERA20C[i,:,:],lat_ERA20C)
    
# Remove the temporal mean 
ERA20C = ERA20C - np.mean(ERA20C,axis=0)
print 'ERA20C shape = ' + str(np.shape(ERA20C))

# compute and remove the 20th century mean
stime = 1900
etime = 1999
smatch, ematch = find_date_indices(ERA20C_time,stime,etime)
era20c_gm = era20c_gm - np.mean(era20c_gm[smatch:ematch])


###############################################################
# END: load verification data (20CR and ERA20C)               #
###############################################################

print '\n regridding data to a common T42 grid...\n'

iplot_loc= False
#iplot_loc= True

# create instance of the spherical harmonics object for each grid
specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')
specob_tcr = Spharmt(nlon_20CR,nlat_20CR,gridtype='regular',legfunc='computed')
specob_era20c = Spharmt(nlon_ERA20C,nlat_ERA20C,gridtype='regular',legfunc='computed')

# truncate to a lower resolution grid (common:21, 42, 62, 63, 85, 106, 255, 382, 799)
ntrunc_new = 42 # T42
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

#lmr_trunc = np.zeros([nyrs,nlat_new,nlon_new])
#print 'lmr_trunc shape: ' + str(np.shape(lmr_trunc))

# loop over years of interest and transform...specify trange at top of file

iw = 0
if nya > 0:
    iw = (nya-1)/2

cyears = range(trange[0],trange[1])
lt_csave = np.zeros([len(cyears)])
le_csave = np.zeros([len(cyears)])
te_csave = np.zeros([len(cyears)])

lmr_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
tcr_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
era20c_allyears = np.zeros([len(cyears),nlat_new,nlon_new])

lmr_zm = np.zeros([len(cyears),nlat_new])
tcr_zm = np.zeros([len(cyears),nlat_new])
era20c_zm = np.zeros([len(cyears),nlat_new])

k = -1
for yr in cyears:
    k = k + 1
    LMR_smatch, LMR_ematch = find_date_indices(LMR_time,yr-iw,yr+iw+1)
    TCR_smatch, TCR_ematch = find_date_indices(TCR_time,yr-iw,yr+iw+1)
    ERA20C_smatch, ERA20C_ematch = find_date_indices(ERA20C_time,yr-iw,yr+iw+1)

    print '------------------------------------------------------------------------'
    print 'working on year...' + str(yr)
    print 'working on year...' + str(yr) + ' LMR index = ' + str(LMR_smatch) + ' = LMR year ' + str(LMR_time[LMR_smatch])
    #print 'working on year...' + str(yr) + ' TCR index = ' + str(TCR_smatch) + ' = TCR year ' + str(TCR_time[TCR_smatch])

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

    # ERA20C
    pdata_era20c = np.mean(ERA20C[ERA20C_smatch:ERA20C_ematch,:,:],0)    
    era20c_trunc = regrid(specob_era20c, specob_new, np.nan_to_num(pdata_era20c), ntrunc=nlat_new-1, smooth=None)
    # ERA20C latitudes upside down
    era20c_trunc = np.flipud(era20c_trunc)
    #print 'shape of old ERA20C data array:' + str(np.shape(pdata_era20c))
    #print 'shape of new ERA20C data array:' + str(np.shape(era20c_trunc))

    # Reanalysis comparison figures (annually-averaged Z500 anomalies)
    #fmin = -80.0; fmax = +80.0; nflevs=41
    fmin = -60.0; fmax = +60.0; nflevs=41
    fig = plt.figure()
#    ax = fig.add_subplot(3,2,1)    
#    LMR_plotter(pdata_lmr,lat2,lon2,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
#    plt.title('LMR Z500 anom. '+ 'Orig. grid'+' '+str(yr))
#    plt.clim(fmin,fmax)
    ax = fig.add_subplot(3,2,2)    
    LMR_plotter(lmr_trunc,lat2_new,lon2_new,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
    plt.title('LMR Z500 anom.'+ ' T'+str(nlat_new-ifix)+' '+str(yr))
    plt.clim(fmin,fmax)
    ax = fig.add_subplot(3,2,3)    
    LMR_plotter(pdata_tcr,lat2_TCR,lon2_TCR,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
    plt.title('TCR Z500 anom. '+ 'Orig. grid'+' '+str(yr))
    plt.clim(fmin,fmax)
    ax = fig.add_subplot(3,2,4)    
    LMR_plotter(tcr_trunc,lat2_new,lon2_new,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
    plt.title('TCR Z500 anom.'+ ' T'+str(nlat_new-ifix)+' '+str(yr))
    plt.clim(fmin,fmax)
    ax = fig.add_subplot(3,2,5)    
    LMR_plotter(pdata_era20c,lat2_ERA20C,lon2_ERA20C,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
    plt.title('ERA20C Z500 anom. '+ 'Orig. grid'+' '+str(yr))
    plt.clim(fmin,fmax)
    ax = fig.add_subplot(3,2,6)    
    LMR_plotter(era20c_trunc,lat2_new,lon2_new,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
    plt.title('ERA20C Z500 anom.'+ ' T'+str(nlat_new-ifix)+' '+str(yr))
    plt.clim(fmin,fmax)
    fig.tight_layout()
    plt.savefig(nexp+'_LMR_TCR_ERA20C_Z500anom_'+str(yr)+'.png') # RT added ".png"


    # save the full grids
    lmr_allyears[k,:,:] = lmr_trunc
    tcr_allyears[k,:,:] = tcr_trunc
    era20c_allyears[k,:,:] = era20c_trunc

    # zonal-mean verification
    lmr_zm[k,:] = np.mean(lmr_trunc,1)
    tcr_zm[k,:] = tcr_trunc.mean(1)
    era20c_zm[k,:] = era20c_trunc.mean(1)
    
    if iplot_loc:
        ncints = 30
        cmap = 'bwr'
        nticks = 6 # number of ticks on the colorbar
        #set contours based on 20CR
        maxabs = np.nanmax(np.abs(tcr_trunc))
        # round the contour interval, and then set limits to fit
        dc = np.round(maxabs*2/ncints,2)
        cl = dc*ncints/2.
        cints = np.linspace(-cl,cl,ncints,endpoint=True)
        
        # compare LMR and TCR and ERA20C
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
        ax.set_title('LMR Z500'+str(ntrunc_new) + ' ' + str(yr))
        
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
        ax.set_title('TCR Z500'+str(ntrunc_new)+ ' ' + str(yr))
        
        ax = fig.add_subplot(2,2,3)
        m3 = bm.Basemap(projection='robin',lon_0=0)
        # maxabs = np.nanmax(np.abs(gis_trunc))
        cs = m3.contourf(lon2_new,lat2_new,era20c_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
        m3.drawcoastlines()
        cb = m1.colorbar(cs)
        tick_locator = ticker.MaxNLocator(nbins=nticks)
        cb.locator = tick_locator
        cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        cb.update_ticks()
        ax.set_title('ERA20C Z500'+str(ntrunc_new)+ ' ' + str(yr))
        
        plt.clim(-maxabs,maxabs)
        
        # get these numbers by adjusting the figure interactively!!!
        plt.subplots_adjust(left=0.05, bottom=0.45, right=0.95, top=0.95, wspace=0.1, hspace=0.0)
        # plt.tight_layout(pad=0.3)
        fig.suptitle('500hPa height for ' +str(nya) +' year centered average')
    
    # anomaly correlation
    lmrvec = np.reshape(lmr_trunc,(1,nlat_new*nlon_new))
    tcrvec = np.reshape(tcr_trunc,(1,nlat_new*nlon_new))
    era20cvec = np.reshape(era20c_trunc,(1,nlat_new*nlon_new))

    lmr_tcr_corr = np.corrcoef(lmrvec,tcrvec)
    #print 'lmr-tcr correlation: '+str(lmr_tcr_corr[0,1])
    lmr_era20c_corr = np.corrcoef(lmrvec,era20cvec)
    #print 'lmr-era20c correlation: '+ str(lmr_era20c_corr[0,1])
    tcr_era20c_corr = np.corrcoef(tcrvec,era20cvec)
    #print 'tcr-era20c correlation: '+ str(tcr_era20c_corr[0,1])

    # save the correlation values
    lt_csave[k] = lmr_tcr_corr[0,1]
    le_csave[k] = lmr_era20c_corr[0,1]
    te_csave[k] = tcr_era20c_corr[0,1]


# plots for anomaly correlation statistics

# number of bins in the histograms
nbins = 10
#nbins = 5

# LMR compared to TCR and ERA20C
fig = plt.figure()
ax = fig.add_subplot(3,2,1)
ax.plot(cyears,lt_csave)
ax.set_title('LMR-TCR')
ax = fig.add_subplot(3,2,2)
ax.hist(lt_csave,bins=nbins)
ax.set_title('LMR-TCR')
ax = fig.add_subplot(3,2,3)
ax.plot(cyears,le_csave)
ax.set_title('LMR-ERA20C')
ax = fig.add_subplot(3,2,4)
ax.hist(le_csave,bins=nbins)
ax.set_title('LMR-ERA20C')

fig.tight_layout()
plt.subplots_adjust(left=0.05, bottom=0.45, right=0.95, top=0.9, wspace=0.5, hspace=0.5)
fig.suptitle('500hPa height anomaly correlation') 
if fsave:
    print 'saving to .png'
    plt.savefig(nexp+'_verify_grid_Z500_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'.png') # RT added ".png"

# ERA20C compared to TCR
fig = plt.figure()
ax = fig.add_subplot(2,2,1)
ax.plot(cyears,te_csave)
ax.set_title('ERA20C-TCR')
ax = fig.add_subplot(2,2,2)
ax.hist(te_csave,bins=nbins)
ax.set_title('ERA20c-TCR')

#fig.tight_layout()
plt.subplots_adjust(left=0.05, bottom=0.45, right=0.95, top=0.9, wspace=0.5, hspace=0.5)
fig.suptitle('500hPa height anomaly correlation') 
if fsave:
    print 'saving to .png'
    plt.savefig(nexp+'_verify_grid_Z500_anomaly_correlation_reference_'+str(trange[0])+'-'+str(trange[1])+'.png') # RT added ".png"
  

#
# BEGIN r and CE calculations
#

# correlation and CE at each (lat,lon) point

lt_err = lmr_allyears - tcr_allyears
le_err = lmr_allyears - era20c_allyears
te_err = tcr_allyears - era20c_allyears

r_lt = np.zeros([nlat_new,nlon_new])
ce_lt = np.zeros([nlat_new,nlon_new])
r_le = np.zeros([nlat_new,nlon_new])
ce_le = np.zeros([nlat_new,nlon_new])
r_te = np.zeros([nlat_new,nlon_new])
ce_te = np.zeros([nlat_new,nlon_new])


for la in range(nlat_new):
    for lo in range(nlon_new):
        # LMR-TCR
        tstmp = np.corrcoef(lmr_allyears[:,la,lo],tcr_allyears[:,la,lo])
        evar = np.var(lt_err[:,la,lo],ddof=1)
        tvar = np.var(tcr_allyears[:,la,lo],ddof=1)
        r_lt[la,lo] = tstmp[0,1]
        ce_lt[la,lo] = 1. - (evar/tvar)
        # LMR-ERA20C
        tstmp = np.corrcoef(lmr_allyears[:,la,lo],era20c_allyears[:,la,lo])
        evar = np.var(le_err[:,la,lo],ddof=1)
        tvar = np.var(era20c_allyears[:,la,lo],ddof=1)
        r_le[la,lo] = tstmp[0,1]
        ce_le[la,lo] = 1. - (evar/tvar)
        # TCR-ERA20C
        tstmp = np.corrcoef(tcr_allyears[:,la,lo],era20c_allyears[:,la,lo])
        evar = np.var(te_err[:,la,lo],ddof=1)
        tvar = np.var(era20c_allyears[:,la,lo],ddof=1)
        r_te[la,lo] = tstmp[0,1]
        ce_te[la,lo] = 1. - (evar/tvar)


lt_rmean = str(float('%.2g' % np.median(np.median(r_lt)) ))
print 'lmr-tcr all-grid median r: ' + str(lt_rmean)
lt_rmean60 = str(float('%.2g' % np.median(np.median(r_lt[7:34,:])) ))
print 'lmr-tcr 60S-60N median r: ' + str(lt_rmean60)
lt_cemean = str(float('%.2g' % np.median(np.median(ce_lt)) ))
print 'lmr-tcr all-grid median ce: ' + str(lt_cemean)
lt_cemean60 = str(float('%.2g' % np.median(np.median(ce_lt[7:34,:])) ))
print 'lmr-tcr 60S-60N median ce: ' + str(lt_cemean60)
le_rmean = str(float('%.2g' % np.median(np.median(r_le)) ))
print 'lmr-era20c all-grid median r: ' + str(le_rmean)
le_rmean60 = str(float('%.2g' % np.median(np.median(r_le[7:34,:])) ))
print 'lmr-era20c 60S-60N median r: ' + str(le_rmean60)
le_cemean = str(float('%.2g' % np.median(np.median(ce_le)) ))
print 'lmr-era20c all-grid median ce: ' + str(le_cemean)
le_cemean60 = str(float('%.2g' % np.median(np.median(ce_le[7:34,:])) ))
print 'lmr-era20c 60S-60N median ce: ' + str(le_cemean60)
te_rmean = str(float('%.2g' % np.median(np.median(r_te)) ))
print 'tcr-era20c all-grid median r: ' + str(te_rmean)
te_rmean60 = str(float('%.2g' % np.median(np.median(r_te[7:34,:])) ))
print 'tcr-era20c 60S-60N median r: ' + str(te_rmean60)
te_cemean = str(float('%.2g' % np.median(np.median(ce_te)) ))
print 'tcr-era20c all-grid median ce: ' + str(te_cemean)
te_cemean60 = str(float('%.2g' % np.median(np.median(ce_te[7:34,:])) ))
print 'tcr-era20c 60S-60N median ce: ' + str(te_cemean60)


# zonal mean verification
r_lt_zm = np.zeros([nlat_new])
ce_lt_zm = np.zeros([nlat_new])
lt_err_zm = lmr_zm - tcr_zm
# era20c verification
r_le_zm = np.zeros([nlat_new])
ce_le_zm = np.zeros([nlat_new])
le_err_zm = lmr_zm - era20c_zm
for la in range(nlat_new):
    # LMR-TCR
    tstmp = np.corrcoef(lmr_zm[:,la],tcr_zm[:,la])
    evar = np.var(lt_err_zm[:,la],ddof=1)
    tvar = np.var(tcr_zm[:,la],ddof=1)
    r_lt_zm[la] = tstmp[0,1]
    ce_lt_zm[la] = 1. - (evar/tvar)
    # LMR-ERA20C
    tstmp = np.corrcoef(lmr_zm[:,la],era20c_zm[:,la])
    evar = np.var(le_err_zm[:,la],ddof=1)
    tvar = np.var(era20c_zm[:,la],ddof=1)
    r_le_zm[la] = tstmp[0,1]
    ce_le_zm[la] = 1. - (evar/tvar)
    
#print 'LMR-TCR zonal mean r:'    
#print r_lt_zm
#print 'LMR-TCR ce zonal mean:'
#print ce_lt_zm
#print 'LMR-ERA20C zonal mean r:'    
#print r_le_zm
#print 'LMR-ERA20C ce zonal mean:'
#print ce_le_zm
#
# END r and CE
#
major_ticks = np.arange(-90, 91, 30)
fig = plt.figure()
ax = fig.add_subplot(1,2,1)    
ax.plot(r_lt_zm,veclat,'k-',linestyle='--')
ax.plot(r_le_zm,veclat,'k-',linestyle='-')
ax.plot([0,0],[-90,90],'k:')
ax.set_yticks(major_ticks)                                                       
plt.ylim([-90,90])
plt.xlim([-1,1])
plt.ylabel('latitude',fontweight='bold')
plt.xlabel('correlation',fontweight='bold')
#plt.title('correlation (TCR dashed; ERA20C solid)')
ax = fig.add_subplot(1,2,2)    
ax.plot(ce_lt_zm,veclat,'k-',linestyle='--')
ax.plot(ce_le_zm,veclat,'k-',linestyle='-')
ax.plot([0,0],[-90,90],'k:')
ax.set_yticks([])                                                       
plt.ylim([-90,90])
plt.xlim([-1.5,1])
plt.xlabel('cofficient of efficiency',fontweight='bold')
#plt.title('CE (TCR dashed; ERA20C solid)')
plt.suptitle('LMR zonal-mean comparison with ERA20C (solid) and 20CR(dashed)')
fig.tight_layout(pad = 2.0)
if fsave:
    print 'saving to .png'
    plt.savefig(nexp+'_verify_grid_Z500_r_ce_zonal_mean_'+str(trange[0])+'-'+str(trange[1])+'.png') # RT added ".png"

plt.show()

#
# r and ce plots
#

# number of contour levels
nlevs = 11

if iplot:
    fig = plt.figure()
    ax = fig.add_subplot(4,2,1)    
    LMR_plotter(r_lt,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither')
    plt.title('LMR-TCR Z r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lt_rmean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,2)    
    LMR_plotter(ce_lt,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither')
    plt.title('LMR-TCR Z CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lt_cemean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,3)    
    LMR_plotter(r_le,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither')
    plt.title('LMR-ERA20C Z r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(le_rmean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,4)    
    LMR_plotter(ce_le,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither')
    plt.title('LMR-ERA20C Z CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(le_cemean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,5)    
    LMR_plotter(r_te,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither')
    plt.title('TCR-ERA20C Z r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(te_rmean60))
    plt.clim(-1,1)

    ax = fig.add_subplot(4,2,6)    
    LMR_plotter(ce_te,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither')
    plt.title('TCR-ERA20C Z CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(te_cemean60))
    plt.clim(-1,1)
  
    fig.tight_layout()
    if fsave:
        print 'saving to .png'
        plt.savefig(nexp+'_verify_grid_Z500_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.png') # RT: added ".png"    
    

if iplot:
    plt.show()

#ensemble calibration

print np.shape(lt_err)
print np.shape(xam_var)
LMR_smatch, LMR_ematch = find_date_indices(LMR_time,trange[0],trange[1])
print LMR_smatch, LMR_ematch
svar = xam_var[LMR_smatch:LMR_ematch,:,:]
print np.shape(svar)

calib = lt_err.var(0)/svar.mean(0)
print np.shape(calib)
print calib[0:-1,:].mean()



fig = plt.figure()
cb = LMR_plotter(calib,lat2_new,lon2_new,'Oranges',10,0,10,extend='neither')
#cb.set_ticks(range(11))
# overlay stations!
plt.title('ratio of ensemble-mean error variance to mean ensemble variance')
if fsave:
    print 'saving to .png'
    plt.savefig(nexp+'_verify_grid_Z500_ensemble_calibration_'+str(trange[0])+'-'+str(trange[1])+'.png') # RT: added ".png"   


# in loop over lat,lon, add a call to the rank histogram function; need to move up the function def

# NEW look at trends over specified time periods as a function of latitude

# zonal means of the original LMR data
