
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
from LMR_utils import global_hemispheric_means, assimilated_proxies, coefficient_efficiency
from load_gridded_data import read_gridded_data_CMIP5_model
from LMR_plot_support import *

# change default value of latlon kwarg to True.
bm.latlon_default = True

##################################
# START:  set user parameters here
##################################

# option to suppress figures
#iplot = False
iplot = True
iplot_individual_years = False

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
# ---
#nexp = 'p3rlrc0_CCSM4_LastMillenium_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p3rlrc0_CCSM4_PiControl_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p3rlrc0_GFDLCM3_PiControl_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p3rlrc0_MPIESMP_LastMillenium_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p3rlrc0_20CR_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p3rlrc0_ERA20C_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p3rlrc0_CCSM4_LastMillenium_ens100_cMLOST_allAnnualProxyTypes_pf0.75'
#nexp = 'p3rlrc0_GFDLCM3_PiControl_ens100_cMLOST_allAnnualProxyTypes_pf0.75'
#nexp = 'p3rlrc0_MPIESMP_LastMillenium_ens100_cMLOST_allAnnualProxyTypes_pf0.75'
#nexp = 'p3rlrc0_20CR_ens100_cMLOST_allAnnualProxyTypes_pf0.75'
#nexp = 'p3rlrc0_ERA20C_ens100_cMLOST_allAnnualProxyTypes_pf0.75'
#nexp = 'p4rlrc0_CCSM4_LastMillenium_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
#nexp = 'p4rlrc0_GFDLCM3_PiControl_ens100_cGISTEMP_allAnnualProxyTypes_pf0.75'
# ---
#nexp = 'production_gis_ccsm4_pagesall_0.75'
nexp = 'production_mlost_ccsm4_pagesall_0.75'
#nexp = 'production_cru_ccsm4_pagesall_0.75'

# override datadir
#datadir_output = '/home/disk/kalman3/rtardif/LMR/output'
#datadir_output = './data/'
datadir_output = '/home/disk/kalman2/wperkins/LMR_output/archive'

# number of contours for plots
nlevs = 21

# plot alpha transparency
alpha = 0.5

# time range for verification (in years CE)
trange = [1880,2000] #works for nya = 0 
#trange = [1900,2000] #works for nya = 0 
#trange = [1885,1995] #works for nya = 5
#trange = [1890,1990] #works for nya = 10

# reference period over which mean is calculated & subtracted 
# from all datasets (in years CE)
ref_period = [1900, 1999] # 20th century

valid_frac = 0.0


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

# load 20th century reanalysis (TCR) reanalysis --------------------------------

datadir = '/home/disk/kalman3/rtardif/LMR/data/model/20cr'
datafile = 'zg_500hPa_Amon_20CR_185101-201112.nc'
vardef = 'zg_500hPa_Amon'

dd = read_gridded_data_CMIP5_model(datadir,datafile,[vardef])

TCR_time = dd[vardef]['years']
lat_TCR = dd[vardef]['lat']
lon_TCR = dd[vardef]['lon']
nlat_TCR = len(lat_TCR)
nlon_TCR = len(lon_TCR)
lon2_TCR, lat2_TCR = np.meshgrid(lon_TCR, lat_TCR)
#TCR = dd[vardef]['value'] + dd[vardef]['climo'] # Full field (long-term mean NOT REMOVED)
TCR = dd[vardef]['value']                        # Anomalies (long-term mean REMOVED)


# load ERA20C reanalysis -------------------------------------------------------

datadir = '/home/disk/kalman3/rtardif/LMR/data/model/era20c'
datafile = 'zg_500hPa_Amon_ERA20C_190001-201212.nc'
vardef = 'zg_500hPa_Amon'

dd = read_gridded_data_CMIP5_model(datadir,datafile,[vardef])

ERA20C_time = dd[vardef]['years']
lat_ERA20C = dd[vardef]['lat']
lon_ERA20C = dd[vardef]['lon']
nlat_ERA20C = len(lat_ERA20C)
nlon_ERA20C = len(lon_ERA20C)
lon2_ERA20C, lat2_ERA20C = np.meshgrid(lon_ERA20C, lat_ERA20C)
#ERA20C = dd[vardef]['value'] + dd[vardef]['climo'] # Full field (long-term mean NOT REMOVED)
ERA20C = dd[vardef]['value']                        # Anomalies (long-term mean REMOVED)


###############################################################
# END: load verification data (20CR and ERA20C)               #
###############################################################

# ----------------------------------------------------------
# Adjust so that all anomaly data pertain to the mean over a 
# user-defined reference period (e.g. 20th century)
# ----------------------------------------------------------
stime = ref_period[0]
etime = ref_period[1]

# LMR
LMR = xam
smatch, ematch = find_date_indices(LMR_time,stime,etime)
LMR = LMR - np.mean(LMR[smatch:ematch,:,:],axis=0)

# TCR
smatch, ematch = find_date_indices(TCR_time,stime,etime)
TCR = TCR - np.mean(TCR[smatch:ematch,:,:],axis=0)

# ERA
smatch, ematch = find_date_indices(ERA20C_time,stime,etime)
ERA20C = ERA20C - np.mean(ERA20C[smatch:ematch,:,:],axis=0)


# -----------------------------------
# Regridding the data for comparisons
# -----------------------------------
print '\n regridding data to a common T42 grid...\n'

iplot_loc= False
#iplot_loc= True

# create instance of the spherical harmonics object for each grid
specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')
specob_tcr = Spharmt(nlon_TCR,nlat_TCR,gridtype='regular',legfunc='computed')
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
    pdata_lmr = np.mean(LMR[LMR_smatch:LMR_ematch,:,:],0)    
    lmr_trunc = regrid(specob_lmr, specob_new, pdata_lmr, ntrunc=nlat_new-1, smooth=None)

    
    # TCR
    if TCR_smatch and TCR_ematch:
        pdata_tcr = np.mean(TCR[TCR_smatch:TCR_ematch,:,:],0)
    else:
        pdata_tcr = np.zeros(shape=[nlat_TCR,nlon_TCR])
        pdata_tcr.fill(np.nan)

    # regrid on LMR grid
    if np.isnan(pdata_tcr).all():
        tcr_trunc = np.zeros(shape=[nlat_new,nlon_new])
        tcr_trunc.fill(np.nan)
    else:
        tcr_trunc = regrid(specob_tcr, specob_new, pdata_tcr, ntrunc=nlat_new-1, smooth=None)


    # ERA20C
    if ERA20C_smatch and ERA20C_ematch:
        pdata_era20c = np.mean(ERA20C[ERA20C_smatch:ERA20C_ematch,:,:],0)
    else:
        pdata_era20c = np.zeros(shape=[nlat_ERA20C,nlon_ERA20C])
        pdata_era20c.fill(np.nan)

    # regrid on LMR grid
    if np.isnan(pdata_era20c).all():
        era20c_trunc = np.zeros(shape=[nlat_new,nlon_new])
        era20c_trunc.fill(np.nan)
    else:
        era20c_trunc = regrid(specob_era20c, specob_new, pdata_era20c, ntrunc=nlat_new-1, smooth=None)

    if iplot_individual_years:
        # Reanalysis comparison figures (annually-averaged Z500 anomalies)
        #fmin = -80.0; fmax = +80.0; nflevs=41
        fmin = -60.0; fmax = +60.0; nflevs=41
        fig = plt.figure()
        ax = fig.add_subplot(3,1,1)    
        LMR_plotter(lmr_trunc,lat2_new,lon2_new,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
        plt.title('LMR Z500 anom.'+ ' T'+str(nlat_new-ifix)+' '+str(yr))
        plt.clim(fmin,fmax)
        ax = fig.add_subplot(3,1,2)    
        LMR_plotter(tcr_trunc,lat2_new,lon2_new,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
        plt.title('20CR-V2 Z500 anom.'+ ' T'+str(nlat_new-ifix)+' '+str(yr))
        plt.clim(fmin,fmax)
        ax = fig.add_subplot(3,1,3)    
        LMR_plotter(era20c_trunc,lat2_new,lon2_new,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
        plt.title('ERA-20C Z500 anom.'+ ' T'+str(nlat_new-ifix)+' '+str(yr))
        plt.clim(fmin,fmax)
        fig.tight_layout()
        plt.savefig(nexp+'_LMR_TCR_ERA20C_Z500anom_'+str(yr)+'.png')
        plt.close()
        
    # save the full grids
    lmr_allyears[k,:,:] = lmr_trunc
    tcr_allyears[k,:,:] = tcr_trunc
    era20c_allyears[k,:,:] = era20c_trunc

    # -----------------------
    # zonal-mean verification
    # -----------------------

    # LMR
    lmr_zm[k,:] = np.mean(lmr_trunc,1)

    # TCR
    fracok    = np.sum(np.isfinite(tcr_trunc),axis=1,dtype=np.float16)/float(nlon_TCR)
    boolok    = np.where(fracok >= valid_frac)
    boolnotok = np.where(fracok < valid_frac)
    for i in boolok:
        tcr_zm[k,i] = np.nanmean(tcr_trunc[i,:],axis=1)
    tcr_zm[k,boolnotok]  = np.NAN

    # ERA
    fracok    = np.sum(np.isfinite(era20c_trunc),axis=1,dtype=np.float16)/float(nlon_ERA20C)
    boolok    = np.where(fracok >= valid_frac)
    boolnotok = np.where(fracok < valid_frac)
    for i in boolok:
        era20c_zm[k,i] = np.nanmean(era20c_trunc[i,:],axis=1)
    era20c_zm[k,boolnotok]  = np.NAN

    
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

    # lmr <-> tcr
    indok = np.isfinite(tcrvec); nbok = np.sum(indok); nball = tcrvec.shape[1]
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        lt_csave[k] = np.corrcoef(lmrvec[indok],tcrvec[indok])[0,1]
    else:
        lt_csave[k] = np.nan
    print '  lmr-tcr correlation  : '+ str(lt_csave[k])

    # lmr <-> era
    indok = np.isfinite(era20cvec); nbok = np.sum(indok); nball = era20cvec.shape[1]
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        le_csave[k] = np.corrcoef(lmrvec[indok],era20cvec[indok])[0,1]
    else:
        le_csave[k] = np.nan
    print '  lmr-era correlation  : '+ str(le_csave[k])

    # tcr <-> era
    indok = np.isfinite(era20cvec); nbok = np.sum(indok); nball = era20cvec.shape[1]
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        te_csave[k] = np.corrcoef(tcrvec[indok],era20cvec[indok])[0,1]
    else:
        te_csave[k] = np.nan
    print '  tcr-era correlation  : '+ str(te_csave[k])


# plots for anomaly correlation statistics

# number of bins in the histograms
nbins = 15
corr_range = [-0.6,1.0]
bins = np.linspace(corr_range[0],corr_range[1],nbins)

# LMR compared to TCR and ERA20C
fig = plt.figure()
# TCR
ax = fig.add_subplot(3,2,1)
ax.plot(cyears,lt_csave,lw=2)
ax.plot([trange[0],trange[-1]],[0,0],'k:')
ax.set_title('LMR - 20CR-V2')
ax.set_ylim(corr_range[0],corr_range[-1])
ax.set_ylabel('Correlation',fontweight='bold')
ax = fig.add_subplot(3,2,2)
ax.hist(lt_csave,bins=bins,histtype='stepfilled',alpha=0.25)
ax.set_title('LMR - 20CR-V2')
ax.set_xlim(corr_range[0],corr_range[-1])
ax.set_ylabel('Counts',fontweight='bold')
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
ypos = ymax-0.15*(ymax-ymin)
xpos = xmin+0.025*(xmax-xmin)
ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lt_csave)),fontsize=11,fontweight='bold')
# ERA20C
ax = fig.add_subplot(3,2,3)
ax.plot(cyears,le_csave,lw=2)
ax.plot([trange[0],trange[-1]],[0,0],'k:')
ax.set_title('LMR - ERA-20C')
ax.set_ylim(corr_range[0],corr_range[-1])
ax.set_ylabel('Correlation',fontweight='bold')
ax = fig.add_subplot(3,2,4)
ax.hist(le_csave,bins=bins,histtype='stepfilled',alpha=0.25)
ax.set_title('LMR - ERA-20C')
ax.set_xlim(corr_range[0],corr_range[-1])
ax.set_ylabel('Counts',fontweight='bold')
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
ypos = ymax-0.15*(ymax-ymin)
xpos = xmin+0.025*(xmax-xmin)
ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(le_csave)),fontsize=11,fontweight='bold')

# ERA20C compared to TCR
ax = fig.add_subplot(3,2,5)
ax.plot(cyears,te_csave,lw=2)
ax.plot([trange[0],trange[-1]],[0,0],'k:')
ax.set_title('ERA-20C - 20CR-V2')
ax.set_ylim(corr_range[0],corr_range[-1])
ax.set_ylabel('Correlation',fontweight='bold')
ax.set_xlabel('Year CE',fontweight='bold')
ax = fig.add_subplot(3,2,6)
ax.hist(te_csave,bins=bins,histtype='stepfilled',alpha=0.25)
ax.set_title('ERA-20C - 20CR-V2')
ax.set_xlim(corr_range[0],corr_range[-1])
ax.set_ylabel('Counts',fontweight='bold')
ax.set_xlabel('Correlation',fontweight='bold')
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
ypos = ymax-0.15*(ymax-ymin)
xpos = xmin+0.025*(xmax-xmin)
ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(te_csave)),fontsize=11,fontweight='bold')
#fig.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.45, right=0.95, top=0.93, wspace=0.5, hspace=0.5)
fig.suptitle('500hPa height anomaly correlation',fontweight='bold') 
if fsave:
    print 'saving to .png'
    plt.savefig(nexp+'_verify_grid_Z500_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'.png')
    plt.savefig(nexp+'_verify_grid_Z500_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'.pdf', bbox_inches='tight', dpi=300, format='pdf')
    plt.close()

    # =================================================================================================================================
    # For paper 1 :

    fig = plt.figure()
    # TCR
    ax = fig.add_subplot(2,2,1)
    ax.plot(cyears,lt_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('LMR - 20CR-V2')
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    ax = fig.add_subplot(2,2,2)
    ax.hist(lt_csave,bins=bins,histtype='stepfilled',alpha=0.25)
    ax.set_title('LMR - 20CR-V2')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lt_csave)),fontsize=11,fontweight='bold')
    # ERA20C
    ax = fig.add_subplot(2,2,3)
    ax.plot(cyears,le_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('LMR - ERA-20C')
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    ax.set_xlabel('Year CE',fontweight='bold')    
    ax = fig.add_subplot(2,2,4)
    ax.hist(le_csave,bins=bins,histtype='stepfilled',alpha=0.25)
    ax.set_title('LMR - ERA-20C')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    ax.set_xlabel('Correlation',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(le_csave)),fontsize=11,fontweight='bold')
    
    fig.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.45, right=0.95, top=0.93, wspace=0.5, hspace=0.5)
    fig.suptitle('500hPa height anomaly correlation',fontweight='bold') 

    plt.savefig(nexp+'_verify_grid_Z500_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'_paper.png')
    plt.savefig(nexp+'_verify_grid_Z500_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'_paper.pdf', bbox_inches='tight', dpi=300, format='pdf')
    plt.close()
    
    # =================================================================================================================================


#
# BEGIN bias, r and CE calculations
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

# bias

# CE
ce_lt = coefficient_efficiency(tcr_allyears,lmr_allyears)
ce_le = coefficient_efficiency(era20c_allyears,lmr_allyears)
ce_te = coefficient_efficiency(era20c_allyears,tcr_allyears)

# Correlation
for la in range(nlat_new):
    for lo in range(nlon_new):
        # LMR-TCR
        indok = np.isfinite(tcr_allyears[:,la,lo])
        nbok = np.sum(indok)
        nball = lmr_allyears[:,la,lo].shape[0]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_lt[la,lo] = np.corrcoef(lmr_allyears[indok,la,lo],tcr_allyears[indok,la,lo])[0,1]
        else:
            r_lt[la,lo] = np.nan

        # LMR-ERA20C
        indok = np.isfinite(era20c_allyears[:,la,lo])
        nbok = np.sum(indok)
        nball = lmr_allyears[:,la,lo].shape[0]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_le[la,lo] = np.corrcoef(lmr_allyears[indok,la,lo],era20c_allyears[indok,la,lo])[0,1]
        else:
            r_le[la,lo] = np.nan

        # TCR-ERA20C
        indok = np.isfinite(era20c_allyears[:,la,lo])
        nbok = np.sum(indok)
        nball = tcr_allyears[:,la,lo].shape[0]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_te[la,lo] = np.corrcoef(tcr_allyears[indok,la,lo],era20c_allyears[indok,la,lo])[0,1]
        else:
            r_te[la,lo] = np.nan

# median
lt_rmedian = str(float('%.2g' % np.median(np.median(r_lt)) ))
print 'lmr-tcr all-grid median r: ' + str(lt_rmedian)
lt_rmedian60 = str(float('%.2g' % np.median(np.median(r_lt[7:34,:])) ))
print 'lmr-tcr 60S-60N median r: ' + str(lt_rmedian60)
lt_cemedian = str(float('%.2g' % np.median(np.median(ce_lt)) ))
print 'lmr-tcr all-grid median ce: ' + str(lt_cemedian)
lt_cemedian60 = str(float('%.2g' % np.median(np.median(ce_lt[7:34,:])) ))
print 'lmr-tcr 60S-60N median ce: ' + str(lt_cemedian60)
le_rmedian = str(float('%.2g' % np.median(np.median(r_le)) ))
print 'lmr-era20c all-grid median r: ' + str(le_rmedian)
le_rmedian60 = str(float('%.2g' % np.median(np.median(r_le[7:34,:])) ))
print 'lmr-era20c 60S-60N median r: ' + str(le_rmedian60)
le_cemedian = str(float('%.2g' % np.median(np.median(ce_le)) ))
print 'lmr-era20c all-grid median ce: ' + str(le_cemedian)
le_cemedian60 = str(float('%.2g' % np.median(np.median(ce_le[7:34,:])) ))
print 'lmr-era20c 60S-60N median ce: ' + str(le_cemedian60)
te_rmedian = str(float('%.2g' % np.median(np.median(r_te)) ))
print 'tcr-era20c all-grid median r: ' + str(te_rmedian)
te_rmedian60 = str(float('%.2g' % np.median(np.median(r_te[7:34,:])) ))
print 'tcr-era20c 60S-60N median r: ' + str(te_rmedian60)
te_cemedian = str(float('%.2g' % np.median(np.median(ce_te)) ))
print 'tcr-era20c all-grid median ce: ' + str(te_cemedian)
te_cemedian60 = str(float('%.2g' % np.median(np.median(ce_te[7:34,:])) ))
print 'tcr-era20c 60S-60N median ce: ' + str(te_cemedian60)

# spatial mean (area weighted)
# LMR-TCR
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lt,veclat)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lt,veclat)
lt_rmean_global  = str(float('%.2f' %rmean_global[0]))
lt_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
lt_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
lt_cemean_global = str(float('%.2f' %cemean_global[0]))
lt_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
lt_cemean_sh     = str(float('%.2f' %cemean_sh[0]))
# LMR-ERA
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_le,veclat)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_le,veclat)
le_rmean_global  = str(float('%.2f' %rmean_global[0]))
le_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
le_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
le_cemean_global = str(float('%.2f' %cemean_global[0]))
le_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
le_cemean_sh     = str(float('%.2f' %cemean_sh[0]))
# TCR-ERA
[rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_te,veclat)
[cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_te,veclat)
te_rmean_global  = str(float('%.2f' %rmean_global[0]))
te_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
te_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
te_cemean_global = str(float('%.2f' %cemean_global[0]))
te_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
te_cemean_sh     = str(float('%.2f' %cemean_sh[0]))


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
    ce_lt_zm[la] = coefficient_efficiency(tcr_zm[:,la],lmr_zm[:,la],valid=valid_frac)
    indok = np.isfinite(tcr_zm[:,la])
    nbok = np.sum(indok)
    nball = len(cyears)
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        r_lt_zm[la] = np.corrcoef(lmr_zm[indok,la],tcr_zm[indok,la])[0,1]
    else:
        r_lt_zm[la]  = np.nan

    # LMR-ERA20C
    ce_le_zm[la] = coefficient_efficiency(era20c_zm[:,la],lmr_zm[:,la],valid=valid_frac)    
    indok = np.isfinite(era20c_zm[:,la])
    nbok = np.sum(indok)
    nball = len(cyears)
    ratio = float(nbok)/float(nball)
    if ratio > valid_frac:
        r_le_zm[la] = np.corrcoef(lmr_zm[indok,la],era20c_zm[indok,la])[0,1]
    else:
        r_le_zm[la]  = np.nan

#
# END r and CE
#
major_ticks = np.arange(-90, 91, 30)
fig = plt.figure()
ax = fig.add_subplot(1,2,1)    
tcrleg, = ax.plot(r_lt_zm,veclat,'k-',linestyle='--',lw=2,label='20CR-V2')
eraleg, = ax.plot(r_le_zm,veclat,'k-',linestyle='-',lw=2,label='ERA-20C')
ax.plot([0,0],[-90,90],'k:')
ax.set_yticks(major_ticks)                                                       
plt.ylim([-90,90])
plt.xlim([-1,1])
plt.ylabel('Latitude',fontweight='bold')
plt.xlabel('Correlation',fontweight='bold')
ax.legend(handles=[tcrleg,eraleg],handlelength=3.0,ncol=1,fontsize=12,loc='upper left',frameon=False)

ax = fig.add_subplot(1,2,2)    
ax.plot(ce_lt_zm,veclat,'k-',linestyle='--',lw=2)
ax.plot(ce_le_zm,veclat,'k-',linestyle='-',lw=2)
ax.plot([0,0],[-90,90],'k:')
ax.set_yticks([])                                                       
plt.ylim([-90,90])
plt.xlim([-1.5,1])
plt.xlabel('Coefficient of efficiency',fontweight='bold')
#plt.title('CE (TCR dashed; ERA20C solid)')
plt.suptitle('LMR zonal-mean verification - 500hPa heights',fontweight='bold')
fig.tight_layout(pad = 2.0)
if fsave:
    print 'saving to .png'
    plt.savefig(nexp+'_verify_grid_Z500_r_ce_zonal_mean_'+str(trange[0])+'-'+str(trange[1])+'.png') 
    plt.savefig(nexp+'_verify_grid_Z500_r_ce_zonal_mean_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')
    plt.close()


#
# r and ce plots
#

cbarfmt = '%4.1f'
nticks = 4 # number of ticks on the colorbar
if iplot:
    fig = plt.figure()
    ax = fig.add_subplot(4,2,1)    
    LMR_plotter(r_lt,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
    #plt.title('LMR - 20CR-V2 Z r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lt_rmean))
    plt.title('LMR - 20CR-V2 Z500 r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lt_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,2)    
    LMR_plotter(ce_lt,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
    #plt.title('LMR - 20CR-V2 Z CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(lt_cemean))
    plt.title('LMR - 20CR-V2 Z500 CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lt_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,3)    
    LMR_plotter(r_le,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
    #plt.title('LMR - ERA-20C Z r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(le_rmean))
    plt.title('LMR - ERA-20C Z500 r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(le_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,4)    
    LMR_plotter(ce_le,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
    #plt.title('LMR - ERA-20C Z CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(le_cemean))
    plt.title('LMR - ERA-20C Z500 CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(le_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,5)    
    LMR_plotter(r_te,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
    #plt.title('TCR - ERA-20C Z r '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(te_rmean))
    plt.title('20CR-V2 - ERA-20C Z500 r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(te_rmean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])

    ax = fig.add_subplot(4,2,6)    
    LMR_plotter(ce_te,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
    #plt.title('20CR-V2 - ERA-20C Z CE '+ 'T'+str(nlat_new-ifix)+' '+str(cyears[0])+'-'+str(cyears[-1]) + ' median='+str(te_cemean))
    plt.title('20CR-V2 - ERA-20C Z500 CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(te_cemean_global))
    plt.clim(-1,1)
    ax.title.set_position([.5, 1.05])
  
    fig.tight_layout()
    if fsave:
        print 'saving to .png'
        plt.savefig(nexp+'_verify_grid_Z500_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_grid_Z500_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')
        plt.close()

        # =================================================================================================================================
        # For paper 1 :

        fig = plt.figure()
        ax = fig.add_subplot(4,2,1)    
        LMR_plotter(r_lt,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - 20CR-V2 Z500 r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lt_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,2)    
        LMR_plotter(ce_lt,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - 20CR-V2 Z500 CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lt_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,3)    
        LMR_plotter(r_le,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - ERA-20C Z500 r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(le_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,4)    
        LMR_plotter(ce_le,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - ERA-20C Z500 CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(le_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        fig.tight_layout()
        print 'saving to .png'
        plt.savefig(nexp+'_verify_grid_Z500_r_ce_'+str(trange[0])+'-'+str(trange[1])+'_paper.png')
        plt.savefig(nexp+'_verify_grid_Z500_r_ce_'+str(trange[0])+'-'+str(trange[1])+'_paper.pdf',bbox_inches='tight', dpi=300, format='pdf')
        plt.close()

        # =================================================================================================================================

if iplot:
    plt.show()

# ensemble calibration

print np.shape(lt_err)
print np.shape(xam_var)
LMR_smatch, LMR_ematch = find_date_indices(LMR_time,trange[0],trange[1])
print LMR_smatch, LMR_ematch
svar = xam_var[LMR_smatch:LMR_ematch,:,:]
print np.shape(svar)

calib = lt_err.var(0)/svar.mean(0)
print np.shape(calib)
print calib[0:-1,:].mean()


# create the plot
mapcolor_calib = truncate_colormap(plt.cm.YlOrBr,0.0,0.8)
fig = plt.figure()
cb = LMR_plotter(calib,lat2_new,lon2_new,mapcolor_calib,11,0,10,extend='max',nticks=10)
#cb.set_ticks(range(11))
# overlay stations!
plt.title('Ratio of ensemble-mean error variance to mean ensemble variance \n 500hPa heights')
if fsave:
    print 'saving to .png'
    plt.savefig(nexp+'_verify_grid_Z500_ensemble_calibration_'+str(trange[0])+'-'+str(trange[1])+'.png')  


# in loop over lat,lon, add a call to the rank histogram function; need to move up the function def

# NEW look at trends over specified time periods as a function of latitude

# zonal means of the original LMR data
