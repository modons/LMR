""" 
Module: LMR_verify_gridPRCP.py

Purpose: Generates spatial verification statistics of LMR gridded precipitation
         against various gridded historical instrumental precipitation datasets
         and precipitation from reanalyses.  

Originator: Robert Tardif, U. of Washington, March 2016

Revisions: 

"""
import matplotlib
# need to do this backend when running remotely or to suppress figures interactively
matplotlib.use('Agg')

# generic imports
import numpy as np
import glob, os, sys, calendar
from datetime import datetime, timedelta
from netCDF4 import Dataset, date2num, num2date
import mpl_toolkits.basemap as bm
import matplotlib.pyplot as plt
from matplotlib import ticker
from spharm import Spharmt, getspecindx, regrid
# LMR specific imports
sys.path.append('../')
from LMR_utils import global_hemispheric_means, assimilated_proxies, coefficient_efficiency
from load_gridded_data import read_gridded_data_CMIP5_model
from LMR_plot_support import *

# change default value of latlon kwarg to True.
bm.latlon_default = True

##################################
# START:  set user parameters here
##################################

# option to suppress figures
iplot = True
iplot_individual_years = False

# centered time mean (nya must be odd! 3 = 3 yr mean; 5 = 5 year mean; etc 0 = none)
nya = 0

# option to print figures
fsave = True
#fsave = False

# set paths, the filename for plots, and global plotting preferences

# override datadir
#datadir_output = './data/'
#datadir_output = '/home/disk/kalman2/wperkins/LMR_output/archive'
#datadir_output = '/home/disk/kalman3/rtardif/LMR/output'
#datadir_output = '/home/disk/ekman4/rtardif/LMR/output'
datadir_output = '/home/disk/kalman3/hakim/LMR'

# Directories where precip and reanalysis data can be found
datadir_precip = '/home/disk/kalman3/rtardif/LMR/data/verification'
datadir_reanl  = '/home/disk/kalman3/rtardif/LMR/data/model'

# file specification
#
# current datasets
# ---
#nexp = 'production_gis_ccsm4_pagesall_0.75'
#nexp = 'production_mlost_ccsm4_pagesall_0.75'
#nexp = 'production_cru_ccsm4_pagesall_0.75'
#nexp = 'production_mlost_era20c_pagesall_0.75'
#nexp = 'production_mlost_era20cm_pagesall_0.75'
# ---
#nexp = 'test'
#nexp = 'pages2_loc25000_seasonal_bilinear_nens200'
#nexp = 'pages2_loc15000_seasonal_bilinear_nens200'
nexp = 'pages2_loc10000_seasonal_bilinear_nens200'

# perform verification using all recon. MC realizations ( MCset = None )
# or over a custom selection ( MCset = (begin,end) )
# ex. MCset = (0,0)    -> only the first MC run
#     MCset = (0,10)   -> the first 11 MC runs (from 0 to 10 inclusively)
#     MCset = (80,100) -> the 80th to 100th MC runs (21 realizations)
MCset = None
#MCset = (0,10)

# Definition of variables to verify
#                       kind   name   variable long name bounds     units   mult. factor
verif_dict = \
    {
    'pr_sfc_Amon'   : ('anom', 'PRCP', 'Precipitation',-400.0,400.0,'(mm/yr)',1.0), \
    }

# time range for verification (in years CE)
#trange = [1979,2000] #works for nya = 0 
trange = [1880,2000] #works for nya = 0 
#trange = [1900,2000] #works for nya = 0 
#trange = [1885,1995] #works for nya = 5
#trange = [1890,1990] #works for nya = 10

# reference period over which mean is calculated & subtracted 
# from all datasets (in years CE)
ref_period = [1979, 1999]

valid_frac = 0.0

# number of contours for plots
nlevs = 21

# plot alpha transparency
alpha = 0.5

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

verif_vars = list(verif_dict.keys())

workdir = datadir_output + '/' + nexp
print('working directory = ' + workdir)

print('\n getting file system information...\n')

# get number of mc realizations from directory count
# RT: modified way to determine list of directories with mc realizations
# get a listing of the iteration directories
dirs = glob.glob(workdir+"/r*")

# selecting the MC iterations to keep
if MCset:
    dirset = dirs[MCset[0]:MCset[1]+1]
else:
    dirset = dirs

mcdir = [item.split('/')[-1] for item in dirset]
niters = len(mcdir)

print('mcdir:' + str(mcdir))
print('niters = ' + str(niters))

# Loop over verif. variables
for var in verif_vars:

    # read ensemble mean data
    print('\n reading LMR ensemble-mean data...\n')

    first = True
    k = -1
    for dir in mcdir:
        k = k + 1
        ensfiln = workdir + '/' + dir + '/ensemble_mean_'+var+'.npz'
        npzfile = np.load(ensfiln)
        print(dir, ':', npzfile.files)
        tmp = npzfile['xam']
        print('shape of tmp: ' + str(np.shape(tmp)))
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

        xam = xam + tmp
        xam_all[k,:,:,:] = tmp
    
    # this is the sample mean computed with low-memory accumulation
    xam = xam/len(mcdir)
    # this is the sample mean computed with numpy on all data
    xam_check = xam_all.mean(0)
    # check..
    max_err = np.max(np.max(np.max(xam_check - xam)))
    if max_err > 1e-4:
        print('max error = ' + str(max_err))
        raise Exception('sample mean does not match what is in the ensemble files!')

    # sample variance
    xam_var = xam_all.var(0)
    print(np.shape(xam_var))

    print('\n shape of the ensemble array: ' + str(np.shape(xam_all)) +'\n')
    print('\n shape of the ensemble-mean array: ' + str(np.shape(xam)) +'\n')


    # Convert units to match verif dataset: from kg m-2 s-1 to mm (per year)
    rho = 1000.0
    for y in range(nyrs):

        if calendar.isleap(int(years[y])):
            xam[y,:,:] = 1000.*xam[y,:,:]*366.*86400./rho
        else:
            xam[y,:,:] = 1000.*xam[y,:,:]*365.*86400./rho


    #################################################################
    # BEGIN: load verification data                                 #
    #################################################################
    print('\nloading verification data...\n')


    # GPCP ----------------------------------------------------------
    infile = datadir_precip+'/'+'GPCP/'+'GPCPv2.2_precip.mon.mean.nc'
    verif_data = Dataset(infile,'r')
    
    # Time
    time = verif_data.variables['time']
    time_obj = num2date(time[:],units=time.units)
    
    time_yrs = np.asarray([time_obj[k].year for k in range(len(time_obj))])
    yrs_range = list(set(time_yrs))

    # lat/lon
    verif_lat = verif_data.variables['lat'][:]
    verif_lon = verif_data.variables['lon'][:]
    nlat_GPCP = len(verif_lat)
    nlon_GPCP = len(verif_lon)
    lon_GPCP, lat_GPCP = np.meshgrid(verif_lon, verif_lat)

    # Precip
    verif_precip_monthly = verif_data.variables['precip'][:]
    [ntime,nlon_v,nlat_v] = verif_precip_monthly.shape
    
    # convert mm/day monthly data to mm/year yearly data
    GPCP_time  = np.zeros(shape=len(yrs_range),dtype=np.int)
    GPCP = np.zeros(shape=[len(yrs_range),nlat_GPCP,nlon_GPCP])
    i = 0
    for yr in yrs_range:
        GPCP_time[i] = int(yr)
        inds = np.where(time_yrs == yr)[0]

        if calendar.isleap(yr):
            nbdays = 366.
        else:
            nbdays = 365.

        accum = np.zeros(shape=[nlat_GPCP, nlon_GPCP])
        for k in range(len(inds)):
            days_in_month = calendar.monthrange(time_obj[inds[k]].year, time_obj[inds[k]].month)[1]
            accum = accum + verif_precip_monthly[inds[k],:,:]*days_in_month

        GPCP[i,:,:] = accum # precip in mm
        #GPCP[i,:,:] = np.mean(verif_precip_monthly[inds,:,:], axis=0) # not mean !? sum ????
        i = i + 1


    # CMAP ----------------------------------------------------------
    infile = datadir_precip+'/'+'CMAP/'+'CMAP_enhanced_precip.mon.mean.nc'
    verif_data = Dataset(infile,'r')
    
    # Time
    time = verif_data.variables['time']
    time_obj = num2date(time[:],units=time.units)
    
    time_yrs = np.asarray([time_obj[k].year for k in range(len(time_obj))])
    yrs_range = list(set(time_yrs))

    # lat/lon
    verif_lat = verif_data.variables['lat'][:]
    verif_lon = verif_data.variables['lon'][:]
    nlat_CMAP = len(verif_lat)
    nlon_CMAP = len(verif_lon)
    lon_CMAP, lat_CMAP = np.meshgrid(verif_lon, verif_lat)

    # Precip
    verif_precip_monthly = verif_data.variables['precip'][:]
    [ntime,nlon_v,nlat_v] = verif_precip_monthly.shape
    
    # convert mm/day monthly data to mm/year yearly data
    CMAP_time  = np.zeros(shape=len(yrs_range),dtype=np.int)
    CMAP = np.zeros(shape=[len(yrs_range),nlat_CMAP,nlon_CMAP])
    i = 0
    for yr in yrs_range:
        CMAP_time[i] = int(yr)
        inds = np.where(time_yrs == yr)[0]

        if calendar.isleap(yr):
            nbdays = 366.
        else:
            nbdays = 365.

        accum = np.zeros(shape=[nlat_CMAP, nlon_CMAP])
        for k in range(len(inds)):
            days_in_month = calendar.monthrange(time_obj[inds[k]].year, time_obj[inds[k]].month)[1]
            accum = accum + verif_precip_monthly[inds[k],:,:]*days_in_month

        CMAP[i,:,:] = accum # precip in mm
        #GPCP[i,:,:] = np.mean(verif_precip_monthly[inds,:,:], axis=0) # not mean !? sum ????
        i = i + 1

    # ----------
    # Reanalyses
    # ----------

    # Define month sequence for the calendar year 
    # (argument needed in upload of reanalysis data)
    annual = list(range(1,13))
    
    # 20th Century reanalysis (TCR) ---------------------------------
    vardict = {var: verif_dict[var][0]}
    vardef   = var
    datadir  = datadir_reanl +'/20cr'
    datafile = vardef +'_20CR_185101-201112.nc'
    
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
    lon2_TCR, lat2_TCR = np.meshgrid(lon_TCR, lat_TCR)

    TCRfull = dd[vardef]['value'] + dd[vardef]['climo'] # Full field
    TCR = dd[vardef]['value']                           # Anomalies

    # Conversion from kg m-2 s-1
    rho = 1000.0
    i = 0
    for y in TCR_time:
        if calendar.isleap(y):
            TCRfull[i,:,:] = 1000.*TCRfull[i,:,:]*366.*86400./rho
            TCR[i,:,:] = 1000.*TCR[i,:,:]*366.*86400./rho
        else:
            TCRfull[i,:,:] = 1000.*TCRfull[i,:,:]*365.*86400./rho
            TCR[i,:,:] = 1000.*TCR[i,:,:]*365.*86400./rho
        i = i + 1


    # ERA 20th Century reanalysis (ERA20C) ---------------------------------
    vardict  = {var: verif_dict[var][0]}
    vardef   = var
    datadir  = datadir_reanl +'/era20c'
    datafile = vardef +'_ERA20C_190001-201012.nc'
    
    dd = read_gridded_data_CMIP5_model(datadir,datafile,vardict,outtimeavg=annual)
    rtime = dd[vardef]['years']
    ERA_time = np.array([d.year for d in rtime])
    lats = dd[vardef]['lat']
    lons = dd[vardef]['lon']
    latshape = lats.shape
    lonshape = lons.shape
    if len(latshape) == 2 & len(lonshape) == 2:
        # stored in 2D arrays
        lat_ERA = np.unique(lats)
        lon_ERA = np.unique(lons)
        nlat_ERA, = lat_ERA.shape
        nlon_ERA, = lon_ERA.shape
    else:
        # stored in 1D arrays
        lon_ERA = lons
        lat_ERA = lats
        nlat_ERA = len(lat_ERA)
        nlon_ERA = len(lon_ERA)
    lon2_ERA, lat2_ERA = np.meshgrid(lon_ERA, lat_ERA)

    ERAfull = dd[vardef]['value'] + dd[vardef]['climo'] # Full field
    ERA = dd[vardef]['value']                           # Anomalies

    # Conversion from kg m-2 s-1
    rho = 1000.0
    i = 0
    for y in ERA_time:
        if calendar.isleap(y):
            ERAfull[i,:,:] = 1000.*ERAfull[i,:,:]*366.*86400./rho
            ERA[i,:,:] = 1000.*ERA[i,:,:]*366.*86400./rho
        else:
            ERAfull[i,:,:] = 1000.*ERAfull[i,:,:]*365.*86400./rho
            ERA[i,:,:] = 1000.*ERA[i,:,:]*365.*86400./rho
        i = i + 1

        
        
    # Plots of precipitation climatologies ---

    # Climatology (annual accumulation)
    GPCP_climo = np.nanmean(GPCP, axis=0)
    CMAP_climo = np.nanmean(CMAP, axis=0)
    TCR_climo = np.nanmean(TCRfull, axis=0)
    ERA_climo = np.nanmean(ERAfull, axis=0)
    
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)    
    fmin = 0; fmax = 4000; nflevs=41
    LMR_plotter(GPCP_climo,lat_GPCP,lon_GPCP,'Reds',nflevs,vmin=fmin,vmax=fmax,extend='max')
    plt.title( 'GPCP '+'orig. grid'+' '+verif_dict[var][1]+' '+verif_dict[var][5]+' '+'climo.', fontweight='bold')
    plt.clim(fmin,fmax)

    ax = fig.add_subplot(2,2,2)    
    fmin = 0; fmax = 4000; nflevs=41
    LMR_plotter(CMAP_climo,lat_CMAP,lon_CMAP,'Reds',nflevs,vmin=fmin,vmax=fmax,extend='max')
    plt.title( 'CMAP '+'orig. grid'+' '+verif_dict[var][1]+' '+verif_dict[var][5]+' '+'climo.', fontweight='bold')
    plt.clim(fmin,fmax)

    ax = fig.add_subplot(2,2,3)    
    fmin = 0; fmax = 4000; nflevs=41
    LMR_plotter(TCR_climo,lat2_TCR,lon2_TCR,'Reds',nflevs,vmin=fmin,vmax=fmax,extend='max')
    plt.title( '20CR-V2 '+'orig. grid'+' '+verif_dict[var][1]+' '+verif_dict[var][5]+' '+'climo.', fontweight='bold')
    plt.clim(fmin,fmax)

    ax = fig.add_subplot(2,2,4)    
    fmin = 0; fmax = 4000; nflevs=41
    LMR_plotter(ERA_climo,lat2_ERA,lon2_ERA,'Reds',nflevs,vmin=fmin,vmax=fmax,extend='max')
    plt.title( 'ERA20C '+'orig. grid'+' '+verif_dict[var][1]+' '+verif_dict[var][5]+' '+'climo.', fontweight='bold')
    plt.clim(fmin,fmax)
    
    fig.tight_layout()
    plt.savefig('GPCP_CMAP_20CR_ERA_climo.png')
    plt.close()
    

    # Precip accum. anomalies ---
    print('Removing the temporal mean (for every gridpoint) from the prior...')
    climo = np.nanmean(GPCP,axis=0)
    GPCP = (GPCP - climo)
    climo = np.nanmean(CMAP,axis=0)
    CMAP = (CMAP - climo)

    print('GPCP : Global: mean=', np.nanmean(GPCP), ' , std-dev=', np.nanstd(GPCP))
    print('CMAP : Global: mean=', np.nanmean(CMAP), ' , std-dev=', np.nanstd(CMAP))
    print('TCR  : Global: mean=', np.nanmean(TCR), ' , std-dev=', np.nanstd(TCR))
    print('ERA  : Global: mean=', np.nanmean(ERA), ' , std-dev=', np.nanstd(ERA))
    print('LMR  : Global: mean=', np.nanmean(xam), ' , std-dev=', np.nanstd(xam))


    ###############################################################
    # END: load verification data                                 #
    ###############################################################

    # ----------------------------------------------------------
    # Adjust so that all anomaly data pertain to the mean over a 
    # user-defined reference period (e.g. 20th century)
    # ----------------------------------------------------------
    
    print('Re-center on %s-%s period' % (str(ref_period[0]), str(ref_period[1])))

    stime = ref_period[0]
    etime = ref_period[1]

    # LMR
    LMR = xam
    smatch, ematch = find_date_indices(LMR_time,stime,etime)
    LMR = LMR - np.mean(LMR[smatch:ematch,:,:],axis=0)
    
    # verif
    smatch, ematch = find_date_indices(GPCP_time,stime,etime)
    GPCP = GPCP - np.mean(GPCP[smatch:ematch,:,:],axis=0)

    smatch, ematch = find_date_indices(CMAP_time,stime,etime)
    CMAP = CMAP - np.mean(CMAP[smatch:ematch,:,:],axis=0)

    smatch, ematch = find_date_indices(TCR_time,stime,etime)
    TCR = TCR - np.mean(TCR[smatch:ematch,:,:],axis=0)

    smatch, ematch = find_date_indices(ERA_time,stime,etime)
    ERA = ERA - np.mean(ERA[smatch:ematch,:,:],axis=0)


    print('GPCP : Global: mean=', np.nanmean(GPCP), ' , std-dev=', np.nanstd(GPCP))
    print('CMAP : Global: mean=', np.nanmean(CMAP), ' , std-dev=', np.nanstd(CMAP))
    print('TCR : Global: mean=', np.nanmean(TCR), ' , std-dev=', np.nanstd(TCR))
    print('ERA : Global: mean=', np.nanmean(ERA), ' , std-dev=', np.nanstd(ERA))
    print('LMR  : Global: mean=', np.nanmean(LMR), ' , std-dev=', np.nanstd(LMR))


    # -----------------------------------
    # Regridding the data for comparisons
    # -----------------------------------
    print('\n regridding data to a common T42 grid...\n')

    iplot_loc= False
    #iplot_loc= True

    # create instance of the spherical harmonics object for each grid
    specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')
    specob_gpcp = Spharmt(nlon_GPCP,nlat_GPCP,gridtype='regular',legfunc='computed')
    specob_cmap = Spharmt(nlon_CMAP,nlat_CMAP,gridtype='regular',legfunc='computed')
    specob_tcr  = Spharmt(nlon_TCR,nlat_TCR,gridtype='regular',legfunc='computed')
    specob_era  = Spharmt(nlon_ERA,nlat_ERA,gridtype='regular',legfunc='computed')

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
    lmr_trunc = np.zeros([nyrs,nlat_new,nlon_new])
    print('lmr_trunc shape: ' + str(np.shape(lmr_trunc)))

    # loop over years of interest and transform...specify trange at top of file
    iw = 0
    if nya > 0:
        iw = (nya-1)/2

    cyears = list(range(trange[0],trange[1]))
    lg_csave = np.zeros([len(cyears)])
    lc_csave = np.zeros([len(cyears)])
    lt_csave = np.zeros([len(cyears)])
    le_csave = np.zeros([len(cyears)])    
    gc_csave = np.zeros([len(cyears)])
    gt_csave = np.zeros([len(cyears)])
    ge_csave = np.zeros([len(cyears)])
    te_csave = np.zeros([len(cyears)])

    lmr_allyears  = np.zeros([len(cyears),nlat_new,nlon_new])
    gpcp_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
    cmap_allyears = np.zeros([len(cyears),nlat_new,nlon_new])
    tcr_allyears  = np.zeros([len(cyears),nlat_new,nlon_new])
    era_allyears  = np.zeros([len(cyears),nlat_new,nlon_new])
    
    lmr_zm  = np.zeros([len(cyears),nlat_new])
    gpcp_zm = np.zeros([len(cyears),nlat_new])
    cmap_zm = np.zeros([len(cyears),nlat_new])
    tcr_zm  = np.zeros([len(cyears),nlat_new])
    era_zm  = np.zeros([len(cyears),nlat_new])

    k = -1
    for yr in cyears:
        k = k + 1
        LMR_smatch, LMR_ematch = find_date_indices(LMR_time,yr-iw,yr+iw+1)
        GPCP_smatch, GPCP_ematch = find_date_indices(GPCP_time,yr-iw,yr+iw+1)
        CMAP_smatch, CMAP_ematch = find_date_indices(CMAP_time,yr-iw,yr+iw+1)
        TCR_smatch, TCR_ematch = find_date_indices(TCR_time,yr-iw,yr+iw+1)
        ERA_smatch, ERA_ematch = find_date_indices(ERA_time,yr-iw,yr+iw+1)

        print('------------------------------------------------------------------------')
        print('working on year... %5s' %(str(yr)))
        print('                   %5s LMR index  = %5s : LMR year  = %5s' %(str(yr), str(LMR_smatch), str(LMR_time[LMR_smatch])))
        if GPCP_smatch:
            print('                   %5s GPCP index = %5s : GPCP year = %5s' %(str(yr), str(GPCP_smatch), str(GPCP_time[GPCP_smatch])))
        if CMAP_smatch:
            print('                   %5s CMAP index = %5s : CMAP year = %5s' %(str(yr), str(CMAP_smatch), str(CMAP_time[CMAP_smatch])))
        if TCR_smatch:
            print('                   %5s TCP index  = %5s : TCR year  = %5s' %(str(yr), str(TCR_smatch), str(TCR_time[TCR_smatch])))
        if ERA_smatch:
            print('                   %5s ERA index  = %5s : ERA year  = %5s' %(str(yr), str(ERA_smatch), str(ERA_time[ERA_smatch])))
            
        # LMR
        pdata_lmr = np.mean(LMR[LMR_smatch:LMR_ematch,:,:],0)    
        lmr_trunc = regrid(specob_lmr, specob_new, pdata_lmr, ntrunc=nlat_new-1, smooth=None)

        # GPCP
        if GPCP_smatch and GPCP_ematch:
            pdata_gpcp = np.mean(GPCP[GPCP_smatch:GPCP_ematch,:,:],0)
        else:
            pdata_gpcp = np.zeros(shape=[nlat_GPCP,nlon_GPCP])
            pdata_gpcp.fill(np.nan)
        # regrid on LMR grid
        if np.isnan(pdata_gpcp).all():
            gpcp_trunc = np.zeros(shape=[nlat_new,nlon_new])
            gpcp_trunc.fill(np.nan)
        else:
            gpcp_trunc = regrid(specob_gpcp, specob_new, pdata_gpcp, ntrunc=nlat_new-1, smooth=None)

        # CMAP
        if CMAP_smatch and CMAP_ematch:
            pdata_cmap = np.mean(CMAP[CMAP_smatch:CMAP_ematch,:,:],0)
        else:
            pdata_cmap = np.zeros(shape=[nlat_CMAP,nlon_CMAP])
            pdata_cmap.fill(np.nan)
        # regrid on LMR grid
        if np.isnan(pdata_cmap).all():
            cmap_trunc = np.zeros(shape=[nlat_new,nlon_new])
            cmap_trunc.fill(np.nan)
        else:
            cmap_trunc = regrid(specob_cmap, specob_new, pdata_cmap, ntrunc=nlat_new-1, smooth=None)

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
        
        # ERA
        if ERA_smatch and ERA_ematch:
            pdata_era = np.mean(ERA[ERA_smatch:ERA_ematch,:,:],0)
        else:
            pdata_era = np.zeros(shape=[nlat_ERA,nlon_ERA])
            pdata_era.fill(np.nan)
        # regrid on LMR grid
        if np.isnan(pdata_era).all():
            era_trunc = np.zeros(shape=[nlat_new,nlon_new])
            era_trunc.fill(np.nan)
        else:
            era_trunc = regrid(specob_era, specob_new, pdata_era, ntrunc=nlat_new-1, smooth=None)

            
        if iplot_individual_years:
            # Precipitation products comparison figures (annually-averaged anomaly fields)
            fmin = verif_dict[var][3]; fmax = verif_dict[var][4]; nflevs=41
            fig = plt.figure()
            ax = fig.add_subplot(5,1,1)    
            LMR_plotter(lmr_trunc*verif_dict[var][6],lat2_new,lon2_new,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
            plt.title('LMR '+'T'+str(nlat_new-ifix)+' '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
            plt.clim(fmin,fmax)
            ax = fig.add_subplot(5,1,2)
            LMR_plotter(gpcp_trunc*verif_dict[var][6],lat2_new,lon2_new,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
            plt.title('GPCP '+'T'+str(nlat_new-ifix)+' '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
            #LMR_plotter(pdata_gpcp*verif_dict[var][6],lat_GPCP,lon_GPCP,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
            #plt.title( 'GPCP '+'orig. grid'+' '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
            plt.clim(fmin,fmax)
            ax = fig.add_subplot(5,1,3)    
            LMR_plotter(cmap_trunc*verif_dict[var][6],lat2_new,lon2_new,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
            plt.title('CMAP '+'T'+str(nlat_new-ifix)+' '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
            #LMR_plotter(pdata_cmap*verif_dict[var][6],lat_GPCP,lon_GPCP,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
            #plt.title( 'CMAP '+'orig. grid'+' '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
            plt.clim(fmin,fmax)
            ax = fig.add_subplot(5,1,4)    
            LMR_plotter(tcr_trunc*verif_dict[var][6],lat2_new,lon2_new,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
            plt.title('20CR-V2 '+'T'+str(nlat_new-ifix)+' '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
            #LMR_plotter(pdata_tcr*verif_dict[var][6],lat_TCR,lon_TCR,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
            #plt.title( '20CR-V2 '+'orig. grid'+' '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
            plt.clim(fmin,fmax)
            ax = fig.add_subplot(5,1,5)    
            LMR_plotter(era_trunc*verif_dict[var][6],lat2_new,lon2_new,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
            plt.title('ERA20C '+'T'+str(nlat_new-ifix)+' '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
            #LMR_plotter(pdata_era*verif_dict[var][6],lat_ERA,lon_ERA,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both')
            #plt.title( 'ERA20C '+'orig. grid'+' '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
            plt.clim(fmin,fmax)

            fig.tight_layout()
            plt.savefig(nexp+'_LMR_GPCP_CMAP_TCR_ERA_'+verif_dict[var][1]+'anom_'+str(yr)+'.png')
            plt.close()


        # save the full grids
        lmr_allyears[k,:,:]  = lmr_trunc
        gpcp_allyears[k,:,:] = gpcp_trunc
        cmap_allyears[k,:,:] = cmap_trunc
        tcr_allyears[k,:,:]  = tcr_trunc
        era_allyears[k,:,:]  = era_trunc

        # -----------------------
        # zonal-mean verification
        # -----------------------

        # LMR
        lmr_zm[k,:] = np.mean(lmr_trunc,1)

        # GPCP
        fracok    = np.sum(np.isfinite(gpcp_trunc),axis=1,dtype=np.float16)/float(nlon_GPCP)
        boolok    = np.where(fracok >= valid_frac)
        boolnotok = np.where(fracok < valid_frac)
        for i in boolok:
            gpcp_zm[k,i] = np.nanmean(gpcp_trunc[i,:],axis=1)
        gpcp_zm[k,boolnotok]  = np.NAN

        # CMAP
        fracok    = np.sum(np.isfinite(cmap_trunc),axis=1,dtype=np.float16)/float(nlon_CMAP)
        boolok    = np.where(fracok >= valid_frac)
        boolnotok = np.where(fracok < valid_frac)
        for i in boolok:
            cmap_zm[k,i] = np.nanmean(cmap_trunc[i,:],axis=1)
        cmap_zm[k,boolnotok]  = np.NAN

        # TCR
        tcr_zm[k,:] = np.mean(tcr_trunc,1)

        # ERA
        era_zm[k,:] = np.mean(era_trunc,1)
        
    
        if iplot_loc:
            ncints = 30
            cmap = 'bwr'
            nticks = 6 # number of ticks on the colorbar
            # set contours based on GPCP
            maxabs = np.nanmax(np.abs(gpcp_trunc))
            # round the contour interval, and then set limits to fit
            dc = np.round(maxabs*2/ncints,2)
            cl = dc*ncints/2.
            cints = np.linspace(-cl,cl,ncints,endpoint=True)
        
            # compare LMR with GPCP, CMAP, TCR and ERA
            fig = plt.figure()
        
            ax = fig.add_subplot(3,2,1)
            m1 = bm.Basemap(projection='robin',lon_0=0)
            # maxabs = np.nanmax(np.abs(lmr_trunc))
            cs = m1.contourf(lon2_new,lat2_new,lmr_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
            m1.drawcoastlines()
            cb = m1.colorbar(cs)
            tick_locator = ticker.MaxNLocator(nbins=nticks)
            cb.locator = tick_locator
            cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            cb.update_ticks()
            ax.set_title('LMR '+verif_dict[var][1]+' '+str(ntrunc_new) + ' ' + str(yr))
        
            ax = fig.add_subplot(3,2,3)
            m2 = bm.Basemap(projection='robin',lon_0=0)
            # maxabs = np.nanmax(np.abs(gpcp_trunc))
            cs = m2.contourf(lon2_new,lat2_new,gpcp_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
            m2.drawcoastlines()
            cb = m1.colorbar(cs)
            tick_locator = ticker.MaxNLocator(nbins=nticks)
            cb.locator = tick_locator
            cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            cb.update_ticks()
            ax.set_title('GPCP '+verif_dict[var][1]+' '+str(ntrunc_new) + ' ' + str(yr))
        
            ax = fig.add_subplot(3,2,4)
            m3 = bm.Basemap(projection='robin',lon_0=0)
            # maxabs = np.nanmax(np.abs(cmap_trunc))
            cs = m3.contourf(lon2_new,lat2_new,cmap_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
            m3.drawcoastlines()
            cb = m1.colorbar(cs)
            tick_locator = ticker.MaxNLocator(nbins=nticks)
            cb.locator = tick_locator
            cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            cb.update_ticks()
            ax.set_title('CMAP '+verif_dict[var][1]+' '+str(ntrunc_new) + ' ' + str(yr))

            ax = fig.add_subplot(3,2,5)
            m3 = bm.Basemap(projection='robin',lon_0=0)
            # maxabs = np.nanmax(np.abs(tcr_trunc))
            cs = m3.contourf(lon2_new,lat2_new,tcr_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
            m3.drawcoastlines()
            cb = m1.colorbar(cs)
            tick_locator = ticker.MaxNLocator(nbins=nticks)
            cb.locator = tick_locator
            cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            cb.update_ticks()
            ax.set_title('20CR-V2 '+verif_dict[var][1]+' '+str(ntrunc_new) + ' ' + str(yr))
            
            ax = fig.add_subplot(3,2,6)
            m3 = bm.Basemap(projection='robin',lon_0=0)
            # maxabs = np.nanmax(np.abs(era_trunc))
            cs = m3.contourf(lon2_new,lat2_new,era_trunc,cints,cmap=plt.get_cmap(cmap),vmin=-maxabs,vmax=maxabs)
            m3.drawcoastlines()
            cb = m1.colorbar(cs)
            tick_locator = ticker.MaxNLocator(nbins=nticks)
            cb.locator = tick_locator
            cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            cb.update_ticks()
            ax.set_title('ERA20C '+verif_dict[var][1]+' '+str(ntrunc_new) + ' ' + str(yr))
            
            plt.clim(-maxabs,maxabs)
        
            # get these numbers by adjusting the figure interactively!!!
            plt.subplots_adjust(left=0.05, bottom=0.45, right=0.95, top=0.95, wspace=0.1, hspace=0.0)
            # plt.tight_layout(pad=0.3)
            fig.suptitle(verif_dict[var][1] + ' for ' +str(nya) +' year centered average')
    
        
        # anomaly correlation
        lmrvec  = np.reshape(lmr_trunc,(1,nlat_new*nlon_new))
        gpcpvec = np.reshape(gpcp_trunc,(1,nlat_new*nlon_new))
        cmapvec = np.reshape(cmap_trunc,(1,nlat_new*nlon_new))
        tcrvec  = np.reshape(tcr_trunc,(1,nlat_new*nlon_new))
        eravec  = np.reshape(era_trunc,(1,nlat_new*nlon_new))
        

        # lmr <-> gpcp
        indok = np.isfinite(gpcpvec); nbok = np.sum(indok); nball = gpcpvec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            lg_csave[k] = np.corrcoef(lmrvec[indok],gpcpvec[indok])[0,1]
        else:
            lg_csave[k] = np.nan
        print('  lmr-gpcp correlation    : %s' % str(lg_csave[k]))
        
        # lmr <-> cmap
        indok = np.isfinite(cmapvec); nbok = np.sum(indok); nball = cmapvec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            lc_csave[k] = np.corrcoef(lmrvec[indok],cmapvec[indok])[0,1]
        else:
            lc_csave[k] = np.nan
        print('  lmr-cmap correlation    : %s' % str(lc_csave[k]))

        # lmr <-> tcr
        indok = np.isfinite(tcrvec); nbok = np.sum(indok); nball = tcrvec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            lt_csave[k] = np.corrcoef(lmrvec[indok],tcrvec[indok])[0,1]
        else:
            lt_csave[k] = np.nan
        print('  lmr-tcr correlation     : %s' % str(lt_csave[k]))

        # lmr <-> era
        indok = np.isfinite(eravec); nbok = np.sum(indok); nball = eravec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            le_csave[k] = np.corrcoef(lmrvec[indok],eravec[indok])[0,1]
        else:
            le_csave[k] = np.nan
        print('  lmr-era correlation     : %s' % str(le_csave[k]))
        
        # gpcp <-> cmap
        indok = np.isfinite(cmapvec); nbok = np.sum(indok); nball = cmapvec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            gc_csave[k] = np.corrcoef(gpcpvec[indok],cmapvec[indok])[0,1]
        else:
            gc_csave[k] = np.nan
        print('  gpcp-cmap correlation   : %s' % str(gc_csave[k]))

        # gpcp <-> tcr
        indok = np.isfinite(gpcpvec); nbok = np.sum(indok); nball = gpcpvec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            gt_csave[k] = np.corrcoef(gpcpvec[indok],tcrvec[indok])[0,1]
        else:
            gt_csave[k] = np.nan
        print('  gpcp-tcr correlation    : %s' % str(gt_csave[k]))

        # gpcp <-> era
        indok = np.isfinite(gpcpvec); nbok = np.sum(indok); nball = gpcpvec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            ge_csave[k] = np.corrcoef(gpcpvec[indok],eravec[indok])[0,1]
        else:
            ge_csave[k] = np.nan
        print('  gpcp-era correlation    : %s' % str(ge_csave[k]))
        
        # tcr <-> era
        indok = np.isfinite(eravec); nbok = np.sum(indok); nball = eravec.shape[1]
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            te_csave[k] = np.corrcoef(tcrvec[indok],eravec[indok])[0,1]
        else:
            te_csave[k] = np.nan
        print('  tcr-era correlation     : %s' % str(te_csave[k]))


    # -- plots for anomaly correlation statistics --

    # number of bins in the histograms
    nbins = 15
    corr_range = [-0.6,1.0]
    bins = np.linspace(corr_range[0],corr_range[1],nbins)

    # LMR compared to GPCP, CMAP, TCR and ERA
    fig = plt.figure()
    # GPCP
    ax = fig.add_subplot(4,2,1)
    ax.plot(cyears,lg_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('LMR - GPCP')
    ax.set_xlim(trange[0],trange[-1])
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    # 
    ax = fig.add_subplot(4,2,2)
    ax.hist(lg_csave[~np.isnan(lg_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
    ax.set_title('LMR - GPCP')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lg_csave)),fontsize=11,fontweight='bold')

    # CMAP
    ax = fig.add_subplot(4,2,3)
    ax.plot(cyears,lc_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('LMR - CMAP')
    ax.set_xlim(trange[0],trange[-1])
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    # 
    ax = fig.add_subplot(4,2,4)
    ax.hist(lc_csave[~np.isnan(lc_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
    ax.set_title('LMR - CMAP')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lc_csave)),fontsize=11,fontweight='bold')

    # TCR
    ax = fig.add_subplot(4,2,5)
    ax.plot(cyears,lt_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('LMR - 20CR-V2')
    ax.set_xlim(trange[0],trange[-1])
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    # 
    ax = fig.add_subplot(4,2,6)
    ax.hist(lt_csave[~np.isnan(lt_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
    ax.set_title('LMR - 20CR-V2')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lt_csave)),fontsize=11,fontweight='bold')

    # ERA
    ax = fig.add_subplot(4,2,7)
    ax.plot(cyears,le_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('LMR - ERA20C')
    ax.set_xlim(trange[0],trange[-1])
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    # 
    ax = fig.add_subplot(4,2,8)
    ax.hist(le_csave[~np.isnan(le_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
    ax.set_title('LMR - ERA20C')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(le_csave)),fontsize=11,fontweight='bold')

    fig.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.95, top=0.93, wspace=0.5, hspace=0.5)
    fig.suptitle(verif_dict[var][2]+' anomaly correlation',fontweight='bold') 
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'.pdf', bbox_inches='tight', dpi=300, format='pdf')
        plt.close()


    # Reference : TCR & ERA compared to GPCP + ERA compared to TCR

    fig = plt.figure()

    # TCR <-> GPCP
    ax = fig.add_subplot(3,2,1)
    ax.plot(cyears,gt_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('20CR-V2 - GPCP')
    ax.set_xlim(trange[0],trange[-1])
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    ax.set_xlabel('Year CE',fontweight='bold')
    #
    ax = fig.add_subplot(3,2,2)
    ax.hist(gt_csave[~np.isnan(gt_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
    ax.set_title('20CR-V2 - GPCP')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    ax.set_xlabel('Correlation',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(gt_csave)),fontsize=11,fontweight='bold')

    # ERA <-> GPCP
    ax = fig.add_subplot(3,2,3)
    ax.plot(cyears,ge_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('ERA20C - GPCP')
    ax.set_xlim(trange[0],trange[-1])
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    ax.set_xlabel('Year CE',fontweight='bold')
    #
    ax = fig.add_subplot(3,2,4)
    ax.hist(ge_csave[~np.isnan(ge_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
    ax.set_title('ERA20C - GPCP')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    ax.set_xlabel('Correlation',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(ge_csave)),fontsize=11,fontweight='bold')

    # ERA <-> TCR
    ax = fig.add_subplot(3,2,5)
    ax.plot(cyears,te_csave,lw=2)
    ax.plot([trange[0],trange[-1]],[0,0],'k:')
    ax.set_title('ERA20C - 20CR-V2')
    ax.set_xlim(trange[0],trange[-1])
    ax.set_ylim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Correlation',fontweight='bold')
    ax.set_xlabel('Year CE',fontweight='bold')
    #
    ax = fig.add_subplot(3,2,6)
    ax.hist(te_csave[~np.isnan(te_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
    ax.set_title('ERA20C - GPCP')
    ax.set_xlim(corr_range[0],corr_range[-1])
    ax.set_ylabel('Counts',fontweight='bold')
    ax.set_xlabel('Correlation',fontweight='bold')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    ypos = ymax-0.15*(ymax-ymin)
    xpos = xmin+0.025*(xmax-xmin)
    ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(te_csave)),fontsize=11,fontweight='bold')
    
    fig.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.93, wspace=0.5, hspace=0.5)
    fig.suptitle(verif_dict[var][2]+' anomaly correlation',fontweight='bold') 
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_anomaly_correlation_'+str(trange[0])+'-'+str(trange[1])+'_reference.png')
        plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_anomaly_correlation_'+str(trange[0])+'-'+str(trange[1])+'_reference.pdf', bbox_inches='tight', dpi=300, format='pdf')
        plt.close()


    #
    # BEGIN bias, r and CE calculations
    #

    # correlation and CE at each (lat,lon) point

    lg_err = lmr_allyears  - gpcp_allyears
    lc_err = lmr_allyears  - cmap_allyears
    lr_err = lmr_allyears  - tcr_allyears
    le_err = lmr_allyears  - era_allyears
    gc_err = gpcp_allyears - cmap_allyears
    tg_err = tcr_allyears  - gpcp_allyears
    eg_err = era_allyears  - gpcp_allyears
    te_err = tcr_allyears  - era_allyears

    r_lg  = np.zeros([nlat_new,nlon_new])
    ce_lg = np.zeros([nlat_new,nlon_new])
    r_lc  = np.zeros([nlat_new,nlon_new])
    ce_lc = np.zeros([nlat_new,nlon_new])
    r_lr  = np.zeros([nlat_new,nlon_new])
    ce_lr = np.zeros([nlat_new,nlon_new])
    r_le  = np.zeros([nlat_new,nlon_new])
    ce_le = np.zeros([nlat_new,nlon_new])
    r_gc  = np.zeros([nlat_new,nlon_new])
    ce_gc = np.zeros([nlat_new,nlon_new])
    r_tg  = np.zeros([nlat_new,nlon_new])
    ce_tg = np.zeros([nlat_new,nlon_new])
    r_eg  = np.zeros([nlat_new,nlon_new])
    ce_eg = np.zeros([nlat_new,nlon_new])
    r_te  = np.zeros([nlat_new,nlon_new])
    ce_te = np.zeros([nlat_new,nlon_new])


    # bias
    # ...
    
    # CE
    ce_lg = coefficient_efficiency(gpcp_allyears,lmr_allyears)
    ce_lc = coefficient_efficiency(cmap_allyears,lmr_allyears)
    ce_lr = coefficient_efficiency(tcr_allyears,lmr_allyears)
    ce_le = coefficient_efficiency(era_allyears,lmr_allyears)
    ce_gc = coefficient_efficiency(cmap_allyears,gpcp_allyears)
    ce_tg = coefficient_efficiency(gpcp_allyears,tcr_allyears)
    ce_eg = coefficient_efficiency(gpcp_allyears,era_allyears)
    ce_te = coefficient_efficiency(era_allyears,tcr_allyears)
    
    # Correlation
    for la in range(nlat_new):
        for lo in range(nlon_new):
            # LMR-GPCP
            indok = np.isfinite(gpcp_allyears[:,la,lo])
            nbok = np.sum(indok)
            nball = lmr_allyears[:,la,lo].shape[0]
            ratio = float(nbok)/float(nball)
            if ratio > valid_frac:
                r_lg[la,lo] = np.corrcoef(lmr_allyears[indok,la,lo],gpcp_allyears[indok,la,lo])[0,1]
            else:
                r_lg[la,lo] = np.nan

            # LMR-CMAP
            indok = np.isfinite(cmap_allyears[:,la,lo])
            nbok = np.sum(indok)
            nball = lmr_allyears[:,la,lo].shape[0]
            ratio = float(nbok)/float(nball)
            if ratio > valid_frac:
                r_lc[la,lo] = np.corrcoef(lmr_allyears[indok,la,lo],cmap_allyears[indok,la,lo])[0,1]
            else:
                r_lc[la,lo] = np.nan

            # LMR-TCR
            indok = np.isfinite(tcr_allyears[:,la,lo])
            nbok = np.sum(indok)
            nball = lmr_allyears[:,la,lo].shape[0]
            ratio = float(nbok)/float(nball)
            if ratio > valid_frac:
                r_lr[la,lo] = np.corrcoef(lmr_allyears[indok,la,lo],tcr_allyears[indok,la,lo])[0,1]
            else:
                r_lr[la,lo] = np.nan

            # LMR-ERA
            indok = np.isfinite(era_allyears[:,la,lo])
            nbok = np.sum(indok)
            nball = lmr_allyears[:,la,lo].shape[0]
            ratio = float(nbok)/float(nball)
            if ratio > valid_frac:
                r_le[la,lo] = np.corrcoef(lmr_allyears[indok,la,lo],era_allyears[indok,la,lo])[0,1]
            else:
                r_le[la,lo] = np.nan

            # GPCP-CMAP
            indok = np.isfinite(cmap_allyears[:,la,lo])
            nbok = np.sum(indok)
            nball = gpcp_allyears[:,la,lo].shape[0]
            ratio = float(nbok)/float(nball)
            if ratio > valid_frac:
                r_gc[la,lo] = np.corrcoef(gpcp_allyears[indok,la,lo],cmap_allyears[indok,la,lo])[0,1]
            else:
                r_gc[la,lo] = np.nan

            # GPCP-TCR
            indok = np.isfinite(gpcp_allyears[:,la,lo])
            nbok = np.sum(indok)
            nball = tcr_allyears[:,la,lo].shape[0]
            ratio = float(nbok)/float(nball)
            if ratio > valid_frac:
                r_tg[la,lo] = np.corrcoef(gpcp_allyears[indok,la,lo],tcr_allyears[indok,la,lo])[0,1]
            else:
                r_tg[la,lo] = np.nan

            # GPCP-ERA
            indok = np.isfinite(gpcp_allyears[:,la,lo])
            nbok = np.sum(indok)
            nball = era_allyears[:,la,lo].shape[0]
            ratio = float(nbok)/float(nball)
            if ratio > valid_frac:
                r_eg[la,lo] = np.corrcoef(gpcp_allyears[indok,la,lo],era_allyears[indok,la,lo])[0,1]
            else:
                r_eg[la,lo] = np.nan


            # ERA-TCR
            indok = np.isfinite(era_allyears[:,la,lo])
            nbok = np.sum(indok)
            nball = era_allyears[:,la,lo].shape[0]
            ratio = float(nbok)/float(nball)
            if ratio > valid_frac:
                r_te[la,lo] = np.corrcoef(era_allyears[indok,la,lo],tcr_allyears[indok,la,lo])[0,1]
            else:
                r_te[la,lo] = np.nan
                

    # median
    # ------

    lat_trunc = np.squeeze(lat2_new[:,0])
    indlat = np.where((lat_trunc[:] > -60.0) & (lat_trunc[:] < 60.0))

    # LMR-GPCP
    print('')
    lg_rmedian = str(float('%.2g' % np.median(np.median(r_lg)) ))
    print('lmr-gpcp all-grid median r   : %s' % str(lg_rmedian))
    lg_rmedian60 = str(float('%.2g' % np.median(np.median(r_lg[indlat,:])) ))
    print('lmr-gpcp 60S-60N median r    : %s' % str(lg_rmedian60))
    lg_cemedian = str(float('%.2g' % np.median(np.median(ce_lg)) ))
    print('lmr-gpcp all-grid median ce  : %s' % str(lg_cemedian))
    lg_cemedian60 = str(float('%.2g' % np.median(np.median(ce_lg[indlat,:])) ))
    print('lmr-gpcp 60S-60N median ce   : %s' % str(lg_cemedian60))
    # LMR-CMAP
    print('')
    lc_rmedian = str(float('%.2g' % np.median(np.median(r_lc)) ))
    print('lmr-cmap all-grid median r   : ' + str(lc_rmedian))
    lc_rmedian60 = str(float('%.2g' % np.median(np.median(r_lc[indlat,:])) ))
    print('lmr-cmap 60S-60N median r    : ' + str(lc_rmedian60))
    lc_cemedian = str(float('%.2g' % np.median(np.median(ce_lc)) ))
    print('lmr-cmap all-grid median ce  : ' + str(lc_cemedian))
    lc_cemedian60 = str(float('%.2g' % np.median(np.median(ce_lc[indlat,:])) ))
    print('lmr-cmap 60S-60N median ce   : ' + str(lc_cemedian60))
    # LMR-TCR
    print('')
    lr_rmedian = str(float('%.2g' % np.median(np.median(r_lr)) ))
    print('lmr-tcr all-grid median r    : ' + str(lr_rmedian))
    lr_rmedian60 = str(float('%.2g' % np.median(np.median(r_lr[indlat,:])) ))
    print('lmr-tcr 60S-60N median r     : ' + str(lr_rmedian60))
    lr_cemedian = str(float('%.2g' % np.median(np.median(ce_lr)) ))
    print('lmr-tcr all-grid median ce   : ' + str(lr_cemedian))
    lr_cemedian60 = str(float('%.2g' % np.median(np.median(ce_lr[indlat,:])) ))
    print('lmr-tcr 60S-60N median ce    : ' + str(lr_cemedian60))
    # LMR-ERA
    print('')
    le_rmedian = str(float('%.2g' % np.median(np.median(r_le)) ))
    print('lmr-era all-grid median r    : ' + str(le_rmedian))
    le_rmedian60 = str(float('%.2g' % np.median(np.median(r_le[indlat,:])) ))
    print('lmr-era 60S-60N median r     : ' + str(le_rmedian60))
    le_cemedian = str(float('%.2g' % np.median(np.median(ce_le)) ))
    print('lmr-era all-grid median ce   : ' + str(le_cemedian))
    le_cemedian60 = str(float('%.2g' % np.median(np.median(ce_le[indlat,:])) ))
    print('lmr-era 60S-60N median ce    : ' + str(le_cemedian60))
    # GPCP-CMAP
    print('')
    gc_rmedian = str(float('%.2g' % np.median(np.median(r_gc)) ))
    print('gpcp-cmap all-grid median r  : ' + str(gc_rmedian))
    gc_rmedian60 = str(float('%.2g' % np.median(np.median(r_gc[indlat,:])) ))
    print('gpcp-cmap 60S-60N median r   : ' + str(gc_rmedian60))
    gc_cemedian = str(float('%.2g' % np.median(np.median(ce_gc)) ))
    print('gpcp-cmap all-grid median ce : ' + str(gc_cemedian))
    gc_cemedian60 = str(float('%.2g' % np.median(np.median(ce_gc[indlat,:])) ))
    print('gpcp-cmap 60S-60N median ce  : ' + str(gc_cemedian60))
    # TCR-GPCP
    print('')
    tg_rmedian = str(float('%.2g' % np.median(np.median(r_tg)) ))
    print('gpcp-tcr all-grid median r   : ' + str(tg_rmedian))
    tg_rmedian60 = str(float('%.2g' % np.median(np.median(r_tg[indlat,:])) ))
    print('gpcp-tcr 60S-60N median r    : ' + str(tg_rmedian60))
    tg_cemedian = str(float('%.2g' % np.median(np.median(ce_tg)) ))
    print('gpcp-tcr all-grid median ce  : ' + str(tg_cemedian))
    tg_cemedian60 = str(float('%.2g' % np.median(np.median(ce_tg[indlat,:])) ))
    print('gpcp-tcr 60S-60N median ce   : ' + str(tg_cemedian60))
    # ERA-GPCP
    print('')
    eg_rmedian = str(float('%.2g' % np.median(np.median(r_eg)) ))
    print('gpcp-era all-grid median r   : ' + str(eg_rmedian))
    eg_rmedian60 = str(float('%.2g' % np.median(np.median(r_eg[indlat,:])) ))
    print('gpcp-era 60S-60N median r    : ' + str(eg_rmedian60))
    eg_cemedian = str(float('%.2g' % np.median(np.median(ce_eg)) ))
    print('gpcp-era all-grid median ce  : ' + str(eg_cemedian))
    eg_cemedian60 = str(float('%.2g' % np.median(np.median(ce_eg[indlat,:])) ))
    print('gpcp-era 60S-60N median ce   : ' + str(eg_cemedian60))
    # TCR-ERA
    print('')
    te_rmedian = str(float('%.2g' % np.median(np.median(r_te)) ))
    print('tcr-era all-grid median r    : ' + str(te_rmedian))
    te_rmedian60 = str(float('%.2g' % np.median(np.median(r_te[indlat,:])) ))
    print('tcr-era 60S-60N median r     : ' + str(te_rmedian60))
    te_cemedian = str(float('%.2g' % np.median(np.median(ce_te)) ))
    print('tcr-era all-grid median ce   : ' + str(te_cemedian))
    te_cemedian60 = str(float('%.2g' % np.median(np.median(ce_te[indlat,:])) ))
    print('tcr-era 60S-60N median ce    : ' + str(te_cemedian60))
    print('')    

    # spatial mean (area weighted)
    # LMR-GPCP
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lg,veclat)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lg,veclat)
    lg_rmean_global  = str(float('%.2f' %rmean_global[0]))
    lg_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
    lg_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
    lg_cemean_global = str(float('%.2f' %cemean_global[0]))
    lg_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
    lg_cemean_sh     = str(float('%.2f' %cemean_sh[0]))
    # LMR-CMAP
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lc,veclat)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lc,veclat)
    lc_rmean_global  = str(float('%.2f' %rmean_global[0]))
    lc_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
    lc_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
    lc_cemean_global = str(float('%.2f' %cemean_global[0]))
    lc_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
    lc_cemean_sh     = str(float('%.2f' %cemean_sh[0]))
    # LMR-TCR
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lr,veclat)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lr,veclat)
    lr_rmean_global  = str(float('%.2f' %rmean_global[0]))
    lr_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
    lr_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
    lr_cemean_global = str(float('%.2f' %cemean_global[0]))
    lr_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
    lr_cemean_sh     = str(float('%.2f' %cemean_sh[0]))
    # LMR-ERA
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_le,veclat)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_le,veclat)
    le_rmean_global  = str(float('%.2f' %rmean_global[0]))
    le_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
    le_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
    le_cemean_global = str(float('%.2f' %cemean_global[0]))
    le_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
    le_cemean_sh     = str(float('%.2f' %cemean_sh[0]))

    
    # GPCP-CMAP
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_gc,veclat)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_gc,veclat)
    gc_rmean_global  = str(float('%.2f' %rmean_global[0]))
    gc_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
    gc_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
    gc_cemean_global = str(float('%.2f' %cemean_global[0]))
    gc_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
    gc_cemean_sh     = str(float('%.2f' %cemean_sh[0]))
    # GPCP-TCR
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_tg,veclat)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_tg,veclat)
    tg_rmean_global  = str(float('%.2f' %rmean_global[0]))
    tg_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
    tg_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
    tg_cemean_global = str(float('%.2f' %cemean_global[0]))
    tg_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
    tg_cemean_sh     = str(float('%.2f' %cemean_sh[0]))
    # GPCP-ERA
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_eg,veclat)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_eg,veclat)
    eg_rmean_global  = str(float('%.2f' %rmean_global[0]))
    eg_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
    eg_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
    eg_cemean_global = str(float('%.2f' %cemean_global[0]))
    eg_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
    eg_cemean_sh     = str(float('%.2f' %cemean_sh[0]))
    # ERA-TCR
    [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_te,veclat)
    [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_te,veclat)
    te_rmean_global  = str(float('%.2f' %rmean_global[0]))
    te_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
    te_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
    te_cemean_global = str(float('%.2f' %cemean_global[0]))
    te_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
    te_cemean_sh     = str(float('%.2f' %cemean_sh[0]))

    
    # zonal mean verification
    # gpcp
    r_lg_zm = np.zeros([nlat_new])
    ce_lg_zm = np.zeros([nlat_new])
    lg_err_zm = lmr_zm - gpcp_zm
    # cmap
    r_lc_zm = np.zeros([nlat_new])
    ce_lc_zm = np.zeros([nlat_new])
    lc_err_zm = lmr_zm - cmap_zm
    # tcr
    r_lr_zm = np.zeros([nlat_new])
    ce_lr_zm = np.zeros([nlat_new])
    lr_err_zm = lmr_zm - tcr_zm
    # era
    r_le_zm = np.zeros([nlat_new])
    ce_le_zm = np.zeros([nlat_new])
    le_err_zm = lmr_zm - era_zm


    for la in range(nlat_new):
        # LMR-GPCP
        ce_lg_zm[la] = coefficient_efficiency(gpcp_zm[:,la],lmr_zm[:,la],valid=valid_frac)
        indok = np.isfinite(gpcp_zm[:,la])
        nbok = np.sum(indok)
        nball = len(cyears)
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_lg_zm[la] = np.corrcoef(lmr_zm[indok,la],gpcp_zm[indok,la])[0,1]
        else:
            r_lg_zm[la]  = np.nan

        # LMR-CMAP
        ce_lc_zm[la] = coefficient_efficiency(cmap_zm[:,la],lmr_zm[:,la],valid=valid_frac)    
        indok = np.isfinite(cmap_zm[:,la])
        nbok = np.sum(indok)
        nball = len(cyears)
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_lc_zm[la] = np.corrcoef(lmr_zm[indok,la],cmap_zm[indok,la])[0,1]
        else:
            r_lc_zm[la]  = np.nan

        # LMR-TCR
        ce_lr_zm[la] = coefficient_efficiency(tcr_zm[:,la],lmr_zm[:,la],valid=valid_frac)    
        indok = np.isfinite(tcr_zm[:,la])
        nbok = np.sum(indok)
        nball = len(cyears)
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_lr_zm[la] = np.corrcoef(lmr_zm[indok,la],tcr_zm[indok,la])[0,1]
        else:
            r_lr_zm[la]  = np.nan

        # LMR-ERA
        ce_le_zm[la] = coefficient_efficiency(era_zm[:,la],lmr_zm[:,la],valid=valid_frac)    
        indok = np.isfinite(era_zm[:,la])
        nbok = np.sum(indok)
        nball = len(cyears)
        ratio = float(nbok)/float(nball)
        if ratio > valid_frac:
            r_le_zm[la] = np.corrcoef(lmr_zm[indok,la],era_zm[indok,la])[0,1]
        else:
            r_le_zm[la]  = np.nan            

    #
    # END r and CE
    #
    major_ticks = np.arange(-90, 91, 30)
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)    
    gpcpleg, = ax.plot(r_lg_zm,veclat,'r-',linestyle='-',lw=2,label='GPCP')
    cmapleg, = ax.plot(r_lc_zm,veclat,'b-',linestyle='-',lw=2,label='CMAP')
    tcrleg,  = ax.plot(r_lr_zm,veclat,'k-',linestyle='--',lw=2,label='20CR-V2')
    eraleg,  = ax.plot(r_le_zm,veclat,'0.55',linestyle='--',lw=2,label='ERA20C')
    ax.plot([0,0],[-90,90],'k:')
    ax.set_yticks(major_ticks)                                                       
    plt.ylim([-90,90])
    plt.xlim([-1,1])
    plt.ylabel('Latitude',fontweight='bold')
    plt.xlabel('Correlation',fontweight='bold')
    ax.legend(handles=[gpcpleg,cmapleg,tcrleg,eraleg],handlelength=3.0,ncol=1,fontsize=12,loc='upper right',frameon=False)

    ax = fig.add_subplot(1,2,2)    
    ax.plot(ce_lg_zm,veclat,'r-',linestyle='-',lw=2)
    ax.plot(ce_lc_zm,veclat,'b-',linestyle='-',lw=2)
    ax.plot(ce_lr_zm,veclat,'k-',linestyle='--',lw=2)
    ax.plot(ce_le_zm,veclat,'0.55',linestyle='--',lw=2)
    ax.plot([0,0],[-90,90],'k:')
    ax.set_yticks([])                                                       
    plt.ylim([-90,90])
    plt.xlim([-1.5,1])
    plt.xlabel('Coefficient of efficiency',fontweight='bold')
    plt.suptitle('LMR zonal-mean verification - '+verif_dict[var][2],fontweight='bold')
    fig.tight_layout(pad = 2.0)
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_r_ce_zonal_mean_'+str(trange[0])+'-'+str(trange[1])+'.png') 
        plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_r_ce_zonal_mean_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')
        plt.close()

        
    #
    # r and ce plots
    #

    cbarfmt = '%4.1f'
    nticks = 4 # number of ticks on the colorbar
    if iplot:
        fig = plt.figure()
        ax = fig.add_subplot(4,2,1)    
        LMR_plotter(r_lg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - GPCP '+verif_dict[var][1]+' r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lg_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,2)    
        LMR_plotter(ce_lg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - GPCP '+verif_dict[var][1]+' CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lg_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,3)    
        LMR_plotter(r_lc,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - CMAP '+verif_dict[var][1]+' r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lc_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,4)    
        LMR_plotter(ce_lc,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - CMAP '+verif_dict[var][1]+' CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lc_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,5)    
        LMR_plotter(r_lr,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - 20CR-V2 '+verif_dict[var][1]+' r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lr_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,6)    
        LMR_plotter(ce_lr,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - 20CR-V2 '+verif_dict[var][1]+' CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lr_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,7)    
        LMR_plotter(r_le,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - ERA20C '+verif_dict[var][1]+' r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(le_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,8)    
        LMR_plotter(ce_le,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('LMR - ERA20C '+verif_dict[var][1]+' CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(le_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        fig.tight_layout()
        if fsave:
            print('saving to .png')
            plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.png')
            plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')
            plt.close()


        # Reference
        fig = plt.figure()

        # GPCP <-> CMAP
        ax = fig.add_subplot(4,2,1)    
        LMR_plotter(r_gc,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('GPCP - CMAP '+verif_dict[var][1]+' r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(gc_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,2)    
        LMR_plotter(ce_gc,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('GPCP - CMAP '+verif_dict[var][1]+' CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(gc_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])
        
        # TCR <-> GPCP
        ax = fig.add_subplot(4,2,3)    
        LMR_plotter(r_tg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('20CR-V2 - GPCP '+verif_dict[var][1]+' r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(tg_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,4)    
        LMR_plotter(ce_tg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('20CR-V2 - GPCP '+verif_dict[var][1]+' CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(tg_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        # ERA <-> GPCP
        ax = fig.add_subplot(4,2,5)    
        LMR_plotter(r_eg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('ERA20C - GPCP '+verif_dict[var][1]+' r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(eg_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,6)    
        LMR_plotter(ce_eg,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('ERA20C - GPCP '+verif_dict[var][1]+' CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(eg_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        # ERA <-> TCR
        ax = fig.add_subplot(4,2,7)    
        LMR_plotter(r_te,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('ERA20C - 20CR-V2 '+verif_dict[var][1]+' r '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(te_rmean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        ax = fig.add_subplot(4,2,8)    
        LMR_plotter(ce_te,lat2_new,lon2_new,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks)
        plt.title('ERA20C - 20CR-V2 '+verif_dict[var][1]+' CE '+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(te_cemean_global))
        plt.clim(-1,1)
        ax.title.set_position([.5, 1.05])

        
        fig.tight_layout()
        if fsave:
            print('saving to .png')
            plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_r_ce_'+str(trange[0])+'-'+str(trange[1])+'_reference.png')
            plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_r_ce_'+str(trange[0])+'-'+str(trange[1])+'_reference.pdf',bbox_inches='tight', dpi=300, format='pdf')
            plt.close()


    if iplot:
        plt.show()


    # ensemble calibration
    print(np.shape(lg_err))
    print(np.shape(xam_var))
    LMR_smatch, LMR_ematch = find_date_indices(LMR_time,trange[0],trange[1])
    print(LMR_smatch, LMR_ematch)
    svar = xam_var[LMR_smatch:LMR_ematch,:,:]
    print(np.shape(svar))

    calib = lg_err.var(0)/svar.mean(0)
    print(np.shape(calib))
    print(calib[0:-1,:].mean())


    # create the plot
    mapcolor_calib = truncate_colormap(plt.cm.YlOrBr,0.0,0.8)
    fig = plt.figure()
    cb = LMR_plotter(calib,lat2_new,lon2_new,mapcolor_calib,11,0,10,extend='max',nticks=10)
    #cb.set_ticks(range(11))
    # overlay stations!
    plt.title('Ratio of ensemble-mean error variance to mean ensemble variance \n '+verif_dict[var][2])
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_ensemble_calibration_'+str(trange[0])+'-'+str(trange[1])+'.png')  


    # in loop over lat,lon, add a call to the rank histogram function; need to move up the function def
        
    # NEW look at trends over specified time periods as a function of latitude

    # zonal means of the original LMR data
