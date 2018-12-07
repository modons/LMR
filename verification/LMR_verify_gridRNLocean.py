""" 
Module: LMR_verify_gridRNLocean.py

Purpose: Generates spatial verification statistics of various LMR gridded fields
         against 20th century ocean reanalyses.  

Originator: Robert Tardif, U. of Washington, March 2016

Revisions: 
         - More flexible handling related to availability of verification datasets
           [R. Tardif, U. of Washington - August 2018]

"""
import matplotlib
# need to do this backend when running remotely or to suppress figures interactively
matplotlib.use('Agg')

# generic imports
import numpy as np
import glob, os, sys
from datetime import datetime, timedelta
from netCDF4 import Dataset
import mpl_toolkits.basemap as bm
import matplotlib.pyplot as plt
from matplotlib import ticker
from spharm import Spharmt, getspecindx, regrid
from scipy import stats
import warnings
import sys

# LMR specific imports
sys.path.append('../')
from LMR_utils import regrid_esmpy
from LMR_utils import global_hemispheric_means, assimilated_proxies, coefficient_efficiency
from load_gridded_data import read_gridded_data_CMIP5_model
from LMR_plot_support import *

# change default value of latlon kwarg to True.
bm.latlon_default = True

warnings.filterwarnings('ignore')

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

# where to find reconstruction data
#datadir_output = './data/'
#datadir_output = '/home/disk/kalman2/wperkins/LMR_output/archive'
datadir_output = '/home/disk/kalman3/rtardif/LMR/output'
#datadir_output = '/home/disk/ekman4/rtardif/LMR/output'
#datadir_output = '/home/disk/kalman3/hakim/LMR'

# Directory where reanalysis data can be found
datadir_reanl = '/home/disk/kalman3/rtardif/LMR/data/analyses/'

# file specification
#
# current datasets
# ---
nexp = 'test'
# ---


# perform verification using all recon. MC realizations ( MCset = None )
# or over a custom selection ( MCset = (begin,end) )
# ex. MCset = (0,0)    -> only the first MC run
#     MCset = (0,10)   -> the first 11 MC runs (from 0 to 10 inclusively)
#     MCset = (80,100) -> the 80th to 100th MC runs (21 realizations)
MCset = None
#MCset = (0,10)

# Definition of variables to verify
#                        kind   name     variable long name        bounds   units   mult. factor
verif_dict = \
    {
        'tos_sfc_Omon'    : ('anom','SST', 'Sea surface temperature',-2.0,2.0,'(K)',1.0), \
        'sos_sfc_Omon'    : ('anom','SSS', 'Sea surface salinity',-1.2,1.2,'(psu)',1.0), \
        'ohc_0-700m_Omon' : ('anom','OHC_0-700', 'Ocean heat content (0-700m)', -4.0,4.0,'(10$^{9}$ J m$^{-2}$)',1.0e-9), \
    }

# time range for verification (in years CE)
trange = [1900,2000] #works for nya = 0 

# reference period over which mean is calculated & subtracted 
# from all datasets (in years CE)
ref_period = [1951, 1980] # as in instrumental-era products (e.g. GISTEMP)

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
print('working directory = %s' % workdir)

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

print('mcdir: %s' % str(mcdir))
print('niters = %s' % str(niters))

# check availability of target variables
vars_to_remove = []
for var in verif_vars:
    available = True
    for dir in mcdir:
        ensfiln = workdir + '/' + dir + '/ensemble_mean_'+var+'.npz'
        if not os.path.exists(ensfiln):
            available = False
            continue
    if not available:
        print('WARNING: Variable %s not found in reconstruction output...' %var)
        vars_to_remove.append(var)
if len(vars_to_remove) > 0:
    for var in vars_to_remove:
        verif_vars.remove(var)

# Finally, loop over available verif. variables
for var in verif_vars:

    # read ensemble mean data
    print('\n reading LMR ensemble-mean data...\n')

    first = True
    k = -1
    for dir in mcdir:
        k = k + 1
        ensfiln = workdir + '/' + dir + '/ensemble_mean_'+var+'.npz'
        npzfile = np.load(ensfiln)
        print(npzfile.files)
        tmp = npzfile['xam']
        print('shape of tmp: %s' % str(np.shape(tmp)))
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
        print('max error = %s' % str(max_err))
        raise Exception('sample mean does not match what is in the ensemble files!')

    # sample variance
    xam_var = xam_all.var(0)
    print(np.shape(xam_var))

    print('\n shape of the ensemble array: %s \n' % str(np.shape(xam_all)))
    print('\n shape of the ensemble-mean array: %s \n' % str(np.shape(xam)))


    # ===============================================================
    # BEGIN: load verification data               
    # ===============================================================

    print('\nloading verification data...\n')

    # Define month sequence for the calendar year 
    # (argument needed in upload of reanalysis data)
    annual = list(range(1,13))

    
    # load SODA reanalysis ----------------------------------
    if var == 'tos_sfc_Omon':
        var_rnl = 'sst'
    elif var == 'sos_sfc_Omon':
        var_rnl = 'sss'
    elif var == 'ohc_0-700m_Omon':
        var_rnl = 'ohc_0-700m'
        
    vardict = {var_rnl: verif_dict[var][0]}
    vardef   = var_rnl
    datadir  = datadir_reanl +'SODA'
    datafile = vardef +'_SODAv2.2.4_187101-200812.nc'

    # check file availability
    infile = datadir+'/'+datafile

    soda_available = False
    if os.path.exists(infile):
        soda_available = True

    if soda_available:
        dd = read_gridded_data_CMIP5_model(datadir,datafile,vardict,outtimeavg=annual,
                                           anom_ref=ref_period)
        rtime = dd[vardef]['years']
        SODA_time = np.array([d.year for d in rtime])
        lats = dd[vardef]['lat']
        lons = dd[vardef]['lon']
        latshape = lats.shape
        lonshape = lons.shape
        if len(latshape) == 2 & len(lonshape) == 2:
            # stored in 2D arrays
            lat_SODA = np.unique(lats)
            lon_SODA = np.unique(lons)
            nlat_SODA, = lat_SODA.shape
            nlon_SODA, = lon_SODA.shape
        else:
            # stored in 1D arrays
            lon_SODA  = lons
            lat_SODA  = lats
            nlat_SODA = len(lat_SODA)
            nlon_SODA = len(lon_SODA)
        lon2d_SODA, lat2d_SODA = np.meshgrid(lon_SODA, lat_SODA)

        #SODA = dd[vardef]['value'] + dd[vardef]['climo'] # Full field
        SODA = dd[vardef]['value']                        # Anomalies


    # load ORA20C reanalysis --------------------------------
    if var == 'tos_sfc_Omon':
        var_rnl = 'sst'
    elif var == 'sos_sfc_Omon':
        var_rnl = 'sss'
    elif var == 'ohc_0-700m_Omon':
        var_rnl = 'ohc_0-700m'

    vardict = {var_rnl: verif_dict[var][0]}
    vardef   = var_rnl
    datadir  = datadir_reanl +'ORA20C'
    datafile = vardef +'_ORA20C_ensemble_mean_190001-200912.nc'

    # check file availability
    infile = datadir+'/'+datafile
    
    ora_available = False
    if os.path.exists(infile):
        ora_available = True

    if ora_available:        
        dd = read_gridded_data_CMIP5_model(datadir,datafile,vardict,outtimeavg=annual,
                                           anom_ref=ref_period)
        rtime = dd[vardef]['years']
        ORA20C_time = np.array([d.year for d in rtime])
        lats = dd[vardef]['lat']
        lons = dd[vardef]['lon']
        latshape = lats.shape
        lonshape = lons.shape
        if len(latshape) == 2 & len(lonshape) == 2:
            # stored in 2D arrays
            lat_ORA20C = np.unique(lats)
            lon_ORA20C = np.unique(lons)
            nlat_ORA20C, = lat_ORA20C.shape
            nlon_ORA20C, = lon_ORA20C.shape
        else:
            # stored in 1D arrays
            lon_ORA20C  = lons
            lat_ORA20C  = lats
            nlat_ORA20C = len(lat_ORA20C)
            nlon_ORA20C = len(lon_ORA20C)
        lon2d_ORA20C, lat2d_ORA20C = np.meshgrid(lon_ORA20C, lat_ORA20C)

        #ORA20C = dd[vardef]['value'] + dd[vardef]['climo'] # Full field
        ORA20C = dd[vardef]['value']                        # Anomalies


    # load HadleyEN4 reanalysis ----------------------------------
    if var == 'tos_sfc_Omon':
        var_rnl = 'sst'
    elif var == 'sos_sfc_Omon':
        var_rnl = 'sss'
    elif var == 'ohc_0-700m_Omon':
        var_rnl = 'ohc_0-700m'
        
    vardict = {var_rnl: verif_dict[var][0]}
    vardef   = var_rnl
    datadir  = datadir_reanl +'HadleyEN4'
    datafile = vardef +'_HadleyEN4.2.1g10_190001-201012.nc'

    # check file availability
    infile = datadir+'/'+datafile

    en4_available = False
    if os.path.exists(infile):
        en4_available = True

    if en4_available:
        dd = read_gridded_data_CMIP5_model(datadir,datafile,vardict,outtimeavg=annual,
                                           anom_ref=ref_period)
        rtime = dd[vardef]['years']
        EN4_time = np.array([d.year for d in rtime])
        lats = dd[vardef]['lat']
        lons = dd[vardef]['lon']
        latshape = lats.shape
        lonshape = lons.shape
        if len(latshape) == 2 & len(lonshape) == 2:
            # stored in 2D arrays
            lat_EN4 = np.unique(lats)
            lon_EN4 = np.unique(lons)
            nlat_EN4, = lat_EN4.shape
            nlon_EN4, = lon_EN4.shape
        else:
            # stored in 1D arrays
            lon_EN4  = lons
            lat_EN4  = lats
            nlat_EN4 = len(lat_EN4)
            nlon_EN4 = len(lon_EN4)
        lon2d_EN4, lat2d_EN4 = np.meshgrid(lon_EN4, lat_EN4)

        #EN4 = dd[vardef]['value'] + dd[vardef]['climo'] # Full field
        EN4 = dd[vardef]['value']                        # Anomalies

        
    if not soda_available and not en4_available and not ora_available:
        raise SystemExit('No verification data acailabe for this variable')

    # =============================================================
    # END: load verification data                                 
    # =============================================================

    
    
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

    # SODA
    if soda_available:
        smatch, ematch = find_date_indices(SODA_time,stime,etime)
        SODA = SODA - np.mean(SODA[smatch:ematch,:,:],axis=0)
        
    # ORA20C
    if ora_available:
        smatch, ematch = find_date_indices(ORA20C_time,stime,etime)
        ORA20C = ORA20C - np.mean(ORA20C[smatch:ematch,:,:],axis=0)

    # EN4
    if en4_available:
        smatch, ematch = find_date_indices(EN4_time,stime,etime)
        EN4 = EN4 - np.mean(EN4[smatch:ematch,:,:],axis=0)        



    # -----------------------------------------
    # Verification of global and regional means
    # -----------------------------------------

    # LMR
    veclat = np.unique(lat) # 1d array
    [LMR_mean_global,LMR_mean_nh,LMR_mean_sh]  = global_hemispheric_means(LMR,veclat)

     # SODA
    if soda_available:
        [SODA_mean_global,SODA_mean_nh,SODA_mean_sh]  = global_hemispheric_means(SODA,lat_SODA)
        
    # ORA20C
    if ora_available:
        [ORA20C_mean_global,ORA20C_mean_nh,ORA20C_mean_sh]  = global_hemispheric_means(ORA20C,lat_ORA20C)

    # EN4
    if en4_available:
        [EN4_mean_global,EN4_mean_nh,EN4_mean_sh]  = global_hemispheric_means(EN4,lat_EN4)



    # Annual data
    # -----------

    yplotmin = -0.5;  yplotmax = 0.5; # x 10e+9
    
    # --- Global mean ---

    vyears = range(trange[0],trange[1]+1)
    lmrvec  = np.zeros([len(vyears)]); lmrvec[:]  = np.nan
    sodavec = np.zeros([len(vyears)]); sodavec[:] = np.nan
    oravec  = np.zeros([len(vyears)]); oravec[:]  = np.nan
    en4vec  = np.zeros([len(vyears)]); en4vec[:]  = np.nan
    
    k = 0
    for yr in vyears:
        yrmatch,_ = find_date_indices(LMR_time,yr,yr+1)
        lmrvec[k] =  LMR_mean_global[yrmatch]*verif_dict[var][6]

        if soda_available:
             yrmatch,_ = find_date_indices(SODA_time,yr,yr+1)
             if yrmatch: sodavec[k] =  SODA_mean_global[yrmatch]*verif_dict[var][6]
        
        if ora_available:
             yrmatch,_ = find_date_indices(ORA20C_time,yr,yr+1)
             if yrmatch: oravec[k] =  ORA20C_mean_global[yrmatch]*verif_dict[var][6]

        if en4_available:
             yrmatch,_ = find_date_indices(EN4_time,yr,yr+1)
             if yrmatch: en4vec[k] =  EN4_mean_global[yrmatch]*verif_dict[var][6]

        k +=1

        
    # plot
    fig = plt.figure()
    plt.plot(vyears, lmrvec, 'k', lw=4, label='LMR')

    if soda_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec) & np.isfinite(sodavec))
        r_soda = np.corrcoef(lmrvec[indok],sodavec[indok])[0,1]
        ce_soda = coefficient_efficiency(sodavec,lmrvec,valid=valid_frac)        
        plt.plot(vyears, sodavec, 'g', lw=2, label='SODA')

    if ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec) & np.isfinite(oravec))
        r_ora = np.corrcoef(lmrvec[indok],oravec[indok])[0,1]
        ce_ora = coefficient_efficiency(oravec,lmrvec,valid=valid_frac)        
        plt.plot(vyears, oravec, 'r', lw=2, label='ORA20C')
        
    if en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec) & np.isfinite(en4vec))
        r_en4 = np.corrcoef(lmrvec[indok],en4vec[indok])[0,1]
        ce_en4 = coefficient_efficiency(en4vec,lmrvec,valid=valid_frac)        
        plt.plot(vyears, en4vec, 'b', lw=2, label='EN4')

    xmin, xmax, ymin, ymax = plt.axis()
    ymin = yplotmin ; ymax = yplotmax
    plt.plot([xmin,xmax], [0,0], 'gray', linestyle=':')
    plt.xlim(trange)
    plt.ylim([ymin,ymax])

    plt.xlabel('year CE',fontweight='bold',fontsize=14)
    plt.ylabel(verif_dict[var][2]+' '+verif_dict[var][5],fontweight='bold',fontsize=14)
    plt.title(verif_dict[var][2]+' - Global mean',fontweight='bold',fontsize=14)
    
    plt.legend(loc=2)


    txl = xmin + (xmax-xmin)*.45
    tyl = ymin + (ymax-ymin)*.2
    offset = (ymax-ymin)*.025

    if soda_available:
        plt.text(txl,tyl,'(LMR,SODA)    : r= ' + str("{:5.2f}".format(r_soda)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if ora_available:
        plt.text(txl,tyl,'(LMR,ORA20C)  : r= ' + str("{:5.2f}".format(r_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available:
        plt.text(txl,tyl,'(LMR,EN4)     : r= ' + str("{:5.2f}".format(r_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset

    tyl = tyl-offset
    if soda_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec) & np.isfinite(sodavec))
        r_soda_ora = np.corrcoef(oravec[indok],sodavec[indok])[0,1]
        ce_soda_ora = coefficient_efficiency(sodavec,oravec,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,ORA20C) : r= ' + str("{:5.2f}".format(r_soda_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec) & np.isfinite(en4vec))
        r_en4_ora = np.corrcoef(oravec[indok],en4vec[indok])[0,1]
        ce_en4_ora = coefficient_efficiency(en4vec,oravec,valid=valid_frac) 
        plt.text(txl,tyl,'(EN4,ORA20C)  : r= ' + str("{:5.2f}".format(r_en4_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if soda_available and en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(sodavec) & np.isfinite(en4vec))
        r_soda_en4 = np.corrcoef(sodavec[indok],en4vec[indok])[0,1]
        ce_soda_en4 = coefficient_efficiency(sodavec,en4vec,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,EN4)    : r= ' + str("{:5.2f}".format(r_soda_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset


    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_GLOBALmean_'+verif_dict[var][1]+'_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_GLOBALmean_'+verif_dict[var][1]+'_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.pdf',
                    bbox_inches='tight', dpi=300, format='pdf')
    plt.close()


    # Detrended global mean ---
   
    lmrvec_detrend  = np.zeros([len(vyears)]); lmrvec_detrend[:]  = np.nan
    sodavec_detrend = np.zeros([len(vyears)]); sodavec_detrend[:] = np.nan
    oravec_detrend  = np.zeros([len(vyears)]); oravec_detrend[:]  = np.nan
    en4vec_detrend  = np.zeros([len(vyears)]); en4vec_detrend[:]  = np.nan
    

    xvar = np.arange(len(vyears))
    
    # for LMR
    indok = np.where(np.isfinite(lmrvec))
    lmr_slope, lmr_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],lmrvec[indok])
    lmr_trend = lmr_slope*np.squeeze(xvar) + lmr_intercept
    lmrvec_detrend = lmrvec - lmr_trend

    # for SODA
    indok = np.where(np.isfinite(sodavec))
    soda_slope, soda_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],sodavec[indok])
    soda_trend = soda_slope*np.squeeze(xvar) + soda_intercept
    sodavec_detrend = sodavec - soda_trend

    # for ORA20C
    indok = np.where(np.isfinite(oravec))
    ora_slope, ora_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],oravec[indok])
    ora_trend = ora_slope*np.squeeze(xvar) + ora_intercept
    oravec_detrend = oravec - ora_trend
    
    # for EN4
    indok = np.where(np.isfinite(en4vec))
    en4_slope, en4_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],en4vec[indok])
    en4_trend = en4_slope*np.squeeze(xvar) + en4_intercept
    en4vec_detrend = en4vec - en4_trend


    # Trends
    trend_lmr  = str("{:5.2f}".format(lmr_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    trend_soda = str("{:5.2f}".format(soda_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    trend_ora  = str("{:5.2f}".format(ora_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    trend_en4  = str("{:5.2f}".format(en4_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    
    # plot
    fig = plt.figure()
    plt.plot(vyears, lmrvec_detrend, 'k', lw=4, label="{:7}".format('LMR')+trend_lmr)

    if soda_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec_detrend) & np.isfinite(sodavec_detrend))
        r_soda = np.corrcoef(lmrvec_detrend[indok],sodavec_detrend[indok])[0,1]
        ce_soda = coefficient_efficiency(sodavec_detrend,lmrvec_detrend,valid=valid_frac)        
        plt.plot(vyears, sodavec_detrend, 'g', lw=2, label="{:7}".format('SODA')+trend_soda)

    if ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec_detrend) & np.isfinite(oravec_detrend))
        r_ora = np.corrcoef(lmrvec_detrend[indok],oravec_detrend[indok])[0,1]
        ce_ora = coefficient_efficiency(oravec_detrend,lmrvec_detrend,valid=valid_frac)        
        plt.plot(vyears, oravec_detrend, 'r', lw=2, label="{:7}".format('ORA20C')+trend_ora)
        
    if en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec_detrend) & np.isfinite(en4vec_detrend))
        r_en4 = np.corrcoef(lmrvec_detrend[indok],en4vec_detrend[indok])[0,1]
        ce_en4 = coefficient_efficiency(en4vec_detrend,lmrvec_detrend,valid=valid_frac)        
        plt.plot(vyears, en4vec_detrend, 'b', lw=2, label="{:7}".format('EN4')+trend_en4)

    xmin, xmax, ymin, ymax = plt.axis()
    ymin = yplotmin ; ymax = yplotmax
    plt.plot([xmin,xmax], [0,0], 'gray', linestyle=':')
    plt.xlim(trange)
    plt.ylim([ymin,ymax])

    plt.xlabel('year CE',fontweight='bold',fontsize=14)
    plt.ylabel(verif_dict[var][2]+' '+verif_dict[var][5],fontweight='bold',fontsize=14)
    plt.title(verif_dict[var][2]+' - Detrended global mean',fontweight='bold',fontsize=14)
    
    plt.legend(loc=2, prop={'family':'monospace','size':12})


    txl = xmin + (xmax-xmin)*.45
    tyl = ymin + (ymax-ymin)*.2
    offset = (ymax-ymin)*.025

    if soda_available:
        plt.text(txl,tyl,'(LMR,SODA)    : r= ' + str("{:5.2f}".format(r_soda)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if ora_available:
        plt.text(txl,tyl,'(LMR,ORA20C)  : r= ' + str("{:5.2f}".format(r_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available:
        plt.text(txl,tyl,'(LMR,EN4)     : r= ' + str("{:5.2f}".format(r_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset

    tyl = tyl-offset
    if soda_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec_detrend) & np.isfinite(sodavec_detrend))
        r_soda_ora = np.corrcoef(oravec_detrend[indok],sodavec_detrend[indok])[0,1]
        ce_soda_ora = coefficient_efficiency(sodavec_detrend,oravec_detrend,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,ORA20C) : r= ' + str("{:5.2f}".format(r_soda_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec_detrend) & np.isfinite(en4vec_detrend))
        r_en4_ora = np.corrcoef(oravec_detrend[indok],en4vec_detrend[indok])[0,1]
        ce_en4_ora = coefficient_efficiency(en4vec_detrend,oravec_detrend,valid=valid_frac) 
        plt.text(txl,tyl,'(EN4,ORA20C)  : r= ' + str("{:5.2f}".format(r_en4_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if soda_available and en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(sodavec_detrend) & np.isfinite(en4vec_detrend))
        r_soda_en4 = np.corrcoef(sodavec_detrend[indok],en4vec_detrend[indok])[0,1]
        ce_soda_en4 = coefficient_efficiency(sodavec_detrend,en4vec_detrend,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,EN4)    : r= ' + str("{:5.2f}".format(r_soda_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset

    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_GLOBALmean_'+verif_dict[var][1]+'_r_ce_detrend_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_GLOBALmean_'+verif_dict[var][1]+'_r_ce_detrend_'+str(trange[0])+'-'+str(trange[1])+'.pdf',
                    bbox_inches='tight', dpi=300, format='pdf')
    plt.close()

    
    
    # --- NH mean ---

    lmrvec  = np.zeros([len(vyears)]); lmrvec[:]  = np.nan
    sodavec = np.zeros([len(vyears)]); sodavec[:] = np.nan
    oravec  = np.zeros([len(vyears)]); oravec[:]  = np.nan
    en4vec  = np.zeros([len(vyears)]); en4vec[:]  = np.nan
    
    k = 0
    for yr in vyears:
        yrmatch,_ = find_date_indices(LMR_time,yr,yr+1)
        lmrvec[k] =  LMR_mean_nh[yrmatch]*verif_dict[var][6]

        if soda_available:
             yrmatch,_ = find_date_indices(SODA_time,yr,yr+1)
             if yrmatch: sodavec[k] =  SODA_mean_nh[yrmatch]*verif_dict[var][6]
        
        if ora_available:
             yrmatch,_ = find_date_indices(ORA20C_time,yr,yr+1)
             if yrmatch: oravec[k] =  ORA20C_mean_nh[yrmatch]*verif_dict[var][6]

        if en4_available:
             yrmatch,_ = find_date_indices(EN4_time,yr,yr+1)
             if yrmatch: en4vec[k] =  EN4_mean_nh[yrmatch]*verif_dict[var][6]

        k +=1

        
    # plot
    fig = plt.figure()
    plt.plot(vyears, lmrvec, 'k', lw=4, label='LMR')

    if soda_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec) & np.isfinite(sodavec))
        r_soda = np.corrcoef(lmrvec[indok],sodavec[indok])[0,1]
        ce_soda = coefficient_efficiency(sodavec,lmrvec,valid=valid_frac)        
        plt.plot(vyears, sodavec, 'g', lw=2, label='SODA')

    if ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec) & np.isfinite(oravec))
        r_ora = np.corrcoef(lmrvec[indok],oravec[indok])[0,1]
        ce_ora = coefficient_efficiency(oravec,lmrvec,valid=valid_frac)        
        plt.plot(vyears, oravec, 'r', lw=2, label='ORA20C')
        
    if en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec) & np.isfinite(en4vec))
        r_en4 = np.corrcoef(lmrvec[indok],en4vec[indok])[0,1]
        ce_en4 = coefficient_efficiency(en4vec,lmrvec,valid=valid_frac)        
        plt.plot(vyears, en4vec, 'b', lw=2, label='EN4')

    xmin, xmax, ymin, ymax = plt.axis()
    ymin = yplotmin ; ymax = yplotmax
    plt.plot([xmin,xmax], [0,0], 'gray', linestyle=':')
    plt.xlim(trange)
    plt.ylim([ymin,ymax])

    plt.xlabel('year CE',fontweight='bold',fontsize=14)
    plt.ylabel(verif_dict[var][2]+' '+verif_dict[var][5],fontweight='bold',fontsize=14)
    plt.title(verif_dict[var][2]+' - NH mean',fontweight='bold',fontsize=14)
    
    plt.legend(loc=2)


    txl = xmin + (xmax-xmin)*.45
    tyl = ymin + (ymax-ymin)*.2
    offset = (ymax-ymin)*.025

    if soda_available:
        plt.text(txl,tyl,'(LMR,SODA)    : r= ' + str("{:5.2f}".format(r_soda)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if ora_available:
        plt.text(txl,tyl,'(LMR,ORA20C)  : r= ' + str("{:5.2f}".format(r_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available:
        plt.text(txl,tyl,'(LMR,EN4)     : r= ' + str("{:5.2f}".format(r_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset

    tyl = tyl-offset
    if soda_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec) & np.isfinite(sodavec))
        r_soda_ora = np.corrcoef(oravec[indok],sodavec[indok])[0,1]
        ce_soda_ora = coefficient_efficiency(sodavec,oravec,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,ORA20C) : r= ' + str("{:5.2f}".format(r_soda_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec) & np.isfinite(en4vec))
        r_en4_ora = np.corrcoef(oravec[indok],en4vec[indok])[0,1]
        ce_en4_ora = coefficient_efficiency(en4vec,oravec,valid=valid_frac) 
        plt.text(txl,tyl,'(EN4,ORA20C)  : r= ' + str("{:5.2f}".format(r_en4_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if soda_available and en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(sodavec) & np.isfinite(en4vec))
        r_soda_en4 = np.corrcoef(sodavec[indok],en4vec[indok])[0,1]
        ce_soda_en4 = coefficient_efficiency(sodavec,en4vec,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,EN4)    : r= ' + str("{:5.2f}".format(r_soda_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
        
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_NHmean_'+verif_dict[var][1]+'_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_NHmean_'+verif_dict[var][1]+'_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.pdf', bbox_inches='tight', dpi=300, format='pdf')
    plt.close()


    # Detrended NH mean ---
   
    lmrvec_detrend  = np.zeros([len(vyears)]); lmrvec_detrend[:]  = np.nan
    sodavec_detrend = np.zeros([len(vyears)]); sodavec_detrend[:] = np.nan
    oravec_detrend  = np.zeros([len(vyears)]); oravec_detrend[:]  = np.nan
    en4vec_detrend  = np.zeros([len(vyears)]); en4vec_detrend[:]  = np.nan
    

    xvar = np.arange(len(vyears))
    
    # for LMR
    indok = np.where(np.isfinite(lmrvec))
    lmr_slope, lmr_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],lmrvec[indok])
    lmr_trend = lmr_slope*np.squeeze(xvar) + lmr_intercept
    lmrvec_detrend = lmrvec - lmr_trend

    # for SODA
    indok = np.where(np.isfinite(sodavec))
    soda_slope, soda_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],sodavec[indok])
    soda_trend = soda_slope*np.squeeze(xvar) + soda_intercept
    sodavec_detrend = sodavec - soda_trend

    # for ORA20C
    indok = np.where(np.isfinite(oravec))
    ora_slope, ora_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],oravec[indok])
    ora_trend = ora_slope*np.squeeze(xvar) + ora_intercept
    oravec_detrend = oravec - ora_trend
    
    # for EN4
    indok = np.where(np.isfinite(en4vec))
    en4_slope, en4_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],en4vec[indok])
    en4_trend = en4_slope*np.squeeze(xvar) + en4_intercept
    en4vec_detrend = en4vec - en4_trend

    # Trends
    trend_lmr  = str("{:5.2f}".format(lmr_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    trend_soda = str("{:5.2f}".format(soda_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    trend_ora  = str("{:5.2f}".format(ora_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    trend_en4  = str("{:5.2f}".format(en4_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    
    # plot
    fig = plt.figure()
    plt.plot(vyears, lmrvec_detrend, 'k', lw=4, label="{:7}".format('LMR')+trend_lmr)

    if soda_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec_detrend) & np.isfinite(sodavec_detrend))
        r_soda = np.corrcoef(lmrvec_detrend[indok],sodavec_detrend[indok])[0,1]
        ce_soda = coefficient_efficiency(sodavec_detrend,lmrvec_detrend,valid=valid_frac)        
        plt.plot(vyears, sodavec_detrend, 'g', lw=2, label="{:7}".format('SODA')+trend_soda)

    if ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec_detrend) & np.isfinite(oravec_detrend))
        r_ora = np.corrcoef(lmrvec_detrend[indok],oravec_detrend[indok])[0,1]
        ce_ora = coefficient_efficiency(oravec_detrend,lmrvec_detrend,valid=valid_frac)        
        plt.plot(vyears, oravec_detrend, 'r', lw=2, label="{:7}".format('ORA20C')+trend_ora)
        
    if en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec_detrend) & np.isfinite(en4vec_detrend))
        r_en4 = np.corrcoef(lmrvec_detrend[indok],en4vec_detrend[indok])[0,1]
        ce_en4 = coefficient_efficiency(en4vec_detrend,lmrvec_detrend,valid=valid_frac)        
        plt.plot(vyears, en4vec_detrend, 'b', lw=2, label="{:7}".format('EN4')+trend_en4)

    xmin, xmax, ymin, ymax = plt.axis()
    ymin = yplotmin ; ymax = yplotmax
    plt.plot([xmin,xmax], [0,0], 'gray', linestyle=':')
    plt.xlim(trange)
    plt.ylim([ymin,ymax])

    plt.xlabel('year CE',fontweight='bold',fontsize=14)
    plt.ylabel(verif_dict[var][2]+' '+verif_dict[var][5],fontweight='bold',fontsize=14)
    plt.title(verif_dict[var][2]+' - Detrended NH mean',fontweight='bold',fontsize=14)
    
    plt.legend(loc=2, prop={'family':'monospace','size':12})


    txl = xmin + (xmax-xmin)*.45
    tyl = ymin + (ymax-ymin)*.2
    offset = (ymax-ymin)*.025

    if soda_available:
        plt.text(txl,tyl,'(LMR,SODA)    : r= ' + str("{:5.2f}".format(r_soda)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if ora_available:
        plt.text(txl,tyl,'(LMR,ORA20C)  : r= ' + str("{:5.2f}".format(r_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available:
        plt.text(txl,tyl,'(LMR,EN4)     : r= ' + str("{:5.2f}".format(r_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset

    tyl = tyl-offset
    if soda_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec_detrend) & np.isfinite(sodavec_detrend))
        r_soda_ora = np.corrcoef(oravec_detrend[indok],sodavec_detrend[indok])[0,1]
        ce_soda_ora = coefficient_efficiency(sodavec_detrend,oravec_detrend,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,ORA20C) : r= ' + str("{:5.2f}".format(r_soda_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec_detrend) & np.isfinite(en4vec_detrend))
        r_en4_ora = np.corrcoef(oravec_detrend[indok],en4vec_detrend[indok])[0,1]
        ce_en4_ora = coefficient_efficiency(en4vec_detrend,oravec_detrend,valid=valid_frac) 
        plt.text(txl,tyl,'(EN4,ORA20C)  : r= ' + str("{:5.2f}".format(r_en4_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if soda_available and en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(sodavec_detrend) & np.isfinite(en4vec_detrend))
        r_soda_en4 = np.corrcoef(sodavec_detrend[indok],en4vec_detrend[indok])[0,1]
        ce_soda_en4 = coefficient_efficiency(sodavec_detrend,en4vec_detrend,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,EN4)    : r= ' + str("{:5.2f}".format(r_soda_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset

    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_NHmean_'+verif_dict[var][1]+'_r_ce_detrend_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_NHmean_'+verif_dict[var][1]+'_r_ce_detrend_'+str(trange[0])+'-'+str(trange[1])+'.pdf',
                    bbox_inches='tight', dpi=300, format='pdf')
    plt.close()


    
    # --- SH mean ---

    lmrvec  = np.zeros([len(vyears)]); lmrvec[:]  = np.nan
    sodavec = np.zeros([len(vyears)]); sodavec[:] = np.nan
    oravec  = np.zeros([len(vyears)]); oravec[:]  = np.nan
    en4vec  = np.zeros([len(vyears)]); en4vec[:]  = np.nan
    
    k = 0
    for yr in vyears:
        yrmatch,_ = find_date_indices(LMR_time,yr,yr+1)
        lmrvec[k] =  LMR_mean_sh[yrmatch]*verif_dict[var][6]

        if soda_available:
             yrmatch,_ = find_date_indices(SODA_time,yr,yr+1)
             if yrmatch: sodavec[k] =  SODA_mean_sh[yrmatch]*verif_dict[var][6]
        
        if ora_available:
             yrmatch,_ = find_date_indices(ORA20C_time,yr,yr+1)
             if yrmatch: oravec[k] =  ORA20C_mean_sh[yrmatch]*verif_dict[var][6]

        if en4_available:
             yrmatch,_ = find_date_indices(EN4_time,yr,yr+1)
             if yrmatch: en4vec[k] =  EN4_mean_sh[yrmatch]*verif_dict[var][6]

        k +=1

        
    # plot
    fig = plt.figure()
    plt.plot(vyears, lmrvec, 'k', lw=4, label='LMR')

    if soda_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec) & np.isfinite(sodavec))
        r_soda = np.corrcoef(lmrvec[indok],sodavec[indok])[0,1]
        ce_soda = coefficient_efficiency(sodavec,lmrvec,valid=valid_frac)        
        plt.plot(vyears, sodavec, 'g', lw=2, label='SODA')

    if ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec) & np.isfinite(oravec))
        r_ora = np.corrcoef(lmrvec[indok],oravec[indok])[0,1]
        ce_ora = coefficient_efficiency(oravec,lmrvec,valid=valid_frac)        
        plt.plot(vyears, oravec, 'r', lw=2, label='ORA20C')
        
    if en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec) & np.isfinite(en4vec))
        r_en4 = np.corrcoef(lmrvec[indok],en4vec[indok])[0,1]
        ce_en4 = coefficient_efficiency(en4vec,lmrvec,valid=valid_frac)        
        plt.plot(vyears, en4vec, 'b', lw=2, label='EN4')

    xmin, xmax, ymin, ymax = plt.axis()
    ymin = yplotmin ; ymax = yplotmax
    plt.plot([xmin,xmax], [0,0], 'gray', linestyle=':')
    plt.xlim(trange)
    plt.ylim([ymin,ymax])

    plt.xlabel('year CE',fontweight='bold',fontsize=14)
    plt.ylabel(verif_dict[var][2]+' '+verif_dict[var][5],fontweight='bold',fontsize=14)
    plt.title(verif_dict[var][2]+' - SH mean',fontweight='bold',fontsize=14)
    
    plt.legend(loc=2)


    txl = xmin + (xmax-xmin)*.45
    tyl = ymin + (ymax-ymin)*.2
    offset = (ymax-ymin)*.025

    if soda_available:
        plt.text(txl,tyl,'(LMR,SODA)    : r= ' + str("{:5.2f}".format(r_soda)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if ora_available:
        plt.text(txl,tyl,'(LMR,ORA20C)  : r= ' + str("{:5.2f}".format(r_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available:
        plt.text(txl,tyl,'(LMR,EN4)     : r= ' + str("{:5.2f}".format(r_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset

    tyl = tyl-offset
    if soda_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec) & np.isfinite(sodavec))
        r_soda_ora = np.corrcoef(oravec[indok],sodavec[indok])[0,1]
        ce_soda_ora = coefficient_efficiency(sodavec,oravec,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,ORA20C) : r= ' + str("{:5.2f}".format(r_soda_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec) & np.isfinite(en4vec))
        r_en4_ora = np.corrcoef(oravec[indok],en4vec[indok])[0,1]
        ce_en4_ora = coefficient_efficiency(en4vec,oravec,valid=valid_frac) 
        plt.text(txl,tyl,'(EN4,ORA20C)  : r= ' + str("{:5.2f}".format(r_en4_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if soda_available and en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(sodavec) & np.isfinite(en4vec))
        r_soda_en4 = np.corrcoef(sodavec[indok],en4vec[indok])[0,1]
        ce_soda_en4 = coefficient_efficiency(sodavec,en4vec,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,EN4)    : r= ' + str("{:5.2f}".format(r_soda_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset

    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_SHmean_'+verif_dict[var][1]+'_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_SHmean_'+verif_dict[var][1]+'_r_ce_'+str(trange[0])+'-'+str(trange[1])+'.pdf',
                    bbox_inches='tight', dpi=300, format='pdf')
    plt.close()


    # Detrended SH mean ---
   
    lmrvec_detrend  = np.zeros([len(vyears)]); lmrvec_detrend[:]  = np.nan
    sodavec_detrend = np.zeros([len(vyears)]); sodavec_detrend[:] = np.nan
    oravec_detrend  = np.zeros([len(vyears)]); oravec_detrend[:]  = np.nan
    en4vec_detrend  = np.zeros([len(vyears)]); en4vec_detrend[:]  = np.nan
    

    xvar = np.arange(len(vyears))
    
    # for LMR
    indok = np.where(np.isfinite(lmrvec))
    lmr_slope, lmr_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],lmrvec[indok])
    lmr_trend = lmr_slope*np.squeeze(xvar) + lmr_intercept
    lmrvec_detrend = lmrvec - lmr_trend

    # for SODA
    indok = np.where(np.isfinite(sodavec))
    soda_slope, soda_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],sodavec[indok])
    soda_trend = soda_slope*np.squeeze(xvar) + soda_intercept
    sodavec_detrend = sodavec - soda_trend

    # for ORA20C
    indok = np.where(np.isfinite(oravec))
    ora_slope, ora_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],oravec[indok])
    ora_trend = ora_slope*np.squeeze(xvar) + ora_intercept
    oravec_detrend = oravec - ora_trend
    
    # for EN4
    indok = np.where(np.isfinite(en4vec))
    en4_slope, en4_intercept, r_value, p_value, std_err = stats.linregress(xvar[indok],en4vec[indok])
    en4_trend = en4_slope*np.squeeze(xvar) + en4_intercept
    en4vec_detrend = en4vec - en4_trend

    # Trends
    trend_lmr  = str("{:5.2f}".format(lmr_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    trend_soda = str("{:5.2f}".format(soda_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    trend_ora  = str("{:5.2f}".format(ora_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    trend_en4  = str("{:5.2f}".format(en4_slope*100.)).ljust(5,' ')+' '+verif_dict[var][5].rstrip(')')+'/100yrs)'
    
    # plot
    fig = plt.figure()
    plt.plot(vyears, lmrvec_detrend, 'k', lw=4, label="{:7}".format('LMR')+trend_lmr)

    if soda_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec_detrend) & np.isfinite(sodavec_detrend))
        r_soda = np.corrcoef(lmrvec_detrend[indok],sodavec_detrend[indok])[0,1]
        ce_soda = coefficient_efficiency(sodavec_detrend,lmrvec_detrend,valid=valid_frac)        
        plt.plot(vyears, sodavec_detrend, 'g', lw=2, label="{:7}".format('SODA')+trend_soda)

    if ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec_detrend) & np.isfinite(oravec_detrend))
        r_ora = np.corrcoef(lmrvec_detrend[indok],oravec_detrend[indok])[0,1]
        ce_ora = coefficient_efficiency(oravec_detrend,lmrvec_detrend,valid=valid_frac)        
        plt.plot(vyears, oravec_detrend, 'r', lw=2, label="{:7}".format('ORA20C')+trend_ora)
        
    if en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(lmrvec_detrend) & np.isfinite(en4vec_detrend))
        r_en4 = np.corrcoef(lmrvec_detrend[indok],en4vec_detrend[indok])[0,1]
        ce_en4 = coefficient_efficiency(en4vec_detrend,lmrvec_detrend,valid=valid_frac)        
        plt.plot(vyears, en4vec_detrend, 'b', lw=2, label="{:7}".format('EN4')+trend_en4)

    xmin, xmax, ymin, ymax = plt.axis()
    ymin = yplotmin ; ymax = yplotmax
    plt.plot([xmin,xmax], [0,0], 'gray', linestyle=':')
    plt.xlim(trange)
    plt.ylim([ymin,ymax])

    plt.xlabel('year CE',fontweight='bold',fontsize=14)
    plt.ylabel(verif_dict[var][2]+' '+verif_dict[var][5],fontweight='bold',fontsize=14)
    plt.title(verif_dict[var][2]+' - Detrended SH mean',fontweight='bold',fontsize=14)
    
    plt.legend(loc=2, prop={'family':'monospace','size':12})


    txl = xmin + (xmax-xmin)*.45
    tyl = ymin + (ymax-ymin)*.2
    offset = (ymax-ymin)*.025

    if soda_available:
        plt.text(txl,tyl,'(LMR,SODA)    : r= ' + str("{:5.2f}".format(r_soda)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if ora_available:
        plt.text(txl,tyl,'(LMR,ORA20C)  : r= ' + str("{:5.2f}".format(r_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available:
        plt.text(txl,tyl,'(LMR,EN4)     : r= ' + str("{:5.2f}".format(r_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset

    tyl = tyl-offset
    if soda_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec_detrend) & np.isfinite(sodavec_detrend))
        r_soda_ora = np.corrcoef(oravec_detrend[indok],sodavec_detrend[indok])[0,1]
        ce_soda_ora = coefficient_efficiency(sodavec_detrend,oravec_detrend,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,ORA20C) : r= ' + str("{:5.2f}".format(r_soda_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if en4_available and ora_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(oravec_detrend) & np.isfinite(en4vec_detrend))
        r_en4_ora = np.corrcoef(oravec_detrend[indok],en4vec_detrend[indok])[0,1]
        ce_en4_ora = coefficient_efficiency(en4vec_detrend,oravec_detrend,valid=valid_frac) 
        plt.text(txl,tyl,'(EN4,ORA20C)  : r= ' + str("{:5.2f}".format(r_en4_ora)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_en4_ora)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset
    if soda_available and en4_available:
        # need to find indices of overlaping valid data in both arrays
        indok = np.where(np.isfinite(sodavec_detrend) & np.isfinite(en4vec_detrend))
        r_soda_en4 = np.corrcoef(sodavec_detrend[indok],en4vec_detrend[indok])[0,1]
        ce_soda_en4 = coefficient_efficiency(sodavec_detrend,en4vec_detrend,valid=valid_frac) 
        plt.text(txl,tyl,'(SODA,EN4)    : r= ' + str("{:5.2f}".format(r_soda_en4)).ljust(5,' ') + ' CE= ' + str("{:5.2f}".format(ce_soda_en4)).ljust(5,' '),
                 fontsize=14, family='monospace')
        tyl = tyl-offset

    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_SHmean_'+verif_dict[var][1]+'_r_ce_detrend_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_SHmean_'+verif_dict[var][1]+'_r_ce_detrend_'+str(trange[0])+'-'+str(trange[1])+'.pdf',
                    bbox_inches='tight', dpi=300, format='pdf')
    plt.close()


    
    # ----------------------------------------------------------
    # Regridding the data for gridpoint-by-gridpoint comparisons
    # ----------------------------------------------------------
    print('\n regridding data to a common grid...\n')

    iplot_loc= False

    # regrid reanalysis data to LMR grid
    target_grid_nlat = nlat
    target_grid_nlon = nlon
    target_grid_include_poles = True
    veclat = np.unique(lat) # 1d array

    
    # loop over years of interest and transform...specify trange at top of file
    iw = 0
    if nya > 0:
        iw = (nya-1)/2

    cyears = list(range(trange[0],trange[1]))
    ls_csave = np.zeros([len(cyears)])
    le_csave = np.zeros([len(cyears)])
    lo_csave = np.zeros([len(cyears)])
    so_csave = np.zeros([len(cyears)])
    se_csave = np.zeros([len(cyears)])
    oe_csave = np.zeros([len(cyears)])
    
    lmr_allyears = np.zeros([len(cyears),target_grid_nlat,target_grid_nlon])
    soda_allyears = np.zeros([len(cyears),target_grid_nlat,target_grid_nlon])
    en4_allyears = np.zeros([len(cyears),target_grid_nlat,target_grid_nlon])
    ora20c_allyears = np.zeros([len(cyears),target_grid_nlat,target_grid_nlon])
    
    lmr_zm = np.zeros([len(cyears),target_grid_nlat])
    soda_zm = np.zeros([len(cyears),target_grid_nlat])
    en4_zm = np.zeros([len(cyears),target_grid_nlat])    
    ora20c_zm = np.zeros([len(cyears),target_grid_nlat])    
    
    k = -1
    for yr in cyears:
        k = k + 1
        LMR_smatch, LMR_ematch = find_date_indices(LMR_time,yr-iw,yr+iw+1)
        if soda_available:
            SODA_smatch, SODA_ematch = find_date_indices(SODA_time,yr-iw,yr+iw+1)
        if en4_available:
            EN4_smatch, EN4_ematch = find_date_indices(EN4_time,yr-iw,yr+iw+1)
        if ora_available:
            ORA20C_smatch, ORA20C_ematch = find_date_indices(ORA20C_time,yr-iw,yr+iw+1)        

        print('------------------------------------------------------------------------')
        print('working on year... %5s' % str(yr))
        print('                   %5s LMR index= %5s : LMR year= %5s' % (str(yr), str(LMR_smatch),str(LMR_time[LMR_smatch])))

        # -- LMR
        lmr_trunc = np.mean(LMR[LMR_smatch:LMR_ematch,:,:],0)    

        # -- SODA
        if soda_available:
            if SODA_smatch and SODA_ematch:
                pdata_soda = np.mean(SODA[SODA_smatch:SODA_ematch,:,:],0)
            else:
                pdata_soda = np.zeros(shape=[nlat_SODA,nlon_SODA])
                pdata_soda.fill(np.nan)

            # regrid reanalyses on LMR grid
            if np.isnan(pdata_soda).all():
                soda_trunc = np.zeros(shape=[target_grid_nlat,target_grid_nlon])
                soda_trunc.fill(np.nan)
            else:
                var_array_orig = np.zeros(shape=[nlat_SODA*nlon_SODA,1])
                var_array_orig[:,0] = pdata_soda.flatten()
                [var_array_new,
                 lat_new,
                 lon_new] = regrid_esmpy(target_grid_nlat,
                                         target_grid_nlon,
                                         1,
                                         var_array_orig,
                                         lat2d_SODA,
                                         lon2d_SODA,
                                         nlat_SODA,
                                         nlon_SODA,
                                         include_poles=target_grid_include_poles,
                                         method='bilinear')

                soda_trunc = np.reshape(var_array_new[:,0],(target_grid_nlat,target_grid_nlon))

        # -- ORA20C
        if ora_available:
            if ORA20C_smatch and ORA20C_ematch:
                pdata_ora20c = np.mean(ORA20C[ORA20C_smatch:ORA20C_ematch,:,:],0)
            else:
                pdata_ora20c = np.zeros(shape=[nlat_ORA20C,nlon_ORA20C])
                pdata_ora20c.fill(np.nan)

            # regrid reanalyses on LMR grid
            if np.isnan(pdata_ora20c).all():
                ora20c_trunc = np.zeros(shape=[target_grid_nlat,target_grid_nlon])
                ora20c_trunc.fill(np.nan)
            else:
                var_array_orig = np.zeros(shape=[nlat_ORA20C*nlon_ORA20C,1])
                var_array_orig[:,0] = pdata_ora20c.flatten()
                [var_array_new,
                 lat_new,
                 lon_new] = regrid_esmpy(target_grid_nlat,
                                         target_grid_nlon,
                                         1,
                                         var_array_orig,
                                         lat2d_ORA20C,
                                         lon2d_ORA20C,
                                         nlat_ORA20C,
                                         nlon_ORA20C,
                                         include_poles=target_grid_include_poles,
                                         method='bilinear')

                ora20c_trunc = np.reshape(var_array_new[:,0],(target_grid_nlat,target_grid_nlon))

        # -- EN4
        if en4_available:
            if EN4_smatch and EN4_ematch:
                pdata_en4 = np.mean(EN4[EN4_smatch:EN4_ematch,:,:],0)
            else:
                pdata_en4 = np.zeros(shape=[nlat_EN4,nlon_EN4])
                pdata_en4.fill(np.nan)

            # regrid reanalyses on LMR grid
            if np.isnan(pdata_en4).all():
                en4_trunc = np.zeros(shape=[target_grid_nlat,target_grid_nlon])
                en4_trunc.fill(np.nan)
            else:
                var_array_orig = np.zeros(shape=[nlat_EN4*nlon_EN4,1])
                var_array_orig[:,0] = pdata_en4.flatten()
                [var_array_new,
                 lat_new,
                 lon_new] = regrid_esmpy(target_grid_nlat,
                                         target_grid_nlon,
                                         1,
                                         var_array_orig,
                                         lat2d_EN4,
                                         lon2d_EN4,
                                         nlat_EN4,
                                         nlon_EN4,
                                         include_poles=target_grid_include_poles,
                                         method='bilinear')

                en4_trunc = np.reshape(var_array_new[:,0],(target_grid_nlat,target_grid_nlon))                

        if iplot_individual_years:
                        
            # Reanalysis comparison figures (annually-averaged anomaly fields)
            fmin = verif_dict[var][3]; fmax = verif_dict[var][4]; nflevs=41
            fig = plt.figure()

            nbframes = 1
            if soda_available: nbframes +=1
            if en4_available: nbframes +=1
            if ora_available: nbframes +=1

            frame = 1
            ax = fig.add_subplot(nbframes,1,frame)    
            LMR_plotter(lmr_trunc*verif_dict[var][6],lat2,lon2,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both',backg='lightgrey')
            plt.title('LMR '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
            plt.clim(fmin,fmax)
            if soda_available:
                frame +=1
                ax = fig.add_subplot(nbframes,1,frame)    
                LMR_plotter(soda_trunc*verif_dict[var][6],lat2,lon2,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both',backg='lightgrey')
                plt.title('SODA '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
                plt.clim(fmin,fmax)
            if ora_available:
                frame +=1
                ax = fig.add_subplot(nbframes,1,frame)    
                LMR_plotter(ora20c_trunc*verif_dict[var][6],lat2,lon2,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both',backg='lightgrey')
                plt.title('ORA-20C '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
                plt.clim(fmin,fmax)
            if en4_available:
                frame +=1
                ax = fig.add_subplot(nbframes,1,frame)    
                LMR_plotter(en4_trunc*verif_dict[var][6],lat2,lon2,'bwr',nflevs,vmin=fmin,vmax=fmax,extend='both',backg='lightgrey')
                plt.title('EN4 '+verif_dict[var][1]+' anom. '+verif_dict[var][5]+' '+str(yr), fontweight='bold')
                plt.clim(fmin,fmax)
                
            fstring = '_LMR_'
            if soda_available: fstring = fstring +'SODA_'
            if ora_available: fstring = fstring +'ORA20C_'
            if soda_available: fstring = fstring +'EN4_'
            
            fig.tight_layout()
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
            plt.savefig(nexp+fstring+verif_dict[var][1]+'anom_'+str(yr)+'.png')
            plt.close()

        
        # save the new grids
        lmr_allyears[k,:,:] = lmr_trunc
        if soda_available: soda_allyears[k,:,:] = soda_trunc
        if ora_available: ora20c_allyears[k,:,:] = ora20c_trunc
        if en4_available: en4_allyears[k,:,:] = en4_trunc
        
        # ---------------
        # zonal-mean data
        # ---------------

        # LMR
        # ocean fields: need to handle missing data (land grid pts)
        fracok    = np.sum(np.isfinite(lmr_trunc),axis=1,dtype=np.float16)/float(nlon)
        boolok    = np.where(fracok >= valid_frac)
        boolnotok = np.where(fracok < valid_frac)
        for i in boolok:
            lmr_zm[k,i] = np.nanmean(lmr_trunc[i,:],axis=1)
        lmr_zm[k,boolnotok]  = np.NAN
        
        # SODA
        if soda_available:
            fracok    = np.sum(np.isfinite(soda_trunc),axis=1,dtype=np.float16)/float(nlon_SODA)
            boolok    = np.where(fracok >= valid_frac)
            boolnotok = np.where(fracok < valid_frac)
            for i in boolok:
                soda_zm[k,i] = np.nanmean(soda_trunc[i,:],axis=1)
            soda_zm[k,boolnotok]  = np.NAN
            
        # ORA20C
        if ora_available:
            fracok    = np.sum(np.isfinite(ora20c_trunc),axis=1,dtype=np.float16)/float(nlon_ORA20C)
            boolok    = np.where(fracok >= valid_frac)
            boolnotok = np.where(fracok < valid_frac)
            for i in boolok:
                ora20c_zm[k,i] = np.nanmean(ora20c_trunc[i,:],axis=1)
            ora20c_zm[k,boolnotok]  = np.NAN

        # EN4
        if en4_available:
            fracok    = np.sum(np.isfinite(en4_trunc),axis=1,dtype=np.float16)/float(nlon_EN4)
            boolok    = np.where(fracok >= valid_frac)
            boolnotok = np.where(fracok < valid_frac)
            for i in boolok:
                en4_zm[k,i] = np.nanmean(en4_trunc[i,:],axis=1)
            en4_zm[k,boolnotok]  = np.NAN
            
        
        # -------------------
        # anomaly correlation
        # -------------------
        lmrvec  = np.reshape(lmr_trunc,(1,nlat*nlon))

        # lmr <-> soda
        if soda_available:
            sodavec = np.reshape(soda_trunc,(1,nlat*nlon))
            indok = np.isfinite(sodavec); nbok = np.sum(indok); nball = sodavec.shape[1]
            ratio = float(nbok)/float(nball)

            if ratio > valid_frac:
                # need to find indices of overlaping valid data in both arrays
                indok2 = np.where(np.isfinite(lmrvec) & np.isfinite(sodavec))
                ls_csave[k] = np.corrcoef(lmrvec[indok2],sodavec[indok2])[0,1]
            else:
                ls_csave[k] = np.nan
            print('  lmr-soda correlation    : %s' % str("{:7.4f}".format(ls_csave[k])))
            
        # lmr <-> ora20c
        if ora_available:
            ora20cvec = np.reshape(ora20c_trunc,(1,nlat*nlon))
            indok = np.isfinite(ora20cvec); nbok = np.sum(indok); nball = ora20cvec.shape[1]
            ratio = float(nbok)/float(nball)

            if ratio > valid_frac:
                # need to find indices of overlaping valid data in both arrays
                indok2 = np.where(np.isfinite(lmrvec) & np.isfinite(ora20cvec))
                lo_csave[k] = np.corrcoef(lmrvec[indok2],ora20cvec[indok2])[0,1]
            else:
                lo_csave[k] = np.nan
            print('  lmr-ora20c correlation  : %s' % str("{:7.4f}".format(lo_csave[k])))

        # lmr <-> en4
        if en4_available:
            en4vec = np.reshape(en4_trunc,(1,nlat*nlon))
            indok = np.isfinite(en4vec); nbok = np.sum(indok); nball = sodavec.shape[1]
            ratio = float(nbok)/float(nball)

            if ratio > valid_frac:
                # need to find indices of overlaping valid data in both arrays
                indok2 = np.where(np.isfinite(lmrvec) & np.isfinite(en4vec))
                le_csave[k] = np.corrcoef(lmrvec[indok2],en4vec[indok2])[0,1]
            else:
                le_csave[k] = np.nan
            print('  lmr-en4 correlation     : %s' % str("{:7.4f}".format(le_csave[k])))


        # soda <-> ora20c
        if soda_available and ora_available:
            indok = np.isfinite(ora20cvec); nbok = np.sum(indok); nball = ora20cvec.shape[1]
            ratio = float(nbok)/float(nball)

            if ratio > valid_frac:
                # need to find indices of overlaping valid data in both arrays
                indok2 = np.where(np.isfinite(sodavec) & np.isfinite(ora20cvec))
                so_csave[k] = np.corrcoef(sodavec[indok2],ora20cvec[indok2])[0,1]
            else:
                so_csave[k] = np.nan
            print('  soda-ora20c correlation : %s' % str("{:7.4f}".format(so_csave[k])))
        
        # soda <-> en4
        if soda_available and en4_available:
            indok = np.isfinite(en4vec); nbok = np.sum(indok); nball = en4vec.shape[1]
            ratio = float(nbok)/float(nball)

            if ratio > valid_frac:
                # need to find indices of overlaping valid data in both arrays
                indok2 = np.where(np.isfinite(sodavec) & np.isfinite(en4vec))
                se_csave[k] = np.corrcoef(sodavec[indok2],en4vec[indok2])[0,1]
            else:
                se_csave[k] = np.nan
            print('  soda-en4 correlation    : %s' % str("{:7.4f}".format(se_csave[k])))
            
        # ora20c <-> en4
        if en4_available and ora_available:
            indok = np.isfinite(ora20cvec); nbok = np.sum(indok); nball = ora20cvec.shape[1]
            ratio = float(nbok)/float(nball)

            if ratio > valid_frac:
                # need to find indices of overlaping valid data in both arrays
                indok2 = np.where(np.isfinite(en4vec) & np.isfinite(ora20cvec))
                oe_csave[k] = np.corrcoef(en4vec[indok2],ora20cvec[indok2])[0,1]
            else:
                oe_csave[k] = np.nan
            print('  ora20c-en4 correlation  : %s' % str("{:7.4f}".format(oe_csave[k])))

            
        
    # ----------------------------------------
    # plots for anomaly correlation statistics
    # ----------------------------------------    

    # number of bins in the histograms
    nbins = 15
    corr_range = [-0.6,1.0]
    bins = np.linspace(corr_range[0],corr_range[1],nbins)

    fig = plt.figure()

    frame = 1
    nbframes = 0
    if soda_available: nbframes +=1
    if en4_available: nbframes +=1
    if ora_available: nbframes +=1
    
    # LMR <-> SODA
    if soda_available:
        # - time series
        ax = fig.add_subplot(nbframes,2,frame)
        ax.plot(cyears,ls_csave,lw=2)
        ax.plot([trange[0],trange[-1]],[0,0],'k:')
        ax.set_title('LMR - SODA')
        ax.set_xlim(trange[0],trange[-1])
        ax.set_ylim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Correlation',fontweight='bold')
        # - histogram
        frame +=1
        ax = fig.add_subplot(nbframes,2,frame)
        ax.hist(ls_csave[~np.isnan(ls_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
        ax.set_title('LMR - SODA')
        ax.set_xlim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Counts',fontweight='bold')
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        ypos = ymax-0.15*(ymax-ymin)
        xpos = xmin+0.025*(xmax-xmin)
        ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(ls_csave)),fontsize=11,fontweight='bold')

    # LMR <-> ORA20C
    if ora_available:
        # - time series
        if frame > 1: frame +=1
        ax = fig.add_subplot(nbframes,2,frame)
        ax.plot(cyears,lo_csave,lw=2)
        ax.plot([trange[0],trange[-1]],[0,0],'k:')
        ax.set_title('LMR - ORA-20C')
        ax.set_xlim(trange[0],trange[-1])
        ax.set_ylim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Correlation',fontweight='bold')
        # - histogram
        frame +=1
        ax = fig.add_subplot(nbframes,2,frame)
        ax.hist(lo_csave[~np.isnan(lo_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
        ax.set_title('LMR - ORA-20C')
        ax.set_xlim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Counts',fontweight='bold')
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        ypos = ymax-0.15*(ymax-ymin)
        xpos = xmin+0.025*(xmax-xmin)
        ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(lo_csave)),fontsize=11,fontweight='bold')
        
    # LMR <-> EN4
    if en4_available:
        # - time series
        if frame > 1: frame +=1
        ax = fig.add_subplot(nbframes,2,frame)
        ax.plot(cyears,le_csave,lw=2)
        ax.plot([trange[0],trange[-1]],[0,0],'k:')
        ax.set_title('LMR - EN4')
        ax.set_xlim(trange[0],trange[-1])
        ax.set_ylim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Correlation',fontweight='bold')
        # - histogram
        frame +=1
        ax = fig.add_subplot(nbframes,2,frame)
        ax.hist(le_csave[~np.isnan(le_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
        ax.set_title('LMR - EN4')
        ax.set_xlim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Counts',fontweight='bold')
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        ypos = ymax-0.15*(ymax-ymin)
        xpos = xmin+0.025*(xmax-xmin)
        ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(le_csave)),fontsize=11,fontweight='bold')        
    
    
    fig.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.45, right=0.95, top=0.93, wspace=0.5, hspace=0.5)
    fig.suptitle(verif_dict[var][2]+' anomaly correlation',fontweight='bold') 
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_anomaly_correlation_LMR_'+str(trange[0])+'-'+str(trange[1])+'.pdf', bbox_inches='tight', dpi=300, format='pdf')
        plt.close()

        
    # Comparison of reanalyses (for reference)
    fig = plt.figure()

    frame = 1
    nbframes = 0
    if (soda_available and en4_available) : nbframes +=1
    if (soda_available and ora_available) : nbframes +=1
    if (en4_available and ora_available) : nbframes +=1

    # SODA <-> ORA20C
    if soda_available and ora_available:
        # - time series
        if frame > 1: frame +=1
        ax = fig.add_subplot(nbframes,2,frame)
        ax.plot(cyears,so_csave,lw=2)
        ax.plot([trange[0],trange[-1]],[0,0],'k:')
        ax.set_title('SODA - ORA-20C')
        ax.set_xlim(trange[0],trange[-1])
        ax.set_ylim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Correlation',fontweight='bold')
#        ax.set_xlabel('Year CE',fontweight='bold')
        # - histogram
        frame +=1
        ax = fig.add_subplot(nbframes,2,frame)
        ax.hist(so_csave[~np.isnan(so_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
        ax.set_title('SODA - ORA-20C')
        ax.set_xlim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Counts',fontweight='bold')
#        ax.set_xlabel('Correlation',fontweight='bold')
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        ypos = ymax-0.15*(ymax-ymin)
        xpos = xmin+0.025*(xmax-xmin)
        ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(so_csave)),fontsize=11,fontweight='bold')

    # SODA <-> EN4
    if soda_available and en4_available:
        # - time series
        if frame > 1: frame +=1
        ax = fig.add_subplot(nbframes,2,frame)
        ax.plot(cyears,se_csave,lw=2)
        ax.plot([trange[0],trange[-1]],[0,0],'k:')
        ax.set_title('SODA - EN4')
        ax.set_xlim(trange[0],trange[-1])
        ax.set_ylim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Correlation',fontweight='bold')
#        ax.set_xlabel('Year CE',fontweight='bold')
        # - histogram
        frame +=1
        ax = fig.add_subplot(nbframes,2,frame)
        ax.hist(se_csave[~np.isnan(se_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
        ax.set_title('SODA - EN4')
        ax.set_xlim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Counts',fontweight='bold')
#        ax.set_xlabel('Correlation',fontweight='bold')
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        ypos = ymax-0.15*(ymax-ymin)
        xpos = xmin+0.025*(xmax-xmin)
        ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(se_csave)),fontsize=11,fontweight='bold')
    
    # ORA20C <-> EN4
    if en4_available and ora_available:
        # - time series
        if frame > 1: frame +=1
        ax = fig.add_subplot(nbframes,2,frame)
        ax.plot(cyears,oe_csave,lw=2)
        ax.plot([trange[0],trange[-1]],[0,0],'k:')
        ax.set_title('ORA-20C - EN4')
        ax.set_xlim(trange[0],trange[-1])
        ax.set_ylim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Correlation',fontweight='bold')
#        ax.set_xlabel('Year CE',fontweight='bold')
        # - histogram
        frame +=1
        ax = fig.add_subplot(nbframes,2,frame)
        ax.hist(oe_csave[~np.isnan(oe_csave)],bins=bins,histtype='stepfilled',alpha=0.25)
        ax.set_title('ORA-20C - EN4')
        ax.set_xlim(corr_range[0],corr_range[-1])
        ax.set_ylabel('Counts',fontweight='bold')
#        ax.set_xlabel('Correlation',fontweight='bold')
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        ypos = ymax-0.15*(ymax-ymin)
        xpos = xmin+0.025*(xmax-xmin)
        ax.text(xpos,ypos,'Mean = %s' %"{:.2f}".format(np.nanmean(oe_csave)),fontsize=11,fontweight='bold')


    fig.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.45, right=0.95, top=0.93, wspace=0.5, hspace=0.5)
    fig.suptitle(verif_dict[var][2]+' anomaly correlation',fontweight='bold') 
    if fsave:
        print('saving to .png')
        plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_anomaly_correlation_Reanalyses_'+str(trange[0])+'-'+str(trange[1])+'.png')
        plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_anomaly_correlation_Reanalyses_'+str(trange[0])+'-'+str(trange[1])+'.pdf', bbox_inches='tight', dpi=300, format='pdf')
        plt.close()


        
    
    # ---------------------------------
    # BEGIN bias, r and CE calculations
    #

    # correlation and CE at each (lat,lon) point

    r_ls = np.zeros([nlat,nlon])
    ce_ls = np.zeros([nlat,nlon])
    r_le = np.zeros([nlat,nlon])
    ce_le = np.zeros([nlat,nlon])
    r_lo = np.zeros([nlat,nlon])
    ce_lo = np.zeros([nlat,nlon])
    r_so = np.zeros([nlat,nlon])
    ce_so = np.zeros([nlat,nlon])
    r_se = np.zeros([nlat,nlon])
    ce_se = np.zeros([nlat,nlon])
    r_oe = np.zeros([nlat,nlon])
    ce_oe = np.zeros([nlat,nlon])
    
    
    # bias & CE
    if soda_available:
        ls_err = lmr_allyears - soda_allyears
        ce_ls = coefficient_efficiency(soda_allyears,lmr_allyears,valid=valid_frac)

    if en4_available:
        le_err = lmr_allyears - en4_allyears
        ce_le = coefficient_efficiency(en4_allyears,lmr_allyears,valid=valid_frac)
        
    if ora_available:
        lo_err = lmr_allyears - ora20c_allyears
        ce_lo = coefficient_efficiency(ora20c_allyears,lmr_allyears,valid=valid_frac)

    if soda_available and en4_available:
        se_err = soda_allyears - en4_allyears
        ce_se = coefficient_efficiency(en4_allyears,soda_allyears,valid=valid_frac)
        
    if soda_available and ora_available:
        so_err = soda_allyears - ora20c_allyears
        ce_so = coefficient_efficiency(ora20c_allyears,soda_allyears,valid=valid_frac)

    if en4_available and ora_available:
        oe_err = ora20c_allyears - en4_allyears
        ce_oe = coefficient_efficiency(en4_allyears,ora20c_allyears,valid=valid_frac)

    
    # Correlation
    for la in range(nlat):
        for lo in range(nlon):

            # LMR-SODA
            if soda_available:
                indok = np.isfinite(soda_allyears[:,la,lo])
                nbok = np.sum(indok)
                nball = lmr_allyears[:,la,lo].shape[0]
                ratio = float(nbok)/float(nball)
                if ratio > valid_frac:
                    indok2 = np.where(np.isfinite(lmr_allyears[:,la,lo]) & np.isfinite(soda_allyears[:,la,lo]))
                    r_ls[la,lo] = np.corrcoef(lmr_allyears[indok2,la,lo],soda_allyears[indok2,la,lo])[0,1]
                else:
                    r_ls[la,lo] = np.nan

                    
            # LMR-ORA20C
            if ora_available:
                indok = np.isfinite(ora20c_allyears[:,la,lo])
                nbok = np.sum(indok)
                nball = lmr_allyears[:,la,lo].shape[0]
                ratio = float(nbok)/float(nball)
                if ratio > valid_frac:
                    indok2 = np.where(np.isfinite(lmr_allyears[:,la,lo]) & np.isfinite(ora20c_allyears[:,la,lo]))
                    r_lo[la,lo] = np.corrcoef(lmr_allyears[indok2,la,lo],ora20c_allyears[indok2,la,lo])[0,1]
                else:
                    r_lo[la,lo] = np.nan

            
            # LMR-EN4
            if en4_available:
                indok = np.isfinite(en4_allyears[:,la,lo])
                nbok = np.sum(indok)
                nball = lmr_allyears[:,la,lo].shape[0]
                ratio = float(nbok)/float(nball)
                if ratio > valid_frac:
                    indok2 = np.where(np.isfinite(lmr_allyears[:,la,lo]) & np.isfinite(en4_allyears[:,la,lo]))
                    r_le[la,lo] = np.corrcoef(lmr_allyears[indok2,la,lo],en4_allyears[indok2,la,lo])[0,1]
                else:
                    r_le[la,lo] = np.nan

            
            # SODA-ORA20C
            if soda_available and ora_available:
                indok = np.isfinite(ora20c_allyears[:,la,lo])
                nbok = np.sum(indok)
                nball = soda_allyears[:,la,lo].shape[0]
                ratio = float(nbok)/float(nball)
                if ratio > valid_frac:
                    indok2 = np.where(np.isfinite(soda_allyears[:,la,lo]) & np.isfinite(ora20c_allyears[:,la,lo]))
                    r_so[la,lo] = np.corrcoef(soda_allyears[indok2,la,lo],ora20c_allyears[indok2,la,lo])[0,1]
                else:
                    r_so[la,lo] = np.nan
                    
            # SODA-EN4
            if soda_available and en4_available:
                indok = np.isfinite(en4_allyears[:,la,lo])
                nbok = np.sum(indok)
                nball = soda_allyears[:,la,lo].shape[0]
                ratio = float(nbok)/float(nball)
                if ratio > valid_frac:
                    indok2 = np.where(np.isfinite(soda_allyears[:,la,lo]) & np.isfinite(en4_allyears[:,la,lo]))
                    r_se[la,lo] = np.corrcoef(soda_allyears[indok2,la,lo],en4_allyears[indok2,la,lo])[0,1]
                else:
                    r_se[la,lo] = np.nan

            # ORA20C-EN4
            if en4_available and ora_available:
                indok = np.isfinite(en4_allyears[:,la,lo])
                nbok = np.sum(indok)
                nball = ora20c_allyears[:,la,lo].shape[0]
                ratio = float(nbok)/float(nball)
                if ratio > valid_frac:
                    indok2 = np.where(np.isfinite(ora20c_allyears[:,la,lo]) & np.isfinite(en4_allyears[:,la,lo]))
                    r_oe[la,lo] = np.corrcoef(ora20c_allyears[indok2,la,lo],en4_allyears[indok2,la,lo])[0,1]
                else:
                    r_oe[la,lo] = np.nan
                    

    # median & spatial mean (area weighted)
    lat_trunc = np.squeeze(lat2[:,0])
    indlat = np.where((lat_trunc[:] > -60.0) & (lat_trunc[:] < 60.0))

    if soda_available:
        ls_rmedian = str(float('%.2g' % np.nanmedian(np.nanmedian(r_ls)) ))
        print('lmr-soda all-grid median r     : %s' % str(ls_rmedian))
        ls_rmedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(r_ls[indlat,:])) ))
        print('lmr-soda 60S-60N median r      : %s' % str(ls_rmedian60))
        ls_cemedian = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_ls)) ))
        print('lmr-soda all-grid median ce    : %s' % str(ls_cemedian))
        ls_cemedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_ls[indlat,:])) ))
        print('lmr-soda 60S-60N median ce     : %s' % str(ls_cemedian60))

        # LMR-SODA
        [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_ls,veclat)
        [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_ls,veclat)
        ls_rmean_global  = str(float('%.2f' %rmean_global[0]))
        ls_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
        ls_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
        ls_cemean_global = str(float('%.2f' %cemean_global[0]))
        ls_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
        ls_cemean_sh     = str(float('%.2f' %cemean_sh[0]))

        
    if ora_available:
        lo_rmedian = str(float('%.2g' % np.nanmedian(np.nanmedian(r_lo)) ))
        print('lmr-ora20c all-grid median r   : %s' % str(lo_rmedian))
        lo_rmedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(r_lo[indlat,:])) ))
        print('lmr-ora20c 60S-60N median r    : %s' % str(lo_rmedian60))
        lo_cemedian = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_lo)) ))
        print('lmr-ora20c all-grid median ce  : %s' % str(lo_cemedian))
        lo_cemedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_lo[indlat,:])) ))
        print('lmr-ora20c 60S-60N median ce   : %s' % str(lo_cemedian60))

        # LMR-ORA20C
        [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_lo,veclat)
        [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_lo,veclat)
        lo_rmean_global  = str(float('%.2f' %rmean_global[0]))
        lo_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
        lo_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
        lo_cemean_global = str(float('%.2f' %cemean_global[0]))
        lo_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
        lo_cemean_sh     = str(float('%.2f' %cemean_sh[0]))

    
    if en4_available:
        le_rmedian = str(float('%.2g' % np.nanmedian(np.nanmedian(r_le)) ))
        print('lmr-en4 all-grid median r      : %s' % str(le_rmedian))
        le_rmedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(r_le[indlat,:])) ))
        print('lmr-en4 60S-60N median r       : %s' % str(le_rmedian60))
        le_cemedian = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_le)) ))
        print('lmr-en4 all-grid median ce     : %s' % str(le_cemedian))
        le_cemedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_le[indlat,:])) ))
        print('lmr-en4 60S-60N median ce      : %s' % str(le_cemedian60))

        # LMR-EN4
        [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_le,veclat)
        [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_le,veclat)
        le_rmean_global  = str(float('%.2f' %rmean_global[0]))
        le_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
        le_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
        le_cemean_global = str(float('%.2f' %cemean_global[0]))
        le_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
        le_cemean_sh     = str(float('%.2f' %cemean_sh[0]))


    if soda_available and ora_available:
        so_rmedian = str(float('%.2g' % np.nanmedian(np.nanmedian(r_so)) ))
        print('soda-ora20c all-grid median r  : %s' % str(so_rmedian))
        so_rmedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(r_so[indlat,:])) ))
        print('soda-ora20c 60S-60N median r   : %s' % str(so_rmedian60))
        so_cemedian = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_so)) ))
        print('soda-ora20c all-grid median ce : %s' % str(so_cemedian))
        so_cemedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_so[indlat,:])) ))
        print('soda-ora20c 60S-60N median ce  : %s' % str(so_cemedian60))

        # SODA-ORA20C
        [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_so,veclat)
        [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_so,veclat)
        so_rmean_global  = str(float('%.2f' %rmean_global[0]))
        so_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
        so_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
        so_cemean_global = str(float('%.2f' %cemean_global[0]))
        so_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
        so_cemean_sh     = str(float('%.2f' %cemean_sh[0]))

    if soda_available and en4_available:
        se_rmedian = str(float('%.2g' % np.nanmedian(np.nanmedian(r_se)) ))
        print('soda-en4 all-grid median r     : %s' % str(se_rmedian))
        se_rmedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(r_se[indlat,:])) ))
        print('soda-en4 60S-60N median r      : %s' % str(se_rmedian60))
        se_cemedian = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_se)) ))
        print('soda-en4 all-grid median ce    : %s' % str(se_cemedian))
        se_cemedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_se[indlat,:])) ))
        print('soda-en4 60S-60N median ce     : %s' % str(se_cemedian60))

        # SODA-EN4
        [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_se,veclat)
        [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_se,veclat)
        se_rmean_global  = str(float('%.2f' %rmean_global[0]))
        se_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
        se_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
        se_cemean_global = str(float('%.2f' %cemean_global[0]))
        se_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
        se_cemean_sh     = str(float('%.2f' %cemean_sh[0]))

    if en4_available and ora_available:
        oe_rmedian = str(float('%.2g' % np.nanmedian(np.nanmedian(r_oe)) ))
        print('ora20c-en4 all-grid median r   : %s' % str(oe_rmedian))
        oe_rmedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(r_oe[indlat,:])) ))
        print('ora20c-en4 60S-60N median r    : %s' % str(oe_rmedian60))
        oe_cemedian = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_oe)) ))
        print('ora20c-en4 all-grid median ce  : %s' % str(oe_cemedian))
        oe_cemedian60 = str(float('%.2g' % np.nanmedian(np.nanmedian(ce_oe[indlat,:])) ))
        print('ora20c-en4 60S-60N median ce  : %s' % str(oe_cemedian60))

        # ORA20C-EN4
        [rmean_global,rmean_nh,rmean_sh]    = global_hemispheric_means(r_oe,veclat)
        [cemean_global,cemean_nh,cemean_sh] = global_hemispheric_means(ce_oe,veclat)
        oe_rmean_global  = str(float('%.2f' %rmean_global[0]))
        oe_rmean_nh      = str(float('%.2f' %rmean_nh[0]))
        oe_rmean_sh      = str(float('%.2f' %rmean_sh[0]))
        oe_cemean_global = str(float('%.2f' %cemean_global[0]))
        oe_cemean_nh     = str(float('%.2f' %cemean_nh[0]))
        oe_cemean_sh     = str(float('%.2f' %cemean_sh[0]))

        

    # -----------------------
    # zonal mean verification
    # -----------------------

    #  LMR-SODA
    if soda_available:
        r_ls_zm = np.zeros([nlat])
        ce_ls_zm = np.zeros([nlat])
        ls_err_zm = lmr_zm - soda_zm

    # LMR-ORA20C
    if ora_available:
        r_lo_zm = np.zeros([nlat])
        ce_lo_zm = np.zeros([nlat])
        lo_err_zm = lmr_zm - ora20c_zm
    
    # LMR-EN4
    if en4_available:
        r_le_zm = np.zeros([nlat])
        ce_le_zm = np.zeros([nlat])
        le_err_zm = lmr_zm - en4_zm
        
    
    for la in range(nlat):

        # LMR-SODA
        if soda_available:
            ce_ls_zm[la] = coefficient_efficiency(soda_zm[:,la],lmr_zm[:,la],valid=valid_frac)
            indok = np.isfinite(soda_zm[:,la])
            nbok = np.sum(indok)
            nball = len(cyears)
            ratio = float(nbok)/float(nball)

            if ratio > valid_frac:
                indok2 = np.where(np.isfinite(lmr_zm[:,la]) & np.isfinite(soda_zm[:,la]))
                r_ls_zm[la] = np.corrcoef(lmr_zm[indok2,la],soda_zm[indok2,la])[0,1]
            else:
                r_ls_zm[la]  = np.nan

        # LMR-ORA20C
        if ora_available:
            ce_lo_zm[la] = coefficient_efficiency(ora20c_zm[:,la],lmr_zm[:,la],valid=valid_frac)    
            indok = np.isfinite(ora20c_zm[:,la])
            nbok = np.sum(indok)
            nball = len(cyears)
            ratio = float(nbok)/float(nball)
            if ratio > valid_frac:
                indok2 = np.where(np.isfinite(lmr_zm[:,la]) & np.isfinite(ora20c_zm[:,la]))
                r_lo_zm[la] = np.corrcoef(lmr_zm[indok2,la],ora20c_zm[indok2,la])[0,1]
            else:
                r_lo_zm[la]  = np.nan
            
        # LMR-EN4
        if en4_available:
            ce_le_zm[la] = coefficient_efficiency(en4_zm[:,la],lmr_zm[:,la],valid=valid_frac)
            indok = np.isfinite(en4_zm[:,la])
            nbok = np.sum(indok)
            nball = len(cyears)
            ratio = float(nbok)/float(nball)

            if ratio > valid_frac:
                indok2 = np.where(np.isfinite(lmr_zm[:,la]) & np.isfinite(en4_zm[:,la]))
                r_le_zm[la] = np.corrcoef(lmr_zm[indok2,la],en4_zm[indok2,la])[0,1]
            else:
                r_le_zm[la]  = np.nan
                
    
    #
    # END bias, r and CE calculations
    # -------------------------------

    major_ticks = np.arange(-90, 91, 30)
    fig = plt.figure()

    # correlation
    ax = fig.add_subplot(1,2,1)
    sodaleg = None
    oraleg = None
    en4leg = None
    if soda_available:
        sodaleg, = ax.plot(r_ls_zm,veclat,'k-',linestyle=':',lw=2,label='SODA')        
    if en4_available:
        en4leg, = ax.plot(r_le_zm,veclat,'k-',linestyle='--',lw=2,label='EN4')        
    if ora_available:
        oraleg, = ax.plot(r_lo_zm,veclat,'k-',linestyle='-',lw=2,label='ORA-20C')
    ax.plot([0,0],[-90,90],'k:')
    ax.set_yticks(major_ticks)                                                       
    plt.ylim([-90,90])
    plt.xlim([-1,1])
    plt.ylabel('Latitude',fontweight='bold')
    plt.xlabel('Correlation',fontweight='bold')

    if sodaleg and en4leg and oraleg:
        ax.legend(handles=[sodaleg,en4leg,oraleg],handlelength=3.0,ncol=1,fontsize=12,loc='lower left',frameon=False)
    elif sodaleg and en4leg and not oraleg:
        ax.legend(handles=[sodaleg,en4leg],handlelength=3.0,ncol=1,fontsize=12,loc='lower left',frameon=False)
    elif sodaleg and oraleg and not en4leg:
        ax.legend(handles=[sodaleg,oraleg],handlelength=3.0,ncol=1,fontsize=12,loc='lower left',frameon=False)
    elif en4leg and oraleg and not sodaleg:
        ax.legend(handles=[en4leg,oraleg],handlelength=3.0,ncol=1,fontsize=12,loc='lower left',frameon=False)
    elif sodaleg and not en4leg and not oraleg:
        ax.legend(handles=[sodaleg],handlelength=3.0,ncol=1,fontsize=12,loc='lower left',frameon=False)
    elif en4leg and not sodaleg and not sodaleg:
        ax.legend(handles=[en4leg],handlelength=3.0,ncol=1,fontsize=12,loc='lower left',frameon=False)
    elif oraleg and not en4leg and not sodaleg:
        ax.legend(handles=[oraleg],handlelength=3.0,ncol=1,fontsize=12,loc='lower left',frameon=False)

        
    # CE
    ax = fig.add_subplot(1,2,2)    
    if soda_available:
        ax.plot(ce_ls_zm,veclat,'k-',linestyle=':',lw=2)
    if en4_available:
        ax.plot(ce_le_zm,veclat,'k-',linestyle='--',lw=2)
    if ora_available:
        ax.plot(ce_lo_zm,veclat,'k-',linestyle='-',lw=2)
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


        
    # ======================
    # spatial r and ce plots
    # ======================

    cbarfmt = '%4.1f'
    nticks = 4 # number of ticks on the colorbar
    if iplot:

        nbframes = 0
        if soda_available: nbframes +=1
        if en4_available: nbframes +=1
        if ora_available: nbframes +=1
        frame = 1

        fig = plt.figure()

        if soda_available:
            ax = fig.add_subplot(nbframes,2,frame)    
            LMR_plotter(r_ls,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('LMR - SODA '+verif_dict[var][1]+' r \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(ls_rmean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])
            frame +=1
            ax = fig.add_subplot(nbframes,2,frame)    
            LMR_plotter(ce_ls,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('LMR - SODA '+verif_dict[var][1]+' CE \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(ls_cemean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])

        if ora_available:
            if frame > 1: frame +=1
            ax = fig.add_subplot(nbframes,2,frame)    
            LMR_plotter(r_lo,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('LMR - ORA-20C '+verif_dict[var][1]+' r \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lo_rmean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])
            frame +=1
            ax = fig.add_subplot(nbframes,2,frame)    
            LMR_plotter(ce_lo,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('LMR - ORA-20C '+verif_dict[var][1]+' CE \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(lo_cemean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])

        if en4_available:
            if frame > 1: frame +=1
            ax = fig.add_subplot(nbframes,2,frame)    
            LMR_plotter(r_le,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('LMR - EN4 '+verif_dict[var][1]+' r \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(le_rmean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])
            frame +=1
            ax = fig.add_subplot(nbframes,2,frame)    
            LMR_plotter(ce_le,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('LMR - EN4 '+verif_dict[var][1]+' CE \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(le_cemean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])
            

        fig.tight_layout()
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.93, wspace=0.15, hspace=0.5)
        if fsave:
            print('saving to .png')
            plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_r_ce_LMR_'+str(trange[0])+'-'+str(trange[1])+'.png')
            plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_r_ce_LMR_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')
            plt.close()




        nbframes = 0
        if soda_available and en4_available: nbframes +=1
        if soda_available and ora_available: nbframes +=1
        if en4_available and ora_available: nbframes +=1
        frame = 1

        fig = plt.figure()

        if soda_available and ora_available:
            if frame > 1: frame +=1
            ax = fig.add_subplot(nbframes,2,frame)    
            LMR_plotter(r_so,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('SODA - ORA-20C '+verif_dict[var][1]+' r \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(so_rmean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])
            frame +=1
            ax = fig.add_subplot(nbframes,2,frame)
            LMR_plotter(ce_so,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('SODA - ORA-20C '+verif_dict[var][1]+' CE \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(so_cemean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])

        if soda_available and en4_available:
            if frame > 1: frame +=1
            ax = fig.add_subplot(nbframes,2,frame)    
            LMR_plotter(r_se,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('SODA - EN4 '+verif_dict[var][1]+' r \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(se_rmean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])
            frame +=1
            ax = fig.add_subplot(nbframes,2,frame)
            LMR_plotter(ce_se,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('SODA - EN4 '+verif_dict[var][1]+' CE \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(se_cemean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])
        
        if en4_available and ora_available:
            if frame > 1: frame +=1
            ax = fig.add_subplot(nbframes,2,frame)    
            LMR_plotter(r_oe,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='neither',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('ORA-20C - EN4 '+verif_dict[var][1]+' r \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(oe_rmean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])
            frame +=1
            ax = fig.add_subplot(nbframes,2,frame)
            LMR_plotter(ce_oe,lat2,lon2,'bwr',nlevs,vmin=-1,vmax=1,extend='min',cbarfmt=cbarfmt,nticks=nticks,backg='lightgrey')
            plt.title('ORA-20C - EN4 '+verif_dict[var][1]+' CE \n'+str(cyears[0])+'-'+str(cyears[-1]) + ' mean='+str(oe_cemean_global))
            plt.clim(-1,1)
            ax.title.set_position([.5, 1.05])

            
        fig.tight_layout()
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.93, wspace=0.15, hspace=0.5)
        if fsave:
            print('saving to .png')
            plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_r_ce_Reanalyses_'+str(trange[0])+'-'+str(trange[1])+'.png')
            plt.savefig(nexp+'_verify_grid_'+verif_dict[var][1]+'_r_ce_Reanalyses_'+str(trange[0])+'-'+str(trange[1])+'.pdf',bbox_inches='tight', dpi=300, format='pdf')
            plt.close()
