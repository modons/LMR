"""
Module: plot_regional_averages.py

Purpose: Calculate the latitude-weighted spatial average of a specified 2D field
         over specific (PAGES2k) regions and plot the resulting time series. 

Originators: Greg Hakim  | U. of Washington
                         | July 2017

Revisions: 

"""

import sys
import glob, os
import numpy as np
import matplotlib

from collections import OrderedDict

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
matplotlib.use('Agg')

# LMR-specific imports
sys.path.append("../")
import LMR_utils


# ------------------------------------------------
# --- Begin section of user-defined parameters ---


# save figures here:
# -----------------
#figdir = '/Users/hakim/lmr/lmr_plots/'
#figdir = '/home/disk/ice4/hakim/lmr/lmr_plots/'
#figdir = '/home/disk/kalman3/rtardif/LMR/output/'
figdir = './'

# LMR input dayta located here:
# ----------------------------
datadir_output = '/Users/hakim/data/LMR/archive'
#datadir_output = '/home/disk/kalman3/rtardif/LMR/output'

# Name of reconstruction experiment
# ---------------------------------
nexp = 'test'
#nexp = 'pages2_loc25000_pages2k2_seasonal_TorP_nens200'
#nexp = 'p2_ccsm4LM_n200_bilin_GISTEMPGPCCseasonPSM_PAGES2kv2_pf0.75_loc25k'


# Variable to plot
# ----------------
var = 'tas_sfc_Amon'
#var = 'tas_sfc_Adec'

# MC iterations to consider
MCset = None
#MCset = (0,0)
#MCset = (0,10)


# ---- End section of user-defined parameters ----
# ------------------------------------------------


# region labels for plotting (push this to LMR_utils.py?)
#  Important: These need to match the hard-coded definitions
#  in PAGES2K_regional_means() found in LMR_utils.py

# more compact definition of regions than original using multiple lists
region_labels = OrderedDict([
    ('Arctic', 'Arctic: north of 60N'),
    ('Europe','Europe: 35-70N, 10W-40E'),
    ('Asia','Asia: 23-55N'),
    ('N. America','North America (trees):30-55N,75-130W'),
    ('S. America','South America: 20S-65S, 30W-80W'),
    ('Australasia','Australasia: 0-50S, 110E-180E'),
    ('Antarctica','Antarctica: south of 60S'),
    ('NH', 'N. Hemisphere'),
    ('SH', 'S. Hemisphere'),
    ('global mean', 'Global')
    ])


labs_short = region_labels.keys()
labs = [region_labels[item] for item in labs_short]


#----------------------------------------------------------------------------
# plot the ensemble mean for a single realization for all regions on a single figure...
#----------------------------------------------------------------------------

workdir = datadir_output + '/' + nexp
dir = '/r0/'
ensfiln = workdir + '/' + dir + '/ensemble_mean_'+var+'.npz'
npzfile = np.load(ensfiln)
print  npzfile.files
xam = npzfile['xam']
print('shape of xam: %s' % str(np.shape(xam)))

lat = npzfile['lat']
lon = npzfile['lon']
nlat = npzfile['nlat']
nlon = npzfile['nlon']
lat2 = np.reshape(lat,(nlat,nlon))
lon2 = np.reshape(lon,(nlat,nlon))
years = npzfile['years']
years = years.astype(float)
nyrs =  len(years)

print 'min lat:' + str(np.min(lat))
print 'max lat:' + str(np.max(lat))
print 'min lon:' + str(np.min(lon))
print 'max lon:' + str(np.max(lon))

[tmp_gm,_,_] = LMR_utils.global_hemispheric_means(xam,lat[:,1])
rm = LMR_utils.PAGES2K_regional_means(xam,lat[:,1],lon[1,:])
nregions = np.shape(rm)[0]
print 'nregions='+ str(nregions)

fontP = FontProperties()
fontP.set_size('small')
for k in range(nregions):
    plt.plot(years,rm[k,:],label=labs[k],alpha=0.5)

lgd = plt.legend(bbox_to_anchor=(0.5,-.1),loc=9,ncol=2,borderaxespad=0,fontsize=10)
art = []
art.append(lgd)
fname = figdir+'regions_all_'+nexp+'.png'
plt.savefig(fname,additional_artists=art,bbox_inches='tight')

plt.figure()
matplotlib.rcParams.update({'font.size':8})
for k in range(nregions):
    plt.subplot(3,3,k+1)
    plt.plot(years,rm[k,:],label=labs_short[k])
    xmin,xmax,ymin,ymax = plt.axis()
    plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
    plt.xlabel('Year CE')
    plt.title(labs_short[k],fontweight='bold')
    
plt.tight_layout()
fname = figdir+'regions_'+nexp+'.png'
print 'saving...'
print fname
plt.savefig(fname,additional_artists=art,bbox_inches='tight')
#plt.show()

#----------------------------------------------------------------------------
# plot the iteration/ensemble mean for all regions on different subplots of a
# single figure, along with iteration uncertainty...
# -- change to ensemble 5/95 when full-field ensemble writing ia available!
#----------------------------------------------------------------------------

# get a listing of the iteration directories, and combine with MCset
dirs = glob.glob(workdir+"/r*")
if MCset:
    dirset = dirs[MCset[0]:MCset[1]+1]
else:
    dirset = dirs
niters = len(dirset)

print('--------------------------------------------------')
print('niters = %s' % str(niters))
print('--------------------------------------------------')

kk = -1
first = True
for dir in dirset:
    kk = kk + 1
    ensfiln = dir + '/ensemble_mean_'+var+'.npz'
    print ensfiln
    npzfile = np.load(ensfiln)
    xam = npzfile['xam']
    lat = npzfile['lat']
    lon = npzfile['lon']
    nlat = npzfile['nlat']
    nlon = npzfile['nlon']
    nyrs =  len(years)
    if first:
        first = False
        rmsave = np.zeros([niters,nregions+3,nyrs])
        
    [tmp_gm,tmp_nh,tmp_sh] = LMR_utils.global_hemispheric_means(xam,lat[:,1])
    rm = LMR_utils.PAGES2K_regional_means(xam,lat[:,1],lon[1,:])
    rmsave[kk,0:nregions] = rm
    rmsave[kk,nregions] = tmp_nh
    rmsave[kk,nregions+1] = tmp_sh
    rmsave[kk,nregions+2] = tmp_gm
    
print np.shape(rmsave)
rms_avg = np.mean(rmsave,0)
rms_5 = np.percentile(rmsave,5,axis=0)
rms_95 = np.percentile(rmsave,95,axis=0)
print np.shape(rms_5)
print np.min(rms_5)
print np.max(rms_5)
print np.min(rms_95)
print np.max(rms_95)

# new plot: 7 regions + global mean + NH + SH
plt.figure()
for k in range(nregions+2):
    plt.subplot(3,3,k+1)
    plt.fill_between(years,rms_5[k,:],rms_95[k,:],facecolor='gray',alpha = 0.5,linewidth=0.)
    plt.plot(years,rms_avg[k,:],color='b',label=labs_short[k],lw=0.5)
    #plt.plot(years,rms_5[k,:],color='gray',alpha=0.5,lw=0.5)
    #plt.plot(years,rms_95[k,:],color='gray',alpha=0.5,lw=0.5)
    xmin,xmax,ymin,ymax = plt.axis()
    plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
    plt.xlabel('Year CE')
    plt.title(labs_short[k],fontweight='bold')
    
plt.tight_layout()
fname = figdir+'regions_'+nexp+'_5_95.png'
print 'saving...'
print fname
plt.savefig(fname,additional_artists=art,bbox_inches='tight')
#plt.show()
