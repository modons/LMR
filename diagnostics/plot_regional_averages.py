<<<<<<< HEAD
import sys
sys.path.append("..")

import glob, os
import LMR_utils
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

# save figures here
#figdir = '/Users/hakim/lmr/lmr_plots/'
figdir = '/home/disk/ice4/hakim/lmr/lmr_plots/'

#datadir_output = '/Users/hakim/data/LMR/archive'
datadir_output = '/home/disk/kalman3/rtardif/LMR/output'

#nexp = 'dadt_test_xbone'
#nexp = 'pages2_loc25000_pages2k2_seasonal_TorP_nens200'
#nexp = 'test'
nexp = 'p2_ccsm4LM_n200_bilin_GISTEMPGPCCseasonPSM_PAGES2kv2_pf0.75_loc25k'

#var = 'tas_sfc_Adec'
var = 'tas_sfc_Amon'

# loop over directories
=======
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
figdir = '/home/disk/ice4/hakim/lmr/lmr_plots/'
#figdir = '/home/disk/kalman3/rtardif/LMR/output/'

# LMR input dayta located here:
# ----------------------------
#datadir_output = '/Users/hakim/data/LMR/archive'
datadir_output = '/home/disk/kalman3/rtardif/LMR/output'

# Name of reconstruction experiment
# ---------------------------------
#nexp = 'test'
#nexp = 'pages2_loc25000_pages2k2_seasonal_TorP_nens200'
nexp = 'p2_ccsm4LM_n200_bilin_GISTEMPGPCCseasonPSM_PAGES2kv2_pf0.75_loc25k'


# Variable to plot
# ----------------
var = 'tas_sfc_Amon'
#var = 'tas_sfc_Adec'


# MC iterations to consider
>>>>>>> refs/remotes/robertsgit/DADT
MCset = None
#MCset = (0,0)
#MCset = (0,10)

<<<<<<< HEAD
# region labels for plotting (push this to LMR_utils.py?)
labs = [
'Arctic: north of 60N ',
'Europe: 35-70N, 10W-40E',
'Asia: 23-55N',
'North America (trees):30-55N,75-130W',
'South America: 20S-65S, 30W-80W',
'Australasia: 0-50S, 110E-180E', 
'Antarctica: south of 60S'
]

labs_short = [
'Arctic',
'Europe',
'Asia',
'N. America',
'S. America',
'Australasia',
'Antarctica',
'NH',
'SH',
'global mean'
]

#----------------------------------------------------------------------------
# end of user-defined parameters
#----------------------------------------------------------------------------
=======

# ---- End section of user-defined parameters ----
# ------------------------------------------------


# region labels for plotting (push this to LMR_utils.py?)
#  Important: These need to match the hard-coded definitions
#  in PAGES2K_regional_means() found in LMR_utils.py

# more compact definition of regions
# than original using multiple lists
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

>>>>>>> refs/remotes/robertsgit/DADT

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

<<<<<<< HEAD
#print np.shape(xam)
#print np.shape(lat)
#print np.shape(lon)
#print nyrs
#print np.max(xam)

[tmp_gm,_,_] = LMR_utils.global_hemispheric_means(xam,lat[:,1])
#print tmp_gm
=======
[tmp_gm,_,_] = LMR_utils.global_hemispheric_means(xam,lat[:,1])
>>>>>>> refs/remotes/robertsgit/DADT
rm = LMR_utils.PAGES2K_regional_means(xam,lat[:,1],lon[1,:])
nregions = np.shape(rm)[0]
print 'nregions='+ str(nregions)

fontP = FontProperties()
fontP.set_size('small')
for k in range(nregions):
    plt.plot(rm[k,:],label=labs[k],alpha=0.5)

<<<<<<< HEAD
lgd = plt.legend(bbox_to_anchor=(0.5,-.1),loc=9,ncol=2,borderaxespad=0)
=======
lgd = plt.legend(bbox_to_anchor=(0.5,-.1),loc=9,ncol=2,borderaxespad=0,fontsize=10)
>>>>>>> refs/remotes/robertsgit/DADT
art = []
art.append(lgd)
fname = figdir+'regions_all_'+nexp+'.png'
plt.savefig(fname,additional_artists=art,bbox_inches='tight')

plt.figure()
matplotlib.rcParams.update({'font.size':8})
for k in range(nregions):
    plt.subplot(3,3,k+1)
    plt.plot(rm[k,:],label=labs_short[k])
<<<<<<< HEAD
    plt.title(labs_short[k])
=======
    xmin,xmax,ymin,ymax = plt.axis()
    plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
    plt.xlabel('Year CE')
    plt.title(labs_short[k],fontweight='bold')
>>>>>>> refs/remotes/robertsgit/DADT
    
plt.tight_layout()
fname = figdir+'regions_'+nexp+'.png'
print 'saving...'
print fname
plt.savefig(fname,additional_artists=art,bbox_inches='tight')
<<<<<<< HEAD
plt.show()

#----------------------------------------------------------------------------
# plot the iteration/ensemble mean for all regions on different subplots of a single figure, along with iteration uncertainty...
#   --change to ensemble 5/95 when full-field ensemble writing ia available!
=======
#plt.show()

#----------------------------------------------------------------------------
# plot the iteration/ensemble mean for all regions on different subplots of a
# single figure, along with iteration uncertainty...
# -- change to ensemble 5/95 when full-field ensemble writing ia available!
>>>>>>> refs/remotes/robertsgit/DADT
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
<<<<<<< HEAD
    plt.fill_between(years,rms_5[k,:],rms_95[k,:],facecolor='gray',alpha = 0.75,linewidth=0.)
    plt.plot(years,rms_avg[k,:],color='k',label=labs_short[k],lw=0.5)
    #plt.plot(years,rms_5[k,:],color='gray',alpha=0.5,lw=0.5)
    #plt.plot(years,rms_95[k,:],color='gray',alpha=0.5,lw=0.5)
    plt.title(labs_short[k])
=======
    plt.fill_between(years,rms_5[k,:],rms_95[k,:],facecolor='gray',alpha = 0.5,linewidth=0.)
    plt.plot(years,rms_avg[k,:],color='b',label=labs_short[k],lw=0.5)
    #plt.plot(years,rms_5[k,:],color='gray',alpha=0.5,lw=0.5)
    #plt.plot(years,rms_95[k,:],color='gray',alpha=0.5,lw=0.5)
    xmin,xmax,ymin,ymax = plt.axis()
    plt.plot([xmin,xmax],[0,0],'--',color='red',linewidth=1)
    plt.xlabel('Year CE')
    plt.title(labs_short[k],fontweight='bold')
>>>>>>> refs/remotes/robertsgit/DADT
    
plt.tight_layout()
fname = figdir+'regions_'+nexp+'_5_95.png'
print 'saving...'
print fname
plt.savefig(fname,additional_artists=art,bbox_inches='tight')
<<<<<<< HEAD


#plt.show()


=======
#plt.show()
>>>>>>> refs/remotes/robertsgit/DADT
