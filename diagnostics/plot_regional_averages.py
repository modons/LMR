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
MCset = None
#MCset = (0,0)
#MCset = (0,10)

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

#print np.shape(xam)
#print np.shape(lat)
#print np.shape(lon)
#print nyrs
#print np.max(xam)

[tmp_gm,_,_] = LMR_utils.global_hemispheric_means(xam,lat[:,1])
#print tmp_gm
rm = LMR_utils.PAGES2K_regional_means(xam,lat[:,1],lon[1,:])
nregions = np.shape(rm)[0]
print 'nregions='+ str(nregions)

fontP = FontProperties()
fontP.set_size('small')
for k in range(nregions):
    plt.plot(rm[k,:],label=labs[k],alpha=0.5)

lgd = plt.legend(bbox_to_anchor=(0.5,-.1),loc=9,ncol=2,borderaxespad=0)
art = []
art.append(lgd)
fname = figdir+'regions_all_'+nexp+'.png'
plt.savefig(fname,additional_artists=art,bbox_inches='tight')

plt.figure()
matplotlib.rcParams.update({'font.size':8})
for k in range(nregions):
    plt.subplot(3,3,k+1)
    plt.plot(rm[k,:],label=labs_short[k])
    plt.title(labs_short[k])
    
plt.tight_layout()
fname = figdir+'regions_'+nexp+'.png'
print 'saving...'
print fname
plt.savefig(fname,additional_artists=art,bbox_inches='tight')
plt.show()

#----------------------------------------------------------------------------
# plot the iteration/ensemble mean for all regions on different subplots of a single figure, along with iteration uncertainty...
#   --change to ensemble 5/95 when full-field ensemble writing ia available!
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
    plt.fill_between(years,rms_5[k,:],rms_95[k,:],facecolor='gray',alpha = 0.75,linewidth=0.)
    plt.plot(years,rms_avg[k,:],color='k',label=labs_short[k],lw=0.5)
    #plt.plot(years,rms_5[k,:],color='gray',alpha=0.5,lw=0.5)
    #plt.plot(years,rms_95[k,:],color='gray',alpha=0.5,lw=0.5)
    plt.title(labs_short[k])
    
plt.tight_layout()
fname = figdir+'regions_'+nexp+'_5_95.png'
print 'saving...'
print fname
plt.savefig(fname,additional_artists=art,bbox_inches='tight')


#plt.show()


