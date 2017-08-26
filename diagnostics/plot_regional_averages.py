import sys
sys.path.append("..")

import glob, os
import LMR_utils as L
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

def regional_mask(lat,lon,southlat,northlat,westlon,eastlon):

    """
    Given vectors for lat and lon, and lat-lon boundaries for a regional domain, 
    return an array of ones and zeros, with ones located within the domain and zeros outside
    the domain as defined by the input lat,lon vectors.
    """

    nlat = len(lat)
    nlon =len(lon)

    tmp = np.ones([nlon,nlat])
    latgrid = np.multiply(lat,tmp).T
    longrid = np.multiply(tmp.T,lon)

    lab = (latgrid >= southlat) & (latgrid <=northlat)
    # check for zero crossing 
    if eastlon < westlon:
        lob1 = (longrid >= westlon) & (longrid <=360.)
        lob2 = (longrid >= 0.) & (longrid <=eastlon)
        lob = lob1+lob2
    else:
        lob = (longrid >= westlon) & (longrid <=eastlon)

    mask = np.multiply(lab*lob,tmp.T)

    return mask

def PAGES2K_regional_means(field,lat,lon):

    """
     compute geographical spatial mean valuee for all times in the input (i.e. field) array. regions are defined by The PAGES2K Consortium (2013) Nature Geosciences paper, Supplementary Information.
     input:  field[ntime,nlat,nlon] or field[nlat,nlon]
             lat[nlat,nlon] in degrees
             lon[nlat,nlon] in degrees

     output: rm[nregions,ntime] : regional means of "field" where nregions = 7 by definition, but could change
     
    """

    # Originator: Greg Hakim
    #             University of Washington
    #             July 2017
    #
    # Modifications:
    #

    # print debug statements
    #debug = True
    debug = False
    
    # number of geographical regions (default, as defined in PAGES2K(2013) paper
    nregions = 7
    
    # set number of times, lats, lons; array indices for lat and lon    
    if len(np.shape(field)) == 3: # time is a dimension
        ntime,nlat,nlon = np.shape(field)
    else: # only spatial dims
        ntime = 1
        nlat,nlon = np.shape(field)
        field = field[None,:] # add time dim of size 1 for consistent array dims

    if debug:
        print 'field dimensions...'
        print np.shape(field)

    # define regions as in PAGES paper

    # lat and lon range for each region (first value is lower limit, second is upper limit)
    rlat = np.zeros([nregions,2]); rlon = np.zeros([nregions,2])
    # 1. Arctic: north of 60N 
    rlat[0,0] = 60.; rlat[0,1] = 90.
    rlon[0,0] = 0.; rlon[0,1] = 360.
    # 2. Europe: 35-70N, 10W-40E
    rlat[1,0] = 35.; rlat[1,1] = 70.
    rlon[1,0] = 350.; rlon[1,1] = 40.
    # 3. Asia: 23-55N; 60-160E (from map)
    rlat[2,0] = 23.; rlat[2,1] = 55.
    rlon[2,0] = 60.; rlon[2,1] = 160.
    # 4. North America 1 (trees):  30-55N, 75-130W 
    rlat[3,0] = 30.; rlat[3,1] = 55.
    rlon[3,0] = 55.; rlon[3,1] = 230.
    # 5. South America: Text: 20S-65S and 30W-80W
    rlat[4,0] = -65.; rlat[4,1] = -20.
    rlon[4,0] = 280.; rlon[4,1] = 330.
    # 6. Australasia: 110E-180E, 0-50S 
    rlat[5,0] = -50.; rlat[5,1] = 0.
    rlon[5,0] = 110.; rlon[5,1] = 180.
    # 7. Antarctica: south of 60S (from map)
    rlat[6,0] = -90.; rlat[6,1] = -60.
    rlon[6,0] = 0.; rlon[6,1] = 360.
    # ...add other regions here...
    
    # latitude weighting 
    lat_weight = np.cos(np.deg2rad(lat))
    tmp = np.ones([nlon,nlat])
    W = np.multiply(lat_weight,tmp).T

    rm  = np.zeros([nregions,ntime])

    # loop over regions
    for region in range(nregions):

        if debug:
            print 'region='+str(region)
            print rlat[region,0],rlat[region,1],rlon[region,0],rlon[region,1]
            
        # regional weighting (ones in region; zeros outside)
        mask = regional_mask(lat,lon,rlat[region,0],rlat[region,1],rlon[region,0],rlon[region,1])
        if debug:
            print 'mask='
            print mask

        # this is the weight mask for the regional domain    
        Wmask = np.multiply(mask,W)

        # make sure data starts at South Pole
        if lat[0] > 0:
            # data has NH -> SH format; reverse
            field = np.flipud(field)

        # Check for valid (non-NAN) values & use numpy average function (includes weighted avg calculation) 
        # Get arrays indices of valid values
        indok    = np.isfinite(field)
        for t in xrange(ntime):
            indok_2d = indok[t,:,:]
            field_2d = np.squeeze(field[t,:,:])
            if np.max(Wmask) >0.:
                rm[region,t] = np.average(field_2d[indok_2d],weights=Wmask[indok_2d])
            else:
                rm[region,t] = np.nan
    return rm


#----- start main----------------------------------------------------

# test #1: define a grid and region
lat = [-90.,-45.,0.,45.,90.]
lon = [0,90.,180,270.]

lat = np.array(lat)
lon = np.array(lon)
nlat = len(lat)
nlon = len(lon)

# define a lat-lon region
southlat = 0.
northlat = 90.
westlon = 230.
eastlon = 90.

mask = regional_mask(lat,lon,southlat,northlat,westlon,eastlon)

# latitude weighting 
lat_weight = np.cos(np.deg2rad(lat))
tmp = np.ones([nlon,nlat])
W = np.multiply(lat_weight,tmp).T

print 'W:'
print W

# the regional mask
Wmask = np.multiply(mask,W)
print 'masked W:'
print Wmask

#---------------------------------------------------------------------
# test #2: averages from LMR_utils for a given field
field =  np.ones([nlon,nlat])
field[1,1] = -7.
[tmp_gm,_,_] = L.global_hemispheric_means(field.T,lat)
print 'global mean: ' + str(tmp_gm)

# check with global weights defined here
gmt_check = np.average(field.T,weights=W)
print 'global mean local: ' + str(gmt_check)

# regional mask w/o function call
gmt_region = np.average(field.T,weights=Wmask)
print 'regional mean: ' + str(gmt_region)

#---------------------------------------------------------------------
# test #3: regional averages for LMR data using function call

# save figures here
figdir = '/Users/hakim/lmr/lmr_plots/'

# apply to LMR output
datadir_output = '/Users/hakim/data/LMR/archive'
#nexp = 'dadt_test_xbone'
nexp = 'pages2_loc25000_pages2k2_seasonal_TorP_nens200'
nexp = 'test'
dir = 'r0'

#var = 'tas_sfc_Adec'
var = 'tas_sfc_Amon'

workdir = datadir_output + '/' + nexp
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

[tmp_gm,_,_] = L.global_hemispheric_means(xam,lat[:,1])
#print tmp_gm
rm = PAGES2K_regional_means(xam,lat[:,1],lon[1,:])
nregions = np.shape(rm)[0]
print 'nregions='+ str(nregions)

# region labels for plotting
labs = [
'Arctic: north of 60N ',
'Europe: 35-70N, 10W-40E',
'Asia: 23-55N',
'North America (trees):30-55N,75-130W',
'South America: 20S-65S, 30W-80W',
'Australasia: 0-50S, 110E-180E', 
'Antarctica: south of 60S'
]

labs2 = [
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
    plt.plot(rm[k,:],label=labs2[k])
    plt.title(labs2[k])
    
plt.tight_layout()
fname = figdir+'regions_'+nexp+'.png'
print 'saving...'
print fname
plt.savefig(fname,additional_artists=art,bbox_inches='tight')
plt.show()

#---------------------------------------------------------------------
# test #4: regional averages for LMR data using function call

# loop over directories
MCset = None
#MCset = (0,2)
#MCset = (0,7)

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
    years = npzfile['years']
    nyrs =  len(years)
    if first:
        first = False
        rmsave = np.zeros([niters,nregions+3,nyrs])
        
    [tmp_gm,tmp_nh,tmp_sh] = L.global_hemispheric_means(xam,lat[:,1])
    rm = PAGES2K_regional_means(xam,lat[:,1],lon[1,:])
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
    plt.plot(years,rms_avg[k,:],color='b',label=labs2[k],lw=1.)
    plt.plot(years,rms_5[k,:],color='gray',alpha=0.75,lw=0.5)
    plt.plot(years,rms_95[k,:],color='gray',alpha=0.75,lw=0.5)
    #plt.fill_between(years,rms_5[k,:],rms_95[k,:],facecolor='gray',alpha = 0.5,linewidth=0.)
    plt.title(labs2[k])
    
plt.tight_layout()
fname = figdir+'regions_'+nexp+'_5_95.png'
print 'saving...'
print fname
plt.savefig(fname,additional_artists=art,bbox_inches='tight')


#plt.show()


