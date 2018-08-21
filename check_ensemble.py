#
# check the output of the ensemble "global mean" calculation with fabricated data
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import LMR_utils


# set number of ensemble members and degrees of freedom, and bogus lat,lon data
workdir = './'
ntims = 10
startim = 1000
Nens = 10
nlat_new = 3
nlon_new = 5

# make some fake lat,lon arrays
dlat = 90./((nlat_new-1)/2.)
dlon = 360./nlon_new
veclat = np.arange(-90.,90.+dlat,dlat)
veclon = np.arange(0.,360.,dlon)
blank = np.zeros([nlat_new,nlon_new])
lat_new = (veclat + blank.T).T  
lon_new = (veclon + blank)
Ndof = nlat_new*nlon_new

Xb_one = np.random.randn(Ndof,Nens)
[stateDim, _] = Xb_one.shape
xbm = np.mean(Xb_one,axis=1)
# Dump prior state vector (Xb_one) to file 
filen = workdir + '/' + 'Xb_one'
np.savez(filen,Xb_one = Xb_one,stateDim = stateDim,lat = lat_new,lon = lon_new, nlat = nlat_new, nlon = nlon_new)
gmt = np.zeros([ntims])
k = -1
for t in range(startim,startim+ntims):
    k = k + 1
    # make up some data with the right shape as in the LMR code (Ndof,Nens)
    Xa = np.random.randn(Ndof,Nens)    
    xam = np.mean(Xa,axis=1)
    print('Xa shape: ' + str(np.shape(Xa)))

    # Dump Xa to file (to be used as prior for next assimilation)
    ypad = LMR_utils.year_fix(t)
    filen = workdir + '/' + 'year' + ypad
    np.save(filen,Xa)

    # compute global mean for check later
    xam_lalo = np.reshape(xam[0:stateDim],(nlat_new,nlon_new))
    [gmt[k],_,_] = LMR_utils.global_hemispheric_means(xam_lalo, lat_new[:, 0])

# generate the ensemble-mean files as in LMR_wrapper.py
LMR_utils.ensemble_mean(workdir)

#
# now "post-process" the file just written as in verify_grid_testing.py
#

print('reading Xb_one file...')

# reset workdir to a real LMR experiment
workdir = '../data/testdev_check_1000_75pct/r0/'
prior_filn = workdir + '/Xb_one.npz'
npzfile = np.load(prior_filn)
npzfile.files
lat = npzfile['lat']
lon = npzfile['lon']
nlat = npzfile['nlat']
nlon = npzfile['nlon']
lat2 = np.reshape(lat,(nlat,nlon))
lon2 = np.reshape(lon,(nlat,nlon))
print(np.shape(lat))
print(np.shape(lat2))

"""
print 'lat check (should be zero): ' + str(np.max(np.max(lat - lat_new)))
print 'lon check (should be zero): ' + str(np.max(np.max(lon - lon_new)))
print 'nlat check:' + str(nlat_new) + ' =? ' + str(nlat)
print 'nlon check:' + str(nlon_new) + ' =? ' + str(nlon)
"""

print('\n reading LMR ensemble-mean data...\n')
ensfiln = workdir + '/ensemble_mean.npz'
print(ensfiln)
npzfile = np.load(ensfiln)
npzfile.files
xam_check = npzfile['xam']

print('\n shape of xam from ensemble_mean.npz = ' + str(np.shape(xam_check)))

"""
print xam_check
print '----------------'
print xam
[gmt_check,_,_] = LMR_utils.global_hemispheric_means(xam_check,lat[:,0])

print gmt_check
print '----------------'
print gmt
"""

[gmt_check,_,_] = LMR_utils.global_hemispheric_means(xam_check, lat[:, 0])

#
# read gmt data (as in LMR_verify_GM.py)
#

gmtpfile =  workdir + '/gmt.npz'
npzfile = np.load(gmtpfile)
npzfile.files
gmt = npzfile['gmt_save']
recon_times = npzfile['recon_times']
apcount = npzfile['apcount']
print(gmtpfile)
print(apcount)
gmt_src = gmt[-1,:]
print('shape gmt = ' + str(np.shape(gmt)))
print('shape gmt_src = ' + str(np.shape(gmt_src)))

gmt_err = gmt_src - gmt_check
print('mx gmt error = ' + str(np.max(gmt_err)))

plt.plot(gmt_src,'r-')
plt.plot(gmt_check,'b-')
plt.show()

