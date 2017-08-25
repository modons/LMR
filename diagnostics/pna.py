#
# compute PNA index as in Wallace and Gutzler (1981)
#
# imputs:
# lat2 : latitude array(nlat,nlon)
# lon2 : longitude array(nlat,nlon)
# z500 : 500 hPa geopotentaial height array (ntimes,nlat,nlon)

import numpy as np

def pna(lat2,lon2,z500):

    nll = np.shape(lat2)
    dlat = 180./(nll[0]-1)
    dlon = 360./nll[1]

    # define the lat,lon of the index locations
    lat = [20.,45.,55.,30.]
    lon = [160.,165.,115.,85.]

    # find the indices for these points in the lat, lon data
    ilat = np.zeros(4)
    kk = -1
    for k in lat:
        kk = kk + 1
        ilat[kk] = np.argmin(abs(k-lat2[:,1]))
        #print lat2[ilat[kk],1],k
        
    ilon = np.zeros(4)
    kk = -1
    for k in lon:
        kk = kk + 1
        ilon[kk] = np.argmin(abs(k-lon2[1,:]))
        #print lon2[1,ilon[kk]],k

    # standardize the z500 field
    zvar = np.var(z500,axis=0,ddof=1)
    print np.shape(z500)
    print np.shape(zvar)
    zs = z500/zvar
    print np.shape(zs)
    
    # compute the pna index
    nd = np.shape(zs)
    pna = 0.25*(zs[:,ilat[0],ilon[0]] - zs[:,ilat[1],ilon[1]] + zs[:,ilat[2],ilon[2]]- zs[:,ilat[3],ilon[3]])
    
    return pna
