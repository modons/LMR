#
# compute ENSO indices
#
# imputs:
# lat2 : latitude array(nlat,nlon)
# lon2 : longitude array(nlat,nlon)
# t2m : 2 m air temperature array (ntimes,nlat,nlon)

import numpy as np

def enso(lat2,lon2,t2m):

    nll = np.shape(lat2)
    dlat = 180./(nll[0]-1)
    dlon = 360./nll[1]

    # http://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Nino12/
    # It is the area averaged SST from 0-10S and 90W-80W.

    slat = np.argmin(abs(-10.-lat2[:,1]))
    nlat = np.argmin(abs(0.-lat2[:,1]))
    wlon = np.argmin(abs(360.-90.-lon2[1,:]))
    elon = np.argmin(abs(360.-80.-lon2[1,:]))
    nino12 = np.mean(np.mean(t2m[:,slat:nlat,wlon:elon],1),1)
    
    # http://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Nino3/
    # It is the area averaged SST from 5S-5N and 150W-90W.

    slat = np.argmin(abs(-5.-lat2[:,1]))
    nlat = np.argmin(abs(5.-lat2[:,1]))
    wlon = np.argmin(abs(360.-150.-lon2[1,:]))
    elon = np.argmin(abs(360.-90.-lon2[1,:]))
    nino3 = np.mean(np.mean(t2m[:,slat:nlat,wlon:elon],1),1)

    # http://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Nino34/
    # It is the area averaged SST from 5S-5N and 170-120W.

    slat = np.argmin(abs(-5.-lat2[:,1]))
    nlat = np.argmin(abs(5.-lat2[:,1]))
    wlon = np.argmin(abs(360.-170.-lon2[1,:]))
    elon = np.argmin(abs(360.-120.-lon2[1,:]))
    nino34 = np.mean(np.mean(t2m[:,slat:nlat,wlon:elon],1),1)

    # http://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Nino4/
    # It is the area averaged SST from 5S-5N and 160E-150W.

    slat = np.argmin(abs(-5.-lat2[:,1]))
    nlat = np.argmin(abs(5.-lat2[:,1]))
    wlon = np.argmin(abs(360.-160.-lon2[1,:]))
    elon = np.argmin(abs(360.-150.-lon2[1,:]))
    nino4 = np.mean(np.mean(t2m[:,slat:nlat,wlon:elon],1),1)

    return nino12,nino3,nino34,nino4
