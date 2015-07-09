
#==========================================================================================
#
# 
#========================================================================================== 

import glob
import numpy as np
from math import radians, cos, sin, asin, sqrt
from scipy import signal
from spharm import Spharmt, getspecindx, regrid

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367.0 * c
    return km

def year_fix(t):

    """
    pad the year with leading zeros for file I/O
    """
    # Originator: Greg Hakim
    #             University of Washington
    #             March 2015
    #
    #             revised 16 June 2015 (GJH)

    # make sure t is an integer
    t = int(t)

    if t < 10:
        ypad = '000'+str(t)
    elif t >= 10 and t < 100:
        ypad = '00'+str(t)
    elif t >= 100 and t < 1000:
        ypad = '0'+str(t)
    else:
        ypad = str(t)
    
    return ypad

def smooth2D(im, n=15):

    """
    Smooth a 2D array im by convolving with a Gaussian kernel of size n
    Input:
    im (2D array): Array of values to be smoothed
    n (int) : number of points to include in the smoothing
    Output:
    improc(2D array): smoothed array (same dimensions as the input array)
    """
    # Originator: Greg Hakim
    #             University of Washington
    #             May 2015


    # Calculate a normalised Gaussian kernel to apply as the smoothing function.
    size = int(n)
    x,y = np.mgrid[-n:n+1,-n:n+1]
    gk = np.exp(-(x**2/float(n)+y**2/float(n)))
    g = gk / gk.sum()

    #bndy = 'symm'
    bndy = 'wrap'
    improc = signal.convolve2d(im, g, mode='same', boundary=bndy)
    return(improc)

def global_mean(field,lat,lon):

    """
     compute global mean value for all times in the input array
     input: field[ntime,nlat,nlon] or field{nlat,nlon]
            lat[nlat,nlon] in degrees
            lon[nlat,nlon] in degrees
    """

    # Originator: Greg Hakim
    #             University of Washington
    #             May 2015
    #
    #             revised 16 June 2015 (GJH)

    # set number of times, lats, lons; array indices for lat and lon    
    if len(np.shape(field)) == 3:
        ntime,nlat,nlon = np.shape(field)
        lati = 1
        loni = 2
    else:
        ntime = 1
        nlat,nlon = np.shape(field)
        lati = 0
        loni = 1

    # latitude weighting for global mean
    lat_weight = np.cos(np.deg2rad(lat))
    #--old
    #tmp = np.ones([len(lat),len(lon)])
    #W = np.multiply(lat_weight,tmp.T).T
    #--new
    tmp = np.ones([nlon,nlat])
    W = np.multiply(lat_weight,tmp).T

    gm = np.zeros(ntime)
    for t in xrange(ntime):
        if lati == 0:
            gm[t] = np.nansum(np.multiply(W,field))/(np.sum(np.sum(W)))
        else:
            gm[t] = np.nansum(np.multiply(W,field[t,:,:]))/(np.sum(np.sum(W)))
 
    return gm

def ensemble_mean(workdir):

    """
    Compute the ensemble mean for files in the input directory

    """

    # Originator: Greg Hakim
    #             University of Washington
    #             May 2015
    #
    #             revised 16 June 2015 (GJH)

    prior_filn = workdir + '/Xb_one.npz'
    
    # get the prior and basic info
    npzfile = np.load(prior_filn)
    npzfile.files
    Xbtmp = npzfile['Xb_one']
    nlat = npzfile['nlat']
    nlon = npzfile['nlon']
    lat = npzfile['lat']
    lon = npzfile['lon']
    stateDim = npzfile['stateDim']
    nens = np.size(Xbtmp,1)
    Xb = np.reshape(Xbtmp,(nlat,nlon,nens))
    xbm = np.mean(Xb,axis=2)
    #[stateDim, _] = Xbtmp.shape

    # get a listing of the analysis files
    files = glob.glob(workdir+"/year*")

    # sorted
    files.sort()
    
    # process the analysis files
    nyears = len(files)
    years = []
    xam = np.zeros([nyears,nlat,nlon])
    xav = np.zeros([nyears,nlat,nlon])
    k = -1
    for f in files:
        #print 'file = ' + f
        k = k + 1
        i = f.find('year')
        year = f[i+4:i+8]
        years.append(year)
        Xatmp = np.load(f)
        Xa = np.reshape(Xatmp[0:stateDim,:],(nlat,nlon,nens))
        xam[k,:,:] = np.mean(Xa,axis=2)
        xav[k,:,:] = np.var(Xa,axis=2,ddof=1)
        
    filen = workdir + '/ensemble_mean'
    print 'writing the new ensemble mean file...' + filen
    np.savez(filen, nlat=nlat, nlon=nlon, nens=nens, years=years, lat=lat, lon=lon, xbm=xbm, xam=xam)
    filen = workdir + '/ensemble_variance'
    print 'writing the new ensemble variance file...' + filen
    np.savez(filen, nlat=nlat, nlon=nlon, nens=nens, years=years, lat=lat, lon=lon, xav=xav)
        
    return


def regrid_sphere(nlat,nlon,Nens,X,ntrunc):

    """
    Truncate lat,lon grid to another resolution in spherical harmonic space. Triangular truncation

    Inputs:
    nlat  : number of latitudes
    nlon  : number of longitudes
    Nens  : number of ensemble members
    X     : data array of shape (nlat*nlon,Nens) 
    ntrunc: triangular truncation (e.g., use 42 for T42)

    Outputs :
    lat_new : 2D latitude array on the new grid (nlat_new,nlon_new)
    lon_new : 2D longitude array on the new grid (nlat_new,nlon_new)
    X_new   : truncated data array of shape (nlat_new*nlon_new, Nens)
    """
    # Originator: Greg Hakim
    #             University of Washington
    #             May 2015

    # create the spectral object on the original grid
    specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')

    # truncate to a lower resolution grid (triangular truncation)
    ifix = np.remainder(ntrunc,2.0).astype(int)
    nlat_new = ntrunc + ifix
    nlon_new = int(nlat_new*1.5)

    # create the spectral object on the new grid
    specob_new = Spharmt(nlon_new,nlat_new,gridtype='regular',legfunc='computed')

    # create new lat,lon grid arrays
    dlat = 90./((nlat_new-1)/2.)
    dlon = 360./nlon_new
    veclat = np.arange(-90.,90.+dlat,dlat)
    veclon = np.arange(0.,360.,dlon)
    blank = np.zeros([nlat_new,nlon_new])
    lat_new = (veclat + blank.T).T  
    lon_new = (veclon + blank)

    # transform each ensemble member, one at a time
    X_new = np.zeros([nlat_new*nlon_new,Nens])
    for k in range(Nens):
        X_lalo = np.reshape(X[:,k],(nlat,nlon))
        Xbtrunc = regrid(specob_lmr, specob_new, X_lalo, ntrunc=nlat_new-1, smooth=None)
        vectmp = Xbtrunc.flatten()
        X_new[:,k] = vectmp

    return X_new,lat_new,lon_new

def assimilated_proxies(workdir):

    """
    Read the files written by LMR_driver_callable as written to directory workdir. Returns a dictionary with a count by proxy type.
    
    """
    # Originator: Greg Hakim
    #             University of Washington
    #             May 2015

    apfile = workdir + 'assimilated_proxies.npy'
    assimilated_proxies = np.load(apfile)
    nrecords = np.size(assimilated_proxies)

    ptypes = {}
    for rec in range(nrecords):
        key = assimilated_proxies[rec].keys()[0]
        if key in ptypes:
            pc = ptypes[key]
            ptypes[key] = pc + 1
        else:
            ptypes[key] = 1
            
    return ptypes,nrecords

def coefficient_efficiency(ref,test):
    """
    Compute the coefficient of efficiency for a test time series, with respect to a reference time series.

    Inputs:
    test: one-dimensional test array
    ref: one-dimensional reference array, of same size as test

    Outputs:
    CE: scalar CE score
    """

    # error
    error = test - ref

    # error variance
    evar = np.var(error,ddof=1)

    # variance in the reference 
    rvar = np.var(ref,ddof=1)

    # CE
    CE = 1. - (evar/rvar)

    return CE

def rank_histogram(ensemble, value):

    """
    Compute the rank of a measurement in the contex of an ensemble. 
    
    Input:
    * the observation (value)
    * the ensemble evaluated at the observation position (ensemble)

    Output:
    * the rank of the observation in the ensemble (rank)
    """
    # Originator: Greg Hakim
    #             University of Washington
    #             July 2015

    # convert the numpy array to a list so that the "truth" can be appended
    Lensemble = ensemble.tolist()
    Lensemble.append(value)

    # convert the list back to a numpy array so we have access to a sorting function
    Nensemble = np.array(Lensemble)
    sort_index = np.argsort(Nensemble)

    # convert the numpy array containing the ranked list indices back to an ordinary list for indexing
    Lsort_index = sort_index.tolist()
    rank = Lsort_index.index(len(Lensemble)-1)

    return rank




