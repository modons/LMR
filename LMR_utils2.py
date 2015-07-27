import glob
import numpy as np
from scipy import signal
from spharm import Spharmt, getspecindx, regrid
from math import radians, cos, sin, asin, sqrt

#==========================================================================================
#
# 
#========================================================================================== 

import glob
import numpy as np
import cPickle
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



def ensemble_stats(workdir, y_assim):

    """
    Compute the ensemble mean and variance for files in the input directory

    """

    # Originator: Greg Hakim
    #             University of Washington
    #             May 2015
    #
    #             revised 16 June 2015 (GJH)
    #             revised 24 June 2015 (R Tardif, UW)
    #               : func. renamed from ensemble_mean to ensemble_stats
    #               : computes and output the ensemble variance as well
    #               : now handles state vector possibly containing multiple variables
    #             revised 15 July 2015 (R Tardif, UW)
    #               : extracts Ye's from augmented state vector (Ye=HXa), match with corresponding
    #                 proxy sites from master list of proxies and output to analysis_Ye.pckl file

    prior_filn = workdir + '/Xb_one.npz'
    
    # get the prior and basic info
    npzfile = np.load(prior_filn)
    npzfile.files
    Xbtmp = npzfile['Xb_one']
    Xb_coords = npzfile['Xb_one_coords']

    # get state vector content info (state variables and their position in vector)
    # note: the .item() is necessary to access a dict stored in a npz file 
    state_info = npzfile['state_info'].item()
    nens = np.size(Xbtmp,1)

    # get a listing of the analysis files
    files = glob.glob(workdir+"/year*")
    # sorted
    files.sort()

    nyears = len(files)

    # loop on state variables 
    for var in state_info.keys():

        print 'State variable:', var

        ibeg = state_info[var]['pos'][0]
        iend = state_info[var]['pos'][1]

        # variable type (2D lat/lon, 2D lat/depth, time series etc ...)

        if state_info[var]['spacedims']: # var has spatial dimensions (not None)
            if len(state_info[var]['spacecoords']) == 2: # 2D variable
                ndim1 = state_info[var]['spacedims'][0]
                ndim2 = state_info[var]['spacedims'][1]
                
                Xb = np.reshape(Xbtmp[ibeg:iend+1,:],(ndim1,ndim2,nens))
                xbm = np.mean(Xb,axis=2) # ensemble mean
                xbv = np.var(Xb,axis=2,ddof=1)  # ensemble variance

                # process the **analysis** files
                years = []
                xam = np.zeros([nyears,ndim1,ndim2])
                xav = np.zeros([nyears,ndim1,ndim2],dtype=np.float64)
                k = -1
                for f in files:
                    k = k + 1
                    i = f.find('year')
                    year = f[i+4:i+8]
                    years.append(year)
                    Xatmp = np.load(f)
                    Xa = np.reshape(Xatmp[ibeg:iend+1,:],(ndim1,ndim2,nens))
                    xam[k,:,:] = np.mean(Xa,axis=2) # ensemble mean
                    xav[k,:,:] = np.var(Xa,axis=2,ddof=1)  # ensemble variance

                # form dictionary containing variables to save, including info on array dimensions
                coordname1 = state_info[var]['spacecoords'][0]
                coordname2 = state_info[var]['spacecoords'][1]
                dimcoord1 = 'n'+coordname1
                dimcoord2 = 'n'+coordname2

                coord1 = np.reshape(Xb_coords[ibeg:iend+1,0],[state_info[var]['spacedims'][0],state_info[var]['spacedims'][1]])
                coord2 = np.reshape(Xb_coords[ibeg:iend+1,1],[state_info[var]['spacedims'][0],state_info[var]['spacedims'][1]])

                vars_to_save_mean = {'nens':nens, 'years':years, dimcoord1:state_info[var]['spacedims'][0], dimcoord2:state_info[var]['spacedims'][1], \
                                         coordname1:coord1, coordname2:coord2, 'xbm':xbm, 'xam':xam}
                vars_to_save_var  = {'nens':nens, 'years':years, dimcoord1:state_info[var]['spacedims'][0], dimcoord2:state_info[var]['spacedims'][1], \
                                         coordname1:coord1, coordname2:coord2, 'xbv':xbv, 'xav':xav}
    
            else:
                print 'ERROR in ensemble_stats: Variable of unrecognized dimensions! Exiting'
                exit(1)


        else: # var has no spatial dims
            Xb = Xbtmp[ibeg:iend+1,:]
            xbm = np.mean(Xb,axis=1) # ensemble mean
            xbv = np.var(Xb,axis=1)  # ensemble variance
            
            # process the **analysis** files
            years = []
            xam = np.zeros([nyears])
            xav = np.zeros([nyears],dtype=np.float64)
            k = -1
            for f in files:
                k = k + 1
                i = f.find('year')
                year = f[i+4:i+8]
                years.append(year)
                Xatmp = np.load(f)
                Xa = Xatmp[ibeg:iend+1,:]
                xam[k] = np.mean(Xa,axis=1) # ensemble mean
                xav[k] = np.var(Xa,axis=1)  # ensemble variance     
                
            vars_to_save_mean = {'nens':nens, 'years':years, 'xbm':xbm, 'xam':xam}
            vars_to_save_var  = {'nens':nens, 'years':years, 'xbv':xbv, 'xav':xav}
            
        # ens. mean to file
        filen = workdir + '/ensemble_mean_' + var
        print 'writing the new ensemble mean file...' + filen
        #np.savez(filen, nlat=nlat, nlon=nlon, nens=nens, years=years, lat=lat, lon=lon, xbm=xbm, xam=xam)
        np.savez(filen, **vars_to_save_mean)

        # ens. variance to file
        filen = workdir + '/ensemble_variance_' + var
        print 'writing the new ensemble variance file...' + filen
        #np.savez(filen, nlat=nlat, nlon=nlon, nens=nens, years=years, lat=lat, lon=lon, xbv=xbv, xav=xav)
        np.savez(filen, **vars_to_save_var)

    # --------------------------------------------------------
    # Extract the analyzed Ye ensemble for diagnostic purposes
    # --------------------------------------------------------
    # get information on dim of state without the Ye's (before augmentation)
    stateDim  = npzfile['stateDim']
    Xbtmp_aug = npzfile['Xb_one_aug']
    # dim of entire state vector (augmented)
    totDim = Xbtmp_aug.shape[0]
    nbye = (totDim - stateDim)
    Ye_s = np.zeros([nbye,nyears,nens])

    # Loop over **analysis** files & extract the Ye's
    years = []
    for k, f in enumerate(files):
        i = f.find('year')
        year = f[i+4:i+8]
        years.append(float(year))
        Xatmp = np.load(f)
        # Extract the Ye's from augmented state (beyond stateDim 'til end of
        #  state vector)
        Ye_s[:, k, :] = Xatmp[stateDim:, :]
    years = np.array(years)

    # Build dictionary
    YeDict = {}
    # loop over assimilated proxies
    for i, pobj in enumerate(y_assim):
        # build boolean of indices to pull from HXa
        yr_idxs = [year in years for year in pobj.time]
        YeDict[(pobj.type, pobj.id)] = {}
        YeDict[(pobj.type, pobj.id)]['lat'] = pobj.lat
        YeDict[(pobj.type, pobj.id)]['lon'] = pobj.lon
        YeDict[(pobj.type, pobj.id)]['R'] = pobj.psm_obj.R
        YeDict[(pobj.type, pobj.id)]['years'] = pobj.time
        YeDict[(pobj.type, pobj.id)]['HXa'] = Ye_s[i, yr_idxs, :]

    # Dump dictionary to pickle file
    outfile = open('{}/analysis_Ye.pckl'.format(workdir), 'w')
    cPickle.dump(YeDict, outfile)
    outfile.close()

    return


def regrid_sphere(nlat,nlon,Nens,X,ntrunc):

    """
    Truncate lat,lon grid to another resolution in spherical harmonic space. Triangular truncation

    Inputs:
    nlat            : number of latitudes
    nlon            : number of longitudes
    Nens            : number of ensemble members
    X               : data array of shape (nlat*nlon,Nens) 
    ntrunc          : triangular truncation (e.g., use 42 for T42)

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


def class_docs_fixer(cls):
    """Decorator to fix docstrings for subclasses"""
    if not cls.__doc__:
        for parent in cls.__bases__:
            if parent.__doc__:
                cls.__doc__ = parent.__doc__

    for name, func in vars(cls).items():
        if not func.__doc__ or '%%aug%%' in func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name)
                if parfunc and getattr(parfunc, '__doc__', None):
                    if not func.__doc__:
                        func.__doc__ = parfunc.__doc__
                        break
                    elif '%%aug%%' in func.__doc__:
                        func.__doc__ = func.__doc__.replace('%%aug%%', '')
                        func.__doc__ = parfunc.__doc__ + func.__doc__
                        break

    return cls

def augment_docstr(func):
    """ Decorator to mark augmented function docstrings. """
    func.func_doc = '%%aug%%' + func.func_doc
    return func
