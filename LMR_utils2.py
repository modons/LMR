import glob
import numpy as np
import cPickle
import tables as tb
from scipy import signal
from spharm import Spharmt, regrid


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    c = 2 * np.arcsin(np.sqrt(a))
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


def fix_lon(lon):
    """
    Fixes negative longitude values.

    Parameters
    ----------
    lon: ndarray like or value
        Input longitude array or single value
    """
    if lon is not None and np.any(lon < 0):
        if isinstance(lon, np.ndarray):
            lon[lon < 0] += 360.
        else:
            lon += 360
    return lon


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
    x, y = np.mgrid[-n:n+1,-n:n+1]
    gk = np.exp(-(x**2/float(n)+y**2/float(n)))
    g = gk / gk.sum()

    # bndy = 'symm'
    bndy = 'wrap'
    improc = signal.convolve2d(im, g, mode='same', boundary=bndy)
    return improc


def global_mean2(field, lat, output_hemispheric=False):

    """
     compute global mean value for all times in the input array
     input: field with dimensions
            [time, lat, lon]
            [time, lat*lon]
            [lat, lon]
            [lat*lon]

            lat in degrees
            has to match field dimensions, either singleton nlat, or nlat*nlon

    Altered by AndreP August 2015
    """

    # Originator: Greg Hakim
    #             University of Washington
    #             May 2015
    #
    #             revised 16 June 2015 (GJH)

    # index for start of spatial dimensions

    if not lat.shape == field.shape[-lat.ndim:]:
        # received singleton lat
        lat_idx = field.shape.index(len(lat))

        tmp = np.ones(field.shape[lat_idx:])
        lat = (tmp.T * lat).T  # broadcast latitude dimension

    # Flatten spatial dimensions if necessary
    if lat.ndim > 1:
        field = field.reshape(list(field.shape[:-lat.ndim]) +
                              [np.product(field.shape[-lat.ndim:])])
        lat = lat.flatten()

    # latitude weighting for global mean
    lat_weight = np.cos(np.deg2rad(lat))
    # If nan-values in the field, result in nan
    field_weighted = field * lat_weight

    # mask lat_weights
    lat_weight = np.ones_like(field_weighted) * lat_weight
    lat_weight[~np.isfinite(field_weighted)] = np.nan

    gm = np.nansum(field_weighted, axis=-1) / np.nansum(lat_weight, axis=-1)

    if output_hemispheric:
        nh_idx, = np.where(lat > 0)
        nh_wgt_field = np.take(field_weighted, nh_idx, axis=-1)
        nh_lat_wgt = np.take(lat_weight, nh_idx, axis=-1)
        nhm = np.nansum(nh_wgt_field, axis=-1)
        nhm /= np.nansum(nh_lat_wgt, axis=-1)

        sh_idx, = np.where(lat < 0)
        sh_wgt_field = np.take(field_weighted, sh_idx, axis=-1)
        sh_lat_wgt = np.take(lat_weight, sh_idx, axis=-1)
        shm = np.nansum(sh_wgt_field, axis=-1)
        shm /= np.nansum(sh_lat_wgt, axis=-1)

        return gm, nhm, shm

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
    #               : now handles state vector possibly containing multiple
    #               : variables
    #             revised 15 July 2015 (R Tardif, UW)
    #               : extracts Ye's from augmented state vector (Ye=HXa), match
    #               : with corresponding
    #               : proxy sites from master list of proxies and output to
    #               : analysis_Ye.pckl file

    prior_filn = workdir + '/Xb_one.npz'
    
    # get the prior and basic info
    npzfile = np.load(prior_filn)
    npzfile.files
    Xbtmp = np.array(npzfile['Xb_one']).mean(axis=0)
    Xb_coords = npzfile['Xb_one_coords'].item()

    # get state vector content info
    # (state variables and their position in vector)
    # note: the .item() is necessary to access a dict stored in a npz file 
    state_info = npzfile['state_info'].item()
    nens = np.size(Xbtmp, 1)

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

        if state_info[var]['spacedims']:  # var has spatial dimensions(not None)
            if len(state_info[var]['spacecoords']) == 2:  # 2D variable
                ndim1 = state_info[var]['spacedims'][0]
                ndim2 = state_info[var]['spacedims'][1]
                
                Xb = np.reshape(Xbtmp[ibeg:iend, :], (ndim1, ndim2, nens))
                xbm = np.mean(Xb, axis=2)  # ensemble mean
                xbv = np.var(Xb, axis=2, ddof=1)  # ensemble variance

                # process the **analysis** files
                years = []
                xam = np.zeros([nyears, ndim1, ndim2])
                xav = np.zeros([nyears, ndim1, ndim2], dtype=np.float64)
                k = -1
                for f in files:
                    k += 1
                    i = f.find('year')
                    year = f[i+4:i+8]
                    years.append(year)
                    Xatmp = np.load(f)
                    Xa = np.reshape(Xatmp[ibeg:iend, :], (ndim1, ndim2, nens))
                    xam[k, :, :] = np.mean(Xa, axis=2)  # ensemble mean
                    # ensemble variance
                    xav[k, :, :] = np.var(Xa, axis=2, ddof=1)

                # form dictionary containing variables to save, including info
                # on array dimensions
                coordname1 = state_info[var]['spacecoords'][0]
                coordname2 = state_info[var]['spacecoords'][1]
                dimcoord1 = 'n'+coordname1
                dimcoord2 = 'n'+coordname2

                coord1 = np.reshape(Xb_coords[var][coordname1],
                                    state_info[var]['spacedims'])
                coord2 = np.reshape(Xb_coords[var][coordname2],
                                    state_info[var]['spacedims'])

                vars_to_save_mean = {'nens': nens, 'years': years,
                                     dimcoord1: state_info[var]['spacedims'][0],
                                     dimcoord2: state_info[var]['spacedims'][1],
                                     coordname1: coord1,
                                     coordname2: coord2,
                                     'xbm': xbm,
                                     'xam': xam}

                vars_to_save_var  = {'nens': nens, 'years': years,
                                     dimcoord1: state_info[var]['spacedims'][0],
                                     dimcoord2: state_info[var]['spacedims'][1],
                                     coordname1: coord1,
                                     coordname2: coord2,
                                     'xbv': xbv,
                                     'xav': xav}
    
            else:
                print ('ERROR in ensemble_stats: Variable of unrecognized'
                       ' dimensions! Exiting')
                exit(1)

        else:  # var has no spatial dims
            Xb = Xbtmp[ibeg:iend, :]  # prior ensemble
            xbm = np.mean(Xb, axis=1)  # ensemble mean
            xbv = np.var(Xb, axis=1)  # ensemble variance

            # process the **analysis** files
            years = []
            xa_ens = np.zeros((nyears, Xb.shape[1]))
            xam = np.zeros([nyears])
            xav = np.zeros([nyears], dtype=np.float64)
            k = -1
            for f in files:
                k += 1
                i = f.find('year')
                year = f[i+4:i+8]
                years.append(year)
                Xatmp = np.load(f)
                Xa = Xatmp[ibeg:iend, :]
                xa_ens[k] = Xa  # total ensemble
                xam[k] = np.mean(Xa, axis=1)  # ensemble mean
                xav[k] = np.var(Xa, axis=1)  # ensemble variance

            vars_to_save_ens = {'nens': nens, 'years': years, 'xb_ens': Xb,
                                'xa_ens': xa_ens}
            vars_to_save_mean = {'nens': nens, 'years': years, 'xbm': xbm,
                                 'xam': xam}
            vars_to_save_var = {'nens': nens, 'years': years, 'xbv': xbv,
                                'xav': xav}

            # ens to file
            filen = workdir + '/ensemble_' + var
            print 'writing the new ensemble file' + filen
            np.savez(filen, **vars_to_save_ens)

        # ens. mean to file
        filen = workdir + '/ensemble_mean_' + var
        print 'writing the new ensemble mean file...' + filen
        np.savez(filen, **vars_to_save_mean)

        # ens. variance to file
        filen = workdir + '/ensemble_variance_' + var
        print 'writing the new ensemble variance file...' + filen
        np.savez(filen, **vars_to_save_var)

    # --------------------------------------------------------
    # Extract the analyzed Ye ensemble for diagnostic purposes
    # --------------------------------------------------------
    # get information on dim of state without the Ye's (before augmentation)
    stateDim = npzfile['stateDim']
    Xbtmp_aug = npzfile['Xb_one_aug'].mean(axis=0)
    # dim of entire state vector (augmented)
    totDim = Xbtmp_aug.shape[0]
    nbye = (totDim - stateDim)
    Ye_s = np.zeros([nbye, nyears, nens])

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


def regrid_sphere(nlat, nlon, Nens, X, ntrunc):

    """
    Truncate lat,lon grid to another resolution in spherical harmonic space.
    Triangular truncation

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
    specob_lmr = Spharmt(nlon, nlat, gridtype='regular', legfunc='computed')

    # truncate to a lower resolution grid (triangular truncation)
    # ifix = np.remainder(ntrunc,2.0).astype(int)
    # nlat_new = ntrunc + ifix
    # nlon_new = int(nlat_new*1.5)

    # truncate to a lower resolution grid (triangular truncation)
    # nlat must be ntrunc+1 per the documentation, want to keep it even
    # so that poles are not included in the grid and because the original
    # experiments were with even nlat
    nlat_new = (ntrunc + 1) + (ntrunc + 1) % 2
    nlon_new = int(nlat_new*1.5)

    # create the spectral object on the new grid
    specob_new = Spharmt(nlon_new, nlat_new, gridtype='regular',
                         legfunc='computed')

    # create new lat,lon grid arrays
    dlat = 90./((nlat_new-1)/2.)
    dlon = 360./nlon_new
    veclat = np.arange(-90., 90.+dlat, dlat)
    veclon = np.arange(0., 360., dlon)
    blank = np.zeros([nlat_new, nlon_new])
    lat_new = (veclat + blank.T).T  
    lon_new = (veclon + blank)

    # transform each ensemble member, one at a time
    X_new = np.zeros([nlat_new*nlon_new, Nens])
    for k in range(Nens):
        X_lalo = np.reshape(X[:, k], (nlat, nlon))
        Xbtrunc = regrid(specob_lmr, specob_new, X_lalo, ntrunc=nlat_new-1,
                         smooth=None)
        vectmp = Xbtrunc.flatten()
        X_new[:, k] = vectmp

    return X_new, lat_new, lon_new


def regrid_sphere2(grid_obj, ntrunc):

    """
    An adaptation of regrid_shpere for new GriddedData objects

    Inputs:
    grid_obj
    ntrunc

    Outputs :
    lat_new : 2D latitude array on the new grid (nlat_new,nlon_new)
    lon_new : 2D longitude array on the new grid (nlat_new,nlon_new)
    X_new   : truncated data array of shape (nlat_new*nlon_new, Nens)
    """
    # Originator: Greg Hakim
    #             University of Washington
    #             May 2015

    # create the spectral object on the original grid
    specob_lmr = Spharmt(len(grid_obj.lon), len(grid_obj.lat),
                         gridtype='regular', legfunc='computed')

    # truncate to a lower resolution grid (triangular truncation)
    # nlat must be ntrunc+1 per the documentation, want to keep it even
    # so that poles are not included in the grid and because the original
    # experiments were with even nlat
    nlat_new = (ntrunc + 1) + (ntrunc + 1) % 2
    nlon_new = int(nlat_new*1.5)

    # truncate to a lower resolution grid (triangular truncation)

    # create the spectral object on the new grid
    specob_new = Spharmt(nlon_new, nlat_new, gridtype='regular',
                         legfunc='computed')

    # create new lat,lon grid arrays
    dlat = 90./((nlat_new-1)/2.)
    dlon = 360./nlon_new
    veclat = np.arange(-90., 90.+dlat, dlat)
    veclon = np.arange(0., 360., dlon)
    blank = np.zeros([nlat_new, nlon_new])
    lat_new = (veclat + blank.T).T
    lon_new = (veclon + blank)

    # transform each ensemble member, one at a time
    gridded_new = np.zeros((len(grid_obj.time), nlat_new, nlon_new))
    for i, time_slice in enumerate(grid_obj.data):
        gridded_new[i] = regrid(specob_lmr, specob_new, time_slice,
                                ntrunc=ntrunc)

    return gridded_new, lat_new, lon_new


def assimilated_proxies(workdir):

    """
    Read the files written by LMR_driver_callable as written to directory
    workdir. Returns a dictionary with a count by proxy type.
    
    """
    # Originator: Greg Hakim
    #             University of Washington
    #             May 2015

    apfile = workdir + 'assimilated_proxies.npy'
    assim_proxies = np.load(apfile)
    nrecords = np.size(assim_proxies)

    ptypes = {}
    for rec in range(nrecords):
        key = assim_proxies[rec].keys()[0]
        if key in ptypes:
            pc = ptypes[key]
            ptypes[key] = pc + 1
        else:
            ptypes[key] = 1
            
    return ptypes, nrecords

def coefficient_efficiency_old(ref,test):
    """
    Compute the coefficient of efficiency for a test time series, with respect
    to a reference time series.

    Inputs:
    test: one-dimensional test array
    ref: one-dimensional reference array, of same size as test

    Outputs:
    CE: scalar CE score
    """

    # error
    error = test - ref

    # error variance
    evar = np.var(error, ddof=1)

    # variance in the reference 
    rvar = np.var(ref, ddof=1)

    # CE
    CE = 1. - (evar/rvar)

    return CE


def coefficient_efficiency(ref,test,valid=None):
    """
    Compute the coefficient of efficiency for a test time series, with respect
    to a reference time series.

    Inputs:
    test:  test array
    ref:   reference array, of same size as test
    valid: fraction of valid data required to calculate the statistic

    Note: Assumes that the first dimension in test and ref arrays is time!!!

    Outputs:
    CE: CE statistic calculated following Nash & Sutcliffe (1970)
    """

    # check array dimensions
    dims_test = test.shape
    dims_ref  = ref.shape
    #print 'dims_test: ', dims_test, ' dims_ref: ', dims_ref

    if len(dims_ref) == 3:   # 3D: time + 2D spatial
        dims = dims_ref[1:3]
    elif len(dims_ref) == 2: # 2D: time + 1D spatial
        dims = dims_ref[1:2]
    elif len(dims_ref) == 1: # 1D: time series
        dims = 1
    else:
        print 'Problem with input array dimension! Exiting...'
        exit(1)

    #print 'dims CE: ', dims
    CE = np.zeros(dims)

    # error
    error = test - ref

    # CE
    numer = np.nansum(np.power(error,2),axis=0)
    denom = np.nansum(np.power(ref-np.nanmean(ref,axis=0),2),axis=0)
    CE    = 1. - np.divide(numer,denom)

    if valid:
        nbok  = np.sum(np.isfinite(ref),axis=0)
        nball = float(dims_ref[0])
        ratio = np.divide(nbok,nball)
        indok  = np.where(ratio >= valid)
        indbad = np.where(ratio < valid)
        dim_indbad = len(indbad)
        testlist = [indbad[k].size for k in range(dim_indbad)]
        if not all(v == 0 for v in testlist):
            if dims>1:
                CE[indbad] = np.nan
            else:
                CE = np.nan

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

    # convert the list back to a numpy array so we have access to a sorting
    # function
    Nensemble = np.array(Lensemble)
    sort_index = np.argsort(Nensemble)

    # convert the numpy array containing the ranked list indices back to an
    # ordinary list for indexing
    Lsort_index = sort_index.tolist()
    rank = Lsort_index.index(len(Lensemble)-1)

    return rank


def global_hemispheric_means(field, lat):

    """
     compute global and hemispheric mean valuee for all times in the input
     (i.e. field) array

     input:  field[ntime,nlat,nlon] or field[nlat,nlon]
             lat[nlat,nlon] in degrees

     output: gm : global mean of "field"
            nhm : northern hemispheric mean of "field"
            shm : southern hemispheric mean of "field"
    """

    # Originator: Greg Hakim
    #             University of Washington
    #             August 2015
    #
    # Modifications:
    #           - Modified to handle presence of missing values (nan) in arrays
    #             in calculation of spatial averages [ R. Tardif, November 2015) ]
    #

    # set number of times, lats, lons; array indices for lat and lon
    if len(np.shape(field)) == 3:  # time is a dimension
        ntime, nlat, nlon = np.shape(field)
        lati = 1
        loni = 2
    else:  # only spatial dims
        ntime = 1
        nlat, nlon = np.shape(field)
        field = field[None, :]  # add time dim of size 1 for consistent dims
        lati = 1
        loni = 2

    # latitude weighting
    lat_weight = np.cos(np.deg2rad(lat))
    tmp = np.ones([nlon, nlat])
    W = np.multiply(lat_weight, tmp).T

    # define hemispheres
    eqind = nlat/2

    if lat[0] > 0:
        # data has NH -> SH format
        W_NH = W[0:eqind]
        field_NH = field[:, 0:eqind, :]
        W_SH = W[eqind:]
        field_SH = field[:, eqind:, :]
    else:
        # data has SH -> NH format
        W_NH = W[eqind:]
        field_NH = field[:, eqind:, :]
        W_SH = W[0:eqind]
        field_SH = field[:, 0:eqind, :]

    gm = np.zeros(ntime)
    nhm = np.zeros(ntime)
    shm = np.zeros(ntime)

    # Check for valid (non-NAN) values & use numpy average function (includes weighted avg calculation)
    # Get arrays indices of valid values
    indok    = np.isfinite(field)
    indok_nh = np.isfinite(field_NH)
    indok_sh = np.isfinite(field_SH)
    for t in xrange(ntime):
        if lati == 0:
            # Global
            gm[t]  = np.average(field[indok],weights=W[indok])
            # NH
            nhm[t] = np.average(field_NH[indok_nh],weights=W_NH[indok_nh])
            # SH
            shm[t] = np.average(field_SH[indok_sh],weights=W_SH[indok_sh])
        else:
            # Global
            indok_2d    = indok[t,:,:]
            field_2d    = np.squeeze(field[t,:,:])
            gm[t]       = np.average(field_2d[indok_2d],weights=W[indok_2d])
            # NH
            indok_nh_2d = indok_nh[t,:,:]
            field_nh_2d = np.squeeze(field_NH[t,:,:])
            nhm[t]      = np.average(field_nh_2d[indok_nh_2d],weights=W_NH[indok_nh_2d])
            # SH
            indok_sh_2d = indok_sh[t,:,:]
            field_sh_2d = np.squeeze(field_SH[t,:,:])
            shm[t]      = np.average(field_sh_2d[indok_sh_2d],weights=W_SH[indok_sh_2d])

# original code (keep for now...)
#    for t in xrange(ntime):
#        if lati == 0:
#            gm[t]  = np.sum(np.multiply(W,field))/(np.sum(np.sum(W)))
#            nhm[t] = np.sum(np.multiply(W_NH,field_NH))/(np.sum(np.sum(W_NH)))
#            shm[t] = np.sum(np.multiply(W_SH,field_SH))/(np.sum(np.sum(W_SH)))
#        else:
#            gm[t]  = np.sum(np.multiply(W,field[t,:,:]))/(np.sum(np.sum(W)))
#            nhm[t] = np.sum(np.multiply(W_NH,field_NH[t,:,:]))/(np.sum(np.sum(W_NH)))
#            shm[t] = np.sum(np.multiply(W_SH,field_SH[t,:,:]))/(np.sum(np.sum(W_SH)))


    return gm, nhm, shm


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


def var_to_hdf5_carray(h5file, group, node, data, **kwargs):
    """
    Take an input data and insert into a PyTables carray in an HDF5 file.

    Parameters
    ----------
    h5file: tables.File
        Writeable HDF5 file to insert the carray into.
    group: str, tables.Group
        PyTables group to insert the data node into
    node: str, tables.Node
        PyTables node of the carray.  If it already exists it will remove
        the existing node and create a new one.
    data: ndarray
        Data to be inserted into the node carray
    kwargs:
        Extra keyword arguments to be passed to the
        tables.File.create_carray method.

    Returns
    -------
    tables.carray
        Pointer to the created carray object.
    """
    assert(type(h5file) == tb.File)

    # Switch to string
    if type(group) != str:
        group = group._v_pathname

    # Join path for node existence check
    if group[-1] == '/':
        node_path = group + node
    else:
        node_path = '/'.join((group, node))

    # Check existence and remove if necessary
    if h5file.__contains__(node_path):
        h5file.remove_node(node_path)

    out_arr = h5file.create_carray(group,
                                   node,
                                   atom=tb.Atom.from_dtype(data.dtype),
                                   shape=data.shape,
                                   **kwargs)
    out_arr[:] = data
    return out_arr


def empty_hdf5_carray(h5file, group, node, in_atom, shape, **kwargs):
    """
    Create an empty PyTables carray.  Replaces node if it already exists.

    Parameters
    ----------
    h5file: tables.File
        Writeable HDF5 file to insert the carray into.
    group: str, tables.Group
        PyTables group to insert the data node into
    node: str, tables.Node
        PyTables node of the carray.  If it already exists it will remove
        the existing node and create a new one.
    in_atom: tables.Atom
        Atomic datatype and chunk size for the carray.
    shape: tuple, list
        Shape of empty carray to be created.
    kwargs:
        Extra keyword arguments to be passed to the
        tables.File.create_carray method.

    Returns
    -------
    tables.carray
        Pointer to the created carray object.
    """
    assert(type(h5file) == tb.File)

    # Switch to string
    if type(group) == tb.Group:
        group = group._v_pathname

    # Join path for node existence check
    if group[-1] == '/':
        node_path = group + node
    else:
        node_path = '/'.join((group, node))

    # Check existence and remove if necessary
    if h5file.__contains__(node_path):
        h5file.remove_node(node_path)

    out_arr = h5file.create_carray(group,
                                   node,
                                   atom=in_atom,
                                   shape=shape,
                                   **kwargs)
    return out_arr
