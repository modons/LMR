"""
Module: LMR_utils.py

Purpose: Contains miscellaneous "utility" functions used throughout the LMR code, 
         including verification modules.

Originators: Greg Hakim & Robert Tardif | U. of Washington
                                        | March 2015

Revisions: 
          - Added the get_distance function for more efficient calculation of distances
            between lat/lon points on the globe. [R. Tardif, U. of Washington, January 2016]
          - Added fexibility in handling missing data in the global_hemispheric_means function.
            [R. Tardif, U. of Washington, Aug. 2017]
          - Added the option to save full fields of variables. [M. Erb, U. Southern California,
            June 2017]
          - Renamed the proxy databases to less-confusing convention. 
            'pages' renamed as 'PAGES2kv1' and 'NCDC' renamed as 'LMRdb'
            [R. Tardif, U. of Washington, Sept 2017]
"""
import glob
import os
import numpy as np
import re
import pickle
import collections
import copy
import ESMF
from time import time
from os.path import join
from math import radians, cos, sin, asin, sqrt
from scipy import signal, special
from scipy.spatial import cKDTree
from spharm import Spharmt, getspecindx, regrid

def atoi(text):
    try:
        return int(text)
    except ValueError:
        return text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''    
    return [ atoi(c) for c in re.split('([-]?\d+)', text) ]

def natural_sort(input):
    return sorted(input, key=natural_keys)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = list(map(np.radians, [lon1, lat1, lon2, lat2]))
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km

def get_distance(lon_pt, lat_pt, lon_ref, lat_ref):
    """
    Vectorized calculation the great circle distances between lat-lon points 
    on the Earth (lat/lon are specified in decimal degrees)

    Input:
    lon_pt , lat_pt  : longitude, latitude of site w.r.t. which distances 
                       are to be calculated. Both should be scalars.
    lon_ref, lat_ref : longitudes, latitudes of reference field
                       (e.g. calibration dataset, reconstruction grid)
                       May be scalar, 1D arrays, or 2D arrays.

    Output: Returns array containing distances between (lon_pt, lat_pt) and all other points
            in (lon_ref,lat_ref). Array has dimensions [dim(lon_ref),dim(lat_ref)].

    Originator: R. Tardif, Atmospheric sciences, U. of Washington, January 2016

    """

    # Convert decimal degrees to radians 
    lon_pt, lat_pt, lon_ref, lat_ref = list(map(np.radians, [lon_pt, lat_pt, lon_ref, lat_ref]))

    # check dimension of lon_ref and lat_ref
    dims_ref = len(lon_ref.shape)

    if dims_ref == 0:
        lats = lat_ref
        lons = lon_ref
    elif dims_ref == 1:
        lon_dim = len(lon_ref)
        lat_dim = len(lat_ref)
        nbpts = lon_dim*lat_dim
        lats = np.array([lat_ref,]*lon_dim).transpose()
        lons = np.array([lon_ref,]*lat_dim)
    elif dims_ref == 2:
        lats = lat_ref
        lons = lon_ref
    else:
        print('ERROR in get_distance!')
        raise SystemExit()
        
    # Haversine formula using arrays as input
    dlon = lons - lon_pt 
    dlat = lats - lat_pt 

    a = np.sin(dlat/2.)**2 + np.cos(lat_pt) * np.cos(lats) * np.sin(dlon/2.)**2
    c = 2. * np.arcsin(np.sqrt(a))
    km = 6367.0 * c

    return km


def get_data_closest_gridpt(data,lon_data,lat_data,lon_pt,lat_pt,getvalid=None):
    """
        Extracts data from a gridded field (data) at the gridpt closest to the 
        reference location (lat_pt, lon_pt) which has valid (i.e. non-NaN) values.

        Parameters
        ----------
        data: ndarray
            Gridded data matching dimensions of (sample, lat, lon) -> gridded form
            or (lat*lon, sample) -> state vector form.
        lon_data: ndarray
            Longitudes pertaining to the input data.  Can have shape as a single
            vector (lat), a grid (lat, lon), or a flattened grid (lat*lon).
        lat_data: ndarray
            Latitudes pertaining to the input data. Can have shape as a single
            vector (lon), a grid (lat, lon), or a flattened grid (lat*lon).
        lon_pt: float
            Longitude of the reference point (proxy site or other).
        lat_pt: float
            Latitude of the reference point (proxy site or other).
        getvalid: bool
            ...

        Returns
        -------
        pt_data: ndarray
            Grid point data closest to the lat/lon of the reference location.
            Dimension of returned array is 1D along sample axis).

       Originator: R. Tardif, Atmospheric sciences, U. of Washington, December 2016

    """
    # Initialize returned array to missing (nan)
    # sample axis is assumed to be axis=0
    pt_data = np.zeros(shape=data.shape[0])
    pt_data[:] = np.nan

    # Representative distance between grid pts.
    dist = haversine(np.roll(lon_data,1), np.roll(lat_data,1), lon_data, lat_data)
    meandist =  np.mean(dist)
    
    # Calculate distances
    dist = haversine(lon_pt, lat_pt, lon_data, lat_data)
    workdist = np.ravel(dist)
    sorteddist = np.sort(workdist)

    # Find closest grid point with *valid* data.
    # Impose max of search distance equal to twice the representative distance
    # Start with min dist.

    distptmin = sorteddist[0]
    i = 0
    distpt = distptmin
    inds2 = np.where(workdist == distpt)[0][0]
    inds = np.unravel_index(inds2, dist.shape)


    if getvalid:
        valid = False
        while not valid and distpt < meandist*2.:

            inds2 = np.where(workdist == distpt)[0][0]
            inds = np.unravel_index(inds2, dist.shape)

            if len(inds) == 2:
                test_valid = np.isfinite(data[:,inds[0],inds[1]])
            elif len(inds) == 1:
                #test_valid = np.isfinite(data[:,inds])
                test_valid = np.isfinite(data[inds,:])
            else:
                print('ERROR in distance calc in get_data_closest_gridpt!')
                raise SystemExit(1)            
            if np.all(test_valid): valid = True
            i +=1
            distpt = sorteddist[i]

    else:
        pass

    # Extract the data at the identified grid pt.
    if len(inds) == 2:
        pt_data = data[:,inds[0],inds[1]]
    elif len(inds) == 1:
        #pt_data = data[:,inds[0]]
        pt_data = data[inds[0],:]
    else:
        pass

    return pt_data


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

def ensemble_stats(workdir, y_assim, y_eval=None, write_posterior_Ye=False, save_full_field=False):
    """
    Compute the ensemble mean and variance for files in the input directory

    Originator: Greg Hakim
                University of Washington
                May 2015
    
      Revised 16 June 2015 (GJH)
      Revised 24 June 2015 (R Tardif, UW)
              : func. renamed from ensemble_mean to ensemble_stats
              : computes and output the ensemble variance as well
              : now handles state vector possibly containing multiple variables
      Revised 15 July 2015 (R Tardif, UW)
              : extracts Ye's from augmented state vector (Ye=HXa), match with corresponding
                proxy sites from master list of proxies and output to analysis_Ye.pckl file
      Rrevised June 2016 (Michael Erb, USC)
              : Fixed a bug where the analysis Ye values were not being
                indexed by year properly in forming analysis_Ye.pckl
      Revised February 2017 (R Tardif, UW)
              : Added flexibility by getting rid of originally hard-coded features.
              : Added use of new "natural_sort" function to sort filenames composed with negative years.
      Revised August 2017 (G. Hakim, UW)
              : Added boolean flag to control the generation of analysis_Ye.pckl file.
      Revised August 2017 (M. Erb; G. Hakim git port)
              : added full-ensemble saving facility
      Revised August 2017 (R. Tardif, UW)
              : Modified load_precalculated_ye_vals_psm_per_proxy function to upload Ye values
                either from assimilated or withheld proxy sets (R. Tardif, UW)

    """
    
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
    sfiles = natural_sort(files)    
    nyears = len(sfiles)

    # loop on state variables 
    for var in state_info.keys():

        print('State variable:', var)

        ibeg = state_info[var]['pos'][0]
        iend = state_info[var]['pos'][1]

        # Determine variable type (2D lat/lon, 2D lat/depth, time series etc ...)
        # Look for the 'vartype' entry in the state vector dict.
        # Possibilities are: '2D:horizontal', '2D:meridional_vertical', '0D:time series'
        if state_info[var]['vartype'] == '2D:horizontal': 
            # 2D:horizontal variable
            ndim1 = state_info[var]['spacedims'][0]
            ndim2 = state_info[var]['spacedims'][1]

            # Prior
            Xb = np.reshape(Xbtmp[ibeg:iend+1,:],(ndim1,ndim2,nens))
            xbm = np.mean(Xb,axis=2)       # ensemble mean
            xbv = np.var(Xb,axis=2,ddof=1) # ensemble variance

            # Process the **analysis** (i.e. posterior) files
            years = []
            xa_ens = np.zeros([nyears,ndim1,ndim2,nens])
            xam = np.zeros([nyears,ndim1,ndim2])
            xav = np.zeros([nyears,ndim1,ndim2],dtype=np.float64)
            k = -1
            for f in sfiles:
                k = k + 1
                i = f.rfind('year')
                fname_end = f[i+4:]
                ii = fname_end.rfind('.')
                year = fname_end[:ii]
                years.append(year)
                Xatmp = np.load(f)
                Xa = np.reshape(Xatmp[ibeg:iend+1,:],(ndim1,ndim2,nens))
                xa_ens[k,:,:,:] = Xa                  # total ensemble
                xam[k,:,:] = np.mean(Xa,axis=2)       # ensemble mean
                xav[k,:,:] = np.var(Xa,axis=2,ddof=1) # ensemble variance


            # form dictionary containing variables to save, including info on array dimensions
            coordname1 = state_info[var]['spacecoords'][0]
            coordname2 = state_info[var]['spacecoords'][1]
            dimcoord1 = 'n'+coordname1
            dimcoord2 = 'n'+coordname2

            coord1 = np.reshape(Xb_coords[ibeg:iend+1,0],(state_info[var]['spacedims'][0],state_info[var]['spacedims'][1]))
            coord2 = np.reshape(Xb_coords[ibeg:iend+1,1],(state_info[var]['spacedims'][0],state_info[var]['spacedims'][1]))

            vars_to_save_ens  = {'nens':nens, 'years':years, dimcoord1:state_info[var]['spacedims'][0], dimcoord2:state_info[var]['spacedims'][1], \
                                     coordname1:coord1, coordname2:coord2, 'xb_ens':Xb, 'xa_ens':xa_ens}
            vars_to_save_mean = {'nens':nens, 'years':years, dimcoord1:state_info[var]['spacedims'][0], dimcoord2:state_info[var]['spacedims'][1], \
                                     coordname1:coord1, coordname2:coord2, 'xbm':xbm, 'xam':xam}
            vars_to_save_var  = {'nens':nens, 'years':years, dimcoord1:state_info[var]['spacedims'][0], dimcoord2:state_info[var]['spacedims'][1], \
                                     coordname1:coord1, coordname2:coord2, 'xbv':xbv, 'xav':xav}


        elif state_info[var]['vartype'] == '2D:meridional_vertical': 

            ndim1 = state_info[var]['spacedims'][0]
            ndim2 = state_info[var]['spacedims'][1]

            # Prior
            Xb = np.reshape(Xbtmp[ibeg:iend+1,:],(ndim1,ndim2,nens))
            xbm = np.mean(Xb,axis=2)       # ensemble mean
            xbv = np.var(Xb,axis=2,ddof=1) # ensemble variance

            # Process the **analysis** (i.e. posterior) files
            years = []
            xa_ens = np.zeros([nyears,ndim1,ndim2,nens])
            xam = np.zeros([nyears,ndim1,ndim2])
            xav = np.zeros([nyears,ndim1,ndim2],dtype=np.float64)
            k = -1
            for f in sfiles:
                k = k + 1
                i = f.rfind('year')
                fname_end = f[i+4:]
                ii = fname_end.rfind('.')
                year = fname_end[:ii]
                years.append(year)
                Xatmp = np.load(f)
                Xa = np.reshape(Xatmp[ibeg:iend+1,:],(ndim1,ndim2,nens))
                xam[k,:,:,:] = Xa                     # total ensemble
                xam[k,:,:] = np.mean(Xa,axis=2)       # ensemble mean
                xav[k,:,:] = np.var(Xa,axis=2,ddof=1) # ensemble variance


            # form dictionary containing variables to save, including info on array dimensions
            coordname1 = state_info[var]['spacecoords'][0]
            coordname2 = state_info[var]['spacecoords'][1]
            dimcoord1 = 'n'+coordname1
            dimcoord2 = 'n'+coordname2

            coord1 = np.reshape(Xb_coords[ibeg:iend+1,0],(state_info[var]['spacedims'][0],state_info[var]['spacedims'][1]))
            coord2 = np.reshape(Xb_coords[ibeg:iend+1,1],(state_info[var]['spacedims'][0],state_info[var]['spacedims'][1]))

            vars_to_save_ens  = {'nens':nens, 'years':years, dimcoord1:state_info[var]['spacedims'][0], dimcoord2:state_info[var]['spacedims'][1], \
                                     coordname1:coord1, coordname2:coord2, 'xb_ens':Xb, 'xa_ens':xa_ens}
            vars_to_save_mean = {'nens':nens, 'years':years, dimcoord1:state_info[var]['spacedims'][0], dimcoord2:state_info[var]['spacedims'][1], \
                                     coordname1:coord1, coordname2:coord2, 'xbm':xbm, 'xam':xam}
            vars_to_save_var  = {'nens':nens, 'years':years, dimcoord1:state_info[var]['spacedims'][0], dimcoord2:state_info[var]['spacedims'][1], \
                                     coordname1:coord1, coordname2:coord2, 'xbv':xbv, 'xav':xav}


        elif state_info[var]['vartype'] == '1D:meridional': 

            ndim1 = state_info[var]['spacedims'][0]

            # Prior
            Xb = np.reshape(Xbtmp[ibeg:iend+1,:],(ndim1,nens))
            xbm = np.mean(Xb,axis=1)       # ensemble mean
            xbv = np.var(Xb,axis=1,ddof=1) # ensemble variance

            # Process the **analysis** (i.e. posterior) files
            years = []
            xa_ens = np.zeros([nyears,ndim1,nens])
            xam = np.zeros([nyears,ndim1])
            xav = np.zeros([nyears,ndim1],dtype=np.float64)
            k = -1
            for f in sfiles:
                k = k + 1
                i = f.rfind('year')
                fname_end = f[i+4:]
                ii = fname_end.rfind('.')
                year = fname_end[:ii]
                years.append(year)
                Xatmp = np.load(f)
                Xa = np.reshape(Xatmp[ibeg:iend+1,:],(ndim1,nens))
                xa_ens[k,:,:] = Xa                  # total ensemble
                xam[k,:] = np.mean(Xa,axis=1)       # ensemble mean
                xav[k,:] = np.var(Xa,axis=1,ddof=1) # ensemble variance

            # form dictionary containing variables to save, including info on array dimensions
            coordname1 = state_info[var]['spacecoords'][0]
            dimcoord1 = 'n'+coordname1

            coord1 = np.reshape(Xb_coords[ibeg:iend+1,0],(state_info[var]['spacedims'][0]))

            vars_to_save_ens  = {'nens':nens, 'years':years, dimcoord1:state_info[var]['spacedims'][0], \
                                     coordname1:coord1, 'xb_ens':Xb, 'xa_ens':xa_ens}
            vars_to_save_mean = {'nens':nens, 'years':years, dimcoord1:state_info[var]['spacedims'][0], \
                                     coordname1:coord1, 'xbm':xbm, 'xam':xam}
            vars_to_save_var  = {'nens':nens, 'years':years, dimcoord1:state_info[var]['spacedims'][0], \
                                     coordname1:coord1, 'xbv':xbv, 'xav':xav}
        
        elif state_info[var]['vartype'] == '0D:time series': 
            # 0D:time series variable (no spatial dims)
            Xb = Xbtmp[ibeg:iend+1,:] # prior (full) ensemble
            xbm = np.mean(Xb,axis=1)  # ensemble mean
            xbv = np.var(Xb,axis=1,ddof=1)   # ensemble variance

            # process the **analysis** files
            years = []
            xa_ens = np.zeros((nyears, Xb.shape[1]))
            xam = np.zeros([nyears])
            xav = np.zeros([nyears],dtype=np.float64)
            k = -1
            for f in sfiles:
                k = k + 1
                i = f.rfind('year')
                fname_end = f[i+4:]
                ii = fname_end.rfind('.')
                year = fname_end[:ii]
                years.append(year)
                Xatmp = np.load(f)
                Xa = Xatmp[ibeg:iend+1,:]
                xa_ens[k] = Xa  # total ensemble
                xam[k] = np.mean(Xa,axis=1) # ensemble mean
                xav[k] = np.var(Xa,axis=1,ddof=1)  # ensemble variance

            vars_to_save_ens = {'nens':nens, 'years':years, 'xb_ens':Xb, 'xa_ens':xa_ens}
            vars_to_save_mean = {'nens':nens, 'years':years, 'xbm':xbm, 'xam':xam}
            vars_to_save_var  = {'nens':nens, 'years':years, 'xbv':xbv, 'xav':xav}

            # (full) ensemble to file for this variable type
            filen = workdir + '/ensemble_full_' + var
            print('writing the new ensemble file' + filen)
            np.savez(filen, **vars_to_save_ens)


        else: 
            print('ERROR in ensemble_stats: Variable of unrecognized (space) dimensions! Skipping variable:', var)
            continue


        if (state_info[var]['vartype'] != '0D:time series') and save_full_field:
            filen = workdir + '/ensemble_full_' + var
            print('writing the new ensemble file' + filen)
            np.savez(filen, **vars_to_save_ens)

        # --- Write data to files ---
        # ens. mean to file
        filen = workdir + '/ensemble_mean_' + var
        print('writing the new ensemble mean file...' + filen)
        np.savez(filen, **vars_to_save_mean)

        # ens. variance to file
        filen = workdir + '/ensemble_variance_' + var
        print('writing the new ensemble variance file...' + filen)
        np.savez(filen, **vars_to_save_var)


    
    # --------------------------------------------------------
    # Extract the analyzed Ye ensemble for diagnostic purposes
    # --------------------------------------------------------
    if write_posterior_Ye:

        # get information on dim of state without the Ye's (before augmentation)
        stateDim  = npzfile['stateDim']
        Xbtmp_aug = npzfile['Xb_one_aug']
        # dim of entire state vector (augmented)
        totDim = Xbtmp_aug.shape[0]
        nbye = (totDim - stateDim)
        Ye_s = np.zeros([nbye,nyears,nens])

        # Loop over **analysis** files & extract the Ye's
        years = []
        for k, f in enumerate(sfiles):
            i = f.rfind('year')
            fname_end = f[i+4:]
            ii = fname_end.rfind('.')
            year = fname_end[:ii]
            years.append(float(year))
            Xatmp = np.load(f)
            # Extract the Ye's from augmented state (beyond stateDim 'til end of
            # state vector).
            # RT Aug 2017: These now include Ye's from assimilated AND withheld proxies.
            Ye_s[:, k, :] = Xatmp[stateDim:, :]
        years = np.array(years)

        int_recon = years[1] - years[0]

        # Build dictionary
        YeDict = {}
        # loop over assimilated proxies
        nbassim = 0
        for i, pobj in enumerate(y_assim):
            # build boolean of indices to pull from HXa
            if int_recon == 1.:
                yr_idxs = np.array([year in pobj.time for year in years],
                                   dtype=np.bool)
            else:
                yr_idxs = np.zeros([len(years)],dtype=np.bool) # init. w/ all False
                for k in range(len(years)):
                    if [ pyr for pyr in pobj.time if ((pyr > years[k]-int_recon/2.)
                                                      and (pyr <= years[k]+int_recon/2.)) ]:
                        yr_idxs[k] = True
                    else:
                        yr_idxs[k] = False
            
            YeDict[(pobj.type, pobj.id)] = {}
            YeDict[(pobj.type, pobj.id)]['status'] = 'assimilated'
            YeDict[(pobj.type, pobj.id)]['lat'] = pobj.lat
            YeDict[(pobj.type, pobj.id)]['lon'] = pobj.lon
            YeDict[(pobj.type, pobj.id)]['R'] = pobj.psm_obj.R
            #YeDict[(pobj.type, pobj.id)]['years'] = pobj.time
            YeDict[(pobj.type, pobj.id)]['years'] = years[yr_idxs]
            YeDict[(pobj.type, pobj.id)]['HXa'] = Ye_s[i, yr_idxs, :]
            YeDict[(pobj.type, pobj.id)]['augStateIndex'] = i
            
            nbassim +=1

        # loop over non-assimilated (withheld) proxies
        if y_eval:
            for j, pobj in enumerate(y_eval):
                # build boolean of indices to pull from HXa
                if int_recon == 1.:
                    yr_idxs = np.array([year in pobj.time for year in years],
                                       dtype=np.bool)
                    ye_years = pobj.time
                else:
                    yr_idxs = np.zeros([len(years)],dtype=np.bool) # init. w/ all False
                    for k in range(len(years)):
                        if [ pyr for pyr in pobj.time if ((pyr > years[k]-int_recon/2.)
                                                          and (pyr <= years[k]+int_recon/2.)) ]:
                            yr_idxs[k] = True
                        else:
                            yr_idxs[k] = False

                YeDict[(pobj.type, pobj.id)] = {}
                YeDict[(pobj.type, pobj.id)]['status'] = 'withheld'
                YeDict[(pobj.type, pobj.id)]['lat'] = pobj.lat
                YeDict[(pobj.type, pobj.id)]['lon'] = pobj.lon
                YeDict[(pobj.type, pobj.id)]['R'] = pobj.psm_obj.R
                #YeDict[(pobj.type, pobj.id)]['years'] = pobj.time
                YeDict[(pobj.type, pobj.id)]['years'] = years[yr_idxs]
                YeDict[(pobj.type, pobj.id)]['HXa'] = Ye_s[nbassim+j, yr_idxs, :]
                YeDict[(pobj.type, pobj.id)]['augStateIndex'] = nbassim+j
            
        # Dump dictionary to pickle file
        # using protocol 2 for more efficient storing
        outfile = open('{}/analysis_Ye.pckl'.format(workdir), 'wb')
        pickle.dump(YeDict, outfile, protocol=2)
        outfile.close()

    return


def lon_lat_to_cartesian(lon, lat, R=6371.):
    """

    """
    
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)

    return x, y, z

def regrid_simple(Nens,X,X_coords,ind_lat,ind_lon,ntrunc):
    """
    Truncate lat,lon grid to another resolution using local distance-weighted 
    averages. 

    Inputs:
    Nens            : number of ensemble members
    X               : data array of shape (nlat*nlon,Nens) 
    X_coords        : array of lat-lon coordinates of variable contained in X
                      w/ shape (nlat*nlon,2)
    ind_lat         : array index (column) at which X_ccords contains latitudes
    ind_lon         : array index (column) at which X_ccords contains longitudes
    ntrunc          : triangular truncation (e.g., use 42 for T42)

    Outputs :
    lat_new : 2D latitude array on the new grid (nlat_new,nlon_new)
    lon_new : 2D longitude array on the new grid (nlat_new,nlon_new)
    X_new   : truncated data array of shape (nlat_new*nlon_new, Nens)
    
    Originator: Robert Tardif
                University of Washington
                March 2017
    """
        
    # truncate to a lower resolution grid (triangular truncation)
    ifix = np.remainder(ntrunc,2.0).astype(int)
    nlat_new = ntrunc + ifix
    nlon_new = int(nlat_new*1.5)

    # create new lat,lon grid arrays
    # Note: AP - According to github.com/jswhit/pyspharm documentation the
    #  latitudes will not include the equator or poles when nlats is even.
    # TODO: Decide whether this should be tied to spherical regridding grids
    if nlat_new % 2 == 0:
        include_poles = False
    else:
        include_poles = True

    lat_new, lon_new, _, _ = generate_latlon(nlat_new, nlon_new,
                                             include_endpts=include_poles)

    # cartesian coords of target grid
    xt,yt,zt = lon_lat_to_cartesian(lon_new.flatten(), lat_new.flatten())

    # cartesian coords of source grid
    lats = X_coords[:, ind_lat]
    lons = X_coords[:, ind_lon]
    xs,ys,zs = lon_lat_to_cartesian(lons, lats)

    # cKDtree object of source grid
    tree = cKDTree(list(zip(xs,ys,zs)))

    # inverse distance weighting (N pts)
    N = 20
    fracvalid = 0.7
    d, inds = tree.query(list(zip(xt,yt,zt)), k=N)
    L = 200.
    w = np.exp(-np.square(d)/np.square(L))

    # transform each ensemble member, one at a time
    X_new = np.zeros([nlat_new*nlon_new,Nens])
    X_new[:] = np.nan
    for k in range(Nens):
        tmp = np.ma.masked_invalid(X[:,k][inds])
        mask = tmp.mask

        # apply tmp mask to weights array
        w = np.ma.masked_where(np.ma.getmask(tmp),w)
        
        # compute weighted-average of surrounding data
        datagrid = np.sum(w*tmp, axis=1)/np.sum(w, axis=1)

        # keep track of valid data involved in averges  
        nbvalid = np.sum(~mask,axis=1)
        nonvalid = np.where(nbvalid < int(fracvalid*N))[0]

        # make sure to mask grid points where too few valid data were used
        datagrid[nonvalid] = np.nan
        X_new[:,k] = datagrid

    # make sure a masked array is returned, if at
    # least one invalid data is found
    if np.isnan(X_new).any():
        X_new = np.ma.masked_invalid(X_new)
    
    return X_new,lat_new,lon_new


def regrid_esmpy(target_nlat, target_nlon, X_nens, X, X_lat2D, X_lon2D, X_nlat,
                 X_nlon, method='bilinear'):
    """
    Regrid field to target resolution using the ESMF package.
    
    Parameters
    ----------
    target_nlat: int
        number of latitude points for destination grid
    target_nlon: int
        number of longitude points for the destination grid
    X_nens: int
        number of ensemble members in the data array
    X: ndarray
        data array to be regridded of shape (nlat*nlon, nens)
    X_lat2D: ndarray
        2D array of latitudes corresponding to the source field of shape (
        nlat, nlon)
    X_lon2D: ndarray
        2D array of longitudes corresponding to the source field of shape (
        nlat, nlon)
    X_nlat: int
        number of latitude points in the source grid
    X_nlon: int
        number of longitude points in the source grid
    method: str
        Regridding method to use. Valid options include 'bilinear' and 'patch'

    Returns
    -------    
    X_new:  
        truncated data array of shape (nlat_new*nlon_new, Nens)
    lat_new: ndarray
        2D latitude array on the new grid (nlat_new,nlon_new)
    lon_new: ndarray
        2D longitude array on the new grid (nlat_new,nlon_new)
        
    Notes
    -----
    This regridding function supports masked grids

    """

    lons_1D = X_lon2D[0]
    lon_diff_negative = np.diff(lons_1D) < 0
    if lon_diff_negative.sum() > 1:
        raise ValueError('Expected monotonic value increase with index '
                         '(excluding cyclic point) for input longitudes.')
    # adjust cyclinc longitude point to boundaries if necessary
    elif lon_diff_negative.sum() == 1:
        cyclic_idx_start, = np.where(lon_diff_negative)
        lon_shift = -(cyclic_idx_start+1)
        X_lon2D = np.roll(X_lon2D, lon_shift, axis=1)
    else:
        lon_shift = None

    # Create grid for the input state data
    grid = ESMF.Grid(max_index=np.array((X_nlon, X_nlat)),
                     num_peri_dims=1,
                     pole_dim=1,
                     coord_sys=ESMF.CoordSys.SPH_DEG,
                     coord_typekind=ESMF.TypeKind.R8,
                     staggerloc=ESMF.StaggerLoc.CENTER)

    # Add grid cell center coordinates
    [x, y] = (0, 1)
    grid_x_center = grid.get_coords(x)
    grid_x_center[:] = X_lon2D.T
    grid_y_center = grid.get_coords(y)
    grid_y_center[:] = X_lat2D.T

    # Add grid cell corner boundaries
    lat_bnds, lon_bnds = calculate_latlon_bnds(X_lat2D[:,0], X_lon2D[0])
    grid.add_coords(staggerloc=ESMF.StaggerLoc.CORNER)
    grid_x_corner = grid.get_coords(x, staggerloc=ESMF.StaggerLoc.CORNER)
    # Cutoff last element because grid assumed periodic, this might be
    # erroneous if we aren't using full global grids.
    grid_x_corner[:] = lon_bnds[:-1, None]
    grid_y_corner = grid.get_coords(y, staggerloc=ESMF.StaggerLoc.CORNER)
    grid_y_corner[:] = lat_bnds[None, :]

    # check for masked values
    masked_regrid = hasattr(X, 'mask')

    if masked_regrid:
        print('Mask detected.  Adding mask to src ESMF grid')
        grid_mask = grid.add_item(ESMF.GridItem.MASK)
        X_mask = X[:, 0].mask.astype(np.int16)
        X_mask = X_mask.reshape(X_nlat, X_nlon)

        if lon_shift:
            X_mask = np.roll(X_mask, lon_shift, axis=1)

        grid_mask[:] = X_mask.T
        mask_values = np.array([1])
    else:
        mask_values = None

    # Create new grid
    new_grid = ESMF.Grid(max_index=np.array((target_nlon, target_nlat)),
                         num_peri_dims=1,
                         pole_dim=1,
                         coord_sys=ESMF.CoordSys.SPH_DEG,
                         coord_typekind=ESMF.TypeKind.R8,
                         staggerloc=ESMF.StaggerLoc.CENTER)

    [new_lat, new_lon,
     new_lat_bnds, new_lon_bnds] = generate_latlon(target_nlat, target_nlon)
    new_grid_x_cen = new_grid.get_coords(x)
    new_grid_x_cen[:] = new_lon.T
    new_grid_y_cen = new_grid.get_coords(y)
    new_grid_y_cen[:] = new_lat.T
    new_grid.add_coords(staggerloc=ESMF.StaggerLoc.CORNER)
    new_grid_x_cor = new_grid.get_coords(x, staggerloc=ESMF.StaggerLoc.CORNER)
    # Cut off last element of lon_bnds because grid assumed periodic
    new_grid_x_cor[:] = new_lon_bnds[:-1, None]
    new_grid_y_cor = new_grid.get_coords(y, staggerloc=ESMF.StaggerLoc.CORNER)
    new_grid_y_cor[:] = new_lat_bnds[None, :]

    new_grid_mask = new_grid.add_item(ESMF.GridItem.MASK)
    new_grid_mask[:] = 0

    if method == 'bilinear':
        use_method = ESMF.RegridMethod.BILINEAR
    # Note: Removed because conservative regridding does not work for irreg
    # grids currently (discovered with rotated pole ocean grids)
    # elif method == 'conserve':
    #     use_method = ESMF. RegridMethod.CONSERVE
    elif method == 'patch':
        use_method = ESMF.RegridMethod.PATCH
    else:
        raise ValueError('Regridding method \'{}\''
                         ' in regrid_esmpy does not exist.'.format(method))

    # create src and dst fields
    src_field = ESMF.Field(grid, name='src', staggerloc=ESMF.StaggerLoc.CENTER)
    dst_field = ESMF.Field(new_grid, name='dst',
                           staggerloc=ESMF.StaggerLoc.CENTER)

    regrid_output = np.empty((target_nlat * target_nlon, X_nens))
    regridder = ESMF.Regrid(src_field, dst_field, regrid_method=use_method,
                            src_mask_values=mask_values,
                            unmapped_action=ESMF.UnmappedAction.IGNORE)

    # Regrid each ensemble member
    for k in range(X_nens):
        grid_data = X[:,k].reshape(X_nlat, X_nlon)

        if lon_shift:
            grid_data = np.roll(grid_data, lon_shift, axis=1)

        src_field.data[:] = grid_data.T

        regridder(src_field, dst_field)

        out_data = dst_field.data[:].T

        # Check for masked values on new grid
        if masked_regrid:
            out_mask = out_data == 0.0  # if it's exactly zero, it's masked
            out_data[out_mask] = np.nan

        regrid_output[:, k] = out_data.flatten()

    if masked_regrid:
        regrid_output = np.ma.masked_invalid(regrid_output)

    # Clean up objects
    src_field.destroy()
    dst_field.destroy()
    grid.destroy()
    new_grid.destroy()
    regridder.destroy()

    return regrid_output, new_lat, new_lon

    
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
    
    Originator: Greg Hakim
                University of Washington
                May 2015
    """

    
    # create the spectral object on the original grid
    specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')

    # truncate to a lower resolution grid (triangular truncation)
    ifix = np.remainder(ntrunc,2.0).astype(int)
    nlat_new = ntrunc + ifix
    nlon_new = int(nlat_new*1.5)

    # create the spectral object on the new grid
    specob_new = Spharmt(nlon_new,nlat_new,gridtype='regular',legfunc='computed')

    # create new lat,lon grid arrays
    # Note: AP - According to github.com/jswhit/pyspharm documentation the
    #  latitudes will not include the equator or poles when nlats is even.
    if nlat_new % 2 == 0:
        include_poles = False
    else:
        include_poles = True

    lat_new, lon_new, _, _ = generate_latlon(nlat_new, nlon_new,
                                             include_endpts=include_poles)

    # transform each ensemble member, one at a time
    X_new = np.zeros([nlat_new*nlon_new,Nens])
    for k in range(Nens):
        X_lalo = np.reshape(X[:,k],(nlat,nlon))
        Xbtrunc = regrid(specob_lmr, specob_new, X_lalo, ntrunc=nlat_new-1, smooth=None)
        vectmp = Xbtrunc.flatten()
        X_new[:,k] = vectmp

    return X_new,lat_new,lon_new


def generate_latlon(nlats, nlons, lat_bnd=(-90,90), lon_bnd=(0, 360),
                    include_endpts=False):
    """
    Generate regularly spaced latitude and longitude arrays where each point 
    is the center of the respective grid cell.
    
    Parameters
    ----------
    nlats: int
        Number of latitude points
    nlons: int
        Number of longitude points
    lat_bnd: tuple(float), optional
        Bounding latitudes for gridcell edges (not centers).  Accepts values 
        in range of [-90, 90].
    lon_bnd: tuple(float), optional
        Bounding longitudes for gridcell edges (not centers).  Accepts values 
        in range of [-180, 360].
    include_endpts: bool
        Include the poles in the latitude array.  

    Returns
    -------
    lat_center_2d:
        Array of central latitide points (nlat x nlon)
    lon_center_2d:
        Array of central longitude points (nlat x nlon)
    lat_corner:
        Array of latitude boundaries for all grid cells (nlat+1)
    lon_corner:
        Array of longitude boundaries for all grid cells (nlon+1)   

    """
    if len(lat_bnd) != 2 or len(lon_bnd) != 2:
        raise ValueError('Bound tuples must be of length 2')
    if np.any(np.diff(lat_bnd) < 0) or np.any(np.diff(lon_bnd) < 0):
        raise ValueError('Lower bounds must be less than upper bounds.')
    if np.any(abs(np.array(lat_bnd)) > 90):
        raise ValueError('Latitude bounds must be between -90 and 90')
    if np.any(abs(np.diff(lon_bnd)) > 360):
        raise ValueError('Longitude bound difference must not exceed 360')
    if np.any(np.array(lon_bnd) < -180) or np.any(np.array(lon_bnd) > 360):
        raise ValueError('Longitude bounds must be between -180 and 360')

    lon_center = np.linspace(lon_bnd[0], lon_bnd[1], nlons, endpoint=False)

    if include_endpts:
        lat_center = np.linspace(lat_bnd[0], lat_bnd[1], nlats)
    else:
        tmp = np.linspace(lat_bnd[0], lat_bnd[1], nlats+1)
        lat_center = (tmp[:-1] + tmp[1:]) / 2.

    lon_center_2d, lat_center_2d = np.meshgrid(lon_center, lat_center)
    lat_corner, lon_corner = calculate_latlon_bnds(lat_center, lon_center)

    return lat_center_2d, lon_center_2d, lat_corner, lon_corner


def calculate_latlon_bnds(lats, lons):
    """
    Calculate the bounds for regularly gridded lats and lons.
    
    Parameters
    ----------
    lats: ndarray
        Regularly spaced latitudes.  Must be 1-dimensional and monotonically 
        increase with index.
    lons:  ndarray
        Regularly spaced longitudes.  Must be 1-dimensional and monotonically 
        increase with index.

    Returns
    -------
    lat_bnds:
        Array of latitude boundaries for each input latitude of length 
        len(lats)+1.
    lon_bnds:
        Array of longitude boundaries for each input longitude of length
        len(lons)+1.

    """
    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError('Expected 1D-array for input lats and lons.')
    if np.any(np.diff(lats) < 0) or np.any(np.diff(lons) < 0):
        raise ValueError('Expected monotonic value increase with index for '
                         'input latitudes and longitudes')

    # Note: assumes lats are monotonic increase with index
    dlat = abs(lats[1] - lats[0]) / 2.
    dlon = abs(lons[1] - lons[0]) / 2.

    # Check that inputs are regularly spaced
    lat_space = np.diff(lats)
    lon_space = np.diff(lons)

    lat_bnds = np.zeros(len(lats)+1)
    lon_bnds = np.zeros(len(lons)+1)

    lat_bnds[1:-1] = lats[:-1] + lat_space/2.
    lat_bnds[0] = lats[0] - lat_space[0]/2.
    lat_bnds[-1] = lats[-1] + lat_space[-1]/2.

    if lat_bnds[0] < -90:
        lat_bnds[0] = -90.
    if lat_bnds[-1] > 90:
        lat_bnds[-1] = 90.

    lon_bnds[1:-1] = lons[:-1] + lon_space/2.
    lon_bnds[0] = lons[0] - lon_space[0]/2.
    lon_bnds[-1] = lons[-1] + lon_space[-1]/2.

    return lat_bnds, lon_bnds


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
        key = list(assimilated_proxies[rec].keys())[0]
        if key in ptypes:
            pc = ptypes[key]
            ptypes[key] = pc + 1
        else:
            ptypes[key] = 1
            
    return ptypes,nrecords

def coefficient_efficiency_old(ref,test):
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

def coefficient_efficiency(ref,test,valid=None):
    """
    Compute the coefficient of efficiency for a test time series, with respect to a reference time series.

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
    elif len(dims_ref) == 1: # 0D: time series
        dims = 1
    else:
        print('Problem with input array dimension! Exiting...')
        exit(1)

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

def crps(forecasts, observations):
    """
    Function to calculate the continuous ranked probability score
    of an ensemble of forecast timeseries against observations.

    Based on _crps.py in the properscoring package from TheClimateCorporation
    on Github

    Parameters
    ----------
    forecasts: ndarray
        Forecast timeseries with trailing dimensions of (..., nens, ntimes)
    observations: ndarray
        Observations for verification with a single dimension of length ntimes

    Returns
    -------
    crps: ndarray
        Continuous ranked probability score over nens and ntimes.  Has shape of
        the dimensions preceding nens (if any).

    """
    # Calculate the mean absolute error of the ensemble-mean forecast
    fcast_error = abs(forecasts - observations)
    mean_error = fcast_error.mean(axis=-2)

    # Calculate the mean forecast ensemble difference (spread)
    fcast_diff = np.expand_dims(forecasts, -3) - np.expand_dims(forecasts, -2)
    fcast_dist = abs(fcast_diff).mean(axis=(-3, -2))

    # Calculate the CRPS (cumulative over all times)
    crps = mean_error - 0.5 * fcast_dist
    crps = crps.sum(axis=-1)

    return crps

def rmsef(predictions, targets):
    """
    """
    val = np.sqrt(((predictions - targets) ** 2).mean())
    
    return val


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

def global_hemispheric_means(field,lat):

    """
     compute global and hemispheric mean valuee for all times in the input (i.e. field) array
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
    #             in calculation of spatial averages [ R. Tardif, November 2015 ]
    #           - Enhanced flexibility in the handling of missing values
    #             [ R. Tardif, Aug. 2017 ]

    # set number of times, lats, lons; array indices for lat and lon    
    if len(np.shape(field)) == 3: # time is a dimension
        ntime,nlat,nlon = np.shape(field)
        lati = 1
        loni = 2
    else: # only spatial dims
        ntime = 1
        nlat,nlon = np.shape(field)
        field = field[None,:] # add time dim of size 1 for consistent array dims
        lati = 1
        loni = 2

    # latitude weighting 
    lat_weight = np.cos(np.deg2rad(lat))
    tmp = np.ones([nlon,nlat])
    W = np.multiply(lat_weight,tmp).T

    # define hemispheres
    eqind = nlat//2

    if lat[0] > 0:
        # data has NH -> SH format
        W_NH = W[0:eqind+1]
        field_NH = field[:,0:eqind+1,:]
        W_SH = W[eqind+1:]
        field_SH = field[:,eqind+1:,:]
    else:
        # data has SH -> NH format
        W_NH = W[eqind:]
        field_NH = field[:,eqind:,:]
        W_SH = W[0:eqind]
        field_SH = field[:,0:eqind,:]

    gm  = np.zeros(ntime)
    nhm = np.zeros(ntime)
    shm = np.zeros(ntime)

    # Check for valid (non-NAN) values & use numpy average function (includes weighted avg calculation) 
    # Get arrays indices of valid values
    indok    = np.isfinite(field)
    indok_nh = np.isfinite(field_NH)
    indok_sh = np.isfinite(field_SH)
    for t in range(ntime):
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
            if indok_2d.any():
                field_2d    = np.squeeze(field[t,:,:])
                gm[t]       = np.average(field_2d[indok_2d],weights=W[indok_2d])
            else:
                gm[t] = np.nan
            # NH
            indok_nh_2d = indok_nh[t,:,:]
            if indok_nh_2d.any():
                field_nh_2d = np.squeeze(field_NH[t,:,:])
                nhm[t]      = np.average(field_nh_2d[indok_nh_2d],weights=W_NH[indok_nh_2d])
            else:
                nhm[t] = np.nan
            # SH
            indok_sh_2d = indok_sh[t,:,:]
            if indok_sh_2d.any():
                field_sh_2d = np.squeeze(field_SH[t,:,:])
                shm[t]      = np.average(field_sh_2d[indok_sh_2d],weights=W_SH[indok_sh_2d])
            else:
                shm[t] = np.nan
                
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


    return gm,nhm,shm


def regional_mask(lat,lon,southlat,northlat,westlon,eastlon):

    """
    Given vectors for lat and lon, and lat-lon boundaries for a regional domain, 
    return an array of ones and zeros, with ones located within the domain and zeros outside
    the domain as defined by the input lat,lon vectors.

    Originator: Greg Hakim
                University of Washington
                July 2017

    """

    nlat = len(lat)
    nlon = len(lon)

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
    Compute geographical spatial mean values for all times in the input (i.e. field) array. 
    Regions are defined following The PAGES2K Consortium (2013) Nature Geosciences paper, 
    Supplementary Information.

    input:  field[ntime,nlat,nlon] or field[nlat,nlon]
             lat[nlat,nlon] in degrees
             lon[nlat,nlon] in degrees

    output: rm[nregions,ntime] : regional means of "field" where nregions = 7 by definition, but could change

    uses functions: regional_mask()

    Originator: Greg Hakim
                University of Washington
                July 2017
    
    Revisions:
     
    """

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
        print('field dimensions...')
        print(np.shape(field))

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
            print('region='+str(region))
            print(rlat[region,0],rlat[region,1],rlon[region,0],rlon[region,1])
            
        # regional weighting (ones in region; zeros outside)
        mask = regional_mask(lat,lon,rlat[region,0],rlat[region,1],rlon[region,0],rlon[region,1])
        if debug:
            print('mask=')
            print(mask)

        # this is the weight mask for the regional domain    
        Wmask = np.multiply(mask,W)

        # make sure data starts at South Pole
        if lat[0] > 0:
            # data has NH -> SH format; reverse
            field = np.flipud(field)

        # Check for valid (non-NAN) values & use numpy average function (includes weighted avg calculation) 
        # Get arrays indices of valid values
        indok    = np.isfinite(field)
        for t in range(ntime):
            indok_2d = indok[t,:,:]
            field_2d = np.squeeze(field[t,:,:])
            if np.max(Wmask) >0.:
                rm[region,t] = np.average(field_2d[indok_2d],weights=Wmask[indok_2d])
            else:
                rm[region,t] = np.nan
    return rm


def class_docs_fixer(cls):
    """Decorator to fix docstrings for subclasses"""
    if not cls.__doc__:
        for parent in cls.__bases__:
            if parent.__doc__:
                cls.__doc__ = parent.__doc__

    for name, func in list(vars(cls).items()):
        if not func.__doc__ or '%%aug%%' in func.__doc__:
            for parent in cls.__bases__:
                if hasattr(parent, name):
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
    func.__doc__ = '%%aug%%' + func.__doc__
    return func


def create_precalc_ye_filename(config,psm_key,prior_kind):
    """
    Create the filename to use for the precalculated Ye file from the provided
    configuration.  Uses the prior, psm, and proxy configuration to determine a
    unique name for the current experiment.

    Parameters
    ----------
    config: Config
        Instance of an LMR_config.Config object.
    psm_key: str
        Indicates the psm type with which Ye values were calculated.
    prior_kind: str
        Indicates whether Ye values were calculated using 
        anomalies ('anom') of full field ('full') as prior data

    Returns
    -------
    str:
        Filename string based on current configuration
    """

    proxy_database = config.proxies.use_from[0]

    # Generate PSM calibration string
    if psm_key == 'linear':
        calib_avgPeriod = config.psm.linear.avgPeriod
        calib_str = config.psm.linear.datatag_calib
        state_vars_for_ye = config.psm.linear.psm_required_variables
        
    elif psm_key == 'linear_TorP':
        calib_avgPeriod = config.psm.linear_TorP.avgPeriod
        calib_str = 'T:{}-PR:{}'.format(config.psm.linear_TorP.datatag_calib_T,
                                        config.psm.linear_TorP.datatag_calib_P)
        state_vars_for_ye = config.psm.linear_TorP.psm_required_variables

    elif psm_key == 'bilinear':
        calib_avgPeriod = config.psm.bilinear.avgPeriod
        calib_str = 'T:{}-PR:{}'.format(config.psm.bilinear.datatag_calib_T,
                                        config.psm.bilinear.datatag_calib_P)
        state_vars_for_ye = config.psm.bilinear.psm_required_variables
        
    elif psm_key == 'h_interp':
        calib_avgPeriod = None
        calib_str = ''
        state_vars_for_ye = config.psm.h_interp.psm_required_variables

    elif  psm_key == 'bayesreg_uk37':
        calib_avgPeriod = ''.join([str(config.prior.avgInterval['multiyear'][0]),'yrs'])
        calib_str = ''
        state_vars_for_ye = config.psm.bayesreg_uk37.psm_required_variables
        
    else:
        raise ValueError('Unrecognized PSM key.')
    
    if calib_avgPeriod:
        psm_str = psm_key +'_'+ calib_avgPeriod + '-' + calib_str
    else:
        psm_str = psm_key + '-' + calib_str

    proxy_str = str(proxy_database)
    if proxy_str == 'LMRdb':
        proxy_str = proxy_str + str(config.proxies.LMRdb.dbversion)
    elif proxy_str == 'NCDCdtda':
        proxy_str = proxy_str + str(config.proxies.NCDCdtda.dbversion)
        
    # Generate appropriate prior string
    prior_str = '-'.join([config.prior.prior_source] +
                         sorted(state_vars_for_ye) + [prior_kind])

    return '{}_{}_{}.npz'.format(prior_str, psm_str, proxy_str)


def load_precalculated_ye_vals(config, proxy_manager, sample_idxs):
    """
    Convenience function to load a precalculated Ye file for the current
    experiment.

    Parameters
    ----------
    config: LMR_config.Config
        Current experiment instance of the configuration object.
    proxy_manager: LMR_proxy_pandas_rework.ProxyManager
        Current experiment proxy manager
    sample_idxs: list(int)
        A list of the current sample indices used to create the prior ensemble.

    Returns
    -------
    ye_all: ndarray
        The array of Ye values for the current ensemble and all proxy records
    """
    load_dir = os.path.join(config.core.lmr_path, 'ye_precalc_files')
    load_fname = create_precalc_ye_filename(config)
    precalc_file = np.load(os.path.join(load_dir, load_fname))

    pid_idx_map = precalc_file['pid_index_map'][()]
    precalc_vals = precalc_file['ye_vals']

    num_proxies_assim = len(proxy_manager.ind_assim)
    num_samples = len(sample_idxs)
    ye_all = np.zeros((num_proxies_assim, num_samples))

    for i, pobj in enumerate(proxy_manager.sites_assim_proxy_objs()):
        pidx = pid_idx_map[pobj.id]
        ye_all[i] = precalc_vals[pidx, sample_idxs]

    return ye_all


def load_precalculated_ye_vals_psm_per_proxy(config, proxy_manager, proxy_set, sample_idxs):
    """
    Convenience function to load a precalculated Ye file for the current
    experiment.

    Parameters
    ----------
    config: LMR_config.Config
        Current experiment instance of the configuration object.
    proxy_manager: LMR_proxy_pandas_rework.ProxyManager
        Current experiment proxy manager
    proxy_set: str
        String to indicate which proxy set to handle ('assim' or 'eval')
    sample_idxs: list(int)
        A list of the current sample indices used to create the prior ensemble.

    Returns
    -------
    ye_all: ndarray
        The array of Ye values for the current ensemble and all records from
        selected proxy set.
    """

    begin_load = time()

    load_dir = os.path.join(config.core.lmr_path, 'ye_precalc_files')
    
    if proxy_set == 'assim':
        num_proxies_assim = len(proxy_manager.ind_assim)
        psm_keys = list(set([pobj.psm_obj.psm_key for pobj in proxy_manager.sites_assim_proxy_objs()]))
    elif proxy_set == 'eval':
        num_proxies_assim = len(proxy_manager.ind_eval)
        psm_keys = list(set([pobj.psm_obj.psm_key for pobj in proxy_manager.sites_eval_proxy_objs()]))
    num_samples = len(sample_idxs)
    ye_all = np.zeros((num_proxies_assim, num_samples))
    ye_all_coords = np.zeros((num_proxies_assim, 2))
    
    precalc_files = {}
    for psm_key in psm_keys:

        pkind = None
        if psm_key == 'linear':
            pkind = list(config.psm.linear.psm_required_variables.values())[0]
        elif psm_key == 'linear_TorP':
            pkind = list(config.psm.linear_TorP.psm_required_variables.values())[0]
        elif psm_key == 'bilinear':
            pkind = list(config.psm.bilinear.psm_required_variables.values())[0]
        elif psm_key == 'h_interp':
            if config.proxies.proxy_timeseries_kind == 'asis':
                pkind = 'full'
            elif config.proxies.proxy_timeseries_kind == 'anom':
                pkind = 'anom'
            else:
                raise ValueError('Unrecognized proxy_timeseries_kind in proxies class')
        elif psm_key == 'bayesreg_uk37':
            pkind = 'full'
        else:
            raise ValueError('Unrecognized PSM key.')

        load_fname = create_precalc_ye_filename(config,psm_key,pkind)
        print('  Loading file:', load_fname)
        # check if file exists
        if not os.path.isfile(os.path.join(load_dir, load_fname)):
            print ('  ERROR: File does not exist!'
                   ' -- run the precalc file builder:'
                   ' misc/build_ye_file.py'
                   ' to generate the missing file')
            raise SystemExit()
        precalc_files[psm_key] = np.load(os.path.join(load_dir, load_fname))

    
    print('  Now extracting proxy type-dependent Ye values...')
    if proxy_set == 'assim':
        for i, pobj in enumerate(proxy_manager.sites_assim_proxy_objs()):
            psm_key = pobj.psm_obj.psm_key
            pid_idx_map = precalc_files[psm_key]['pid_index_map'][()]
            precalc_vals = precalc_files[psm_key]['ye_vals']

            pidx = pid_idx_map[pobj.id]
            ye_all[i] = precalc_vals[pidx, sample_idxs]

            ye_all_coords[i,:] = np.asarray([pobj.lat, pobj.lon], dtype=np.float64)
    elif proxy_set == 'eval':
        for i, pobj in enumerate(proxy_manager.sites_eval_proxy_objs()):
            psm_key = pobj.psm_obj.psm_key
            pid_idx_map = precalc_files[psm_key]['pid_index_map'][()]
            precalc_vals = precalc_files[psm_key]['ye_vals']

            pidx = pid_idx_map[pobj.id]
            ye_all[i] = precalc_vals[pidx, sample_idxs]

            ye_all_coords[i,:] = np.asarray([pobj.lat, pobj.lon], dtype=np.float64)        
            
    print('  Completed in ',  time() - begin_load, 'secs')
        
    return ye_all, ye_all_coords


def load_precalculated_ye_vals_psm_per_proxy_onlyobjs(config, proxy_objs, sample_idxs):
    """
    Convenience function to load a precalculated Ye file for the current
    experiment.

    Parameters
    ----------
    config: LMR_config.Config
        Current experiment instance of the configuration object.
    proxy_manager: LMR_proxy_pandas_rework.ProxyManager
        Current experiment proxy manager
    sample_idxs: list(int)
        A list of the current sample indices used to create the prior ensemble.

    Returns
    -------
    ye_all: ndarray
        The array of Ye values for the current ensemble and all proxy records
    """

    begin_load = time()

    load_dir = os.path.join(config.core.lmr_path, 'ye_precalc_files')

    num_proxies_assim = len(proxy_objs)
    num_samples = len(sample_idxs)
    ye_all = np.zeros((num_proxies_assim, num_samples))
    ye_all_coords = np.zeros((num_proxies_assim, 2))

    
    #psm_keys = list(set([pobj.psm_obj.psm_key for pobj in proxy_manager.sites_assim_proxy_objs()]))
    psm_keys = list(set([pobj.psm_obj.psm_key for pobj in proxy_objs]))    
    precalc_files = {}
    for psm_key in psm_keys:

        pkind = None
        if psm_key == 'linear':
            pkind = list(config.psm.linear.psm_required_variables.values())[0]
        elif psm_key == 'linear_TorP':
            pkind = list(config.psm.linear_TorP.psm_required_variables.values())[0]
        elif psm_key == 'bilinear':
            pkind = list(config.psm.bilinear.psm_required_variables.values())[0]
        elif psm_key == 'h_interp':
            if config.proxies.proxy_timeseries_kind == 'asis':
                pkind = 'full'
            elif config.proxies.proxy_timeseries_kind == 'anom':
                pkind = 'anom'
            else:
                raise ValueError('Unrecognized proxy_timeseries_kind in proxies class')
        elif psm_key == 'bayesreg_uk37':
            pkind = 'full'
        else:
            raise ValueError('Unrecognized PSM key.')

        load_fname = create_precalc_ye_filename(config,psm_key,pkind)
        print('  Loading file:', load_fname)
        # check if file exists
        if not os.path.isfile(os.path.join(load_dir, load_fname)):
            print ('  ERROR: File does not exist!'
                   ' -- run the precalc file builder:'
                   ' misc/build_ye_file.py'
                   ' to generate the missing file')
            raise SystemExit()
        precalc_files[psm_key] = np.load(os.path.join(load_dir, load_fname))

    
    print('  Now extracting proxy type-dependent Ye values...')
    #for i, pobj in enumerate(proxy_manager.sites_assim_proxy_objs()):
    for i, pobj in enumerate(proxy_objs):
        psm_key = pobj.psm_obj.psm_key
        pid_idx_map = precalc_files[psm_key]['pid_index_map'][()]
        precalc_vals = precalc_files[psm_key]['ye_vals']
        
        pidx = pid_idx_map[pobj.id]
        ye_all[i] = precalc_vals[pidx, sample_idxs]

        ye_all_coords[i,:] = np.asarray([pobj.lat, pobj.lon], dtype=np.float64)

    print('  Completed in ',  time() - begin_load, 'secs')
        
    return ye_all, ye_all_coords


def gaussianize(X):
    """
    Transforms a (proxy) timeseries to Gaussian distribution.

    Originator: Michael Erb, Univ. of Southern California - April 2017

    """

    # Give every record at least one dimensions, or else the code will crash.
    X = np.atleast_1d(X)

    # Make a blank copy of the array, retaining the data type of the original data variable.
    Xn = copy.deepcopy(X)
    Xn[:] = np.NAN

    if len(X.shape) == 1:
        Xn = gaussianize_single(X)
    else:
        for i in range(X.shape[1]):
            Xn[:,i] = gaussianize_single(X[:,i])

    return Xn


def gaussianize_single(X_single):
    """
    Transforms a single (proxy) timeseries to Gaussian distribution.

    Originator: Michael Erb, Univ. of Southern California - April 2017

    """

    # Count only elements with data.
    n = X_single[~np.isnan(X_single)].shape[0]

    # Create a blank copy of the array.
    Xn_single = copy.deepcopy(X_single)
    Xn_single[:] = np.NAN

    nz = np.logical_not(np.isnan(X_single))

    index = np.argsort(X_single[nz])
    rank = np.argsort(index)

    CDF = 1.*(rank+1)/(1.*n) -1./(2*n)
    Xn_single[nz] = np.sqrt(2)*special.erfinv(2*CDF -1)

    return Xn_single


def validate_config(config):
    """
    Function to check for inconsistencies in the experiment 
    configuration.

    Parameters
    ----------
    config: LMR_config.Config
        Current experiment instance of the configuration object.

    Returns
    -------
    proceed_ok: bool
        Boolean indicating if the configuration settings were validated or not.
    """

    proxy_database = config.proxies.use_from[0]
    if proxy_database == 'LMRdb':
        proxy_cfg = config.proxies.LMRdb
    elif proxy_database == 'PAGES2kv1':
        proxy_cfg = config.proxies.PAGES2kv1
    elif proxy_database == 'NCDCdtda':
        proxy_cfg = config.proxies.NCDCdtda
    else:
        print('ERROR in specification of proxy database.')
        raise SystemExit()

    # proxy types activated in configuration
    proxy_types = proxy_cfg.proxy_order 
    # associated psm's
    psm_keys = list(set([proxy_cfg.proxy_psm_type[p] for p in proxy_types]))

    # Forming list of required state variables
    psmclasses = dict([(name,cls)  for name, cls in list(config.psm.__dict__.items())])
    psm_required_variables = []
    for psm_type in psm_keys:
        #print psm_type, ':', psmclasses[psm_type].psm_required_variables
        psm_required_variables.extend(psmclasses[psm_type].psm_required_variables)
    # keep unique values
    psm_required_variables = list(set(psm_required_variables))

    
    proceed_ok = True

    
    # Begin checking configuration
    print('Checking configuration ... ')

    # Conditions when use_precalc_ye = False
    if not config.core.use_precalc_ye:

        # 1) Check whether all variables needed by PSMs are in prior.state_variables
        
        for psm_required_var in psm_required_variables:
            if psm_required_var not in list(config.prior.state_variables.keys()):
                print(' ERROR: Missing state variable:', psm_required_var)
                print (' Could not calculate required ye_values from prior.'
                       ' Add the required variable to the state variable'
                       ' list -- OR -- run the precalc file builder:'
                       ' misc/build_ye_file.py'
                       ' and use precalculated Ye values')
                proceed_ok = False

        # 2) Check if psm_avg is set to 'season' when use_precalc_ye = False
        #    This combination of optios is not currently no possible as management
        #    of seasonal prior state variables is not enabled.

        if config.psm.avgPeriod == 'season':
            print (' ERROR: Conflicting options in configuration :'
                   ' Trying to use seasonally-calibrated PSM'
                   ' (avgPeriod=season in class psm) while'
                   ' the use_precalc_ye option is set to False. '
                   ' Combination of options not enabled!')
            proceed_ok = False


    # Constraints irrespective of value chosen for use_precalc_ye

    # 3) Cannot use seasonally-calibrated PSMs with PAGES2kv1 proxies
    #    Required metadata not available in that dataset.

    if config.psm.avgPeriod == 'season' and proxy_database == 'PAGES2kv1':
        print (' ERROR: Conflicting options in configuration :'
               ' Trying to use seasonally-calibrated PSM'
               ' (avgPeriod=season in class psm) with PAGES2kv1 proxies.'
               ' No seasonality metadata available in that dataset.'
               ' Change avgPeriod to "annual" in your configuration.')
        proceed_ok = False
        
    # 4) Check compatibility between variable 'kind' ('anom' or 'full')
    #    between prior state variables and requirements of chosen PSMs

    # For every PSM class to be used
    for psm_type in psm_keys:

        required_variables = list(psmclasses[psm_type].psm_required_variables.keys())
        for var in required_variables:
            if var in list(config.prior.state_variables.keys()):
                if psmclasses[psm_type].psm_required_variables[var] != config.prior.state_variables[var]:
                    print((' ERROR: Conflict detected in configuration :'
                           ' Selected variable kind for var='+var+' ('+config.prior.state_variables[var]+')'
                           ' is incompatible with requirements of '+ psm_type+' PSM ('
                           + psmclasses[psm_type].psm_required_variables[var]+') using this variable as input'))
                    proceed_ok = False
    
    
    return proceed_ok


def _param_str_to_update_dict(param_str, value):
    split_params = param_str.split('.')
    try:
        value_str = '{:g}'.format(value)
    except ValueError as e:
        value_str = '{}'.format(value)
    param_dir_str = split_params[-1] + '-' + value_str

    while split_params:
        curr_param = split_params.pop()
        update_dict = {curr_param: value}
        value = update_dict

    return update_dict, param_dir_str


def nested_dict_update(orig_dict, update_dict):
    for key, val in update_dict.items():
        if isinstance(val, collections.Mapping):
            res = nested_dict_update(orig_dict.get(key, {}), val)
            orig_dict[key] = res
        else:
            orig_dict[key] = val
    return orig_dict


def param_cfg_update(param_str, val, cfg_dict=None):

    if cfg_dict is None:
        cfg_dict = {}

    update_dict, _ = _param_str_to_update_dict(param_str, val)
    return nested_dict_update(cfg_dict, update_dict)


def psearch_list_cfg_update(param_str_list, val_list, cfg_dict=None):

    if cfg_dict is None:
        cfg_dict = {}

    param_dir_str = []
    for param_str, val in zip(param_str_list, val_list):
        update_dict, dir_str = _param_str_to_update_dict(param_str, val)
        param_dir_str.append(dir_str)
        cfg_dict = nested_dict_update(cfg_dict, update_dict)

    param_dir_str = '_'.join(param_dir_str)
    return cfg_dict, param_dir_str


class FlagError(ValueError):
    """
    Error for exiting code sections
    """
    pass

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
        print('field dimensions...')
        print(np.shape(field))

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
            print('region='+str(region))
            print(rlat[region,0],rlat[region,1],rlon[region,0],rlon[region,1])
            
        # regional weighting (ones in region; zeros outside)
        mask = regional_mask(lat,lon,rlat[region,0],rlat[region,1],rlon[region,0],rlon[region,1])
        if debug:
            print('mask=')
            print(mask)

        # this is the weight mask for the regional domain    
        Wmask = np.multiply(mask,W)

        # make sure data starts at South Pole
        if lat[0] > 0:
            # data has NH -> SH format; reverse
            field = np.flipud(field)

        # Check for valid (non-NAN) values & use numpy average function (includes weighted avg calculation) 
        # Get arrays indices of valid values
        indok    = np.isfinite(field)
        for t in range(ntime):
            indok_2d = indok[t,:,:]
            field_2d = np.squeeze(field[t,:,:])
            if np.max(Wmask) >0.:
                rm[region,t] = np.average(field_2d[indok_2d],weights=Wmask[indok_2d])
            else:
                rm[region,t] = np.nan
    return rm

def find_date_indices(time,stime,etime):

    # find start and end times that match specific values
    # input: time: an array of time values
    #        stime: the starting time
    #        etime: the ending time

    # initialize returned variables
    begin_index = None
    end_index = None
    
    smatch = np.where(time==stime)
    ematch = np.where(time==etime)

    # make sure valid integers are returned
    if type(smatch) is tuple:
        smatch = smatch[0]
        ematch = ematch[0]
        
    if type(smatch) is np.ndarray:
        try:
            smatch = smatch[0]
        except IndexError:
            pass
        try:
            ematch = ematch[0]
        except IndexError:
            pass

    
    if isinstance(smatch,(int,np.integer)):
        begin_index = smatch
    if isinstance(ematch,(int,np.integer)):
        end_index = ematch

    return begin_index, end_index
