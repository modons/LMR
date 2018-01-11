#==========================================================================================
# Data assimilation function. 
#
# Define the data assimilation function
# This version uses passed arrays, and updates the ensemble for a single time
# (& single ob).
#
#==========================================================================================

import numpy as np
import LMR_utils

def enkf_update_array(Xb, obvalue, Ye, ob_err, loc=None, inflate=None):
    """
    Function to do the ensemble square-root filter (EnSRF) update
    (ref: Whitaker and Hamill, Mon. Wea. Rev., 2002)

    Originator: G. J. Hakim, with code borrowed from L. Madaus
                Dept. Atmos. Sciences, Univ. of Washington

    Revisions:

    1 September 2017: 
                    - changed varye = np.var(Ye) to varye = np.var(Ye,ddof=1) 
                    for an unbiased calculation of the variance. 
                    (G. Hakim - U. Washington)
    
    -----------------------------------------------------------------
     Inputs:
          Xb: background ensemble estimates of state (Nx x Nens) 
     obvalue: proxy value
          Ye: background ensemble estimate of the proxy (Nens x 1)
      ob_err: proxy error variance
         loc: localization vector (Nx x 1) [optional]
     inflate: scalar inflation factor [optional]
    """

    # Get ensemble size from passed array: Xb has dims [state vect.,ens. members]
    Nens = Xb.shape[1]

    # ensemble mean background and perturbations
    xbm = np.mean(Xb,axis=1)
    Xbp = np.subtract(Xb,xbm[:,None])  # "None" means replicate in this dimension

    # ensemble mean and variance of the background estimate of the proxy 
    mye   = np.mean(Ye)
    varye = np.var(Ye,ddof=1)

    # lowercase ye has ensemble-mean removed 
    ye = np.subtract(Ye, mye)

    # innovation
    try:
        innov = obvalue - mye
    except:
        print('innovation error. obvalue = ' + str(obvalue) + ' mye = ' + str(mye))
        print('returning Xb unchanged...')
        return Xb
    
    # innovation variance (denominator of serial Kalman gain)
    kdenom = (varye + ob_err)

    # numerator of serial Kalman gain (cov(x,Hx))
    kcov = np.dot(Xbp,np.transpose(ye)) / (Nens-1)

    # Option to inflate the covariances by a certain factor
    if inflate is not None:
        kcov = inflate * kcov

    # Option to localize the gain
    if loc is not None:
        kcov = np.multiply(kcov,loc)
   
    # Kalman gain
    kmat = np.divide(kcov, kdenom)

    # update ensemble mean
    xam = xbm + np.multiply(kmat,innov)

    # update the ensemble members using the square-root approach
    beta = 1./(1. + np.sqrt(ob_err/(varye+ob_err)))
    kmat = np.multiply(beta,kmat)
    ye   = np.array(ye)[np.newaxis]
    kmat = np.array(kmat)[np.newaxis]
    Xap  = Xbp - np.dot(kmat.T, ye)

    # full state
    Xa = np.add(xam[:,None], Xap)

    # if masked array, making sure that fill_value = nan in the new array 
    if np.ma.isMaskedArray(Xa): np.ma.set_fill_value(Xa, np.nan)

    
    # Return the full state
    return Xa


#========================================================================================== 
#
#========================================================================================== 

def cov_localization(locRad, Y, X, X_coords):
    """

    Originator: R. Tardif, 
                Dept. Atmos. Sciences, Univ. of Washington
    -----------------------------------------------------------------
     Inputs:
        locRad : Localization radius (distance in km beyond which cov are forced to zero)
             Y : Proxy object, needed to get ob site lat/lon (to calculate distances w.r.t. grid pts
             X : Prior object, needed to get state vector info. 
      X_coords : Array containing geographic location information of state vector elements

     Output:
        covLoc : Localization vector (weights) applied to ensemble covariance estimates.
                 Dims = (Nx x 1), with Nx the dimension of the state vector.

     Note: Uses the Gaspari-Cohn localization function.

    """

    # declare the localization array, filled with ones to start with (as in no localization)
    stateVectDim, nbdimcoord = X_coords.shape
    covLoc = np.ones(shape=[stateVectDim],dtype=np.float64)

    # Mask to identify elements of state vector that are "localizeable"
    # i.e. fields with (lat,lon)
    localizeable = covLoc == 1. # Initialize as True
    
    for var in X.trunc_state_info.keys():
        [var_state_pos_begin,var_state_pos_end] =  X.trunc_state_info[var]['pos']
        # if variable is not a field with lats & lons, tag localizeable as False
        if X.trunc_state_info[var]['spacecoords'] != ('lat', 'lon'):
            localizeable[var_state_pos_begin:var_state_pos_end+1] = False
    
    # array of distances between state vector elements & proxy site
    # initialized as zeros: this is important!
    dists = np.zeros(shape=[stateVectDim])

    # geographic location of proxy site
    site_lat = Y.lat
    site_lon = Y.lon
    # geographic locations of elements of state vector
    X_lon = X_coords[:,1]
    X_lat = X_coords[:,0]

    # calculate distances for elements tagged as "localizeable". 
    dists[localizeable] = np.array(LMR_utils.haversine(site_lon, site_lat,
                                                       X_lon[localizeable],
                                                       X_lat[localizeable]),dtype=np.float64)

    # those not "localizeable" are assigned with a disdtance of "nan"
    # so these elements will not be included in the indexing
    # according to distances (see below)
    dists[~localizeable] = np.nan
    
    # Some transformation to variables used in calculating localization weights
    hlr = 0.5*locRad; # work with half the localization radius
    r = dists/hlr;
    
    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)    # closest
    ind_outer = np.where(dists >  hlr)    # close
    ind_out   = np.where(dists >  2.*hlr) # out

    # Gaspari-Cohn function
    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (((-0.25*r[ind_inner]+0.5)*r[ind_inner]+0.625)* \
                         r[ind_inner]-(5.0/3.0))*(r[ind_inner]**2)+1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = ((((r[ind_outer]/12. - 0.5) * r[ind_outer] + 0.625) * \
                          r[ind_outer] + 5.0/3.0) * r[ind_outer] - 5.0) * \
                          r[ind_outer] + 4.0 - 2.0/(3.0*r[ind_outer])
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0

    # prevent negative values: calc. above may produce tiny negative
    # values for distances very near the localization radius
    # TODO: revisit calculations to minimize round-off errors
    covLoc[covLoc < 0.0] = 0.0

    
    return covLoc
