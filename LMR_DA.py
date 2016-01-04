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
    varye = np.var(Ye)

    # lowercase ye has ensemble-mean removed 
    ye = np.subtract(Ye, mye)

    # innovation
    try:
        innov = obvalue - mye
    except:
        print 'innovation error. obvalue = ' + str(obvalue) + ' mye = ' + str(mye)
        print 'returning Xb unchanged...'
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

    # Return the full state
    return Xa


#========================================================================================== 
#
#
#========================================================================================== 

def cov_localization(locRad,X,Y):
    """

    Originator: R. Tardif, 
                Dept. Atmos. Sciences, Univ. of Washington
    -----------------------------------------------------------------
     Inputs:
      locRad : Localization radius (distance for which cov are forced to zero
           X : Prior object, needed to get grid info. 
           Y : Proxy object, needed to get ob site lat/lon (to calculate distances w.r.t. grid pts
     Output:
      covLoc : Localization vector (weights) applied to ensemble covariance estimates (Nx x 1),
               with Nx the dimension of the state vector

     Note: Uses the Gaspari-Cohn localization function.

    """

    nlat  = X.nlat
    nlon  = X.nlon

    site_lat = Y.lat
    site_lon = Y.lon

    # declare the localization array (Nx = nlat x nlon for now ... assume single state variable) 
    covLoc = np.zeros(shape=[nlat*nlon])

    dists = np.array(LMR_utils.haversine(site_lon, site_lat, X.lon, X.lat))
    
    hlr = 0.5*locRad; # work with half the localization radius
    r = dists/hlr;

    ind_inner = np.where(dists <= hlr)
    ind_outer = np.where(dists >  hlr)
    ind_out   = np.where(dists >  2*hlr)

    # Gaspari-Cohn function
    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (((-0.25*r[ind_inner]+0.5)*r[ind_inner]+0.625)*r[ind_inner]-(5.0/3.0))*(r[ind_inner]**2)+1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = ((((r[ind_outer]/12. - 0.5) * r[ind_outer] + 0.625) * r[ind_outer] + 5./3) * r[ind_outer] - 5.0) * r[ind_outer]\
        + 4.0 - 2.0/(3.0*r[ind_outer])
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0

    return covLoc
