
# -------------------------------------------------------------------------------
# *** Prior source assignment  --------------------------------------------------
# -------------------------------------------------------------------------------
# All logic for prior object assignment
def prior_assignment(iprior):
    if iprior == 'generic':
        prior_object = prior_generic()
    elif iprior == 'GISTEMP':
        prior_object = prior_gistemp()
    elif iprior == 'ccsm4_last_millenium':
        prior_object = prior_ccsm4_last_millenium()
      
    return prior_object


# -------------------------------------------------------------------------------
# *** Master class for model data as prior --------------------------------------
# -------------------------------------------------------------------------------
class prior_master(object):
    '''
    This is the master class for the prior data. Inherent to create classes for each prior source.
    '''

    # Populate the prior ensemble from gridded model/analysis data
    def populate_ensemble(self,prior_source):

        import numpy as np
        from random import sample
        
        # Load prior **annually averaged** data from file
        self.read_prior()

        [ntime,nlat,nlon] = self.value.shape
        print 'Nb of years in prior data:', ntime

        nbvars = len(self.statevars)
        Nx = nbvars*nlat*nlon
        # Array containing the prior ensemble
        Xb = np.zeros(shape=[Nx,self.Nens]) # no time dimension now...

        print 'Random selection of', str(self.Nens), 'ensemble members'
        # Populate prior ensemble from randomly sampled states
        ind_ens = sample(range(0,ntime-1),self.Nens)

        # Keep lat/lon of gridpoints (needed geo. information)
        X_lat = np.zeros(shape=[nlat,nlon])
        X_lon = np.zeros(shape=[nlat,nlon])
        for j in xrange(0,nlat):
            for k in xrange(0,nlon):
                X_lat[j,k] = self.lat[j]
                X_lon[j,k] = self.lon[k]
        Xb_lat = X_lat.flatten()
        Xb_lon = X_lon.flatten()

        # NOTE: CODE NOW ONLY WORKS FOR A SINGLE STATE VARIABLE (i.e. Tsfc) !!!
        # Loop over ensemble members
        for i in xrange(0,self.Nens):
            # Pick 2D slice from specific year & flatten into a 1D array 
            # to populate member "i" of the ensemble Xb
            Xb[:,i] = self.value[ind_ens[i],:,:].flatten()

        # Some cleanup
        del X_lat
        del X_lon

        # Assign return variables
        self.ens  = Xb
        self.lat  = Xb_lat
        self.lon  = Xb_lon
        self.nlat = nlat
        self.nlon = nlon

        return


# -------------------------------------------------------------------------------
# Classes for specific model/simulation -----------------------------------------
# -------------------------------------------------------------------------------

# class for generic object
class prior_generic(prior_master):
    pass

# class for GISTEMP gridded surface temperature dataset
class prior_gistemp(prior_master):
    pass

# class for BerkeleyEarth gridded surface temperature dataset
class prior_BerkeleyEarth(prior_master):
    pass

# class for the CCSM4 Last Millenium simulation
class prior_ccsm4_last_millenium(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_ccsm4_last_millenium
        [self.time,self.lat,self.lon,self.value] = read_gridded_data_ccsm4_last_millenium(self.prior_datadir,self.prior_datafile,self.statevars)

        return

