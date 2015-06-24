
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
    elif iprior == 'ccsm4_preindustrial_control':
        prior_object = prior_ccsm4_preindustrial_control()      

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
        
        # Load prior **annually averaged** data from file(s) - multiple state variables
        self.read_prior()

        nbvars = len(self.statevars)
        # Check consistency between specified state variables and uploaded dictionary
        if len(self.prior_dict.keys()) != nbvars:
            print 'Problem with load of prior state variables. Exiting!'
            exit(1)

        # Defining content of state vector => dictionary: state_vect_content
        # NOTE: now assumes that dims of state variables are (lat,lon) only !!!
        state_vect_info = {}

        # Loop over state variables
        Nx = 0
        for var in self.prior_dict.keys():
            [ntime,nlat,nlon] = self.prior_dict[var]['value'].shape
            print 'Nb of years in prior data for var = %s : %d' % (var,ntime)
            state_vect_info[var] = (Nx,Nx+(nlat*nlon)-1)
            # determining length of state vector
            Nx = Nx + (nlat*nlon)

        print 'Nx =', Nx
        print 'state_vect_info=', state_vect_info

        # Array containing the prior ensemble
        Xb = np.zeros(shape=[Nx,self.Nens]) # no time dimension now...

        # To keep lat/lon of gridpoints (needed geo. information)
        Xb_lat = np.zeros(shape=[Nx])
        Xb_lon = np.zeros(shape=[Nx])
        
        print 'Random selection of', str(self.Nens), 'ensemble members'
        # Populate prior ensemble from randomly sampled states
        ind_ens = sample(range(0,ntime-1),self.Nens)

        for var in self.prior_dict.keys():
            indstart = state_vect_info[var][0]
            indend   = state_vect_info[var][1]
            # Loop over ensemble members
            for i in range(0,self.Nens):
                Xb[indstart:indend+1,i] = self.prior_dict[var]['value'][ind_ens[i],:,:].flatten()

            lat = self.prior_dict[var]['lat']
            lon = self.prior_dict[var]['lon']
            nlat = lat.shape[0]
            nlon = lon.shape[0]
            X_lat = np.zeros(shape=[nlat,nlon])
            X_lon = np.zeros(shape=[nlat,nlon])
            for j in xrange(0,nlat):
                for k in xrange(0,nlon):
                    X_lat[j,k] = lat[j]
                    X_lon[j,k] = lon[k]

            Xb_lat[indstart:indend+1] = X_lat.flatten()
            Xb_lon[indstart:indend+1] = X_lon.flatten()

        # Some cleanup
        del X_lat
        del X_lon

        # Assign return variables
        self.ens  = Xb
        self.lat  = Xb_lat
        self.lon  = Xb_lon
        self.nlat = nlat
        self.nlon = nlon
        self.full_state_info = state_vect_info

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
        self.prior_dict = read_gridded_data_ccsm4_last_millenium(self.prior_datadir,self.prior_datafile,self.statevars)

        return

# class for the CCSM4 Pre-Industrial Control simulation
class prior_ccsm4_preindustrial_control(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_ccsm4_preindustrial_control
        [self.time,self.lat,self.lon,self.value] = read_gridded_data_ccsm4_preindustrial_control(self.prior_datadir,self.prior_datafile,self.statevars)

        return

