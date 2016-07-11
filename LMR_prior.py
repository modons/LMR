"""
Module: LMR_prior.py

Purpose: Contains definitions of classes defining the various sources
         (i.e. model simulations and reanalyses) which may be used as to
         populate the prior in the LMR. Also contains the code used to 
         randomly pick model states along the temporal domain to populate
         the prior ensemble. 

Originator: Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
                          | January 2015

Revisions: 
          - Added the ERA20CM (ECMWF 20th century ensemble simulation) as a 
            possible source of prior data to be used by the LMR.
            [R. Tardif, U. of Washington, December 2015]
          - Added the option of detrending the prior
            [R. Tardif, U. of Washington, March 2016]
          - Added the 'ccsm4_isotope_controlrun' as a possible prior source.
            Contains simulated isotope (d18O) field. 
            [R. Tardif, U. of Washington, May 2016]

"""
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
    elif iprior == 'ccsm4_isotope_controlrun':
        prior_object = prior_ccsm4_isotope_controlrun()
    elif iprior == 'mpi-esm-p_last_millenium':
        prior_object = prior_mpi_esm_p_last_millenium()
    elif iprior == 'gfdl-cm3_preindustrial_control':
        prior_object = prior_gfdl_cm3_preindustrial_control()
    elif iprior == '20cr':
        prior_object = prior_20cr()
    elif iprior == 'era20c':
        prior_object = prior_era20c()
    elif iprior == 'era20cm':
        prior_object = prior_era20cm()


    return prior_object


# -------------------------------------------------------------------------------
# *** Master class for model data as prior --------------------------------------
# -------------------------------------------------------------------------------
class prior_master(object):
    '''
    This is the master class for the prior data. Inherent to create classes for each prior source.
    '''

    # Populate the prior ensemble from gridded model/analysis data
    def populate_ensemble(self,prior_source, prior_cfg):

        import numpy as np
        from random import sample, seed
        

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
        timedim = []
        for var in self.prior_dict.keys():
            dct = {}
            timedim.append(len(self.prior_dict[var]['years']))
            spacecoords = self.prior_dict[var]['spacecoords']
            #print '==>', var, spacecoords
            if spacecoords:
                #print 'spacecoords is not None: variable with space coordinates'
                dim1, dim2 = spacecoords
                ndim1 = len(self.prior_dict[var][dim1])
                ndim2 = len(self.prior_dict[var][dim2])
                ndimtot = ndim1*ndim2
                dct['pos'] = (Nx,Nx+(ndimtot)-1)
                dct['spacecoords'] = spacecoords
                dct['spacedims'] = (ndim1,ndim2)
            else:
                #print 'spacecoords is None: variable is simple time series'
                ndimtot = 1
                dct['pos'] = (Nx,Nx+(ndimtot)-1)
                dct['spacecoords'] = None
                dct['spacedims'] = None

            # assign to master dictionary
            state_vect_info[var] = dct
            
            # determining length of state vector
            Nx = Nx + (ndimtot)               

        # Looped through all state variables, now a summary:
        print ' '
        print 'State vector information:'
        print 'Nx =', Nx
        print 'state_vect_info=', state_vect_info

        # Array that will contain the prior ensemble
        Xb = np.zeros(shape=[Nx,self.Nens]) # no time dimension now...


        # TODO: AP sorting might help read faster
        # time dimension consistent across variables?
        if all(x == timedim[0] for x in timedim):
            ntime = timedim[0]
        else: 
            print 'ERROR: time dimension not consistent across all state variables. Exiting!'
            exit(1)

        # ***NOTE: Following code assumes that data for a given year are located at same array time index across all state variables
        print 'Random selection of', str(self.Nens), 'ensemble members'
        # Populate prior ensemble from randomly sampled states
        seed(prior_cfg.seed)
        ind_ens = sample(range(ntime), self.Nens)
        self.prior_sample_indices = ind_ens


        # To keep spatial coords of gridpoints (needed geo. information)
        Xb_coords = np.empty(shape=[Nx,2]) # 2 is max nb of spatial dim a variable can take
        Xb_coords[:,:] = np.NAN # initialize with Nan's

        for var in self.prior_dict.keys():

            indstart = state_vect_info[var]['pos'][0]
            indend   = state_vect_info[var]['pos'][1]
            try:
                nbspacecoords = len(state_vect_info[var]['spacecoords'])
                if nbspacecoords == 2:
                    # Loop over ensemble members
                    for i in range(0,self.Nens):
                        Xb[indstart:indend+1,i] = self.prior_dict[var]['value'][ind_ens[i],:,:].flatten()

                    # get the name of the spatial coordinates for state variable 'var'
                    coordname1, coordname2 = state_vect_info[var]['spacecoords']
                    # load in the coord values from data dictionary 
                    coord1 = self.prior_dict[var][coordname1]
                    coord2 = self.prior_dict[var][coordname2]                    
                    ndim1 = coord1.shape[0]
                    ndim2 = coord2.shape[0]

                    X_coord1 =  np.array([coord1,]*ndim2).transpose()
                    X_coord2 =  np.array([coord2,]*ndim1)

                    Xb_coords[indstart:indend+1,0] = X_coord1.flatten()
                    Xb_coords[indstart:indend+1,1] = X_coord2.flatten()

                    # Some cleanup
                    del coord1
                    del coord2
                    del X_coord1
                    del X_coord2

                else:
                    print 'ERROR: variable of unrecognized spatial dimensions... Exiting!'
                    exit(1)

            except:
                # No spacecoords: time series
                # Loop over ensemble members
                for i in range(0,self.Nens):
                    Xb[indstart:indend+1,i] = self.prior_dict[var]['value'][ind_ens[i]].flatten()

        # Assign return variables
        self.ens    = Xb
        self.coords = Xb_coords        
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

# class for the CCSM4 Last Millennium simulation
class prior_ccsm4_last_millenium(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,self.prior_datafile,self.statevars,
                                                        self.detrend,self.kind)
        return

# class for the CCSM4 Pre-Industrial Control simulation
class prior_ccsm4_preindustrial_control(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,self.prior_datafile,self.statevars,
                                                        self.detrend,self.kind)
        return

# class for the CCSM4 isotope-enabled control simulation (from D. Noone)
class prior_ccsm4_isotope_controlrun(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,self.prior_datafile,self.statevars,
                                                        self.detrend,self.kind)
        return
    
# class for the MPI-ESM-P Last Millenniun simulation
class prior_mpi_esm_p_last_millenium(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,self.prior_datafile,self.statevars,
                                                        self.detrend,self.kind)
        return

# class for the GFDL-CM3 Pre-Industrial Control simulation
class prior_gfdl_cm3_preindustrial_control(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,self.prior_datafile,self.statevars,
                                                        self.detrend,self.kind)
        return

# class for NOAA's 20th century reanalysis (20CR)
class prior_20cr(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,self.prior_datafile,self.statevars,
                                                        self.detrend,self.kind)
        return

# class for ECMWF's 20th century reanalysis (ERA20C)
class prior_era20c(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,self.prior_datafile,self.statevars,
                                                        self.detrend,self.kind)
        return

# class for ECMWF's 20th century model ensemble (ERA20CM)
class prior_era20cm(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model_ensemble
        self.prior_dict = read_gridded_data_CMIP5_model_ensemble(self.prior_datadir,self.prior_datafile,self.statevars)
        return

