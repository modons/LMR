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
          - Added 'ccsm3_trace21ka' (simulation of the transient climate of 
            the last 21k years) as a possible prior source
            [R. Tardif, U. of Washington, Nov 2016]

"""

import numpy as np
from random import sample, seed
from copy import deepcopy

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
    elif iprior == 'ccsm3_trace21ka':
        prior_object = prior_ccsm3_trace21ka()

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

        # Load prior data from file(s) - multiple state variables
        self.read_prior()

        Nens_max = len(self.prior_dict[self.prior_dict.keys()[0]]['years'])
        if self.Nens and self.Nens > Nens_max:
            raise SystemExit('ERROR in populate_ensemble! Specified ensemble size too large for available nb of states. '
            'Max allowed with current configuration: %d' %Nens_max)

        
        nbvars = len(self.statevars)
        # Check consistency between specified state variables and uploaded dictionary
        if len(self.prior_dict.keys()) != nbvars:
            raise SystemExit('Problem with load of prior state variables. Exiting!')

        
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
            if spacecoords:
                #print 'spacecoords is not None: variable with space coordinates'
                dim1, dim2 = spacecoords

                # How are these defined? Check dims of arrays
                if len(self.prior_dict[var][dim1].shape) == 2 and len(self.prior_dict[var][dim2].shape) == 2:
                    # we have a field defined on an irregular lat/lon grid, requiring lat & lon
                    # each be defined with a 2d array
                    ndim1 = self.prior_dict[var][dim1].shape[0]
                    ndim2 = self.prior_dict[var][dim1].shape[1]
                
                elif len(self.prior_dict[var][dim1].shape) == 1 and len(self.prior_dict[var][dim2].shape) == 1:
                    # regular lat/lon array : lat and lon can be defined with 1d arrays
                    ndim1 = len(self.prior_dict[var][dim1])
                    ndim2 = len(self.prior_dict[var][dim2])

                else:
                    raise SystemExit('ERROR in populate_ensemble: Unrecognized info on spatial dimensions. Exiting!')

                ndimtot = ndim1*ndim2
                
                dct['pos'] = (Nx,Nx+(ndimtot)-1)
                dct['spacecoords'] = spacecoords
                dct['spacedims'] = (ndim1,ndim2)

                if 'lat' in spacecoords and 'lon' in spacecoords:
                    dct['vartype'] = '2D:horizontal'
                elif 'lat' in spacecoords and 'lev' in spacecoords:
                    dct['vartype'] = '2D:meridional_vertical'
                else:
                    raise SystemExit('ERROR in populate_ensemble: Unrecognized info on spatial dimensions. Exiting!')
                    
            else:
                #print 'spacecoords is None: variable is simple time series'
                ndimtot = 1
                dct['pos'] = (Nx,Nx+(ndimtot)-1)
                dct['spacecoords'] = None
                dct['spacedims'] = None
                dct['vartype'] = '1D:time series'
                
            # assign to master dictionary
            state_vect_info[var] = dct

            # determining length of state vector
            Nx = Nx + (ndimtot)

        # Looped through all state variables, now a summary:
        print ' '
        print 'State vector information:'
        print 'Nx =', Nx
        print 'state_vect_info=', state_vect_info

        # time dimension consistent across variables?
        if all(x == timedim[0] for x in timedim):
            ntime = timedim[0]
        else:
            print 'ERROR: time dimension not consistent across all state variables. Exiting!'
            exit(1)

        # If Nens is None, use all of prior with no random sampling
        if self.Nens is None:
            take_sample = False
            self.Nens = ntime
        else:
            take_sample = True

        # Array that will contain the prior ensemble (state vector)
        Xb = np.zeros(shape=[Nx,self.Nens]) # no time dimension now...
        # ***NOTE: Following code assumes that data for a given year are located at same array time index across all state variables

        if take_sample:
            print 'Random selection of', str(self.Nens), 'ensemble members'
            # Populate prior ensemble from randomly sampled states
            seed(prior_cfg.seed)
            ind_ens = sample(range(ntime), self.Nens)
        else:
            print 'Using entire consecutive years in prior dataset.'
            ind_ens = range(ntime)

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

                    # check how coords are defined:
                    # 1d (regular lat/lon grid) or 2d (irregular lat/lon grid)
                    if len(coord1.shape) == 1 and len(coord2.shape) == 1:
                        ndim1 = coord1.shape[0]
                        ndim2 = coord2.shape[0]
                        X_coord1 =  np.array([coord1,]*ndim2).transpose()
                        X_coord2 =  np.array([coord2,]*ndim1)
                    elif len(coord1.shape) == 2 and len(coord2.shape) == 2:
                        ndim1, ndim2 = coord1.shape
                        X_coord1 =  coord1
                        X_coord2 =  coord2

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


        """ Code in development ... commented out for now ...
        # ---------------------------------------------
        # Check for missing values in arrays and delete
        # corresponding entries from state vector.
        # ---------------------------------------------

        # nb of missing values along ensemble member axis (columns),
        # for every data pt. in state vect.
        state_isnan      = np.sum(np.isnan(Xb), axis=1)
        # Tag data pt as missing if one or more ens. member has missing data
        state_isnan_bool = np.greater(state_isnan,0)
        # Binary version (0/1) of this array due to funkiness in array
        # slicing with boolean arrays
        state_isnan_binary = np.zeros([len(state_isnan)],dtype=np.int32)
        state_isnan_binary[state_isnan_bool] = 1

        # Total count of missinfg data pts.
        totnbnan = np.sum(state_isnan_binary)

        # If some missing values found, perform check per state variable
        # and update info on state vector content. Otherwise leave unchanged.
        if totnbnan > 0:
            print('Eliminating missing values from state vector...')

            print ' Xb before        :', Xb.shape
            print ' Xb_coords before :', Xb_coords.shape

            
            Xb = np.delete(Xb, np.where(state_isnan_bool == True), axis=0)
            Xb_coords = np.delete(Xb_coords, np.where(state_isnan_bool == True), axis=0)
            
            # Updating state_vector_info:
            # ---------------------------
            # order of variables in state vector
            indbeg =  [state_vect_info[item]['pos'][0] for item in state_vect_info.keys()]

            #sorted_vars = [k for k,v in state_vect_info.iteritems() if v['pos'][0] in np.sort(indbeg)] # <================= !?!?
            sorted_vars = []
            indbeg_sorted = np.sort(indbeg)
            for i in range(len(indbeg)):
                for v in state_vect_info.keys():
                    if state_vect_info[v]['pos'][0] == indbeg_sorted[i]: sorted_vars.append(v)
            
            state_vect_info_orig = deepcopy(state_vect_info)

            nbmissing = np.zeros([len(sorted_vars)],dtype=np.int32)
            var_nb = 0
            for var in sorted_vars:
                #print '=>', var, state_vect_info_orig[var]['pos']
                ibegin = state_vect_info_orig[var]['pos'][0]
                iend  = state_vect_info_orig[var]['pos'][1]
                #print '=>', ibegin, iend                
                nbmissing[var_nb] = np.sum(state_isnan_binary[ibegin:iend+1])
                #print '=>', var_nb, state_isnan_bool[ibegin:iend+1], state_isnan_binary[ibegin:iend+1], nbmissing
                
                if var_nb == 0:
                    indstart_new = 0
                else:
                    # new start is end position of previous variable plus one
                    indstart_new = state_vect_info[sorted_vars[var_nb-1]]['pos'][1]+1

                nbelem = (iend - ibegin) - nbmissing[var_nb]
                indend_new = indstart_new + nbelem

                
                #if nbmissing[var_nb] > 0:
                    # what about spacedims ????
                    # through the updated Xb_coords ?
                    
                toto = Xb_coords[indstart_new:indend_new+1]
                print ' '
                print '::>', var, toto.shape
                tmplat = toto[:,0]
                tmplon = toto[:,1]
                print '::>', np.unique(tmplat).shape, np.unique(tmplon).shape
                

                state_vect_info[var]['pos'] = (indstart_new, indend_new)
                    
                var_nb += 1

            print ' Xb after        :', Xb.shape
            print ' Xb_coords after :', Xb_coords.shape
            print ' state_vect_info (after)=', state_vect_info
        
        # RT ... ... ...
        """
        
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
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,
                                                        self.prior_datafile,
                                                        self.statevars,
                                                        self.avgInterval,
                                                        self.detrend,
                                                        self.statevars_info)
        return

# class for the CCSM4 Pre-Industrial Control simulation
class prior_ccsm4_preindustrial_control(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,
                                                        self.prior_datafile,
                                                        self.statevars,
                                                        self.avgInterval,
                                                        self.detrend,
                                                        self.statevars_info)
        return

# class for the CCSM4 isotope-enabled control simulation (from D. Noone)
class prior_ccsm4_isotope_controlrun(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,
                                                        self.prior_datafile,
                                                        self.statevars,
                                                        self.avgInterval,
                                                        self.detrend,
                                                        self.statevars_info)
        return
    
# class for the MPI-ESM-P Last Millenniun simulation
class prior_mpi_esm_p_last_millenium(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,
                                                        self.prior_datafile,
                                                        self.statevars,
                                                        self.avgInterval,
                                                        self.detrend,
                                                        self.statevars_info)
        return

# class for the GFDL-CM3 Pre-Industrial Control simulation
class prior_gfdl_cm3_preindustrial_control(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,
                                                        self.prior_datafile,
                                                        self.statevars,
                                                        self.avgInterval,
                                                        self.detrend,
                                                        self.statevars_info)
        return

# class for NOAA's 20th century reanalysis (20CR)
class prior_20cr(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,
                                                        self.prior_datafile,
                                                        self.statevars,
                                                        self.avgInterval,
                                                        self.detrend,
                                                        self.statevars_info)
        return

# class for ECMWF's 20th century reanalysis (ERA20C)
class prior_era20c(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model
        self.prior_dict = read_gridded_data_CMIP5_model(self.prior_datadir,
                                                        self.prior_datafile,
                                                        self.statevars,
                                                        self.avgInterval,
                                                        self.detrend,
                                                        self.statevars_info)
        return

# class for ECMWF's 20th century model ensemble (ERA20CM)
class prior_era20cm(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_CMIP5_model_ensemble
        self.prior_dict = read_gridded_data_CMIP5_model_ensemble(self.prior_datadir,
                                                                 self.prior_datafile,
                                                                 self.statevars)
        return

# class for the simulation of the transient climate of the last 21k years (TraCE21ka)
class prior_ccsm3_trace21ka(prior_master):

    def read_prior(self):
        from load_gridded_data import read_gridded_data_TraCE21ka
        self.prior_dict = read_gridded_data_TraCE21ka(self.prior_datadir,
                                                      self.prior_datafile,
                                                      self.statevars,
                                                      self.avgInterval,
                                                      self.detrend)
        return
