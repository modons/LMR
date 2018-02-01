"""
Module: LMR_driver_callable.py

Purpose: This is the "main" module of the LMR code.
         Generates a paleoclimate reconstruction (single Monte-Carlo
         realization) through the assimilation of a set of proxy data.

Options: None.
         Experiment parameters defined in LMR_config.

Originators: Greg Hakim    | Dept. of Atmospheric Sciences, Univ. of Washington
             Robert Tardif | January 2015

Revisions:
  April 2015:
            - This version is callable by an outside script, accepts a single
              object, called state, which has everything needed for the driver
              (G. Hakim - U. of Washington)

            - Re-organisation of code around PSM calibration and calculation of
              Ye. Code now assumes PSM parameters have been pre-calulated and
              Ye's are calculated up-front for all proxy types/sites. All
              proxy data are now also loaded up-front, prior to any loops.
              Ye's are appended to state vector to form an augmented state
              vector and are also updated by DA. (R. Tardif - U. of Washington)
    May 2015:
            - Bug fix in calculation of global mean temperature + function
              now part of LMR_utils.py (G. Hakim - U. of Washington)
   July 2015:
            - Switched time & proxy loops, simplified logic so more of the
              proxy and PSM specifics are contained within their classes,
              formatted to mostly adhere to PEP8 guidlines
              (A. Perkins - U. of Washington)
  April 2016:
            - Added handling of the "sensitivity" attribute now attached to
              proxy psm objects that defines the climate variable to which
              each proxy record is deemed sensitive to.
              (R. Tardif - U. of Washington)
   July 2016:
            - Slight code adjustments for handling possible use of PSM calibrated 
              on the basis of proxy records seasonality metadata.
              (R. Tardif - U. of Washington)
 August 2016:
            - Introduced new function that loads pre-calculated Ye values 
              generated using psm types assigned to individual proxy types
              as defined in the experiment configuration. 
              (R. Tardif - U. of Washington)
   Feb. 2017:
            - Modifications to temporal loop to allow the production of 
              reconstructions at lower temporal resolution (i.e. other
              than annual).
              (R. Tardif - U. of Washington)
  March 2017:
            - Added possibility to by-pass the regridding (truncation of the state).
              (R. Tardif - U. of Washington)
            - Added another option for regridding that works on gridded 
              fields with missing values (masked grid points. e.g. ocean fields) 
              (R. Tardif - U. of Washington)
            - Replaced the hared-coded truncation resolution (T42) of spatial fields 
              updated during the DA (i.e. reconstruction resolution) by a 
              user-specified value set in the configuration.
 August 2017:
            - Included the Ye's from withheld proxies to state vector so they get 
              updated during DA as well for easier & complete proxy-based evaluation
              of reconstruction. (R. Tardif - U. of Washington)
"""
import numpy as np
from os.path import join
from time import time

import LMR_proxy_pandas_rework
import LMR_prior
import LMR_utils
import LMR_config as BaseCfg
from LMR_DA import enkf_update_array, cov_localization
from LMR_utils import FlagError


def LMR_driver_callable(cfg=None):

    if cfg is None:
        cfg = BaseCfg.Config()  # Use base configuration from LMR_config

    # Temporary fix for old 'state usage'
    core = cfg.core
    prior = cfg.prior

    # verbose controls print comments (0 = none; 1 = most important;
    #  2 = many; 3 = a lot; >=4 = all)
    verbose = cfg.LOG_LEVEL

    nexp = core.nexp
    workdir = core.datadir_output
    recon_period = core.recon_period
    recon_timescale = core.recon_timescale
    online = core.online_reconstruction
    nens = core.nens
    loc_rad = core.loc_rad
    inflation_fact = core.inflation_fact
    prior_source = prior.prior_source
    datadir_prior = prior.datadir_prior
    datafile_prior = prior.datafile_prior
    state_variables = prior.state_variables
    state_variables_info = prior.state_variables_info
    regrid_method = prior.regrid_method
    regrid_resolution = prior.regrid_resolution

    
    # ==========================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN CODE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ==========================================================================
    # TODO: AP Logging instead of print statements
    if verbose > 0:
        print('')
        print('=====================================================')
        print('Running LMR reconstruction...')
        print('=====================================================')
        print('Name of experiment: ', nexp)
        print(' Monte Carlo iter : ', core.curr_iter)
        print('')
        
    begin_time = time()

    # Define the number of years of the reconstruction
    # (nb of assimilation times)
    recon_times = np.arange(recon_period[0], recon_period[1]+1,recon_timescale)
    ntimes, = recon_times.shape

    # ==========================================================================
    # Load prior data ----------------------------------------------------------
    # ==========================================================================
    if verbose > 0:
        print('-------------------------------------------')
        print('Uploading gridded (model) data as prior ...')
        print('-------------------------------------------')
        print('Source for prior: ', prior_source)

    # Assign prior object according to "prior_source" (from namelist)
    X = LMR_prior.prior_assignment(prior_source)

    # TODO: AP explicit requirements
    # add namelist attributes to the prior object
    X.prior_datadir = datadir_prior
    X.prior_datafile = datafile_prior
    X.statevars = state_variables
    X.statevars_info = state_variables_info
    X.Nens = nens
    # Use a specified reference period for state variable anomalies 
    X.anom_reference = prior.anom_reference
    # new option: detrending the prior
    X.detrend = prior.detrend
    print('detrend:', X.detrend)
    X.avgInterval = prior.avgInterval
    
    # Read data file & populate initial prior ensemble
    X.populate_ensemble(prior_source, prior)
    Xb_one_full = X.ens

    
    # Prepare to check for files in the prior (work) directory (this object just
    # points to a directory)
    prior_check = np.DataSource(workdir)

    load_time = time() - begin_time
    if verbose > 2:
        print('-----------------------------------------------------')
        print('Loading completed in ' + str(load_time)+' seconds')
        print('-----------------------------------------------------')

    # check covariance inflation from config
    inflate = None
    if inflation_fact is not None:
        inflate = inflation_fact
        if verbose > 2:            
            print(('\nUsing covariance inflation factor: %8.2f' %inflate))
        
    # ==========================================================================
    # Get information on proxies to assimilate ---------------------------------
    # ==========================================================================

    begin_time_proxy_load = time()
    if verbose > 0:
        print('')
        print('-----------------------------------')
        print('Uploading proxy data & PSM info ...')
        print('-----------------------------------')

    # Build dictionaries of proxy sites to assimilate and those set aside for
    # verification
    prox_manager = LMR_proxy_pandas_rework.ProxyManager(cfg, recon_period)
    type_site_assim = prox_manager.assim_ids_by_group

    if verbose > 3:
        print('Assimilating proxy types/sites:', type_site_assim)

    if verbose > 0:
        print('--------------------------------------------------------------------')
        print('Proxy counts for experiment:')
        # count the total number of proxies
        assim_proxy_count = len(prox_manager.ind_assim)
        for pkey, plist in sorted(type_site_assim.items()):
            print(('%45s : %5d' % (pkey, len(plist))))
        print(('%45s : %5d' % ('TOTAL', assim_proxy_count)))
        print('--------------------------------------------------------------------')

    if verbose > 2:
        proxy_load_time = time() - begin_time_proxy_load
        print('-----------------------------------------------------')
        print('Loading completed in ' + str(proxy_load_time) + ' seconds')
        print('-----------------------------------------------------')


        
    # ==========================================================================
    # Calculate truncated state from prior, if option chosen -------------------
    # ==========================================================================
    if regrid_method:
        
        # Declare dictionary w/ info on content of truncated state vector
        new_state_info = {}

        # Transform every 2D state variable, one at a time
        Nx = 0
        for var in list(X.full_state_info.keys()):
            dct = {}

            dct['vartype'] = X.full_state_info[var]['vartype']

            # variable indices in full state vector
            ibeg_full = X.full_state_info[var]['pos'][0]
            iend_full = X.full_state_info[var]['pos'][1]
            # extract array corresponding to state variable "var"
            var_array_full = Xb_one_full[ibeg_full:iend_full+1, :]
            # corresponding spatial coordinates
            coords_array_full = X.coords[ibeg_full:iend_full+1, :]

            # Are we truncating this variable? (i.e. is it a 2D lat/lon variable?)

            if X.full_state_info[var]['vartype'] == '2D:horizontal':
                print(var, ' : 2D lat/lon variable, truncating this variable')
                # lat/lon column indices in X.coords
                ind_lon = X.full_state_info[var]['spacecoords'].index('lon')
                ind_lat = X.full_state_info[var]['spacecoords'].index('lat')
                nlat = X.full_state_info[var]['spacedims'][ind_lat]
                nlon = X.full_state_info[var]['spacedims'][ind_lon]

                # calculate the truncated fieldNtimes
                if regrid_method == 'simple':
                    [var_array_new, lat_new, lon_new] = \
                        LMR_utils.regrid_simple(nens, var_array_full, coords_array_full, \
                                                ind_lat, ind_lon, regrid_resolution)
                elif regrid_method == 'spherical_harmonics':
                    [var_array_new, lat_new, lon_new] = \
                        LMR_utils.regrid_sphere(nlat, nlon, nens, var_array_full, regrid_resolution)
                elif regrid_method == 'esmpy':
                    target_grid = prior.esmpy_grid_def

                    lat_2d = coords_array_full[:, ind_lat].reshape(nlat, nlon)
                    lon_2d = coords_array_full[:, ind_lon].reshape(nlat, nlon)

                    [var_array_new,
                     lat_new,
                     lon_new] = LMR_utils.regrid_esmpy(target_grid['nlat'],
                                                       target_grid['nlon'],
                                                       nens,
                                                       var_array_full,
                                                       lat_2d,
                                                       lon_2d,
                                                       nlat,
                                                       nlon,
                                                       method=prior.esmpy_interp_method)
                else:
                    print('Exiting! Unrecognized regridding method.')
                    raise SystemExit
                
                nlat_new = np.shape(lat_new)[0]
                nlon_new = np.shape(lat_new)[1]
                
                print(('=> Full array:      ' + str(np.min(var_array_full)) + ' ' +
                       str(np.max(var_array_full)) + ' ' + str(np.mean(var_array_full)) +
                       ' ' + str(np.std(var_array_full))))
                print(('=> Truncated array: ' + str(np.min(var_array_new)) + ' ' +
                       str(np.max(var_array_new)) + ' ' + str(np.mean(var_array_new)) +
                       ' ' + str(np.std(var_array_new))))

                # corresponding indices in truncated state vector
                ibeg_new = Nx
                iend_new = Nx+(nlat_new*nlon_new)-1
                # for new state info dictionary
                dct['pos'] = (ibeg_new, iend_new)
                dct['spacecoords'] = X.full_state_info[var]['spacecoords']
                dct['spacedims'] = (nlat_new, nlon_new)
                # updated dimension
                new_dims = (nlat_new*nlon_new)

                # array with new spatial coords
                coords_array_new = np.zeros(shape=[new_dims, 2])
                coords_array_new[:, 0] = lat_new.flatten()
                coords_array_new[:, 1] = lon_new.flatten()
                
            else:
                print(var,\
                    ' : not truncating this variable: no changes from full state')

                var_array_new = var_array_full
                coords_array_new = coords_array_full
                # updated dimension
                new_dims = var_array_new.shape[0]
                ibeg_new = Nx
                iend_new = Nx + new_dims - 1
                dct['pos'] = (ibeg_new, iend_new)
                dct['spacecoords'] = X.full_state_info[var]['spacecoords']
                dct['spacedims'] = X.full_state_info[var]['spacedims']

            
            # fill in new state info dictionary
            new_state_info[var] = dct

            # if 1st time in loop over state variables, create Xb_one array as copy
            # of var_array_new
            if Nx == 0:
                Xb_one = np.copy(var_array_new)
                Xb_one_coords = np.copy(coords_array_new)
            else:  # if not 1st time, append to existing array
                Xb_one = np.append(Xb_one, var_array_new, axis=0)
                Xb_one_coords = np.append(Xb_one_coords, coords_array_new, axis=0)
            
            # making sure Xb_one has proper mask, if it contains
            # at least one invalid value
            if np.isnan(Xb_one).any():        
                Xb_one = np.ma.masked_invalid(Xb_one)
                np.ma.set_fill_value(Xb_one, np.nan)
            
            # updating dimension of new state vector
            Nx = Nx + new_dims

        X.trunc_state_info = new_state_info

    else: # no truncation: carry over full state to working array
         X.trunc_state_info = X.full_state_info
         Xb_one = Xb_one_full
         Xb_one_coords = X.coords
         
         [Nx, _] = Xb_one.shape

    # Keep dimension of pre-augmented version of state vector
    [state_dim, _] = Xb_one.shape
    
    
    # ==========================================================================
    # Calculate all Ye's (for all sites in sites_assim) ------------------------
    # ==========================================================================
    
    # Load or generate Ye Values for assimilation
    if not online:
        # Load pre calculated ye values if desired or possible
        try:
            if not cfg.core.use_precalc_ye:
                raise FlagError('use_precalc_ye=False: forego loading precalcul'
                                'ated ye values.')

            print('Loading precalculated Ye values for proxies to be assimilated.')
            [Ye_assim, Ye_assim_coords] = LMR_utils.load_precalculated_ye_vals_psm_per_proxy(cfg, prox_manager,
                                                    'assim', X.prior_sample_indices)

            eval_proxy_count = 0
            if prox_manager.ind_eval:
                print('Loading precalculated Ye values for withheld proxies.')
                [Ye_eval, Ye_eval_coords] = LMR_utils.load_precalculated_ye_vals_psm_per_proxy(cfg,
                                                    prox_manager, 'eval', X.prior_sample_indices)
                [eval_proxy_count,_] = Ye_eval.shape
            
        except (IOError, FlagError) as e:
            print(e)

            # Manually calculate ye_values from state vector
            print('Calculating ye_values from the prior...')
            Ye_assim = np.empty(shape=[assim_proxy_count, nens])
            Ye_assim_coords = np.empty(shape=[assim_proxy_count, 2])
            for k, proxy in enumerate(prox_manager.sites_assim_proxy_objs()):
                Ye_assim[k, :] = proxy.psm(Xb_one_full,
                                         X.full_state_info,
                                         X.coords)
                Ye_assim_coords[k, :] = np.asarray([proxy.lat, proxy.lon], dtype=np.float64)


            eval_proxy_count = 0
            if prox_manager.ind_eval:
                eval_proxy_count = len(prox_manager.ind_eval)
                Ye_eval = np.empty(shape=[eval_proxy_count, nens])
                Ye_eval_coords = np.empty(shape=[eval_proxy_count, 2])
                for k, proxy in enumerate(prox_manager.sites_eval_proxy_objs()):
                    Ye_eval[k, :] = proxy.psm(Xb_one_full,
                                              X.full_state_info,
                                              X.coords)
                    Ye_eval_coords[k, :] = np.asarray([proxy.lat, proxy.lon], dtype=np.float64)


        # ----------------------------------
        # Augment state vector with the Ye's
        # ----------------------------------
        # Append ensemble of Ye's of assimilated proxies to prior state vector
        Xb_one_aug = np.append(Xb_one, Ye_assim, axis=0)
        Xb_one_coords = np.append(Xb_one_coords, Ye_assim_coords, axis=0)

        if prox_manager.ind_eval:
            # Append ensemble of Ye's of withheld proxies to prior state vector
            Xb_one_aug = np.append(Xb_one_aug, Ye_eval, axis=0)
            Xb_one_coords = np.append(Xb_one_coords, Ye_eval_coords, axis=0)
        
    else:
        Xb_one_aug = Xb_one

    
    # Dump entire prior state vector (Xb_one) to file 
    filen = workdir + '/' + 'Xb_one'
    try:
        out_Xb_one = Xb_one.filled()
        out_Xb_one_aug = Xb_one_aug.filled()
    except AttributeError as e:
        out_Xb_one = Xb_one
        out_Xb_one_aug = Xb_one_aug

    np.savez(filen, Xb_one=out_Xb_one, Xb_one_aug=out_Xb_one_aug,
             stateDim=state_dim,
             Xb_one_coords=Xb_one_coords, state_info=X.trunc_state_info)

    # NEW: Dump prior state vector (Xb_one) to file, one file per state variable
    print('\n ---------- saving Xb_one for each variable to separate file -----------\n')
    for var in list(X.trunc_state_info.keys()):
        print(var)
        # now need to pluck off the index region that goes with var
        ibeg = X.trunc_state_info[var]['pos'][0]
        iend = X.trunc_state_info[var]['pos'][1]

        if X.trunc_state_info[var]['vartype'] == '2D:horizontal':
            # if no truncation: lat_new and lon_new are not defined...rather get actual lats/lons info from state vector
            ind_lon = X.trunc_state_info[var]['spacecoords'].index('lon')
            ind_lat = X.trunc_state_info[var]['spacecoords'].index('lat')

            nlon_new = X.trunc_state_info[var]['spacedims'][ind_lon]
            nlat_new = X.trunc_state_info[var]['spacedims'][ind_lat]

            lat_sv = Xb_one_coords[ibeg:iend+1, ind_lat]
            lon_sv = Xb_one_coords[ibeg:iend+1, ind_lon]

            lat_new = np.unique(lat_sv)
            lon_new = np.unique(lon_sv)

            Xb_var = np.reshape(out_Xb_one[ibeg:iend+1,:],(nlat_new,nlon_new,nens))

            filen = workdir + '/' + 'Xb_one' + '_' + var 
            np.savez(filen,Xb_var=Xb_var,nlat=nlat_new,nlon=nlon_new,nens=nens,lat=lat_new,lon=lon_new)

        else:
            print(('Warning: Only saving 2D:horizontal variable. Variable (%s) is of another type' %(var)))
            # TODO: Code mods above are a quick fix. Should allow saving other types of variables here!
    # END new file save
    
    
    # ==========================================================================
    # Loop over all years & proxies and perform assimilation -------------------
    # ==========================================================================

    # Array containing the global and hemispheric-mean state
    # (for diagnostic purposes)
    # Now doing surface air temperature only (var = tas_sfc_Amon)!

    # TODO: AP temporary fix for no TAS in state
    tas_var = [item for item in cfg.prior.state_variables.keys() if 'tas_sfc_' in item]
    if tas_var:
        gmt_save = np.zeros([assim_proxy_count+1,ntimes])
        nhmt_save = np.zeros([assim_proxy_count+1,ntimes])
        shmt_save = np.zeros([assim_proxy_count+1,ntimes])
        # get state vector indices where to find surface air temperature
        ibeg_tas = X.trunc_state_info[tas_var[0]]['pos'][0]
        iend_tas = X.trunc_state_info[tas_var[0]]['pos'][1]
        xbm = np.mean(Xb_one[ibeg_tas:iend_tas+1, :], axis=1)  # ensemble-mean

        nlat_new = X.trunc_state_info[tas_var[0]]['spacedims'][0]
        nlon_new = X.trunc_state_info[tas_var[0]]['spacedims'][1]
        xbm_lalo = xbm.reshape(nlat_new, nlon_new)
        lat_coords = Xb_one_coords[ibeg_tas:iend_tas+1, 0]
        lat_lalo = lat_coords.reshape(nlat_new, nlon_new)

        [gmt,nhmt,shmt] = LMR_utils.global_hemispheric_means(xbm_lalo, lat_lalo[:, 0])

        # First row is prior GMT
        gmt_save[0, :] = gmt
        nhmt_save[0,:] = nhmt
        shmt_save[0,:] = shmt
        # Prior for first proxy assimilated
        gmt_save[1, :] = gmt
        nhmt_save[1,:] = nhmt
        shmt_save[1,:] = shmt

    # -------------------------------------
    # Loop over years of the reconstruction
    # -------------------------------------
    lasttime = time()
    for yr_idx, t in enumerate(range(recon_period[0], recon_period[1]+1, recon_timescale)):
        
        start_yr = int(t-recon_timescale//2)
        end_yr = int(t+recon_timescale//2)
        
        if verbose > 0:
            if start_yr == end_yr:
                time_str = 'year: '+str(t)
            else:
                time_str = 'time period (yrs): ['+str(start_yr)+','+str(end_yr)+']'
            print('\n==== Working on ' + time_str)

        ypad = '{:07d}'.format(t)
        filen = join(workdir, 'year' + ypad + '.npy')
        if prior_check.exists(filen) and not core.clean_start:
            if verbose > 2:
                print('prior file exists: ' + filen)
            Xb = np.load(filen)
        else:
            if verbose > 2:
                print('Prior file ', filen, ' does not exist...')
            Xb = Xb_one_aug.copy()

            
        # -----------------
        # Loop over proxies
        # -----------------
        for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
            # Check if we have proxy ob for current time interval
            try:
                if recon_timescale > 1:
                    # exclude lower bound to not include same obs in adjacent time intervals
                    Yvals = Y.values[(Y.values.index > start_yr) & (Y.values.index <= end_yr)]
                else:
                    Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]
                if Yvals.empty: raise KeyError()
                nYobs = len(Yvals)
                Yobs =  Yvals.mean()
                
            except KeyError:
                # Make sure GMT spot filled from previous proxy
                # TODO: AP temporary fix for no TAS in state
                if tas_var:
                    gmt_save[proxy_idx+1, yr_idx] = gmt_save[proxy_idx, yr_idx]
                continue # skip to next loop iteration (proxy record)

            if verbose > 1:
                print('--------------- Processing proxy: ' + Y.id)
            if verbose > 2:
                print('Site:', Y.id, ':', Y.type)
                print(' latitude, longitude: ' + str(Y.lat), str(Y.lon))

            loc = None
            if loc_rad is not None:
                if verbose > 2:
                    print('...computing localization...')
                loc = cov_localization(loc_rad, Y, X, Xb_one_coords)

            # Get Ye values for current proxy
            if online:
                # Calculate from latest updated prior
                Ye = Y.psm(Xb)
            else:
                # Extract latest updated Ye from appended state vector
                Ye = Xb[proxy_idx - (assim_proxy_count+eval_proxy_count)]

            # Define the ob error variance
            ob_err = Y.psm_obj.R

            # if ob is an average of several values, adjust its ob error variance
            if nYobs > 1: ob_err = ob_err/float(nYobs)
            
            # ------------------------------------------------------------------
            # Do the update (assimilation) -------------------------------------
            # ------------------------------------------------------------------
            if verbose > 2:
                print(('updating time: ' + str(t) + ' proxy value : ' +
                       str(Yobs) + ' (nobs=' + str(nYobs) +') | mean prior proxy estimate: ' +
                       str(Ye.mean())))

            # Update the state
            Xa = enkf_update_array(Xb, Yobs, Ye, ob_err, loc, inflate)

            
            # TODO: AP Temporary fix for no TAS in state
            if tas_var:
                xam = Xa.mean(axis=1)
                xam_lalo = xam[ibeg_tas:(iend_tas+1)].reshape(nlat_new, nlon_new)
                [gmt, nhmt, shmt] = \
                    LMR_utils.global_hemispheric_means(xam_lalo, lat_lalo[:, 0])
                gmt_save[proxy_idx+1, yr_idx] = gmt
                nhmt_save[proxy_idx+1, yr_idx] = nhmt
                shmt_save[proxy_idx+1, yr_idx] = shmt

            # check the variance change for sign
            thistime = time()
            if verbose > 2:
                xbvar = Xb.var(axis=1, ddof=1)
                xavar = Xa.var(ddof=1, axis=1)
                vardiff = xavar - xbvar
                print('min/max change in variance: ('+str(np.min(vardiff))+','+str(np.max(vardiff))+')')
                print('update took ' + str(thistime-lasttime) + 'seconds')
            lasttime = thistime

            # Put analysis Xa in Xb for next assimilation
            Xb = Xa

            # End of loop on proxies

        # Dump Xa to file (use Xb in case no proxies assimilated for
        # current year)
        try:
            np.save(filen, Xb.filled())
        except AttributeError as e:
            np.save(filen, Xb)

    end_time = time() - begin_time

    # End of loop on years
    if verbose > 0:
        print('')
        print('=====================================================')
        print('Reconstruction completed in ' + str(end_time/60.0)+' mins')
        print('=====================================================')

    # 3 July 2015: compute and save the GMT,NHMT,SHMT for the full ensemble
    # need to fix this so that every year is counted
    # TODO: AP temporary fix for no TAS
    if tas_var:
        gmt_ensemble = np.zeros([ntimes, nens])
        nhmt_ensemble = np.zeros([ntimes,nens])
        shmt_ensemble = np.zeros([ntimes,nens])
        for iyr, yr in enumerate(range(recon_period[0], recon_period[1]+1, recon_timescale)):
            filen = join(workdir, 'year{:07d}'.format(yr))
            Xa = np.load(filen+'.npy')
            for k in range(nens):
                xam_lalo = Xa[ibeg_tas:iend_tas+1, k].reshape(nlat_new,nlon_new)
                [gmt, nhmt, shmt] = \
                    LMR_utils.global_hemispheric_means(xam_lalo, lat_lalo[:, 0])
                gmt_ensemble[iyr, k] = gmt
                nhmt_ensemble[iyr, k] = nhmt
                shmt_ensemble[iyr, k] = shmt

        filen = join(workdir, 'gmt_ensemble')
        np.savez(filen, gmt_ensemble=gmt_ensemble, nhmt_ensemble=nhmt_ensemble,
                 shmt_ensemble=shmt_ensemble, recon_times=recon_times)

        # save global mean temperature history and the proxies assimilated
        print(('saving global mean temperature update history and ',
               'assimilated proxies...'))
        filen = join(workdir, 'gmt')
        np.savez(filen, gmt_save=gmt_save, nhmt_save=nhmt_save, shmt_save=shmt_save,
                 recon_times=recon_times,
                 apcount=assim_proxy_count,
                 tpcount=assim_proxy_count)

    # TODO: (AP) The assim/eval lists of lists instead of lists of 1-item dicts
    assimilated_proxies = [{p.type: [p.id, p.lat, p.lon, p.time,
                                     p.psm_obj.sensitivity]}
                           for p in prox_manager.sites_assim_proxy_objs()]
    filen = join(workdir, 'assimilated_proxies')
    np.save(filen, assimilated_proxies)
    
    # collecting info on non-assimilated proxies and save to file
    nonassimilated_proxies = [{p.type: [p.id, p.lat, p.lon, p.time,
                                        p.psm_obj.sensitivity]}
                              for p in prox_manager.sites_eval_proxy_objs()]
    if nonassimilated_proxies:
        filen = join(workdir, 'nonassimilated_proxies')
        np.save(filen, nonassimilated_proxies)

    exp_end_time = time() - begin_time
    if verbose > 0:
        print('')
        print('=====================================================')
        print('Experiment completed in ' + str(exp_end_time/60.0) + ' mins')
        print('=====================================================')

    # TODO: best method for Ye saving?
    return prox_manager.sites_assim_proxy_objs(), prox_manager.sites_eval_proxy_objs()
# ------------------------------------------------------------------------------
# --------------------------- end of main code ---------------------------------
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    LMR_driver_callable()
