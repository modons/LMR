
# ==============================================================================
# Program: LMR_driver_callable.py
# 
# Purpose: 
#
# Options: None. 
#          Experiment parameters through namelist, passed through object called
#          "state"
# 
# Originators: Greg Hakim   | Dept. of Atmospheric Sciences, Univ. of Washington
#              Robert Tardif | January 2015
# 
# Revisions: 
#  April 2015:
#            - This version is callable by an outside script, accepts a single
#              object, called state, which has everything needed for the driver
#              (G. Hakim)

#            - Re-organisation of code around PSM calibration and calculation of
#              Code now assumes PSM parameters have been pre-calulated and
#              Ye's are calculated up-front for all proxy types/sites. All
#              proxy data are now also loaded up-front, prior to any loops.
#              Ye's are appended to state vector to form an augmented state
#              vector and are also updated by DA. (R. Tardif)
#  May 2015:
#            - Bug fix in calculation of global mean temperature + function
#              now part of LMR_utils.py (G. Hakim)
#  July 2015:
#            - Switched time & proxy loops, simplified logic so more of the
#              proxy and PSM specifics are contained within their classes,
#              formatted to mostly adhere to PEP8 guidlines
#              (A. Perkins)
# ==============================================================================

import numpy as np
from os.path import join
from time import time

import LMR_proxy_pandas_rework
import LMR_prior
import LMR_utils
import LMR_config as BaseCfg
from LMR_DA import enkf_update_array, cov_localization


def LMR_driver_callable(cfg=None):

    if cfg is None:
        cfg = BaseCfg  # Use base configuration from LMR_config

    # Temporary fix for old 'state usage'
    core = cfg.core
    prior = cfg.prior

    # verbose controls print comments (0 = none; 1 = most important;
    #  2 = many; >=3 = all)
    verbose = 1

    nexp = core.nexp
    workdir = core.datadir_output
    recon_period = core.recon_period
    online = core.online_reconstruction
    nens = core.nens
    loc_rad = core.loc_rad
    prior_source = prior.prior_source
    datadir_prior = prior.datadir_prior
    datafile_prior = prior.datafile_prior
    state_variables = prior.state_variables

    # ==========================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN CODE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ==========================================================================
    # TODO: AP Logging instead of print statements
    if verbose > 0:
        print ''
        print '====================================================='
        print 'Running LMR reconstruction...'
        print '====================================================='
        print 'Name of experiment: ', nexp
        print ' Monte Carlo iter : ', core.curr_iter
        print ''
        
    begin_time = time()

    # Define the number of years of the reconstruction (nb of assimilation
    # times)
    # Note: recon_period is defined in namelist
    ntimes = recon_period[1] - recon_period[0] + 1
    recon_times = np.arange(recon_period[0], recon_period[1]+1)

    # ==========================================================================
    # Load prior data ----------------------------------------------------------
    # ==========================================================================
    if verbose > 0:
        print '-------------------------------------------'
        print 'Uploading gridded (model) data as prior ...'
        print '-------------------------------------------'
        print 'Source for prior: ', prior_source

    # Assign prior object according to "prior_source" (from namelist)
    X = LMR_prior.prior_assignment(prior_source)

    # TODO: AP explicit requirements
    # add namelist attributes to the prior object
    X.prior_datadir = datadir_prior
    X.prior_datafile = datafile_prior
    X.statevars = state_variables
    X.Nens = nens

    # Read data file & populate initial prior ensemble
    X.populate_ensemble(prior_source)
    Xb_one_full = X.ens

    # Prepare to check for files in the prior (work) directory (this object just
    #  points to a directory)
    prior_check = np.DataSource(workdir)

    load_time = time() - begin_time
    if verbose > 2:
        print '-----------------------------------------------------'
        print 'Loading completed in ' + str(load_time)+' seconds'
        print '-----------------------------------------------------'

    # ==========================================================================
    # Get information on proxies to assimilate ---------------------------------
    # ==========================================================================

    begin_time_proxy_load = time()
    if verbose > 0:
        print ''
        print '-----------------------------------'
        print 'Uploading proxy data & PSM info ...'
        print '-----------------------------------'

    # Build dictionaries of proxy sites to assimilate and those set aside for
    # verification
    prox_manager = LMR_proxy_pandas_rework.ProxyManager(BaseCfg, recon_period)
    type_site_assim = prox_manager.assim_ids_by_group

    if verbose > 0:
        print 'Assimilating proxy types/sites:', type_site_assim

    # ==========================================================================
    # Calculate all Ye's (for all sites in sites_assim) ------------------------
    # ==========================================================================

    print '--------------------------------------------------------------------'
    print 'Proxy counts for experiment:'
    # count the total number of proxies
    total_proxy_count = len(prox_manager.ind_assim)
    for pkey, plist in type_site_assim.iteritems():
        print('%45s : %5d' % (pkey, len(plist)))
    print('%45s : %5d' % ('TOTAL', total_proxy_count))
    print '--------------------------------------------------------------------'

    proxy_load_time = time() - begin_time_proxy_load
    if verbose > 2:
        print '-----------------------------------------------------'
        print 'Loading completed in ' + str(proxy_load_time) + ' seconds'
        print '-----------------------------------------------------'

    # ==========================================================================
    # Calculate truncated state from prior, if option chosen -------------------
    # ==========================================================================

        # Handle state vector with multiple state variables

    # Declare dictionary w/ info on content of truncated state vector
    new_state_info = {}

    # Transform every 2D state variable, one at a time
    Nx = 0
    for var in X.full_state_info.keys():
        dct = {}
        # variable indices in full state vector
        ibeg_full = X.full_state_info[var]['pos'][0]
        iend_full = X.full_state_info[var]['pos'][1]
        # extract array corresponding to state variable "var"
        var_array_full = Xb_one_full[ibeg_full:iend_full+1, :]
        # corresponding spatial coordinates
        coords_array_full = X.coords[ibeg_full:iend_full+1, :]

        # Are we truncating this variable? (i.e. is it a 2D lat/lon variable?)
        if (X.full_state_info[var]['spacecoords'] and
            'lat' in X.full_state_info[var]['spacecoords'] and
            'lon' in X.full_state_info[var]['spacecoords']):

            print var, ' : 2D lat/lon variable, truncating this variable'
            # lat/lon column indices in X.coords
            ind_lon = X.full_state_info[var]['spacecoords'].index('lon')
            ind_lat = X.full_state_info[var]['spacecoords'].index('lat')
            nlat = X.full_state_info[var]['spacedims'][ind_lat]
            nlon = X.full_state_info[var]['spacedims'][ind_lon]

            # calculate the truncated fieldNtimes
            [var_array_new, lat_new, lon_new] = \
                LMR_utils.regrid_sphere(nlat, nlon, nens, var_array_full, 42)
            nlat_new = np.shape(lat_new)[0]
            nlon_new = np.shape(lat_new)[1]

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
            print var,\
                ' : not truncating this variable: no changes from full state'
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

        # updating dimension of new state vector
        Nx = Nx + new_dims

    X.trunc_state_info = new_state_info

    # Keep dimension of pre-augmented version of state vector
    [state_dim, _] = Xb_one.shape

    # ----------------------------------
    # Augment state vector with the Ye's
    # ----------------------------------

    # Extract all the Ye's from master list of proxy objects into numpy array
    if not online:
        Ye_all = np.empty(shape=[total_proxy_count, nens])
        for k, proxy in enumerate(prox_manager.sites_assim_proxy_objs()):
            Ye_all[k, :] = proxy.psm(Xb_one_full, X.full_state_info, X.coords)

        # Append ensemble of Ye's to prior state vector
        Xb_one_aug = np.append(Xb_one, Ye_all, axis=0)
    else:
        Xb_one_aug = Xb_one

    # Dump prior state vector (Xb_one) to file 
    filen = workdir + '/' + 'Xb_one'
    np.savez(filen, Xb_one=Xb_one, Xb_one_aug=Xb_one_aug, stateDim=state_dim,
             Xb_one_coords=Xb_one_coords, state_info=X.trunc_state_info)

    # ==========================================================================
    # Loop over all proxies and perform assimilation ---------------------------
    # ==========================================================================

    # ---------------------
    # Loop over proxy types
    # ---------------------

    # Array containing the global and hemispheric-mean state (for diagnostic purposes)
    # Now doing surface air temperature only (var = tas_sfc_Amon)!
    gmt_save = np.zeros([total_proxy_count+1,recon_period[1] - recon_period[0] + 1])
    nhmt_save = np.zeros([total_proxy_count+1,recon_period[1]-recon_period[0]+1])
    shmt_save = np.zeros([total_proxy_count+1,recon_period[1]-recon_period[0]+1])
    # get state vector indices where to find surface air temperature
    ibeg_tas = X.trunc_state_info['tas_sfc_Amon']['pos'][0]
    iend_tas = X.trunc_state_info['tas_sfc_Amon']['pos'][1]
    xbm = np.mean(Xb_one[ibeg_tas:iend_tas+1, :], axis=1)  # ensemble-mean
    xbm_lalo = xbm.reshape(nlat_new, nlon_new)
    [gmt,nhmt,shmt] = LMR_utils.global_hemispheric_means(xbm_lalo, lat_new[:, 0])
    # First row is prior GMT
    gmt_save[0, :] = gmt
    nhmt_save[0,:] = nhmt
    shmt_save[0,:] = shmt
    # Prior for first proxy assimilated
    gmt_save[1, :] = gmt 
    nhmt_save[1,:] = nhmt
    shmt_save[1,:] = shmt

    lasttime = time()
    for yr_idx, t in enumerate(xrange(recon_period[0], recon_period[1]+1)):

        if verbose > 0:
            print 'working on year: ' + str(t)

        ypad = '{:04d}'.format(t)
        filen = join(workdir, 'year' + ypad + '.npy')
        if prior_check.exists(filen) and not core.clean_start:
            if verbose > 2:
                print 'prior file exists: ' + filen
            Xb = np.load(filen)
        else:
            if verbose > 2:
                print 'Prior file ', filen, ' does not exist...'
            Xb = Xb_one_aug.copy()

        for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
            # Crude check if we have proxy ob for current time
            try:
                Y.values[t]
            except KeyError:
                # Make sure GMT spot filled from previous proxy
                gmt_save[proxy_idx+1, yr_idx] = gmt_save[proxy_idx, yr_idx]
                continue

            if verbose > 2:
                print '--------------- Processing proxy: ' + Y.id

            if verbose > 1:
                print ''
                print 'Site:', Y.id, ':', Y.type
                print ' latitude, longitude: ' + str(Y.lat), str(Y.lon)

            loc = None
            if loc_rad is not None:
                if verbose > 2:
                    print '...computing localization...'
                    loc = cov_localization(loc_rad, X, Y)

            # Get Ye values for current proxy
            if online:
                Ye = Y.psm(Xb)
            else:
                Ye = Xb[proxy_idx - total_proxy_count]

            # Define the ob error variance
            ob_err = Y.psm_obj.R

            # ------------------------------------------------------------------
            # Do the update (assimilation) -------------------------------------
            # ------------------------------------------------------------------
            if verbose > 2:
                print ('updating time: ' + str(t) + ' proxy value : ' +
                       str(Y.values[t]) + ' | mean prior proxy estimate: ' +
                       str(Ye.mean()))

            # Update the state
            Xa = enkf_update_array(Xb, Y.values[t], Ye, ob_err, loc)
            xam = Xa.mean(axis=1)
            xam_lalo = xam[ibeg_tas:(iend_tas+1)].reshape(nlat_new, nlon_new)
            [gmt, nhmt, shmt] = \
                LMR_utils.global_hemispheric_means(xam_lalo, lat_new[:, 0])
            gmt_save[proxy_idx+1, yr_idx] = gmt
            nhmt_save[proxy_idx+1, yr_idx] = nhmt
            shmt_save[proxy_idx+1, yr_idx] = shmt

            # check the variance change for sign
            thistime = time()
            if verbose > 2:
                xbvar = Xb.var(axis=1, ddof=1)
                xavar = Xa.var(ddof=1, axis=1)
                vardiff = xavar - xbvar
                print 'max change in variance:' + str(np.max(vardiff))
                print 'update took ' + str(thistime-lasttime) + 'seconds'
            lasttime = thistime
            # Put analysis Xa in Xb for next assimilation
            Xb = Xa

        # Dump Xa to file (use Xb in case no proxies assimilated for current year)
        np.save(filen, Xb)

    end_time = time() - begin_time

    # End of loop on proxy types
    if verbose > 0:
        print ''
        print '====================================================='
        print 'Reconstruction completed in ' + str(end_time/60.0)+' mins'
        print '====================================================='

    # 3 July 2015: compute and save the GMT,NHMT,SHMT for the full ensemble
    # need to fix this so that every year is counted
    gmt_ensemble = np.zeros([ntimes, nens])
    nhmt_ensemble = np.zeros([ntimes,nens])
    shmt_ensemble = np.zeros([ntimes,nens])
    for iyr, yr in enumerate(xrange(recon_period[0], recon_period[1]+1)):
        filen = join(workdir, 'year{:04d}'.format(yr))
        Xa = np.load(filen+'.npy')
        for k in xrange(nens):
            xam_lalo = Xa[ibeg_tas:iend_tas+1, k].reshape(nlat_new,nlon_new)
            [gmt, nhmt, shmt] = \
                LMR_utils.global_hemispheric_means(xam_lalo, lat_new[:, 0])
            gmt_ensemble[iyr, k] = gmt
            nhmt_ensemble[iyr, k] = nhmt
            shmt_ensemble[iyr, k] = shmt

    filen = join(workdir, 'gmt_ensemble')
    np.savez(filen, gmt_ensemble=gmt_ensemble, nhmt_ensemble=nhmt_ensemble,
             shmt_ensemble=shmt_ensemble, recon_times=recon_times)

    # save global mean temperature history and the proxies assimilated
    print ('saving global mean temperature update history and ',
           'assimilated proxies...')
    filen = join(workdir, 'gmt')
    np.savez(filen, gmt_save=gmt_save, nhmt_save=nhmt_save, shmt_save=shmt_save,
             recon_times=recon_times,
             apcount=total_proxy_count,
             tpcount=total_proxy_count)

    # TODO: (AP) The assim/eval lists of lists instead of lists of 1-item dicts
    assimilated_proxies = [{p.type: [p.id, p.lat, p.lon, p.time]}
                           for p in prox_manager.sites_assim_proxy_objs()]
    filen = join(workdir, 'assimilated_proxies')
    np.save(filen, assimilated_proxies)
    
    # collecting info on non-assimilated proxies and save to file
    nonassimilated_proxies = [{p.type: [p.id, p.lat, p.lon, p.time]}
                              for p in prox_manager.sites_eval_proxy_objs()]
    if nonassimilated_proxies:
        filen = join(workdir, 'nonassimilated_proxies')
        np.save(filen, nonassimilated_proxies)

    exp_end_time = time() - begin_time
    if verbose > 0:
        print ''
        print '====================================================='
        print 'Experiment completed in ' + str(exp_end_time/60.0) + ' mins'
        print '====================================================='

    # TODO: best method for Ye saving?
    return prox_manager.sites_assim_proxy_objs()
# ------------------------------------------------------------------------------
# --------------------------- end of main code ---------------------------------
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    LMR_driver_callable()
