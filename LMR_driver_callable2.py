
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
import tables as tb

import LMR_proxy2
import LMR_gridded
from LMR_utils2 import global_mean2, empty_hdf5_carray
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
    trunc_state = prior.truncate_state
    assim_res_vals = core.assimilation_time_res
    prior_source = prior.prior_source
    base_res = core.assimilation_time_res[0]
    sub_base_res = core.sub_base_res
    res_assim_freq = (np.array(assim_res_vals)/base_res).astype(np.int16)
    res_yr_shift = core.res_yr_shift

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
    # Load calibration data ----------------------------------------------------
    # # ========================================================================
    # if verbose > 0:
    #     print '------------------------------'
    #     print 'Creating calibration object...'
    #     print '------------------------------'
    #     print 'Source for calibration: ' + datatag_calib
    #     print ''

    # ==========================================================================
    # Load prior data ----------------------------------------------------------
    # ==========================================================================
    if verbose > 0:
        print '-------------------------------------------'
        print 'Uploading gridded (model) data as prior ...'
        print '-------------------------------------------'
        print 'Source for prior: ', prior_source

    # Create initial state vector of desired variables at smallest time res
    Xb_one_full = LMR_gridded.State.from_config(cfg)


    # Prepare to check for files in the prior (work) directory (this object just
    #  points to a directory)
    #TODO: might not be necessary
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
    prox_manager = LMR_proxy2.ProxyManager(BaseCfg, recon_period)
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

    if trunc_state:
        Xb_one = Xb_one_full.truncate_state()
    else:
        Xb_one = Xb_one_full.copy()

    # Keep dimension of pre-augmented version of state vector
    state_dim = Xb_one.shape[0]

    # ----------------------------------
    # Augment state vector with the Ye's
    # ----------------------------------

    # TODO: Figure out how to handle precalculated YE Vals larger than 1 yr
    # Extract all the Ye's from master list of proxy objects into numpy array
    if not online:
        ye_all = np.empty(shape=[total_proxy_count, nens])
        for res in assim_res_vals:
            Xb_full_copy = Xb_one_full.copy_state()

            shift = res_yr_shift[res]

            Xb_full_copy.avg_to_res(res, shift)
            for i, proxy in enumerate(
                    prox_manager.sites_assim_res_proxy_objs(res)):
                ye_all[i, :] = proxy.psm(Xb_one_full, proxy.subannual_idx)


        # Append ensemble of Ye's to prior state vector
        Xb_one.augment_state(ye_all)

    # TODO: Switch to cPickled prior object... right now hardcoded for annual
    # case saving
    # Dump prior state vector (Xb_one) to file 
    filen = workdir + '/' + 'Xb_one'
    np.savez(filen, Xb_one=Xb_one.get_var_data('state'),
             Xb_one_aug=Xb_one.state_list,
             stateDim=state_dim,
             Xb_one_coords=Xb_one.var_coords,
             state_info=Xb_one.old_state_info)

    # ==========================================================================
    # Loop over all years and proxies, and perform assimilation ----------------
    # ==========================================================================

    # Create sub_base_resolution output container
    fname = 'xa_output_res{:1.2f}.h5'.format(sub_base_res)
    Xb_one.create_h5_state_container(join(workdir, fname), ntimes)


    # Array containing the global-mean state (for diagnostic purposes)
    gmt_save = np.zeros([total_proxy_count+1, ntimes])
    xbm = Xb_one.annual_avg('tas_sfc_Amon')
    xbm = xbm.mean(axis=1)  # ensemble-mean
    gmt = global_mean2(xbm, Xb_one.var_coords['tas_sfc_Amon']['lat'])
    gmt_save[0, :] = gmt  # First row is prior GMT
    gmt_save[1, :] = gmt  # Prior for first proxy assimilated

    nelem_pr_yr = np.ceil(1.0 / base_res)
    start_yr, end_yr = recon_period
    assim_times = np.arange(start_yr, end_yr+1, base_res)

    # ---------------------
    # Loop over proxy types
    # ---------------------
    lasttime = time()
    for iyr, t in enumerate(assim_times):

        if verbose > 0:
            print 'working on year: ' + str(t)

        # TODO: Loading prior file
        if t % 1.0 == 0:
            Xb_one.insert_upcoming_prior(int(iyr//nelem_pr_yr))
        #     ypad = '{:04d}'.format(int(t))
        #     filen = join(workdir, 'year' + ypad + '.npy')
        #
        #     Xb = Xb_one.copy_state()
        #     if prior_check.exists(filen) and not core.clean_start:
        #         if verbose > 2:
        #             print 'prior file exists: ' + filen
        #         Xb.state_list = np.load(filen)

        # Which resolutions to assimilate for given year
        res_to_assim = [res for res, freq in zip(assim_res_vals, res_assim_freq)
                        if (iyr+1) % freq == 0 and (iyr+1)/freq > 0]

        iproxy = -1
        for res in res_to_assim:

            # Create prior for resolution
            Xb_one.xb_from_h5(int(iyr//nelem_pr_yr), res, res_yr_shift[res])

            for iproxy, Y in enumerate(
                    prox_manager.sites_assim_res_proxy_objs(res),
                    iproxy+1):

                # Crude check if we have proxy ob for current time
                try:
                    Y.values[t]
                except KeyError:
                    # Make sure GMT spot filled from previous proxy
                    if (iyr+1) % nelem_pr_yr == 0:
                        gmt_save[iproxy+1, iyr//nelem_pr_yr] = \
                            gmt_save[iproxy, iyr//nelem_pr_yr]
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
                    Ye = Y.psm(Xb_one)
                else:
                    Ye = Xb_one.get_var_data('ye_vals',
                                             idx=Y.subannual_idx)[iproxy]

                # Define the ob error variance
                ob_err = Y.psm_obj.R

                # --------------------------------------------------------------
                # Do the update (assimilation) ---------------------------------
                # --------------------------------------------------------------
                if verbose > 2:
                    print ('updating time: ' + str(t) + ' proxy value : ' +
                           str(Y.values[t]) + ' | mean prior proxy estimate: ' +
                           str(Ye.mean()))

                # Update the state
                Xa = enkf_update_array(Xb_one.state_list[Y.subannual_idx],
                                       Y.values[t], Ye, ob_err, loc)
                Xb_one.state_list[Y.subannual_idx] = Xa

                if (iyr+1) % nelem_pr_yr == 0:
                    xam = Xb_one.get_var_data('tas_sfc_Amon',
                                              idx=Y.subannual_idx).mean(axis=1)
                    gmt = global_mean2(xam,
                                       Xb_one.var_coords['tas_sfc_Amon']['lat'])
                    gmt_save[iproxy+1, iyr//nelem_pr_yr] = gmt

                # check the variance change for sign
                thistime = time()
                # if verbose > 2:
                #     xbvar = Xb_one.var(axis=1, ddof=1)
                #     xavar = Xa.var(ddof=1, axis=1)
                #     vardiff = xavar - xbvar
                #     print 'max change in variance:' + str(np.max(vardiff))
                #     print 'update took ' + str(thistime-lasttime) + 'seconds'
                lasttime = thistime

            # Assimilated all proxies at given res, propagate mean to base res
            Xb_one.propagate_avg_to_h5(int(iyr//nelem_pr_yr), res_yr_shift[res])

        if (iyr+1) % nelem_pr_yr == 0:
            ypad = '{:04d}'.format(int(t))
            filen = join(workdir, 'year' + ypad + '.npy')
            np.save(filen, Xb_one.annual_avg())

    end_time = time() - begin_time

    # End of loop on proxy types
    if verbose > 0:
        print ''
        print '====================================================='
        print 'Reconstruction completed in ' + str(end_time/60.0)+' mins'
        print '====================================================='

    # Close H5 file
    Xb_one.close_xb_container()

    # 3 July 2015: compute and save the GMT for the full ensemble
    # need to fix this so that every year is counted
    gmt_ensemble = np.zeros([ntimes, nens])
    for iyr, yr in enumerate(assim_times[0::nelem_pr_yr]):
        filen = join(workdir, 'year{:04d}'.format(int(yr)))
        Xb_one.state_list = [np.load(filen+'.npy')]
        Xa = np.squeeze(Xb_one.get_var_data('tas_sfc_Amon'))
        gmt_ensemble[iyr] = \
            global_mean2(Xa.T,
                         Xb_one.var_coords['tas_sfc_Amon']['lat'])

    filen = join(workdir, 'gmt_ensemble')
    np.savez(filen, gmt_ensemble=gmt_ensemble, recon_times=recon_times)

    # save global mean temperature history and the proxies assimilated
    print ('saving global mean temperature update history and ',
           'assimilated proxies...')
    filen = join(workdir, 'gmt')
    np.savez(filen, gmt_save=gmt_save, recon_times=recon_times,
             apcount=total_proxy_count, tpcount=total_proxy_count)

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
