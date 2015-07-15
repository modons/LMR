
#==========================================================================================
# Program: LMR_driver_callable.py
# 
# Purpose: 
#
# Options: None. 
#          Experiment parameters through namelist, passed through object called "state"
#  
# 
# Originators: Greg Hakim    | Dept. of Atmospheric Sciences, Univ. of Washington
#              Robert Tardif | January 2015
# 
# Revisions: 
#  April 2015:
#            - This version is callable by an outside script, accepts a single object, 
#              called state, which has everything needed for the driver (G. Hakim)
#
#            - Re-organisation of code around PSM calibration and calculation of the Ye's.
#              Code now assumes PSM parameters have been pre-calulated and Ye's are 
#              calculated up-front for all proxy types/sites. All proxy data are now also 
#              loaded up-front, prior to any loops. Ye's are appended to state vector to
#              form an augmented state vector and are also updated by DA. (R. Tardif)
#  May 2015:
#            - Bug fix in calculation of global mean temperature + function now part of
#              LMR_utils.py (G. Hakim)
#
#  July 2015:
#            - Version with state vector composed of several state variables, which may
#              be 2D lat/lon, 2D lat/depth (e.g. AMOC streamfunction) or time series
#              (e.g. AMOC index, basin averaged ocean heat content etc.) (R. Tardif)
#
#==========================================================================================

def LMR_driver_callable(state):

    import numpy as np
    import os.path
    from time import time
    from random import sample
    import cPickle

    import LMR_proxy
    import LMR_prior
    import LMR_calibrate
    import LMR_utils
    from LMR_DA import enkf_update_array, cov_localization
    from load_proxy_data import create_proxy_lists_from_metadata_S1csv as create_proxy_lists_from_metadata

    verbose = 1 # verbose controls print comments (0 = none; 1 = most important; 2 = many; >=3 = all)

    # daylight the variables passed in the state object (easier for code migration than leaving attached)
    nexp             = state.nexp
    workdir          = state.workdir
    recon_period     = state.recon_period
    datatag_calib    = state.datatag_calib
    datadir_calib    = state.datadir_calib
    prior_source     = state.prior_source
    datadir_prior    = state.datadir_prior
    datafile_prior   = state.datafile_prior
    state_variables  = state.state_variables
    Nens             = state.Nens
    datadir_proxy    = state.datadir_proxy
    datafile_proxy   = state.datafile_proxy
    regions          = state.regions
    proxy_resolution = state.proxy_resolution
    proxy_assim      = state.proxy_assim
    proxy_frac       = state.proxy_frac
    locRad           = state.locRad
    PSM_r_crit       = state.PSM_r_crit
    LMRpath          = state.LMRpath

    # ===============================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN CODE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ===============================================================================
    if verbose > 0:
        print ''
        print '====================================================='
        print 'Running LMR reconstruction...'
        print '====================================================='
        print 'Name of experiment: ', state.nexp
        print ' Monte Carlo iter : ', state.iter
        print ''
        
    begin_time = time()

    # Define the number of years of the reconstruction (nb of assimilation times)
    # Note: recon_period is defined in namelist
    Ntimes = recon_period[1]-recon_period[0] + 1
    recon_times = np.arange(recon_period[0], recon_period[1]+1)

    # ===============================================================================
    # Load calibration data ---------------------------------------------------------
    # ===============================================================================
    if verbose > 0:
        print '------------------------------'
        print 'Creating calibration object...'
        print '------------------------------'
        print 'Source for calibration: ' + datatag_calib
        print ''

    # Assign calibration object according to "datatag_calib" (from namelist)
    C = LMR_calibrate.calibration_assignment(datatag_calib)

    # the path to the calibration directory is specified in the namelist file; bind it here
    C.datadir_calib = datadir_calib;

    # read the data !!!!!!!!!!!!!!!!!! don't need this with all pre-calculated PSMs !!!!!!!!!!!!!!!!!!
    C.read_calibration()

    # ===============================================================================
    # Load prior data ---------------------------------------------------------------
    # ===============================================================================
    if verbose > 0:
        print '-------------------------------------------'
        print 'Uploading gridded (model) data as prior ...'
        print '-------------------------------------------'
        print 'Source for prior: ', prior_source

    # Assign prior object according to "prior_source" (from namelist)
    X = LMR_prior.prior_assignment(prior_source)

    # add namelist attributes to the prior object
    X.prior_datadir = datadir_prior
    X.prior_datafile = datafile_prior
    X.statevars = state_variables
    X.Nens = Nens

    # Read data file & populate initial prior ensemble
    X.populate_ensemble(prior_source)
    Xb_one_full = X.ens

    load_time = time() - begin_time
    if verbose > 2:
        print '-----------------------------------------------------'
        print 'Loading completed in '+ str(load_time)+' seconds'
        print '-----------------------------------------------------'


    # ===============================================================================
    # Get information on proxies to assimilate --------------------------------------
    # ===============================================================================

    begin_time_proxy_load = time()
    if verbose > 0:
        print ''
        print '-----------------------------------'
        print 'Uploading proxy data & PSM info ...'
        print '-----------------------------------'

    # Read proxy metadata & extract list of proxy sites for wanted proxy type & measurement
    if verbose > 0: print 'Reading the proxy metadata & building list of chronologies to assimilate...'

    # Load pre-calibrated PSM parameters 
    fnamePSM = LMRpath+'/PSM/PSMs_'+datatag_calib+'.pckl'
    infile   = open(fnamePSM,'rb')
    psm_data = cPickle.load(infile)
    infile.close()

    # Build dictionaries of proxy sites to assimilate and those set aside for verification
    [sites_assim, sites_eval] = create_proxy_lists_from_metadata(datadir_proxy,datafile_proxy,regions,proxy_resolution,proxy_assim,proxy_frac,psm_data,PSM_r_crit)

    if verbose > 0: print 'Assimilating proxy types/sites:', sites_assim

    # ================================================================================
    # Calculate all Ye's (for all sites in sites_assim) ------------------------------
    # ================================================================================

    proxy_types_assim = sites_assim.keys()

    print '-----------------------------------------------------------------------'
    print 'Proxy counts for experiment:'
    # count the total number of proxies
    total_proxy_count = 0
    for proxy_key in sorted(proxy_types_assim):
        proxy_count = len(sites_assim[proxy_key]) 
        total_proxy_count = total_proxy_count + proxy_count
        print('%45s : %5d' % (proxy_key,proxy_count))
    print('%45s : %5d' %('TOTAL', total_proxy_count))
    print '-----------------------------------------------------------------------'

    # RT-NEW: Master list of ***proxy objects***
    Yall = []

    # Loop over proxy types
    scount = -1 # counter of all proxy sites
    for proxy_key in sorted(proxy_types_assim):
        # Strip the assim. order info & keep only proxy type
        proxy = proxy_key.split(':', 1)[1]

        # Loop over sites (chronologies) for this proxy type
        # & populate proxy object with site info and data
        for site in sites_assim[proxy_key]:
            scount = scount + 1
            Ywk = LMR_proxy.proxy_assignment(proxy)

            # Add attributes to the proxy object
            Ywk.pid = (proxy,site)
            Ywk.proxy_datadir = datadir_proxy
            Ywk.proxy_datafile = datafile_proxy
            Ywk.proxy_region = regions
            Ywk.nobs = 0            

            # Check if PSM for (proxy,site) has been pre-calibrated
            if Ywk.pid in psm_data.keys():
                Ywk.calibrate = True
                Ywk.lat       = psm_data[(proxy,site)]['lat']
                Ywk.lon       = psm_data[(proxy,site)]['lon']
                Ywk.corr      = psm_data[(proxy,site)]['PSMcorrel']
                Ywk.slope     = psm_data[(proxy,site)]['PSMslope']
                Ywk.intercept = psm_data[(proxy,site)]['PSMintercept']
                Ywk.R         = psm_data[(proxy,site)]['PSMmse']
            else:
                print 'PSM not calibrated for:' + str((proxy,site))

            # --------------------------------------------------------------
            # Call PSM to get ensemble of prior estimates of proxy data (Ye)
            # --------------------------------------------------------------
            Ywk.Ye = Ywk.psm(C,Xb_one_full,X.full_state_info,X.coords)

            # -------------------------------------------
            # Read data for current proxy type/chronology
            # -------------------------------------------
            Ywk.read_proxy(site)

            # Append proxy object to master list
            Yall.append(Ywk)

    # attach Yall to state
    state.Yall = Yall

    proxy_load_time = time() - begin_time_proxy_load
    if verbose > 2:
        print '-----------------------------------------------------'
        print 'Loading completed in '+ str(proxy_load_time)+' seconds'
        print '-----------------------------------------------------'


    # ================================================================================
    # Calculate truncated state from prior, if option chosen -------------------------
    # ================================================================================
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
        var_array_full = Xb_one_full[ibeg_full:iend_full+1,:]
        # corresponding spatial coordinates
        coords_array_full = X.coords[ibeg_full:iend_full+1,:]
        
        # Are we truncating this variable? (i.e. is it a 2D lat/lon variable?)
        if X.full_state_info[var]['spacecoords'] and 'lat' in X.full_state_info[var]['spacecoords'] and 'lon' in X.full_state_info[var]['spacecoords']:
            print var, ' : 2D lat/lon variable, truncating this variable'
            # lat/lon column indices in X.coords 
            ind_lon = X.full_state_info[var]['spacecoords'].index('lon')
            ind_lat = X.full_state_info[var]['spacecoords'].index('lat')
            nlat = X.full_state_info[var]['spacedims'][ind_lat]
            nlon = X.full_state_info[var]['spacedims'][ind_lon]
            
            # calculate the truncated field        
            [var_array_new,lat_new,lon_new] = LMR_utils.regrid_sphere(nlat,nlon,Nens,var_array_full,42)
            nlat_new = np.shape(lat_new)[0]
            nlon_new = np.shape(lat_new)[1]

            # corresponding indices in truncated state vector
            ibeg_new = Nx
            iend_new = Nx+(nlat_new*nlon_new)-1
            # for new state info dictionary
            dct['pos'] = (ibeg_new,iend_new)
            dct['spacecoords'] =  X.full_state_info[var]['spacecoords']
            dct['spacedims'] = (nlat_new,nlon_new)
            # updated dimension
            new_dims = (nlat_new*nlon_new)

            # array with new spatial coords
            coords_array_new = np.zeros(shape=[new_dims,2])
            coords_array_new[:,0] = lat_new.flatten()
            coords_array_new[:,1] = lon_new.flatten()

        else:
            print var, ' : not truncating this variable: no changes from full state'
            var_array_new = var_array_full
            coords_array_new = coords_array_full
            # updated dimension
            new_dims = var_array_new.shape[0]
            ibeg_new = Nx
            iend_new = Nx+(new_dims)-1
            dct['pos'] = (ibeg_new,iend_new)
            dct['spacecoords'] =  X.full_state_info[var]['spacecoords']
            dct['spacedims'] = X.full_state_info[var]['spacedims']



        # fill in new state info dictionary
        new_state_info[var] = dct

        if Nx == 0: # if 1st time in loop over state variables, create Xb_one array as copy of var_array_new
            Xb_one = np.copy(var_array_new)
            Xb_one_coords = np.copy(coords_array_new)
        else: # if not 1st time, append to existing array
            Xb_one = np.append(Xb_one,var_array_new,axis=0)
            Xb_one_coords = np.append(Xb_one_coords,coords_array_new,axis=0)

        # updating dimension of new state vector
        Nx = Nx + new_dims


    X.trunc_state_info = new_state_info

    # Keep dimension of pre-augmented version of state vector
    [stateDim, _] = Xb_one.shape

    # ----------------------------------
    # Augment state vector with the Ye's
    # ----------------------------------

    # Extract all the Ye's from master list of proxy objects into numpy array
    Ye_all = np.empty(shape=[len(Yall),Nens])
    for k in range(len(Yall)): Ye_all[k,:] = Yall[k].Ye

    # Append ensemble of Ye's to prior state vector 
    Xb_one_aug = np.append(Xb_one,Ye_all,axis=0)

    # Dump prior state vector (Xb_one) to file 
    filen = workdir + '/' + 'Xb_one'
    #np.savez(filen,Xb_one = Xb_one,stateDim = stateDim,lat = X.lat,lon = X.lon, nlat = X.nlat, nlon = X.nlon)
    #np.savez(filen,Xb_one = Xb_one,Xb_one_aug = Xb_one_aug,stateDim = stateDim,lat = lat_new,lon = lon_new,nlat = nlat_new,nlon = nlon_new,state_info=X.trunc_state_info)
    np.savez(filen,Xb_one = Xb_one,Xb_one_aug = Xb_one_aug,stateDim = stateDim,Xb_one_coords = Xb_one_coords,state_info=X.trunc_state_info)

    # 25 June 2015: save a prior file, with the augmented state, for each year as a starting point for the analysis
    for yr in range(recon_period[0],recon_period[1]+1):
        ypad = LMR_utils.year_fix(yr)          
        filen = workdir + '/' + 'year' + ypad
        #print 'writing initial prior file...' + filen
        np.save(filen,Xb_one_aug)

    # Prepare to check for files in the prior (work) directory (this object just points to a directory)
    prior_check = np.DataSource(workdir)

    # ===============================================================================
    # Loop over all proxies and perform assimilation --------------------------------
    # ===============================================================================

    # ---------------------
    # Loop over proxy types
    # ---------------------
    apcount = 0 # running counter of accepted proxies

    # Array containing the global-mean state (for diagnostic purposes)
    # Now doing surface air temperature only (var = tas_sfc_Amon)!
    gmt_save = np.zeros([total_proxy_count+1,recon_period[1]-recon_period[0]+1])
    # get state vector indices where to find surface air temperature 
    ibeg_tas = X.trunc_state_info['tas_sfc_Amon']['pos'][0]
    iend_tas = X.trunc_state_info['tas_sfc_Amon']['pos'][1]
    xbm = np.mean(Xb_one[ibeg_tas:iend_tas+1,:],axis=1) # ensemble-mean
    xbm_lalo = np.reshape(xbm,(nlat_new,nlon_new))
    gmt = LMR_utils.global_mean(xbm_lalo,lat_new[:,0],lon_new[0,:])
    gmt_save[0,:] = gmt # First prior
    gmt_save[1,:] = gmt # Prior for first proxy assimilated

    assimilated_proxies= []
    for proxy_key in sorted(proxy_types_assim):
        # Strip the assim. order info & keep only proxy type
        proxy = proxy_key.split(':', 1)[1]

        # --------------------------------------------------------------------
        # Loop over sites (chronologies) to be assimilated for this proxy type
        # --------------------------------------------------------------------
        for site in sites_assim[proxy_key]:
            
            if verbose > 0: print '--------------- Processing proxy: ' + site 
            
            # Find (proxy,site) in proxy object master list 
            indmaster = [k for k in range(len(Yall)) if Yall[k].pid == (proxy,site)]
            if indmaster:
                indx = indmaster[0]
                Y = Yall[indx]

                # Index of array element in augmented state vector where to find the Ye's from the current (proxy,site)
                # *** Note: position of Ye's for current (proxy,site) in augmented part of state vector is same as ***
                # *** position in master list of proxy objects (Yall)                                              ***
                indYe = stateDim + indx

            else:
                print 'Error:', site, 'not found!'
                continue

            if Y.nobs == 0: # if no obs uploaded, move to next proxy type
                continue

            if verbose > 1:
                print ''
                print 'Site:', Y.proxy_type, ':', site
                print ' latitude, longitude: ' + str(Y.lat), str(Y.lon)

            # Calculate array for covariance localization
            if locRad is not None:
                if verbose > 2: print '...computing localization...'
                loc = cov_localization(locRad,X,Y)
            else:
                loc = None

            # ---------------------------------------
            # Loop over all times in the proxy record
            # ---------------------------------------
            lasttime = time()
            first_time = True
            for t in Y.time:

                # make sure t is an integer (necessary due to erratic cell formatting in PAGES excel proxy data file!)
                t = int(t)

                # if proxy ob is outside of period of reconstruction, continue to next ob time
                if t < recon_period[0] or t > recon_period[1]:
                    continue
                
                if verbose > 2: print 'working on year: ' + str(t)

                indt = Y.time.index(t)

                # Load the prior for current year. first check to see if it exists; if not, use ensemble template
                ypad = LMR_utils.year_fix(t)          
                filen = workdir + '/' + 'year' + ypad
                if prior_check.exists(filen+'.npy'):
                    if verbose > 2: print 'prior file exists:' + filen
                    Xb = np.load(filen+'.npy')
                else:
                    if verbose > 2: print 'prior file ' + filen+'.npy does not exist...using template for prior'
                    Xb = np.copy(Xb_one_aug)

                # Extract the Ye values for current (proxy,site) from augmented state vector
                Ye = Xb[indYe]
        
                # reaching this point means the proxy will be assimilated
                if first_time:
                    apcount = apcount + 1
                    print '...This proxy record is accepted for assimilation...it is # ' + str(apcount) + ' out of a total proxy count of ' + str(total_proxy_count)
                    years_in_recon = [k for k in Y.time if k >= recon_period[0] and k <= recon_period[1]]                    
                    site_dict = {proxy:[site,Y.lat,Y.lon,years_in_recon]}
                    assimilated_proxies.append(site_dict)
                    first_time = False
                
                # Define the ob error variance
                ob_err = Y.R

                # -------------------------------------------------------------------
                # Do the update (assimilation) --------------------------------------
                # -------------------------------------------------------------------
                if verbose > 2: print 'updating time: '+str(t)+' proxy value : '+str(Y.value[indt]) + ' | mean prior proxy estimate: '+str(np.mean(Ye))

                # Update the state
                Xa  = enkf_update_array(Xb,Y.value[indt],Ye,ob_err,loc)
                xam = np.mean(Xa,axis=1)

                # check the variance change for sign
                xbvar = np.var(Xb,axis=1)
                xavar = np.var(Xa,axis=1)
                vardiff = xavar - xbvar
                if verbose > 2: print 'max change in variance:' + str(np.max(vardiff))

                # Dump Xa to file (to be used as prior for next assimilation)
                np.save(filen,Xa)

                # compute and store the global mean (temperature only)
                xam_lalo = np.reshape(xam[ibeg_tas:iend_tas+1],(nlat_new,nlon_new))
                gmt = LMR_utils.global_mean(xam_lalo,lat_new[:,0],lon_new[0,:])
                gmt_save[apcount,int(t-recon_period[0])] = gmt
                
                thistime = time()
                if verbose > 2: print 'update took ' + str(thistime-lasttime) + ' seconds'
                lasttime = thistime

                end_time = time() - begin_time

            # End of loop on time:
            # Propagate the updated gmt_save as prior for next assimilated proxy
            if apcount < total_proxy_count:
                gmt_save[apcount+1,:] = gmt_save[apcount,:]

    # End of loop on proxy types
    if verbose > 0:
        print ''
        print '====================================================='
        print 'Reconstruction completed in '+ str(end_time/60.0)+' mins'
        print '====================================================='

    #
    # save global mean temperature history and the proxies assimilated
    #

    # 3 July 2015: compute and save the GMT for the full ensemble
    # need to fix this so that every year is counted
    gmt_ensemble = np.zeros([Ntimes,Nens])
    iyr = -1
    for yr in range(recon_period[0],recon_period[1]+1):
        iyr = iyr + 1
        ypad = LMR_utils.year_fix(yr)          
        filen = workdir + '/' + 'year' + ypad
        Xa = np.load(filen+'.npy')
        for k in range(Nens):
            xam_lalo = np.reshape(Xa[ibeg_tas:iend_tas+1,k],(nlat_new,nlon_new))
            gmt = LMR_utils.global_mean(xam_lalo,lat_new[:,0],lon_new[0,:])
            gmt_ensemble[iyr,k] = gmt

    filen = workdir + '/' + 'gmt_ensemble'
    np.savez(filen,gmt_ensemble=gmt_ensemble,recon_times=recon_times)
    
    print 'saving global mean temperature update history and assimilated proxies...'
    filen = workdir + '/' + 'gmt'
    np.savez(filen,gmt_save=gmt_save,recon_times=recon_times,apcount=apcount,tpcount=total_proxy_count)    
    filen = workdir + '/' + 'assimilated_proxies'
    np.save(filen,assimilated_proxies)
    
    # collecting info on non-assimilated proxies and save to file
    nonassimilated_proxies = []
    proxy_types_eval = sites_eval.keys()
    for proxy_key in sorted(proxy_types_eval):
        # Strip the assim. order info & keep only proxy type
        proxy = proxy_key.split(':', 1)[1]
        for site in sites_eval[proxy_key]:
            if (proxy,site) in psm_data.keys():
                site_dict = {proxy:[site,psm_data[(proxy,site)]['lat'],psm_data[(proxy,site)]['lon']]}
                nonassimilated_proxies.append(site_dict)
    filen = workdir + '/' + 'nonassimilated_proxies'
    np.save(filen,nonassimilated_proxies)


    exp_end_time = time() - begin_time
    if verbose > 0:
        print ''
        print '====================================================='
        print 'Experiment completed in '+ str(exp_end_time/60.0)+' mins'
        print '====================================================='
        

    # -------------------------------------------------------------------------------
    # --------------------------- end of main code ----------------------------------
    # -------------------------------------------------------------------------------
