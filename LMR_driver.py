
#==========================================================================================
# Program: LMR_driver.py
# 
# Purpose:
#
# Options: 
# 
# 
# 
# Originators: Greg Hakim    | Dept. of Atmospheric Sciences, Univ. of Washington
#              Robert Tardif | January 2015
# 
#
# ------ Notes ------
# to do:
# * time averaging prior
# * proposed convention for year indexing: index 0 = 1000 A.D.; 1000 = 2000 A.D.
#
#==========================================================================================

import LMR_proxy
import LMR_prior
import LMR_calibrate
import LMR_utils
from LMR_DA import enkf_update_array, cov_localization
import numpy as np
import os.path
from time import time
from random import sample
from load_proxy_data import create_proxy_lists_from_metadata_S1csv as create_proxy_lists_from_metadata
# this sets variables used at runtime
from LMR_exp_NAMELIST import *
recon_eval = False
if proxy_frac < 1.0:
    recon_eval = True
    import LMR_evaluate

# verbose controls print comments (0 = none; 1 = most important; 2 = many; >=3 = all)
verbose = 1

# ===============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN CODE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ===============================================================================
if verbose > 0:
    print ''
    print '====================================================='
    print 'Running LMR reconstruction...'
    print '====================================================='
    print 'Name of experiment: ', nexp
    print ''

begin_time = time()

# Define work directory
workdir = datadir_output + '/' + nexp
# Check if it exists, if not create it
if not os.path.isdir(workdir):
    os.system('mkdir %s' % workdir)
elif os.path.isdir(workdir) and clean_start:
    if verbose > 0: print ' **** clean start --- removing existing files in output directory'
    os.system('rm -f %s' % workdir+'/*')

# Define the number of years of the reconstruction (nb of assimilation times)
# Note: recon_period is defined in namelist
Ntimes = recon_period[1]-recon_period[0] + 1
#print 'Ntimes=',Ntimes
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
# read the data
C.read_calibration()

# ===============================================================================
# Load prior data ---------------------------------------------------------------
# ===============================================================================
if verbose > 0:
    print '-------------------------------------------'
    print 'Uploading gridded (model) data as prior ...'
    print '-------------------------------------------'
    print 'Source for prior: ', prior_source
    print ''

# Assign prior object according to "prior_source" (from namelist)
X = LMR_prior.prior_assignment(prior_source)
# add namelist attributes to the prior object
X.prior_datadir = datadir_prior
X.prior_datafile = datafile_prior
X.statevars = state_variables
X.Nens = Nens

# Read data file & populate initial prior ensemble
X.populate_ensemble(prior_source)
Xb_one = X.ens
# recall that Xb_one is not an object, but an array, so we can't do
#Xb_one.lat = X.lat
#Xb_one.lon = X.lon

# Dump Xb_one to file 
filen = workdir + '/' + 'Xb_one'
np.savez(filen,Xb_one = Xb_one,lat = X.lat,lon = X.lon, nlat = X.nlat, nlon = X.nlon)

# these are used to compute the global mean 
nlat = X.nlat
nlon = X.nlon
lat2 = np.reshape(X.lat,(nlat,nlon))
lon2 = np.reshape(X.lon,(nlat,nlon))
lat_weight = np.cos(np.deg2rad(lat2[:,0]))

# Prepare to check for files in the prior (work) directory 
# (this object just points to a directory)
prior_check = np.DataSource(workdir)

load_time = time() - begin_time
if verbose > 2:
    print '-----------------------------------------------------'
    print 'Loading completed in '+ str(load_time)+' seconds'
    print '-----------------------------------------------------'


# ===============================================================================
# Get information on proxies to assimilate --------------------------------------
# ===============================================================================

# Read proxy metadata & extract list of proxy sites for wanted proxy type & measurement
if verbose > 0: print 'Reading the proxy metadata & building list of chronologies to assimilate...'
[sites_assim, sites_eval] = create_proxy_lists_from_metadata(datadir_proxy,datafile_proxy,regions,proxy_resolution,proxy_assim,proxy_frac)

# override for testing purposes
#sites_assim = {'01:Tree ring_Width': ['Eur_04']}
#sites_assim = {'01:Coral_d18O': ['Aus_03']}
#sites_assim = {'01:Ice core_d18O': ['Arc_05']}

if verbose > 0: print 'Assimilating proxy types/sites:', sites_assim

# ===============================================================================
# Loop over all proxies and perform assimilation --------------------------------
# ===============================================================================

proxy_types_assim = sites_assim.keys()

# count the total number of proxies
total_proxy_count = 0
for proxy_key in sorted(proxy_types_assim):
    proxy_count = len(sites_assim[proxy_key]) 
    total_proxy_count = total_proxy_count + proxy_count

print '-----------------------------------------------------------------------'
print '-------total proxy count for this experiment: ' + str(total_proxy_count)
print '-----------------------------------------------------------------------'

# ---------------------
# Loop over proxy types
# ---------------------
apcount = 0 # running counter of accepted proxies
tpcount = 0 # running counter of all proxies
gmt_save = np.zeros([total_proxy_count,recon_period[1]-recon_period[0]+1])
assimilated_proxies= []
for proxy_key in sorted(proxy_types_assim):
    # Strip the assim. order info & keep only proxy type
    proxy = proxy_key.split(':', 1)[1]

    # --------------------------------------------------------------------
    # Loop over sites (chronologies) to be assimilated for this proxy type
    # --------------------------------------------------------------------
    for site in sites_assim[proxy_key]:

        tpcount = tpcount + 1
        if verbose > 0: print '--------------- Processing proxy: ' + site 

        Y = LMR_proxy.proxy_assignment(proxy)
        # add namelist attributes to the proxy object
        Y.proxy_datadir = datadir_proxy
        Y.proxy_datafile = datafile_proxy
        Y.proxy_region = regions
                
        # -------------------------------------------
        # Read data for current proxy type/chronology
        # -------------------------------------------
        Y.read_proxy(site)

        if Y.nobs == 0: # if no obs uploaded, move to next proxy type
            continue

        if verbose > 1:
            print ''
            print 'Site:', Y.proxy_type, ':', site
            print ' latitude, longitude: ' + str(Y.lat), str(Y.lon)

        # Define the ob error variance
        #ob_err = Y.error() 
        #ob_err = Y.R

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

            # if proxy ob is outside of period of reconstruction, continue to next ob time
            if t < recon_period[0] or t > recon_period[1]:
                continue
            
            if verbose > 0: print 'working on year: ' + str(t)

            indt = Y.time.index(t)

            # Load the prior for current year. first check to see if it exists; if not, use ensemble template
            ypad = LMR_utils.year_fix(t)          
            filen = workdir + '/' + 'year' + ypad
            if prior_check.exists(filen+'.npy'):
                if verbose > 2: print 'prior file exists:' + filen
                Xb = np.load(filen+'.npy')
            else:
                if verbose > 2: print 'prior file does not exist...using template for prior'
                Xb = np.copy(Xb_one)

            # --------------------------------------------------------------
            # Call PSM to get ensemble of model estimates of proxy data (Ye)
            # --------------------------------------------------------------
            Ye = Y.psm(C,Xb,X.lat,X.lon)
        
            # Reasons to abandon this proxy record due to calibration problems
            r_crit = 0.2
            if Ye is None:
                print 'PSM could not be built for this proxy chronology...too little overlap for calibration'
                break
            elif abs(Y.corr) < r_crit:
                print 'PSM could not be built for this proxy chronology...calibration correlation below threshold'
                break

            # reaching this point means the proxy will be assimilated
            if first_time:
                apcount = apcount + 1
                print '...This proxy record is accepted for assimilation...it is # ' + str(apcount) + ' out of a total proxy count of ' + str(tpcount)
                site_dict = {proxy:site}
                assimilated_proxies.append(site_dict)
                first_time = False
                
            # Define the ob error variance
            #ob_err = Y.error() 
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

            # NEW---compute ensemble-mean, global mean, temperature, and keep track of it for every proxy assimilated
            # gmt_save(proxy,year)
            # also keep a log of proxies assimilated
            GMTstart = time()
            xam_lalo = np.reshape(xam,(nlat,nlon))
            xam_lat = np.mean(xam_lalo,1)
            gmt = np.mean(np.multiply(lat_weight,xam_lat))
            gmt_save[apcount,t-recon_period[0]] = gmt
            GMTend = time()
            #if verbose > 0: print 'GMT calculation took ' + str(GMTend-GMTstart) + ' seconds'

            thistime = time()
            if verbose > 2: print 'update took ' + str(thistime-lasttime) + ' seconds'
            lasttime = thistime

end_time = time() - begin_time
if verbose > 0:
    print ''
    print '====================================================='
    print 'Reconstruction completed in '+ str(end_time/60.0)+' mins'
    print '====================================================='

#
# save global mean temperature history and the proxies assimilated
#
print 'saving global mean temperature update history and assimilated proxies...'
filen = workdir + '/' + 'gmt'
np.save(filen,gmt_save)
filen = workdir + '/' + 'assimilated_proxies'
np.save(filen,assimilated_proxies)

# ===============================================================================
# Evaluation of reconstruction using data from unassimilated sites  -------------
# ===============================================================================

if recon_eval: # if not empty lists, do recon. evaluation
    if verbose > 1:
        print '-----------------------------'
        print 'Evaluating reconstruction ...'
        print '-----------------------------'
        LMR_evaluate.recon_eval_stats(sites_eval,recon_period,workdir,C,X,datadir_proxy,datafile_proxy,regions)

# ===============================================================================
# Postprocessing of ensemble reconstruction -------------------------------------
# ===============================================================================

exp_end_time = time() - begin_time
if verbose > 0:
    print ''
    print '====================================================='
    print 'Experiment completed in '+ str(exp_end_time/60.0)+' mins'
    print '====================================================='


# -------------------------------------------------------------------------------
# --------------------------- end of main code ----------------------------------
# -------------------------------------------------------------------------------
