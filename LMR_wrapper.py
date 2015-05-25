
#==========================================================================================
# Program: LMR_wrapper.py
# 
# Purpose: Wrapper around the callable version of LMR_driver
#          prototype for Monte Carlo iterations
#
# Options: None. 
#          Experiment parameters defined through namelist, 
#          passed through object called "state"
#   
# Originators: Greg Hakim    | Dept. of Atmospheric Sciences, Univ. of Washington
#                            | April 2015
# 
# Revisions: 
#          R. Tardif (April 15 2015): ... 
#
#==========================================================================================

class state():
    pass

import os
import numpy as np
import LMR_driver_callable as LMR
import LMR_utils
from LMR_exp_NAMELIST import *

# object a from empty class state is a convenience to pass information from the "namelist"
a = state()
a.nexp             = nexp
a.datadir_output   = datadir_output
a.clean_start      = clean_start
a.recon_period     = recon_period
a.datatag_calib    = datatag_calib
a.datadir_calib    = datadir_calib
a.prior_source     = prior_source
a.datadir_prior    = datadir_prior
a.datafile_prior   = datafile_prior
a.state_variables  = state_variables
a.Nens             = Nens
a.datadir_proxy    = datadir_proxy
a.datafile_proxy   = datafile_proxy
a.regions          = regions
a.proxy_resolution = proxy_resolution
a.proxy_assim      = proxy_assim
a.proxy_frac       = proxy_frac
a.locRad           = locRad
a.PSM_r_crit       = PSM_r_crit
a.LMRpath          = LMRpath

# Define main experiment output directory
expdir = datadir_output + '/' + nexp
# Check if it exists, if not, create it
if not os.path.isdir(expdir):
    os.system('mkdir %s' % expdir)

# Monte-Carlo approach: loop over iterations (range of iterations defined in namelist)
MCiters = np.arange(iter_range[0], iter_range[1]+1)
for iter in MCiters:
    a.iter = iter
    # Define work directory
    a.workdir = datadir_output + '/' + nexp + '/r' + str(iter)
    # Check if it exists, if not create it
    if not os.path.isdir(a.workdir):
        os.system('mkdir %s' % a.workdir)
    elif os.path.isdir(a.workdir) and clean_start:
        print ' **** clean start --- removing existing files in iteration output directory'
        os.system('rm -f %s' % a.workdir+'/*')

    # Call the driver
    LMR.LMR_driver_callable(a)

    # write the ensemble mean to a separate file
    LMR_utils.ensemble_mean(a.workdir)

    # remove the individual years
    cmd = 'rm -f ' + a.workdir + '/year* ' 

    
    # start: DO NOT DELETE
    # move files from local disk to an archive location
    src_dir = a.workdir
    exp_dir = '/home/disk/kalman3/hakim/LMR/' + nexp
    mc_dir = exp_dir + '/r' + str(iter)
    
    # Check if the experiment directory exists; if not create it
    if not os.path.isdir(exp_dir):
        os.system('mkdir %s' % exp_dir)

    # scrub the monte carlo subdirectory if this is a clean start
    if os.path.isdir(mc_dir) and clean_start:
        print ' **** clean start --- removing existing files in iteration output directory'
        os.system('rm -f -r %s' % mc_dir)

    # option to move the whole directory
    #cmd = 'mv -f ' + src_dir + ' ' + mc_dir
    #print cmd
    #os.system(cmd)

    # or just move select files and delete the rest NEED TO CREATE THE DIRECTORY IN THIS CASE!
    if not os.path.isdir(mc_dir):
        os.system('mkdir %s' % mc_dir)

    cmd = 'mv -f ' + src_dir+'/*.npz' + ' ' + mc_dir + '/'
    print cmd
    os.system(cmd)
    cmd = 'mv -f ' + src_dir+'/assim*' + ' ' + mc_dir + '/'
    print cmd
    os.system(cmd)    
    cmd = 'rm -f -r ' + src_dir
    print cmd
    os.system(cmd)    
    #   end: DO NOT DELETE
    
#==========================================================================================
