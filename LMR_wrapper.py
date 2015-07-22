

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
import datetime
import LMR_driver_callable as LMR
import LMR_utils
from datetime import datetime
from LMR_exp_NAMELIST import *

print '\n' + str(datetime.now()) + '\n'

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

    # write the analysis ensemble mean and variance to separate files (per state variable)
    LMR_utils.ensemble_stats(a.workdir,a.Yall)

    # start: DO NOT DELETE
    # move files from local disk to an archive location

    loc_dir = a.workdir
    arc_dir = archive_dir + '/' + nexp
    mc_dir  = arc_dir + '/r' + str(iter)
    
    # Check if the experiment archive directory exists; if not create it
    if not os.path.isdir(arc_dir):
        os.system('mkdir %s' % arc_dir)
    # scrub the monte carlo subdirectory if this is a clean start
    if os.path.isdir(mc_dir) and clean_start:
        print ' **** clean start --- removing existing files in iteration output directory'
        os.system('rm -f -r %s' % mc_dir)

    # remove the individual years
    cmd = 'rm -f ' + a.workdir + '/year* ' 
    # option to move the whole directory
    #cmd = 'mv -f ' + loc_dir + ' ' + mc_dir
    #print cmd
    #os.system(cmd)

    # or just move select files and delete the rest NEED TO CREATE THE DIRECTORY IN THIS CASE!
    if not os.path.isdir(mc_dir):
        os.system('mkdir %s' % mc_dir)

    cmd = 'mv -f ' + loc_dir+'/*.npz' + ' ' + mc_dir + '/'
    print cmd
    os.system(cmd)
    cmd = 'mv -f ' + loc_dir+'/*.pckl' + ' ' + mc_dir + '/'
    print cmd
    os.system(cmd)
    cmd = 'mv -f ' + loc_dir+'/assim*' + ' ' + mc_dir + '/'
    print cmd
    os.system(cmd)    
    cmd = 'mv -f ' + loc_dir+'/nonassim*' + ' ' + mc_dir + '/'
    print cmd
    os.system(cmd)    
    cmd = 'mv -f ' + loc_dir+'/gmt_ensemble*' + ' ' + mc_dir + '/'
    print cmd
    os.system(cmd)

    # removing the work output directory once selected files have been moved
    cmd = 'rm -f -r ' + loc_dir
    print cmd
    os.system(cmd)
    
    # copy the namelist file to archive directory
    cmd = 'cp ./LMR_exp_NAMELIST.py ' + mc_dir + '/'
    print cmd
    os.system(cmd)
    
    print '\n' + str(datetime.now()) + '\n'

    #   end: DO NOT DELETE
    
#==========================================================================================
