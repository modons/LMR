
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


#==========================================================================================
