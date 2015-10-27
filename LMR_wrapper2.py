# ==============================================================================
# Program: LMR_wrapper.py
# 
# Purpose: Wrapper around the callable version of LMR_driver
#          prototype for Monte Carlo iterations
#
# Options: None. 
#          Experiment parameters defined through namelist, 
#          passed through object called "state"
#   
# Originators: Greg Hakim   | Dept. of Atmospheric Sciences, Univ. of Washington
#                           | April 2015
# 
# Revisions: 
#          R. Tardif (April 15 2015): ... 
#
# ==============================================================================

import os
import numpy as np
import datetime
import itertools
import LMR_driver_callable2 as LMR
import LMR_config as cfg
import random

from LMR_utils2 import ensemble_stats

print '\n' + str(datetime.datetime.now()) + '\n'

# Define main experiment output directory
core = cfg.core
iter_range = core.iter_range
expdir = os.path.join(core.datadir_output, core.nexp)

# Check if it exists, if not, create it
if not os.path.isdir(expdir):
    os.system('mkdir {}'.format(expdir))

# Temporary for parameter sweep
a = np.arange(0.3, 0.91, 0.1)
d = np.arange(0.0, 0.55, 0.05)
seeds = random.sample(range(3, 1000), 20)


# Monte-Carlo approach: loop over iterations (range of iterations defined in
# namelist)
MCiters = np.arange(iter_range[0], iter_range[1]+1)
for iter_num in MCiters:

    cfg.core.curr_iter = iter_num
    itr_dir = os.path.join(expdir, 'r' + str(iter_num))
    cfg.core.seed = seeds[iter_num]

    for a_val, d_val in itertools.product(a, d):
        cfg.core.hybrid_a = a_val
        cfg.forecaster.LIM.eig_adjust = d_val

        # Define work directory
        ad_folder_name = 'a{:1.1f}_d{:1.2f}'.format(a_val, d_val)
        core.datadir_output = os.path.join(itr_dir, ad_folder_name)

        # Check if it exists, if not create it
        if not os.path.isdir(core.datadir_output):
            os.makedirs(core.datadir_output)
        elif os.path.isdir(core.datadir_output) and core.clean_start:
            print (' **** clean start --- removing existing files in iteration'
                   ' output directory')
            os.system('rm -f {}'.format(core.datadir_output + '/*'))

        # Call the driver
        all_proxy_objs = LMR.LMR_driver_callable(cfg)

        # write the analysis ensemble mean and variance to separate files (per
        # state variable)
        ensemble_stats(core.datadir_output, all_proxy_objs)

        # start: DO NOT DELETE
        # move files from local disk to an archive location

        loc_dir = core.datadir_output
        arc_dir = os.path.join(core.archive_dir, core.nexp)
        mc_dir = os.path.join(arc_dir, 'r' + str(iter_num), ad_folder_name)

        # scrub the monte carlo subdirectory if this is a clean start
        if os.path.isdir(mc_dir):
            if core.clean_start:
                print (' **** clean start --- removing existing files in'
                       ' iteration output directory')
                os.system('rm -f -r {}'.format(mc_dir + '/*'))
        else:
            os.makedirs(mc_dir)

        # or just move select files and delete the rest

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

        # removing the work output directory once selected files have been
        #  moved
        cmd = 'rm -f -r ' + loc_dir
        print cmd
        os.system(cmd)

        # copy the namelist file to archive directory
        cmd = 'cp ./LMR_config.py ' + mc_dir + '/'
        print cmd
        os.system(cmd)

        print '\n' + str(datetime.datetime.now()) + '\n'

    #   end: DO NOT DELETE
    
# ==============================================================================
