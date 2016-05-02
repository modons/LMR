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
import LMR_driver_callable as LMR
import LMR_config

from LMR_utils import ensemble_stats

print '\n' + str(datetime.datetime.now()) + '\n'

# Define main experiment output directory
cfg = LMR_config.Config()

iter_range = cfg.wrapper.iter_range
expdir = os.path.join(cfg.core.datadir_output, cfg.core.nexp)

# Check if it exists, if not, create it
if not os.path.isdir(expdir):
    os.system('mkdir {}'.format(expdir))

# Monte-Carlo approach: loop over iterations (range of iterations defined in
# namelist)
MCiters = np.arange(iter_range[0], iter_range[1]+1)
for iter_num in MCiters:

    # Use seed list if specified
    if cfg.wrapper.multi_seed:
        LMR_config.core.seed = cfg.wrapper.multi_seed[iter_num]
        print ('Setting current iteration seed:'
               ' {}'.format(cfg.wrapper.multi_seed[iter_num]))

    # Set iteration number
    LMR_config.core.curr_iter = iter_num

    # Define work directory
    LMR_config.core.datadir_output = os.path.join(expdir, 'r' + str(iter_num))

    curr_cfg = LMR_config.Config()
    core = curr_cfg.core
    core.curr_iter = iter_num

    # Check if it exists, if not create it
    if not os.path.isdir(core.datadir_output):
        os.system('mkdir {}'.format(core.datadir_output))
    elif os.path.isdir(core.datadir_output) and core.clean_start:
        print (' **** clean start --- removing existing files in iteration'
               ' output directory')
        os.system('rm -f {}'.format(core.datadir_output + '/*'))

    # Call the driver
    all_proxy_objs = LMR.LMR_driver_callable(curr_cfg)

    # write the analysis ensemble mean and variance to separate files (per
    # state variable)
    ensemble_stats(core.datadir_output, all_proxy_objs)

    # start: DO NOT DELETE
    # move files from local disk to an archive location

    loc_dir = core.datadir_output
    arc_dir = os.path.join(core.archive_dir, core.nexp)
    mc_dir = os.path.join(arc_dir, 'r' + str(iter_num))
    
    # Check if the experiment archive directory exists; if not create it
    if not os.path.isdir(arc_dir):
        os.system('mkdir {}'.format(arc_dir))
    # scrub the monte carlo subdirectory if this is a clean start
    if os.path.isdir(mc_dir) and core.clean_start:
        print (' **** clean start --- removing existing files in iteration '
               'output directory')
        os.system('rm -f -r {}'.format(mc_dir))

    # remove the individual years
    # cmd = 'rm -f ' + core.datadir_output + '/year* '
    # option to move the whole directory
    # cmd = 'mv -f ' + loc_dir + ' ' + mc_dir
    # print cmd
    # os.system(cmd)

    # or just move select files and delete the rest NEED TO CREATE THE
    # DIRECTORY IN THIS CASE!
    if not os.path.isdir(mc_dir):
        os.system('mkdir {}'.format(mc_dir))

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

    # removing the work output directory once selected files have been moved
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
