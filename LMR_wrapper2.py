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
import itertools
import datetime
import LMR_driver_callable2 as LMR
import LMR_config as cfg

import LMR_utils2 as util2
from LMR_utils2 import ensemble_stats

print '\n' + str(datetime.datetime.now()) + '\n'

# Define main experiment output directory
core = cfg.core
iter_range = core.iter_range
expdir = os.path.join(core.datadir_output, core.nexp)

# Check if it exists, if not, create it
if not os.path.isdir(expdir):
    os.system('mkdir {}'.format(expdir))

# Monte-Carlo approach: loop over iterations (range of iterations defined in
# namelist)
MCiters = xrange(iter_range[0], iter_range[1]+1)
param_iterables = [MCiters]

# get other parameters to sweep over in the reconstruction
param_search = cfg.wrapper.param_search
if param_search is not None:
    sort_params = param_search.keys()
    sort_params.sort(key=lambda x: x.split('.')[-1])
    param_values = [param_search[key] for key in sort_params]
    param_iterables = [MCiters] + param_values


for iter_and_params in itertools.product(*param_iterables):

    iter_num = iter_and_params[0]
    cfg.core.curr_iter = iter_num

    if cfg.wrapper.multi_seed is not None:
        cfg.core.seed = cfg.wrapper.multi_seed[iter_num]

    itr_str = 'r{:d}'.format(iter_num)
    working_dir = os.path.join(expdir, itr_str)
    arc_dir = os.path.join(core.archive_dir, core.nexp)
    mc_dir = os.path.join(arc_dir, itr_str)

    # If parameter space search is being performed then set the current
    # search space values and create a sub-directory
    if param_search is not None:
        curr_param_values = iter_and_params[1:]
        psearch_dir = util2.set_paramsearch_attributes(sort_params,
                                                       curr_param_values,
                                                       cfg)
        working_dir = os.path.join(working_dir, psearch_dir)
        mc_dir = os.path.join(mc_dir, psearch_dir)

    core.datadir_output = working_dir

    # Check if it exists, if not create it
    if not os.path.isdir(core.datadir_output):
        os.makedirs(core.datadir_output)
    elif os.path.isdir(core.datadir_output) and core.clean_start:
        print (' **** clean start --- removing existing files in iteration'
               ' output directory')
        os.system('rm -f {}'.format(core.datadir_output + '/*'))

    # Call the driver
    try:
        all_proxy_objs = LMR.LMR_driver_callable(cfg)
    except LMR.FilterDivergenceError as e:
        print e

        # removing the work output directory
        cmd = 'rm -f -r ' + working_dir
        print cmd
        os.system(cmd)
        continue

    # write the analysis ensemble mean and variance to separate files (per
    # state variable)
    ensemble_stats(core.datadir_output, all_proxy_objs)

    # start: DO NOT DELETE
    # move files from local disk to an archive location

    # scrub the monte carlo subdirectory if this is a clean start
    if os.path.isdir(mc_dir):
        if core.clean_start:
            print (' **** clean start --- removing existing files in'
                   ' iteration output directory')
            os.system('rm -f -r {}'.format(mc_dir + '/*'))
    else:
        os.makedirs(mc_dir)

    # or just move select files and delete the rest

    cmd = 'mv -f ' + working_dir + '/*.npz' + ' ' + mc_dir + '/'
    print cmd
    os.system(cmd)
    cmd = 'mv -f ' + working_dir + '/*.pckl' + ' ' + mc_dir + '/'
    print cmd
    os.system(cmd)
    cmd = 'mv -f ' + working_dir + '/assim*' + ' ' + mc_dir + '/'
    print cmd
    os.system(cmd)
    cmd = 'mv -f ' + working_dir + '/nonassim*' + ' ' + mc_dir + '/'
    print cmd
    os.system(cmd)

    # removing the work output directory once selected files have been
    #  moved
    cmd = 'rm -f -r ' + working_dir
    print cmd
    os.system(cmd)

    # copy the namelist file to archive directory
    cmd = 'cp ./LMR_config.py ' + mc_dir + '/'
    print cmd
    os.system(cmd)

    print '\n' + str(datetime.datetime.now()) + '\n'

    #   end: DO NOT DELETE
    
# ==============================================================================
