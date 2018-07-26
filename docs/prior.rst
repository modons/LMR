.. _prior:

Adding new priors
=================

What prior folders and file names should look like
--------------------------------------------------
If you can easily move your prior data put it in the ../LMR/data/model folder.
Your prior netcdf files should be saved in a folder with the same name as the
prior as defined in the config.yml file. If you are adding a new prior, create
a new folder with the folder name that will be called in the config.yml file.
For example, the ccsm4 past1000 prior uses this folder name:
ccsm4_last_millenium, so this folder name is defined as the prior source in the
config file. This folder name will also need to be added in various places in
LMR_prior.py and datasets.yml - see details below in ‘How to add a new prior to
the LMR’. If your prior data is too larger or cumbersome to move you can leave
it where it is and specify the path at a later step (Step 2). However, make sure
that the filenames adhere to the convention described below.

Within this new prior folder, place all the new prior files. Prior files used
in the LMR should start with the variable name, followed by the level, the model
component, etc.

For example, a few relevant ccsm4 past1000
filenames:
tas_sfc_Amon_CCSM4_past1000_085001-185012.nc
pr_sfc_Amon_CCSM4_past1000_085001-185012.nc
zg_500hPa_Amon_CCSM4_past1000_085001-185012.nc
Each prior file should contain four variables: the variable of interest, time,
latitude and longitude. The dimensions also need to be named: time, lat, and
lon.


How to add a new prior to the LMR
---------------------------------
1. Create a class for your new prior in LMR_prior.py. Creating a new class
   involves changing the LMR_prior.py in two locations (you can copy, paste, and
   change previous prior information to fit your new prior folder name):

   Add the prior folder name in another ‘elif’ statements in the
   prior_assignment()
   function (between lines ~50 and ~100). You can do this by copying and
   pasting an
   ‘elif’ statement from a previous prior and changing the iprior variable to a
   string of the new prior folder name.
   Define new class for the prior (after line ~337). Do this by copying a class
   from another prior (~10 lines of code) and simply change the name of the
   class
   to reflect your prior folder name. This class should contain one read_prior()
   function.

2. Add your new prior to datasets.yml. The instrumental-based datasets are
   listed first, followed by the model prior dataset information (after line
   ~75).
   Copy and paste the information from one of the model priors and change the
   relevant folder name in ‘datafile’ and variable names you have data for and
   wish to reconstruct in ‘available_vars’. Make sure that all the relevant
   variable names are listed here, otherwise the LMR will not be able to
   reconstruct them. Note also that ‘datadir=null’ will default to LMR_path
   (../LMR/data/model folder). If you’re data is in another location, you can
   add that directory here (datadir = ‘path’) and LMR will know where to look
   for the data.

3. Add the appropriate model prior source information to the config.yml file
   (prior_source, line ~350). Although not necessary, you may want to rename the
   experiment to match the new prior information. For example, change the nexp
   (experiment name, line ~20-25) to: ccsm4_past1000_PAGES2kdbv2_annualPSM.

4. In the LMR/misc folder, precalculate ye’s by running build_ye_file.py.
   You will need to follow the above steps 1 through 4 every time you add a new
   prior.

5. Run the main LMR code (LMR_wrapper.py) once you have precalculated the ye’s.

