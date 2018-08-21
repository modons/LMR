"""
Module: LMR_convertNPZtoNETCDF_gmt.py

Purpose: Converts LMR output from .npz files to netcdf files. 
         Now restricted to gmt reconstruction variables, for every
         Monte-Carlo realization included in the experiment.

Originator: Robert Tardif | Univ. of Washington, Dept. of Atmospheric Sciences
            Michael Erb   | University of Southern California
                          | June 2017

Revisions: None

"""
import glob
import numpy as np
from netCDF4 import Dataset

# LMR-specific import
from LMR_utils import natural_sort

# --- Begin section of user-defined parameters ---

# name of directory where the output of LMR experiments are located
#datadir = '/home/disk/ekman4/rtardif/LMR/output'
datadir = '/home/disk/kalman3/rtardif/LMR/output'
#datadir = '/home/scec-00/lmr/erbm/LMR/archive_output/older_experiments'

# name of the experiment
nexp = 'test'

# --- End section of user-defined parameters ---

expdir = datadir + '/'+nexp

# where the netcdf files are created 
outdir = expdir

print('\n Getting information on Monte-Carlo realizations...\n')

dirs = glob.glob(expdir+"/r*")
# keep names of MC directories (i.r. "r...") only 
mcdirs = [item.split('/')[-1] for item in dirs]
# Make sure list is properly sorted
mcdirs = natural_sort(mcdirs)
# number of MC realizations found
niters = len(mcdirs) 

print('mcdirs:' + str(mcdirs))
print('niters = ' + str(niters))

print('\n Getting information on reconstructed variables...\n')

# look in first "mcdirs" only. It should be the same for all. 
workdir = expdir+'/'+mcdirs[0]

print('\n Global-mean temperature file.')
# Loop over realizations
r = 0
for dir in mcdirs:
    fname = expdir+'/'+dir+'/gmt_ensemble.npz'
    npzfile = np.load(fname)
    
    # Get the reconstructed field
    gmt_values = npzfile['gmt_ensemble']
    nhmt_values = npzfile['nhmt_ensemble']
    shmt_values = npzfile['shmt_ensemble']
    
    if r == 0: # first realization
    
        npzcontent = npzfile.files
        print('  file contents:', npzcontent)
        
        # get the years in the reconstruction
        years =  npzfile['recon_times']
    
        # no spatial coordinate, must be a scalar (time series)
        field_type='1D:time_series'
        print('  field type:', field_type)
   
        # declare master array that will contain data from all the M-C realizations 
        # (i.e. the "Monte-Carlo ensemble")
        dims = gmt_values.shape
        print('  gmt field dimensions', dims)
        tmp_gmt = np.expand_dims(gmt_values, axis=0)
        tmp_nhmt = np.expand_dims(nhmt_values, axis=0)
        tmp_shmt = np.expand_dims(shmt_values, axis=0)
        # Form the array with the right total dimensions
        mc_ens_gmt = np.repeat(tmp_gmt,niters,axis=0)
        mc_ens_nhmt = np.repeat(tmp_nhmt,niters,axis=0)
        mc_ens_shmt = np.repeat(tmp_shmt,niters,axis=0)
        
    else:
        mc_ens_gmt[r,:,:] = gmt_values
        mc_ens_nhmt[r,:,:] = nhmt_values
        mc_ens_shmt[r,:,:] = shmt_values
        
    r = r + 1
    
    
# Roll array to get dims as [time, niters, nens]
mc_ens_outarr_gmt = np.swapaxes(mc_ens_gmt,0,1)
mc_ens_outarr_nhmt = np.swapaxes(mc_ens_nhmt,0,1)
mc_ens_outarr_shmt = np.swapaxes(mc_ens_shmt,0,1)

# Create the netcdf file for the current variable
outfile_nc = outdir+'/gmt_ensemble_MCiters.nc'
outfile = Dataset(outfile_nc, 'w', format='NETCDF4')
outfile.description = 'LMR climate reconstruction for global-mean, NH-mean, and SH-mean surface air temperature (K).'
outfile.experiment = nexp
outfile.comment = 'File contains all ensemble values (ensemble_member) for each Monte-Carlo realization (iteration_member)'

# define dimensions
ntime = years.shape[0]
nensemble = mc_ens_outarr_gmt.shape[2]
    
outfile.createDimension('time', ntime)
outfile.createDimension('iteration_member', niters)
outfile.createDimension('ensemble_member', nensemble)
    
    
# define variables & upload the data to file
   
# time
time = outfile.createVariable('time', 'i', ('time',))
time.description = 'time'
time.long_name = 'year CE'
    
# reconstructed fields
varout_gmt = outfile.createVariable('gmt', 'f', ('time','iteration_member','ensemble_member'))
varout_gmt.description = 'gmt'
varout_gmt.long_name = 'Global-mean surface air temperature anomaly'
varout_gmt.units = 'K'

varout_nhmt = outfile.createVariable('nhmt', 'f', ('time','iteration_member','ensemble_member'))
varout_nhmt.description = 'nhmt'
varout_nhmt.long_name = 'Northern Hemisphere-mean surface air temperature anomaly'
varout_nhmt.units = 'K'

varout_shmt = outfile.createVariable('shmt', 'f', ('time','iteration_member','ensemble_member'))
varout_shmt.description = 'shmt'
varout_shmt.long_name = 'Southern Hemisphere-mean surface air temperature anomaly'
varout_shmt.units = 'K'
    
# upload the data to file
time[:]   = years
varout_gmt[:] = mc_ens_outarr_gmt
varout_nhmt[:] = mc_ens_outarr_nhmt
varout_shmt[:] = mc_ens_outarr_shmt

# Closing the file
outfile.close()
