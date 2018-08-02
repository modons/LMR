#=============================================================================
# This script computes climate indices from the posterior.  Climate indices are
# calculated from the posterir rather than the prior because of the use of a 
# localization radius in the data assimilation.  Climate indices calculated from
# the prior and posterior will not be the same.  Calculations are done on the
# "subsampled" files.
#    author: Michael P. Erb
#    date  : 1/23/2018
#=============================================================================

import numpy as np
import xarray as xr
from netCDF4 import Dataset
import calculate_climate_indices_LMR_annual

data_dir = '/home/scec-00/lmr/erbm/LMR/archive_output/'
experiment_name = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'

experiment = data_dir+experiment_name

### LOAD DATA
handle = xr.open_dataset(data_dir+experiment_name+'/sst_MCruns_ensemble_subsample.nc',decode_times=False)
sst_all = handle['sst'].values
lon = handle['lon'].values
lat = handle['lat'].values
time_days = handle['time'].values
handle.close()

handle = xr.open_dataset(data_dir+experiment_name+'/prmsl_MCruns_ensemble_subsample.nc',decode_times=False)
psl_all = handle['prmsl'].values
handle.close()

years = time_days/365
years = years.astype(int)


### CALCULATIONS

# Calculate PDO, AMO, and NINO3.4 from the LMR results
nyears = years.shape[0]
niter  = sst_all.shape[1]
nens   = sst_all.shape[4]
pdo_pattern = {}
pdo    = np.zeros((nyears,niter,nens)); pdo[:]    = np.nan
amo    = np.zeros((nyears,niter,nens)); amo[:]    = np.nan
soi    = np.zeros((nyears,niter,nens)); soi[:]    = np.nan
nino34 = np.zeros((nyears,niter,nens)); nino34[:] = np.nan

for iteration in range(niter):
    pdo_pattern[iteration] = {}
    for ens_member in range(nens):
        print('\n === Calculating climate indices.  Iteration: '+str(iteration+1)+'/'+str(niter)+',  Ensemble member: '+str(ens_member+1)+'/'+str(nens)+' ===')
        pdo_pattern[iteration][ens_member], pdo[:,iteration,ens_member], lat_NPac, lon_NPac = calculate_climate_indices_LMR_annual.calculate_PDO(sst_all[:,iteration,:,:,ens_member],lat,lon,experiment)
        amo[:,iteration,ens_member]    = calculate_climate_indices_LMR_annual.calculate_AMO(sst_all[:,iteration,:,:,ens_member],lat,lon,experiment)
        soi[:,iteration,ens_member]    = calculate_climate_indices_LMR_annual.calculate_SOI(psl_all[:,iteration,:,:,ens_member],lat,lon,years)
        nino34[:,iteration,ens_member] = calculate_climate_indices_LMR_annual.calculate_Nino34(sst_all[:,iteration,:,:,ens_member],lat,lon,experiment)


### OUTPUT THE DATA TO A FILE

var_desc = \
    {
        'amo_sfc_Omon'    : ('amo', 'Atlantic Multidecadal Oscillation', ''), \
        'pdo_sfc_Omon'    : ('pdo', 'Pacific Decadal Oscillation', ''),       \
        'soi_sfc_Amon'    : ('soi', 'Southern Oscillation Index', ''),        \
        'nino34_sfc_Omon' : ('nino34', 'Nino3.4', ''),                        \
    }

# Create the netcdf file for the current variable
outfile_nc = 'data/climate_indices_MCruns_ensemble_subsample_calc_from_posterior.nc'
outfile = Dataset(outfile_nc, 'w', format='NETCDF4')
outfile.description = 'Climate indices calculated from LMR reconstruction'
outfile.experiment = experiment_name
outfile.comment = 'File contains subsampled values for each Monte-Carlo realization (member)'

# define dimensions
outfile.createDimension('time', nyears)
outfile.createDimension('member', niter)
outfile.createDimension('ensemble_member', nens)

# define variables & upload the data to file

# time
time = outfile.createVariable('time', 'i', ('time',))
time.description = 'time'
time.long_name = 'time since 0000-01-01 00:00:00'

varout_amo = outfile.createVariable('amo', 'f', ('time','member','ensemble_member'))
varout_amo.description = var_desc['amo_sfc_Omon'][0]
varout_amo.long_name = var_desc['amo_sfc_Omon'][1]
varout_amo.units = var_desc['amo_sfc_Omon'][2]
varout_amo.level = 'sfc'

varout_pdo = outfile.createVariable('pdo', 'f', ('time','member','ensemble_member'))
varout_pdo.description = var_desc['pdo_sfc_Omon'][0]
varout_pdo.long_name = var_desc['pdo_sfc_Omon'][1]
varout_pdo.units = var_desc['pdo_sfc_Omon'][2]
varout_pdo.level = 'sfc'

varout_soi = outfile.createVariable('soi', 'f', ('time','member','ensemble_member'))
varout_soi.description = var_desc['soi_sfc_Amon'][0]
varout_soi.long_name = var_desc['soi_sfc_Amon'][1]
varout_soi.units = var_desc['soi_sfc_Amon'][2]
varout_soi.level = 'sfc'

varout_nino34 = outfile.createVariable('nino34', 'f', ('time','member','ensemble_member'))
varout_nino34.description = var_desc['nino34_sfc_Omon'][0]
varout_nino34.long_name = var_desc['nino34_sfc_Omon'][1]
varout_nino34.units = var_desc['nino34_sfc_Omon'][2]
varout_nino34.level = 'sfc'

# upload the data to file
time[:]          = time_days
varout_amo[:]    = amo
varout_pdo[:]    = pdo
varout_soi[:]    = soi
varout_nino34[:] = nino34

# Closing the file
outfile.close()
