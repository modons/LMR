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
import numpy.ma as ma
import os
from netCDF4 import Dataset

def main():
    data_dir = '/home/disk/kalman3/rtardif/LMR/output'
    experiment_name = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'
    output_dir = '/home/katabatic/wperkins/LMR_output/production_indices'

    psl_exp_name_postfix = '_psl'
    sst_exp_name_postfix = '_tos'

    ### LOAD DATA
    experiment_sst = os.path.join(data_dir,
                                  experiment_name + sst_exp_name_postfix,
                                  'sst_MCruns_ensemble_subsample.nc')
    with Dataset(experiment_sst, 'r') as sst_ncf:
        sst_all = sst_ncf.variables['sst'][:]
        lon = sst_ncf.variables['lon'][:]
        lat = sst_ncf.variables['lat'][:]
        time_days = sst_ncf['time'][:]

    experiment_psl = os.path.join(data_dir,
                                  experiment_name + psl_exp_name_postfix,
                                  'prmsl_MCruns_ensemble_subsample.nc')

    with Dataset(experiment_psl, 'r') as psl_ncf:
        psl_all = psl_ncf.variables['prmsl'][:]

    years = time_days/365
    years = years.astype(int)


    ### CALCULATIONS

    # Calculate PDO, AMO, and NINO3.4 from the LMR results
    nyears = years.shape[0]
    niter = sst_all.shape[1]
    nens = sst_all.shape[4]
    pdo_pattern = {}

    # Initialize nan arrays
    pdo = np.empty((nyears,niter,nens)) * np.nan
    amo = np.empty((nyears,niter,nens)) * np.nan
    soi = np.empty((nyears,niter,nens)) * np.nan
    nino34 = np.empty((nyears,niter,nens)) * np.nan

    for iteration in range(niter):
        pdo_pattern[iteration] = {}
        for ens_member in range(nens):
            print('\n === Calculating climate indices.  Iteration: ' +
                  str(iteration+1) + '/' + str(niter) +
                  ',  Ensemble member: ' + str(ens_member+1) +
                  '/' + str(nens) + ' ===')

            curr_sst = sst_all[:, iteration, :, :, ens_member]
            curr_psl = psl_all[:, iteration, :, :, ens_member]

            [curr_pdo_patt,
             pdo_vals,
             lat_NPac,
             lon_NPac] = calculate_PDO(curr_sst, lat, lon)
            pdo_pattern[iteration][ens_member] = curr_pdo_patt
            pdo[:, iteration, ens_member] = pdo_vals

            curr_amo = calculate_AMO(curr_sst, lat, lon)
            amo[:, iteration, ens_member] = curr_amo

            curr_soi = calculate_SOI(curr_psl, lat, lon, years)
            soi[:, iteration, ens_member] = curr_soi

            curr_nino34 = calculate_Nino34(curr_sst, lat, lon)
            nino34[:, iteration, ens_member] = curr_nino34


    ### OUTPUT THE DATA TO A FILE

    # Create the netcdf file for the current variable
    outfile_nc = 'posterior_climate_indices_MCruns_ensemble_subsample.nc'
    outpath = os.path.join(output_dir, outfile_nc)
    outfile = Dataset(outpath, 'w', format='NETCDF4')
    outfile.description = 'Climate indices calculated from an LMR reconstruction'
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
    varout_amo.description = 'amo'
    varout_amo.long_name = 'Atlantic Multidecadal Oscillation'
    varout_amo.units = ''
    varout_amo.level = 'sfc'

    varout_pdo = outfile.createVariable('pdo', 'f', ('time','member','ensemble_member'))
    varout_pdo.description = 'pdo'
    varout_pdo.long_name = 'Pacific Decadal Oscillation'
    varout_pdo.units = ''
    varout_pdo.level = 'sfc'

    varout_soi = outfile.createVariable('soi', 'f', ('time','member','ensemble_member'))
    varout_soi.description = 'soi'
    varout_soi.long_name = 'Southern Oscillation Index'
    varout_soi.units = ''
    varout_soi.level = 'sfc'

    varout_nino34 = outfile.createVariable('nino34', 'f', ('time','member','ensemble_member'))
    varout_nino34.description = 'nino34'
    varout_nino34.long_name = 'Nino3.4'
    varout_nino34.units = ''
    varout_nino34.level = 'sfc'

    # upload the data to file
    time[:]          = time_days
    varout_amo[:]    = amo
    varout_pdo[:]    = pdo
    varout_soi[:]    = soi
    varout_nino34[:] = nino34

    # Closing the file
    outfile.close()


if __name__ == "__main__":
    main()