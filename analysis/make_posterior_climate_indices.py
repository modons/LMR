#=============================================================================
# This script computes climate indices from the posterior.  Climate indices are
# calculated from the posterir rather than the prior because of the use of a 
# localization radius in the data assimilation.  Climate indices calculated from
# the prior and posterior will not be the same.  Calculations are done on the
# "subsampled" files.
#    author: Michael P. Erb
#    date  : 1/23/2018
#
#
# Revisions
# ** Andre Perkins 8/3/2018
#    - Removed dependencies on xarray and eofs and instead use netCDF4 and
#      scipy.linalg.svd to load data and calculate eofs
#    - Moved index functions from calculate_climate_indices into this script
#      so that it's self contained
#    - Cleaned up formatting for clairty and some move towards PEP8 compliance
#    - Simplified some of the numpy usage for weighted means, anomaly
#      calculations, and standardization
#=============================================================================

import numpy as np
import numpy.ma as ma
import os
from netCDF4 import Dataset
from scipy.linalg import svd

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
    sst_ncf = Dataset(experiment_sst, 'r')
    sst_all = sst_ncf.variables['sst']
    lon = sst_ncf.variables['lon'][:]
    lat = sst_ncf.variables['lat'][:]
    time_days = sst_ncf['time'][:]

    experiment_psl = os.path.join(data_dir,
                                  experiment_name + psl_exp_name_postfix,
                                  'prmsl_MCruns_ensemble_subsample.nc')

    psl_ncf = Dataset(experiment_psl, 'r')
    psl_all = psl_ncf.variables['prmsl']

    years = time_days/365
    years = years.astype(int)

    # CALCULATIONS

    # Calculate PDO, AMO, and NINO3.4 from the LMR results
    nyears = years.shape[0]
    niter = sst_all.shape[1]
    nens = sst_all.shape[4]
    pdo_pattern = {}

    # Initialize nan arrays
    pdo = np.empty((nyears, niter, nens)) * np.nan
    amo = np.empty((nyears, niter, nens)) * np.nan
    soi = np.empty((nyears, niter, nens)) * np.nan
    nino34 = np.empty((nyears, niter, nens)) * np.nan

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
             pdo_vals] = calculate_pdo(curr_sst, lat, lon)
            pdo_pattern[iteration][ens_member] = curr_pdo_patt
            pdo[:, iteration, ens_member] = pdo_vals

            curr_amo = calculate_amo(curr_sst, lat, lon)
            amo[:, iteration, ens_member] = curr_amo

            curr_soi = calculate_soi(curr_psl, lat, lon, years)
            soi[:, iteration, ens_member] = curr_soi

            curr_nino34 = calculate_nino34(curr_sst, lat, lon)
            nino34[:, iteration, ens_member] = curr_nino34

    sst_ncf.close()
    psl_ncf.close()

    # OUTPUT THE DATA TO A FILE

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


# This function takes a time-lat-lon variable and computes the global-mean for
# masked files.
def global_mean_masked(variable, lats):
    lat_weights = np.cos(np.radians(lats))
    wgt_variable = variable * lat_weights[:, None]
    variable_global = np.ma.mean(wgt_variable, axis=(1,2))
    return variable_global


# This function takes a time-lat-lon variable and computes the mean for a given
# range of i and j.
def spatial_mean(variable, lats, j_min, j_max, i_min, i_max):
    print('Computing spatial mean. i={}-{}, j={}-{}. Points are inclusive.'
          ''.format(i_min, i_max, j_min, j_max))
    variable_zonal = np.nanmean(variable[:, :, i_min:i_max+1], axis=2)
    lat_weights = np.cos(np.radians(lats))
    wgt_zonal = variable_zonal * lat_weights
    variable_mean = wgt_zonal[:, j_min:j_max+1].mean(axis=1)
    return variable_mean


def calculate_pdo(sst, lat, lon):
    """
    Pacific Decadal Oscillation (PDO) index:
    The PDO index is determined by removing the climatological seasonal cycle
    and the global mean (to remove global climate change) from the Pacific SSTs
    north of 20N, then finding the leading EOF.  The PDO index is the first
    principle component.

    References
    ----------
    http://research.jisao.washington.edu/pdo/, Mantua et al., 1997, BAMS

    """
    print("Calculating the Pacific Decadal Oscillation (PDO)")

    # Mask out land.
    sst = ma.masked_invalid(sst)

    # Remove the mean SSTs from the data.
    sst = sst - np.mean(sst, axis=0)

    # TODO: Is this correct?
    # Compute the global-mean and remove the global-mean SST from the data.
    sst_globalmean = global_mean_masked(sst, lat)
    sst_anomalies = sst - sst_globalmean[:, None, None]
    sst_anomalies = sst_anomalies.filled(np.nan)

    # Find the i and j values which cover the Pacific Ocean north of 20N
    # All latitudes between 20 and 66N
    j_indices = np.where((lat >= 20) & (lat <= 66))[0]
    # All longitudes between 100E and 100W
    i_indices = np.where((lon >= 100) & (lon <= (360-100)))[0]
    j_min = min(j_indices)
    j_max = max(j_indices)
    i_min = min(i_indices)
    i_max = max(i_indices)
    print('Computing mean of indices.  j: {}-{}, i: {}-{}'
          ''.format(j_min, j_max, i_min, i_max))
    sst_for_PDO_NPac = sst_anomalies[:, j_min:j_max+1, i_min:i_max+1]

    lon_NPac = lon[i_min:i_max+1]
    lat_NPac = lat[j_min:j_max+1]
    lon_NPac_2D,lat_NPac_2D = np.meshgrid(lon_NPac, lat_NPac)

    # Area weights are equilivent to the cosine of the latitude.
    weights_NPac = np.cos(np.radians(lat_NPac_2D))

    ntimes = sst_for_PDO_NPac.shape[0]
    spatial_shape = sst_for_PDO_NPac.shape[1:3]

    # Flatten spatial dimension
    flat_spatial_sst = sst_for_PDO_NPac.reshape(ntimes, -1)
    spatial_len = flat_spatial_sst.shape[1]

    # Find NaN values in spatial dimension and remove them
    nan_vals = np.isnan(flat_spatial_sst)
    total_nan_vals = nan_vals.sum(axis=0)
    finite_mask = total_nan_vals == 0
    compressed_sst = np.compress(finite_mask, flat_spatial_sst, axis=1)

    # Weight compressed field by latitude and calculate EOFs
    wgt_compressed_sst = compressed_sst * weights_NPac.flatten()[finite_mask]
    eofs, svals, pcs = svd(wgt_compressed_sst.T, full_matrices=False)

    # Put EOFs back into non-compressed field
    full_space_eof = np.empty(spatial_len) * np.nan
    full_space_eof[finite_mask] = eofs[:, 0]
    eof_1 = full_space_eof.reshape(spatial_shape)
    pc_1 = pcs[0]

    # Compute EOF
    # solver = Eof(sst_for_PDO_NPac,weights=weights_NPac)
    # eof_1 = solver.eofs(neofs=1)  # First EOF
    # pc_1 = solver.pcs(npcs=1)     # First principle component

    PDO_pattern = np.squeeze(eof_1)
    PDO_index = np.squeeze(pc_1)

    # Make sure that a positive value indicates a cooler northwest Pacific
    loc_sign_check_lat = np.abs(lat_NPac-39).argmin()
    loc_sign_check_lon = np.abs(lon_NPac-168).argmin()
    if PDO_pattern[loc_sign_check_lat, loc_sign_check_lon] > 0:
        PDO_pattern = -1*PDO_pattern
        PDO_index = -1*PDO_index

    # Normalize the PDO index
    PDO_index_normalized = PDO_index/np.std(PDO_index)

    return PDO_pattern, PDO_index_normalized


def calculate_amo(sst, lat, lon):
    """
    Atlantic Multidecadal Oscillation (AMO) index:
    The AMO index is computed by removing the seasonal cycle from the SST
    data, averaging SSTs anomalies over the north Atlantic (0-60N, 0-80W),
    then removing the 60S-60N mean SST anomalies, to remove the global trend.

    Reference
    ---------
    Trenberth and Shea, Geophys. Res. Lett., 2006

    """
    print("Calculating the Atlantic Multidecadal Oscillation (AMO)")

    # Mask out land.
    sst = ma.masked_invalid(sst)

    # Remove the mean SSTs from the data.
    sst = sst - np.mean(sst, axis=0)

    # Compute the mean over the North Atlantic region (0-60N, 80W-0)
    # All latitudes between 0 and 60N
    j_indices = np.where((lat >= 0) & (lat <= 60))[0]
    # All longitudes between 80W and 0W
    i_indices = np.where((lon >= (360-80)) & (lon <= 360))[0]
    j_min = min(j_indices)
    j_max = max(j_indices)
    i_min = min(i_indices)
    i_max = max(i_indices)
    print('Computing mean of indices.  j: {}-{}, i:{}-{}'
          ''.format(j_min, j_max, i_min, i_max))
    sst_mean_NAtl = spatial_mean(sst, lat, j_min, j_max, i_min, i_max)

    # Compute the mean SST from 60S-60N.
    # All latitudes between 60S and 60N
    j_indices = np.where((lat >= -60) & (lat <= 60))[0]
    # All longitudes
    i_indices = np.where((lon >= -0) & (lon <= 360))[0]
    j_min = min(j_indices)
    j_max = max(j_indices)
    i_min = min(i_indices)
    i_max = max(i_indices)
    print('Computing mean of indices.  j: {}-{}, i: {}-{}'
          ''.format(j_min, j_max, i_min, i_max))
    sst_mean_60S_60N = spatial_mean(sst, lat, j_min, j_max, i_min, i_max)
    #
    # Compute the AMO index
    AMO_index = sst_mean_NAtl - sst_mean_60S_60N
    #
    return AMO_index


def calculate_soi(psl, lat, lon, years, mean_year_begin=1951,
                  mean_year_end=1980):
    """
    Southern Oscillation (SO) index:
    The SO index is calculated based on the sea level pressure in Tahiti and
    Darwin.  It is calculated according to the equations on the NCDC site
    referenced below.

    Reference
    ---------
    https://www.ncdc.noaa.gov/teleconnections/enso/indicators/soi/
    Tahiti and Darwin lats and lons: Stenseth et al. 2003, Royal Society
    """

    print("Calculating the Southern Oscillation Index (SOI)")

    # Remove the mean PSLs from the data.
    psl = psl - np.mean(psl,axis=0)

    # Sea level pressure at closest model grid cell to Tahiti
    j_Tahiti = np.abs(lat-17.55).argmin()         # Latitude:   17.55S
    i_Tahiti = np.abs(lon-(360-149.617)).argmin()  # Longitude: 149.617W
    print('Indices for Tahiti.  j: {}, i: {}'.format(j_Tahiti, i_Tahiti))
    psl_Tahiti = psl[:, j_Tahiti, i_Tahiti]
    #
    # Sea level pressure at closest model grid cell to Darwin, Australia
    j_Darwin = np.abs(lat-12.467).argmin()  # Latitude:   12.467S
    i_Darwin = np.abs(lon-130.85).argmin()   # Longitude: 130.85E
    print('Indices for Darwin.  j: {}, i: {}'.format(j_Darwin, i_Darwin))
    psl_Darwin = psl[:, j_Darwin, i_Darwin]

    # Compute the SOI (based on the equations from NCDC)
    # Mean quantities are calculated over the period 1951-1980 unless otherwise
    # specified.
    print('Baseline years: {}-{}'.format(mean_year_begin, mean_year_end))
    year_mask = (years >= mean_year_begin) & (years <= mean_year_end)

    psl_Tahiti_anom = psl_Tahiti - psl_Tahiti[year_mask].mean()
    psl_Tahiti_standardized = psl_Tahiti_anom / psl_Tahiti_anom.std()

    psl_Darwin_anom = psl_Darwin - psl_Darwin[year_mask].mean()
    psl_Darwin_standardized = psl_Darwin_anom / psl_Darwin_anom.std()

    SO_index = psl_Tahiti_standardized - psl_Darwin_standardized
    SO_index = SO_index / SO_index.std()

    return SO_index


def calculate_nino34(sst, lat, lon):
    """
    Nino3.4 index:
    Nino3.4 is calculating by removing the seasonal cycle from the SST data,
    then averaging over the Nino3.4 region (5S-5N, 170W-120W).

    References
    ---------
    https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Nino34/
    https://www.ncdc.noaa.gov/teleconnections/enso/indicators/sst.php
    """

    print("Calculating the Nino3.4 index")

    # Mask out land.
    sst = ma.masked_invalid(sst)

    # Remove the mean SSTs from the data.
    sst = sst - np.mean(sst, axis=0)

    # Compute the mean over the Nino3.4 region (0-60N, 80W-0)
    # All latitudes between 5S and 5N
    j_indices = np.where((lat >= -5) & (lat <= 5))[0]
    # All longitudes between 170W and 120W
    i_indices = np.where((lon >= (360-170)) & (lon <= (360-120)))[0]
    j_min = min(j_indices)
    j_max = max(j_indices)
    i_min = min(i_indices)
    i_max = max(i_indices)
    print('Computing mean of indices.  j: {}-{}, i: {}-{}'
          ''.format(j_min, j_max, i_min, i_max))

    Nino34 = spatial_mean(sst, lat, j_min, j_max, i_min, i_max)

    return Nino34


if __name__ == "__main__":
    main()
