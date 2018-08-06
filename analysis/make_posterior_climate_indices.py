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
#    - Changed the PDO calculation to find the pattern on the grand ensemble
#      mean and then use that pattern to determine an index for each ensemble
#      member.  Still assumes that the first ensemble mean EOF is the PDO.
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

    # Load Ensemble Mean and take average over MC-iterations for PDO calc
    exp_sst_ensmean = os.path.join(data_dir,
                                   experiment_name + sst_exp_name_postfix,
                                   'sst_MCruns_ensemble_mean.nc')
    with Dataset(exp_sst_ensmean, 'r') as sst_ensmean_ncf:
        sst_grand_mean = sst_ensmean_ncf.variables['sst'][:].mean(axis=1)

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

    # Initialize nan arrays
    pdo = np.empty((nyears, niter, nens)) * np.nan
    amo = np.empty((nyears, niter, nens)) * np.nan
    soi = np.empty((nyears, niter, nens)) * np.nan
    nino34 = np.empty((nyears, niter, nens)) * np.nan

    [ensmean_pdo_patt,
     ensmean_pdo_idx,
     lat_npac,
     lon_npac] = calculate_pdo(sst_grand_mean, lat, lon)

    pdo_pat_out = os.path.join(output_dir,
                               'ensmean_pdo_pattern_index.npz')
    np.savez(pdo_pat_out, pdo_pattern=ensmean_pdo_patt,
             pdo_idx=ensmean_pdo_idx, lat_npac=lat_npac, lon_npac=lon_npac)

    for iteration in range(niter):
        for ens_member in range(nens):
            print('\n === Calculating climate indices.  Iteration: ' +
                  str(iteration+1) + '/' + str(niter) +
                  ',  Ensemble member: ' + str(ens_member+1) +
                  '/' + str(nens) + ' ===')

            curr_sst = sst_all[:, iteration, :, :, ens_member]
            curr_psl = psl_all[:, iteration, :, :, ens_member]

            curr_pdo_idx = calculate_pdo_index(ensmean_pdo_patt,
                                               curr_sst, lat, lon)
            pdo[:, iteration, ens_member] = curr_pdo_idx

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
    outfile.createDimension('lat_npac', len(lat_npac))
    outfile.createDimension('lon_npac', len(lon_npac))

    # define variables & upload the data to file

    # time
    time = outfile.createVariable('time', 'i', ('time',))
    time.description = 'time'
    time.units = 'time since 0000-01-01 00:00:00'

    varout_ensmean_pdo_patt = outfile.createVariable('ensmean_pdo_pattern',
                                                     'f',
                                                     ('lat_npac', 'lon_npac'))
    varout_ensmean_pdo_patt.description = 'ensmean_pdo_pattern'
    varout_ensmean_pdo_patt.long_name = 'Pacific Decadal Oscillation Pattern'
    varout_ensmean_pdo_patt.units = ''
    varout_ensmean_pdo_patt.level = 'sfc'

    varout_lat_npac = outfile.createVariable('lat_npac', 'f', ('lat_npac',))
    varout_lat_npac.description = 'North Pacific latitudes for PDO Pattern'
    varout_lat_npac.long_name = 'North Pacific latitudes'
    varout_lat_npac.units = 'Degrees latitude'

    varout_lon_npac = outfile.createVariable('lon_npac', 'f', ('lon_npac',))
    varout_lon_npac.description = 'North Pacific longitudes for PDO Pattern'
    varout_lon_npac.long_name = 'North Pacific longitudes'
    varout_lon_npac.units = 'Degrees longitude'

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
    varout_ensmean_pdo_patt[:] = ensmean_pdo_patt
    varout_lat_npac[:] = lat_npac
    varout_lon_npac[:] = lon_npac

    # Closing the file
    outfile.close()


# This function takes a time-lat-lon variable and computes the global-mean for
# masked files.
def global_mean_masked(variable, lats):
    lat_weights = np.cos(np.radians(lats))
    wgt_variable = variable * lat_weights[:, None]
    variable_global = np.ma.mean(wgt_variable, axis=(1, 2))
    return variable_global


def spatial_mean_bounded(variable, lat, lon, lat_bnds, lon_bnds):
    print('Computing spatial mean. lat={}-{}, lon={}-{}. Points are inclusive.'
          ''.format(*lat_bnds, *lon_bnds))

    [reduced_var,
     reduced_lat,
     reduced_lon] = reduce_to_lat_lon_box(variable, lat, lon,
                                          lat_bnds, lon_bnds)

    avg_over_lat = global_mean_masked(reduced_var, reduced_lat)

    return avg_over_lat


# This functions takes a field reduces it to lat/lon boundaries
def reduce_to_lat_lon_box(field, lat, lon, lat_bnds, lon_bnds):
    lat_lb, lat_ub = lat_bnds
    lon_lb, lon_ub = lon_bnds

    lat_mask = (lat >= lat_lb) & (lat <= lat_ub)
    lon_mask = (lon >= lon_lb) & (lon <= lon_ub)

    lon_compressed = lon[lon_mask]
    lat_compressed = lat[lat_mask]

    field_compressed = np.compress(lon_mask, field, axis=2)
    field_compressed = np.compress(lat_mask, field_compressed, axis=1)

    return field_compressed, lat_compressed, lon_compressed


def remove_spatial_nans(field):

    assert field.ndim > 1

    # Flatten spatial dimension
    ntimes = field.shape[0]
    flat_field = field.reshape(ntimes, -1)

    # Find NaN values in spatial dimension and remove them
    nan_vals = np.isnan(flat_field)
    total_nan_vals = nan_vals.sum(axis=0)
    # No NaNs if sum == 0
    finite_mask = total_nan_vals == 0

    compressed_field = np.compress(finite_mask, flat_field, axis=1)

    return compressed_field, finite_mask



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

    # Compute the global-mean and remove the global-mean SST from the data.
    sst_globalmean = global_mean_masked(sst, lat)
    sst_anomalies = sst - sst_globalmean[:, None, None]
    sst_anomalies = sst_anomalies.filled(np.nan)

    # Reduce field to all latitudes between 20-66N and 100E-100W
    [sst_for_PDO_NPac,
     lat_NPac,
     lon_NPac] = reduce_to_lat_lon_box(sst_anomalies, lat, lon,
                                       lat_bnds=(20, 66),
                                       lon_bnds=(100, 260))

    # lon_NPac = lon[i_min:i_max+1]
    # lat_NPac = lat[j_min:j_max+1]
    lon_NPac_2D, lat_NPac_2D = np.meshgrid(lon_NPac, lat_NPac)

    # Area weights are equilivent to the cosine of the latitude.
    weights_NPac = np.cos(np.radians(lat_NPac_2D))
    spatial_shape = sst_for_PDO_NPac.shape[1:3]

    [compressed_sst,
     valid_data] = remove_spatial_nans(sst_for_PDO_NPac)

    # Weight compressed field by latitude and calculate EOFs
    wgt_compressed_sst = compressed_sst * weights_NPac.flatten()[valid_data]
    eofs, svals, pcs = svd(wgt_compressed_sst.T, full_matrices=False)

    # Put EOFs back into non-compressed field
    full_space_eof = np.empty_like(valid_data,
                                   dtype=np.float) * np.nan
    full_space_eof[valid_data] = eofs[:, 0]
    eof_1 = full_space_eof.reshape(spatial_shape)
    pc_1 = pcs[0]

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

    return PDO_pattern, PDO_index_normalized, lat_NPac, lon_NPac


def calculate_pdo_index(pdo_pattern, sst, lat, lon):
    """
    Projects PDO pattern on to data.  Assumes sst field is time x lat x lon.
    """
    if np.ma.is_masked(sst):
        sst = sst.filled(np.nan)

    # Reduce field to all latitudes between 20-66N and 100E-100W
    sst_for_PDO, _, _ = reduce_to_lat_lon_box(sst, lat, lon,
                                              lat_bnds=(20, 66),
                                              lon_bnds=(100, 260))

    # Remove NaN values
    [compressed_sst,
     valid_data] = remove_spatial_nans(sst_for_PDO)

    # Remove NaN value locations in sst field from PDO pattern
    pdo_pattern = pdo_pattern.flatten()[valid_data]

    pdo_index = compressed_sst @ pdo_pattern
    pdo_index = pdo_index / pdo_index.std()

    return pdo_index


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

    # Reduce field and average to lats from 0-60N and 80W - 0W
    sst_mean_NAtl = spatial_mean_bounded(sst, lat, lon,
                                         lat_bnds=(0, 60),
                                         lon_bnds=(280, 360))

    # Reduce field and average to lats from 60S-60N
    sst_mean_60S_60N = spatial_mean_bounded(sst, lat, lon,
                                            lat_bnds=(-60, 60),
                                            lon_bnds=(0, 360))

    # Compute the AMO index
    AMO_index = sst_mean_NAtl - sst_mean_60S_60N

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

    # Reduce field and average Nino3.4 region (5S-5N, 170W-120W)
    Nino34 = spatial_mean_bounded(sst, lat, lon, lat_bnds=(-5, 5),
                                  lon_bnds=(190, 240))

    return Nino34


if __name__ == "__main__":
    main()
