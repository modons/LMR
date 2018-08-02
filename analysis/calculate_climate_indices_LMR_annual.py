#=============================================================================
# This script calculates the following climate indices:
#
#  * Pacific Decadal Oscillation (PDO) index:
#    The PDO index is determined by removing the climatological seasonal cycle
#    and the global mean (to remove global climate change) from the Pacific SSTs
#    north of 20N, then finding the leading EOF.  The PDO index is the first
#    principle component.
#       References: http://research.jisao.washington.edu/pdo/
#                   Mantua et al., 1997, BAMS
#
#  * Atlantic Multidecadal Oscillation (AMO) index:
#    The AMO index is computed by removing the seasonal cycle from the SST
#    data, averaging SSTs anomalies over the north Atlantic (0-60N, 0-80W),
#    then removing the 60S-60N mean SST anomalies, to remove the global trend.
#       Reference: Trenberth and Shea, Geophys. Res. Lett., 2006
#
#  * Southern Oscillation (SO) index:
#    The SO index is calculated based on the sea level pressure in Tahiti and
#    Darwin.  It is calculated according to the equations on the NCDC site
#    referenced below.
#       Reference: https://www.ncdc.noaa.gov/teleconnections/enso/indicators/soi/
#       Tahiti and Darwin lats and lons: Stenseth et al. 2003, Royal Society
#
#  * Nino3.4 index:
#    Nino3.4 is calculating by removing the seasonal cycle from the SST data,
#    then averaging over the Nino3.4 region (5S-5N, 170W-120W).
#       References: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Nino34/
#                   https://www.ncdc.noaa.gov/teleconnections/enso/indicators/sst.php
#
#    author: Michael P. Erb
#    date  : January 23, 2018
#=============================================================================

import numpy as np
import numpy.ma as ma
from eofs.standard import Eof


# This function takes a time-lat-lon variable and computes the global-mean for masked files.
def global_mean_masked(variable,lats):
    variable_zonal = np.ma.mean(variable,axis=2)
    lat_weights = np.cos(np.radians(lats))
    variable_global = np.zeros(variable.shape[0])
    variable_global[:] = np.nan
    for time in range(variable.shape[0]):
        variable_global[time] = np.ma.average(variable_zonal[time,:],axis=0,weights=lat_weights)
    #
    return variable_global


# This function takes a time-lat-lon variable and computes the mean for a given range of i and j.
def spatial_mean(variable,lats,j_min,j_max,i_min,i_max):
    print 'Computing spatial mean. i='+str(i_min)+'-'+str(i_max)+', j='+str(j_min)+'-'+str(j_max)+'.  Points are inclusive.'
    variable_zonal = np.nanmean(variable[:,:,i_min:i_max+1],axis=2)
    lat_weights = np.cos(np.radians(lats))
    variable_mean = np.zeros(variable.shape[0])
    variable_mean[:] = np.nan
    for time in range(variable.shape[0]):
        variable_mean[time] = np.average(variable_zonal[time,j_min:j_max+1],axis=0,weights=lat_weights[j_min:j_max+1])
    #
    return variable_mean


# Calculate PDO
def calculate_PDO( sst,lat,lon,experiment ):
    print("Calculating the Pacific Decadal Oscillation (PDO)")
    #
    # Mask out land.
    sst = ma.masked_invalid(sst)
    #
    # Remove the mean SSTs from the data.
    sst = sst - np.mean(sst,axis=0)
    #
    # Compute the global-mean and remove the global-mean SST from the data.
    sst_globalmean = global_mean_masked(sst,lat)
    sst_anomalies = sst - sst_globalmean[:,None,None]
    #
    # Find the i and j values which cover the Pacific Ocean north of 20N
    j_indices = np.where((lat >= 20) & (lat <= 66))[0]          # All latitudes between 20 and 66N
    i_indices = np.where((lon >= 100) & (lon <= (360-100)))[0]  # All longitudes between 100E and 100W
    j_min = min(j_indices); j_max = max(j_indices)
    i_min = min(i_indices); i_max = max(i_indices)
    print('Computing mean of indices.  j: '+str(j_min)+'-'+str(j_max)+', i: '+str(i_min)+'-'+str(i_max))
    sst_for_PDO_NPac = sst_anomalies[:,j_min:j_max+1,i_min:i_max+1]
    #
    lon_NPac = lon[i_min:i_max+1]
    lat_NPac = lat[j_min:j_max+1]
    lon_NPac_2D,lat_NPac_2D = np.meshgrid(lon_NPac,lat_NPac)
    #
    # Area weights are equilivent to the cosine of the latitude.
    weights_NPac = np.cos(np.radians(lat_NPac_2D))
    #
    # Compute EOF
    solver = Eof(sst_for_PDO_NPac,weights=weights_NPac)
    eof_1 = solver.eofs(neofs=1)  # First EOF
    pc_1 = solver.pcs(npcs=1)     # First principle component
    #
    PDO_pattern = np.squeeze(eof_1)
    PDO_index = np.squeeze(pc_1)
    #
    # Make sure that a positive value indicates a cooler northwest Pacific
    if PDO_pattern[np.abs(lat_NPac-39).argmin(),np.abs(lon_NPac-168).argmin()] > 0:
        PDO_pattern = -1*PDO_pattern
        PDO_index = -1*PDO_index
    #
    # Rescale the PDO pattern - not done.  Probably unnecessary.
    #
    # Normalize the PDO index
    PDO_index_normalized = PDO_index/np.std(PDO_index)
    #
    return PDO_pattern, PDO_index_normalized, lat_NPac, lon_NPac


# Calculate AMO
def calculate_AMO( sst,lat,lon,experiment ):
    print("Calculating the Atlantic Multidecadal Oscillation (AMO)")
    #
    # Mask out land.
    sst = ma.masked_invalid(sst)
    #
    # Remove the mean SSTs from the data.
    sst = sst - np.mean(sst,axis=0)
    #
    # Compute the mean over the North Atlantic region (0-60N, 80W-0)
    j_indices = np.where((lat >= 0) & (lat <= 60))[0]          # All latitudes between 0 and 60N
    i_indices = np.where((lon >= (360-80)) & (lon <= 360))[0]  # All longitudes between 80W and 0W
    j_min = min(j_indices); j_max = max(j_indices)
    i_min = min(i_indices); i_max = max(i_indices)
    print('Computing mean of indices.  j: '+str(j_min)+'-'+str(j_max)+', i: '+str(i_min)+'-'+str(i_max))
    sst_mean_NAtl = spatial_mean(sst,lat,j_min,j_max,i_min,i_max)
    #
    # Compute the mean SST from 60S-60N.
    j_indices = np.where((lat >= -60) & (lat <= 60))[0]  # All latitudes between 60S and 60N
    i_indices = np.where((lon >= -0) & (lon <= 360))[0]  # All longitudes
    j_min = min(j_indices); j_max = max(j_indices)
    i_min = min(i_indices); i_max = max(i_indices)
    print('Computing mean of indices.  j: '+str(j_min)+'-'+str(j_max)+', i: '+str(i_min)+'-'+str(i_max))
    sst_mean_60S_60N = spatial_mean(sst,lat,j_min,j_max,i_min,i_max)
    #
    # Compute the AMO index
    AMO_index = sst_mean_NAtl - sst_mean_60S_60N
    #
    return AMO_index


# Calculate SOI
def calculate_SOI( psl,lat,lon,years,mean_year_begin=1951,mean_year_end=1980 ):
    print("Calculating the Southern Oscillation Index (SOI)")
    #
    # Remove the mean PSLs from the data.
    psl = psl - np.mean(psl,axis=0)
    #
    # Sea level pressure at closest model grid cell to Tahiti
    j_Tahiti = np.abs(lat--17.55).argmin()         # Latitude:   17.55S
    i_Tahiti = np.abs(lon-(360-149.617)).argmin()  # Longitude: 149.617W
    print('Indices for Tahiti.  j: '+str(j_Tahiti)+', i: '+str(i_Tahiti))
    psl_Tahiti = psl[:,j_Tahiti,i_Tahiti]
    #
    # Sea level pressure at closest model grid cell to Darwin, Australia
    j_Darwin = np.abs(lat--12.467).argmin()  # Latitude:   12.467S
    i_Darwin = np.abs(lon-130.85).argmin()   # Longitude: 130.85E
    print('Indices for Darwin.  j: '+str(j_Darwin)+', i: '+str(i_Darwin))
    psl_Darwin = psl[:,j_Darwin,i_Darwin]
    #
    # Compute the SOI (based on the equations from NCDC)
    # Mean quantities are calculated over the period 1951-1980 unless otherwise specified.
    print('Baseline years: '+str(mean_year_begin)+'-'+str(mean_year_end))
    index_begin = np.where(years == mean_year_begin)[0][0]
    index_end   = np.where(years == mean_year_end)[0][0]
    psl_Tahiti_standardized = (psl_Tahiti - np.mean(psl_Tahiti[index_begin:index_end+1])) / np.sqrt(np.mean(np.square(psl_Tahiti - np.mean(psl_Tahiti[index_begin:index_end+1]))))
    psl_Darwin_standardized = (psl_Darwin - np.mean(psl_Darwin[index_begin:index_end+1])) / np.sqrt(np.mean(np.square(psl_Darwin - np.mean(psl_Darwin[index_begin:index_end+1]))))
    SO_index = (psl_Tahiti_standardized - psl_Darwin_standardized) / np.sqrt(np.mean(np.square(psl_Tahiti_standardized - psl_Darwin_standardized)))
    #
    return SO_index


# Calculate Nino3.4
def calculate_Nino34( sst,lat,lon,experiment ):
    print("Calculating the Nino3.4 index")
    #
    # Mask out land.
    sst = ma.masked_invalid(sst)
    #
    # Remove the mean SSTs from the data.
    sst = sst - np.mean(sst,axis=0)
    #
    # Compute the mean over the Nino3.4 region (0-60N, 80W-0)
    j_indices = np.where((lat >= -5) & (lat <= 5))[0]                 # All latitudes between 5S and 5N
    i_indices = np.where((lon >= (360-170)) & (lon <= (360-120)))[0]  # All longitudes between 170W and 120W
    j_min = min(j_indices); j_max = max(j_indices)
    i_min = min(i_indices); i_max = max(i_indices)
    print('Computing mean of indices.  j: '+str(j_min)+'-'+str(j_max)+', i: '+str(i_min)+'-'+str(i_max))
    Nino34 = spatial_mean(sst,lat,j_min,j_max,i_min,i_max)
    #
    return Nino34

