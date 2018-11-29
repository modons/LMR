"""
Module: load_gridded_data.py

Purpose: Contains functions used in the uploading of the various gridded datasets 
         (calibration and prior) needed by the LMR. 

Originator: Robert Tardif, U. of Washington, January 2015

Revisions:
          - Added function for the upload of the GPCC historical precipitation dataset.
            [R. Tardif, U of Washington, December 2015]
          - Added function for uploading the ERA20CM *ensemble* to be used as prior.
            [R. Tardif, U of Washington, February 2016]
          - Added function for the upload of the Dai historical PDSI dataset.
            [R. Tardif, U of Washington, May 2016]
          - Added function to upload the data from the TraCE21ka climate simulation.
            [R. Tardif, U of Washington, December 2016]
          - Modified the read_gridded_data_CMIP5_model function to handle fields with 
            lats/lons defined using 2d arrays (on irregular grids), and added
            possibility of returning multiyear averages.
            [R. Tardif, U of Washington, March 2017]
          - Modified the read_gridded_data_CMIP5_model function to handle fields with
            dims as [time,lat] (latitudinally-averaged) and [time,lev,lat] 
            (latitude-depth cross-sections).
            [R. Tardif, U of Washington, April 2017]
          - Minor fix to detrending functionality to handle fields with masked values
            [R. Tardif, U of Washington, April 2017]
          - Added the function read_gridded_data_cGENIE_model to read data files 
            derived from output of the cGENIE EMIC. 
            [R. Tardif, U of Washington, Aug 2017]
          - Modified the read_gridded_data_CMIP5_model function for greater flexibility
            in handling names of geographical coordinates in input .nc files.
            [R. Tardif, U of Washington, Aug 2017]
          - Bug fix to calculation of anomalies to specific reference period in 
            read_gridded_data_GPCC, read_gridded_data_DaiPDSI and read_gridded_data_SPEI
            [M. Erb, N. Arizona U. & R. Tardif, U of Washington, Feb 2018]
          - Reference period w.r.t. which anomalies are computed are now passed as argument
            to functions tasked with uploading instrumental-era calibration datasets. 
            [R. Tardif, U. of Washington, February 2018]
"""
from netCDF4 import Dataset, date2num, num2date
from datetime import datetime, timedelta
from calendar import monthrange
from scipy import stats
from collections import OrderedDict
import numpy as np
import pylab as pl
import os.path
import string
import math

# -------------------
class Object(object):
    pass
# -------------------

def read_gridded_data_GISTEMP(data_dir,data_file,data_vars,outfreq,ref_period):
#==========================================================================================
#
# Reads the monthly data of surface air temperature anomalies from the GISTEMP gridded 
# product.
# 
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : List of variable names to read. (string list)
#                       Here should simply be ['Tsfc'], as only sfc temperature 
#                       data (anomalies) are contained in the file.
#
#      - outfreq      : string indicating whether to return monthly or annual averages
#
#      - ref_period   : two-element list indicating the period w.r.t. which anomalies
#                       are to be referenced. Set to None if no processing required,
#                       otherwise expects [start,end] ex. [1951,1980]
#
# Output: (numpy arrays)
#      - time_yrs     : Array with years over which data is available.
#                       dims: [nb_years]
#      - lat          : Array containing the latitudes of gridded  data. 
#                       dims: [lat]
#      - lon          : Array containing the longitudes of gridded  data. 
#                       dims: [lon]
#      - value        : Array with the annually-averaged data calculated from monthly data 
#                       dims: [time,lat,lon]
# 
#========================================================================================== 


    nbmaxnan = 0 # max nb of nan's allowed in calculation of annual average
    
    # Check if file exists
    infile = data_dir+'/GISTEMP/'+data_file
    if not os.path.isfile(infile):
        raise SystemExit(('Error in specification of gridded dataset. '
                          'File {} does not exist. Exiting.').format(infile))
        
    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        raise SystemExit('Too many variables to read! This file only contains'
                         ' surface air temperature (anomalies). Exiting.')
        
    dateref = datetime(1800,1,1,0)
    data = Dataset(infile,'r')

    lat   = data.variables['lat'][:]
    lon   = data.variables['lon'][:]
    
    indneg = np.where(lon < 0)[0]
    if len(indneg) > 0: # if non-empty
        lon[indneg] = 360.0 + lon[indneg]

    # ----------------------------------------------------------------------------------
    # Convert time from "nb of days from dateref" to actual date/time as datetime object 
    # ----------------------------------------------------------------------------------
    ntime = len(data.dimensions['time'])    
    daysfromdateref = data.variables['time'][:]
    dates = np.array([dateref + timedelta(days=int(i)) for i in daysfromdateref])
    
    fillval = np.power(2,15)-1
    value = np.copy(data.variables['tempanomaly'])
    value[value == fillval] = np.NAN

    if ref_period:
        climo_month = np.zeros([12, len(lat), len(lon)], dtype=float)
        # loop over months
        for i in range(12):
            m = i+1
            indsmref = [j for j,v in enumerate(dates) if v.year >= ref_period[0] and v.year <= ref_period[1] and v.month == m]
            indsm = [j for j,v in enumerate(dates) if v.month == m]
            climo_month[i] = np.nanmean(value[indsmref], axis=0)
            value[indsm] = (value[indsm] - climo_month[i])
    else:
        print('Warning: using default reference period defining temperature anomalies for GISTEMP product.')
    
    if outfreq == 'annual':
        # List years available in dataset and sort
        years = list(set([d.year for d in dates])) # 'set' is used to get unique values in list
        years.sort # sort the list
        dates_annual = np.array([datetime(y,1,1,0,0) for y in years])

        value_annual = np.empty([len(years), len(lat), len(lon)], dtype=float)
        value_annual[:] = np.nan # initialize with nan's
        
        # Loop over years in dataset
        for i in range(0,len(years)):        
            # find indices in time array where "years[i]" appear
            ind = [j for j, k in enumerate(dates) if k.year == years[i]]
            # ---------------------------------------
            # Calculate annual mean from monthly data
            # Note: data has dims [time,lat,lon]
            # ---------------------------------------
            tmp = np.nanmean(value[ind],axis=0)
            # apply check of max nb of nan values allowed
            nancount = np.isnan(value[ind]).sum(axis=0)
            tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
            value_annual[i,:,:] = tmp

        dates_ret = dates_annual
        value_ret = value_annual

    else:
        dates_ret = dates
        value_ret = value
        
    return dates_ret, lat, lon, value_ret

#==========================================================================================


def read_gridded_data_HadCRUT(data_dir,data_file,data_vars,outfreq,ref_period):
#==========================================================================================
#
# Reads the monthly data of surface air temperature anomalies from the HadCRUT gridded 
# product.
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : List of variable names to read. (string list)
#                       Here should simply be ['Tsfc'], as only sfc temperature 
#                       data (anomalies) are contained in the file.
#      - outfreq      : string indicating whether to return monthly or annual averages
#
#      - ref_period   : two-element list indicating the period w.r.t. which anomalies
#                       are to be referenced. Set to None if no processing required,
#                       otherwise expects [start,end] ex. [1951,1980]
#
# Output: (numpy arrays)
#      - time_yrs     : Array with years over which data is available.
#                       dims: [nb_years]
#      - lat          : Array containing the latitudes of gridded  data. 
#                       dims: [lat]
#      - lon          : Array containing the longitudes of gridded  data. 
#                       dims: [lon]
#      - value        : Array with the annually-averaged data calculated from monthly data 
#                       dims: [time,lat,lon]
# 
#========================================================================================== 

    nbmaxnan = 0 # max nb of nan's allowed in calculation of annual average

    # Check if file exists
    infile = data_dir+'/HadCRUT/'+data_file
    if not os.path.isfile(infile):
        raise SystemExit(('Error in specification of gridded dataset. '
                          'File {} does not exist. Exiting.').format(infile))

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        raise SystemExit('Too many variables to read! This file only contains'
                         ' surface air temperature (anomalies). Exiting.')

    dateref = datetime(1850,1,1,0)
    data = Dataset(infile,'r')

    lat   = data.variables['latitude'][:]
    lon   = data.variables['longitude'][:]

    indneg = np.where(lon < 0)[0]
    if len(indneg) > 0: # if non-empty
        lon[indneg] = 360.0 + lon[indneg]

    # -------------------------------------------------------------
    # Convert time from "nb of days from dateref" to absolute years 
    # -------------------------------------------------------------
    ntime = len(data.dimensions['time'])    
    daysfromdateref = data.variables['time'][:]
    dates = np.array([dateref + timedelta(days=int(i)) for i in daysfromdateref])

    value = np.copy(data.variables['temperature_anomaly'])
    value[value == -1e+30] = np.NAN

    if ref_period:
        climo_month = np.zeros([12, len(lat), len(lon)], dtype=float)
        # loop over months
        for i in range(12):
            m = i+1
            indsmref = [j for j,v in enumerate(dates) if v.year >= ref_period[0] and v.year <= ref_period[1] and v.month == m]
            indsm = [j for j,v in enumerate(dates) if v.month == m]
            climo_month[i] = np.nanmean(value[indsmref], axis=0)
            value[indsm] = (value[indsm] - climo_month[i])
    else:
        print('Warning: using default reference period defining temperature anomalies for HadCRUT product.')

    if outfreq == 'annual':
        # List years available in dataset and sort
        years = list(set([d.year for d in dates])) # 'set' is used to get unique values in list
        years.sort # sort the list
        dates_annual = np.array([datetime(y,1,1,0,0) for y in years])

        value_annual = np.empty([len(years), len(lat), len(lon)], dtype=float)
        value_annual[:] = np.nan # initialize with nan's

        # Loop over years in dataset
        for i in range(0,len(years)):        
            # find indices in time array where "years[i]" appear
            ind = [j for j, k in enumerate(dates) if k.year == years[i]]
            # ---------------------------------------
            # Calculate annual mean from monthly data
            # Note: data has dims [time,lat,lon]
            # ---------------------------------------
            tmp = np.nanmean(value[ind],axis=0)
            # apply check of max nb of nan values allowed
            nancount = np.isnan(value[ind]).sum(axis=0)
            tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
            value_annual[i,:,:] = tmp
            
        dates_ret = dates_annual
        value_ret = value_annual

    else:
        dates_ret = dates
        value_ret = value       

    return dates_ret, lat, lon, value_ret

#==========================================================================================


def read_gridded_data_BerkeleyEarth(data_dir,data_file,data_vars,outfreq,ref_period):
#==========================================================================================
#
# Reads the monthly data of surface air temperature anomalies from the BerkeleyEarth gridded 
# product.
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : List of variable names to read. (string list)
#                       Here should simply be ['Tsfc'], as only sfc temperature 
#                       data (anomalies) are contained in the file.
#
#      - outfreq      : string indicating whether to return monthly or annual averages
#
#      - ref_period   : two-element list indicating the period w.r.t. which anomalies
#                       are to be referenced. Set to None if no processing required,
#                       otherwise expects [start,end] ex. [1951,1980]
#
# Output: (numpy arrays)
#      - time_yrs     : Array with years over which data is available.
#                       dims: [nb_years]
#      - lat          : Array containing the latitudes of gridded  data. 
#                       dims: [lat]
#      - lon          : Array containing the longitudes of gridded  data. 
#                       dims: [lon]
#      - value        : Array with the annually-averaged data calculated from monthly data 
#                       dims: [time,lat,lon]
# 
#========================================================================================== 

    nbmaxnan = 0 # max nb of nan's allowed in calculation of annual average

    # Check if file exists
    infile = data_dir+'/BerkeleyEarth/'+data_file
    if not os.path.isfile(infile):
        raise SystemExit(('Error in specification of gridded dataset. '
                          'File {} does not exist. Exiting.').format(infile))

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        raise SystemExit('Too many variables to read! This file only contains'
                         ' surface air temperature (anomalies). Exiting.')

    data = Dataset(infile,'r')

    lat   = data.variables['latitude'][:]
    lon   = data.variables['longitude'][:]

    indneg = np.where(lon < 0)[0]
    if len(indneg) > 0: # if non-empty
        lon[indneg] = 360.0 + lon[indneg]

    # -------------------------------------------------------------
    # Time is in year A.D. (in decimal real number)
    # -------------------------------------------------------------
    ntime = len(data.dimensions['time'])

    time_yrs = []
    for i in range(0,len(data.variables['time'][:])):
        yrAD = data.variables['time'][i]
        year = int(yrAD)
        rem = yrAD - year
        base = datetime(year, 1, 1)
        time_yrs.append(base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem))

    dates = np.array(time_yrs)

    fillval = data.variables['temperature'].missing_value
    value = np.copy(data.variables['temperature'])    
    value[value == fillval] = np.NAN

    if ref_period:
        climo_month = np.zeros([12, len(lat), len(lon)], dtype=float)
        # loop over months
        for i in range(12):
            m = i+1
            indsmref = [j for j,v in enumerate(dates) if v.year >= ref_period[0] and v.year <= ref_period[1] and v.month == m]
            indsm = [j for j,v in enumerate(dates) if v.month == m]
            climo_month[i] = np.nanmean(value[indsmref], axis=0)
            value[indsm] = (value[indsm] - climo_month[i])
    else:
        print('Warning: using default reference period defining temperature anomalies for BEST product.')
    
    if outfreq == 'annual':
        # List years available in dataset and sort
        years = list(set([d.year for d in dates])) # 'set' is used to get unique values in list
        years.sort # sort the list
        dates_annual = np.array([datetime(y,1,1,0,0) for y in years])

        value_annual = np.empty([len(years), len(lat), len(lon)], dtype=float)
        value_annual[:] = np.nan # initialize with nan's

        # Loop over years in dataset
        for i in range(0,len(years)):        
            # find indices in time array where "years[i]" appear
            ind = [j for j, k in enumerate(dates) if k.year == years[i]]
            # ---------------------------------------
            # Calculate annual mean from monthly data
            # Note: data has dims [time,lat,lon]
            # ---------------------------------------
            tmp = np.nanmean(value[ind],axis=0)
            # apply check of max nb of nan values allowed
            nancount = np.isnan(value[ind]).sum(axis=0)
            tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
            value_annual[i,:,:] = tmp

        dates_ret = dates_annual
        value_ret = value_annual            

    else:
        dates_ret = dates
        value_ret = value
        
    return dates_ret, lat, lon, value_ret

#==========================================================================================

def read_gridded_data_MLOST(data_dir,data_file,data_vars,outfreq,ref_period):
#==========================================================================================
#
# Reads the monthly data of surface air temperature anomalies from the MLOST NOAA/NCDC 
# gridded product.
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : List of variable names to read. (string list)
#                       Here should simply be ['Tsfc'], as only sfc temperature 
#                       data (anomalies) are contained in the file.
#
#      - outfreq      : string indicating whether to return monthly or annual averages
#
#      - ref_period   : two-element list indicating the period w.r.t. which anomalies
#                       are to be referenced. Set to None if no processing required,
#                       otherwise expects [start,end] ex. [1951,1980]
#
# Output: (numpy arrays)
#      - time_yrs     : Array with years over which data is available.
#                       dims: [nb_years]
#      - lat          : Array containing the latitudes of gridded  data. 
#                       dims: [lat]
#      - lon          : Array containing the longitudes of gridded  data. 
#                       dims: [lon]
#      - value        : Array with the annually-averaged data calculated from monthly data 
#                       dims: [time,lat,lon]
# 
#========================================================================================== 

    nbmaxnan = 0 # max nb of nan's allowed in calculation of annual average

    # Check if file exists
    if 'MLOST' in data_file:
        infile = data_dir+'/MLOST/'+data_file
    elif 'NOAAGlobalTemp' in data_file:
        infile = data_dir+'/NOAAGlobalTemp/'+data_file
    else:
        print('In read_gridded_data_MLOST: error in specification of',
              ' datadirectory for this calibration dataset.')
        raise SystemExit()

    if not os.path.isfile(infile):
        raise SystemExit(('Error in specification of gridded dataset. '
                          'File {} does not exist. Exiting.').format(infile))

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        raise SystemExit('Too many variables to read! This file only contains'
                         ' surface air temperature (anomalies). Exiting.')

    data = Dataset(infile,'r')

    lat   = data.variables['lat'][:]
    lon   = data.variables['lon'][:]

    indneg = np.where(lon < 0)[0]
    if len(indneg) > 0: # if non-empty
        lon[indneg] = 360.0 + lon[indneg]

    # -----------------------------------------------------------------
    # Time is in "days since 1800-1-1 0:0:0":convert to calendar years
    # -----------------------------------------------------------------        
    dateref = datetime(1800,1,1,0)
    ntime = len(data.dimensions['time']) 
    daysfromdateref = data.variables['time'][:]
    dates = np.array([dateref + timedelta(days=int(i)) for i in daysfromdateref])

    fillval = data.variables['air'].missing_value
    value = np.copy(data.variables['air'])
    value[value == fillval] = np.NAN

    if ref_period:
        climo_month = np.zeros([12, len(lat), len(lon)], dtype=float)
        # loop over months
        for i in range(12):
            m = i+1
            indsmref = [j for j,v in enumerate(dates) if v.year >= ref_period[0] and v.year <= ref_period[1] and v.month == m]
            indsm = [j for j,v in enumerate(dates) if v.month == m]
            climo_month[i] = np.nanmean(value[indsmref], axis=0)
            value[indsm] = (value[indsm] - climo_month[i])
    else:
        print('Warning: using default reference period defining temperature anomalies for MLOST product.')
    
    if outfreq == 'annual':
        # List years available in dataset and sort
        years = list(set([d.year for d in dates])) # 'set' is used to get unique values in list
        years.sort # sort the list
        dates_annual = np.array([datetime(y,1,1,0,0) for y in years])

        value_annual = np.empty([len(years), len(lat), len(lon)], dtype=float)
        value_annual[:] = np.nan # initialize with nan's
        
        # Loop over years in dataset
        for i in range(0,len(years)):    
            # find indices in time array where "years[i]" appear
            ind = [j for j, k in enumerate(dates) if k.year == years[i]]
            # ---------------------------------------
            # Calculate annual mean from monthly data
            # Note: data has dims [time,lat,lon]
            # --------------------------------------- 
            tmp = np.nanmean(value[ind],axis=0)
            # apply check of max nb of nan values allowed
            nancount = np.isnan(value[ind]).sum(axis=0)
            tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
            value_annual[i,:,:] = tmp

        dates_ret = dates_annual
        value_ret = value_annual

    else:
        dates_ret = dates
        value_ret = value
        
    return dates_ret, lat, lon, value_ret      


#==========================================================================================

def read_gridded_data_GPCC(data_dir,data_file,data_vars,out_anomalies,ref_period,outfreq):
#==========================================================================================
#
# Reads the monthly data of surface air temperature anomalies from the GPCC 
# gridded product.
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : List of variable names to read. (string list)
#                       Here should simply be ['precip'], as only precipitation 
#                       data (anomalies) are contained in the file.
#
#      - out_anomalies: Boolean indicating whether anomalies w.r.t. a referenced period
#                       are to be csalculated and provided as output
#
#      - ref_period   : two-element list indicating the period w.r.t. which anomalies
#                       are to be referenced. Use [start,end] ex. [1951,1980]
#
#      - outfreq      : string indicating whether to return monthly or annual averages
#
# Output: (numpy arrays)
#      - time_yrs     : Array with years over which data is available.
#                       dims: [nb_years]
#      - lat          : Array containing the latitudes of gridded  data. 
#                       dims: [lat]
#      - lon          : Array containing the longitudes of gridded  data. 
#                       dims: [lon]
#      - value        : Array with the annually-averaged data calculated from monthly data 
#                       dims: [time,lat,lon]
# 
#========================================================================================== 

    nbmaxnan = 0 # max nb of nan's allowed in calculation of annual average

    # Check if file exists
    infile = data_dir+'/GPCC/'+data_file
    if not os.path.isfile(infile):
        raise SystemExit(('Error in specification of gridded dataset. '
                          'File {} does not exist. Exiting.').format(infile))

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        raise SystemExit('Too many variables to read! This file only contains'
                         ' precipitation accumulation or flux data. Exiting.')

    data = Dataset(infile,'r')

    lat   = data.variables['lat'][:]
    lon   = data.variables['lon'][:]

    indneg = np.where(lon < 0)[0]
    if len(indneg) > 0: # if non-empty
        lon[indneg] = 360.0 + lon[indneg]

    # -----------------------------------------------------------------
    # Time is in "days since 1800-1-1 0:0:0":convert to calendar years
    # -----------------------------------------------------------------        
    dateref = datetime(1800,1,1,0)
    ntime = len(data.dimensions['time']) 
    daysfromdateref = data.variables['time'][:]
    dates = np.array([dateref + timedelta(days=int(i)) for i in daysfromdateref])

    fillval = data.variables['precip'].missing_value
    value = np.copy(data.variables['precip'])
    value[value == fillval] = np.NAN

    # Calculate anomalies w.r.t. reference period, if out_anomalies is set to True in
    # class calibration_precip_GPCC() in LMR_calibrate.py
    if out_anomalies:
        if ref_period and type(ref_period) in [list,tuple] and len(ref_period) == 2:
            climo_month = np.zeros([12, len(lat), len(lon)], dtype=float)
            # loop over months
            for i in range(12):
                m = i+1
                indsmref = [j for j,v in enumerate(dates) if v.year >= ref_period[0] and v.year <= ref_period[1] and v.month == m]
                indsm = [j for j,v in enumerate(dates) if v.month == m]
                climo_month[i] = np.nanmean(value[indsmref], axis=0)
                value[indsm] = (value[indsm] - climo_month[i])
        else:
            raise SystemExit('In read_gridded_data_GPCC: out_anomalies is set to True,'
                             ' but a reference period is not properly defined. Exiting.')

    if outfreq == 'annual':
        # List years available in dataset and sort
        years = list(set([d.year for d in dates])) # 'set' is used to get unique values in list
        years.sort # sort the list
        dates_annual = np.array([datetime(y,1,1,0,0) for y in years])

        value_annual = np.empty([len(years), len(lat), len(lon)], dtype=float)
        value_annual[:] = np.nan # initialize with nan's
        
        # Loop over years in dataset
        for i in range(0,len(years)):        
            # find indices in time array where "years[i]" appear
            ind = [j for j, k in enumerate(dates) if k.year == years[i]]
            # ---------------------------------------
            # Calculate annual mean from monthly data
            # Note: data has dims [time,lat,lon]
            # ---------------------------------------
            tmp = np.nanmean(value[ind],axis=0)
            # apply check of max nb of nan values allowed
            nancount = np.isnan(value[ind]).sum(axis=0)
            tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
            value_annual[i,:,:] = tmp

        dates_ret = dates_annual
        value_ret = value_annual

    else:
        dates_ret = dates
        value_ret = value

    return dates_ret, lat, lon, value_ret

    
#==========================================================================================

def read_gridded_data_DaiPDSI(data_dir,data_file,data_vars,out_anomalies,ref_period,outfreq):
#==========================================================================================
#
# Reads the monthly data of Palmer Drought Severity Index (PDSI) anomalies from the
# "Dai" PDSI gridded product obtained from NOAA/ESRL at: 
# http://www.esrl.noaa.gov/psd/data/gridded/data.pdsi.html
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : List of variable names to read. (string list)
#                       Here should simply be ['pdsi'], as only PDSI
#                       data are contained in the file.
#
#      - out_anomalies: Boolean indicating whether anomalies w.r.t. a referenced period
#                       are to be calculated and provided as output
#
#      - ref_period   : two-element list indicating the period w.r.t. which anomalies
#                       are to be referenced. Use [start,end] ex. [1951,1980]
#
#      - outfreq      : string indicating whether to return monthly or annual averages
#
# Output: (numpy arrays)
#      - time_yrs     : Array with years over which data is available.
#                       dims: [nb_years]
#      - lat          : Array containing the latitudes of gridded  data. 
#                       dims: [lat]
#      - lon          : Array containing the longitudes of gridded  data. 
#                       dims: [lon]
#      - value        : Array with the annually-averaged data calculated from monthly data 
#                       dims: [time,lat,lon]
# 
#========================================================================================== 

    nbmaxnan = 0 # max nb of nan's allowed in calculation of annual average

    # Check if file exists
    infile = data_dir+'/DaiPDSI/'+data_file
    if not os.path.isfile(infile):
        raise SystemExit(('Error in specification of gridded dataset. '
                          'File {} does not exist. Exiting.').format(infile))

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        raise SystemExit('Too many variables to read! This file only contains'
                         ' Palmer Drought Severity Index (PDSI). Exiting.')

    data = Dataset(infile,'r')

    lat   = data.variables['lat'][:]
    lon   = data.variables['lon'][:]

    indneg = np.where(lon < 0)[0]
    if len(indneg) > 0: # if non-empty
        lon[indneg] = 360.0 + lon[indneg]

    # -----------------------------------------------------------------
    # Time is in "hours since 1800-1-1 0:0:0":convert to calendar years
    # -----------------------------------------------------------------        
    dateref = datetime(1800,1,1,0)
    ntime = len(data.dimensions['time'])    
    hoursfromdateref = data.variables['time'][:]
    dates = np.array([dateref + timedelta(hours=int(i)) for i in hoursfromdateref])

    fillval = data.variables['pdsi'].missing_value
    value = np.copy(data.variables['pdsi'])
    value[value == fillval] = np.NAN

    # Calculate anomalies w.r.t. reference period, if out_anomalies is set to True in class calibration_precip_DaiPDSI()
    # in LMR_calibrate.py
    if out_anomalies:
        if ref_period and type(ref_period) in [list,tuple] and len(ref_period) == 2:
            climo_month = np.zeros([12, len(lat), len(lon)], dtype=float)
            # loop over months
            for i in range(12):
                m = i+1
                indsmref = [j for j,v in enumerate(dates) if v.year >= ref_period[0] and v.year <= ref_period[1] and v.month == m]
                indsm = [j for j,v in enumerate(dates) if v.month == m]
                climo_month[i] = np.nanmean(value[indsmref], axis=0)
                value[indsm] = (value[indsm] - climo_month[i])
        else:
            raise SystemExit('In read_gridded_data_DaiPDSI: out_anomalies is set to True,'
                             ' but a reference period is not properly defined. Exiting.')

    if outfreq == 'annual':
        # List years available in dataset and sort
        years = list(set([d.year for d in dates])) # 'set' is used to get unique values in list
        years.sort # sort the list
        dates_annual = np.array([datetime(y,1,1,0,0) for y in years])

        value_annual = np.empty([len(years), len(lat), len(lon)], dtype=float)
        value_annual[:] = np.nan # initialize with nan's
        
        # Loop over years in dataset
        for i in range(0,len(years)):        
            # find indices in time array where "years[i]" appear
            ind = [j for j, k in enumerate(dates) if k.year == years[i]]
            # ---------------------------------------
            # Calculate annual mean from monthly data
            # Note: data has dims [time,lat,lon]
            # ---------------------------------------
            tmp = np.nanmean(value[ind],axis=0)
            # apply check of max nb of nan values allowed
            nancount = np.isnan(value[ind]).sum(axis=0)
            tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
            value_annual[i,:,:] = tmp
        
        dates_ret = dates_annual
        value_ret = value_annual

    else:
        dates_ret = dates
        value_ret = value

    return dates_ret, lat, lon, value_ret


#==========================================================================================

def read_gridded_data_SPEI(data_dir,data_file,data_vars,out_anomalies,ref_period,outfreq):
#==========================================================================================
#
# Reads the monthly data of Standardized Precipitation Evapotranspiration Index from
# Begueria S., Vicente-Serrano S., Reig F., Latorre B. (2014) Standardized precipitation
# evapotranspiration index (SPEI) revisited: Parameter fitting, evapotranspiration models,
# tools, datasets and drought monitoring. International Journal of Climatology 34, 3001-3023.
# SPEI gridded product obtained from the Consejo Superior de Investigaciones Cientificas
# (CSIC) at http://sac.csic.es/spei/index.html
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : List of variable names to read. (string list)
#                       Here should simply be ['spei'], as only SPEI
#                       data are contained in the file.
#
#      - out_anomalies: Boolean indicating whether anomalies w.r.t. a referenced period
#                       are to be calculated and provided as output
#
#      - ref_period   : two-element list indicating the period w.r.t. which anomalies
#                       are to be referenced. Use [start,end] ex. [1951,1980]
#
#      - outfreq      : string indicating whether to return monthly or annual averages
#
# Output: (numpy arrays)
#      - time_yrs     : Array with years over which data is available.
#                       dims: [nb_years]
#      - lat          : Array containing the latitudes of gridded  data. 
#                       dims: [lat]
#      - lon          : Array containing the longitudes of gridded  data. 
#                       dims: [lon]
#      - value        : Array with the annually-averaged data calculated from monthly data 
#                       dims: [time,lat,lon]
# 
#========================================================================================== 

    nbmaxnan = 0 # max nb of nan's allowed in calculation of annual average

    # Check if file exists
    infile = data_dir+'/SPEI/'+data_file
    if not os.path.isfile(infile):
        raise SystemExit(('Error in specification of gridded dataset. '
                          'File {} does not exist. Exiting.').format(infile))

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        raise SystemExit('Too many variables to read! This file only contains'
                         ' Standardized Precipitation Evapotranspiration Index (SPEI).'
                         ' Exiting.')

    data = Dataset(infile,'r')

    lat   = data.variables['lat'][:]
    lon   = data.variables['lon'][:]

    indneg = np.where(lon < 0)[0]
    if len(indneg) > 0: # if non-empty
        lon[indneg] = 360.0 + lon[indneg]

    # -----------------------------------------------------------------
    # Time is in "days since 1900-1-1 0:0:0":convert to calendar years
    # -----------------------------------------------------------------        
    dateref = datetime(1900,1,1,0)
    ntime = len(data.dimensions['time'])    
    daysfromdateref = data.variables['time'][:]
    dates = np.array([dateref + timedelta(days=int(i)) for i in daysfromdateref])

    fillval = data.variables['spei']._FillValue
    value = np.copy(data.variables['spei'])
    value[value == fillval] = np.NAN

    # Calculate anomalies w.r.t. reference period, if out_anomalies is set to True in class calibration_precip_DaiPDSI()
    # in LMR_calibrate.py
    if out_anomalies:
        if ref_period and type(ref_period) in [list,tuple] and len(ref_period) == 2:
            climo_month = np.zeros([12, len(lat), len(lon)], dtype=float)
            # loop over months
            for i in range(12):
                m = i+1
                indsmref = [j for j,v in enumerate(dates) if v.year >= ref_period[0] and v.year <= ref_period[1] and v.month == m]
                indsm = [j for j,v in enumerate(dates) if v.month == m]
                climo_month[i] = np.nanmean(value[indsmref], axis=0)
                value[indsm] = (value[indsm] - climo_month[i])
        else:
            raise SystemExit('In read_gridded_data_SPEI: out_anomalies is set to True,'
                             ' but a reference period is not properly defined. Exiting.')
    
    if outfreq == 'annual':
        # List years available in dataset and sort
        years = list(set([d.year for d in dates])) # 'set' is used to get unique values in list
        years.sort # sort the list
        dates_annual = np.array([datetime(y,1,1,0,0) for y in years])

        value_annual = np.empty([len(years), len(lat), len(lon)], dtype=float)
        value_annual[:] = np.nan # initialize with nan's
        
        # Loop over years in dataset
        for i in range(0,len(years)):        
            # find indices in time array where "years[i]" appear
            ind = [j for j, k in enumerate(dates) if k.year == years[i]]
            # ---------------------------------------
            # Calculate annual mean from monthly data
            # Note: data has dims [time,lat,lon]
            # ---------------------------------------
            tmp = np.nanmean(value[ind],axis=0)
            # apply check of max nb of nan values allowed
            nancount = np.isnan(value[ind]).sum(axis=0)
            tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
            value_annual[i,:,:] = tmp
        
        dates_ret = dates_annual
        value_ret = value_annual

    else:
        dates_ret = dates
        value_ret = value

    return dates_ret, lat, lon, value_ret

anom_ref=None
#==========================================================================================

def read_gridded_data_CMIP5_model(data_dir,data_file,data_vars,outtimeavg,
                                  detrend=None,anom_ref=None,var_info=None):
#==========================================================================================
#
# Reads the monthly data from a CMIP5 model and return yearly averaged values
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : Variables names to be read, and info on whether each
#                       variable is to be returned as anomalies of as full field
#                       (dict)
#
#      - outtimeavg   : Dictionary indicating the type of averaging (key) and associated
#                       information on averaging period (integer list)
#                       if the type is "annual": list of integers indicating the months of 
#                                                the year over which to average the data.
#                                                Requires availability of monthly data.
#                       if type is "multiyear" : list of single integer indicating the length
#                                                of averaging period (in years).
#                                                Requires availability of data with a
#                                                resolution of at least the averaging
#                                                interval.
#
#                       ex 1: outtimeavg = {'annual': [1,2,3,4,5,6,7,8,9,10,11,12]}
#                       ex 2: outtimeavg = {'multiyear': [100]} -> 100yr average
#
#                       *or*:
#                       Integer list or tuple of integer lists indicating the sequence of
#                       months over which to average the data.
#
#      - detrend      : Boolean to indicate if detrending is to be applied to the prior
#
#      - anom_ref     : Reference period (in years CE) used in calculating anomalies (tuple)
#
#      - var_info     : Dict. containing information about whether some state variables
#                       represent temperature or moisture (used to extract proper 
#                       seasonally-avg. data to be used in calculation of proxy estimates) 
#
# Output: 
#      - datadict     : Master dictionary containing dictionaries, one for each state 
#                       variable, themselves containing the following numpy arrays:
#                       - time_yrs  : Array with years over which data is available.
#                                     dims: [nb_years]
#                       - lat       : Array containing the latitudes of gridded  data. 
#                                     dims: [lat]
#                       - lon       : Array containing the longitudes of gridded  data. 
#                                     dims: [lon]
#                       - value     : Array with the averaged data calculated from 
#                                     monthly data dims: [time,lat,lon]
# 
#  ex. data access : datadict['tas_sfc_Amon']['years'] => array containing years of the 
#                                                         'tas' data
#                    datadict['tas_sfc_Amon']['lat']   => array of lats for 'tas' data
#                    datadict['tas_sfc_Amon']['lon']   => array of lons for 'tas' data
#                    datadict['tas_sfc_Amon']['value'] => array of 'tas' data values
#
#========================================================================================== 

    datadict = {}

    # Loop over state variables to load
    for v in range(len(data_vars)):
        vardef = list(data_vars.keys())[v]
        data_file_read = data_file.replace('[vardef_template]', vardef)
        
        # Check if file exists
        infile = data_dir + '/' + data_file_read
        if not os.path.isfile(infile):
            print('Error in specification of gridded dataset')
            print('File ', infile, ' does not exist! - Exiting ...')
            raise SystemExit()
        else:
            print('Reading file: ', infile)

        # Get file content
        data = Dataset(infile,'r')

        # Dimensions used to store the data
        nc_dims = [dim for dim in data.dimensions]
        dictdims = {}
        for dim in nc_dims:
            dictdims[dim] = len(data.dimensions[dim])

        # Define the name of the variable to extract from the variable definition (from namelist)
        var_to_extract = vardef.split('_')[0]

        # Query its dimensions
        vardims = data.variables[var_to_extract].dimensions
        nbdims  = len(vardims)
        # names of variable dims
        vardimnames = []
        for d in vardims:
            vardimnames.append(d)
        
        # put everything in lower case for homogeneity
        vardimnames = [item.lower() for item in vardimnames]

        # extract info on variable units
        if hasattr(data.variables[var_to_extract], 'units'):
            units = data.variables[var_to_extract].units
        else:
            units = None
        
        # One of the dims has to be time! 
        if 'time' not in vardimnames:
            print('Variable does not have *time* as a dimension! Exiting!')
            raise SystemExit()
        else:
            # read in the time netCDF4.Variable
            time = data.variables['time']

        # Transform into calendar dates using netCDF4 variable attributes (units & calendar)
        # TODO: may not want to depend on netcdf4.num2date...
        try:
            if hasattr(time, 'calendar'):
                # if time is defined as "months since":not handled by datetime functions
                if 'months since' in time.units:
                    new_time = np.zeros(time.shape)
                    nmonths, = time.shape
                    basedate = time.units.split('since')[1].lstrip()
                    new_time_units = "days since "+basedate        
                    start_date = pl.datestr2num(basedate)        
                    act_date = start_date*1.0
                    new_time[0] = act_date
                    for i in range(int(nmonths)): #increment months
                        d = pl.num2date(act_date)
                        ndays = monthrange(d.year,d.month)[1] #number of days in current month
                        act_date += ndays
                        new_time[i] = act_date

                    time_yrs = num2date(new_time[:],units=new_time_units,calendar=time.calendar)
                else:                    
                    time_yrs = num2date(time[:],units=time.units,
                                    calendar=time.calendar)
            else:
                time_yrs = num2date(time[:],units=time.units)
            time_yrs_list = time_yrs.tolist()
        except ValueError:
            # num2date needs calendar year start >= 0001 C.E. (bug submitted
            # to unidata about this
            fmt = '%Y-%d-%m %H:%M:%S'
            tunits = time.units
            since_yr_idx = tunits.index('since ') + 6
            year = int(tunits[since_yr_idx:since_yr_idx+4])
            year_diff = year - 1
            new_start_date = datetime(1, 1, 1, 0, 0, 0)

            new_units = tunits[:since_yr_idx] + '0001-01-01 00:00:00'
            if hasattr(time, 'calendar'):
                time_yrs = num2date(time[:], new_units, calendar=time.calendar)
            else:
                time_yrs = num2date(time[:], new_units)

            time_yrs_list = [datetime(d.year + year_diff, d.month, d.day,
                                      d.hour, d.minute, d.second)
                             for d in time_yrs]


        # Query info on spatial coordinates ...
        # get rid of time in list in vardimnames
        varspacecoordnames = [item for item in vardimnames if item != 'time'] 
        nbspacecoords = len(varspacecoordnames)

        if nbspacecoords == 0: # data => simple time series
            vartype = '0D:time series'
            spacecoords = None
        elif nbspacecoords == 1: # data => 1D data
            if 'lat' in varspacecoordnames or 'latitude' in varspacecoordnames:
                # latitudinally-averaged  variable
                vartype = '1D:meridional' 
                spacecoords = ('lat',)
                if 'lat' in varspacecoordnames:
                    spacevar1 = data.variables['lat'][:]
                elif 'latitude' in varspacecoordnames:
                    spacevar1 = data.variables['latitude'][:]
                
        elif ((nbspacecoords == 2) or (nbspacecoords == 3 and 'plev' in vardimnames and dictdims['plev'] == 1)): # data => 2D data
            # get rid of plev in list        
            varspacecoordnames = [item for item in varspacecoordnames if item != 'plev'] 
            spacecoords = (varspacecoordnames[0],varspacecoordnames[1])
            spacevar1 = data.variables[spacecoords[0]][:]
            spacevar2 = data.variables[spacecoords[1]][:]
            
            # Allow for use of 'latitude'/'longitude' as definitions of spatial coords.
            # in input file, but use the harmonized 'lat'/'lon' for remainder of process
            # after the input from file. 
            if 'latitude' in spacecoords or 'longitude' in spacecoords:
                tmplst = list(spacecoords) # to list for modification
                if 'latitude' in tmplst:
                    indc = tmplst.index('latitude')
                    tmplst[indc] = 'lat'
                if 'longitude' in tmplst:
                    indc = tmplst.index('longitude')
                    tmplst[indc] = 'lon'
                spacecoords = tuple(tmplst) # back to tuple
                
            if 'lat' in spacecoords and 'lon' in spacecoords:
                vartype = '2D:horizontal'
            elif 'lat' in spacecoords and 'lev' in spacecoords:
                vartype = '2D:meridional_vertical'
            else:
                print('Cannot handle this variable yet! 2D variable of unrecognized dimensions... Exiting!')
                raise SystemExit()
        else:
            print('Cannot handle this variable yet! Too many dimensions... Exiting!')
            raise SystemExit()
        
        
        # -----------------
        # Upload data array
        # -----------------
        data_var = data.variables[var_to_extract][:]

        data_var_shape = data_var.shape
        if vartype == '2D:horizontal' and len(data_var_shape) > 3: data_var = np.squeeze(data_var)
        print(data_var.shape)

        ntime = len(data.dimensions['time'])
        dates = time_yrs

        
        # if 2D:horizontal variable, check grid & standardize grid orientation to lat=>[-90,90] & lon=>[0,360] if needed
        if vartype == '2D:horizontal':

            vardims = data_var.shape
            
            # which dim is lat & which is lon?
            indlat = spacecoords.index('lat')
            indlon = spacecoords.index('lon')
            print('indlat=', indlat, ' indlon=', indlon)
            
            if indlon == 0:
                vlon = spacevar1
                vlat = spacevar2
                nlat = vardims[2]
                nlon = vardims[1]
            elif indlon == 1:
                vlon = spacevar2
                vlat = spacevar1
                nlat = vardims[1]
                nlon = vardims[2]

            # are coordinates defined as 1d or 2d array?
            spacevardims = len(vlat.shape)
            if spacevardims == 1:
                varlat = np.array([vlat,]*nlon).transpose()
                varlon = np.array([vlon,]*nlat)
            else:
                varlat = vlat
                varlon = vlon

            varlatdim = len(varlat.shape)
            varlondim = len(varlon.shape)

            # -----------------------------------------------------
            # Check if latitudes are defined in the [-90,90] domain
            # and if longitudes are in the [0,360] domain
            # -----------------------------------------------------
            # Latitudes:
            fliplat = None

            # check for monotonically increasing or decreasing values
            monotone_increase = np.all(np.diff(varlat[:,0]) > 0)
            monotone_decrease = np.all(np.diff(varlat[:,0]) < 0)
            if not monotone_increase and not monotone_decrease:
                # funky grid
                fliplat = False
                
            if fliplat is None:
                if varlatdim == 2: # 2D lat array
                    if varlat[0,0] > varlat[-1,0]: # lat not as [-90,90] => array upside-down
                        fliplat = True
                    else:
                        fliplat = False
                elif varlatdim == 1: # 1D lat array
                    if varlat[0] > varlat[-1]: # lat not as [-90,90] => array upside-down
                        fliplat = True
                    else:
                        fliplat = False
                else:
                    print('ERROR!')
                    raise SystemExit(1)

            if fliplat:
                varlat = np.flipud(varlat)
                # flip data variable
                if indlat == 0:
                    tmp = data_var[:,::-1,:]
                else:
                    tmp = data_var[:,:,::-1]
                data_var = tmp

            # Longitudes:
            # Transform longitudes from [-180,180] domain to [0,360] domain if needed
            indneg = np.where(varlon < 0)
            if len(indneg) > 0: # if non-empty
                varlon[indneg] = 360.0 + varlon[indneg]

            # Back into right arrays
            if indlon == 0:
                spacevar1 = varlon
                spacevar2 = varlat
            elif indlon == 1:
                spacevar2 = varlon
                spacevar1 = varlat

        # if 2D:meridional_vertical variable
        elif vartype == '2D:meridional_vertical':

            vardims = data_var.shape


            print('::vardims=', vardims)
            
            # which dim is lat and which is lev?
            indlat = spacecoords.index('lat')
            indlev = spacecoords.index('lev')
            print('indlat=', indlat, ' inlev=', indlev)

            if indlev == 0:
                vlev = spacevar1
                vlat = spacevar2
                nlat = vardims[2]
                nlev = vardims[1]
            elif indlev == 1:
                vlev = spacevar2
                vlat = spacevar1
                nlat = vardims[1]
                nlev = vardims[2]

            # are coordinates defined as 1d or 2d array?
            spacevardims = len(vlat.shape)
            
            if spacevardims == 1:
                if indlev == 0:
                    varlev = np.array([vlev,]*nlat).transpose()
                    varlat = np.array([vlat,]*nlev)
                    
                else:
                    varlat = np.array([vlat,]*nlev).transpose()
                    varlev = np.array([vlev,]*nlat)

            else:
                varlat = vlat
                varlev = vlev

            varlatdim = len(varlat.shape)
            varlevdim = len(varlev.shape)

            
            # Check if latitudes are defined in the [-90,90] domain
            fliplat = None

            # check for monotonically increasing or decreasing values
            monotone_increase = np.all(np.diff(varlat[:,0]) > 0)
            monotone_decrease = np.all(np.diff(varlat[:,0]) < 0)
            if not monotone_increase and not monotone_decrease:
                # funky grid
                fliplat = False
                
            if fliplat is None:
                if varlatdim == 2: # 2D lat array
                    if indlat == 0:
                        if varlat[0,0] > varlat[-1,0]: # lat not as [-90,90] => array upside-down
                            fliplat = True
                        else:
                            fliplat = False
                    else:
                        if varlat[0,0] > varlat[0,-1]: # lat not as [-90,90] => array upside-down
                            fliplat = True
                        else:
                            fliplat = False
                elif varlatdim == 1: # 1D lat array
                    if varlat[0] > varlat[-1]: # lat not as [-90,90] => array upside-down
                        fliplat = True
                    else:
                        fliplat = False
                else:
                    print('ERROR!')
                    raise SystemExit(1)

            if fliplat:
                varlat = np.flipud(varlat)
                # flip data variable
                if indlat == 0:
                    tmp = data_var[:,::-1,:]
                else:
                    tmp = data_var[:,:,::-1]
                data_var = tmp

            # Back into right arrays
            if indlev == 0:
                spacevar1 = varlev
                spacevar2 = varlat
            elif indlev == 1:
                spacevar2 = varlev
                spacevar1 = varlat

        
        # if 1D:meridional (latitudinally-averaged) variable
        elif vartype == '1D:meridional':

            vardims = data_var.shape
            
            # which dim is lat?
            indlat = spacecoords.index('lat')
            print('indlat=', indlat)
            
            # Check if latitudes are defined in the [-90,90] domain
            fliplat = None
            # check for monotonically increasing or decreasing values
            monotone_increase = np.all(np.diff(spacevar1) > 0)
            monotone_decrease = np.all(np.diff(spacevar1) < 0)
            if not monotone_increase and not monotone_decrease:
                # funky grid
                fliplat = False

            if fliplat is None:
                if spacevar1[0] > spacevar1[-1]: # lat not as [-90,90] => array upside-down
                    fliplat = True

            if fliplat:
                spacevar1 = np.flipud(spacevar1)
                tmp = data_var[:,::-1]
                data_var = tmp
            

        # ====== other data processing ======
            
        # Calculate anomalies?
        kind = data_vars[vardef]
        
        # monthly climatology
        if vartype == '0D:time series':
            climo_month = np.zeros((12))
        elif vartype == '1D:meridional':
            climo_month = np.zeros([12, vardims[1]], dtype=float)
        elif '2D' in vartype:
            climo_month = np.zeros([12, vardims[1], vardims[2]], dtype=float)
        
        if not kind or kind == 'anom':
            print('Anomalies provided as the prior: Removing the temporal mean (for every gridpoint)...')
            
            # prior data overlap with anomaly reference period?
            # if not, take anomalies w.r.t. to mean over entire length of the data
            overlap = False
            if anom_ref:
                indsmref = [j for j,v in enumerate(dates) if ((v.year >= anom_ref[0])
                                                              and (v.year <= anom_ref[1]))]
                if len(indsmref) > 0: # overlap exists
                    overlap = True
            # loop over months
            for i in range(12):
                m = i+1
                indsm = [j for j,v in enumerate(dates) if v.month == m]
                if overlap:
                    indsmref = [j for j,v in enumerate(dates) if ((v.year >= anom_ref[0])
                                                                  and (v.year <= anom_ref[1])
                                                                  and (v.month == m))]
                else:
                    indsmref = indsm

                climo_month[i] = np.nanmean(data_var[indsmref], axis=0)
                data_var[indsm] = (data_var[indsm] - climo_month[i])
                
        elif kind == 'full':
            print('Full field provided as the prior')
            # do nothing else...
        else:
            print('ERROR in the specification of type of prior. Should be "full" or "anom"! Exiting...')
            raise SystemExit()

        print(var_to_extract, ': Global(monthly): mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var))

        # Possibly detrend the prior
        if detrend:
            print('Detrending the prior for variable: '+var_to_extract)

            data_var_copy = np.copy(data_var)

            if vartype == '0D:time series':
                xdim = data_var.shape[0]
                xvar = list(range(xdim))
                slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy)
                trend = slope*np.squeeze(xvar) + intercept
                data_var = data_var_copy - trend
            elif vartype == '1D:meridional':
                [xdim,dim1] = data_var.shape
                xvar = list(range(xdim))
                # loop over grid points
                for i in range(dim1):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy[:,i])
                    if np.isfinite(slope) and np.isfinite(intercept):
                        trend = slope*np.squeeze(xvar) + intercept
                        data_var[:,i] = data_var_copy[:,i] - trend
                    else:
                        data_var[:,i] = np.nan
            elif '2D' in vartype: 
                [xdim,dim1,dim2] = data_var.shape
                xvar = list(range(xdim))
                # loop over grid points
                for i in range(dim1):
                    for j in range(dim2):
                        slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy[:,i,j])
                        if np.isfinite(slope) and np.isfinite(intercept):
                            trend = slope*np.squeeze(xvar) + intercept
                            data_var[:,i,j] = data_var_copy[:,i,j] - trend
                        else:
                            data_var[:,i,j] = np.nan
                        
            print(var_to_extract, ': Global(monthly/detrend): mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var))


        # ----------------------------------------------------------------
        # Average monthly data over the monthly sequence in outtimeavg.
        # Note: Means one output data per year, but averaged over specific 
        #       sequence of months.
        # ----------------------------------------------------------------

        # for compatibility with new definition possibly using a dict.
        if type(outtimeavg) is dict:
            outtimeavg_dict = outtimeavg
            # check key - there should be only one...
            outtimeavg_key = list(outtimeavg_dict.keys())[0]
            # here, it should be 'annual'. No other definition allowed.
            if outtimeavg_key == 'annual':
                outtimeavg_val = outtimeavg_dict['annual']
            else:
                # Set to calendar year first, to perform averaging over
                # annual cycle before averaging over multiple years
                outtimeavg_val = list(range(1,13))
        else:
            # not a dict, must be a list or tuple of lists providing
            # sequence(s) of months over which to average
            outtimeavg_key = 'annual'
            outtimeavg_val = outtimeavg
            
        # outtimeavg_val is a tuple, or a list?
        if type(outtimeavg_val) is tuple:
            # Is var_info defined? 
            if var_info:
                # assign appropriate seasonality whether variable represents temperature *or* moisture
                if vardef in var_info['temperature']:
                    outtimeavg_var =  outtimeavg_val[0]
                elif vardef in var_info['moisture']:
                    outtimeavg_var =  outtimeavg_val[1]
                else:
                    # variable not representing temperature or moisture
                    print('ERROR: outtimeavg is a tuple but variable is not' \
                       ' temperature nor moisture...')
                    raise SystemExit()
            else:
                print('ERROR: var_info undefined. outtimeavg is a tuple and info' \
                       ' contained in this dict. is required to assign proper' \
                       ' seasonality to temperature and moisture variables')
                raise SystemExit()
        elif type(outtimeavg_val) is list:
            outtimeavg_var =  outtimeavg_val
        else:
            print('ERROR: outtimeavg has to be a list or a tuple of lists, but is:', outtimeavg)
            raise SystemExit()


        print('Averaging over month sequence:', outtimeavg_var)
        
        year_current = [m for m in outtimeavg_var if m>0 and m<=12]
        year_before  = [abs(m) for m in outtimeavg_var if m < 0]        
        year_follow  = [m-12 for m in outtimeavg_var if m > 12]
        
        avgmonths = year_before + year_current + year_follow
        indsclimo = sorted([item-1 for item in avgmonths])
        
        # List years available in dataset and sort
        years_all = [d.year for d in time_yrs_list]
        years     = list(set(years_all)) # 'set' used to retain unique values in list
        years.sort() # sort the list
        ntime = len(years)
        datesYears = np.array([datetime(y,1,1,0,0) for y in years])
        
        if vartype == '0D:time series':
            value = np.zeros([ntime], dtype=float) 
        elif vartype == '1D:meridional':
            value = np.zeros([ntime, vardims[1]], dtype=float)
        elif '2D' in vartype:
            value = np.zeros([ntime, vardims[1], vardims[2]], dtype=float)


        # Loop over years in dataset (less memory intensive...otherwise need to deal with large arrays) 
        for i in range(ntime):
            tindsyr   = [k for k,d in enumerate(dates) if d.year == years[i]    and d.month in year_current]
            tindsyrm1 = [k for k,d in enumerate(dates) if d.year == years[i]-1. and d.month in year_before]
            tindsyrp1 = [k for k,d in enumerate(dates) if d.year == years[i]+1. and d.month in year_follow]
            indsyr = tindsyrm1+tindsyr+tindsyrp1

            if vartype == '0D:time series':
                value[i] = np.nanmean(data_var[indsyr],axis=0)
            elif vartype == '1D:meridional':
                value[i,:] = np.nanmean(data_var[indsyr],axis=0)
            elif '2D' in vartype: 
                if nbdims > 3:
                    value[i,:,:] = np.nanmean(np.squeeze(data_var[indsyr]),axis=0)
                else:
                    value[i,:,:] = np.nanmean(data_var[indsyr],axis=0)

        
        print(var_to_extract, ': Global(time-averaged): mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value))

        climo = np.mean(climo_month[indsclimo], axis=0)
        

        # Returning multiyear averages if option enabled
        if outtimeavg_key == 'multiyear':
            # multiyear: Averaging available data over a time interval 
            # corresponding to the specified number of years.
            print(outtimeavg_dict)
            print('Averaging period (years): ', outtimeavg_dict['multiyear'][0])

            avg_period = float(outtimeavg_dict['multiyear'][0])
            time_resolution = 1.
            
            # How many averaged data can be calculated w/ the dataset?
            nbpts = int(avg_period/time_resolution)
            years_range = years[-1] - years[0]
            nbintervals = int(math.modf(years_range/avg_period)[1])

            years_avg = np.zeros([nbintervals],dtype=int)
            if vartype == '0D:time series':
                value_avg = np.zeros([nbintervals], dtype=float)
            elif vartype == '1D:meridional':
                value_avg = np.zeros([nbintervals, vardims[1]], dtype=float)
            elif '2D' in vartype: 
                value_avg = np.zeros([nbintervals, vardims[1], vardims[2]], dtype=float)
            # really initialize with missing values (NaNs)
            value_avg[:] = np.nan

            for i in range(nbintervals):
                edgel = i*nbpts
                edger = edgel+nbpts
                years_avg[i] = int(round(np.mean(years[edgel:edger])))
                value_avg[i] = np.nanmean(value[edgel:edger], axis=0)

            # into the returned arrays
            value = value_avg
            datesYears = np.array([datetime(y,1,1,0,0) for y in years_avg])
            

        # Dictionary of dictionaries
        # ex. data access : datadict['tas_sfc_Amon']['years'] => arrays containing years of the 'tas' data
        #                   datadict['tas_sfc_Amon']['value'] => array of 'tas' data values etc ...
        d = {}
        d['vartype'] = vartype
        d['units']   = units
        d['years']   = datesYears
        d['value']   = value
        d['climo']   = climo
        d['spacecoords'] = spacecoords

        if vartype == '1D:meridional':
            d[spacecoords[0]] = spacevar1
        elif '2D' in vartype:
            d[spacecoords[0]] = spacevar1
            d[spacecoords[1]] = spacevar2

        datadict[vardef] = d


    return datadict


#==========================================================================================

def read_gridded_data_CMIP5_model_ensemble(data_dir,data_file,data_vars):
#==========================================================================================
#
# Reads the monthly data from a CMIP5 model *ensemble* and return yearly averaged values
# for all ensemble members. 
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : List of variable names to read. (string list)
#
# Output: 
#      - datadict     : Master dictionary containing dictionaries, one for each state 
#                       variable, themselves containing the following numpy arrays:
#                       - time_yrs  : Array with years over which data is available.
#                                     dims: [nb_years]
#                       - lat       : Array containing the latitudes of gridded  data. 
#                                     dims: [lat]
#                       - lon       : Array containing the longitudes of gridded  data. 
#                                     dims: [lon]
#                       - value     : Array with the annually-averaged data calculated from 
#                                     monthly data dims: [time*members,lat,lon]
#                                     where members is the number of members in the original
#                                     ensemble
# 
#  ex. data access : datadict['tas_sfc_Amon']['years'] => array containing years of the 
#                                                         'tas' data
#                    datadict['tas_sfc_Amon']['lat']   => array of lats for 'tas' data
#                    datadict['tas_sfc_Amon']['lon']   => array of lons for 'tas' data
#                    datadict['tas_sfc_Amon']['value'] => array of 'tas' data values
#
#========================================================================================== 

    datadict = {}

    # Loop over state variables to load
    for v in range(len(data_vars)):
        vardef = data_vars[v]
        data_file_read = data_file.replace('[vardef_template]',vardef)
        
        # Check if file exists
        infile = data_dir + '/' + data_file_read
        if not os.path.isfile(infile):
            print('Error in specification of gridded dataset')
            print('File ', infile, ' does not exist! - Exiting ...')
            exit(1)
        else:
            print('Reading file: ', infile)

        # Load entire dataset from file
        data = Dataset(infile,'r')

        # Dimensions used to store the data
        nc_dims = [dim for dim in data.dimensions]
        dictdims = {}
        for dim in nc_dims:
            dictdims[dim] = len(data.dimensions[dim])

        # Define the name of the variable to extract from the variable definition (from namelist)
        var_to_extract = vardef.split('_')[0]

        # Query its dimensions
        vardims = data.variables[var_to_extract].dimensions
        nbdims  = len(vardims)
        # names of variable dims
        vardimnames = []
        for d in vardims:
            vardimnames.append(d)
        
        # put everything in lower case for homogeneity
        vardimnames = [item.lower() for item in vardimnames]

        # One of the dims has to be time! 
        if 'time' not in vardimnames:
            print('Variable does not have *time* as a dimension! Exiting!')
            exit(1)
        else:
            # read in the time netCDF4.Variable
            time = data.variables['time']

        # Transform into calendar dates using netCDF4 variable attributes (units & calendar)
        # TODO: may not want to depend on netcdf4.num2date...
        try:
            time_yrs = num2date(time[:],units=time.units,calendar=time.calendar)
            time_yrs_list = time_yrs.tolist()
        except ValueError:
            # num2date needs calendar year start >= 0001 C.E. (bug submitted
            # to unidata about this
            fmt = '%Y-%d-%m %H:%M:%S'
            tunits = time.units
            since_yr_idx = tunits.index('since ') + 6
            year = int(tunits[since_yr_idx:since_yr_idx+4])
            year_diff = year - 1
            new_start_date = datetime(1, 1, 1, 0, 0, 0)

            new_units = tunits[:since_yr_idx] + '0001-01-01 00:00:00'
            time_yrs = num2date(time[:], new_units, calendar=time.calendar)
            time_yrs_list = [datetime(d.year + year_diff, d.month, d.day,
                                      d.hour, d.minute, d.second)
                             for d in time_yrs]

        # To convert monthly data to annual: List years available in dataset and sort
        years_all = [d.year for d in time_yrs_list]
        years     = list(set(years_all)) # 'set' is used to retain unique values in list
        years.sort() # sort the list
        time_yrs  = np.empty(len(years), dtype=int)
        
        # Query about ensemble members
        if 'member' in vardimnames:
            indmem = vardimnames.index('member')
            memberdims = data.variables['member'][:]
            nbmems = len(memberdims)
        else:
            nbmems = 1
        print('nbmems=', nbmems, 'indmem=', indmem)

        # Query info on spatial coordinates ...
        # get rid of "time" and "member" in list        
        varspacecoordnames = [item for item in vardimnames if item != 'time' and item != 'member'] 
        nbspacecoords = len(varspacecoordnames)

        if nbspacecoords == 0: # data => simple time series
            vartype = '0D:time series'
            value = np.empty([len(years),nbmems], dtype=float)            
            spacecoords = None
        elif ((nbspacecoords == 2) or (nbspacecoords == 3 and 'plev' in vardimnames and dictdims['plev'] == 1)): # data => 2D data
            # get rid of plev in list        
            varspacecoordnames = [item for item in varspacecoordnames if item != 'plev'] 
            spacecoords = (varspacecoordnames[0],varspacecoordnames[1])
            spacevar1 = data.variables[spacecoords[0]][:]
            spacevar2 = data.variables[spacecoords[1]][:]
            value = np.empty([len(years), nbmems, len(spacevar1), len(spacevar2)], dtype=float)

            if 'lat' in spacecoords and 'lon' in spacecoords:
                vartype = '2D:horizontal'
            elif 'lat' in spacecoords and 'lev' in spacecoords:
                vartype = '2D:meridional_vertical'
            else:
                print('Cannot handle this variable yet! 2D variable of unrecognized dimensions... Exiting!')
                exit(1)
        else:
            print('Cannot handle this variable yet! To many dimensions... Exiting!')
            exit(1)

        # data array
        data = data.variables[var_to_extract][:]
        print(data.shape)

        # if 2D:horizontal variable, check grid & standardize grid orientation to lat=>[-90,90] & lon=>[0,360] if needed
        if vartype == '2D:horizontal':
            # which dim is lat & which is lon?
            indlat = spacecoords.index('lat')
            indlon = spacecoords.index('lon')
            print('indlat=', indlat, ' indlon=', indlon)

            if indlon == 0:
                varlon = spacevar1
                varlat = spacevar2
            elif indlon == 1:
                varlon = spacevar2
                varlat = spacevar1

            # Transform latitudes to [-90,90] domain if needed
            if varlat[0] > varlat[-1]: # not as [-90,90] => array upside-down
                # flip coord variable
                varlat = np.flipud(varlat)

                # flip data variable
                if indlat == 0:
                    tmp = data[:,:,::-1,:]             
                else:
                    tmp = data[:,:,:,::-1] 
                data = tmp

            # Transform longitudes from [-180,180] domain to [0,360] domain if needed
            indneg = np.where(varlon < 0)[0]
            if len(indneg) > 0: # if non-empty
                varlon[indneg] = 360.0 + varlon[indneg]

            # Back into right arrays
            if indlon == 0:
                spacevar1 = varlon
                spacevar2 = varlat
            elif indlon == 1:
                spacevar2 = varlon
                spacevar1 = varlat
        

        # if 2D:meridional_vertical variable,
        # TO DO ...


            
        # Loop over years in dataset
        for i in range(0,len(years)): 
            # find indices in time array where "years[i]" appear
            ind = [j for j, k in enumerate(years_all) if k == years[i]]
            time_yrs[i] = years[i]

            # -----------------------------------------
            # Calculate annual mean from monthly data
            # Note: assume data has dims [time,lat,lon]
            # -----------------------------------------
            if vartype == '0D:time series':
                value[i,:] = np.nanmean(data[ind],axis=0)
            elif '2D' in vartype: 
                if nbdims > 3:
                    value[i,:,:,:] = np.nanmean(np.squeeze(data[ind]),axis=0)
                else:
                    value[i,:,:,:] = np.nanmean(data[ind],axis=0)


        # Model data, so need to standardize (i.e. calculate anomalies)
        #print 'Standardizing the prior...'
        #print 'mean=', np.nanmean(value), ' std-dev=', np.nanstd(value)
        #value = (value - np.nanmean(value))/np.nanstd(value)

        print('Removing the temporal mean (for every gridpoint) from the prior...')
        climo = np.nanmean(value,axis=0)
        value = (value - climo)
        print(var_to_extract, ': Global: mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value))

        time_yrs_all = np.tile(time_yrs,nbmems)
        value_all = np.squeeze(value[:,0,:,:]) # 1st member
        for mem in range(1,nbmems):
            value_all = np.append(value_all,np.squeeze(value[:,mem,:,:]),axis=0)

        
        # Dictionary of dictionaries
        # ex. data access : datadict['tas_sfc_Amon']['years'] => arrays containing years of the 'tas' data
        #                   datadict['tas_sfc_Amon']['value'] => array of 'tas' data values etc ...
        d = {}
        d['vartype'] = vartype
        d['years']   = time_yrs_all
        d['value']   = value_all
        d['climo']   = climo
        d['spacecoords'] = spacecoords
        if '2D' in vartype:
            d[spacecoords[0]] = spacevar1
            d[spacecoords[1]] = spacevar2

        datadict[vardef] = d


    return datadict

#==========================================================================================


def read_gridded_data_TraCE21ka(data_dir,data_file,data_vars,outtimeavg,detrend=None,anom_ref=None):
#==========================================================================================
#
# Reads the monthly data from the TraCE21ka climate model simulation and returns values of
# specified model fields averaged over a user-specified period.
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : Variables names to be read, and info on whether each
#                       variable is to be returned as anomalies of as full field
#                       (dict)
#
#      - outtimeavg   : Dictionary indicating the type of averaging (key) and associated
#                       information on averaging period (integer list)
#                       if the type is "annual": list of integers indicating the months of 
#                                                the year over which to average the data.
#                                                Requires availability of monthly data.
#                       if type is "multiyear" : list of single integer indicating the length
#                                                of averaging period (in years).
#                                                Requires availability of data with a
#                                                resolution of at least the averaging
#                                                interval.
#
#                       ex 1: outtimeavg = {'annual': [1,2,3,4,5,6,7,8,9,10,11,12]}
#                       ex 2: outtimeavg = {'multiyear': [100]} -> 100yr average
#
#      - detrend      : Boolean to indicate if detrending is to be applied to the prior
#
#      - anom_ref     : Reference period (in year CE) used in calculating anomalies
#
#
# Output: 
#      - datadict     : Master dictionary containing dictionaries, one for each state 
#                       variable, themselves containing the following numpy arrays:
#                       - time_yrs  : Array with years over which data is available.
#                                     dims: [nb_years]
#                       - lat       : Array containing the latitudes of gridded  data. 
#                                     dims: [lat]
#                       - lon       : Array containing the longitudes of gridded  data. 
#                                     dims: [lon]
#                       - value     : Array with the annually-averaged data calculated from 
#                                     monthly data dims: [time,lat,lon]
# 
#  ex. data access : datadict['tas_sfc_Amon']['years'] => array containing years of the 
#                                                         'tas' data
#                    datadict['tas_sfc_Amon']['lat']   => array of lats for 'tas' data
#                    datadict['tas_sfc_Amon']['lon']   => array of lons for 'tas' data
#                    datadict['tas_sfc_Amon']['value'] => array of 'tas' data values
#
#========================================================================================== 

    datadict = {}

    # Loop over state variables to load
    for v in range(len(data_vars)):

        vardef = list(data_vars.keys())[v]
        data_file_read = data_file.replace('[vardef_template]',vardef)
        
        # Check if file exists
        infile = data_dir + '/' + data_file_read
        if not os.path.isfile(infile):
            print('Error in specification of gridded dataset')
            print('File ', infile, ' does not exist! - Exiting ...')
            raise SystemExit(1)
        else:
            print('Reading file: ', infile)

        # Load dataset from file
        data = Dataset(infile,'r')

        # Dimensions used to store the data
        nc_dims = [dim for dim in data.dimensions]
        dictdims = {}
        for dim in nc_dims:
            dictdims[dim] = len(data.dimensions[dim])

        # Define the name of the variable to extract from the variable definition (from namelist)
        var_to_extract = vardef.split('_')[0]

        # Query dimensions of input array
        vardims = data.variables[var_to_extract].dimensions
        nbdims  = len(vardims)

        # put everything in lower case for homogeneity
        vardims = tuple([item.lower() for item in vardims])

        # extract info on variable units
        if hasattr(data.variables[var_to_extract], 'units'):
            units = data.variables[var_to_extract].units
        else:
            units = None
        
        
        # Get info about time coordinate(s)
        # ---------------------------------

        # One of the dims has to be time!
        if 'time' not in vardims:
            raise SystemExit('In read_gridded_data_TraCE21ka: '
                             'Variable does not have *time* as a dimension! Exiting!')
        else:
            # Query information on time included in file
            timedims = [item for item in vardims if item == 'time' or item == 'months']

            timecoords_dict = OrderedDict()
            for dim in timedims:
                timecoords_dict[dim] = vardims.index(dim)

            if len(timedims) == 1 and timedims[0] == 'time':
                timetype = 'time_continuous'
            elif len(timedims) == 2 and 'time' in timedims and 'months' in timedims:
                timetype = 'time_months'
            else:
                raise SystemExit('In read_gridded_data_TraCE21ka: '
                                 'Unrecognized time coordinates. Exiting.')

            # read in the time netCDF4.Variable
            time = data.variables['time']

            # Make sure time units are what is expected
            if time.units == 'ka BP':            
                # Transform "ka BP" year to CE years
                #dates_array = time[:] *1000.0 + 1950.0
                dates_array = time[:] *1000.0 + 1950.958904109 # BP = Dec 15th 1950 # mod RT 11/18 ... ... ... 
            else:
                raise SystemExit('In read_gridded_data_TraCE21ka: '
                                 'Unrecognized time units. Exiting.')

            # check on temporal resolution (in years)
            time_diff = np.diff(dates_array)
            time_diff_mean = np.mean(time_diff)
            monthly = 1./12.
            if time_diff_mean < 1.0:
                # subannual data available
                # test for monthly
                if np.isclose(time_diff_mean,monthly, rtol=1e-05, atol=1e-07, equal_nan=False):
                    time_resolution = monthly
                else:
                    time_resolution = time_diff_mean
            else:
                # annual or interannual data available
                time_resolution = np.rint(time_diff_mean)

            print(':: Data temporal resolution = ', time_resolution, 'yrs')

            # store in "date-like" objects (i.e. with .year and .month attributes)
            dates = Object()
            dates.year = dates_array.astype(int)

            # convert to months
            residuals = dates_array - dates.year
            if np.any(residuals<0.):
                residuals[residuals<0.] = 1. + residuals[residuals<0.]
            dates.month = (residuals*365./30.4167 + 1).astype(int)
            
            # get unique years in dataset
            years = np.asarray(list(set(dates.year)))
            years.sort(axis=0) # sort


        # Get info about spatial coordinates
        # ----------------------------------

        # get rid of time in list of coordinates
        varspacecoordnames = [item for item in vardims if item != 'time' and item != 'months']
        # remaining must be spatial dims
        nbspacecoords = len(varspacecoordnames)
        
        if nbspacecoords == 0:
            # data => simple time series
            vartype = '0D:time series'
            value = np.zeros([len(years)], dtype=float)
            
        elif nbspacecoords == 1:
            # data => 1D data
            if 'lat' in varspacecoordnames:
                # latitudinally-averaged  variable
                vartype = '1D:meridional' 
                spacecoords_dict = OrderedDict()
                spacecoords_dict['lat'] = data.variables['lat'][:]
                
        elif ((nbspacecoords == 2) or 
            (nbspacecoords == 3 and 'plev' in vardims and dictdims['plev'] == 1)):
            # data => 2D data

            # get rid of plev (pressure level) in list
            varspacecoordnames = [item for item in varspacecoordnames if item != 'plev'] 
            # keep info/values on spatial coordinates in ordered dict
            spacecoords_dict = OrderedDict()
            for coord in varspacecoordnames:
                spacecoords_dict[coord] = data.variables[coord][:]
            
            if 'lat' in spacecoords_dict.keys() and 'lon' in spacecoords_dict.keys():
                vartype = '2D:horizontal'
                #vlat  = data.variables['lat'][:]
                #vlon  = data.variables['lon'][:]
                
            elif 'lat' in spacecoords_dict.keys() and 'lev' in spacecoords_dict.keys():
                vartype = '2D:meridional_vertical'
                #vlat  = data.variables['lat'][:]
                #vlev  = data.variables['lev'][:]
                
            else:
                print('In read_gridded_data_TraCE21ka: Cannot handle this variable yet! '
                      '2D variable of unrecognized dimensions... Exiting!')
                raise SystemExit(1)

        else:
            print('In read_gridded_data_TraCE21ka: Cannot handle this variable yet! '
                  'Unrecognized array structure/dimensions... Exiting!')
            raise SystemExit(1)


        # load data array
        data_var = data.variables[var_to_extract][:]
        print('Array dims :', vardims)
        print('Array shape:', data_var.shape)

        
        # --------------------------------------
        # processing depending on variable type:
        # --------------------------------------

        # if 2D:horizontal variable
        # -------------------------
        if vartype == '2D:horizontal':

            print('Type of variable:', vartype)
            
            # which spatial dim is lat & which is lon?
            # -- in original array (from file)
            indlatorig = vardims.index('lat')
            indlonorig = vardims.index('lon')
            # -- in array that will be the output of this function
            indlat = list(spacecoords_dict.keys()).index('lat')
            indlon = list(spacecoords_dict.keys()).index('lon')

            print('indlat(orig)=', indlatorig, ' indlon(orig)=', indlonorig)            
            print('indlat      =', indlat,     ' indlon      =', indlon)

            nlat = data.dimensions['lat'].size
            nlon = data.dimensions['lon'].size

            vlat = spacecoords_dict['lat']
            vlon = spacecoords_dict['lon']

            # check grid & standardize grid orientation to lat=>[-90,90] & lon=>[0,360] if needed
            
            # are coordinates defined as 1d or 2d array?
            spacevardims = len(vlat.shape)
            if spacevardims == 1:
                # 1D to 2D
                varlat = np.array([vlat,]*nlon).transpose()
                varlon = np.array([vlon,]*nlat)
            else:
                # 2D already
                varlat = vlat
                varlon = vlon

            varlatdim = len(varlat.shape)
            varlondim = len(varlon.shape)

            
            # Check if latitudes are defined in the [-90,90] domain
            # and if longitudes are in the [0,360] domain
            fliplat = None

            # check for monotonically increasing or decreasing values
            monotone_increase = np.all(np.diff(varlat[:,0]) > 0)
            monotone_decrease = np.all(np.diff(varlat[:,0]) < 0)
            if not monotone_increase and not monotone_decrease:
                # funky grid
                fliplat = False

            if fliplat is None:
                if varlatdim == 2: # 2D lat array
                    if varlat[0,0] > varlat[-1,0]: # lat not as [-90,90] => array upside-down
                        fliplat = True
                    else:
                        fliplat = False
                elif varlatdim == 1: # 1D lat array
                    if varlat[0] > varlat[-1]: # lat not as [-90,90] => array upside-down
                        fliplat = True
                    else:
                        fliplat = False
                else:
                    print('In read_gridded_data_TraCE21ka: ERROR!')
                    raise SystemExit(1)

            
            if fliplat:
                varlat = np.flipud(varlat)
                # flip the data along the appropriate axis
                tmp = np.flip(data_var, axis=indlatorig)
                data_var = tmp
                
            # Transform longitudes from [-180,180] domain to [0,360] domain if needed
            indneg = np.where(varlon < 0)
            if len(indneg) > 0: # if non-empty
                varlon[indneg] = 360.0 + varlon[indneg]
                
            # Back into right arrays
            spacecoords_dict['lat'] = varlat
            spacecoords_dict['lon'] = varlon
            

        # if 2D:meridional_vertical variable
        # ----------------------------------
        elif vartype == '2D:meridional_vertical':

            print('Type of variable:', vartype)

            # which spatial dim is lat and which is lev?
            # -- in original array (from file)
            indlatorig = vardims.index('lat')
            indlevorig = vardims.index('lev')
            # -- in array that will be the output of this function
            indlat = list(spacecoords_dict.keys()).index('lat')
            indlev = list(spacecoords_dict.keys()).index('lev')

            print('indlat(orig)=', indlatorig, ' indlev(orig)=', indlevorig)            
            print('indlat      =', indlat,     ' indlev      =', indlev)

            nlat = data.dimensions['lat'].size
            nlev = data.dimensions['lev'].size

            vlat = spacecoords_dict['lat']
            vlev = spacecoords_dict['lev']


            # check grid & standardize grid orientation to lat=>[-90,90] if needed
            
            # are coordinates defined as 1d or 2d array?
            spacevardims = len(vlat.shape)
            if spacevardims == 1:
                # 1D to 2D
                varlat = np.array([vlat,]*nlev).transpose()
                varlev = np.array([vlev,]*nlat)
            else:
                # 2D already
                varlat = vlat
                varlev = vlev

            varlatdim = len(varlat.shape)
            varlevdim = len(varlev.shape)


            # Check if latitudes are defined in the [-90,90] domain
            fliplat = None

            # check for monotonically increasing or decreasing values
            monotone_increase = np.all(np.diff(varlat[:,0]) > 0)
            monotone_decrease = np.all(np.diff(varlat[:,0]) < 0)
            if not monotone_increase and not monotone_decrease:
                # funky grid
                fliplat = False
                
            if fliplat is None:
                if varlatdim == 2: # 2D lat array
                    if varlat[0,0] > varlat[-1,0]: # lat not as [-90,90] => array upside-down
                        fliplat = True
                    else:
                        fliplat = False
                elif varlatdim == 1: # 1D lat array
                    if varlat[0] > varlat[-1]: # lat not as [-90,90] => array upside-down
                        fliplat = True
                    else:
                        fliplat = False
                else:
                    print('In read_gridded_data_TraCE21ka: ERROR!')
                    raise SystemExit(1)

            if fliplat:
                varlat = np.flipud(varlat)
                # flip the data along the appropriate axis
                tmp = np.flip(data_var, axis=indlatorig)
                data_var = tmp
                

            # Back into right arrays
            spacecoords_dict['lat'] = varlat
            spacecoords_dict['lev'] = varlev

                
        # if 1D:meridional (latitudinally-averaged) variable
        # --------------------------------------------------
        elif vartype == '1D:meridional':

            print('Type of variable:', vartype)
            
            # which dim is lat?
            # -- in original array (from file)
            indlatorig = vardims.index('lat')
            # -- in array that will be the output of this function
            indlat = list(spacecoords_dict.keys()).index('lat')

            print('indlat(orig)=', indlatorig, ' indlon(orig)=', indlonorig)            
            print('indlat      =', indlat,     ' indlon      =', indlon)

            nlat = data.dimensions['lat'].size
            vlat = spacecoords_dict['lat']

            # check grid & standardize grid orientation to lat=>[-90,90] if needed
            fliplat = None
            # check for monotonically increasing or decreasing values
            monotone_increase = np.all(np.diff(vlat) > 0)
            monotone_decrease = np.all(np.diff(vlat) < 0)
            if not monotone_increase and not monotone_decrease:
                # funky grid
                fliplat = False

            if fliplat is None:
                if vlat[0] > vlat[-1]: # lat not as [-90,90] => array upside-down
                    fliplat = True

            if fliplat:
                vlat = np.flipud(vlat)
                # flip the data along the appropriate axis
                tmp = np.flip(data_var, axis=indlatorig)
                data_var = tmp

            # Back into right array
            spacecoords_dict['lat'] = vlat



        # ------------------------------------     
        # ====== other data processing ======
        
        # --------------------
        # Calculate anomalies?
        # --------------------
        kind = data_vars[vardef]
        
        if not kind or kind == 'anom':
            print('Anomalies provided as the prior: Removing the temporal mean (for every gridpoint)...')

            # monthly data?
            if timetype == 'time_continuous' and time_resolution == monthly:
                # monthly data available
                # ----------------------
                
                # monthly climatology: declaring the array depending on data type
                if vartype == '0D:time series':
                    climo_month = np.zeros((12))
                elif  vartype == '1D:meridional':
                    _,ndim1 = data_var.shape
                    climo_month = np.zeros([12, ndim1], dtype=float)
                elif '2D' in vartype:
                    _,ndim1,ndim2 = data_var.shape
                    climo_month = np.zeros([12, ndim1, ndim2], dtype=float)
                
                # indices corresponding to reference period   
                if anom_ref is not None:
                    indsyrref = [j for j,v in enumerate(dates.year) if ((v >= anom_ref[0]) and (v <= anom_ref[1]))]
                    # overlap?
                    if len(indsyrref) == 0:
                        raise SystemExit('ERROR in anomaly calculation: '
                                         ' No overlap between prior simulation'
                                         ' and specified reference period. Exiting!')
                else:
                    # no reference period specified: indices over entire length of the simulation
                    indsyrref = [j for j,v in enumerate(dates.year)]

                # loop over months
                for i in range(12):
                    m = i+1.
                    print('...calculating climo for month:', m)
                    indsm = [j for j,v in enumerate(dates.month) if v == m]
                    indsmref = [j for j,v in enumerate(dates.month) if ((v == m) and (j in indsyrref))]
                    climo_month[i] = np.nanmean(data_var[indsmref], axis=0)
                    data_var[indsm] = (data_var[indsm] - climo_month[i])
                    
                climo = climo_month

        
            elif timetype == 'time_months':
                # monthly data, but time-averaged already
                # ---------------------------------------

                axtime  = timecoords_dict['time']
                axmonth = timecoords_dict['months']
                print(axtime, axmonth)
                print(data_var.shape)
                
                # monthly climatology: declaring the array depending on data type
                if vartype == '0D:time series':
                    climo_month = np.zeros((12))
                elif  vartype == '1D:meridional':
                    _,_,ndim1 = data_var.shape
                    climo_month = np.zeros([12, ndim1], dtype=float)
                elif '2D' in vartype:
                    _,_,ndim1,ndim2 = data_var.shape
                    climo_month = np.zeros([12, ndim1, ndim2], dtype=float)

                # indices corresponding to reference period, if specified                
                if anom_ref is not None:
                    indsyrref = [j for j,v in enumerate(dates.year) if ((v >= anom_ref[0]) and (v <= anom_ref[1]))]
                    # overlap?
                    if len(indsyrref) == 0:
                        raise SystemExit('ERROR in anomaly calculation: '
                                         ' No overlap between prior simulation'
                                         ' and specified reference period. Exiting!')
                else:
                    # no reference period specified: indices over entire length of the simulation
                    indsyrref = [j for j,v in enumerate(dates.year)]
                
                # loop over months
                for i in range(12):
                    m = i+1
                    print('...calculating climo for month:', m)
                    climo_month[i] = np.nanmean(data_var[indsyrref,i], axis=0)
                    data_var[:,i] = (data_var[:,i] - climo_month[i])
                
                climo = climo_month
                
            else:
                # monthly data not available
                # --------------------------

                # indices corresponding to reference period                
                if anom_ref is not None:
                    indsyrref = [j for j,v in enumerate(dates) if ((v >= anom_ref[0]) and (v <= anom_ref[1]))]
                    # overlap?
                    if len(indsyrref) > 0:
                        climo = np.nanmean(data_var[indsyrref],axis=0)
                    else:
                        raise SystemExit('ERROR in anomaly calculation: No overlap between prior'
                                         ' simulation and specified reference period. Exiting!')

                else:
                    # anomalies w.r.t. mean over entire length of the simulation
                    climo = np.nanmean(data_var,axis=0)

                # calculate anomalies
                data_var = (data_var - climo)
            

        elif kind == 'full':
            print('Full field provided as the prior')
            
            # Calculating climo nevertheless. Needed as output.

            # monthly or not ?
            if timetype == 'time_continuous' and time_resolution == monthly:
                # monthly data available
                # ----------------------

                # monthly climatology: declaring the array depending on data type
                if vartype == '0D:time series':
                    climo_month = np.zeros((12))
                elif  vartype == '1D:meridional':
                    _,ndim1 = data_var.shape
                    climo_month = np.zeros([12, ndim1], dtype=float)
                elif '2D' in vartype:
                    _,ndim1,ndim2 = data_var.shape
                    climo_month = np.zeros([12, ndim1, ndim2], dtype=float)
                # loop over months
                for i in range(12):
                    m = i+1.
                    print('...calculating climo for month:', m)
                    indsm = [j for j,v in enumerate(dates.month) if v == m]
                    climo_month[i] = np.nanmean(data_var[indsm], axis=0)                    
                climo = climo_month

            elif timetype == 'time_months':
                # monthly data, but time-averaged already
                # ---------------------------------------
                
                # monthly climatology: declaring the array depending on data type
                if vartype == '0D:time series':
                    climo_month = np.zeros((12))
                elif  vartype == '1D:meridional':
                    _,_,ndim1 = data_var.shape
                    climo_month = np.zeros([12, ndim1], dtype=float)
                elif '2D' in vartype:
                    _,_,ndim1,ndim2 = data_var.shape
                    climo_month = np.zeros([12, ndim1, ndim2], dtype=float)
                
                # loop over months
                for i in range(12):
                    m = i+1
                    print('...calculating climo for month:', m)
                    climo_month[i] = np.nanmean(data_var[:,i], axis=0)                
                climo = climo_month
                
            else:
                # monthly data not available, climo is straight average
                climo = np.nanmean(data_var,axis=0)

        else:
            raise SystemExit('ERROR in the specification of type of prior.'
                             ' Should be "full" or "anom"! Exiting...')

        print(var_to_extract, ': Global: mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var))        

        
        # --------------------------
        # Possibly detrend the prior
        # --------------------------
        if detrend:

            # RT ... ... ... NOT YET FULLY FUNCTIONAL ... ... ... 
            # ... not compatible with all the data types (dims) that are now handled by this function ...
            raise SystemExit('Detrending of the prior not yet completely functional for the TraCE data.'

            
            print('Detrending the prior for variable: '+var_to_extract)
            if vartype == '0D:time series':
                xdim = data_var.shape[0]
                xvar = list(range(xdim))
                data_var_copy = np.copy(data_var)
                slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy)
                trend = slope*np.squeeze(xvar) + intercept
                data_var = data_var_copy - trend
            elif vartype == '1D:meridional':
                [xdim,dim1] = data_var.shape
                xvar = list(range(xdim))
                # loop over grid points
                for i in range(dim1):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy[:,i])
                    if np.isfinite(slope) and np.isfinite(intercept):
                        trend = slope*np.squeeze(xvar) + intercept
                        data_var[:,i] = data_var_copy[:,i] - trend
                    else:
                        data_var[:,i] = np.nan
            elif '2D' in vartype: 
                data_var_copy = np.copy(data_var)
                [xdim,dim1,dim2] = data_var.shape
                xvar = list(range(xdim))
                # loop over grid points
                for i in range(dim1):
                    for j in range(dim2):
                        slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy[:,i,j])
                        if np.isfinite(slope) and np.isfinite(intercept):
                            trend = slope*np.squeeze(xvar) + intercept
                            data_var[:,i,j] = data_var_copy[:,i,j] - trend
                        else:
                            data_var[:,i,j] = np.nan
                            
            print(var_to_extract, ': Global(detrended): mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var))


        # ------------------------------------------------------------------------
        # Average the data over user-specified period. Two configurations: 
        # 1) Annual: Averaging over the monthly sequence specified in outtimeavg
        #    One output data per year is provided as output, but averaged over
        #    specified sequence of months.
        #    Requires availability of monthly data.
        #
        # 2) multiyear: Averaging available data over a time interval 
        #    corresponding to the specified number of years. 
        #    Requires availability of data with a resolution of at least the
        #    averaging interval.
        # ------------------------------------------------------------------------

        # parse/validate information on time averaging
        # --------------------------------------------
        if 'annual' in outtimeavg.keys():
            year_avg = 1
            season_avg = outtimeavg['annual']

        elif 'multiyear' in outtimeavg.keys():
            year_avg = outtimeavg['multiyear'][0]
            if 'season' in outtimeavg.keys():
                season_avg = outtimeavg['season']
            else:
                season_avg = None
        else:
            raise SystemExit('In read_gridded_data_TraCE21ka: Unrecognized '
                             'time averaging configuration. Exiting.')

                
        if year_avg == 1:
            # Annual/seasonal averaging
            # -------------------------
            print('Annual averaging over month sequence:', season_avg)

            # check availability of monthly data
            if timetype == 'time_continuous' and time_resolution == monthly:

                year_current = [m for m in outtimeavg['annual'] if m>0 and m<=12]
                year_before  = [abs(m) for m in outtimeavg['annual'] if m < 0]        
                year_follow  = [m-12 for m in outtimeavg['annual'] if m > 12]

                # List years available in dataset and sort
                years_all = [d for d in dates.year]
                years     = list(set(years_all)) # 'set' used to retain unique values in list
                years.sort() # sort the list
                ntime = len(years)
                datesYears = years
        
                if vartype == '0D:time series':
                    value = np.zeros([ntime], dtype=float) # vartype = '0D:time series' 
                elif vartype == '1D:meridional':
                    _, ndim1 = data_var.shape
                    value = np.zeros([ntime, ndim1], dtype=float)
                elif '2D' in vartype:
                    _, ndim1, ndim2 = data_var.shape
                    value = np.zeros([ntime, ndim1, ndim2], dtype=float)

                # really initialize with missing values (NaNs)
                value[:] = np.nan 
                
                # Loop over years in dataset (less memory intensive...otherwise need to deal with large arrays) 
                for i in range(ntime):
                    tindsyr   = [k for k,d in enumerate(dates.year) if d == years[i]    and dates.month[k] in year_current]
                    tindsyrm1 = [k for k,d in enumerate(dates.year) if d == years[i]-1. and dates.month[k] in year_before]
                    tindsyrp1 = [k for k,d in enumerate(dates.year) if d == years[i]+1. and dates.month[k] in year_follow]
                    indsyr = tindsyrm1+tindsyr+tindsyrp1

                    if vartype == '0D:time series':
                        value[i] = np.nanmean(data_var[indsyr],axis=0)
                    elif vartype == '1D:meridional':
                        value[i,:] = np.nanmean(data_var[indsyr],axis=0)
                    elif '2D' in vartype: 
                        if nbdims > 3:
                            value[i,:,:] = np.nanmean(np.squeeze(data_var[indsyr]),axis=0)
                        else:
                            value[i,:,:] = np.nanmean(data_var[indsyr],axis=0)
                        
                print(var_to_extract, ': Global(time-averaged): mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value))


            else:
                raise SystemExit('In read_gridded_data_TraCE21ka: '
                                 'Specified averaging requires monthly data '
                                 'but data has temporal resolution of %f years. '
                                 'Exiting' %time_resolution)

        elif year_avg > 1:
            # Multiyear averaging, possible seasonal as well
            # ----------------------------------------------
            print('Averaging period (years):', year_avg)

            avg_period = float(year_avg)

            # check if specified avg. period compatible with the available data
            if avg_period < time_resolution:
                raise SystemExit('In read_gridded_data_TraCE21ka: '
                                 'Specified averaging requires data with higher temporal resolution. '
                                 'Data has temporal resolution of %f years, '
                                 'while specified averaging period is %f years. '
                                 'Exiting.' %(time_resolution, avg_period))
            else:
                pass # ok, do nothing here


            # How many averaged data can be calculated w/ the dataset?
            years = dates.year
            nbpts = int(avg_period/time_resolution)
            years_range = years[-1] - years[0]
            nbintervals = int(math.modf(years_range/avg_period)[1])            

            # Is averaging over specific season requested?
            # If so, is monthly data available?
            if season_avg:
                if timetype != 'time_months':
                    raise SystemExit('In read_gridded_data_TraCE21ka: '
                                     'Averaging over a season is requested, but monthly '
                                     'information is not available in data. Exiting.')

                # monthly data available, do the seasonal averaging
                # use np.take to extract the proper data slice
                print('...averaging period (season):', season_avg)
                indices = [item-1 for item in season_avg]
                data_var_to_average = np.nanmean(np.take(data_var, indices, axis=1), axis=1)
                
            else:
                # no seasonal averaging requested
                if timetype == 'time_months':
                    # monthly info available, average over the annual period
                    indices = list(range(0,12))
                    data_var_to_average = np.nanmean(np.take(data_var, indices, axis=1), axis=1)
                else:
                    # simply use original array used for next step
                    data_var_to_average = data_var


            # Do the multiyear averaging
            # --------------------------
            datesYears = np.zeros([nbintervals])
            if vartype == '0D:time series':
                value = np.zeros([nbintervals], dtype=float)
            elif vartype == '1D:meridional':
                _, ndim1 = data_var_to_average.shape
                value = np.zeros([nbintervals, ndim1], dtype=float)
            elif '2D' in vartype:
                _, ndim1, ndim2 = data_var_to_average.shape
                value = np.zeros([nbintervals, ndim1, ndim2])

            # initialize array with missing values (NaNs)
            value[:] = np.nan

            """
            for i in range(nbintervals):
                edgel = i*nbpts
                edger = edgel+nbpts
                datesYears[i] = np.mean(years[edgel:edger])
                value[i] = np.nanmean(data_var_to_average[edgel:edger], axis=0)
            """
            
            # More efficent method, without loop and with use of np.cumsum
            # Time variable
            ret = np.cumsum(years)
            ret[nbpts:] = ret[nbpts:] - ret[:-nbpts]
            avg = ret[nbpts-1:]/nbpts
            datesYears = avg[::nbpts][:nbintervals]

            # the field itself
            ret = np.cumsum(data_var_to_average, axis=0)
            ret[nbpts:] = ret[nbpts:] - ret[:-nbpts]
            avg = ret[nbpts-1:]/nbpts
            value = avg[::nbpts][:nbintervals]

            # if variable with mask, making sure mask is carried forward with
            # NaN as the missing flag.
            if hasattr(value,'mask'):
                value[value.mask] = np.nan
                value.mask = np.isnan(value)

            
            print(var_to_extract, ': Global(time-averaged): mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value))

            
        # ------------
        # For output :
        # ------------
        # Dictionary of dictionaries
        # ex. data access : datadict['tas_sfc_Amon']['years'] => arrays containing years of the 'tas' data
        #                   datadict['tas_sfc_Amon']['value'] => array of 'tas' data values etc ...
        d = {}
        d['vartype'] = vartype
        d['years']   = datesYears
        d['value']   = value
        d['climo']   = climo
        d['units']   = units

        if vartype != '0D:time series':
            d['spacecoords'] = tuple(spacecoords_dict.keys())
            for coord in tuple(spacecoords_dict.keys()):
                d[coord] = spacecoords_dict[coord]
        else:
            d['spacecoords'] = None


        datadict[vardef] = d

    return datadict

#==========================================================================================


def read_gridded_data_cGENIE_model(data_dir,data_file,data_vars,outtimeavg,detrend=None,anom_ref=None):
#==========================================================================================
#
# Reads the data from the cGENIE climate model simulation and returns values of
# specified model fields averaged over a user-specified period.
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : Variables names to be read, and info on whether each
#                       variable is to be returned as anomalies of as full field
#                       (dict)
#
#      - outtimeavg   : Dictionary indicating the type of averaging (key) and associated
#                       information on averaging period (integer list)
#                       if the type is "annual": list of integers indicating the months of 
#                                                the year over which to average the data.
#                                                Requires availability of monthly data.
#                       if type is "multiyear" : list of single integer indicating the length
#                                                of averaging period (in years).
#                                                Requires availability of data with a
#                                                resolution of at least the averaging
#                                                interval.
#
#                       ex 1: outtimeavg = {'annual': [1,2,3,4,5,6,7,8,9,10,11,12]}
#                       ex 2: outtimeavg = {'multiyear': [100]} -> 100yr average
#
#      - detrend      : Boolean to indicate if detrending is to be applied to the prior
#
#      - anom_ref     : Reference period (in year CE) used in calculating anomalies
#
#
# Output: 
#      - datadict     : Master dictionary containing dictionaries, one for each state 
#                       variable, themselves containing the following numpy arrays:
#                       - time_yrs  : Array with years over which data is available.
#                                     dims: [nb_years]
#                       - lat       : Array containing the latitudes of gridded  data. 
#                                     dims: [lat]
#                       - lon       : Array containing the longitudes of gridded  data. 
#                                     dims: [lon]
#                       - value     : Array with the annually-averaged data calculated from 
#                                     monthly data dims: [time,lat,lon]
# 
#  ex. data access : datadict['tas_sfc_Amon']['years'] => array containing years of the 
#                                                         'tas' data
#                    datadict['tas_sfc_Amon']['lat']   => array of lats for 'tas' data
#                    datadict['tas_sfc_Amon']['lon']   => array of lons for 'tas' data
#                    datadict['tas_sfc_Amon']['value'] => array of 'tas' data values
#
#========================================================================================== 

    datadict = {}

    # Loop over state variables to load
    for v in range(len(data_vars)):

        vardef = list(data_vars.keys())[v]
        data_file_read = data_file.replace('[vardef_template]',vardef)
        
        # Check if file exists
        infile = data_dir + '/' + data_file_read
        if not os.path.isfile(infile):
            print('Error in specification of gridded dataset')
            print('File ', infile, ' does not exist! - Exiting ...')
            exit(1)
        else:
            print('Reading file: ', infile)

        # Load entire dataset from file
        data = Dataset(infile,'r')

        # Dimensions used to store the data
        nc_dims = [dim for dim in data.dimensions]
        dictdims = {}
        for dim in nc_dims:
            dictdims[dim] = len(data.dimensions[dim])

        # Define the name of the variable to extract from the variable definition (from namelist)
        var_to_extract = vardef.split('_')[0]

        # Query its dimensions
        vardims = data.variables[var_to_extract].dimensions
        nbdims  = len(vardims)
        # names of variable dims
        vardimnames = []
        for d in vardims:
            vardimnames.append(d)
        
        # put everything in lower case for homogeneity
        vardimnames = [item.lower() for item in vardimnames]

        # One of the dims has to be time! 
        if 'time' not in vardimnames:
            print('Variable does not have *time* as a dimension! Exiting!')
            exit(1)
        else:
            # read in the time netCDF4.Variable
            time = data.variables['time']

        # keep time in "model years"
        dates = time[:]
        time_yrs_list = dates.tolist()
        
        time_diff = np.diff(time_yrs_list)
        time_diff_mean = np.mean(time_diff)
        monthly = 1./12.
        if time_diff_mean < 1.0:
            # subannual data available
            # test for monthly
            if np.isclose(time_diff_mean,monthly, rtol=1e-05, atol=1e-07, equal_nan=False):
                time_resolution = monthly
            else:
                time_resolution = time_diff_mean
        else:
            # annual or interannual data available
            time_resolution = np.rint(time_diff_mean)

        print(':: Data temporal resolution = ', time_resolution, 'yrs')

        # List years available in dataset and sort
        years_all = [int(d) for d in np.rint(dates)]
        years     = list(set(years_all)) # 'set' is used to retain unique values in list
        years.sort() # sort the list
        time_yrs  = np.zeros(len(years), dtype=int)

                
        # Query info on spatial coordinates ...
        # get rid of time in list        
        varspacecoordnames = [item for item in vardimnames if item != 'time'] 
        nbspacecoords = len(varspacecoordnames)
        #print vardimnames, nbspacecoords, dictdims

        if nbspacecoords == 0: # data => simple time series
            vartype = '0D:time series'
            value = np.empty([len(years)], dtype=float)
            spacecoords = None
        elif nbspacecoords == 1: # data => 1D data
            if 'lat' in varspacecoordnames:
                # latitudinally-averaged  variable
                vartype = '1D:meridional' 
                spacecoords = ('lat',)
                spacevar1 = data.variables['lat'][:]
        elif ((nbspacecoords == 2) or (nbspacecoords == 3 and 'plev' in vardimnames and dictdims['plev'] == 1)): # data => 2D data
            # get rid of plev in list        
            varspacecoordnames = [item for item in varspacecoordnames if item != 'plev'] 
            spacecoords = (varspacecoordnames[0],varspacecoordnames[1])
            spacevar1 = data.variables[spacecoords[0]][:]
            spacevar2 = data.variables[spacecoords[1]][:]

            if 'lat' in spacecoords and 'lon' in spacecoords:
                vartype = '2D:horizontal'
            elif 'lat' in spacecoords and 'lev' in spacecoords:
                vartype = '2D:meridional_vertical'
            else:
                print('Cannot handle this variable yet! 2D variable of unrecognized dimensions... Exiting!')
                exit(1)
        else:
            print('Cannot handle this variable yet! Too many dimensions... Exiting!')
            exit(1)

        # data array
        data_var = data.variables[var_to_extract][:]
        print(data_var.shape)
        
        # if 2D:horizontal variable, check grid & standardize grid orientation to lat=>[-90,90] & lon=>[0,360] if needed
        if vartype == '2D:horizontal':

            vardims = data_var.shape

            # which dim is lat & which is lon?
            indlat = spacecoords.index('lat')
            indlon = spacecoords.index('lon')
            print('indlat=', indlat, ' indlon=', indlon)
            
            if indlon == 0:
                vlon = spacevar1
                vlat = spacevar2
                nlat = vardims[2]
                nlon = vardims[1]
            elif indlon == 1:
                vlon = spacevar2
                vlat = spacevar1
                nlat = vardims[1]
                nlon = vardims[2]

            # are coordinates defined as 1d or 2d array?
            spacevardims = len(vlat.shape)
            if spacevardims == 1:
                varlat = np.array([vlat,]*nlon).transpose()
                varlon = np.array([vlon,]*nlat)
            else:
                varlat = vlat
                varlon = vlon

            varlatdim = len(varlat.shape)
            varlondim = len(varlon.shape)
            
            # Check if latitudes are defined in the [-90,90] domain
            # and if longitudes are in the [0,360] domain
            fliplat = None

            # check for monotonically increasing or decreasing values
            monotone_increase = np.all(np.diff(varlat[:,0]) > 0)
            monotone_decrease = np.all(np.diff(varlat[:,0]) < 0)
            if not monotone_increase and not monotone_decrease:
                # funky grid
                fliplat = False

            if fliplat is None:
                if varlatdim == 2: # 2D lat array
                    if varlat[0,0] > varlat[-1,0]: # lat not as [-90,90] => array upside-down
                        fliplat = True
                    else:
                        fliplat = False
                elif varlatdim == 1: # 1D lat array
                    if varlat[0] > varlat[-1]: # lat not as [-90,90] => array upside-down
                        fliplat = True
                    else:
                        fliplat = False
                else:
                    print('ERROR!')
                    raise SystemExit(1)

            if fliplat:
                varlat = np.flipud(varlat)
                # flip data variable
                if indlat == 0:
                    tmp = data_var[:,::-1,:]
                else:
                    tmp = data_var[:,:,::-1]
                data_var = tmp

            # ---------------------------------------------------------------------------------
            # Transform longitudes from [-180,180] domain to [0,360] domain if needed
            # flip or roll along longitudes?
            # This code assumes a possible longitude array structure as (example)
            # [-175. -165. -155. ...,  155.  165.  175.] to be transformed to
            # [   5.,   15.,   25., ...,  335.,  345.,  355.]
            indneg = np.where(varlon[0,:] < 0)
            nbneg = len(indneg[0])
            varlon = np.roll(varlon,nbneg,axis=1)
            # do same for data array
            tmp = np.roll(data_var,nbneg,axis=2)
            data_var = tmp
                
            # negative lon values to positive
            indneg = np.where(varlon < 0)
            if len(indneg) > 0: # if non-empty
                varlon[indneg] = 360.0 + varlon[indneg]
            # ---------------------------------------------------------------------------------

            # Coords back into right arrays
            if indlon == 0:
                spacevar1 = varlon
                spacevar2 = varlat
            elif indlon == 1:
                spacevar2 = varlon
                spacevar1 = varlat
        

        # if 2D:meridional_vertical variable
        elif vartype == '2D:meridional_vertical':
            
            vardims = data_var.shape

            # which dim is lat and which is lev?
            indlat = spacecoords.index('lat')
            indlev = spacecoords.index('lev')
            print('indlat=', indlat, ' inlev=', indlev)

            if indlev == 0:
                vlev = spacevar1
                vlat = spacevar2
                nlat = vardims[2]
                nlev = vardims[1]
            elif indlev == 1:
                vlev = spacevar2
                vlat = spacevar1
                nlat = vardims[1]
                nlev = vardims[2]

            # are coordinates defined as 1d or 2d array?
            spacevardims = len(vlat.shape)
            if spacevardims == 1:
                varlat = np.array([vlat,]*nlev).transpose()
                varlev = np.array([vlev,]*nlat)
            else:
                varlat = vlat
                varlev = vlev

            varlatdim = len(varlat.shape)
            varlevdim = len(varlev.shape)


            # Check if latitudes are defined in the [-90,90] domain
            fliplat = None

            # check for monotonically increasing or decreasing values
            monotone_increase = np.all(np.diff(varlat[:,0]) > 0)
            monotone_decrease = np.all(np.diff(varlat[:,0]) < 0)
            if not monotone_increase and not monotone_decrease:
                # funky grid
                fliplat = False
                
            if fliplat is None:
                if varlatdim == 2: # 2D lat array
                    if varlat[0,0] > varlat[-1,0]: # lat not as [-90,90] => array upside-down
                        fliplat = True
                    else:
                        fliplat = False
                elif varlatdim == 1: # 1D lat array
                    if varlat[0] > varlat[-1]: # lat not as [-90,90] => array upside-down
                        fliplat = True
                    else:
                        fliplat = False
                else:
                    print('ERROR!')
                    raise SystemExit(1)

            if fliplat:
                varlat = np.flipud(varlat)
                # flip data variable
                if indlat == 0:
                    tmp = data_var[:,::-1,:]
                else:
                    tmp = data_var[:,:,::-1]
                data_var = tmp

            # Back into right arrays
            if indlev == 0:
                spacevar1 = varlev
                spacevar2 = varlat
            elif indlev == 1:
                spacevar2 = varlev
                spacevar1 = varlat
            

        # if 1D:meridional (latitudinally-averaged) variable
        elif vartype == '1D:meridional':
            vardims = data_var.shape
            
            # which dim is lat?
            indlat = spacecoords.index('lat')
            print('indlat=', indlat)
            
            # Check if latitudes are defined in the [-90,90] domain
            fliplat = None
            # check for monotonically increasing or decreasing values
            monotone_increase = np.all(np.diff(spacevar1) > 0)
            monotone_decrease = np.all(np.diff(spacevar1) < 0)
            if not monotone_increase and not monotone_decrease:
                # funky grid
                fliplat = False

            if fliplat is None:
                if spacevar1[0] > spacevar1[-1]: # lat not as [-90,90] => array upside-down
                    fliplat = True

            if fliplat:
                spacevar1 = np.flipud(spacevar1)
                tmp = data_var[:,::-1]
                data_var = tmp

                
        # ====== other data processing ======
        
        # --------------------
        # Calculate anomalies?
        # --------------------
        kind = data_vars[vardef]
        
        if not kind or kind == 'anom':
            print('Anomalies provided as the prior: Removing the temporal mean (for every gridpoint)...')

            # monthly data?
            if time_resolution == monthly:
                # monthly climatology
                if vartype == '0D:time series':
                    climo_month = np.zeros((12))
                elif  vartype == '1D:meridional':
                    climo_month = np.zeros([12, vardims[1]], dtype=float)
                elif '2D' in vartype:
                    climo_month = np.zeros([12, vardims[1], vardims[2]], dtype=float)
                
                dates_years = np.array([math.modf(item)[1] for item in dates])
                tmp = np.array([abs(math.modf(item)[0]) for item in dates])
                dates_months = np.rint((tmp/monthly)+1.)

                # indices corresponding to reference period   
                if anom_ref:                    
                    indsyr = [j for j,v in enumerate(dates_years) if ((v >= anom_ref[0]) and (v <= anom_ref[1]))]
                    # overlap?
                    if len(indsyr) == 0:
                        raise SystemExit('ERROR in anomaly calculation: No overlap between prior simulation and specified reference period. Exiting!')
                else:
                    # indices over entire length of the simulation
                    indsyr = [j for j,v in enumerate(dates_years)]

                # loop over months
                for i in range(12):
                    m = i+1.
                    indsm_ref = [j for j,v in enumerate(dates_months[indsyr]) if v == m]
                    climo_month[i] = np.nanmean(data_var[indsm_ref], axis=0)
                    indsm_all = [j for j,v in enumerate(dates_months) if v == m]
                    data_var[indsm_all] = (data_var[indsm_all] - climo_month[i])
                climo = climo_month
            else:
                # other than monthly data
                # indices corresponding to reference period                
                if anom_ref:
                    indsyr = [j for j,v in enumerate(dates) if ((v >= anom_ref[0]) and (v <= anom_ref[1]))]
                    # overlap?
                    if len(indsyr) > 0:
                        climo = np.nanmean(data_var[indsyr],axis=0)
                    else:
                        raise SystemExit('ERROR in anomaly calculation: No overlap between prior simulation and specified reference period. Exiting!')
                else:
                    # anomalies w.r.t. mean over entire length of the simulation
                    climo = np.nanmean(data_var,axis=0)

                # calculate anomalies
                data_var = (data_var - climo)
                
        elif kind == 'full':
            print('Full field provided as the prior')
            # Calculating climo nevertheless. Needed as output.
            climo = np.nanmean(data_var,axis=0)
            # do nothing else...
        else:
            raise SystemExit('ERROR in the specification of type of prior. Should be "full" or "anom"! Exiting...')

        print(var_to_extract, ': Global: mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var))        

        
        # --------------------------
        # Possibly detrend the prior
        # --------------------------
        if detrend:
            print('Detrending the prior for variable: '+var_to_extract)
            if vartype == '0D:time series':
                xdim = data_var.shape[0]
                xvar = list(range(xdim))
                data_var_copy = np.copy(data_var)
                slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy)
                trend = slope*np.squeeze(xvar) + intercept
                data_var = data_var_copy - trend
            elif vartype == '1D:meridional':
                [xdim,dim1] = data_var.shape
                xvar = list(range(xdim))
                # loop over grid points
                for i in range(dim1):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy[:,i])
                    if np.isfinite(slope) and np.isfinite(intercept):
                        trend = slope*np.squeeze(xvar) + intercept
                        data_var[:,i] = data_var_copy[:,i] - trend
                    else:
                        data_var[:,i] = np.nan
            elif '2D' in vartype: 
                data_var_copy = np.copy(data_var)
                [xdim,dim1,dim2] = data_var.shape
                xvar = list(range(xdim))
                # loop over grid points
                for i in range(dim1):
                    for j in range(dim2):
                        slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy[:,i,j])
                        if np.isfinite(slope) and np.isfinite(intercept):
                            trend = slope*np.squeeze(xvar) + intercept
                            data_var[:,i,j] = data_var_copy[:,i,j] - trend
                        else:
                            data_var[:,i,j] = np.nan
                            
            print(var_to_extract, ': Global(detrended): mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var))


        # ------------------------------------------------------------------------
        # Average the data over user-specified period. Two configurations: 
        # 1) Annual: Averaging over the monthly sequence specified in outtimeavg
        #    One output data per year is provided as output, but averaged over
        #    specified sequence of months.
        #    Requires availability of monthly data.
        #
        # 2) multiyear: Averaging available data over a time interval 
        #    corresponding to the specified number of years. 
        #    Requires availability of data with a resolution of at least the
        #    averaging interval.
        # ------------------------------------------------------------------------

        if list(outtimeavg.keys())[0] == 'annual':
            print('Averaging over month sequence:', outtimeavg['annual'])

            # check availability of monthly data
            if time_resolution == monthly:

                year_current = [m for m in outtimeavg['annual'] if m>0 and m<=12]
                year_before  = [abs(m) for m in outtimeavg['annual'] if m < 0]        
                year_follow  = [m-12 for m in outtimeavg['annual'] if m > 12]
        
                avgmonths = year_before + year_current + year_follow
                indsclimo = sorted([item-1 for item in avgmonths])
        
                # List years available in dataset and sort
                years_all = [d for d in dates_years]
                years     = list(set(years_all)) # 'set' used to retain unique values in list
                years.sort() # sort the list
                ntime = len(years)
                datesYears = years
        
                if vartype == '0D:time series':
                    value = np.zeros([ntime], dtype=float) # vartype = '0D:time series' 
                elif vartype == '1D:meridional':
                    value = np.zeros([ntime, vardims[1]], dtype=float)
                elif '2D' in vartype:
                    value = np.zeros([ntime, vardims[1], vardims[2]], dtype=float)
                # really initialize with missing values (NaNs)
                value[:] = np.nan 
                
                # Loop over years in dataset (less memory intensive...otherwise need to deal with large arrays) 
                for i in range(ntime):
                    tindsyr   = [k for k,d in enumerate(dates_years) if d == years[i]    and dates_months[k] in year_current]
                    tindsyrm1 = [k for k,d in enumerate(dates_years) if d == years[i]-1. and dates_months[k] in year_before]
                    tindsyrp1 = [k for k,d in enumerate(dates_years) if d == years[i]+1. and dates_months[k] in year_follow]
                    indsyr = tindsyrm1+tindsyr+tindsyrp1

                    if vartype == '0D:time series':
                        value[i] = np.nanmean(data_var[indsyr],axis=0)
                    elif vartype == '1D:meridional':
                        value[i,:] = np.nanmean(data_var[indsyr],axis=0)
                    elif '2D' in vartype: 
                        if nbdims > 3:
                            value[i,:,:] = np.nanmean(np.squeeze(data_var[indsyr]),axis=0)
                        else:
                            value[i,:,:] = np.nanmean(data_var[indsyr],axis=0)
                        
                print(var_to_extract, ': Global(time-averaged): mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value))
                    
                climo = np.mean(climo_month[indsclimo], axis=0)

            else:
                print('ERROR: Specified averaging requires monthly data')
                print('       Here we have data with temporal resolution of ', time_resolution, 'years')
                print('       Exiting!')
                raise SystemExit()
                
        elif list(outtimeavg.keys())[0] == 'multiyear':
            print('Averaging period (years):', outtimeavg['multiyear'])

            avg_period = float(outtimeavg['multiyear'][0])

            # check if specified avg. period compatible with the available data
            if avg_period < time_resolution:
                print('ERROR: Specified averaging requires data with higher temporal resolution!')
                print('       Here we have data with temporal resolution of ', time_resolution, 'years')
                print('       while specified averaging period is:', avg_period, ' yrs')
                print('       Exiting!')
                raise SystemExit()
            else:
                pass # ok, do nothing here

            # How many averaged data can be calculated w/ the dataset?
            nbpts = int(avg_period/time_resolution)
            years_range = years[-1] - years[0]
            nbintervals = int(math.modf(years_range/avg_period)[1])

            datesYears = np.zeros([nbintervals])
            if vartype == '0D:time series':
                value = np.zeros([nbintervals], dtype=float)
            elif vartype == '1D:meridional':
                value = np.zeros([nbintervals, vardims[1]], dtype=float)
            elif '2D' in vartype: 
                #value = np.zeros([nbintervals, vardims[1], vardims[2]], dtype=float)
                value = np.zeros([nbintervals, vardims[1], vardims[2]])
            # really initialize with missing values (NaNs)
            value[:] = np.nan

            for i in range(nbintervals):
                edgel = i*nbpts
                edger = edgel+nbpts
                datesYears[i] = np.mean(years[edgel:edger])
                value[i] = np.nanmean(data_var[edgel:edger], axis=0)

        
        # Dictionary of dictionaries
        # ex. data access : datadict['tas_sfc_Amon']['years'] => arrays containing years of the 'tas' data
        #                   datadict['tas_sfc_Amon']['value'] => array of 'tas' data values etc ...
        d = {}
        d['vartype'] = vartype
        d['years']   = datesYears
        d['value']   = value
        d['climo']   = climo
        d['spacecoords'] = spacecoords
        if vartype == '1D:meridional':
            d[spacecoords[0]] = spacevar1
        elif '2D' in vartype:
            d[spacecoords[0]] = spacevar1
            d[spacecoords[1]] = spacevar2

        datadict[vardef] = d


    return datadict

#==========================================================================================
