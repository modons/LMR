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

"""
from netCDF4 import Dataset, date2num, num2date
from datetime import datetime, timedelta
from scipy import stats
import numpy as np
import os.path
import string


def read_gridded_data_GISTEMP(data_dir,data_file,data_vars,outfreq):
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
        print 'Error in specification of gridded dataset'
        print 'File ', infile, ' does not exist! - Exiting ...'
        exit(1)

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        print 'Too many variables to read!'
        print 'This file only contains surface air temperature (anomalies)'
        print 'Exiting!'
        exit(1)

        
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


def read_gridded_data_HadCRUT(data_dir,data_file,data_vars,outfreq):
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
        print 'Error in specification of gridded dataset'
        print 'File ', infile, ' does not exist! - Exiting ...'
        exit(1)

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        print 'Too many variables to read!'
        print 'This file only contains surface air temperature (anomalies)'
        print 'Exiting!'
        exit(1)


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


def read_gridded_data_BerkeleyEarth(data_dir,data_file,data_vars,outfreq):
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
        print 'Error in specification of gridded dataset'
        print 'File ', infile, ' does not exist! - Exiting ...'
        exit(1)

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        print 'Too many variables to read!'
        print 'This file only contains surface air temperature (anomalies)'
        print 'Exiting!'
        exit(1)

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
    for i in xrange(0,len(data.variables['time'][:])):
        yrAD = data.variables['time'][i]
        year = int(yrAD)
        rem = yrAD - year
        base = datetime(year, 1, 1)
        time_yrs.append(base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem))

    dates = np.array(time_yrs)

    fillval = data.variables['temperature'].missing_value
    value = np.copy(data.variables['temperature'])    
    value[value == fillval] = np.NAN

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

def read_gridded_data_MLOST(data_dir,data_file,data_vars,outfreq):
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
    infile = data_dir+'/MLOST/'+data_file
    if not os.path.isfile(infile):
        print 'Error in specification of gridded dataset'
        print 'File ', infile, ' does not exist! - Exiting ...'
        exit(1)

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        print 'Too many variables to read!'
        print 'This file only contains surface air temperature (anomalies)'
        print 'Exiting!'
        exit(1)

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

def read_gridded_data_GPCC(data_dir,data_file,data_vars,out_anomalies,outfreq):
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
        print 'Error in specification of gridded dataset'
        print 'File ', infile, ' does not exist! - Exiting ...'
        exit(1)

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        print 'Too many variables to read!'
        print 'This file only contains precipitation accumulation or flux data'
        print 'Exiting!'
        exit(1)

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
        ref_period = [1951,1980] # same as GISTEMP temperature anomalies
        climo_month = np.zeros([12, len(lat), len(lon)], dtype=float)
        # loop over months
        for i in range(12):
            m = i+1
            indsm = [j for j,v in enumerate(dates) if v.year >= ref_period[0] and v.year <= ref_period[1] and v.month == m]
            climo_month[i] = np.nanmean(value[indsm], axis=0)
            value[indsm] = (value[indsm] - climo_month[i])
    
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

def read_gridded_data_DaiPDSI(data_dir,data_file,data_vars,out_anomalies,outfreq):
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
        print 'Error in specification of gridded dataset'
        print 'File ', infile, ' does not exist! - Exiting ...'
        exit(1)

    # Sanity check on number of variables to read
    if len(data_vars) > 1:
        print 'Too many variables to read!'
        print 'This file only contains surface PDSI (anomalies)'
        print 'Exiting!'
        exit(1)


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
        ref_period = [1951,1980] # same as GISTEMP temperature anomalies
        climo_month = np.zeros([12, len(lat), len(lon)], dtype=float)
        # loop over months
        for i in range(12):
            m = i+1
            indsm = [j for j,v in enumerate(dates) if v.year >= ref_period[0] and v.year <= ref_period[1] and v.month == m]
            climo_month[i] = np.nanmean(value[indsm], axis=0)
            value[indsm] = (value[indsm] - climo_month[i])
    
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

def read_gridded_data_CMIP5_model(data_dir,data_file,data_vars,outtimeavg,detrend=None):
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
#      - outtimeavg   : List indicating the months over which to average the data.
#                       (integer list)
#
#      - detrend      : Boolean to indicate if detrending is to be applied to the prior
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
        vardef = data_vars.keys()[v]
        data_file_read = string.replace(data_file,'[vardef_template]',vardef)
        
        # Check if file exists
        infile = data_dir + '/' + data_file_read
        if not os.path.isfile(infile):
            print 'Error in specification of gridded dataset'
            print 'File ', infile, ' does not exist! - Exiting ...'
            raise SystemExit()
        else:
            print 'Reading file: ', infile

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

        # One of the dims has to be time! 
        if 'time' not in vardimnames:
            print 'Variable does not have *time* as a dimension! Exiting!'
            raise SystemExit()
        else:
            # read in the time netCDF4.Variable
            time = data.variables['time']

        # Transform into calendar dates using netCDF4 variable attributes (units & calendar)
        # TODO: may not want to depend on netcdf4.num2date...
        try:
            if hasattr(time, 'calendar'):
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
            year_diff = year - 0001
            new_start_date = datetime(0001, 01, 01, 0, 0, 0)

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
            vartype = '1D:time series'
            spacecoords = None
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
                print 'Cannot handle this variable yet! 2D variable of unrecognized dimensions... Exiting!'
                raise SystemExit()
        else:
            print 'Cannot handle this variable yet! Too many dimensions... Exiting!'
            raise SystemExit()



        # -----------------
        # Upload data array
        # -----------------
        data_var = data.variables[var_to_extract][:]
        print data_var.shape

        ntime = len(data.dimensions['time'])
        dates = time_yrs

        # if 2D:horizontal variable, check grid & standardize grid orientation to lat=>[-90,90] & lon=>[0,360] if needed
        if vartype == '2D:horizontal':

            # which dim is lat & which is lon?
            indlat = spacecoords.index('lat')
            indlon = spacecoords.index('lon')
            #print 'indlat=', indlat, ' indlon=', indlon

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
                    tmp = data_var[:,::-1,:]
                else:
                    tmp = data_var[:,:,::-1]
                data_var = tmp

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
        

        # if 2D:meridional_vertical variable
        #if vartype == '2D:meridional_vertical':
        #    value = np.zeros([ntime, len(spacevar1), len(spacevar2)], dtype=float)
        # TODO ...

        
        # Calculate anomalies?
        kind = data_vars[vardef]
        
        # monthly climatology
        if vartype == '1D:time series':
            climo_month = np.zeros((12))
        elif '2D' in vartype:
            climo_month = np.zeros([12, len(spacevar1), len(spacevar2)], dtype=float)

        if not kind or kind == 'anom':
            print 'Anomalies provided as the prior: Removing the temporal mean (for every gridpoint)...'
            # loop over months
            for i in range(12):
                m = i+1
                indsm = [j for j,v in enumerate(dates) if v.month == m]
                climo_month[i] = np.nanmean(data_var[indsm], axis=0)
                data_var[indsm] = (data_var[indsm] - climo_month[i])

        elif kind == 'full':
            print 'Full field provided as the prior'
            # do nothing else...
        else:
            print 'ERROR in the specification of type of prior. Should be "full" or "anom"! Exiting...'
            raise SystemExit()

        print var_to_extract, ': Global(monthly): mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var)

        # Possibly detrend the prior
        if detrend:
            print 'Detrending the prior for variable: '+var_to_extract
            if vartype == '1D:time series':
                xdim = data_var.shape[0]
                xvar = range(xdim)
                data_var_copy = np.copy(data_var)
                slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy)
                trend = slope*np.squeeze(xvar) + intercept
                data_var = data_var_copy - trend
            elif '2D' in vartype: 
                data_var_copy = np.copy(data_var)
                [xdim,dim1,dim2] = data_var.shape
                xvar = range(xdim)
                # loop over grid points
                for i in range(dim1):
                    for j in range(dim2):
                        slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy[:,i,j])
                        trend = slope*np.squeeze(xvar) + intercept
                        data_var[:,i,j] = data_var_copy[:,i,j] - trend

            print var_to_extract, ': Global(monthly/detrend): mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var)


        # ----------------------------------------------------------------
        # Average monthly data over the monthly sequence in outtimeavg.
        # Note: Means one output data per year, but averaged over specific 
        #       sequence of months.
        # ----------------------------------------------------------------

        print 'Averaging over month sequence:', outtimeavg
        
        year_current = [m for m in outtimeavg if m>0 and m<=12]
        year_before  = [abs(m) for m in outtimeavg if m < 0]        
        year_follow  = [m-12 for m in outtimeavg if m > 12]
        
        avgmonths = year_before + year_current + year_follow
        indsclimo = sorted([item-1 for item in avgmonths])
        
        # List years available in dataset and sort
        years_all = [d.year for d in time_yrs_list]
        years     = list(set(years_all)) # 'set' used to retain unique values in list
        years.sort() # sort the list
        ntime = len(years)
        datesYears = np.array([datetime(y,1,1,0,0) for y in years])
        
        if vartype == '1D:time series':
            value = np.zeros([ntime], dtype=float) # vartype = '1D:time series' 
        elif vartype == '2D:horizontal':
            value = np.zeros([ntime, len(spacevar1), len(spacevar2)], dtype=float)


        # Loop over years in dataset (less memory intensive...otherwise need to deal with large arrays) 
        for i in range(ntime):
            tindsyr   = [k for k,d in enumerate(dates) if d.year == years[i]    and d.month in year_current]
            tindsyrm1 = [k for k,d in enumerate(dates) if d.year == years[i]-1. and d.month in year_before]
            tindsyrp1 = [k for k,d in enumerate(dates) if d.year == years[i]+1. and d.month in year_follow]
            indsyr = tindsyrm1+tindsyr+tindsyrp1

            if vartype == '1D:time series':
                value[i] = np.nanmean(data_var[indsyr],axis=0)
            elif '2D' in vartype: 
                if nbdims > 3:
                    value[i,:,:] = np.nanmean(np.squeeze(data_var[indsyr]),axis=0)
                else:
                    value[i,:,:] = np.nanmean(data_var[indsyr],axis=0)

        
        print var_to_extract, ': Global(time-averaged): mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value)

        climo = np.mean(climo_month[indsclimo], axis=0)
        
        # Dictionary of dictionaries
        # ex. data access : datadict['tas_sfc_Amon']['years'] => arrays containing years of the 'tas' data
        #                   datadict['tas_sfc_Amon']['value'] => array of 'tas' data values etc ...
        d = {}
        d['vartype'] = vartype
        d['years']   = datesYears
        d['value']   = value
        d['climo']   = climo
        d['spacecoords'] = spacecoords
        if '2D' in vartype:
            d[spacecoords[0]] = spacevar1
            d[spacecoords[1]] = spacevar2

        datadict[vardef] = d


    return datadict


def read_gridded_data_CMIP5_model_old(data_dir,data_file,data_vars,outfreq,detrend=None,kind=None):
#==========================================================================================
#
# Reads the monthly data from a CMIP5 model and return yearly averaged values
#
# Input: 
#      - data_dir     : Full name of directory containing gridded 
#                       data. (string)
#      - data_file    : Name of file containing gridded data. (string)
#
#      - data_vars    : List of variable names to read. (string list)
#
#      - outfreq      : String indicating whether monthly or annually-averaged data 
#                       are to be returned
#
#      - detrend      : Boolean to indicate if detrending is to be applied to the prior
#
#      - kind         : String indicating whether the prior is to be returned as
#                       a full field or as anomalies (w.r.t. to the gridpt temporal mean)
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
        vardef = data_vars[v]
        data_file_read = string.replace(data_file,'[vardef_template]',vardef)
        
        # Check if file exists
        infile = data_dir + '/' + data_file_read
        if not os.path.isfile(infile):
            print 'Error in specification of gridded dataset'
            print 'File ', infile, ' does not exist! - Exiting ...'
            raise SystemExit()
        else:
            print 'Reading file: ', infile

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

        # One of the dims has to be time! 
        if 'time' not in vardimnames:
            print 'Variable does not have *time* as a dimension! Exiting!'
            raise SystemExit()
        else:
            # read in the time netCDF4.Variable
            time = data.variables['time']

        # Transform into calendar dates using netCDF4 variable attributes (units & calendar)
        # TODO: may not want to depend on netcdf4.num2date...
        try:
            if hasattr(time, 'calendar'):
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
            year_diff = year - 0001
            new_start_date = datetime(0001, 01, 01, 0, 0, 0)

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
            vartype = '1D:time series'
            spacecoords = None
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
                print 'Cannot handle this variable yet! 2D variable of unrecognized dimensions... Exiting!'
                raise SystemExit()
        else:
            print 'Cannot handle this variable yet! Too many dimensions... Exiting!'
            raise SystemExit()



        # -----------------
        # Upload data array
        # -----------------
        data_var = data.variables[var_to_extract][:]
        print data_var.shape

        ntime = len(data.dimensions['time'])
        dates = time_yrs

        # if 2D:horizontal variable, check grid & standardize grid orientation to lat=>[-90,90] & lon=>[0,360] if needed
        if vartype == '2D:horizontal':

            # which dim is lat & which is lon?
            indlat = spacecoords.index('lat')
            indlon = spacecoords.index('lon')
            print 'indlat=', indlat, ' indlon=', indlon

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
                    tmp = data_var[:,::-1,:]             
                else:
                    tmp = data_var[:,:,::-1] 
                data_var = tmp

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
        

        # if 2D:meridional_vertical variable
        #if vartype == '2D:meridional_vertical':
        #    value = np.zeros([ntime, len(spacevar1), len(spacevar2)], dtype=float)
        # TODO ...

        
        # Calculate anomalies?
        # monthly climatology
        if vartype == '1D:time series':
            climo_month = np.zeros((12))
        elif '2D' in vartype:
            climo_month = np.zeros([12, len(spacevar1), len(spacevar2)], dtype=float)

        if not kind or kind == 'anom':
            print 'Removing the temporal mean (for every gridpoint) from the prior...'
            # loop over months
            for i in range(12):
                m = i+1
                indsm = [j for j,v in enumerate(dates) if v.month == m]
                climo_month[i] = np.nanmean(data_var[indsm], axis=0)
                data_var[indsm] = (data_var[indsm] - climo_month[i])

        elif kind == 'full':
            print 'Full field provided as the prior'
            # do nothing else...
        else:
            print 'ERROR in the specification of type of prior. Should be "full" or "anom"! Exiting...'
            raise SystemExit()

        print var_to_extract, ': Global: mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var)

        # Possibly detrend the prior
        if detrend:
            print 'Detrending the prior for variable: '+var_to_extract
            if vartype == '1D:time series':
                xdim = data_var.shape[0]
                xvar = range(xdim)
                data_var_copy = np.copy(data_var)
                slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy)
                trend = slope*np.squeeze(xvar) + intercept
                data_var = data_var_copy - trend
            elif '2D' in vartype: 
                data_var_copy = np.copy(data_var)
                [xdim,dim1,dim2] = data_var.shape
                xvar = range(xdim)
                # loop over grid points
                for i in range(dim1):
                    for j in range(dim2):
                        slope, intercept, r_value, p_value, std_err = stats.linregress(xvar,data_var_copy[:,i,j])
                        trend = slope*np.squeeze(xvar) + intercept
                        data_var[:,i,j] = data_var_copy[:,i,j] - trend

            print var_to_extract, ': Global: mean=', np.nanmean(data_var), ' , std-dev=', np.nanstd(data_var)


        if outfreq == 'annual':
            # -----------------------------------------
            # Calculate annual mean from monthly data
            # Note: assume data has dims [time,lat,lon]
            # -----------------------------------------
            # List years available in dataset and sort
            years_all = [d.year for d in time_yrs_list]
            years     = list(set(years_all)) # 'set' is used to retain unique values in list
            years.sort() # sort the list
            ntime = len(years)
            dates = np.array([datetime(y,1,1,0,0) for y in years])
            
            if vartype == '1D:time series':
                value = np.zeros([ntime], dtype=float) # vartype = '1D:time series' 
            elif vartype == '2D:horizontal':
                value = np.zeros([ntime, len(spacevar1), len(spacevar2)], dtype=float)
            
            # Loop over years in dataset
            for i in range(0,len(years)): 
                # find indices in time array where "years[i]" appear
                ind = [j for j, k in enumerate(years_all) if k == years[i]]
                time_yrs[i] = years[i]

                if vartype == '1D:time series':
                    value[i] = np.nanmean(data_var[ind],axis=0)
                elif '2D' in vartype: 
                    if nbdims > 3:
                        value[i,:,:] = np.nanmean(np.squeeze(data_var[ind]),axis=0)
                    else:
                        value[i,:,:] = np.nanmean(data_var[ind],axis=0)

            climo = np.mean(climo_month, axis=0)
                        
        elif outfreq == 'monthly':
            value = data_var
            climo = climo_month
        else:
            print 'ERROR: Unsupported averaging interval for prior!'
            raise SystemExit()

        print var_to_extract, ': Global: mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value)

        
        # Dictionary of dictionaries
        # ex. data access : datadict['tas_sfc_Amon']['years'] => arrays containing years of the 'tas' data
        #                   datadict['tas_sfc_Amon']['value'] => array of 'tas' data values etc ...
        d = {}
        d['vartype'] = vartype
        d['years']   = dates
        d['value']   = value
        d['climo']   = climo
        d['spacecoords'] = spacecoords
        if '2D' in vartype:
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
        data_file_read = string.replace(data_file,'[vardef_template]',vardef)
        
        # Check if file exists
        infile = data_dir + '/' + data_file_read
        if not os.path.isfile(infile):
            print 'Error in specification of gridded dataset'
            print 'File ', infile, ' does not exist! - Exiting ...'
            exit(1)
        else:
            print 'Reading file: ', infile

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
            print 'Variable does not have *time* as a dimension! Exiting!'
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
            year_diff = year - 0001
            new_start_date = datetime(0001, 01, 01, 0, 0, 0)

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
        print 'nbmems=', nbmems, 'indmem=', indmem

        # Query info on spatial coordinates ...
        # get rid of "time" and "member" in list        
        varspacecoordnames = [item for item in vardimnames if item != 'time' and item != 'member'] 
        nbspacecoords = len(varspacecoordnames)

        if nbspacecoords == 0: # data => simple time series
            vartype = '1D:time series'
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
                print 'Cannot handle this variable yet! 2D variable of unrecognized dimensions... Exiting!'
                exit(1)
        else:
            print 'Cannot handle this variable yet! To many dimensions... Exiting!'
            exit(1)

        # data array
        data = data.variables[var_to_extract][:]
        print data.shape

        # if 2D:horizontal variable, check grid & standardize grid orientation to lat=>[-90,90] & lon=>[0,360] if needed
        if vartype == '2D:horizontal':
            # which dim is lat & which is lon?
            indlat = spacecoords.index('lat')
            indlon = spacecoords.index('lon')
            print 'indlat=', indlat, ' indlon=', indlon

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
            if vartype == '1D:time series':
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

        print 'Removing the temporal mean (for every gridpoint) from the prior...'
        climo = np.nanmean(value,axis=0)
        value = (value - climo)
        print var_to_extract, ': Global: mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value)

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

