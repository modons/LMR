from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import os.path

#==========================================================================================
# 
# 
#========================================================================================== 


def read_gridded_data_GISTEMP(data_dir,data_file,data_vars):
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

    # -------------------------------------------------------------
    # Convert time from "nb of days from dateref" to absolute years 
    # -------------------------------------------------------------
    time_yrs = []
    for i in xrange(0,len(data.variables['time'][:])):
        time_yrs.append(dateref + timedelta(days=int(data.variables['time'][i])))

    # ------------------------------
    # Convert monthly data to annual
    # ------------------------------
    # List years available in dataset and sort
    years_all = []
    for i in xrange(0,len(time_yrs)):
        isotime = time_yrs[i].isoformat()
        years_all.append(int(isotime.split("-")[0]))
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years.sort # sort the list

    time_yrs  = np.empty(len(years), dtype=int)
    value = np.empty([len(years), len(lat), len(lon)], dtype=float)

    fillval = np.power(2,15)-1
    tmp = np.copy(data.variables['tempanomaly'])
    tmp[tmp == fillval] = np.NAN

    # Loop over years in dataset
    # TODO: AP finds indices corresponding to year and averages them, list comp.
    #       is somewhat inefficient (searches entire series even though it's sorted)
    for i in xrange(0,len(years)):        
        # find indices in time array where "years[i]" appear
        ind = [j for j, k in enumerate(years_all) if k == years[i]]
        time_yrs[i] = years[i]
        # ---------------------------------------
        # Calculate annual mean from monthly data
        # Note: data has dims [time,lat,lon]
        # ---------------------------------------
        #value[i,:,:] = np.nanmean(data.variables['tempanomaly'][ind],axis=0)
        value[i,:,:] = np.nanmean(tmp[ind],axis=0)

    return time_yrs, lat, lon, value

#==========================================================================================


def read_gridded_data_HadCRUT(data_dir,data_file,data_vars):
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
    time_yrs = []
    for i in xrange(0,len(data.variables['time'][:])):
        time_yrs.append(dateref + timedelta(days=int(data.variables['time'][i])))

    # ------------------------------
    # Convert monthly data to annual
    # ------------------------------
    # List years available in dataset and sort
    years_all = []
    for i in xrange(0,len(time_yrs)):
        isotime = time_yrs[i].isoformat()
        years_all.append(int(isotime.split("-")[0]))
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years.sort # sort the list

    time_yrs  = np.empty(len(years), dtype=int)
    value = np.empty([len(years), len(lat), len(lon)], dtype=float)

    tmp = np.copy(data.variables['temperature_anomaly'])
    tmp[tmp == -1e+30] = np.NAN

    # Loop over years in dataset
    for i in xrange(0,len(years)):        
        # find indices in time array where "years[i]" appear
        ind = [j for j, k in enumerate(years_all) if k == years[i]]
        time_yrs[i] = years[i]
        # ---------------------------------------
        # Calculate annual mean from monthly data
        # Note: data has dims [time,lat,lon]
        # ---------------------------------------
        #value[i,:,:] = np.nanmean(data.variables['temperature_anomaly'][ind],axis=0)
        value[i,:,:] = np.nanmean(tmp[ind],axis=0)



#    # ... test RT ... return monthly values ...
#    print '=>', np.min(value), np.max(value)
#    time_yrs  = np.empty(len(years_all), dtype=int)
#    value = np.empty([len(years_all), len(lat), len(lon)], dtype=float)
#    time_yrs = years_all
#    value = np.copy(data.variables['temperature_anomaly'])
#    value[value == -1e+30] = np.NAN
#    print '=>', np.nanmin(value), np.nanmax(value)
#    # ... test RT ... return monthly values ...


    return time_yrs, lat, lon, value

#==========================================================================================


def read_gridded_data_BerkeleyEarth(data_dir,data_file,data_vars):
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
    time_yrs = []
    for i in xrange(0,len(data.variables['time'][:])):
        
        yrAD = data.variables['time'][i]
        year = int(yrAD)
        rem = yrAD - year
        base = datetime(year, 1, 1)
        time_yrs.append(base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem))

    # ------------------------------
    # Convert monthly data to annual
    # ------------------------------
    # List years available in dataset and sort
    years_all = []
    for i in xrange(0,len(time_yrs)):
        isotime = time_yrs[i].isoformat()
        years_all.append(int(isotime.split("-")[0]))
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years.sort # sort the list

    time_yrs  = np.empty(len(years), dtype=int)
    value = np.empty([len(years), len(lat), len(lon)], dtype=float)

    fillval = data.variables['temperature'].missing_value
    tmp = np.copy(data.variables['temperature'])    
    tmp[tmp == fillval] = np.NAN
    # Loop over years in dataset
    for i in xrange(0,len(years)):        
        # find indices in time array where "years[i]" appear
        ind = [j for j, k in enumerate(years_all) if k == years[i]]
        time_yrs[i] = years[i]
        # ---------------------------------------
        # Calculate annual mean from monthly data
        # Note: data has dims [time,lat,lon]
        # ---------------------------------------
        #value[i,:,:] = np.nanmean(data.variables['temperature'][ind],axis=0)
        value[i,:,:] = np.nanmean(tmp[ind],axis=0)

    return time_yrs, lat, lon, value

#==========================================================================================


def read_gridded_data_NOAA(data_dir,data_file,data_vars):
#==========================================================================================
#
# Reads the monthly data of surface air temperature anomalies from the NOAA/NCDC gridded 
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

    # Check if file exists
    infile = data_dir+'/NOAA/'+data_file
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
    # Time is in "hours since 1800-1-1 0:0:0":convert to calendar years
    # -----------------------------------------------------------------
    time_yrs = []
    dateref = datetime(1800,1,1,0)
    for i in xrange(0,len(data.variables['time'][:])):
        time_yrs.append(dateref + timedelta(hours=int(data.variables['time'][i])))

    # ------------------------------
    # Convert monthly data to annual
    # ------------------------------
    # List years available in dataset and sort
    years_all = []
    for i in xrange(0,len(time_yrs)):
        isotime = time_yrs[i].isoformat()
        years_all.append(int(isotime.split("-")[0]))
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years.sort # sort the list

    time_yrs  = np.empty(len(years), dtype=int)
    value = np.empty([len(years), len(lat), len(lon)], dtype=float)

    fillval = np.power(2,15)-1
    tmp = np.copy(data.variables['data'])
    tmp[tmp == fillval] = np.NAN
    # Loop over years in dataset
    for i in xrange(0,len(years)):        
        # find indices in time array where "years[i]" appear
        ind = [j for j, k in enumerate(years_all) if k == years[i]]
        time_yrs[i] = years[i]
        # ---------------------------------------
        # Calculate annual mean from monthly data
        # Note: data has dims [time,lat,lon]
        # ---------------------------------------
        #value[i,:,:] = np.nanmean(data.variables['data'][ind],axis=0)
        value[i,:,:] = np.nanmean(tmp[ind],axis=0)

    return time_yrs, lat, lon, value


#==========================================================================================

def read_gridded_data_ccsm4_last_millenium(data_dir,data_file,data_vars):
#==========================================================================================
#
# Reads the monthly data of surface air temperature anomalies from the CCSM4 model
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

    # Check if file exists
    # TODO: AP why is the directory hard coded when we specify it in Namelist?
    infile = data_dir+'ccsm4_last_mil/'+data_file
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

    dateref = datetime(850,1,1,0)

    data = Dataset(infile,'r')

    lat   = data.variables['lat'][:]
    lon   = data.variables['lon'][:]

    # Transform longitudes from [-180,180] domain to [0,360] domain if needed
    indneg = np.where(lon < 0)[0]
    if len(indneg) > 0: # if non-empty
        lon[indneg] = 360.0 + lon[indneg]

    # -------------------------------------------------------------
    # Convert time from "nb of days from dateref" to absolute years 
    # -------------------------------------------------------------
    time_yrs = []
    for i in xrange(0,len(data.variables['time'][:])):
        time_yrs.append(dateref + timedelta(days=int(data.variables['time'][i])))

    # ------------------------------
    # Convert monthly data to annual
    # ------------------------------
    # List years available in dataset and sort
    years_all = []
    for i in xrange(0,len(time_yrs)):
        isotime = time_yrs[i].isoformat()
        years_all.append(int(isotime.split("-")[0]))
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years.sort # sort the list

    time_yrs  = np.empty(len(years), dtype=int)
    value = np.empty([len(years), len(lat), len(lon)], dtype=float)

    # Loop over years in dataset
    for i in xrange(0,len(years)):        
        # find indices in time array where "years[i]" appear
        ind = [j for j, k in enumerate(years_all) if k == years[i]]
        time_yrs[i] = years[i]
        # ---------------------------------------
        # Calculate annual mean from monthly data
        # Note: data has dims [time,lat,lon]
        # ---------------------------------------
        value[i,:,:] = np.nanmean(data.variables['tas'][ind],axis=0)


    # Model data, so need to standardize (i.e. calculate anomalies)
    #print 'Standardizing the prior...'
    #print 'mean=', np.nanmean(value), ' std-dev=', np.nanstd(value)
    #value = (value - np.nanmean(value))/np.nanstd(value)

    #print 'Removing the mean (global over entire length of experiment) from the prior...'
    #print 'mean=', np.nanmean(value), ' std-dev=', np.nanstd(value)
    #value = (value - np.nanmean(value))

    print 'Removing the temporal mean (for every gridpoint) from the prior...'
    value = (value - np.nanmean(value,axis=0))
    print 'Global: mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value)

    return time_yrs, lat, lon, value

#==========================================================================================
