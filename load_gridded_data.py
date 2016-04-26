
#==========================================================================================
# 
# 
#========================================================================================== 

from netCDF4 import Dataset, date2num, num2date
from datetime import datetime, timedelta
import numpy as np
import os.path
import string



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
    for i in range(0,len(time_yrs)):
        isotime = time_yrs[i].isoformat()
        years_all.append(int(isotime.split("-")[0]))
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years.sort # sort the list

    time_yrs  = np.empty(len(years), dtype=int)
    value = np.empty([len(years), len(lat), len(lon)], dtype=float)
    value[:] = np.nan # initialize with nan's

    fillval = np.power(2,15)-1
    cpy = np.copy(data.variables['tempanomaly'])
    cpy[cpy == fillval] = np.NAN
    # Loop over years in dataset
    for i in range(0,len(years)):        
        # find indices in time array where "years[i]" appear
        ind = [j for j, k in enumerate(years_all) if k == years[i]]
        time_yrs[i] = years[i]
        # ---------------------------------------
        # Calculate annual mean from monthly data
        # Note: data has dims [time,lat,lon]
        # ---------------------------------------
        tmp = np.nanmean(cpy[ind],axis=0)
        # apply check of max nb of nan values allowed
        nancount = np.isnan(cpy[ind]).sum(axis=0)
        tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
        value[i,:,:] = tmp

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
    time_yrs = []
    for i in xrange(0,len(data.variables['time'][:])):
        time_yrs.append(dateref + timedelta(days=int(data.variables['time'][i])))

    # ------------------------------
    # Convert monthly data to annual
    # ------------------------------
    # List years available in dataset and sort
    years_all = []
    for i in range(0,len(time_yrs)):
        isotime = time_yrs[i].isoformat()
        years_all.append(int(isotime.split("-")[0]))
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years.sort # sort the list

    time_yrs  = np.empty(len(years), dtype=int)
    value = np.empty([len(years), len(lat), len(lon)], dtype=float)
    value[:] = np.nan # initialize with nan's

    cpy = np.copy(data.variables['temperature_anomaly'])
    cpy[cpy == -1e+30] = np.NAN
    # Loop over years in dataset
    for i in range(0,len(years)):        
        # find indices in time array where "years[i]" appear
        ind = [j for j, k in enumerate(years_all) if k == years[i]]
        time_yrs[i] = years[i]
        # ---------------------------------------
        # Calculate annual mean from monthly data
        # Note: data has dims [time,lat,lon]
        # ---------------------------------------
        tmp = np.nanmean(cpy[ind],axis=0)
        # apply check of max nb of nan values allowed
        nancount = np.isnan(cpy[ind]).sum(axis=0)
        tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
        value[i,:,:] = tmp

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
    for i in range(0,len(time_yrs)):
        isotime = time_yrs[i].isoformat()
        years_all.append(int(isotime.split("-")[0]))
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years.sort # sort the list

    time_yrs  = np.empty(len(years), dtype=int)
    value = np.empty([len(years), len(lat), len(lon)], dtype=float)
    value[:] = np.nan # initialize with nan's

    fillval = data.variables['temperature'].missing_value
    cpy = np.copy(data.variables['temperature'])    
    cpy[cpy == fillval] = np.NAN
    # Loop over years in dataset
    for i in range(0,len(years)):        
        # find indices in time array where "years[i]" appear
        ind = [j for j, k in enumerate(years_all) if k == years[i]]
        time_yrs[i] = years[i]
        # ---------------------------------------
        # Calculate annual mean from monthly data
        # Note: data has dims [time,lat,lon]
        # ---------------------------------------
        tmp = np.nanmean(cpy[ind],axis=0)
        # apply check of max nb of nan values allowed
        nancount = np.isnan(cpy[ind]).sum(axis=0)
        tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
        value[i,:,:] = tmp

    return time_yrs, lat, lon, value

#==========================================================================================

def read_gridded_data_MLOST(data_dir,data_file,data_vars):
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
    time_yrs = []
    dateref = datetime(1800,1,1,0)
    for i in xrange(0,len(data.variables['time'][:])):
        time_yrs.append(dateref + timedelta(days=int(data.variables['time'][i])))

    # ------------------------------
    # Convert monthly data to annual
    # ------------------------------
    # List years available in dataset and sort
    years_all = []
    for i in range(0,len(time_yrs)):
        isotime = time_yrs[i].isoformat()
        years_all.append(int(isotime.split("-")[0]))
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years.sort # sort the list

    time_yrs  = np.zeros(len(years), dtype=int)
    value = np.zeros([len(years), len(lat), len(lon)], dtype=float)
    value[:] = np.nan # initialize with nan's
    
    fillval = data.variables['air'].missing_value
    cpy = np.copy(data.variables['air'])
    cpy[cpy == fillval] = np.NAN
    # Loop over years in dataset
    for i in range(0,len(years)):        
        # find indices in time array where "years[i]" appear
        ind = [j for j, k in enumerate(years_all) if k == years[i]]
        time_yrs[i] = years[i]
        # ---------------------------------------
        # Calculate annual mean from monthly data
        # Note: data has dims [time,lat,lon]
        # ---------------------------------------
        tmp = np.nanmean(cpy[ind],axis=0)
        # apply check of max nb of nan values allowed
        nancount = np.isnan(cpy[ind]).sum(axis=0)
        tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
        value[i,:,:] = tmp

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

    nbmaxnan = 0 # max nb of nan's allowed in calculation of annual average

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
    for i in range(0,len(time_yrs)):
        isotime = time_yrs[i].isoformat()
        years_all.append(int(isotime.split("-")[0]))
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years.sort # sort the list

    time_yrs  = np.empty(len(years), dtype=int)
    value = np.empty([len(years), len(lat), len(lon)], dtype=float)
    value[:] = np.nan # initialize with nan's

    fillval = np.power(2,15)-1
    cpy = np.copy(data.variables['data'])
    cpy[cpy == fillval] = np.NAN
    # Loop over years in dataset
    for i in range(0,len(years)):        
        # find indices in time array where "years[i]" appear
        ind = [j for j, k in enumerate(years_all) if k == years[i]]
        time_yrs[i] = years[i]
        # ---------------------------------------
        # Calculate annual mean from monthly data
        # Note: data has dims [time,lat,lon]
        # ---------------------------------------
        tmp = np.nanmean(cpy[ind],axis=0)
        # apply check of max nb of nan values allowed
        nancount = np.isnan(cpy[ind]).sum(axis=0)
        tmp[nancount > nbmaxnan] = np.nan # put nan back if nb of nan's in current year above threshold
        value[i,:,:] = tmp

    return time_yrs, lat, lon, value


#==========================================================================================

def read_gridded_data_CMIP5_model(data_dir,data_file,data_vars):
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

        # To convert monthly data to annual: List years available in dataset and sort
        years_all = [d.year for d in time_yrs_list]
        years     = list(set(years_all)) # 'set' is used to retain unique values in list
        years.sort() # sort the list
        time_yrs  = np.empty(len(years), dtype=int)

        # Query info on spatial coordinates ...
        # get rid of time in list        
        varspacecoordnames = [item for item in vardimnames if item != 'time'] 
        nbspacecoords = len(varspacecoordnames)

        #print vardimnames, nbspacecoords, dictdims

        if nbspacecoords == 0: # data => simple time series
            vartype = '1D:time series'
            value = np.empty([len(years)], dtype=float)            
            spacecoords = None
        elif ((nbspacecoords == 2) or (nbspacecoords == 3 and 'plev' in vardimnames and dictdims['plev'] == 1)): # data => 2D data
            # get rid of plev in list        
            varspacecoordnames = [item for item in varspacecoordnames if item != 'plev'] 
            spacecoords = (varspacecoordnames[0],varspacecoordnames[1])
            spacevar1 = data.variables[spacecoords[0]][:]
            spacevar2 = data.variables[spacecoords[1]][:]
            value = np.empty([len(years), len(spacevar1), len(spacevar2)], dtype=float)

            #print spacecoords
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
                    tmp = data[:,::-1,:]             
                else:
                    tmp = data[:,:,::-1] 
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
                #value[i] = np.nanmean(data.variables[var_to_extract][ind],axis=0)
                value[i] = np.nanmean(data[ind],axis=0)
            elif '2D' in vartype: 
                if nbdims > 3:
                    #value[i,:,:] = np.nanmean(np.squeeze(data.variables[var_to_extract][ind]),axis=0)
                    value[i,:,:] = np.nanmean(np.squeeze(data[ind]),axis=0)
                else:
                    #value[i,:,:] = np.nanmean(data.variables[var_to_extract][ind],axis=0)
                    value[i,:,:] = np.nanmean(data[ind],axis=0)


        # Model data, so need to standardize (i.e. calculate anomalies)
        #print 'Standardizing the prior...'
        #print 'mean=', np.nanmean(value), ' std-dev=', np.nanstd(value)
        #value = (value - np.nanmean(value))/np.nanstd(value)

        print 'Removing the temporal mean (for every gridpoint) from the prior...'
        climo = np.nanmean(value,axis=0)
        value = (value - climo)
        print var_to_extract, ': Global: mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value)

        # Dictionary of dictionaries
        # ex. data access : datadict['tas_sfc_Amon']['years'] => arrays containing years of the 'tas' data
        #                   datadict['tas_sfc_Amon']['value'] => array of 'tas' data values etc ...
        d = {}
        d['vartype'] = vartype
        d['years']   = time_yrs
        d['value']   = value
        d['climo']   = climo
        d['spacecoords'] = spacecoords
        if '2D' in vartype:
            d[spacecoords[0]] = spacevar1
            d[spacecoords[1]] = spacevar2

        datadict[vardef] = d


    return datadict

#==========================================================================================
