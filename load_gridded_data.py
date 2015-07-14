


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

    from netCDF4 import Dataset
    from datetime import datetime, timedelta
    import numpy as np
    import os.path

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

    from netCDF4 import Dataset
    from datetime import datetime, timedelta
    import numpy as np
    import os.path

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

    from netCDF4 import Dataset
    from datetime import datetime, timedelta
    import numpy as np
    import os.path

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

    from netCDF4 import Dataset
    from datetime import datetime, timedelta
    import numpy as np
    import os.path

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

    from netCDF4 import Dataset, date2num, num2date
    from datetime import datetime, timedelta
    import numpy as np
    import os.path
    import string

    datadict = {}

    # Loop over state variables to load
    for v in range(len(data_vars)):
        vardef = data_vars[v]
        data_file_read = string.replace(data_file,'[vardef_template]',vardef)
        
        # Check if file exists
        infile = data_dir+'/ccsm4_last_mil/'+data_file_read
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
        time_yrs = num2date(time[:],units=time.units,calendar=time.calendar)
        time_yrs_list = time_yrs.tolist()

        # To convert monthly data to annual: List years available in dataset and sort
        years_all = [int(time_yrs_list[i].strftime('%Y')) for i in range(0,len(time_yrs_list))]
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

        # Transform longitudes from [-180,180] domain to [0,360] domain if needed
        if vartype == '2D:horizontal':
            # which dim is lon?
            indlon = spacecoords.index('lon')
            if indlon == 0:
                vartmp = spacevar1
            elif indlon == 1:
                vartmp = spacevar2
            indneg = np.where(vartmp < 0)[0]
            if len(indneg) > 0: # if non-empty
                vartmp[indneg] = 360.0 + vartmp[indneg]
            # Back into right array
            if indlon == 0:
                spacevar1 = vartmp
            elif indlon == 1:
                spacevar2 = vartmp
            
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
                value[i] = np.nanmean(data.variables[var_to_extract][ind],axis=0)    
            elif '2D' in vartype: 
                if nbdims > 3:
                    value[i,:,:] = np.nanmean(np.squeeze(data.variables[var_to_extract][ind]),axis=0)
                else:
                    value[i,:,:] = np.nanmean(data.variables[var_to_extract][ind],axis=0)

        # Model data, so need to standardize (i.e. calculate anomalies)
        #print 'Standardizing the prior...'
        #print 'mean=', np.nanmean(value), ' std-dev=', np.nanstd(value)
        #value = (value - np.nanmean(value))/np.nanstd(value)

        print 'Removing the temporal mean (for every gridpoint) from the prior...'
        value = (value - np.nanmean(value,axis=0))
        print var_to_extract, ': Global: mean=', np.nanmean(value), ' , std-dev=', np.nanstd(value)

        # Dictionary of dictionaries
        # ex. data access : datadict['tas_sfc_Amon']['years'] => arrays containing years of the 'tas' data
        #                   datadict['tas_sfc_Amon']['value'] => array of 'tas' data values etc ...
        d = {}
        d['vartype'] = vartype
        d['years']   = time_yrs
        d['value']   = value
        d['spacecoords'] = spacecoords
        if '2D' in vartype:
            d[spacecoords[0]] = spacevar1
            d[spacecoords[1]] = spacevar2

        datadict[vardef] = d


    return datadict

#==========================================================================================
