class EmptyError(Exception):
    print(Exception)

#==========================================================================================
#
# 
#========================================================================================== 

def read_proxy_metadata_S1csv(datadir_proxy, datafile_proxy, proxy_region, proxy_resolution, \
                              proxy_definition):
#==========================================================================================
#
# ... reads metadata worksheet from PAGES2K_DatabaseS1 dataset ...
# 
#========================================================================================== 

    import sys
    import numpy as np
    from random import sample

    # Library needed to read CSV file format
    xlrd_dir = '/home/disk/ekman/rtardif/nobackup/lib/pylibs/xlrd/xlrd/'
    sys.path.append(xlrd_dir)
    import xlrd

    # Parsing dictionary of proxy definitions
    proxy_list  = {}; # dict list containing proxy types and associated proxy id's (sites)
    sites_assim = {}
    sites_eval  = {}
    proxy_types = list(proxy_definition.keys())

    # check if dealing with with "order" digits or not in definition of proxies
    try:
        proxy_types_unordered = [i.split(':', 1)[1] for i in list(proxy_definition.keys())]
    except:
        proxy_types_unordered = proxy_types

    for t in proxy_types:
        proxy_list[t] = []
        sites_assim[t] = []
        sites_eval[t] = []

    proxy_category = [item.split('_')[0] for item in proxy_types_unordered]

    # Define name of file & open
    proxy_file = datadir_proxy + '/'+datafile_proxy;
    print('Reading metadata file: ', proxy_file)
    workbook = xlrd.open_workbook(proxy_file);

    # ====================
    # Read in the metadata
    # ====================
    metadata = workbook.sheet_by_name('Metadata');
    # Get columns headers
    meta_fields = [metadata.cell(0,col_index).value for col_index in range(metadata.ncols)];    
    proxy_metadata = []; # dict list containing proxy metadata
    for row_index in range(1,metadata.nrows):
        d = {meta_fields[col_index]: metadata.cell(row_index, col_index).value
             for col_index in range(metadata.ncols)};
        proxy_metadata.append(d)

    # =================================================================
    # Restrict to proxy_region and proxy_assim items listed in NAMELIST
    # =================================================================
    for row_index in range(0,metadata.nrows-1):
        if proxy_metadata[row_index]['PAGES 2k Region'] in proxy_region:
            if proxy_metadata[row_index]['Archive type'] in proxy_category:
                if proxy_metadata[row_index]['Resolution (yr)'] in proxy_resolution:
                    indt = [i for i, s in enumerate(proxy_definition) if proxy_metadata[row_index]['Archive type'] in s]
                    proxy_measurement = [proxy_definition[proxy_types[indt[k]]] for k in range(len(indt))]
                    indm = [i for i, s in enumerate(proxy_measurement) if proxy_metadata[row_index]['Proxy measurement'] in s]
                    if indm: 
                        indtype = indt[indm[0]]
                        # Add chronology ID to appropriate list in dictionary
                        proxy_list[proxy_types[indtype]].append(str(proxy_metadata[row_index]['PAGES ID']))                        

    return proxy_list



def create_proxy_lists_from_metadata_S1csv(datadir_proxy, datafile_proxy, proxy_region, proxy_resolution, \
                                           proxy_definition, proxy_frac, psm_data, psm_r_crit):
#==========================================================================================
#
# ... reads metadata worksheet from PAGES2K_DatabaseS1 dataset ...
# 
#========================================================================================== 

    import sys
    import numpy as np
    from random import sample

    # Library needed to read CSV file format
    xlrd_dir = '/home/disk/ekman/rtardif/nobackup/lib/pylibs/xlrd/xlrd/'
    sys.path.append(xlrd_dir)
    import xlrd

    # Parsing dictionary of proxy definitions
    proxy_list = {}; # dict list containing proxy types and associated proxy id's (sites)
    sites_assim = {}
    sites_eval = {}
    proxy_types = list(proxy_definition.keys())
    proxy_types_unordered = [i.split(':', 1)[1] for i in list(proxy_definition.keys())]

    for t in proxy_types:
        proxy_list[t] = []
        sites_assim[t] = []
        sites_eval[t] = []

    proxy_category = [item.split('_')[0] for item in proxy_types_unordered]


    # Define name of file & open
    proxy_file = datadir_proxy + '/'+datafile_proxy;
    print('Reading metadata file: ', proxy_file)
    workbook = xlrd.open_workbook(proxy_file);

    # ====================
    # Read in the metadata
    # ====================
    metadata = workbook.sheet_by_name('Metadata');
    # Get columns headers
    meta_fields = [metadata.cell(0,col_index).value for col_index in range(metadata.ncols)];    
    proxy_metadata = []; # dict list containing proxy metadata
    for row_index in range(1,metadata.nrows):
        d = {meta_fields[col_index]: metadata.cell(row_index, col_index).value
             for col_index in range(metadata.ncols)};
        proxy_metadata.append(d)

    # Restrict to proxy_region and proxy_assim items listed in NAMELIST
    for row_index in range(0,metadata.nrows-1):
        if proxy_metadata[row_index]['PAGES 2k Region'] in proxy_region:
            if proxy_metadata[row_index]['Archive type'] in proxy_category:
                if proxy_metadata[row_index]['Resolution (yr)'] in proxy_resolution:
                    indt = [i for i, s in enumerate(proxy_definition) if proxy_metadata[row_index]['Archive type'] in s]
                    proxy_measurement = [proxy_definition[proxy_types[indt[k]]] for k in range(len(indt))]
                    indm = [i for i, s in enumerate(proxy_measurement) if proxy_metadata[row_index]['Proxy measurement'] in s]
                    if indm: 
                        indtype = indt[indm[0]]
                        # Add chronology ID to appropriate list in dictionary
                        proxy_list[proxy_types[indtype]].append(str(proxy_metadata[row_index]['PAGES ID']))                        

    # =========================================================================
    # Filter list to retain sites with PSM calibration correlation > PSM_r_crit
    # =========================================================================
    if psm_data is not None:
        proxy_TypesSites_psm    = list(psm_data.keys())
        proxy_TypesSites_psm_ok = [t for t in proxy_TypesSites_psm if abs(psm_data[t]['PSMcorrel']) > psm_r_crit]
        proxy_list_ok = {}
        for t in list(proxy_list.keys()):
            proxy = t.split(':', 1)[1]
            list_ok = [proxy_TypesSites_psm_ok[k][1] for k in range(len(proxy_TypesSites_psm_ok)) if proxy_TypesSites_psm_ok[k][0] == proxy]
            proxy_list_ok[t] = list_ok

    else:
        proxy_list_ok = proxy_list

    # ================================================================
    # Create lists of sites to assimilate / keep for recon. evaluation
    # ================================================================
    if proxy_frac < 1.0:
        # List all sites, regardless of proxy type
        mergedlist = []
        tmp = [proxy_list_ok[x] for x in proxy_list_ok]
        nbtype = len(tmp)

        for k in range(nbtype):
            mergedlist.extend(tmp[k])

        nbsites = len(mergedlist)
        nbsites_assim = int(nbsites*proxy_frac)
        # random selection over entire site list
        ind_assim = sample(list(range(0, nbsites)), nbsites_assim)
        ind_eval = set(range(0,nbsites)) - set(ind_assim) # list indices of sites not chosen
        p_assim = [mergedlist[p] for p in ind_assim]
        p_eval = [mergedlist[p] for p in ind_eval]

        #ind = [i for i, s in enumerate(proxy_definition) if proxy_metadata[row_index]['Archive type'] in s]
        # Re-populate lists by proxy type        
        for t in proxy_types:
            inda = [i for i, s in enumerate(p_assim) if s in proxy_list_ok[t]]
            sites_assim[t] = [p_assim[k] for k in inda]
            inde = [i for i, s in enumerate(p_eval) if s in proxy_list_ok[t]]
            sites_eval[t] = [p_eval[k] for k in inde]
    else:
        sites_assim = proxy_list_ok
        # leave sites_eval list empty

    return sites_assim, sites_eval



def read_proxy_metadata_S1csv_old(datadir_proxy, datafile_proxy, proxy_region, proxy_resolution, \
                                  proxy_type, proxy_measurement):
#==========================================================================================
#
# ... reads metadata worksheet from PAGES2K_DatabaseS1 dataset ...
# 
#========================================================================================== 

    import sys
    import numpy as np

    # Library needed to read CSV file format
    xlrd_dir = '/home/disk/ekman/rtardif/nobackup/lib/pylibs/xlrd/xlrd/'
    sys.path.append(xlrd_dir)
    import xlrd

    # Uploading proxy data
    proxy_file = datadir_proxy + '/'+datafile_proxy;
    print('Reading metadata file: ', proxy_file)
    workbook = xlrd.open_workbook(proxy_file);

    # Read in the metadata
    metadata = workbook.sheet_by_name('Metadata');
    # Get columns headers
    meta_fields = [metadata.cell(0,col_index).value for col_index in range(metadata.ncols)];
    
    proxy_metadata = []; # dict list containing proxy metadata
    for row_index in range(1,metadata.nrows):
        d = {meta_fields[col_index]: metadata.cell(row_index, col_index).value
             for col_index in range(metadata.ncols)};
        proxy_metadata.append(d)

    # Restrict to proxy_region and proxy_assim items listed in NAMELIST
    proxy_type_to_assim = [];
    proxy_id_to_assim   = [];
    proxy_lat_to_assim  = [];
    proxy_lon_to_assim  = [];
    for row_index in range(0,metadata.nrows-1):
        if proxy_metadata[row_index]['PAGES 2k Region'] in proxy_region:
            if proxy_metadata[row_index]['Archive type'] in proxy_type:
                if proxy_metadata[row_index]['Proxy measurement'] in proxy_measurement:
                    if proxy_metadata[row_index]['Resolution (yr)'] in proxy_resolution:
                        proxy_id_to_assim.append(proxy_metadata[row_index]['PAGES ID'])
                        proxy_type_to_assim.append(proxy_metadata[row_index]['Archive type'])
                        proxy_lat_to_assim.append(proxy_metadata[row_index]['Lat (N)'])
                        proxy_lon_to_assim.append(proxy_metadata[row_index]['Lon (E)'])
    
    site_list = [str(item) for item in proxy_id_to_assim]; # getting rid of unicode
    site_lat  = proxy_lat_to_assim
    site_lon  = proxy_lon_to_assim

    return site_list, site_lat, site_lon


def read_proxy_data_S1csv_site(datadir_proxy, datafile_proxy, proxy_site):
#==========================================================================================
#
# ... reads data from a selected site (chronology) in PAGES2K_DatabaseS1 ... 
# ... site is passed as argument ...
# 
#========================================================================================== 

    import sys
    import numpy as np

    # Library needed to read CSV file format
    xlrd_dir = '/home/disk/ekman/rtardif/nobackup/lib/pylibs/xlrd/xlrd/'
    sys.path.append(xlrd_dir)
    import xlrd

    # Uploading proxy data
    proxy_file = datadir_proxy + '/'+datafile_proxy;
    #print 'Reading file: ', proxy_file
    workbook = xlrd.open_workbook(proxy_file);

    # Getting general (number & names of worksheets) info on file content
    nb_worksheets  = workbook.nsheets;
    #worksheet_list = workbook.sheet_names();
    worksheet_list = [str(item) for item in workbook.sheet_names()]; # getting rid of unicode
    
    # Create list of worksheet names containing data
    worksheet_list_data = worksheet_list;
    del worksheet_list[worksheet_list_data.index('ReadMe')]
    del worksheet_list[worksheet_list_data.index('Metadata')]

    # Read in the metadata
    metadata = workbook.sheet_by_name('Metadata');
    # Get columns headers
    meta_fields = [metadata.cell(0,col_index).value for col_index in range(metadata.ncols)];
    
    proxy_metadata = []; # dict list containing proxy metadata
    for row_index in range(1,metadata.nrows):
        d = {meta_fields[col_index]: metadata.cell(row_index, col_index).value
             for col_index in range(metadata.ncols)};
        proxy_metadata.append(d)

    # Restrict to proxy_region and proxy_assim items listed in NAMELIST
    proxy_type_to_assim = [];
    proxy_id_to_assim   = [];
    proxy_lat_to_assim  = [];
    proxy_lon_to_assim  = [];
    for row_index in range(0,metadata.nrows-1):
        if proxy_metadata[row_index]['PAGES ID'] in proxy_site:
            proxy_id_to_assim.append(proxy_metadata[row_index]['PAGES ID'])
            proxy_type_to_assim.append(proxy_metadata[row_index]['Archive type'])
            proxy_lat_to_assim.append(proxy_metadata[row_index]['Lat (N)'])
            proxy_lon_to_assim.append(proxy_metadata[row_index]['Lon (E)'])

    proxy_id_to_assim = [str(item) for item in proxy_id_to_assim]; # getting rid of unicode encoding
    proxy_type_to_assim = [str(item) for item in proxy_type_to_assim]; # getting rid of unicode encoding


    # ------------------------------------------
    # Loop over worksheets containing proxy data
    # ------------------------------------------
    # Dictionary containing proxy metadata & data
    proxy_data = {}
    nb_ob = -1
    for worksheet in worksheet_list_data:

        data = workbook.sheet_by_name(worksheet)
        num_cols = data.ncols - 1

        # Get columns headers
        tmp_headers = [data.cell(0,col_index).value for col_index in range(data.ncols)]
        data_headers = [str(item) for item in tmp_headers]; # getting rid of unicode encoding
        tmp_refs = [data.cell(1,col_index).value for col_index in range(data.ncols)]
        data_refs = [str(item) for item in tmp_refs] # getting rid of unicode encoding
        data_headers[0] = data_refs[0]; # correct tag for years

        # Column indices of proxy id's in proxy_id_to_assim list
        col_assim = [i for i, item in enumerate(data_headers) if item in proxy_id_to_assim]        
        if col_assim: # if non-empty list
            for row_index in range(2,data.nrows):
                for col_index in col_assim:
                    found = False
                    # associate metadata to data record
                    for meta_row_index in range(1,len(proxy_metadata)):                        
                        if proxy_metadata[meta_row_index]['PAGES ID'] == data_headers[col_index]:
                            found = True
                            typedat    = proxy_metadata[meta_row_index]['Archive type']
                            measure    = proxy_metadata[meta_row_index]['Proxy measurement']
                            resolution = proxy_metadata[meta_row_index]['Resolution (yr)']
                            lat        = proxy_metadata[meta_row_index]['Lat (N)']
                            lon        = proxy_metadata[meta_row_index]['Lon (E)']
                            alt        = 0.0 # no altitude info in data file
                            if lon < 0:
                                lon = 360 + lon
                            

                    if found:
                        if data.cell(row_index, col_index).value: # only keep those with non-empty values
                            nb_ob = nb_ob + 1
                            proxy_data[nb_ob] = {}
                            proxy_data[nb_ob]['id']    = data_headers[col_index]
                            proxy_data[nb_ob]['type']  = str(typedat)
                            proxy_data[nb_ob]['meas']  = str(measure)
                            proxy_data[nb_ob]['resol'] = resolution
                            proxy_data[nb_ob]['lat']   = lat
                            proxy_data[nb_ob]['lon']   = lon
                            proxy_data[nb_ob]['alt']   = alt
                            proxy_data[nb_ob]['time']  = data.cell(row_index, 0).value
                            proxy_data[nb_ob]['value'] = data.cell(row_index, col_index).value

    id    = proxy_data[0]['id']
    lat   = proxy_data[0]['lat']
    lon   = proxy_data[0]['lon']
    alt   = proxy_data[0]['alt']
    # proxy time series
    time  = [proxy_data[k]['time'] for k in range(0,len(proxy_data))]
    value = [proxy_data[k]['value'] for k in range(0,len(proxy_data))]

    return id, lat, lon, alt, time, value # could add more output here as we develop further
    #return proxy_data



def read_proxy_data_S1csv(self, datadir_proxy, datafile_proxy, proxy_region, proxy_type, proxy_measurement):
#==========================================================================================
#
# ... reads data from all sites (chronologies) in PAGES2K_DatabaseS1 dataset meeting 
#     selection criteria from NAMELIST ... 
# 
#========================================================================================== 

    import sys
    import numpy as np

    # Library needed to read CSV file format
    xlrd_dir = '/home/disk/ekman/rtardif/nobackup/lib/pylibs/xlrd/xlrd/'
    sys.path.append(xlrd_dir)
    import xlrd

    # Uploading proxy data
    proxy_file = datadir_proxy + '/'+datafile_proxy;
    print('Reading file: ', proxy_file)
    workbook = xlrd.open_workbook(proxy_file);

    # Getting general (number & names of worksheets) info on file content
    nb_worksheets  = workbook.nsheets;
    #worksheet_list = workbook.sheet_names();
    worksheet_list = [str(item) for item in workbook.sheet_names()]; # getting rid of unicode
    
    # Create list of worksheet names containing the data
    worksheet_list_data = worksheet_list;
    del worksheet_list[worksheet_list_data.index('ReadMe')]
    del worksheet_list[worksheet_list_data.index('Metadata')]

    # Read in the metadata
    metadata = workbook.sheet_by_name('Metadata');
    # Get columns headers
    meta_fields = [metadata.cell(0,col_index).value for col_index in range(metadata.ncols)];
    
    proxy_metadata = []; # dict list containing proxy metadata
    for row_index in range(1,metadata.nrows):
        d = {meta_fields[col_index]: metadata.cell(row_index, col_index).value
             for col_index in range(metadata.ncols)};
        proxy_metadata.append(d)

    # Restrict to proxy_region and proxy_assim items listed in NAMELIST
    proxy_type_to_assim = [];
    proxy_id_to_assim   = [];
    proxy_lat_to_assim  = [];
    proxy_lon_to_assim  = [];
    for row_index in range(0,metadata.nrows-1):
        if proxy_metadata[row_index]['PAGES 2k Region'] in proxy_region:
            if proxy_metadata[row_index]['Archive type'] in proxy_type:
                if proxy_metadata[row_index]['Proxy measurement'] in proxy_measurement:
                    proxy_id_to_assim.append(proxy_metadata[row_index]['PAGES ID'])
                    proxy_type_to_assim.append(proxy_metadata[row_index]['Archive type'])
                    proxy_lat_to_assim.append(proxy_metadata[row_index]['Lat (N)'])
                    proxy_lon_to_assim.append(proxy_metadata[row_index]['Lon (E)'])

    # Loop over worksheets containing proxy data
    # dictionary containing proxy metadata & data
    proxy_data = {}
    nb_ob = -1
    for worksheet in worksheet_list_data:

        #print 'worksheet: ', worksheet

        data = workbook.sheet_by_name(worksheet)
        num_cols = data.ncols - 1

        # Get columns headers
        tmp_headers = [data.cell(0,col_index).value for col_index in range(data.ncols)]
        data_headers = [str(item) for item in tmp_headers]; # getting rid of unicode encoding
        tmp_refs = [data.cell(1,col_index).value for col_index in range(data.ncols)]
        data_refs = [str(item) for item in tmp_refs] # getting rid of unicode encoding
        data_headers[0] = data_refs[0]; # correct tag for years

        # Column indices of proxy id's in proxy_id_to_assim list
        col_assim = [i for i, item in enumerate(data_headers) if item in proxy_id_to_assim]

        if col_assim: # if non-empty list
            for row_index in range(2,data.nrows):
                for col_index in col_assim:
                    found = False
                    # associate metadata to data record
                    for meta_row_index in range(1,len(proxy_metadata)):
                        if proxy_metadata[meta_row_index]['PAGES ID'] == data_headers[col_index]:
                            found = True
                            typedat    = proxy_metadata[meta_row_index]['Archive type']
                            measure    = proxy_metadata[meta_row_index]['Proxy measurement']
                            resolution = proxy_metadata[meta_row_index]['Resolution (yr)']
                            lat        = proxy_metadata[meta_row_index]['Lat (N)']
                            lon        = proxy_metadata[meta_row_index]['Lon (E)']
                            alt        = 0.0 # no altitude info in data file
                
                    if found:
                        if data.cell(row_index, col_index).value: # only keep those with non-empty values
                            nb_ob = nb_ob + 1
                            proxy_data[nb_ob] = {}
                            proxy_data[nb_ob]['id']    = data_headers[col_index]
                            proxy_data[nb_ob]['type']  = str(typedat)
                            proxy_data[nb_ob]['meas']  = str(measure)
                            proxy_data[nb_ob]['resol'] = resolution
                            proxy_data[nb_ob]['lat']   = lat
                            proxy_data[nb_ob]['lon']   = lon
                            proxy_data[nb_ob]['alt']   = alt
                            proxy_data[nb_ob]['time']  = data.cell(row_index, 0).value
                            proxy_data[nb_ob]['value'] = data.cell(row_index, col_index).value

    id    = [proxy_data[k]['id'] for k in range(0,len(proxy_data))]
    lat   = [proxy_data[k]['lat'] for k in range(0,len(proxy_data))]
    lon   = [proxy_data[k]['lon'] for k in range(0,len(proxy_data))]
    alt   = [proxy_data[k]['alt'] for k in range(0,len(proxy_data))]
    time  = [proxy_data[k]['time'] for k in range(0,len(proxy_data))]
    value = [proxy_data[k]['value'] for k in range(0,len(proxy_data))]

    return id, lat, lon, alt, time, value # should add more output here as we develop further
    #return proxy_data


#==========================================================================================
#
# 
#========================================================================================== 


# =========================================================================================
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

# =========================================================================================
def create_proxy_lists_from_metadata_NCDC(datadir_proxy, datafile_proxy, proxy_resolution, \
                                          proxy_definition, proxy_frac):

#==========================================================================================
#
# ... reads metadata worksheet for NCDC formatted proxy dataset ...
# 
#========================================================================================== 

    import sys
    import numpy as np
    from random import sample

    # NEED TO THINK OF SOMETHING ELSE HERE... ... ... ... ... ... ... ... ...
    # ... provide this library as part of LMR distribution?
    # Library needed to read CSV file format
    xlrd_dir = '/home/disk/ekman/rtardif/nobackup/lib/pylibs/xlrd/xlrd/'
    sys.path.append(xlrd_dir)
    import xlrd

    # Parsing dictionary of proxy definitions
    proxy_list = {}; # dict list containing proxy types and associated proxy id's (sites)
    sites_assim = {}
    sites_eval = {}
    proxy_types = list(proxy_definition.keys())
    proxy_types_unordered = [i.split(':', 1)[1] for i in list(proxy_definition.keys())]

    for t in proxy_types:
        proxy_list[t]  = []
        sites_assim[t] = []
        sites_eval[t]  = []

    proxy_category = [item.split('_')[0] for item in proxy_types_unordered]

    # Define name of file & open
    proxy_file = datadir_proxy + '/'+datafile_proxy;
    print('Reading metadata file: ', proxy_file)
    workbook = xlrd.open_workbook(proxy_file);

    # ====================
    # Read in the metadata
    # ====================
    metadata = workbook.sheet_by_name('Master Metadata File');
    # Get columns headers
    meta_fields = [metadata.cell(0,col_index).value for col_index in range(metadata.ncols)];    
    proxy_metadata = []; # dict list containing proxy metadata
    for row_index in range(1,metadata.nrows):
        d = {meta_fields[col_index]: metadata.cell(row_index, col_index).value
             for col_index in range(metadata.ncols)};
        proxy_metadata.append(d)

    # =================================================================
    # Restrict to proxy_assim items listed in NAMELIST
    # =================================================================

    for row_index in range(0,metadata.nrows-1):
        if proxy_metadata[row_index]['Archive'] in proxy_category:
            if proxy_metadata[row_index]['Resolution'] in proxy_resolution:
                indt = [i for i, s in enumerate(proxy_definition) if proxy_metadata[row_index]['Archive'] in s]
                proxy_measurement = [proxy_definition[proxy_types[indt[k]]] for k in range(len(indt))]
                l1 = proxy_metadata[row_index]['Variable Short Names'].split(",")
                l2 = [item.strip("[").strip("]").strip("'").strip().strip("'") for item in l1] # clean the crud...
                l3 = [str(l2[k]) for k in range(len(l2))]

                # Common elements in lists?
                for indm in range(len(proxy_measurement)):
                    common_set = set(l3)&set(proxy_measurement[indm])
                    if common_set: # if common element has been found
                        indtype = indt[indm]
                        # Add chronology ID to appropriate list in dictionary
                        # Do a check on consistency between 'Unique Identifier' & 'Filename.txt' ... sometimes doesn't match!
                        siteid_from_filename = proxy_metadata[row_index]['Filename.txt'][:-4] # strip the '.txt'
                        if str(proxy_metadata[row_index]['Unique Identifier']) != siteid_from_filename:
                            print('Filename & Unique Identifier DO NOT MATCH: using filename instead ...', siteid_from_filename, \
                                'vs', str(proxy_metadata[row_index]['Unique Identifier']))
                            proxy_list[proxy_types[indtype]].append(str(siteid_from_filename))
                        else:
                            proxy_list[proxy_types[indtype]].append(str(proxy_metadata[row_index]['Unique Identifier']))


    # Create lists of sites to assimilate / keep for recon. evaluation
    if proxy_frac < 1.0:
        # List all sites, regardless of proxy type
        mergedlist = []
        tmp = [proxy_list[x] for x in proxy_list]
        nbtype = len(tmp)

        for k in range(nbtype):
            mergedlist.extend(tmp[k])

        nbsites = len(mergedlist)
        nbsites_assim = int(nbsites*proxy_frac)
        # random selection over merged site list
        ind_assim = sample(list(range(0, nbsites)), nbsites_assim)
        ind_eval = set(range(0,nbsites)) - set(ind_assim) # list indices of sites not chosen
        p_assim = [mergedlist[p] for p in ind_assim]
        p_eval = [mergedlist[p] for p in ind_eval]

        #ind = [i for i, s in enumerate(proxy_definition) if proxy_metadata[row_index]['Archive type'] in s]
        # Re-populate lists by proxy type        
        for t in proxy_types:
            inda = [i for i, s in enumerate(p_assim) if s in proxy_list[t]]
            sites_assim[t] = [p_assim[k] for k in inda]
            inde = [i for i, s in enumerate(p_eval) if s in proxy_list[t]]
            sites_eval[t] = [p_eval[k] for k in inde]
    else:
        sites_assim = proxy_list
        # leave sites_eval list empty



#    print ' '
#    for t in proxy_types:
#        print t, proxy_list[t]
#        print ' '

    print('Assim:', sites_assim)
    print(' ')
    print('Eval:', sites_eval)


    return sites_assim, sites_eval


# =========================================================================================
def colonReader(string, fCon, fCon_low, end):
    '''This function seeks a specified string (or list of strings) within
    the transcribed file fCon (lowercase version fCon_low) until a specified
    character (typically end of the line) is found.x
    If a list of strings is provided, make sure they encompass all possibilities

    From Julien Emile-Geay (Univ. of Southern California)
    '''

    if isinstance(string, str):
        lstr = string + ': ' # append the annoying stuff
        Index = fCon_low.find(lstr)
        Len = len(lstr)

        if Index != -1:
            endlIndex = fCon_low[Index:].find(end)
            rstring = fCon[Index+Len:Index+endlIndex]  # returned string
            if rstring[-1:] == '\r':  # strip the '\r' character if it appears
                rstring = rstring[:-1]
            return rstring.strip()
        else:
            #print "Error: property " + string + " not found"           
            return ""
    else:
        num_str = len(string)
        rstring = "" # initialize returned string

        for k in range(0,num_str):  # loop over possible strings
            lstr = string[k] + ': ' # append the annoying stuff  
            Index = fCon_low.find(lstr)
            Len = len(lstr)
            if Index != -1:
                endlIndex = fCon_low[Index:].find(end)
                rstring = fCon[Index+Len:Index+endlIndex]
                if rstring[-1:] == '\r':  # strip the '\r' character if it appears
                    rstring = rstring[:-1]

        if rstring == "":
            #print "Error: property " + string[0] + " not found"           
            return ""
        else:
            return rstring.strip()

# =========================================================================================


def read_proxy_data_NCDCtxt_site(datadir, site, measurement):
#==========================================================================================
# Purpose: Reads data from a selected site (chronology) in NCDC proxy dataset
# 
# Input   :
#      - datadir     : Directory where proxy data files are located.
#      - site        : Site ID (ex. 00aust01a)
#      - measurement : List of possible proxy measurement labels for specific proxy type
#        (ex. ['d18O','d18o','d18o_stk','d18o_int','d18o_norm'] for delta 18 oxygen isotope
#        measurements)
#
# Returns :
#      - id      : Site id read from the data file
#      - lat/lon : latitude & longitude of the site
#      - alt     : Elevation of the site
#      - time    : Array containing the time of uploaded data  
#      - value   : Array of uploaded proxy data
#
# Author(s): Robert Tardif, Univ. of Washington, Dept. of Atmospheric Sciences 
#            based on "ncdc_file_parser.py" code from Julien Emile-Geay 
#            (Univ. of Southern California)
#
# Date     : March 2015
#
# Revision : None
# 
#========================================================================================== 
    
    import os
    import numpy as np
    

    # Possible header definitions of time in data files ...
    time_defs = ['age','Age_AD','age_AD','age_AD_ass','age_AD_int','Midpt_year',\
                     'age_yb1950','yb_1950','yrb_1950',\
                     'yb_1989','age_yb1989',\
                     'yr_b2k','yb_2k','ky_b2k','kyb_2k','kab2k','ka_b2k','ky_BP','kyr_BP','ka_BP','age_kaBP',\
                     'yr_BP','calyr_BP','Age(yrBP)','age_calBP']

    filename = datadir+'/'+site+'.txt'

    if os.path.isfile(filename):

        print('File:', filename)

        # Define root string for filename
        file_s   = filename.replace(" ", '_')  # strip all whitespaces if present
        fileroot = '_'.join(file_s.split('.')[:-1])

        # Open the file and port content to a string object
        filein      = open(filename,'U') # use the "universal newline mode" (U) to handle DOS formatted files
        fileContent = filein.read()
        fileContent_low = fileContent.lower()

        # Initialize empty dictionary
        d = {}

        # Assign default values to some metadata  
        d['ElevationUnit'] = 'm'
        d['TimeUnit']      = 'y_ad'

        # note: 8240/2030 ASCII code for "permil"

        # ===========================================================================
        # Extract metadata from file 
        # ===========================================================================
        try:
            # 'Archive' is the proxy type
            d['Archive']              = colonReader('archive', fileContent, fileContent_low, '\n')
            # Other info
            d['Title']                = colonReader('study_name', fileContent, fileContent_low, '\n')
            investigators             = colonReader('investigators', fileContent, fileContent_low, '\n')
            d['Investigators']        = investigators.replace(';',' and') # take out the ; so that turtle doesn't freak out.
            d['PubDOI']               = colonReader('doi', fileContent, fileContent_low, '\n')
            d['SiteName']             = colonReader('site_name', fileContent, fileContent_low, '\n')
            str_lst = ['northernmost_latitude', 'northernmost latitude'] # documented instances of this field property
            d['NorthernmostLatitude'] = float(colonReader(str_lst, fileContent, fileContent_low, '\n'))  
            str_lst = ['southernmost_latitude', 'southernmost latitude'] # documented instances of this field property
            d['SouthernmostLatitude'] = float(colonReader(str_lst, fileContent, fileContent_low, '\n'))
            str_lst = ['easternmost_longitude', 'easternmost longitude'] # documented instances of this field property
            d['EasternmostLongitude'] = float(colonReader(str_lst, fileContent, fileContent_low, '\n'))
            str_lst = ['westernmost_longitude', 'westernmost longitude'] # documented instances of this field property
            d['WesternmostLongitude'] = float(colonReader(str_lst, fileContent, fileContent_low, '\n'))
            elev = colonReader('elevation', fileContent, fileContent_low, '\n')
            if elev != 'nan' and len(elev)>0:
                elev_s = elev.split(' ')
                d['Elevation'] = float(''.join(c for c in elev_s[0] if c.isdigit())) # to only keep digits ...            
            else:   
                d['Elevation'] = float('NaN')

            d['CollectionName']       = colonReader('collection_name', fileContent, fileContent_low, '\n')
            d['EarliestYear']         = float(colonReader('earliest_year', fileContent, fileContent_low, '\n'))
            d['MostRecentYear']       = float(colonReader('most_recent_year', fileContent, fileContent_low, '\n'))
            d['TimeUnit']             = colonReader('time_unit', fileContent, fileContent_low, '\n')
            if not d['TimeUnit']:
                d['TimeUnit']         = colonReader('time unit', fileContent, fileContent_low, '\n')

        except EmptyError as e:
            print(e)

        # ===========================================================================
        # Extract information from the "Variables" section of the file
        # ===========================================================================

        # Find beginning of block
        sline_begin = fileContent.find('# Variables:')
        if sline_begin == -1:
            sline_begin = fileContent.find('# Variables')
        # Find end of block
        sline_end = fileContent.find('# Data:')
        if sline_end == -1:
            sline_end = fileContent.find('# Data\n')

        VarDesc = fileContent[sline_begin:sline_end].splitlines()
        nvar = 0 # counter for variable number
        for line in VarDesc:  # handle all the NCDC convention changes
            # (TODO: more clever/general exception handling)
            if line and line[0] != '' and line[0] != ' ' and line[0:2] != '#-' and line[0:2] != '# ' and line != '#':
                #print line
                nvar = nvar + 1
                line2 = line.replace('\t',',') # clean up
                sp_line = line2.split(',')     # split line along commas
                if len(sp_line) < 9:
                    continue
                else:
                    d['DataColumn' + format(nvar, '02') + '_ShortName']   = sp_line[0].strip('#').strip(' ')
                    d['DataColumn' + format(nvar, '02') + '_LongName']    = sp_line[1]
                    d['DataColumn' + format(nvar, '02') + '_Material']    = sp_line[2]
                    d['DataColumn' + format(nvar, '02') + '_Uncertainty'] = sp_line[3]
                    d['DataColumn' + format(nvar, '02') + '_Units']       = sp_line[4]
                    d['DataColumn' + format(nvar, '02') + '_Seasonality'] = sp_line[5]
                    d['DataColumn' + format(nvar, '02') + '_Archive']     = sp_line[6]
                    d['DataColumn' + format(nvar, '02') + '_Detail']      = sp_line[7]
                    d['DataColumn' + format(nvar, '02') + '_Method']      = sp_line[8]
                    d['DataColumn' + format(nvar, '02') + '_CharOrNum']   = sp_line[9].strip(' ')


        # ===========================================================================
        # Extract the data from the "Data" section of the file
        # ===========================================================================

        # Find line number at beginning of data block
        sline = fileContent.find('# Data:')
        if sline == -1:
            sline = fileContent.find('# Data\n')

        fileContent_datalines = fileContent[sline:].splitlines()
        start_line_index = 0
        line_nb = 0
        for line in fileContent_datalines:  # skip lines without actual data
            #print line
            if not line or line[0]=='#' or line[0] == ' ':
                start_line_index += 1
            else:
                start_line_index2 = line_nb
                break

            line_nb +=1
        
        #print start_line_index, start_line_index2

        # Extract column descriptions (headers) of the data matrix    
        DataColumn_headers = fileContent_datalines[start_line_index].splitlines()[0].split('\t')
        # Strip possible blanks in column headers 
        DataColumn_headers = [item.strip() for item in  DataColumn_headers]
        nc = len(DataColumn_headers)

        #print '-:' + str(nvar) + ' variables identified in metadata'
        #print '-:' + str(nc) + ' columns in data matrix'    

        # Which column contains the important data (time & proxy values) to be extracted?
        time_list = []
        data_list = []

        # Time
        TimeColumn_ided = False
        TimeColumn_tag = list(set(DataColumn_headers).intersection(time_defs))
        if len(TimeColumn_tag) > 0:
            if len(TimeColumn_tag) == 1: # single match -> ok
                time_col_index = DataColumn_headers.index(', '.join(TimeColumn_tag))
                TimeColumn_ided = True
            else:
                print('TimeColumn: More than one match ...do what then?')                

        # Proxy data
        DataColumn_ided = False
        DataColumn_tag = list(set(DataColumn_headers).intersection(measurement))
        if len(DataColumn_tag) > 0:
            if len(DataColumn_tag) == 1: # single match -> ok
                data_col_index = DataColumn_headers.index(', '.join(DataColumn_tag))
                DataColumn_ided = True
            else:
                print('DataColumn: More than one match ...do what then?')
                print('Taking first one...')
                DataColumn_tag.remove(DataColumn_tag[1])
                data_col_index = DataColumn_headers.index(', '.join(DataColumn_tag))
                DataColumn_ided = True

        # If both columns identified, then load arrays with the data
        if TimeColumn_ided and DataColumn_ided:
            datalines = fileContent_datalines[start_line_index+1:] # +1 to skip 1st line (header line)
            for line in datalines:
                datalist = line.split()
                # if line not empty
                if datalist:
                    try:
                        # If data not empty, not NaN & only digits -> OK then fill lists
                        if datalist and datalist[time_col_index] and datalist[data_col_index] and \
                                is_number(datalist[data_col_index]) and datalist[data_col_index].lower() != 'nan':
                            time_list.append(datalist[time_col_index])
                            data_list.append(datalist[data_col_index])
                    except:
                        continue
        
        # transform to numpy arrays => proxy time series
        time  = np.asarray(time_list,dtype=np.float64)
        value = np.asarray(data_list,dtype=np.float64)

        # proxy identifier and geo location
        id  = d['CollectionName']
        alt = d['Elevation']
    
        # Something crude in assignement of lat/lon:
        if d['NorthernmostLatitude'] != d['SouthernmostLatitude']:
            lat = (d['NorthernmostLatitude'] + d['SouthernmostLatitude'])/2.0
        else:
            lat = d['NorthernmostLatitude']
        if d['EasternmostLongitude'] != d['WesternmostLongitude']:
            lon = (d['EasternmostLongitude'] + d['WesternmostLongitude'])/2.0
        else:
            lon = d['EasternmostLongitude']

        # Modify "time" array into "years AD" if not already
        #print 'TimeUnit:', d['TimeUnit']
        tdef = d['TimeUnit']
        tdef_parsed = tdef.split('_')
        if len(tdef_parsed) == 2 and tdef_parsed[0] and tdef_parsed[1]:
            # tdef has expected structure ...
            if tdef_parsed[0] == 'yb' and is_number(tdef_parsed[1]):
                time = float(tdef_parsed[1]) - time
            elif tdef_parsed[0] == 'kyb' and is_number(tdef_parsed[1]):
                time = float(tdef_parsed[1]) - 1000.0*time
            elif tdef_parsed[0] == 'y' and tdef_parsed[1] == 'ad':
                pass # do nothing, time already in years_AD
            else:
                print('Unrecognized time definition. Returning empty arrays!')
                time  = np.asarray([],dtype=np.float64)
                value = np.asarray([],dtype=np.float64)                
        else:
            print('*** WARNING *** Unexpected time definition: string has more elements than expected. Returning empty arrays!')
            time  = np.asarray([],dtype=np.float64)
            value = np.asarray([],dtype=np.float64)

# Old code ... to handle the mishmash of time definitions ...
#        if tdef == 'kaB1950' or tdef == 'KYrBP' or tdef == 'kyr BP' or tdef == 'ky BP':
#            time = 1950.0 - 1000.0*time
#        elif tdef == 'yb 1950' or tdef == 'yb_1950' or tdef == 'yrb_1950' or tdef == 'age_yb1950' or tdef == 'years before 1950' \
#                or tdef == 'years before 1950 AD' or tdef == 'cal yr BP' or tdef == 'cal year BP' or tdef == 'yrs BP':
#            time = 1950.0 - time
#        elif tdef == 'yb 1989' or tdef == 'age_yb1989':
#            time = 1989.0 - time
#        elif tdef == 'years before 2000 AD' or tdef == 'yrs b2k' or tdef == 'yb 2k' or tdef == 'yrs b2k' or tdef == 'yrs b2k AD':
#            time = 2000.0 - time
#        elif tdef == 'ka b2k' or tdef == 'kyb 2k' or tdef == 'kyb_2k' or tdef == 'ky b2k':
#            time = 2000.0 - 1000.0*time
#        else:
#            # be careful: all else is considered years_AD
#            time = time



        # If subannual, average up to annual --------------------------------------------------------
        years_all = [int(time[k]) for k in range(0,len(time))]
        years = list(set(years_all)) # 'set' is used to get unique values in list
        years.sort() # sort the list
        time_annual  = np.asarray(years,dtype=np.float64)
        value_annual = np.empty(len(years), dtype=np.float64)
        # Loop over years in dataset
        for i in range(0,len(years)):     
            ind = [j for j, k in enumerate(years_all) if k == years[i]]
            value_annual[i] = np.nanmean(value[ind],axis=0)




        # Transform longitude in [0,360] domain
        if lon < 0:
            lon = 360 + lon

    else:
        print('File NOT FOUND:', filename)
        # return empty arrays
        id    = site
        lat   = []
        lon   = []
        alt   = []
        time  = np.asarray([],dtype=np.float64)
        value = np.asarray([],dtype=np.float64)

    #return id, lat, lon, alt, time, value
    return id, lat, lon, alt, time_annual, value_annual

