
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
    proxy_types = proxy_definition.keys()

    # check if dealing with with "order" digits or not in definition of proxies
    try:
        proxy_types_unordered = [i.split(':', 1)[1] for i in proxy_definition.keys()]
    except:
        proxy_types_unordered = proxy_types

    for t in proxy_types:
        proxy_list[t] = []
        sites_assim[t] = []
        sites_eval[t] = []

    proxy_category = [item.split('_')[0] for item in proxy_types_unordered]

    # Define name of file & open
    proxy_file = datadir_proxy + '/'+datafile_proxy;
    print 'Reading metadata file: ', proxy_file
    workbook = xlrd.open_workbook(proxy_file);

    # ====================
    # Read in the metadata
    # ====================
    metadata = workbook.sheet_by_name('Metadata');
    # Get columns headers
    meta_fields = [metadata.cell(0,col_index).value for col_index in xrange(metadata.ncols)];    
    proxy_metadata = []; # dict list containing proxy metadata
    for row_index in xrange(1,metadata.nrows):
        d = {meta_fields[col_index]: metadata.cell(row_index, col_index).value
             for col_index in xrange(metadata.ncols)};
        proxy_metadata.append(d)

    # =================================================================
    # Restrict to proxy_region and proxy_assim items listed in NAMELIST
    # =================================================================
    for row_index in xrange(0,metadata.nrows-1):
        if proxy_metadata[row_index]['PAGES 2k Region'] in proxy_region:
            if proxy_metadata[row_index]['Archive type'] in proxy_category:
                if proxy_metadata[row_index]['Resolution (yr)'] in proxy_resolution:
                    indt = [i for i, s in enumerate(proxy_definition) if proxy_metadata[row_index]['Archive type'] in s]
                    proxy_measurement = [proxy_definition[proxy_types[indt[k]]] for k in xrange(len(indt))]
                    indm = [i for i, s in enumerate(proxy_measurement) if proxy_metadata[row_index]['Proxy measurement'] in s]
                    if indm: 
                        indtype = indt[indm[0]]
                        # Add chronology ID to appropriate list in dictionary
                        proxy_list[proxy_types[indtype]].append(str(proxy_metadata[row_index]['PAGES ID']))                        

    return proxy_list



def create_proxy_lists_from_metadata_S1csv(datadir_proxy, datafile_proxy, proxy_region, proxy_resolution, \
                                           proxy_definition, proxy_frac):
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
    proxy_types = proxy_definition.keys()
    proxy_types_unordered = [i.split(':', 1)[1] for i in proxy_definition.keys()]

    for t in proxy_types:
        proxy_list[t] = []
        sites_assim[t] = []
        sites_eval[t] = []

    proxy_category = [item.split('_')[0] for item in proxy_types_unordered]


    # Define name of file & open
    proxy_file = datadir_proxy + '/'+datafile_proxy;
    print 'Reading metadata file: ', proxy_file
    workbook = xlrd.open_workbook(proxy_file);

    # ====================
    # Read in the metadata
    # ====================
    metadata = workbook.sheet_by_name('Metadata');
    # Get columns headers
    meta_fields = [metadata.cell(0,col_index).value for col_index in xrange(metadata.ncols)];    
    proxy_metadata = []; # dict list containing proxy metadata
    for row_index in xrange(1,metadata.nrows):
        d = {meta_fields[col_index]: metadata.cell(row_index, col_index).value
             for col_index in xrange(metadata.ncols)};
        proxy_metadata.append(d)

    # =================================================================
    # Restrict to proxy_region and proxy_assim items listed in NAMELIST
    # =================================================================
    for row_index in xrange(0,metadata.nrows-1):
        if proxy_metadata[row_index]['PAGES 2k Region'] in proxy_region:
            if proxy_metadata[row_index]['Archive type'] in proxy_category:
                if proxy_metadata[row_index]['Resolution (yr)'] in proxy_resolution:
                    indt = [i for i, s in enumerate(proxy_definition) if proxy_metadata[row_index]['Archive type'] in s]
                    proxy_measurement = [proxy_definition[proxy_types[indt[k]]] for k in xrange(len(indt))]
                    indm = [i for i, s in enumerate(proxy_measurement) if proxy_metadata[row_index]['Proxy measurement'] in s]
                    if indm: 
                        indtype = indt[indm[0]]
                        # Add chronology ID to appropriate list in dictionary
                        proxy_list[proxy_types[indtype]].append(str(proxy_metadata[row_index]['PAGES ID']))                        

    # Create lists of sites to assimilate / keep for recon. evaluation
    if proxy_frac < 1.0:
        # List all sites, regardless of proxy type
        mergedlist = []
        tmp = [proxy_list[x] for x in proxy_list]
        nbtype = len(tmp)

        for k in xrange(nbtype):
            mergedlist.extend(tmp[k])

        nbsites = len(mergedlist)
        nbsites_assim = int(nbsites*proxy_frac)
        # random selection over entire site list
        ind_assim = sample(range(0, nbsites), nbsites_assim)
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
    print 'Reading metadata file: ', proxy_file
    workbook = xlrd.open_workbook(proxy_file);

    # Read in the metadata
    metadata = workbook.sheet_by_name('Metadata');
    # Get columns headers
    meta_fields = [metadata.cell(0,col_index).value for col_index in xrange(metadata.ncols)];
    
    proxy_metadata = []; # dict list containing proxy metadata
    for row_index in xrange(1,metadata.nrows):
        d = {meta_fields[col_index]: metadata.cell(row_index, col_index).value
             for col_index in xrange(metadata.ncols)};
        proxy_metadata.append(d)

    # Restrict to proxy_region and proxy_assim items listed in NAMELIST
    proxy_type_to_assim = [];
    proxy_id_to_assim   = [];
    proxy_lat_to_assim  = [];
    proxy_lon_to_assim  = [];
    for row_index in xrange(0,metadata.nrows-1):
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
    meta_fields = [metadata.cell(0,col_index).value for col_index in xrange(metadata.ncols)];
    
    proxy_metadata = []; # dict list containing proxy metadata
    for row_index in xrange(1,metadata.nrows):
        d = {meta_fields[col_index]: metadata.cell(row_index, col_index).value
             for col_index in xrange(metadata.ncols)};
        proxy_metadata.append(d)

    # Restrict to proxy_region and proxy_assim items listed in NAMELIST
    proxy_type_to_assim = [];
    proxy_id_to_assim   = [];
    proxy_lat_to_assim  = [];
    proxy_lon_to_assim  = [];
    for row_index in xrange(0,metadata.nrows-1):
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
        tmp_headers = [data.cell(0,col_index).value for col_index in xrange(data.ncols)]
        data_headers = [str(item) for item in tmp_headers]; # getting rid of unicode encoding
        tmp_refs = [data.cell(1,col_index).value for col_index in xrange(data.ncols)]
        data_refs = [str(item) for item in tmp_refs] # getting rid of unicode encoding
        data_headers[0] = data_refs[0]; # correct tag for years

        # Column indices of proxy id's in proxy_id_to_assim list
        col_assim = [i for i, item in enumerate(data_headers) if item in proxy_id_to_assim]        
        if col_assim: # if non-empty list
            for row_index in xrange(2,data.nrows):
                for col_index in col_assim:
                    found = False
                    # associate metadata to data record
                    for meta_row_index in xrange(1,len(proxy_metadata)):                        
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
    time  = [proxy_data[k]['time'] for k in xrange(0,len(proxy_data))]
    value = [proxy_data[k]['value'] for k in xrange(0,len(proxy_data))]

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
    print 'Reading file: ', proxy_file
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
    meta_fields = [metadata.cell(0,col_index).value for col_index in xrange(metadata.ncols)];
    
    proxy_metadata = []; # dict list containing proxy metadata
    for row_index in xrange(1,metadata.nrows):
        d = {meta_fields[col_index]: metadata.cell(row_index, col_index).value
             for col_index in xrange(metadata.ncols)};
        proxy_metadata.append(d)

    # Restrict to proxy_region and proxy_assim items listed in NAMELIST
    proxy_type_to_assim = [];
    proxy_id_to_assim   = [];
    proxy_lat_to_assim  = [];
    proxy_lon_to_assim  = [];
    for row_index in xrange(0,metadata.nrows-1):
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
        tmp_headers = [data.cell(0,col_index).value for col_index in xrange(data.ncols)]
        data_headers = [str(item) for item in tmp_headers]; # getting rid of unicode encoding
        tmp_refs = [data.cell(1,col_index).value for col_index in xrange(data.ncols)]
        data_refs = [str(item) for item in tmp_refs] # getting rid of unicode encoding
        data_headers[0] = data_refs[0]; # correct tag for years

        # Column indices of proxy id's in proxy_id_to_assim list
        col_assim = [i for i, item in enumerate(data_headers) if item in proxy_id_to_assim]

        if col_assim: # if non-empty list
            for row_index in xrange(2,data.nrows):
                for col_index in col_assim:
                    found = False
                    # associate metadata to data record
                    for meta_row_index in xrange(1,len(proxy_metadata)):
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

    id    = [proxy_data[k]['id'] for k in xrange(0,len(proxy_data))]
    lat   = [proxy_data[k]['lat'] for k in xrange(0,len(proxy_data))]
    lon   = [proxy_data[k]['lon'] for k in xrange(0,len(proxy_data))]
    alt   = [proxy_data[k]['alt'] for k in xrange(0,len(proxy_data))]
    time  = [proxy_data[k]['time'] for k in xrange(0,len(proxy_data))]
    value = [proxy_data[k]['value'] for k in xrange(0,len(proxy_data))]

    return id, lat, lon, alt, time, value # should add more output here as we develop further
    #return proxy_data
