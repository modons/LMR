"""
Module: LMR_proxy_preprocess.py

Purpose: Takes proxy data in their native format (.xlsx file for PAGES or collection of
          .txt files for NCDC) and generates Pandas DataFrames stored in pickle files
          containing metadata and actual data from proxy records. The "pickled" DataFrames
          are used as input by the Last Millennium Reanalysis software.
          Currently, the data is stored as *annual averages*, for original records with
          subannual data.  
 
 Originator : Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
                            | January 2016
              (Based on code written by Andre Perkins (U. of Washington) to handle PAGES
              proxies)

Update 4/14/2017, Michael Erb: Code for the new PAGES2k has been added, mostly in three places:
    - A few additional import statements.
    - Some new code in "main" for the pages2k_v2 data.
    - A new function: pages2kv2_pickle_to_dataframes

"""
import glob
import numpy as np
import pandas as pd
from scipy import stats
import string
import re
import six
import ast
from os.path import join
import pickle
import gzip
import calendar
import gaussianize

# =========================================================================================

class EmptyError(Exception):
    print Exception

# =========================================================================================
# ---------------------------------------- MAIN -------------------------------------------
# =========================================================================================
def main():

    # ***************************************************************
    # Section for User-defined options: begin
    # 

    #proxy_data_source = 'PAGES2K'
    #proxy_data_source = 'NCDC'
    # version of the NCDC proxy db to process
    #dbversion = 'v0.0.0' 
    #dbversion = 'v0.1.0' 

    eliminate_duplicates = True

    # *** Begin pages2k v2.0.0 options ***
    proxy_data_source = 'PAGES2Kv2'

    # This option transforms all data to a Gaussian distribution.  It should only be used for
    # regressions, not physically-based PSMs.
    gaussianize_data = False
    
    # Specify the type of year to use for data averaging.  Currently only affects Pages2k v2 proxies.
    # "calendar year" (Jan-Dec) or "tropical year" (Apr-Mar)
    year_type = "calendar year"
    #year_type = "tropical year"
    # *** End pages2k v2.0.0 options ***

    # datadir: directory where the original proxy datafiles are located
    #datadir = '/home/chaos2/wperkins/data/LMR/proxies/'
    #datadir = '/home/disk/kalman3/rtardif/LMR/data/proxies/'
    datadir = '/home/scec-00/lmr/erbm/LMR/data/proxies/'
    
    # outdir: directory where the proxy database files will be created
    #         The piece before /data/proxies should correspond to your "lmr_path" set in LMR_config.py 
    #outdir  = '/home/disk/kalman3/rtardif/LMR/data/proxies/'
    outdir  = '/home/scec-00/lmr/erbm/LMR/data/proxies/'

    # 
    # Section for User-defined options: end
    # ***************************************************************

    if proxy_data_source == 'PAGES2K':
        # ============================================================================
        # PAGES2K proxy data ---------------------------------------------------------
        # ============================================================================

        take_average_out = False

        fname = datadir + 'Pages2k_DatabaseS1-All-proxy-records.xlsx'
        meta_outfile = outdir + 'Pages2k_Metadata.df.pckl'
        outfile = outdir + 'Pages2k_Proxies.df.pckl'
        pages_xcel_to_dataframes(fname, meta_outfile, outfile, take_average_out)


    elif proxy_data_source == 'PAGES2Kv2':
        # ============================================================================
        # PAGES2K v2.0.0 proxy data -------------------------------------------------------
        # ============================================================================
        
        if year_type == 'tropical year':
            if gaussianize_data == True:
                meta_outfile = outdir + 'NCDC_Pages2kv2_tropicalyear_gaussianized_Metadata.df.pckl'
                data_outfile = outdir + 'NCDC_Pages2kv2_tropicalyear_gaussianized_Proxies.df.pckl'
            else:
                meta_outfile = outdir + 'NCDC_Pages2kv2_tropicalyear_Metadata.df.pckl'
                data_outfile = outdir + 'NCDC_Pages2kv2_tropicalyear_Proxies.df.pckl'
        else:
            if gaussianize_data == True:
                meta_outfile = outdir + 'NCDC_Pages2kv2_gaussianized_Metadata.df.pckl'
                data_outfile = outdir + 'NCDC_Pages2kv2_gaussianized_Proxies.df.pckl'
            else:
                meta_outfile = outdir + 'NCDC_Pages2kv2_Metadata.df.pckl'
                data_outfile = outdir + 'NCDC_Pages2kv2_Proxies.df.pckl'

        pages2kv2_pickle_to_dataframes(datadir, meta_outfile, data_outfile, eliminate_duplicates, year_type, gaussianize_data)

        
    elif  proxy_data_source == 'NCDC':
        # ============================================================================
        # NCDC proxy data ------------------------------------------------------------
        # ============================================================================
        
        datadir = datadir+'NCDC/ToPandas_'+dbversion+'/'

        meta_outfile = outdir + 'NCDC_'+dbversion+'_Metadata.df.pckl'
        data_outfile = outdir + 'NCDC_'+dbversion+'_Proxies.df.pckl'

        # Specify all proxy types & associated proxy measurements to look for & extract from the data files
        # This is to take into account all the possible different names found in the NCDC data files.
        proxy_def = \
            {
                'Corals and Sclerosponges_d18O'   : ['d18O','delta18O','d18o','d18O_stk','d18O_int','d18O_norm','d18o_avg','d18o_ave','dO18','d18O_4'],\
                'Corals and Sclerosponges_SrCa'   : ['Sr/Ca','Sr_Ca','Sr/Ca_norm','Sr/Ca_anom','Sr/Ca_int'],\
                'Corals and Sclerosponges_Rates'  : ['ext','calc'],\
#                'Corals and Sclerosponges_d14C'   : ['d14C','d14c','ac_d14c'],\
#                'Corals and Sclerosponges_d13C'   : ['d13C','d13c','d13c_ave','d13c_ann_ave','d13C_int'],\
#                'Corals and Sclerosponges_Sr'     : ['Sr'],\
#                'Corals and Sclerosponges_BaCa'   : ['Ba/Ca'],\
#                'Corals and Sclerosponges_CdCa'   : ['Cd/Ca'],\
#                'Corals and Sclerosponges_MgCa'   : ['Mg/Ca'],\
#                'Corals and Sclerosponges_UCa'    : ['U/Ca','U/Ca_anom'],\
#                'Corals and Sclerosponges_Pb'     : ['Pb'],\
                'Ice Cores_d18O'                  : ['d18O','delta18O','delta18o','d18o','d18o_int','d18O_int','d18O_norm','d18o_norm','dO18','d18O_anom'],\
                'Ice Cores_dD'                    : ['deltaD','delD','dD'],\
                'Ice Cores_Accumulation'          : ['accum','accumu'],\
                'Ice Cores_MeltFeature'           : ['MFP'],\
                'Lake Cores_Varve'                : ['varve', 'varve_thickness', 'varve thickness'],\
                'Lake Cores_BioMarkers'           : ['Uk37', 'TEX86'],\
                'Lake Cores_GeoChem'              : ['Sr/Ca', 'Mg/Ca','Cl_cont'],\
                'Marine Cores_d18O'               : ['d18O'],\
#                'Speleothems_d18O'                : ['d18O'],\
#                'Speleothems_d13C'                : ['d13C'],\
                'Tree Rings_WidthBreit'           : ['trsgi'],\
                'Tree Rings_WidthPages2'          : ['trsgi'],\
                'Tree Rings_WidthPages'           : ['TRW','ERW','LRW'],\
                'Tree Rings_WoodDensity'          : ['max_d','min_d','early_d','earl_d','late_d','MXD','density'],\
                'Tree Rings_Isotopes'             : ['d18O'],\
                #'Climate Reconstructions'         : ['sst_ORSTOM','sss_ORSTOM','temp_anom'],\
            }

        ncdc_txt_to_dataframes(datadir, proxy_def, meta_outfile, data_outfile, eliminate_duplicates)

    else:
        print 'ERROR: Unkown proxy data source! Exiting!'
        exit(1)

# =========================================================================================
# ------------------------------------- END OF MAIN ---------------------------------------
# =========================================================================================


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


# ===================================================================================
# For PAGES2k S1 proxy data ---------------------------------------------------------
# ===================================================================================

def pages_xcel_to_dataframes(filename, metaout, dataout, take_average_out):
    """
    Takes in Pages2K CSV and converts it to dataframe storage.  This increases
    size on disk due to the joining along the time index (lots of null values).

    Makes it easier to query and grab data for the proxy experiments.

    :param filename:
    :param metaout:
    :param dataout:
    :return:

    Author: Andre Perkins, Univ. of Washington

    """

    meta_sheet_name = 'Metadata'
    metadata = pd.read_excel(filename, meta_sheet_name)
    metadata.to_pickle(metaout)

    record_sheet_names = ['AntProxies', 'ArcProxies', 'AsiaProxies',
                          'AusProxies', 'EurProxies', 'NAmPol', 'NAmTR',
                          'SAmProxies']

    for i, sheet in enumerate(record_sheet_names):
        tmp = pd.read_excel(filename, sheet)
        # for key, series in tmp.iteritems():
        #     h5store[key] = series[series.notnull()]

        if i == 0:
            df = tmp
        else:
            # SQL like table join along index
            df = df.merge(tmp, how='outer', on='PAGES 2k ID')

    #fix index and column name
    col0 = df.columns[0]
    newcol0 = df[col0][0]
    df.set_index(col0, drop=True, inplace=True)
    df.index.name = newcol0
    df = df.ix[1:]
    df.sort_index(inplace=True)

    if take_average_out:
        # copy of dataframe
        df_tmp = df                          
        # fill dataframe with new values where temporal averages over proxy records are subtracted
        df = df_tmp.sub(df_tmp.mean(axis=0), axis=1)
    
        
    # TODO: make sure year index is consecutive
    #write data to file
    df.to_pickle(dataout)


def compute_annual_means(time_raw,data_raw,valid_frac,year_type):
    """
    Computes annual-means from raw data.
    Inputs:
        time_raw   : Original time axis
        data_raw   : Original data
        valid_frac : The fraction of sub-annual data necessary to create annual mean.  Otherwise NaN.
        year_type  : "calendar year" (Jan-Dec) or "tropical year" (Apr-Mar)
    Outputs: time_annual, data_annual

    Authors: R. Tardif, Univ. of Washington; M. Erb, Univ. of Southern California

    """

    time_between_records = np.diff(time_raw, n=1)

    # Temporal resolution of the data, calculated as the mode of time difference.
    time_resolution = abs(stats.mode(time_between_records)[0][0])

    # check if time_resolution = 0.0 !!! sometimes adjacent records are tagged at same time ... 
    if time_resolution == 0.0:
        print '***WARNING! Found adjacent records with same times!'
        inderr = np.where(time_between_records == 0.0)
        print inderr
        time_between_records = np.delete(time_between_records,inderr)
        time_resolution = abs(stats.mode(time_between_records)[0][0])

    max_nb_per_year = int(1.0/time_resolution)

    if time_resolution <=1.0:
        proxy_resolution = int(1.0)
    else:
        proxy_resolution = int(time_resolution)

    # Get the integer values of all years.
    years_all = [int(time_raw[k]) for k in range(0,len(time_raw))]
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years = sorted(years) # sort the list
    years = np.insert(years,0,years[0]-1) # Add a year prior to the first year because of tropical year adjustments.

    time_annual = np.asarray(years,dtype=np.float64)
    data_annual = np.zeros(shape=[len(years)], dtype=np.float64)
    # fill with NaNs for default values
    data_annual[:] = np.NAN
        
    # If some of the time values are floats and year_type is tropical_year,
    # adjust the years to cover the tropical year (Apr-Mar).
    if np.equal(np.mod(time_raw,1),0).all() == False and year_type == 'tropical year':
        print "Tropical year method"

        # Make a variable for the time axis, adjusted to a tropical year
        time_adjusted = np.zeros(len(time_raw))
        time_adjusted[:] = np.nan

        for i in range(0,len(time_raw)):
            year_int = int(time_raw[i])
            if calendar.isleap(year_int):
                time_adjusted[i] = time_raw[i]-((31+29+31)/float(366))
            else:
                time_adjusted[i] = time_raw[i]-((31+28+31)/float(365))
                
        time_raw = time_adjusted
       
    # Calculate the mean of all data points with the same year.
    for i in range(len(years)): 
        ind = [j for j, k in enumerate(years_all) if k == years[i]]
        nbdat = len(ind)

        # TODO: check nb of non-NaN values !!!!! ... ... ... ... ... ...

        if time_resolution <= 1.0: 
            frac = float(nbdat)/float(max_nb_per_year)
            if frac > valid_frac:
                data_annual[i] = np.nanmean(data_raw[ind],axis=0)
        else:
            if nbdat > 1:
                print '***WARNING! Found multiple records in same year in data with multiyear resolution!'
                print '   year=', years[i], nbdat 
            # Note: this calculates the mean if multiple entries found
            data_annual[i] = np.nanmean(data_raw[ind],axis=0)

        #print years[i], nbdat, max_nb_per_year, data_annual[i,:]

    return time_annual, data_annual, proxy_resolution


# ===================================================================================
# For PAGES2k v2.0.0 proxy data ---------------------------------------------------------
# ===================================================================================

def pages2kv2_pickle_to_dataframes(datadir, metaout, dataout, eliminate_duplicates, year_type, gaussianize_data):
    """
    Takes in a Pages2k pckl file and converts it to dataframe storage.  
    Authors: R. Tardif, Univ. of Washington, Jan 2016.
             M. Erb, Univ. of Southern California, Feb 2017
    """

    # ===============================================================================
    # Upload proxy data from Pages2k v2 pickle file
    # ===============================================================================

    # Open the pickle file containing the Pages2k data
    f = gzip.open(datadir+'PAGES2k_v2.0.0_tempOnly.pklz','rb')
    pages2k_data = pickle.load(f)
    f.close()

    # ===============================================================================
    # Produce a summary of uploaded proxy data & 
    # generate integrated database in pandas DataFrame format
    # ===============================================================================

    # Summary of the final_proxy_list
    nbsites = len(pages2k_data)
    print '----------------------------------------------------------------------'
    print ' SUMMARY: '
    print '  Total nb of records : ', nbsites
    print '  ------------------------------------------------------'

    tot = []
    # Loop over proxy types specified in *main*
    counter = 0
    # Build up pandas DataFrame
    metadf  = pd.DataFrame()
    headers = ['NCDC ID','Site name','Lat (N)','Lon (E)','Elev','Archive type','Proxy measurement','Resolution (yr)',\
               'Oldest (C.E.)','Youngest (C.E.)','Location','climateVariable','Realm','Relation_to_climateVariable',\
               'Seasonality', 'Databases']

    nb = []
    for counter in range(0,len(pages2k_data)):
        
        #counter = 13 # An example of a sub-annual record

        # Give each record a unique descriptive name
        pages2k_data[counter]['siteID'] = "PAGES2kv2_"+pages2k_data[counter]['dataSetName']+"_"+pages2k_data[counter]['paleoData_pages2kID']+":"+pages2k_data[counter]['paleoData_variableName']
        nb.append(pages2k_data[counter]['siteID'])
        print "Processing metadata", counter+1, "/", len(pages2k_data), ":", pages2k_data[counter]['paleoData_pages2kID']

        # If the time axis goes backwards (i.e. newer to older), reverse it.
        if pages2k_data[counter]['year'][-1] - pages2k_data[counter]['year'][-2] < 0:
            pages2k_data[counter]['year'].reverse()
            pages2k_data[counter]['paleoData_values'].reverse()
            
        # If subannual, average up to annual --------------------------------------------------------
        time_raw = np.array(pages2k_data[counter]['year'],dtype=np.float)
        data_raw = np.array(pages2k_data[counter]['paleoData_values'],dtype=np.float)

        # Remove values where either time or data is nan. 
        nan_indices = np.isnan(time_raw)+np.isnan(data_raw)
        time_raw = time_raw[~nan_indices]
        data_raw = data_raw[~nan_indices]
                            
        valid_frac = 0.5

        # Use the following function to make annual-means.
        # Inputs: time_raw, data_raw, valid_frac, year_type.  Outputs: time_annual, data_annual
        time_annual, data_annual, proxy_resolution = compute_annual_means(time_raw,data_raw,valid_frac,year_type)

        # If gaussianize_data is set to true, transform the proxy data to Gaussian.
        # This option should only be used when using regressions, not physically-based PSMs.
        if gaussianize_data == True:
            data_annual = gaussianize.gaussianize(data_annual)

        # Write the annual data to the dictionary, so they can use written to
        # the data file outside of this loop.
        pages2k_data[counter]['time_annual'] = time_annual
        pages2k_data[counter]['data_annual'] = data_annual

        # Rename the proxy types in the same convention as the NCDC dataset.
        # Proxy types not renamed: bivalve, borehole, documents, hybrid
        if (pages2k_data[counter]['archiveType'] == 'coral') or (pages2k_data[counter]['archiveType'] == 'sclerosponge'):
            pages2k_data[counter]['archiveType'] = 'Corals and Sclerosponges'
        elif pages2k_data[counter]['archiveType'] == 'glacier ice':
            pages2k_data[counter]['archiveType'] = 'Ice Cores'
        elif pages2k_data[counter]['archiveType'] == 'lake sediment':
            pages2k_data[counter]['archiveType'] = 'Lake Cores'
        elif pages2k_data[counter]['archiveType'] == 'marine sediment':
            pages2k_data[counter]['archiveType'] = 'Marine Cores'
        elif pages2k_data[counter]['archiveType'] == 'speleothem':
            pages2k_data[counter]['archiveType'] = 'Speleothems'
        elif pages2k_data[counter]['archiveType'] == 'tree':
            pages2k_data[counter]['archiveType'] = 'Tree Rings'
                
        # Rename some of the the proxy measurements to be more standard.
        if (pages2k_data[counter]['archiveType'] == 'Ice Cores') and (pages2k_data[counter]['paleoData_variableName'] == 'd18O1'):
            pages2k_data[counter]['paleoData_variableName'] = 'd18O'
        elif (pages2k_data[counter]['archiveType'] == 'Lake Cores') and (pages2k_data[counter]['paleoData_variableName'] == 'temperature1'):
            pages2k_data[counter]['paleoData_variableName'] = 'temperature'
        elif (pages2k_data[counter]['archiveType'] == 'Lake Cores') and (pages2k_data[counter]['paleoData_variableName'] == 'temperature3'):
            pages2k_data[counter]['paleoData_variableName'] = 'temperature'

        # Not all records have data for elevation.  In these cases, set elevation to nan.
        if 'geo_meanElev' not in pages2k_data[counter]:
            pages2k_data[counter]['geo_meanElev'] = np.nan

        # Ensure lon is in [0,360] domain
        if pages2k_data[counter]['geo_meanLon'] < 0.0:
            pages2k_data[counter]['geo_meanLon'] = 360 + pages2k_data[counter]['geo_meanLon']

        # Determine the seasonality of the record.
        # Seasonal names were mapped the three-month climatological seasons.
        # 'early summer' was mapped to the first two months of summer only.  Is this right????????????
        # 'growing season' was mapped to summer.
        season_orig = pages2k_data[counter]['climateInterpretation_seasonality']
        if any(char.isdigit() for char in season_orig):
            pages2k_data_seasonality = map(int,season_orig.split(' '))
        elif season_orig == 'annual':
            if year_type == 'tropical year': pages2k_data_seasonality = [4,5,6,7,8,9,10,11,12,13,14,15]
            else: pages2k_data_seasonality = [1,2,3,4,5,6,7,8,9,10,11,12]
        elif season_orig == 'summer':
            if pages2k_data[counter]['geo_meanLat'] >= 0: pages2k_data_seasonality = [6,7,8]
            else: pages2k_data_seasonality = [-12,1,2]
        elif season_orig == 'winter':
            if pages2k_data[counter]['geo_meanLat'] >= 0: pages2k_data_seasonality = [-12,1,2]
            else: pages2k_data_seasonality = [6,7,8]
        elif season_orig == 'winter/spring':
            if pages2k_data[counter]['geo_meanLat'] >= 0: pages2k_data_seasonality = [-12,1,2,3,4,5]
            else: pages2k_data_seasonality = [6,7,8,9,10,11]
        elif season_orig == 'early summer':
            if pages2k_data[counter]['geo_meanLat'] >= 0: pages2k_data_seasonality = [6,7]
            else: pages2k_data_seasonality = [-12,1]
        elif season_orig == 'growing season':
            if pages2k_data[counter]['geo_meanLat'] >= 0: pages2k_data_seasonality = [6,7,8]
            else: pages2k_data_seasonality = [-12,1,2]
        else:
            if year_type == 'tropical year': pages2k_data_seasonality = [4,5,6,7,8,9,10,11,12,13,14,15]
            else: pages2k_data_seasonality = [1,2,3,4,5,6,7,8,9,10,11,12]

        # Spell out the name of the interpretation variable.
        if pages2k_data[counter]['climateInterpretation_variable'] == 'T':
            pages2k_data[counter]['climateInterpretation_variable'] = 'temperature'

        # Save to a dataframe
        frame  = pd.DataFrame({'a':pages2k_data[counter]['siteID'], 'b':pages2k_data[counter]['geo_siteName'], 'c':pages2k_data[counter]['geo_meanLat'], \
                               'd':pages2k_data[counter]['geo_meanLon'], 'e':pages2k_data[counter]['geo_meanElev'], \
                               'f':pages2k_data[counter]['archiveType'], 'g':pages2k_data[counter]['paleoData_variableName'], \
                               'h':proxy_resolution, 'i':pages2k_data[counter]['time_annual'][0], 'j':pages2k_data[counter]['time_annual'][-1], \
                               'k':pages2k_data[counter]['geo_pages2kRegion'], 'l':pages2k_data[counter]['climateInterpretation_variable'], \
                               'm':pages2k_data[counter]['climateInterpretation_variableDetail'], \
                               'n':pages2k_data[counter]['climateInterpretation_interpDirection'], 'o':None, 'p':None}, index=[counter])
        # To get seasonality & databases *lists* into columns 'o' and 'p' of DataFrame
        frame.set_value(counter,'o',pages2k_data_seasonality)
        frame.set_value(counter,'p',['PAGES2kv2'])
        # Append to main DataFrame
        metadf = metadf.append(frame)

        #print '   ', '{:40}'.format(key), ' : ', len(nb)
        tot.append(len(nb))
    nbtot = sum(tot)
    print '  ------------------------------------------------------'
    print '   ','{:40}'.format('Total:'), ' : ', nbtot
    print '----------------------------------------------------------------------'
    print ' '

    
    # Redefine column headers
    metadf.columns = headers

    # Write metadata to file
    print 'Now writing metadata to file:', metaout
    metadf.to_pickle(metaout)

    # -----------------------------------------------------
    # Build the proxy **data** DataFrame and output to file
    # -----------------------------------------------------
    print ' '
    print 'Now creating & loading the data in the pandas DataFrame...'
    print ' '

    for counter in range(0,len(pages2k_data)):
        print "Processing metadata", counter+1, "/", len(pages2k_data), ":", pages2k_data[counter]['paleoData_pages2kID']

        nbdata = len(pages2k_data[counter]['time_annual'])

        # Load data in numpy array
        frame_data = np.zeros(shape=[nbdata,2])
        frame_data[:,0] = pages2k_data[counter]['time_annual']
        frame_data[:,1] = pages2k_data[counter]['data_annual']

        if counter == 0:
            # Build up pandas DataFrame
            header = ['NCDC ID', pages2k_data[counter]['siteID']]
            df = pd.DataFrame({'a':frame_data[:,0], 'b':frame_data[:,1]})
            df.columns = header
        else:
            frame = pd.DataFrame({'NCDC ID':frame_data[:,0], pages2k_data[counter]['siteID']:frame_data[:,1]})
            df = df.merge(frame, how='outer', on='NCDC ID')

            
    # Fix DataFrame index and column name
    col0 = df.columns[0]
    df.set_index(col0, drop=True, inplace=True)
    df.index.name = 'Year C.E.'
    df.sort_index(inplace=True)

    # Write data to file
    print 'Now writing to file:', dataout
    df.to_pickle(dataout)
    print ' '
    print 'Done!'


# ===================================================================================
# For NCDC proxy data files ---------------------------------------------------------
# ===================================================================================

# ===================================================================================
def colonReader(string, fCon, fCon_low, end):
    '''This function seeks a specified string (or list of strings) within
    the transcribed file fCon (lowercase version fCon_low) until a specified
    character (typically end of the line) is found.x
    If a list of strings is provided, make sure they encompass all possibilities

    From Julien Emile-Geay (Univ. of Southern California)
    '''

    if isinstance(string, basestring):
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

# ===================================================================================

# ===================================================================================

def read_proxy_data_NCDCtxt(site, proxy_def):
#====================================================================================
# Purpose: Reads data from a selected site (chronology) in NCDC proxy dataset
# 
# Input   :
#      - site        : Full name of proxy data file, including the directory where 
#                      file is located.
#      - proxy_def   : Dictionary containing information on proxy types to look for 
#                      and associated characteristics, such as possible proxy 
#                      measurement labels for the specific proxy type 
#                      (ex. ['d18O','d18o','d18o_stk','d18o_int','d18o_norm'] 
#                      for delta 18 oxygen isotope measurements)
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
#====================================================================================
    
    import os
    import numpy as np
    

    # Possible header definitions of time in data files ...
    time_defs = ['age', 'age_int',\
                 'y_ad','Age_AD','age_AD','age_AD_ass','age_AD_int','Midpt_year','AD',\
                 'age_yb1950','yb_1950','yrb_1950',\
                 'kyb_1950',\
                 'yb_1989','age_yb1989',\
                 'yb_2000','yr_b2k','yb_2k','ky_b2k','kyb_2000','kyb_2k','kab2k','ka_b2k',\
                 'ky_BP','kyr_BP','ka_BP','age_kaBP','yr_BP','calyr_BP','Age(yrBP)','age_calBP']

    filename = site

    if os.path.isfile(filename):
        print ' '
        print 'File:', filename

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
        # ===========================================================================
        # Extract metadata from file ------------------------------------------------
        # ===========================================================================
        # ===========================================================================
        try:
            # 'Archive' is the proxy type
            d['Archive']              = colonReader('archive', fileContent, fileContent_low, '\n')
            # Other info
            d['Title']                = colonReader('study_name', fileContent, fileContent_low, '\n')
            investigators             = colonReader('investigators', fileContent, fileContent_low, '\n')
            d['Investigators']        = investigators.replace(';',' and') # take out the ; so that turtle doesn't freak out.
            d['PubDOI']               = colonReader('doi', fileContent, fileContent_low, '\n')

            # ===========================================================================
            # Extract information from the "Site_Information" section of the file -------
            # ===========================================================================
            # Find beginning of block
            sline_begin = fileContent.find('# Site_Information:')
            if sline_begin == -1:
                sline_begin = fileContent.find('# Site_Information')
            if sline_begin == -1:
                sline_begin = fileContent.find('# Site Information')
            # Find end of block
            sline_end = fileContent.find('# Data_Collection:')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection\n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection\n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection  \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection   \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data_Collection    \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data Collection\n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data Collection \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data Collection  \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data Collection   \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Data Collection    \n')

            SiteInfo = fileContent[sline_begin:sline_end]
            SiteInfo_low = SiteInfo.lower()

            d['SiteName'] = colonReader('site_name', SiteInfo, SiteInfo_low, '\n')
            d['Location'] = colonReader('location', SiteInfo, SiteInfo_low, '\n')

            #print SiteInfo

            str_lst = ['northernmost_latitude', 'northernmost latitude'] # documented instances of this field property
            d['NorthernmostLatitude'] = float(colonReader(str_lst, SiteInfo, SiteInfo_low, '\n'))  
            str_lst = ['southernmost_latitude', 'southernmost latitude'] # documented instances of this field property
            d['SouthernmostLatitude'] = float(colonReader(str_lst, SiteInfo, SiteInfo_low, '\n'))
            str_lst = ['easternmost_longitude', 'easternmost longitude'] # documented instances of this field property
            d['EasternmostLongitude'] = float(colonReader(str_lst, SiteInfo, SiteInfo_low, '\n'))
            str_lst = ['westernmost_longitude', 'westernmost longitude'] # documented instances of this field property
            d['WesternmostLongitude'] = float(colonReader(str_lst, SiteInfo, SiteInfo_low, '\n'))
            elev = colonReader('elevation', SiteInfo, SiteInfo_low, '\n')
            #if elev != 'nan' and len(elev)>0:
            if 'nan' not in elev and len(elev)>0:
                elev_s = elev.split(' ')
                # is elevation negative (depth)?
                if '-' in elev_s[0]:
                    negative = True
                    sign = '-'
                else:
                    negative = False
                    sign = ''
                # is there a decimal in elev_s?
                if '.' in elev_s[0]:
                    elev_s_split = elev_s[0].split('.')
                    elev_s_int = ''.join(c for c in elev_s_split[0] if c.isdigit())
                    elev_s_dec = ''.join(c for c in elev_s_split[1] if c.isdigit())
                    d['Elevation'] = float(sign+elev_s_int+'.'+elev_s_dec)
                else:
                    d['Elevation'] = float(sign+''.join(c for c in elev_s[0] if c.isdigit())) # to only keep digits ...            
                
            else:   
                d['Elevation'] = float('NaN')


            # ===========================================================================
            # Extract information from the "Data_Collection" section of the file --------
            # ===========================================================================
            # Find beginning of block
            sline_begin = fileContent.find('# Data_Collection:')
            if sline_begin == -1:
                sline_begin = fileContent.find('# Data_Collection')
            if sline_begin == -1:
                sline_begin = fileContent.find('# Data_Collection\n')

            # Find end of block
            sline_end = fileContent.find('# Variables:')
            if sline_end == -1:
                sline_end = fileContent.find('# Variables\n')
            if sline_end == -1:
                sline_end = fileContent.find('# Variables \n')
            if sline_end == -1:
                sline_end = fileContent.find('# Variables')
            if sline_end == -1:
                sline_end = fileContent.find('# Variables ')

            DataColl = fileContent[sline_begin:sline_end]
            DataColl_low = DataColl.lower()

            d['CollectionName']       = colonReader('collection_name', DataColl, DataColl_low, '\n')
            d['EarliestYear']         = float(colonReader('earliest_year', DataColl, DataColl_low, '\n'))
            d['MostRecentYear']       = float(colonReader('most_recent_year', DataColl, DataColl_low, '\n'))
            d['TimeUnit']             = colonReader('time_unit', DataColl, DataColl_low, '\n')
            if not d['TimeUnit']:
                d['TimeUnit']         = colonReader('time unit', DataColl, DataColl_low, '\n')
            if d['TimeUnit'] not in time_defs:
                print '***Time_Unit '+d['TimeUnit']+' not in recognized time definitions! Exiting!'
                return

            # Get Notes: information, if it exists
            notes = colonReader('notes', DataColl, DataColl_low, '\n')
            if notes: # not empty


                #print 'notes=>', notes
                #print ' '


                # database info is in form {"database":db1}{"database":db2} ...
                # extract fields that are in {}. This produces a list.
                jsdata = re.findall('\{.*?\}',notes)
                bad_chars = '{}"'
                jsdata = [item.translate(string.maketrans("", "", ), bad_chars) for item in jsdata]

                # Look for database information
                # -----------------------------
                # item in jsdata list with database info?

                # TODO: ... think about using try/except instead ...
                dbinfo = None
                jsdata_db = [item for i, item in enumerate(jsdata) if 'database:' in item]
                if jsdata_db: 
                    db_lst = re.sub('database:', '', jsdata_db[0]).split(',')
                    if len(db_lst) > 1:
                        dbinfo = [item.split(':')[1] for item in db_lst]
                    else:                    
                        dbinfo = db_lst

                # check if some db info exists
                if dbinfo:
                    d['Databases'] = dbinfo
                else:
                    # Set to default value if not found.
                    #d['Databases'] = None
                    d['Databases'] = ['LMR']


                # Look for information on "climate interpretation" of proxy record
                # ----------------------------------------------------------------

                # Initialize metadata to be extracted
                seasonality = [1,2,3,4,5,6,7,8,9,10,11,12] # annual (calendar)
                climateVariable = None
                climateVariableRealm = None
                climateVariableDirec = None
                
                jsdata_clim = [item for i, item in enumerate(jsdata) if 'climateInterpretation:' in item]
                if jsdata_clim: 
                    clim_lst = re.sub('climateInterpretation:', '', jsdata_clim[0])
                    clim_lst = clim_lst.replace('[','(').replace(']',')')
                    tmp =  re.split(r',\s*(?![^()]*\))',clim_lst)
                    clim_elements = [item.replace('(','[').replace(')',']') for item in tmp]
                    
                    seasonality          = [item.split(':')[1] for item in clim_elements if 'seasonality:' in item][0]
                    climateVariable      = [item.split(':')[1] for item in clim_elements if 'climateVariable:' in item][0]
                    climateVariableRealm = [item.split(':')[1] for item in clim_elements if 'climateVariableDetail:' in item][0]
                    climateVariableDirec = [item.split(':')[1] for item in clim_elements if 'interpDirection:' in item][0]
                    
                    if len(seasonality) == 0: seasonality = [1,2,3,4,5,6,7,8,9,10,11,12]
                    if len(climateVariable) == 0: climateVariable = None
                    if len(climateVariableRealm) == 0: climateVariableRealm = None
                    if len(climateVariableDirec) == 0: climateVariableDirec = None

                    # Some translation...
                    if climateVariable == 'T': climateVariable = 'temperature'
                    if climateVariable == 'M': climateVariable = 'moisture'

                # test whether seasonality is a string or already a list
                # if a string, convert to list                
                if type(seasonality) is not list: 
                    if isinstance(seasonality,six.string_types):
                        seasonality = ast.literal_eval(seasonality)
                    else:
                        print 'Problem with seasonality metadata! Exiting!'
                        exit(1)
                
                d['Seasonality'] = seasonality
                d['climateVariable'] =  climateVariable
                d['climateVariableRealm'] = climateVariableRealm
                d['climateVariableDirec'] = climateVariableDirec

                # Look for information about duplicate proxy records
                # --------------------------------------------------
                dup_lst = []
                jsdata_dup = [item for i, item in enumerate(jsdata) if 'duplicate:' in item]
                if jsdata_dup:
                    tmp = re.sub('duplicate:', '', jsdata_dup[0]).split(',')
                    if len(tmp) > 1:
                        dup_lst = [item.split(':')[1].rstrip('.txt') for item in tmp]
                    else:
                        dup_lst = [item.rstrip('.txt') for item in tmp]

                d['Duplicates'] = dup_lst
                

                
                """
                # Old code that worked for NCDC v0.0.0
                
                # Look for information on relation to temperature
                # -----------------------------------------------
                clim_temp_relation = [item.split(':')[1] for item in jsdata if item.split(':')[0] == 'relationship']
                if clim_temp_relation:
                    d['Relation_to_temp'] = clim_temp_relation[0]
                else:
                    d['Relation_to_temp'] = None

                # Look for information on the nature of sensitivity of the proxy data
                # (i.e. temperature or moisture or etc.)
                # -------------------------------------------------------------------
                clim_sensitivity = [item.split(':')[1] for item in jsdata if item.split(':')[0] == 'sensitivity']
                if clim_sensitivity:
                    d['Sensitivity'] = clim_sensitivity[0]
                else:
                    d['Sensitivity'] = None

                """

                d['Relation_to_temp'] = None
                d['Sensitivity'] = None

                
                    
            else:
                # Default values if not found.
                #d['Databases'] = None
                d['Databases'] = ['LMR']

                d['Seasonality'] = [1,2,3,4,5,6,7,8,9,10,11,12]
                d['climateVariable'] =  None
                d['climateVariableRealm'] = None
                d['climateVariableDirec'] = None

                d['Duplicates'] = []
                
                d['Relation_to_temp'] = None
                d['Sensitivity'] = None


            #print '<==>'
            #print DataColl

        except EmptyError, e:
            print e
            return

        # ===========================================================================
        # ===========================================================================
        # Extract the data from file ------------------------------------------------
        # ===========================================================================
        # ===========================================================================
        
        # ===========================================================================
        # Extract information from the "Variables" section of the file --------------
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


        print 'Site ID: ', d['CollectionName'], ' Archive:', d['Archive']

        # Cross-reference "ShortName" entries with possible proxy measurements specified in proxy_def dictionary
        proxy_types_all = proxy_def.keys()

        # Restrict to those matching d['Archive']
        proxy_types_keep = [s for s in proxy_types_all if d['Archive'] in s]

        # Which columns contain the important data (time & proxy values) to be extracted?
        # Referencing variables (time/age & proxy data) with data column IDsx
        # Time/age
        TimeColumn_ided = False
        for ivar in range(nvar):
            if d['DataColumn' + format(ivar+1, '02') + '_ShortName'] in time_defs:
                TimeColumn_ided = True
                TimeColumn_id = ivar
        if TimeColumn_ided:
            print '  Time/Age data in data column:', TimeColumn_id
        else:
            print ' '  


        # Proxy data
        # Dictionary containing info on proxy type and column ID where to find the data
        DataColumns_ided = False
        proxy_types_in_file = {}
        for ivar in range(nvar):
            proxy_types = [s for s in proxy_types_keep if d['DataColumn' + format(ivar+1, '02') + '_ShortName'] in proxy_def[s]]

            if proxy_types: # if non-empty list
                # Crude logic to distinguish between PAGES vs Breitenmoser Tree Rings data at proxy type level
                if len(proxy_types) > 1 and [item for item in proxy_types if 'Tree Rings' in item ]:
                    if 'Breitenmoser' in d['Investigators'].split(',')[0]:
                        treetag = 'Breit'
                    else:
                        treetag = 'Pages2'

                    ind = [i for i, s in enumerate(proxy_types) if treetag in s][0]
                    proxy_types_in_file[proxy_types[ind]] = (d['DataColumn' + format(ivar+1, '02') + '_ShortName'], ivar)
                    
                else:
                    proxy_types_in_file[proxy_types[0]] = (d['DataColumn' + format(ivar+1, '02') + '_ShortName'], ivar)

        
        dkeys = proxy_types_in_file.keys()
        nbvalid = len(dkeys)
        if nbvalid > 0:
            DataColumns_ided = True
            print '  Found ', nbvalid, ' valid proxy variables:'
            for i in range(nbvalid):
                print ' ', i, ' : ',  dkeys[i], proxy_types_in_file[dkeys[i]]

        
        # Check status of what has been found in the data file
        # If nothing found, just return (exit function by returning None)
        if not TimeColumn_ided or not DataColumns_ided:
            print '***Valid data was not found in file!'
            return (None, None)


        # ===========================================================================
        # Extract the numerical data from the "Data" section of the file ------------
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

        # Extract column descriptions (headers) of the data matrix    
        DataColumn_headers = fileContent_datalines[start_line_index].splitlines()[0].split('\t')
        # Strip possible blanks in column headers 
        DataColumn_headers = [item.strip() for item in  DataColumn_headers]
        nc = len(DataColumn_headers)
        #print nc, DataColumn_headers

        # ---------------
        # Now the data !!
        # ---------------
        inds_to_extract = []
        for dkey in dkeys:
            inds_to_extract.append(proxy_types_in_file[dkey][1])

        # from start of data block to end, in a list
        datalist = fileContent_datalines[start_line_index+1:]
        # Strip any empty lines
        datalist = [x for x in datalist if x]
        nbdata = len(datalist)

        # into numpy arrays
        time_raw = np.zeros(shape=[nbdata])
        data_raw = np.zeros(shape=[nbdata,nbvalid])
        # fill with NaNs for default values
        data_raw[:] = np.NAN

        for i in range(nbdata):
            tmp = datalist[i].split('\t')
            #print tmp
            # any empty element replaced by NANs
            tmp = ['NAN' if x == '' else x for x in tmp]
            time_raw[i]   = tmp[TimeColumn_id]
            # strip possible "()" in data before conversion to float
            # not sure why these are found sometimes ...
            tmp = [tmp[j].replace('(','') for j in range(len(tmp))]
            tmp = [tmp[j].replace(')','') for j in range(len(tmp))]
            data_raw[i,:] = [float(tmp[j]) for j in inds_to_extract]


        # Modify "time" array into "years CE" if not already
        #print 'TimeUnit:', d['TimeUnit']
        tdef = d['TimeUnit']
        tdef_parsed = tdef.split('_')
        if len(tdef_parsed) == 2 and tdef_parsed[0] and tdef_parsed[1]:
            # tdef has expected structure ...
            if tdef_parsed[0] == 'yb' and is_number(tdef_parsed[1]):
                time_raw = float(tdef_parsed[1]) - time_raw
                d['EarliestYear'] = float(tdef_parsed[1]) - d['EarliestYear']
                d['MostRecentYear'] = float(tdef_parsed[1]) - d['MostRecentYear']
            elif tdef_parsed[0] == 'kyb' and is_number(tdef_parsed[1]):
                time_raw = float(tdef_parsed[1]) - 1000.0*time_raw
                d['EarliestYear'] = float(tdef_parsed[1]) - 1000.0*d['EarliestYear']
                d['MostRecentYear'] = float(tdef_parsed[1]) - 1000.0*d['MostRecentYear']
            elif tdef_parsed[0] == 'y' and tdef_parsed[1] == 'ad':
                pass # do nothing, time already in years_AD
            else:
                print 'Unrecognized time definition. Returning empty arrays!'
                #time  = np.asarray([],dtype=np.float64)
                #value = np.asarray([],dtype=np.float64)                
                exit(1)
        else:
            print '*** WARNING *** Unexpected time definition: string has more elements than expected. Returning empty arrays!'
            #time  = np.asarray([],dtype=np.float64)
            #value = np.asarray([],dtype=np.float64)
            exit(1)


        # If subannual, average up to annual --------------------------------------------------------

        valid_frac = 0.5
            
        time_between_records = np.diff(time_raw, n=1)
        #print '=>', time_raw
        #print '=>', time_between_records

        # Temporal resolution of the data
        # Use MINIMUM (shortest) time difference ????????????????????????????????????
        #time_resolution = np.min(abs(time_between_records))
        # Use MODE of distribution (most frequent time difference) ????????????????????????????????????
        time_resolution = abs(stats.mode(time_between_records)[0][0])

        # check if time_resolution = 0.0 !!! sometimes adjacent records are tagged at same time ... 
        if time_resolution == 0.0:
            print '***WARNING! Found adjacent records with same times!'
            inderr = np.where(time_between_records == 0.0)
            print inderr
            time_between_records = np.delete(time_between_records,inderr)
            #time_resolution = np.min(abs(time_between_records))
            time_resolution = abs(stats.mode(time_between_records)[0][0])
        #print '=>', time_between_records
        #print '=>', time_resolution
        max_nb_per_year = int(1.0/time_resolution)

        if  time_resolution <=1.0:
            proxy_resolution = int(1.0)
        else:
            proxy_resolution = int(time_resolution)

        years_all = [int(time_raw[k]) for k in range(0,len(time_raw))]
        years = list(set(years_all)) # 'set' is used to get unique values in list
        years = sorted(years) # sort the list

        time_annual = np.asarray(years,dtype=np.float64)
        data_annual = np.zeros(shape=[len(years),nbvalid], dtype=np.float64)        
        # fill with NaNs for default values
        data_annual[:] = np.NAN

        # Loop over years in dataset
        for i in range(len(years)): 
            ind = [j for j, k in enumerate(years_all) if k == years[i]]
            nbdat = len(ind)

            # TODO: check nb of non-NaN values !!!!! ... ... ... ... ... ...

            if time_resolution <= 1.0: 
                frac = float(nbdat)/float(max_nb_per_year)
                if frac > valid_frac:
                    data_annual[i,:] = np.nanmean(data_raw[ind,:],axis=0)
            else:
                if nbdat > 1:
                    print '***WARNING! Found multiple records in same year in data with multiyear resolution!'
                    print '   year=', years[i], nbdat 
                # Note: this calculates the mean if multiple entries found
                data_annual[i,:] = np.nanmean(data_raw[ind,:],axis=0)

            #print years[i], nbdat, max_nb_per_year, data_annual[i,:]


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

        # Ensure lon is in [0,360] domain
        if lon < 0.0:
            lon = 360 + lon

        # Range in years for which data is available
        yearRange = (int('%.0f' % d['EarliestYear']),int('%.0f' %d['MostRecentYear']))



        # Define and fill list of dictionaries to be returned by function
        # 
        returned_list = []
        duplicate_list = []
        for k in range(len(dkeys)):            
            key = dkeys[k]
            
            ind = proxy_types_in_file[key][1]
            proxy_units = d['DataColumn' + format(ind+1, '02') + '_Units']
            proxy_measurement = key.split('_')[1]            
            proxy_measurement = d['DataColumn' + format(ind+1, '02') + '_ShortName']

            if key == 'Tree Rings_WidthBreit': proxy_measurement = proxy_measurement + '_breit'
            proxy_name = d['CollectionName']+':'+proxy_measurement

            proxy_dict = {}
            proxy_dict[proxy_name] = {}
            proxy_dict[proxy_name]['Archive']              = d['Archive']
            proxy_dict[proxy_name]['SiteName']             = d['SiteName']
            proxy_dict[proxy_name]['Location']             = d['Location']
            proxy_dict[proxy_name]['Resolution (yr)']      = proxy_resolution
            proxy_dict[proxy_name]['Lat']                  = lat
            proxy_dict[proxy_name]['Lon']                  = lon
            proxy_dict[proxy_name]['Elevation']            = alt
            proxy_dict[proxy_name]['YearRange']            = yearRange
            proxy_dict[proxy_name]['Measurement']          = proxy_measurement
            proxy_dict[proxy_name]['DataUnits']            = proxy_units
            proxy_dict[proxy_name]['Databases']            = d['Databases']

            proxy_dict[proxy_name]['Seasonality']          = d['Seasonality']
            proxy_dict[proxy_name]['climateVariable']      = d['climateVariable']
            proxy_dict[proxy_name]['Realm']                = d['climateVariableRealm']
            proxy_dict[proxy_name]['climateVariableDirec'] = d['climateVariableDirec']
            

            
            # *** for v.0.1.0:
            #proxy_dict[proxy_name]['Relation_to_temp'] = d['Relation_to_temp']
            #proxy_dict[proxy_name]['Sensitivity']      = d['Sensitivity']
            
            
            proxy_dict[proxy_name]['Years']            = time_annual
            proxy_dict[proxy_name]['Data']             = data_annual[:,k]

            if d['Duplicates']:
                duplicate_list.extend(d['Duplicates'])
            
            returned_list.append(proxy_dict)

    else:
        print '***File NOT FOUND:', filename
        returned_list = []
        duplicate_list = []
        
    return returned_list, duplicate_list

# =========================================================================================

def ncdc_txt_to_dataframes(datadir, proxy_def, metaout, dataout, eliminate_duplicates):
    """
    Takes in NCDC text proxy data and converts it to dataframe storage.  

    Caveat: This increases size on disk due to the joining along the time index (lots of null values).
    But: Makes it easier to query and grab data for the proxy data assimilation experiments.

    :param datadir   :
    :param proxy_def : 
    :param metaout   :
    :param dataout   :
    :return:
    
    Author: R. Tardif, Univ. of Washington, Jan 2016.

    """

    # ===============================================================================
    # Upload proxy data from NCDC-formatted text files
    # ===============================================================================

    # List filenames im the data directory (dirname)
    # files is a python list contining file names to be read
    sites_data = glob.glob(datadir+"/*.txt")

    # For specific testing :
    #sites_data = ['/home/disk/ekman/rtardif/kalman3/LMR/data/proxies/NCDC/LMR_data_files-master/00aust01a.txt']
    #sites_data = ['/home/disk/ekman/rtardif/kalman3/LMR/data/proxies/NCDC/LMR_data_files-master/SAm_8.txt']
    # --
    #sites_data = ['/home/disk/kalman3/rtardif/LMR/data/proxies/NCDC/ToPandas_v0.1.0/00Epic01.txt']
    #sites_data = ['/home/disk/kalman3/rtardif/LMR/data/proxies/NCDC/ToPandas_v0.1.0/PAGES2kv2_Arc-MackenzieDelta_Porter_2013LMR63.txt']
    #sites_data = ['/home/disk/kalman3/rtardif/LMR/data/proxies/NCDC/ToPandas_v0.1.0/PAGES2kv2_SAm-chil017_Szeicz_2000LMR681.txt']
    #sites_data = ['/home/disk/kalman3/rtardif/LMR/data/proxies/NCDC/ToPandas_v0.1.0/PAGES2kv2_Arc-Kittelfjall_Bjorkelund_2013LMR54.txt']
    #sites_data = ['/home/disk/kalman3/rtardif/LMR/data/proxies/NCDC/ToPandas_v0.1.0/PAGES2kv2_Ocean2kHR-PacificLinsley2006Rarotongad18O2RLMR652.txt']
    #sites_data = ['/home/disk/kalman3/rtardif/LMR/data/proxies/NCDC/ToPandas_v0.1.0/PAGES2kv2_Ant-BerknerIslan_Mulvaney_2002LMR5.txt']

    nbsites = len(sites_data)
    
    # Master list containing dictionaries of all proxy chronologies extracted from the data files.
    master_proxy_list = []
    master_duplicate_list = []

    # Loop over files
    nbsites_valid = 0
    duplicate_list = []
    for file_site in sites_data:
        proxy_list, duplicate_list = read_proxy_data_NCDCtxt(file_site,proxy_def)
        if eliminate_duplicates and duplicate_list:
             master_duplicate_list.extend(duplicate_list)
        if proxy_list: # if returned list is not empty
            # extract dictionary and populate the master proxy list
            for item in proxy_list:
                master_proxy_list.append(item)
            nbsites_valid = nbsites_valid + 1
        else: # returned list is empty, just move to next site
            pass

    nbduplicates = len(master_duplicate_list)
    nbextracted = len(master_proxy_list)

    # eliminate duplicates if option activated
    if eliminate_duplicates:
        final_proxy_list = []
        master_proxy_siteIDs = [item.keys()[0].split(':')[0] for item in master_proxy_list]
        inds = [i for i, item in enumerate(master_proxy_siteIDs) if item not in master_duplicate_list]
        nbduplicates = len(master_proxy_list) - len(inds)
        # extract those not in list of duplicates
        for k in inds: final_proxy_list.append(master_proxy_list[k])
    else:
        final_proxy_list = master_proxy_list
        
    
    # ===============================================================================
    # Produce a summary of uploaded proxy data & 
    # generate integrated database in pandas DataFrame format
    # ===============================================================================

    # Summary of the final_proxy_list
    nbchronol = len(final_proxy_list)
    print ' '
    print ' '
    print '----------------------------------------------------------------------'
    print ' SUMMARY: '
    print '  Total nb of files found & queried           : ', nbsites
    print '  Total nb of files with valid data           : ', nbsites_valid
    print '  Number of proxy chronologies extracted      : ', nbextracted
    print '  Number of identified duplicate chronologies : ', nbduplicates
    print '  Number of proxy chronologies included in df : ', nbchronol
    print '  ------------------------------------------------------'

    tot = []
    # Loop over proxy types specified in *main*
    counter = 0
    # Build up pandas DataFrame
    metadf  = pd.DataFrame()

    # for v0.0.0:
    #headers = ['NCDC ID','Site name','Lat (N)','Lon (E)','Elev','Archive type','Proxy measurement','Resolution (yr)',\
    #               'Oldest (C.E.)','Youngest (C.E.)','Location','Sensitivity','Relation_to_temp','Databases']
    # for v0.1.0:
    headers = ['NCDC ID','Site name','Lat (N)','Lon (E)','Elev','Archive type','Proxy measurement','Resolution (yr)',\
               'Oldest (C.E.)','Youngest (C.E.)','Location','climateVariable','Realm','Relation_to_climateVariable',\
               'Seasonality', 'Databases']

    
    
    for key in sorted(proxy_def.keys()):
        proxy_archive = key.split('_')[0]

        # change the associated between proxy type and proxy measurement for Breitenmoser tree ring data
        if key == 'Tree Rings_WidthBreit':
            proxy_def[key] = [item+'_breit' for item in proxy_def[key]]
            
        nb = []
        for item in final_proxy_list:
            siteID = item.keys()[0]
            if item[siteID]['Archive'] == proxy_archive and item[siteID]['Measurement'] in proxy_def[key]:
                nb.append(siteID)
                # *** for v.0.0.0:
                #frame  = pd.DataFrame({'a':siteID, 'b':item[siteID]['SiteName'], 'c':item[siteID]['Lat'], 'd':item[siteID]['Lon'], \
                #                       'e':item[siteID]['Elevation'], 'f':item[siteID]['Archive'], 'g':item[siteID]['Measurement'], \
                #                       'h':item[siteID]['Resolution (yr)'], 'i':item[siteID]['YearRange'][0], \
                #                       'j':item[siteID]['YearRange'][1], 'k':item[siteID]['Location'], \
                #                       'l':item[siteID]['Sensitivity'], 'm':item[siteID]['Relation_to_temp'], 'n':None}, index=[counter])
                ## To get database *list* into column 'm' of DataFrame
                #frame.set_value(counter,'n',item[siteID]['Databases'])
                ## Append to main DataFrame
                #metadf = metadf.append(frame)
                # *** for v.0.1.0:
                frame  = pd.DataFrame({'a':siteID, 'b':item[siteID]['SiteName'], 'c':item[siteID]['Lat'], 'd':item[siteID]['Lon'], \
                                       'e':item[siteID]['Elevation'], 'f':item[siteID]['Archive'], 'g':item[siteID]['Measurement'], \
                                       'h':item[siteID]['Resolution (yr)'], 'i':item[siteID]['YearRange'][0], \
                                       'j':item[siteID]['YearRange'][1], 'k':item[siteID]['Location'], \
                                       'l':item[siteID]['climateVariable'], 'm':item[siteID]['Realm'], \
                                       'n':item[siteID]['climateVariableDirec'], 'o':None, 'p':None}, index=[counter])
                # To get seasonality & databases *lists* into columns 'o' and 'p' of DataFrame
                frame.set_value(counter,'o',item[siteID]['Seasonality'])
                frame.set_value(counter,'p',item[siteID]['Databases'])
                # Append to main DataFrame
                metadf = metadf.append(frame)

                counter = counter + 1

        print '   ', '{:40}'.format(key), ' : ', len(nb)
        tot.append(len(nb))
    nbtot = sum(tot)
    print '  ------------------------------------------------------'
    print '   ','{:40}'.format('Total:'), ' : ', nbtot
    print '----------------------------------------------------------------------'
    print ' '

    
    # Redefine column headers
    metadf.columns = headers

    # Write metadata to file
    print 'Now writing metadata to file:', metaout
    metadf.to_pickle(metaout)

    # -----------------------------------------------------
    # Build the proxy **data** DataFrame and output to file
    # -----------------------------------------------------
    print ' '
    print 'Now creating & loading the data in the pandas DataFrame...'
    print ' '

    counter = 0
    for item in final_proxy_list:
        siteID = item.keys()[0]

        years = item[siteID]['Years']
        data = item[siteID]['Data']
        [nbdata,] = years.shape

        # Load data in numpy array
        frame_data = np.zeros(shape=[nbdata,2])
        frame_data[:,0] = years
        frame_data[:,1] = data

        if counter == 0:
            # Build up pandas DataFrame
            header = ['NCDC ID', siteID]
            df = pd.DataFrame({'a':frame_data[:,0], 'b':frame_data[:,1]})
            df.columns = header
        else:
            frame = pd.DataFrame({'NCDC ID':frame_data[:,0], siteID:frame_data[:,1]})
            df = df.merge(frame, how='outer', on='NCDC ID')

        counter = counter + 1
    
    # Fix DataFrame index and column name
    col0 = df.columns[0]
    df.set_index(col0, drop=True, inplace=True)
    df.index.name = 'Year C.E.'
    df.sort_index(inplace=True)

    # Write data to file
    print 'Now writing to file:', dataout
    df.to_pickle(dataout)
    print ' '
    print 'Done!'


# =========================================================================================
# =========================================================================================
if __name__ == "__main__":
    main()


