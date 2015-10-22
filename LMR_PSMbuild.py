


#==========================================================================================
# 
# 
# 
# 
# Originator : Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
#                            | April 2015
#  
#========================================================================================== 

psm_info = \
"""
Proxies built using a linear PSM calibrated against 2m air temperature.
"""

# Parameters for the PSM build

# Section 1: High-level parameters

# set the absolute path the experiment (could make this cwd with some os coding)
LMRpath = '/home/disk/kalman3/rtardif/LMR'
#LMRpath = '/home/disk/ekman/rtardif/nobackup/LMR'

# Section 2: PROXIES

# Proxy data directory & file
datadir_proxy    = LMRpath+'/data/proxies';
datafile_proxy   = 'Pages2k_DatabaseS1-All-proxy-records.xlsx';
dataformat_proxy = 'CSV';

regions = ['Antarctica','Arctic','Asia','Australasia','Europe','North America','South America']

# Define proxies to be assimilated
# Use dictionary with structure: {<<proxy type>>:[... list of all measurement tags ...]}
# where "proxy type" is written as "<<archive type>>_<<measurement type>>"
# ---DO NOT CHANGE FORMAT of the how the dictionary text is laid out below
proxy_calib = {\
    '01:Tree ring_Width': ['Ring width','Tree ring width','Total ring width','TRW'],\
    '02:Tree ring_Density': ['Maximum density','Minimum density','Earlywood density','Latewood density','MXD'],\
    '03:Ice core_d18O': ['d18O'],\
    '04:Ice core_d2H': ['d2H'],\
    '05:Ice core_Accumulation':['Accumulation'],\
    '06:Coral_d18O': ['d18O'],\
    '07:Coral_Luminescence':['Luminescence'],\
    '08:Lake sediment_All':['Varve thickness','Thickness','Mass accumulation rate','Particle-size distribution','Organic matter','X-ray density'],\
    '09:Marine sediment_All':['Mg/Ca'],\
    '10:Speleothem_All':['Lamina thickness'],\
    }

# Proxy temporal resolution (in yrs)
#proxy_resolution = [1.0,5.0]
proxy_resolution = [1.0]

# Section 3: Calibration

# Source of calibration data (for PSM)
datatag_calib = 'GISTEMP'
#datatag_calib = 'HadCRUT'
#datatag_calib = 'BerkeleyEarth'
#datatag_calib = 'MLOST'
#datatag_calib = 'NOAA'
datadir_calib = LMRpath+'/data/analyses';

# Section 4: Output

#psm_output  = '/home/disk/kalman3/hakim/LMR/PSM/PAGES2kS1'
psm_output  = '/home/disk/kalman3/rtardif/LMR/PSM/PAGES2kS1'
#psm_output  = '/home/disk/ekman/rtardif/nobackup/LMR/PSM'


# =============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Main code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# =============================================================================
def main():

    import os
    import numpy as np
    import cPickle    
    from time import time

    import LMR_calibrate
    import LMR_proxy

    from load_proxy_data import create_proxy_lists_from_metadata_S1csv as create_proxy_lists_from_metadata

    begin_time = time()

    # ===========================
    # Creating calibration object
    # ===========================
    # Assign calibration object according to "datatag_calib" (from namelist)
    C = LMR_calibrate.calibration_assignment(datatag_calib)
    # the path to the calibration directory is specified in the namelist file; bind it here
    C.datadir_calib = datadir_calib;
    # read the data
    C.read_calibration()
 
    # ===============================================================================
    # Get information on proxies to calibrate ---------------------------------------
    # ===============================================================================

    # Read proxy metadata & extract list of proxy sites for wanted proxy type & measurement
    print 'Reading the proxy metadata & building list of chronologies to assimilate...'
    [sites_calib, _] = create_proxy_lists_from_metadata(datadir_proxy,datafile_proxy,regions,proxy_resolution,proxy_calib,1.0,None,None)

    proxy_types = sites_calib.keys()

    print '-----------------------------------'
    print 'Sites to be processed:             '
    totalsites = 0
    for proxy_key in proxy_types:
        proxy = proxy_key.split(':', 1)[1]
        print proxy, ':', len(sites_calib[proxy_key]), 'sites'
        totalsites = totalsites + len(sites_calib[proxy_key])
    print '-----------------------------------'
    print 'Total:', totalsites
    print ' '

    # Output dictionary
    psm_dict = {}

    sitecount = 0
    # Loop over proxy types
    for proxy_key in sorted(proxy_types):
        proxy = proxy_key.split(':', 1)[1]

        # --------------------------------------------------------------------
        # Loop over sites (chronologies) to be assimilated for this proxy type
        # --------------------------------------------------------------------
        for site in sites_calib[proxy_key]:

            sitecount = sitecount + 1

            Y = LMR_proxy.proxy_assignment(proxy)
            # add namelist attributes to the proxy object
            Y.proxy_datadir  = datadir_proxy
            Y.proxy_datafile = datafile_proxy
            Y.proxy_region   = regions

            # -------------------------------------------
            # Read data for current proxy type/chronology
            # -------------------------------------------
            Y.read_proxy(site)

            if Y.nobs == 0: # if no obs uploaded, move to next proxy site
                continue

            print ''
            print 'Site:', Y.proxy_type, ':', site, '=> nb', sitecount, 'out of', totalsites, '(',(np.float(sitecount)/np.float(totalsites))*100,'% )'
            print ' latitude, longitude: ' + str(Y.lat), str(Y.lon)

            # --------------------------------------------------------------
            # Call PSM to calibrate 
            # --------------------------------------------------------------
            Ynot = Y.psm(C,None,None,None)

            if hasattr(Y, 'corr'):
                # Populate dictionary containing details of PSMs

                # Site info
                sitetag = (proxy, site)
                psm_dict[sitetag] = {}
                psm_dict[sitetag]['calib'] = datatag_calib
                psm_dict[sitetag]['lat']   = Y.lat
                psm_dict[sitetag]['lon']   = Y.lon
                psm_dict[sitetag]['alt']   = Y.alt
                
                # PSM info
                psm_dict[sitetag]['PSMslope']     = Y.slope
                psm_dict[sitetag]['PSMintercept'] = Y.intercept
                psm_dict[sitetag]['PSMcorrel']    = Y.corr
                psm_dict[sitetag]['PSMmse']       = Y.R

            else:
                print 'PSM could not be built for this proxy chronology...too little overlap for calibration'
                continue


    # Dump dictionary to pickle file
    if not os.path.isdir(psm_output):
        os.system('mkdir %s' % psm_output)
    outfile = open('%s/PSMs_%s.pckl' % (psm_output, datatag_calib),'w')
    cPickle.dump(psm_dict,outfile)
    cPickle.dump(psm_info,outfile)
    outfile.close()

    end_time = time() - begin_time
    print '====================================================='
    print 'PSMs completed in '+ str(end_time/60.0)+' mins'
    print '====================================================='

# =============================================================================

if __name__ == '__main__':
    main()
