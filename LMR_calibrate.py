"""
Module containing definitions pertaining to linear PSM calibration sources.

Revisions:
- Addition of the GPCC precipitation dataset as a possible calibration source.
  [R. Tardif, U. of Washington, Spring 2016] 

"""
# -------------------------------------------------------------------------------
# *** Calibration type assignment  ----------------------------------------------
# -------------------------------------------------------------------------------
# All logic for proxy object assignment

def calibration_assignment(icalib):
    if icalib == 'GISTEMP':
        calib_object = calibration_GISTEMP()
    elif icalib == 'HadCRUT':
        calib_object = calibration_HadCRUT()
    elif icalib == 'BerkeleyEarth':
        calib_object = calibration_BerkeleyEarth()
    elif icalib == 'MLOST':
        calib_object = calibration_MLOST()
    elif icalib == 'GPCC':
        calib_object = calibration_precip_GPCC()
    else:
        print 'Error in calibration data specification! Exiting ...'
        exit(1)
      
    return calib_object

# -------------------------------------------------------------------------------
# *** Master class for calibration ----------------------------------------------
# -------------------------------------------------------------------------------
class calibration_master(object):
    '''
    This is the master calibration class. Turn this into a metaclass so one cannot instantiate directly; 
    it is an abstract class.
    '''
    pass

# -------------------------------------------------------------------------------
# *** GISTEMP class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_GISTEMP(calibration_master):

    source = 'GISTEMP'
    datafile_calib   = 'gistemp1200_ERSST.nc'
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']

    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_GISTEMP
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_GISTEMP(self.datadir_calib,self.datafile_calib,self.calib_vars)


# -------------------------------------------------------------------------------
# *** HadCRUT class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_HadCRUT(calibration_master):

    source = 'HadCRUT'
    datafile_calib   = 'HadCRUT.4.3.0.0.median.nc'
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']

    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_HadCRUT
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_HadCRUT(self.datadir_calib,self.datafile_calib,self.calib_vars)


# -------------------------------------------------------------------------------
# *** BerkeleyEarth class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_BerkeleyEarth(calibration_master):

    source = 'BerkeleyEarth'
    datafile_calib   = 'Land_and_Ocean_LatLong1.nc'
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']

    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_BerkeleyEarth

        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_BerkeleyEarth(self.datadir_calib,self.datafile_calib,self.calib_vars)

# -------------------------------------------------------------------------------
# *** MLOST class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_MLOST(calibration_master):

    source = 'MLOST'
    datafile_calib   = 'MLOST_air.mon.anom_V3.5.4.nc'
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']

    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_MLOST

        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_MLOST(self.datadir_calib,self.datafile_calib,self.calib_vars)


# -------------------------------------------------------------------------------
# *** NOAA class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_NOAA(calibration_master):

    source = 'NOAA'
    datafile_calib   = 'er-ghcn-sst.nc'
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']

    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_NOAA

        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_NOAA(self.datadir_calib,self.datafile_calib,self.calib_vars)

# -------------------------------------------------------------------------------
# *** GPCC class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_precip_GPCC(calibration_master):

    source = 'GPCC'
    #datafile_calib   = 'GPCC_precip.mon.total.0.5x0.5.v6.nc'
    #datafile_calib   = 'GPCC_precip.mon.total.1x1.v6.nc'
    #datafile_calib   = 'GPCC_precip.mon.total.2.5x2.5.v6.nc'
    datafile_calib   = 'GPCC_precip.mon.flux.1x1.v6.nc'

    dataformat_calib = 'NCD'
    calib_vars = ['precip']
    out_anomalies = True
    
    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_GPCC

        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_GPCC(self.datadir_calib,self.datafile_calib,self.calib_vars,self.out_anomalies)

