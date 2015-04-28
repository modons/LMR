
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
    elif icalib == 'ccsm4_last_millenium':
        calib_object = calibration_ccsm4_last_millenium()
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

