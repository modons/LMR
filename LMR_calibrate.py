"""
Module containing definitions pertaining to linear PSM calibration sources.

Revisions:
- Addition of the GPCC precipitation dataset as a possible calibration 
  moisture source.
  [R. Tardif, U. of Washington, February 2016]
- Addition of the Dai PDSI dataset as a possible calibration
  moisture source.
  [R. Tardif, U. of Washington, May 2016]
- Addition of the SPEI dataset as a possible calibration moisture source.
  [R. Tardif, U. of Washington, December 2016]
- Added parameter explicitely defining a reference period in the calculation
  of anomalies in functions tasked with uploading instrumental-era calibration 
  datasets. 
  Parameter now defined in configuration. 
  [R. Tardif, U. of Washington, February 2018]

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
    elif icalib == 'NOAAGlobalTemp':
        calib_object = calibration_NOAAGlobalTemp()
    elif icalib == 'GPCC':
        calib_object = calibration_precip_GPCC()
    elif icalib == 'DaiPDSI':
        calib_object = calibration_precip_DaiPDSI()
    elif icalib == 'SPEI':
        calib_object = calibration_precip_SPEI()
    else:
        print('Error in calibration data specification! Exiting ...')
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
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']
    outfreq = 'monthly'
    
    # read the data
    def read_calibration(self):        
        from load_gridded_data import read_gridded_data_GISTEMP
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_GISTEMP(self.datadir_calib,
                                                                                    self.datafile_calib,
                                                                                    self.calib_vars,
                                                                                    self.outfreq,
                                                                                    self.anom_reference_period)


# -------------------------------------------------------------------------------
# *** HadCRUT class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_HadCRUT(calibration_master):

    source = 'HadCRUT'
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']
    outfreq = 'monthly'

    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_HadCRUT
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_HadCRUT(self.datadir_calib,
                                                                                    self.datafile_calib,
                                                                                    self.calib_vars,
                                                                                    self.outfreq,
                                                                                    self.anom_reference_period)


# -------------------------------------------------------------------------------
# *** BerkeleyEarth class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_BerkeleyEarth(calibration_master):

    source = 'BerkeleyEarth'
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']
    outfreq = 'monthly'
    
    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_BerkeleyEarth
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_BerkeleyEarth(self.datadir_calib,
                                                                                          self.datafile_calib,
                                                                                          self.calib_vars,
                                                                                          self.outfreq,
                                                                                          self.anom_reference_period)

# -------------------------------------------------------------------------------
# *** MLOST class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_MLOST(calibration_master):

    source = 'MLOST'
    dataformat_calib = 'NCD'
    calib_vars = ['air']
    outfreq = 'monthly'
    
    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_MLOST
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_MLOST(self.datadir_calib,
                                                                                  self.datafile_calib,
                                                                                  self.calib_vars,
                                                                                  self.outfreq,
                                                                                  self.anom_reference_period)


# -------------------------------------------------------------------------------
# *** NOAAGlobalTemp class ----------------------------------------
# -------------------------------------------------------------------------------
class calibration_NOAAGlobalTemp(calibration_master):

    source = 'NOAAGlobalTemp'
    dataformat_calib = 'NCD'
    calib_vars = ['air']
    outfreq = 'monthly'
    
    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_MLOST
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_MLOST(self.datadir_calib,
                                                                                  self.datafile_calib,
                                                                                  self.calib_vars,
                                                                                  self.outfreq,
                                                                                  self.anom_reference_period)


# -------------------------------------------------------------------------------
# *** GPCC class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_precip_GPCC(calibration_master):

    source = 'GPCC'
    dataformat_calib = 'NCD'
    calib_vars = ['precip']
    outfreq = 'monthly'
    # read_calibration() to return anomalies w.r.t. a reference period (True or False)
    out_anomalies = True

    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_GPCC
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_GPCC(self.datadir_calib,
                                                                                 self.datafile_calib,
                                                                                 self.calib_vars,
                                                                                 self.out_anomalies,
                                                                                 self.anom_reference_period,
                                                                                 self.outfreq)


# -------------------------------------------------------------------------------
# *** DaiPDSI class -----------------------------------------------
# -------------------------------------------------------------------------------
class calibration_precip_DaiPDSI(calibration_master):

    source = 'DaiPDSI'
    dataformat_calib = 'NCD'
    calib_vars = ['pdsi']
    outfreq = 'monthly'
    # read_calibration() to return anomalies w.r.t. a reference period (True or False)
    out_anomalies = True

    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_DaiPDSI
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_DaiPDSI(self.datadir_calib,
                                                                                    self.datafile_calib,
                                                                                    self.calib_vars,
                                                                                    self.out_anomalies,
                                                                                    self.anom_reference_period,
                                                                                    self.outfreq)


# -------------------------------------------------------------------------------
# *** SPEI class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_precip_SPEI(calibration_master):

    source = 'SPEI'
    dataformat_calib = 'NCD'
    calib_vars = ['spei']
    outfreq = 'monthly'
    # read_calibration() to return anomalies w.r.t. a reference period (True or False)
    out_anomalies = True
    
    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_SPEI
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_SPEI(self.datadir_calib,
                                                                                 self.datafile_calib,
                                                                                 self.calib_vars,
                                                                                 self.out_anomalies,
                                                                                 self.anom_reference_period,
                                                                                 self.outfreq)
