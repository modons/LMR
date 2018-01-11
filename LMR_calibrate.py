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
    datafile_calib   = 'gistemp1200_ERSSTv4.nc'
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']
    outfreq = 'monthly'
    
    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_GISTEMP
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_GISTEMP(self.datadir_calib,
                                                                                    self.datafile_calib,
                                                                                    self.calib_vars,
                                                                                    self.outfreq)


# -------------------------------------------------------------------------------
# *** HadCRUT class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_HadCRUT(calibration_master):

    source = 'HadCRUT'
    datafile_calib   = 'HadCRUT.4.3.0.0.median.nc'
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']
    outfreq = 'monthly'

    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_HadCRUT
        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_HadCRUT(self.datadir_calib,
                                                                                    self.datafile_calib,
                                                                                    self.calib_vars,
                                                                                    self.outfreq)


# -------------------------------------------------------------------------------
# *** BerkeleyEarth class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_BerkeleyEarth(calibration_master):

    source = 'BerkeleyEarth'
    datafile_calib   = 'Land_and_Ocean_LatLong1.nc'
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']
    outfreq = 'monthly'
    
    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_BerkeleyEarth

        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_BerkeleyEarth(self.datadir_calib,
                                                                                          self.datafile_calib,
                                                                                          self.calib_vars,
                                                                                          self.outfreq)

# -------------------------------------------------------------------------------
# *** MLOST class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_MLOST(calibration_master):

    source = 'MLOST'
    datafile_calib   = 'MLOST_air.mon.anom_V3.5.4.nc'
    dataformat_calib = 'NCD'
    calib_vars = ['Tsfc']
    outfreq = 'monthly'
    
    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_MLOST

        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_MLOST(self.datadir_calib,
                                                                                  self.datafile_calib,
                                                                                  self.calib_vars,
                                                                                  self.outfreq)


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
    outfreq = 'monthly'
    
    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_GPCC

        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_GPCC(self.datadir_calib,
                                                                                 self.datafile_calib,
                                                                                 self.calib_vars,
                                                                                 self.out_anomalies,
                                                                                 self.outfreq)


# -------------------------------------------------------------------------------
# *** DaiPDSI class -----------------------------------------------
# -------------------------------------------------------------------------------
class calibration_precip_DaiPDSI(calibration_master):

    source = 'DaiPDSI'
    datafile_calib   = 'Dai_pdsi.mon.mean.selfcalibrated_185001-201412.nc'

    dataformat_calib = 'NCD'
    calib_vars = ['pdsi']
    out_anomalies = True
    outfreq = 'monthly'
    
    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_DaiPDSI

        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_DaiPDSI(self.datadir_calib,
                                                                                    self.datafile_calib,
                                                                                    self.calib_vars,
                                                                                    self.out_anomalies,
                                                                                    self.outfreq)


# -------------------------------------------------------------------------------
# *** SPEI class --------------------------------------------------
# -------------------------------------------------------------------------------
class calibration_precip_SPEI(calibration_master):

    source = 'SPEI'
    datafile_calib   = 'spei_monthly_v2.4_190001-201412.nc'

    dataformat_calib = 'NCD'
    calib_vars = ['spei']
    out_anomalies = True
    outfreq = 'monthly'
    
    # read the data
    def read_calibration(self):
        from load_gridded_data import read_gridded_data_SPEI

        [self.time,self.lat,self.lon,self.temp_anomaly] = read_gridded_data_SPEI(self.datadir_calib,
                                                                                 self.datafile_calib,
                                                                                 self.calib_vars,
                                                                                 self.out_anomalies,
                                                                                 self.outfreq)
