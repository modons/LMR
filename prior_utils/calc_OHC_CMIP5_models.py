"""
Module: calc_OHC_CMIP5_models.py

Purpose: Calculates gridded ocean heat content over ocean layers of specified depths
         using grin info and gridded sea water potential temperature as input.

 Required files:
          1) 'thetao_...'     : File containing sea water potential temperature for all
                                times (monthly) and depths available in the model simulation. 
                                ex. thetao_Omon_CCSM4_past1000_085001_185012.nc
          2) 'volcello_fx...' : File providing information on the volume of grid cells
                                of the ocean model.
                                ex. volcello_fx_CCSM4_past1000_r0i0p0.nc
          3) 'areacello_fx...': File providing information on the area (horizontal) of
                                grid cells of the ocean model.
                                ex. areacello_fx_CCSM4_past1000_r0i0p0.nc

Originator: Robert Tardif, University of Washington, August 2017

            Adapted from code originally written by 
            Dan Amrhein, University of Washington, July 2017

Revisions: 

"""

import numpy as np
from netCDF4 import Dataset, date2num, num2date
from os.path import join
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

# ==============================================================================

def main():

    
    # --------------------- begin user input -----------------------------

    #datadir = '/home/disk/kalman3/rtardif/CMIP5data'
    datadir = '/home/disk/ekman4/rtardif/CMIP5data'

    # ---    
    model_exp = 'ccsm4_last_millenium'
    exper_tag = 'CCSM4_past1000'
    period = '085001-185012'
    realm_freq = 'Omon'
    # ---
    

    # layer (depths in m) over which OHC is to be calculated
    layer = (0.,700.)
    #layer = (0.,2000.)

    # options for time averaging
    time_avg = None                                    # keep as in the file
    #time_avg = {'annual':[1,2,3,4,5,6,7,8,9,10,11,12]} # calendar year
    #time_avg = {'annual':[-12,1,2]}                    # DJF
    #time_avg = {'annual':[6,7,8]}                      # JJA
    
    # ---------------------- end user input ------------------------------


    
    filename_d = ''.join(['_'.join(['thetao',realm_freq, exper_tag,period]),'.nc'])
    datafile_d = join(datadir,model_exp,filename_d)
    fhd = Dataset(datafile_d, mode='r')

    filename_v = ''.join(['_'.join(['volcello_fx',exper_tag,'r0i0p0']),'.nc'])
    datafile_v = join(datadir,model_exp,filename_v)
    fhv = Dataset(datafile_v, mode='r')

    filename_a = ''.join(['_'.join(['areacello_fx',exper_tag,'r0i0p0']),'.nc'])
    datafile_a = join(datadir,model_exp,filename_a)
    fha = Dataset(datafile_a, mode='r')

    
    startdepth = float(layer[0])
    stopdepth  = float(layer[1])

    # get time attributes from input file
    time_in = fhd.variables['time']
    units = time_in.units # from original data
    calendar = time_in.calendar # from original data

    # do calculations
    ohc,time,lat,lon = depth_average_OHC(fhd,fhv,fha,startdepth,stopdepth,time_avg)

    
    # -------------------------------------
    # Save the data in a LMR-ready .nc file
    # -------------------------------------
    
    missing_val = -9999.0

    if time_avg:    
        realm_freq = 'Oann' # annual only for now
        
    
    varname = 'ohc'
    levname = ''.join(['-'.join([str(int(startdepth)),str(int(stopdepth))]),'m'])
    fname = ''.join(['_'.join([varname,levname,realm_freq,exper_tag,period]),'.nc'])
    outfilename = join(datadir,model_exp,fname)
    outfile = Dataset(outfilename, 'w', format='NETCDF4')

    # define dimensions
    ntime,nlat,nlon = ohc.shape

    outfile.createDimension('time', ntime)
    outfile.createDimension('lat', nlat)
    outfile.createDimension('lon', nlon)

    # define variables & upload the data to file
    # ------------------------------------------
    # time
    # ----

    # convert time from datetime object to integer
    time_to_file = date2num(time,units=units,calendar=calendar)
    
    time = outfile.createVariable('time', 'i', ('time',))
    time.description = 'time'
    time.long_name = 'time'
    time.standard_name = 'time'

    time.units = units
    time.calendar = calendar

    # lat
    # ---
    latf = outfile.createVariable('lat', 'f', ('lat','lon'))
    latf.description = 'latitude'
    latf.long_name = 'latitude coordinate'
    latf.standard_name = 'latitude'
    latf.units = 'degrees_north'

    # lon
    # ---
    lonf = outfile.createVariable('lon', 'f', ('lat','lon'))
    lonf.description = 'longitude'
    lonf.long_name = 'longitude coordinate'
    lonf.standard_name = 'longitude'
    lonf.units = 'degrees_east'

    # ohc
    ohcf = outfile.createVariable('ohc', 'f8', ('time', 'lat','lon'),fill_value=missing_val)
    ohcf.description = 'Gridded ocean heat content in %s layer' %(levname)
    ohcf.long_name = 'Ocean heat content'
    ohcf.standard_name = 'Ocean heat content'
    ohcf.units='J m^-2'

    # set up masked array and corresponding missing_value attribute
    ohc_masked = np.ma.masked_equal(ohc,0.0)
    # Set fill_value to -9999.0
    np.ma.set_fill_value(ohc_masked, missing_val)
    # add missing_value attribute
    ohcf.missing_value = missing_val
    
    
    # upload the data to file
    time[:] = time_to_file
    latf[:] = lat 
    lonf[:] = lon
    ohcf[:] = ohc_masked

    
    # Closing the file
    outfile.close()
        
    
    return

# ---------------------------- end of main -------------------------------------
# ------------------------------------------------------------------------------


def depth_average_OHC(filedata,filevol,filearea,startdepth,stopdepth,time_avg=None):
    """
    Inputs :
    ------
    filedata   : file handler for the (ocean potential temperatude) data
    filevol    : file handler for the (ocean grid) volume of cells
    filearea   : file handler for the (ocean grid) area of cells
    startdepth : shallow limit of region to compute heat content
    stopdepth  : deep limit
    time_avg   : whether or not to time average and over which period

    # Outputs
    # ohc            : 2D field of ocean heat content density in J/m^2
    # time, lat, lon : obtained from model output

    """

    # specific heat salt water (J/kg/K)
    cp_sw_mks = 3850.
    # density of salt water (kg/m^3)
    rho_sw_mks = 1025.
    
    # Boundaries between ocean levels in m. Use this to interpolate over depths
    lev_bnds = filedata.variables['lev_bnds'][:]

    # Shape of filevol.variables['volcello'] is (depth, lat, lon)
    # This is the volume of ocean cells in m^3
    volcello = filevol.variables['volcello']

    ndepth,nlat,nlon = volcello.shape

    # This is the area of ocean cells in m^2
    areacello = filearea.variables['areacello']

    
    # Make a field of weights. These will be equal to the grid box volume above the vertical level
    # stoplev that includes the value of user-specified bottom boundary depth, and 0 below dlev. At dlev,
    # the weights will be equal to grid box volumes multiplied by fraction of depth spanned by stopdepth
    # over the grid box at dlev.
    wt = np.copy(volcello[:])
    
    # Eliminate missing values (land) (force weights to zero)
    if hasattr(volcello,'missing_value'):
        missing = volcello.missing_value
    elif hasattr(volcello,'_FillValue'):
        missing = volcello.missing_value
    else:
        raise SystemExit('ERROR: Missing values undefined. Cannot continue. Exiting!')

    wt[wt==missing] = 0. 

    
    if stopdepth < lev_bnds[-1,1]:
        # All levels whose shallower bounds are greater than stopdepth are set to 0 
        wt[lev_bnds[:,0] > stopdepth,:,:] = 0.
        # Finding dlev (which includes stopdepth) as the first level whose deeper boundary is larger than stopdepth
        dlev = np.argmax(lev_bnds[:,1] > stopdepth)
        # determine the length of overlap between the level indexed by dlev and the distance over which we are integrating
        d_in_dlev = stopdepth-lev_bnds[dlev,0]
        # determine the fraction of flev that we want to include in the calculation
        frac_in_dlev = d_in_dlev/(lev_bnds[dlev,1]-lev_bnds[dlev,0])
        wt[dlev,:,:] = frac_in_dlev*volcello[dlev,:,:]
        
    if startdepth > lev_bnds[0,0]:
        # All levels whose deeper bounds are greater than startdepth are set to 0 
        wt[lev_bnds[:,1]>startdepth,:,:] = 0

        # Finding flev (which includes stopdepth) as the level above the first level whose shallower boundary is larger than startdepth
        flev = np.argmax(lev_bnds[:,0] > startdepth) - 1

        # determine the length of overlap between the level indexed by flev and the distance over which we are integrating
        d_in_flev = startdepth - lev_bnds[flev,1]

        # determine the fraction of flev that we want to include in the calculation
        frac_in_flev = d_in_flev/(lev_bnds[flev,1]-lev_bnds[flev,0])
        wt[flev,:,:] = frac_in_flev*volcello[flev,:,:]

    
    # times in the simulation
    time = filedata.variables['time']
    ntime, = time.shape 
    time_inds = np.arange(ntime)

    if hasattr(time, 'calendar'):
        dates = num2date(time[:], time.units, calendar=time.calendar)
    else:
        dates = num2date(time[:], time.units)

    
    # The heat content in Joules at time index t is given by
    # wt[:]*(filedata.variables['thetao'][t,:,:,:]))*rho_sw_mks*cp_sw_mks
    # Output in J/m2 is obtained by dividing by cell areas (areacello) 

    # NB thetao is in K
    thetao = filedata.variables['thetao']
    
    # declare output array
    ohc = np.nan*np.empty([ntime,nlat,nlon])

    # breaking down calculations: applying vectorized calculations over array slices (along time dimension)
    # speeds up compared to looping over single time elements
    # while vectorized calculations over entire array lead to "memory error" on my machine (RT)

    inter = 120 # nb of indices per time slice
    
    ntimeint = int(ntime / inter)
    # looping over time slices
    for i in range(ntimeint):
        ibeg = i*inter
        iend = ibeg+inter
        print(('%d : times from %s to %s' %(i, str(dates[ibeg]), str(dates[iend]))))
        ohc[ibeg:iend,:,:] = np.sum(wt*thetao[ibeg:iend,:,:,:],1)*rho_sw_mks*cp_sw_mks/areacello

    # do the remaining (residual) array elements not included in the interval chunks above
    time_resids = np.arange(ntimeint*inter,ntime)
    # looping over single time elements
    for i in time_resids:
        print('single time : %s' %str(dates[i]))
        ohc[i,:,:] = np.sum(wt*thetao[i,:,:,:],0)*rho_sw_mks*cp_sw_mks/areacello

        
    """ amrhein code ...
    # The full field is
    ohcFLD = np.nan*np.empty([ntime,wt.shape[1],wt.shape[2]])
    for ii in time_inds:
        if doAnn:
            ohcFLD[ii,:,:] = np.sum(wt* np.mean(filedata.variables['thetao'][(ii*12):((ii+1)*12),:,:,:],0),0)*rho_sw_mks*cp_sw_mks
            time = filedata.variables['time'][5::12]
        else:
            ohcFLD[ii,:,:,ii] = wt*(filedata.variables['thetao'][ii,:,:,:])*rho_sw_mks*cp_sw_mks
            time = filedata.variables['time']
        print ii
    """


    # ------------------------------------------------------------
    # do time (annual or other) averaging here, if option selected
    if time_avg:

        avg_def = list(time_avg.keys())[0]
        avg_seq = time_avg[avg_def]
        
        if hasattr(time, 'calendar'):
            dates = num2date(time[:], time.units, calendar=time.calendar)
        else:
            dates = num2date(time[:], time.units)
            
        dates_list = [datetime(d.year, d.month, d.day,
                                  d.hour, d.minute, d.second)
                      for d in dates]

        # List years available in dataset and sort
        years_all = [d.year for d in dates_list]
        years     = list(set(years_all)) # 'set' used to retain unique values in list
        years.sort() # sort the list
        ntime = len(years)
        datesYears = np.array([datetime(y,1,1,0,0) for y in years])

        print('Averaging over month sequence:', avg_seq)
        year_current = [m for m in avg_seq if m>0 and m<=12]
        year_before  = [abs(m) for m in avg_seq if m < 0]        
        year_follow  = [m-12 for m in avg_seq if m > 12]
        
        # declare output array
        ohc_out = np.nan*np.empty([ntime,nlat,nlon])

        # Loop over years in dataset
        for i in range(ntime):
            tindsyr   = [k for k,d in enumerate(dates) if d.year == years[i]    and d.month in year_current]
            tindsyrm1 = [k for k,d in enumerate(dates) if d.year == years[i]-1. and d.month in year_before]
            tindsyrp1 = [k for k,d in enumerate(dates) if d.year == years[i]+1. and d.month in year_follow]
            indsyr = tindsyrm1+tindsyr+tindsyrp1
        
            ohc_out[i,:,:] = np.nanmean(ohc[indsyr],axis=0)

            time_out = datesYears
            
    else:
        ohc_out = ohc
        time_out = dates

        
    lat = filedata.variables['lat'][:]
    lon = filedata.variables['lon'][:]

    return ohc_out, time_out, lat, lon


# ==============================================================================
if __name__ == '__main__':
    main()
