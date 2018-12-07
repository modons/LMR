"""
Module: LMR_convertNPZtoNETCDF.py

Purpose: Converts LMR output from .npz files to netcdf files. 
         Collects data of specified variable from .npz LMR output files 
         in the various MC directories (r*) and writes out the data
         in netcdf file(s) located in the experiment main directory.  

Originator: Robert Tardif | Univ. of Washington, Dept. of Atmospheric Sciences
                          | April 2016

Revisions: 
         - Output to netcdf using of 20CRv2 variable names and conventions 
           whenever possible.
         - Added capabilities and streamlined processing by enabling output 
           of full ensemble, or subsample of ensemble members or ensemble mean 
           and spread, or ensemble mean only.
           Processing of global mean temperature output (gmt) is also 
           incorporated here. 
           *** This script therefore is meant to replace the 
               LMR_convertNPZtoNETCDF_fullfield.py and 
               LMR_convertNPZtoNETCDF_gmt.py scripts. ***
           [R. Tardif, U. of Washington - May 2018]

"""
import os, glob, re
import numpy as np
from netCDF4 import Dataset, num2date
import time as clock
import subprocess

Nsample = None

# ------------------------------------------------
# --- Begin section of user-defined parameters ---

# -- root name of directory where the output of LMR experiments are located --
datadir = '/home/disk/kalman3/rtardif/LMR/output'

# -- name of the experiment --
nexp = 'test'

# -- Which type of output to netcdf file(s) --
#    uncomment one of the lines below to choose
archive_type = 'ensemble_mean'
#archive_type = 'ensemble_mean_spread'
#archive_type = 'ensemble_subsample'; Nsample = 10
#archive_type = 'ensemble_full'

# Select MC realizations from which to extract the data
#  All recon. MC realizations ( MCset = None )
#  or over a custom selection ( MCset = (begin,end) )
#  ex. MCset = (0,0)    -> only the first MC run
#      MCset = (0,10)   -> the first 11 MC runs (from 0 to 10 inclusively)
#      MCset = (80,100) -> the 80th to 100th MC runs (21 realizations)
MCset = None
#MCset = (0,10)

dataset_tag = 'NOAA Last Millennium Reanalysis version 1 Annual Averages'


# Dictionary containing definitions of variables that handled by this code
# ------------------------------------------------------------------------
#      LMR variable names             Variable names/attributes in netcdf
var_desc = \
    {
        'tas_sfc_Amon'              : {'variable_name': 'air',
                                       'long_name': 'Air Temperature at Surface',
                                       'standard_name': 'air_temperature',
                                       'var_desc': 'Air temperature',
                                       'level_desc': '2m',
                                       'units': 'degK',
                                       'GRIB_id': '11',
                                       'GRIB_name': 'TMP',
                                       'valid_range': (-20.,20.),
                                       'dtype': 'f',
                                      },
        'psl_sfc_Amon'              : {'variable_name': 'prmsl',
                                       'long_name': 'Pressure at Mean Sea Level',
                                       'standard_name': 'air_pressure',
                                       'var_desc': 'Mean Sea Level Pressure',
                                       'level_desc': 'Mean Sea Level',
                                       'units': 'Pa',
                                       'GRIB_id': '2',
                                       'GRIB_name': 'PRMSL',
                                       'valid_range': (-2000.,2000.),
                                       'dtype': 'f',
                                      },
        'pr_sfc_Amon'               : {'variable_name': 'prate',
                                       'long_name': 'Precipitation Rate at Surface',
                                       'standard_name': 'precipitation_flux',
                                       'var_desc': 'Precipitation rate at surface',
                                       'level_desc': 'Surface',
                                       'units': 'kg/m^2/s',
                                       'GRIB_id': '59',
                                       'GRIB_name': 'PRATEsfcAvg',
                                       'valid_range': (-2.e-4,2.e-4),
                                       'dtype': 'f',
                                      },
        'prw_int_Amon'              : {'variable_name': 'pr_wtr',
                                       'long_name': 'Precipitable Water for Entire Atmosphere',
                                       'standard_name': 'atmosphere_water_vapor_content',
                                       'var_desc': 'Precipitable water',
                                       'level_desc': 'Surface',
                                       'units': 'kg/m^2',
                                       'GRIB_id': '54',
                                       'GRIB_name': 'PWATeatm',
                                       'valid_range': (-20.,20.),
                                       'dtype': 'f',
                                      },
        'scpdsipm_sfc_Amon'         : {'variable_name': 'pdsi',
                                       'long_name': 'Self-Calibrated Palmer Drought Severity Index (Penman-Monteith evapotranspiration)',
                                       'standard_name': 'self_calibrated_palmer_drought_severity_index',
                                       'var_desc': 'Self-Calibrated Palmer Drought Severity Index',
                                       'level_desc': 'Surface',
                                       'units': '',
                                       'GRIB_id': '',
                                       'GRIB_name': '',
                                       'valid_range': (-40.,40.),
                                       'dtype': 'f',
                                      },
        'zg_500hPa_Amon'            : {'variable_name': 'hgt500',
                                       'long_name': 'Geopotential Height on 500hPa pressure level',
                                       'standard_name': 'geopotential_height_500hPa',
                                       'var_desc': 'Geopotential height at 500hPa',
                                       'level_desc': '500hPa',
                                       'units': 'm',
                                       'GRIB_id': '7',
                                       'GRIB_name': 'HGT',
                                       'valid_range': (-200.,200.),
                                       'dtype': 'f',
                                      },
        'hfss_sfc_Amon'             : {'variable_name': 'shtfl',
                                       'long_name': 'Sensible Heat Net Flux at surface',
                                       'standard_name': 'surface_upward_sensible_heat_flux',
                                       'var_desc': 'Sensible heat flux at the surface',
                                       'level_desc': 'Surface',
                                       'units': 'W/m^2',
                                       'GRIB_id': '122',
                                       'GRIB_name': 'SHTFLsfcAvg',
                                       'valid_range': (-100.,100.),
                                       'dtype': 'f',
                                      },        
        'hfls_sfc_Amon'             : {'variable_name': 'lhtfl',
                                       'long_name': 'Latent Heat Net Flux at surface',
                                       'standard_name': 'surface_upward_latent_heat_flux',
                                       'var_desc': 'Latent heat flux at the surface',
                                       'level_desc': 'Surface',
                                       'units': 'W/m^2',
                                       'GRIB_id': '121',
                                       'GRIB_name': 'LHTFLsfcAvg',
                                       'valid_range': (-100.,100.),
                                       'dtype': 'f',
                                      },
        'rlds_sfc_Amon'             : {'variable_name': 'dlwrfSFC',
                                       'long_name': 'Downward Longwave Radiation Flux at surface',
                                       'standard_name': 'surface_downward_longwave_radiation_flux',
                                       'var_desc': 'Downward longwave radiation flux at the surface',
                                       'level_desc': 'Surface',
                                       'units': 'W/m^2',
                                       'GRIB_id': '205',
                                       'GRIB_name': 'DLWRFsfcAvg',
                                       'valid_range': (-100.,100.),
                                       'dtype': 'f',
                                      },
        'rlus_sfc_Amon'             : {'variable_name': 'ulwrfSFC',
                                       'long_name': 'Upward Longwave Radiation Flux at surface',
                                       'standard_name': 'surface_upward_longwave_radiation_flux',
                                       'var_desc': 'Upward longwave radiation flux at the surface',
                                       'level_desc': 'Surface',
                                       'units': 'W/m^2',
                                       'GRIB_id': '212',
                                       'GRIB_name': 'ULWRFsfcAvg',
                                       'valid_range': (-100.,100.),
                                       'dtype': 'f',
                                      },
        'rsds_sfc_Amon'             : {'variable_name': 'dswrfSFC',
                                       'long_name': 'Downward Solar Radiation Flux at surface',
                                       'standard_name': 'surface_downward_solar_radiation_flux',
                                       'var_desc': 'Downward solar radiation flux at the surface',
                                       'level_desc': 'Surface',
                                       'units': 'W/m^2',
                                       'GRIB_id': '204',
                                       'GRIB_name': 'DSWRFsfcAvg',
                                       'valid_range': (-100.,100.),
                                       'dtype': 'f',
                                      },
        'rsus_sfc_Amon'             : {'variable_name': 'uswrfSFC',
                                       'long_name': 'Upward Solar Radiation Flux at surface',
                                       'standard_name': 'surface_upward_solar_radiation_flux',
                                       'var_desc': 'Upward solar radiation flux at the surface',
                                       'level_desc': 'Surface',
                                       'units': 'W/m^2',
                                       'GRIB_id': '211',
                                       'GRIB_name': 'USWRFsfcAvg',
                                       'valid_range': (-100.,100.),
                                       'dtype': 'f',
                                      },
        'rlut_toa_Amon'             : {'variable_name': 'ulwrfTOA',
                                       'long_name': 'Upward Longwave Radiation Flux at nominal top of atmosphere',
                                       'standard_name': 'toa_upward_longwave_radiation_flux',
                                       'var_desc': 'Upward longwave radiation flux at TOA',
                                       'level_desc': 'Nominal top of atmosphere',
                                       'units': 'W/m^2',
                                       'GRIB_id': '212',
                                       'GRIB_name': 'ULWRFntatAvg',
                                       'valid_range': (-100.,100.),
                                       'dtype': 'f',
                                      },
        'rsut_toa_Amon'             : {'variable_name': 'uswrfTOA',
                                       'long_name': 'Upward Solar Radiation Flux at nominal top of atmosphere',
                                       'standard_name': 'toa_upward_solar_radiation_flux',
                                       'var_desc': 'Upward solar radiation flux at TOA',
                                       'level_desc': 'Nominal top of atmosphere',
                                       'units': 'W/m^2',
                                       'GRIB_id': '211',
                                       'GRIB_name': 'USWRFntatAvg',
                                       'valid_range': (-100.,100.),
                                       'dtype': 'f',
                                      },
        'rsdt_toa_Amon'             : {'variable_name': 'dswrfTOA',
                                       'long_name': 'Downward Solar Radiation Flux at nominal top of atmosphere',
                                       'standard_name': 'toa_downward_solar_radiation_flux',
                                       'var_desc': 'Downward solar radiation flux at TOA',
                                       'level_desc': 'Nominal top of atmosphere',
                                       'units': 'W/m^2',
                                       'GRIB_id': '204',
                                       'GRIB_name': 'DSWRFntatAvg',
                                       'valid_range': (-10.,10.),
                                       'dtype': 'f',
                                      },
        'tos_sfc_Omon'              : {'variable_name': 'sst',
                                       'long_name': 'Sea Surface Water Temperature',
                                       'standard_name': 'sea_surface_water_temperature',
                                       'var_desc': 'Sea surface water temperature',
                                       'level_desc': 'surface',
                                       'units': 'degK',
                                       'GRIB_id': '',
                                       'GRIB_name': 'WTMP',
                                       'valid_range': (-20.,20.),
                                       'dtype': 'f',
                                      },
        'sos_sfc_Omon'              : {'variable_name': 'sss',
                                       'long_name': 'Sea Surface Water Salinity',
                                       'standard_name': 'sea_surface_water_salinity',
                                       'var_desc': 'Sea surface water salinity',
                                       'level_desc': 'surface',
                                       'units': 'psu',
                                       'GRIB_id': '',
                                       'GRIB_name': '',
                                       'valid_range': (-10.,10.),
                                       'dtype': 'f',
                                      },
        'ohc_0-700m_Omon'           : {'variable_name': 'ohc0-700',
                                       'long_name': 'Ocean Heat Content in the 0-700m layer',
                                       'standard_name': 'ocean_heat_content_0-700m',
                                       'var_desc': 'Ocean heat content in 0-700m layer',
                                       'level_desc': '0-700m layer',
                                       'units': 'J/m^2',
                                       'GRIB_id': '',
                                       'GRIB_name': '',
                                       'valid_range': (-1.e11,1.e11),
                                       'dtype': 'f',
                                      },
        'ohc_0-2000m_Omon'          : {'variable_name': 'ohc0-2000',
                                       'long_name': 'Ocean Heat Content in the 0-2000m layer',
                                       'standard_name': 'ocean_heat_content_0-2000m',
                                       'var_desc': 'Ocean heat content in 0-2000m layer',
                                       'level_desc': '0-2000m layer',
                                       'units': 'J/m^2',
                                       'GRIB_id': '',
                                       'GRIB_name': '',
                                       'valid_range': (-1.e12,1.e12),
                                       'dtype': 'f',
                                      },
        'gmt_ensemble'              : {'variable_name': 'gmt',
                                       'long_name': 'Global-Mean Air Temperature at Surface',
                                       'standard_name': 'global_mean_air_temperature',
                                       'var_desc': 'Air temperature',
                                       'level_desc': '2m',
                                       'units': 'degK',
                                       'GRIB_id': '',
                                       'GRIB_name': '',
                                       'valid_range': (-2.,2.),
                                       'dtype': 'f',
                                      },
        'AMOCindex_Omon'            : {'variable_name': 'AMOCindex',
                                       'long_name': 'AMOC index',
                                       'standard_name': 'AMOCindex',
                                       'var_desc': 'Maximum value of Atlantic overturning streamfunction between 30N and 70N and between depths of 500m and 2000m',
                                       'level_desc': '',
                                       'units': 'kg s-1',
                                       'GRIB_id': '',
                                       'GRIB_name': '',
                                       'valid_range': (-1.e10,1.e10),
                                       'dtype': 'f',
                                      },
        'AMOC26Nmax_Omon'            : {'variable_name': 'AMOC26Nmax',
                                       'long_name': 'AMOC maximum at 26N',
                                       'standard_name': 'AMOC26Nmax',
                                       'var_desc': 'Maximum value of Atlantic overturning streamfunction in water column at 26N',
                                       'level_desc': '',
                                       'units': 'kg s-1',
                                       'GRIB_id': '',
                                       'GRIB_name': '',
                                       'valid_range': (-1.e10,1.e10),
                                       'dtype': 'f',
                                      },
        'AMOC26N1000m_Omon'          : {'variable_name': 'AMOC26N1000m',
                                       'long_name': 'AMOC at 26N and 1000m depth',
                                       'standard_name': 'AMOC26N1000m',
                                       'var_desc': 'Atlantic overturning streamfunction at 26N and 1000m depth',
                                       'level_desc': '1000m',
                                       'units': 'kg s-1',
                                       'GRIB_id': '',
                                       'GRIB_name': '',
                                       'valid_range': (-1.e10,1.e10),
                                       'dtype': 'f',
                                      },
        'AMOC45N1000m_Omon'          : {'variable_name': 'AMOC45N1000m',
                                       'long_name': 'AMOC at 45N and 1000m depth',
                                       'standard_name': 'AMOC45N1000m',
                                       'var_desc': 'Atlantic overturning streamfunction at 45N and 1000m depth',
                                       'level_desc': '1000m',
                                       'units': 'kg s-1',
                                       'GRIB_id': '',
                                       'GRIB_name': '',
                                       'valid_range': (-1.e10,1.e10),
                                       'dtype': 'f',
                                      },
    }


# attempt additional compression (Note: possible issues with use of FillValue depending of NCO version)
run_ncpdq = False

# --- End section of user-defined parameters ---
# ----------------------------------------------

def main():

    begin_time = clock.time()
    
    #missing_val = np.nan
    missing_val = -9.96921e+36 # value used in 20CR files
    
    expdir = datadir + '/'+nexp

    # where the netcdf files are to be generated
    outdir = expdir

    print('\n Getting information on Monte-Carlo realizations...\n')

    dirs = glob.glob(expdir+"/r*")
    # keep names of MC directories (i.r. "r...") only 
    mcdirs = [item.split('/')[-1] for item in dirs]
    # Make sure list is properly sorted
    mcdirs = natural_sort(mcdirs)
    # number of MC realizations found
    niters = len(mcdirs) 

    print('mcdirs available:' + str(mcdirs))
    print('niters = ' + str(niters))

    # user-modified selection
    if MCset:
        mcdirset = mcdirs[MCset[0]:MCset[1]+1]
    else:
        mcdirset = mcdirs
    niters = len(mcdirset)

    print('mcdirs selected:' + str(mcdirset))
    print('niters = ' + str(niters))
    
    print('\n Getting information on files and reconstructed variables...\n')

    # look in first "mcdirset" only. It should be the same for all. 
    workdir = expdir+'/'+mcdirset[0]

    # Assess data files present in directory and determine available input
    # --------------------------------------------------------------------
    allfiles = glob.glob(workdir+'/ensemble_*.npz')
    # strip directory name, keep file names only
    listfiles = [item.split('/')[-1] for item in allfiles]

    # Assume file name structure as: ensemble_<<type>>_<<variable name>>.npz
    # where <<type>> can be "full", "subsample", "mean" or "variance"
    availtypes = list(set([item.split('_')[0]+'_'+item.split('_')[1] for item in listfiles]))

    # Check available input vs. desired output
    if archive_type == 'ensemble_full':

        if 'ensemble_full' in availtypes:
            input_type = 'ensemble_full'
            statistic = ('Ensemble Members', )
        else:
            print('Full ensemble archiving selected, but full ensemble not available in input! '
                  'Cannot proceed.')
            raise SystemExit(1)

    elif archive_type == 'ensemble_subsample':
        if 'ensemble_subsample' in availtypes:
            input_type = 'ensemble_subsample'
        elif 'ensemble_full' in availtypes:
            input_type = 'ensemble_full'
        else:
            print('Archiving of subset of ensemble members selected, but needed full ensemble'
                  ' not available! Cannot proceed.')
            raise SystemExit(1)
        statistic = ('Ensemble Members (subset)', )

    elif archive_type == 'ensemble_mean_spread':
        if ('ensemble_mean' in availtypes) and \
           ('ensemble_variance' in availtypes):
            input_type = ('ensemble_mean', 'ensemble_variance')
        elif 'ensemble_full' in availtypes:
            input_type = 'ensemble_full'
        else:
            print('Archiving of ensemble mean & spread selected, but necessary input'
                  ' (full ensemble, or ensemble-mean and variance) not available.'
                  ' Cannot proceed.')
            raise SystemExit(1)
        statistic = ('Ensemble Mean', 'Ensemble Spread')

    elif archive_type == 'ensemble_mean':
        if 'ensemble_mean' in availtypes:
            input_type = 'ensemble_mean'
        elif 'ensemble_full' in availtypes:
            input_type = 'ensemble_full'
        else:
            print('Archiving of ensemble mean selected, but necessary input'
                  ' (ensemble-mean itself or full ensemble) not available.'
                  ' Cannot proceed.')
            raise SystemExit(1)
        statistic = ('Ensemble Mean', )

    else:
        print('Unrecognized archiving selection. Allowed options are:  ensemble_full,'
              ' ensemble_subsample, ensemble_mean_spread or ensemble_mean')
        raise SystemExit(1)

    # determine which array(s) to extract from npz files
    if input_type == 'ensemble_full':
        npfile_to_extract = ('xa_ens',)
    elif input_type == 'ensemble_subsample':
        npfile_to_extract = ('xa_subsample',)
    elif input_type == ('ensemble_mean', 'ensemble_variance'):
        npfile_to_extract = ('xam', 'xav')
    elif input_type == 'ensemble_mean':
        npfile_to_extract = ('xam',)
    else:
        print('ERROR: Non-valid option specified for input data type: %s' %input_type)
        raise SystemExit(1)


    # look for files corresponding to desired input_type
    if type(input_type) is tuple:
        input_file = input_type[0]
    else:
        input_file = input_type
    listdirfiles = glob.glob(workdir+'/'+input_file+'_*')

    # including gmt_ensemble
    listdirfiles.extend(glob.glob(workdir+'/gmt_ensemble.npz'))
    # strip directory name, keep file name only
    listfiles = [item.split('/')[-1] for item in listdirfiles]
    # strip everything but variable name
    listvars = [(item.replace(input_file+'_','')).replace('.npz','') for item in listfiles]

    print('Variables:', listvars, '\n')

    # Loop over variables to process
    for var in listvars:
        print('\n Variable:', var)

        if var == 'gmt_ensemble': npfile_to_extract = (var,)

        if var not in list(var_desc.keys()):
            print(' ***WARNING: Variable %s does not have a corresponding entry in variable definitions'
                  ' (var_desc) at the top of this program. Please make necessary edits to have this'
                  ' variable included in the format conversion output' %var)
            continue

        missing_val = np.array(missing_val, dtype=var_desc[var]['dtype'])
        
        # Loop over Monte Carlo realizations
        r = 0
        for dir in mcdirset:
            print('  MCdir:', dir)

            if var == 'gmt_ensemble':
                fname = expdir+'/'+dir+'/'+var+'.npz'
                time_name = 'recon_times'
                # reset archive and input types appropriate for this variable
                archive_type_var = 'ensemble_full'
                input_type = 'ensemble_full'
            else:
                fname = expdir+'/'+dir+'/'+input_file+'_'+var+'.npz'
                time_name = 'years'
                archive_type_var = archive_type

            # load file
            npzfile = np.load(fname)

            # extract reconstructed field
            field_values = npzfile[npfile_to_extract[0]]

            if r == 0: # first realization
                npzcontent = npzfile.files
                print('  file contents:', npzcontent)

                # get the years in the reconstruction ... for some reason stored in an array of strings ...
                years_str =  npzfile[time_name]

                # convert to array of floats
                years = np.asarray([float(item) for item in years_str])

                # Determine ensemble size
                Nens = None
                if 'nens' in npzcontent:
                    Nens = npzfile['nens']

                # Determine type of variable, get spatial coordinates if present
                if 'lat' in npzcontent and 'lon' in npzcontent:
                    field_type = '2D:horizontal'
                    print('  field type:', field_type)
                    # get lat/lon data
                    lat2d = npzfile['lat']
                    lon2d = npzfile['lon']
                    lat1d = lat2d[:,0]
                    lon1d = lon2d[0,:]
                    print('  nlat/nlon=', lat1d.shape, lon1d.shape)
                    nlat, = lat1d.shape
                    nlon, = lon1d.shape
                elif 'lat' in npzcontent and 'lev' in npzcontent:
                    field_type = '2D:meridional_vertical'
                    print('  field type:', field_type)
                    # get lat/lev data
                    lat2d = npzfile['lat']
                    lev2d = npzfile['lev']
                    lat1d = lat2d[:,0]
                    lev1d = lev2d[0,:]
                    print('  nlat/nlev=', lat1d.shape, lev1d.shape)
                    nlat, = lat1d.shape
                    nlev, = lev1d.shape
                elif 'lat' in npzcontent and ('lon' not in npzcontent and 'lev' not in npzcontent):
                    # lat is the only spatial coord
                    field_type = '1D:meridional'
                    print('  field type:', field_type)
                    # get lat data
                    lat2d = npzfile['lat']
                    lat1d = lat2d[:,0]
                    print('  nlat=', lat1d.shape)
                    nlat, = lat1d.shape
                elif 'lat' not in npzcontent and 'lon' not in npzcontent and 'lev' not in npzcontent:
                    # no spatial coordinate, must be a scalar (time series)
                    field_type='0D:time_series'
                    print('  field type:', field_type)
                else:
                    print('Cannot handle this variable yet! Variable of unrecognized dimensions... Exiting!')
                    raise SystemExit(1)

                # declare master array(s) that will contain output data from
                # all the Monte-Carlo realizations, depending on archive_type
                ntime = years.shape[0]
                if archive_type_var == 'ensemble_full':
                    if  field_type == '2D:horizontal':
                        mc_ens = np.zeros(shape=[1, niters, ntime, nlat, nlon, Nens])
                        axis_ens = 3
                    elif field_type == '2D:meridional_vertical':
                        mc_ens = np.zeros(shape=[1, niters, ntime, nlat, nlev, Nens])
                        axis_ens = 3
                    elif field_type == '1D:meridional':
                        mc_ens = np.zeros(shape=[1, niters, ntime, nlat, Nens])
                        axis_ens = 2
                    elif field_type == '0D:time_series':
                        if not Nens:
                            print(field_values.shape)
                            _, Nens = field_values.shape                        
                        mc_ens = np.zeros(shape=[1, niters, ntime, Nens])
                        axis_ens = 1

                    mc_min = np.zeros(shape=[1, niters])
                    mc_max = np.zeros(shape=[1, niters])

                elif archive_type_var == 'ensemble_subsample':
                    if  field_type == '2D:horizontal':
                        mc_ens = np.zeros(shape=[1, niters, ntime, nlat, nlon, Nsample])
                        axis_ens = 3
                    elif field_type == '2D:meridional_vertical':
                        mc_ens = np.zeros(shape=[1, niters, ntime, nlat, nlev, Nsample])
                        axis_ens = 3
                    elif field_type == '1D:meridional':
                        mc_ens = np.zeros(shape=[1, niters, ntime, nlat, Nsample])
                        axis_ens = 2
                    elif field_type == '0D:time_series':
                        if not Nens:
                            print(field_values.shape)
                            _, Nens = field_values.shape                        
                        mc_ens = np.zeros(shape=[1, niters, ntime, Nsample])
                        axis_ens = 1

                    mc_min = np.zeros(shape=[1, niters])
                    mc_max = np.zeros(shape=[1, niters])

                elif archive_type_var == 'ensemble_mean_spread':
                    if  field_type == '2D:horizontal':
                        mc_ens   = np.zeros(shape=[2, niters, ntime, nlat, nlon])
                        axis_ens = 3
                    elif field_type == '2D:meridional_vertical':
                        mc_ens   = np.zeros(shape=[2, niters, ntime, nlat, nlev])
                        axis_ens = 3
                    elif field_type == '1D:meridional':
                        mc_ens   = np.zeros(shape=[2, niters, ntime, nlat])
                        axis_ens = 2
                    elif field_type == '0D:time_series':
                        mc_ens   = np.zeros(shape=[2, niters, ntime])
                        axis_ens = 1

                    mc_min = np.zeros(shape=[2, niters])
                    mc_max = np.zeros(shape=[2, niters])

                elif archive_type_var == 'ensemble_mean':
                    if  field_type == '2D:horizontal':
                        mc_ens   = np.zeros(shape=[1, niters, ntime, nlat, nlon])
                        axis_ens = 3
                    elif field_type == '2D:meridional_vertical':
                        mc_ens   = np.zeros(shape=[1, niters, ntime, nlat, nlev])
                        axis_ens = 3
                    elif field_type == '1D:meridional':
                        mc_ens   = np.zeros(shape=[1, niters, ntime, nlat])
                        axis_ens = 2
                    elif field_type == '0D:time_series':
                        mc_ens   = np.zeros(shape=[1, niters, ntime])
                        axis_ens = 1

                    mc_min = np.zeros(shape=[1, niters])
                    mc_max = np.zeros(shape=[1, niters])

            # if we need to extact another field from input (e.g. ensemble variance)
            if len(npfile_to_extract) > 1:

                # do we need to look in a separate file?
                if input_type == ('ensemble_mean', 'ensemble_variance'):
                    # get only file name
                    ftmp = fname.split('/')[-1]
                    # rename file to other part of input
                    ftmp2 = ftmp.replace(input_type[0], input_type[1])
                    # reform complete dir & file name 
                    fname2 = expdir+'/'+dir+'/'+ftmp2
                    # read the separate file
                    npzfile2 = np.load(fname2)
                    # extract the needed array
                    field_values2 = npzfile2[npfile_to_extract[1]]
                else:
                    field_values2 = npzfile[npfile_to_extract[1]]

            
            # Any data processing required ?
            if archive_type_var != input_type: # some processing required

                if archive_type_var == 'ensemble_subsample':
                    if input_type == 'ensemble_full':
                        # extract first Nsample members from full ensemble                    
                        indices = range(0,Nsample)
                        mc_ens[0,r,:] = np.take(field_values,indices,axis=axis_ens)

                        mc_min[0,r] = np.nanmin(mc_ens[0,r,:])
                        mc_max[0,r] = np.nanmax(mc_ens[0,r,:])
                        
                elif archive_type_var == 'ensemble_mean_spread':
                    if input_type == 'ensemble_full':
                        # calculate ensemble mean and standard-deviation from full ensemble
                        mc_ens[0,r,:] = np.mean(field_values, axis=axis_ens)
                        mc_ens[1,r,:] = np.std(field_values, axis=axis_ens)

                    elif input_type == ('ensemble_mean', 'ensemble_variance'):
                        # extract ensemble mean
                        mc_ens[0,r,:] = field_values
                        # calculate spread (standard-deviation) from variance
                        # (stored in field_values2)
                        mc_ens[1,r,:] = np.sqrt(field_values2)

                    mc_min[0,r] = np.nanmin(mc_ens[0,r,:])
                    mc_max[0,r] = np.nanmax(mc_ens[0,r,:])
                    mc_min[1,r] = np.nanmin(mc_ens[1,r,:])
                    mc_max[1,r] = np.nanmax(mc_ens[1,r,:])
                        
                elif archive_type_var == 'ensemble_mean':
                    if input_type == 'ensemble_full':
                        # calculate ensemble mean from full ensemble members
                        mc_ens[0,r,:] = np.mean(field_values, axis=axis_ens)

                        mc_min[0,r] = np.nanmin(mc_ens[0,r,:])
                        mc_max[0,r] = np.nanmax(mc_ens[0,r,:])
                        
            else: # no further processing needed
                mc_ens[0,r,:] = field_values

                mc_min[0,r] = np.nanmin(mc_ens[0,r,:])
                mc_max[0,r] = np.nanmax(mc_ens[0,r,:])

            # expected that any missing values in field_values, and now in mc_ens,
            # are indicated by NaNs. Change these to specified value to be written
            # in netcdf file
            arrshape = mc_ens.shape
            for i in range(arrshape[0]):
                indsmiss = np.isnan(mc_ens[i,r,:])
                mc_ens[i,r,indsmiss] = missing_val
                

            # next MC realization
            r = r + 1

            # end of loop on MC runs


        # Roll array to get dims as [...,time, niters, ...]
        mc_ens_outarr = np.swapaxes(mc_ens,1,2)


        # ----------------------------------------------------------------
        # Now writing the data out in netcdf file(s)
        # ----------------------------------------------------------------

        # number of files to write-out for current variable ...
        if archive_type_var == 'ensemble_mean_spread':
            nboutfiles = 2
            filename_suffix = ('ensemble_mean', 'ensemble_spread')
        else:
            nboutfiles = 1
            filename_suffix = (archive_type_var, )


        # loop over the files to generate
        for k in range(nboutfiles):

            # Create the netcdf file for the current variable
            varname = var_desc[var]['variable_name']
            outfile_nc = outdir+'/'+varname+'_MCruns_'+filename_suffix[k]+'.nc'
            outfile = Dataset(outfile_nc, 'w', format='NETCDF4')
            outfile.description = 'Last Millennium Reanalysis climate field reconstruction for variable: %s' % varname
            outfile.experiment = nexp

            # define dimensions
            ntime = years.shape[0]
            outfile.createDimension('time', ntime)
            outfile.createDimension('MCrun', niters)
            if archive_type_var == 'ensemble_full':
                if not Nens:
                    Nens = mc_ens_outarr.shape[-1]
                outfile.createDimension('members', Nens)
                outfile.comment = 'File contains full ensemble values for each Monte-Carlo reconstruction (MCrun)'

            elif archive_type_var == 'ensemble_subsample':
                outfile.createDimension('members', Nsample)
                outfile.comment = 'File contains values for a subsample of ensemble members for each Monte-Carlo reconstruction (MCrun)'

            elif archive_type_var == 'ensemble_mean_spread':
                if k == 0:
                    outfile.comment = 'File contains ensemble-mean values for each Monte-Carlo reconstruction (MCrun)'
                elif k == 1:
                    outfile.comment = 'File contains ensemble spread values for each Monte-Carlo reconstruction (MCrun)'

            elif archive_type_var == 'ensemble_mean':
                outfile.comment = 'File contains ensemble-mean values for each Monte-Carlo reconstruction (MCrun)'

            # create dimensions according to spatial coords.
            if field_type == '2D:horizontal':
                nlat  = lat1d.shape[0]
                nlon  = lon1d.shape[0]
                outfile.createDimension('lat', nlat)
                outfile.createDimension('lon', nlon)
            elif field_type == '2D:meridional_vertical':
                nlat  = lat1d.shape[0]
                nlev  = lev1d.shape[0]
                outfile.createDimension('lat', nlat)
                outfile.createDimension('lev', nlev)
            elif field_type == '1D:meridional':
                nlat  = lat1d.shape[0]
                outfile.createDimension('lat', nlat)
            else:
                # '0D:time_series': no need to pull in info on spatial coords.
                pass

            # Arrays to store min & max field values, to define variable actual range
            minval = np.min(mc_min[k,:])
            maxval = np.max(mc_max[k,:])
            
            # define variables & upload the data to file
            # ------------------------------------------

            # -- time --
            time_output = years*365. # time in nb of days (no leap years)
            time = outfile.createVariable('time', 'f8', ('time',))
            time.description = 'time'
            time.long_name = 'Time'
            time.standard_name = 'time'
            time.units = 'days since 0000-01-01 00:00:00'
            time.calendar = 'noleap'
            time.actual_range = np.array((np.min(time_output), np.max(time_output)))


            if field_type == '2D:horizontal':
                # lat
                lat = outfile.createVariable('lat', 'f', ('lat',))
                lat.description = 'latitude'
                lat.units = 'degrees_north'
                lat.long_name = 'Latitude'
                lat.standard_name = 'latitude'
                lat.axis = 'Y'
                lat.coordinate_defines = 'point'
                lat.actual_range = np.array((np.min(lat1d), np.max(lat1d)),dtype=np.float)

                # lon
                lon = outfile.createVariable('lon', 'f', ('lon',))
                lon.description = 'longitude'
                lon.units = 'degrees_east'
                lon.long_name = 'Longitude'
                lon.standard_name = 'longitude'
                lon.axis = 'X'
                lon.coordinate_defines = 'point'
                lon.actual_range = np.array((np.min(lon1d), np.max(lon1d)),dtype=np.float)

                # reconstructed field itself
                if (archive_type_var == 'ensemble_full') or (archive_type_var == 'ensemble_subsample'):
                    varout = outfile.createVariable(varname, var_desc[var]['dtype'], ('time','MCrun','lat','lon','members'),
                                                    fill_value=missing_val,
                                                    zlib=True,complevel=4,fletcher32=True)
                else:
                    varout = outfile.createVariable(varname, var_desc[var]['dtype'], ('time','MCrun','lat','lon'),
                                                    fill_value=missing_val,
                                                    zlib=True,complevel=4,fletcher32=True)
                varout.var_desc      = var_desc[var]['var_desc']
                varout.long_name     = var_desc[var]['long_name']
                varout.standard_name = var_desc[var]['standard_name']
                varout.units         = var_desc[var]['units']
                varout.level_desc    = var_desc[var]['level_desc']
                varout.statistic     = statistic[k]
                varout.GRIB_id       = var_desc[var]['GRIB_id']
                varout.GRIB_name     = var_desc[var]['GRIB_name']        
                varout.valid_range   = np.array(var_desc[var]['valid_range'],dtype=var_desc[var]['dtype'])
                varout.dataset       = dataset_tag
                varout.missing_value = missing_val
                varout.actual_range  = np.array((minval, maxval), dtype=var_desc[var]['dtype'])
                
                # upload this data to file
                lat[:]    = lat1d
                lon[:]    = lon1d
                time[:]   = time_output
                varout[:] = mc_ens_outarr[k,:]


            elif field_type == '2D:meridional_vertical':
                # lat
                lat = outfile.createVariable('lat', 'f', ('lat',))
                lat.description = 'latitude'
                lat.units = 'degrees_north'
                lat.long_name = 'Latitude'
                lat.standard_name = 'latitude'
                lat.axis = 'Y'
                lat.coordinate_defines = 'point'
                lat.actual_range = np.array((np.min(lat1d), np.max(lat1d)),dtype=np.float)

                # lev
                lev = outfile.createVariable('lev', 'f', ('lev',))
                lev.description = 'depth'
                lev.units = 'm'
                lev.long_name = 'Depth'
                lev.standard_name = 'depth'
                lev.axis = 'Z'
                lev.coordinate_defines = 'level'
                lev.actual_range = np.array((np.min(lev1d), np.max(lev1d)),dtype=np.float)

                # reconstructed field itself
                if (archive_type_var == 'ensemble_full') or (archive_type_var == 'ensemble_subsample'):
                    varout = outfile.createVariable(varname, var_desc[var]['dtype'], ('time','MCrun','lat','lev','members'),
                                                    fill_value=missing_val,
                                                    zlib=True,complevel=4,fletcher32=True)
                else:
                    varout = outfile.createVariable(varname, var_desc[var]['dtype'], ('time','MCrun','lat','lev'),
                                                    fill_value=missing_val,
                                                    zlib=True,complevel=4,fletcher32=True)
                varout.var_desc      = var_desc[var]['var_desc']
                varout.long_name     = var_desc[var]['long_name']
                varout.standard_name = var_desc[var]['standard_name']
                varout.units         = var_desc[var]['units']
                varout.level_desc    = var_desc[var]['level_desc']
                varout.statistic     = statistic[k]
                varout.GRIB_id       = var_desc[var]['GRIB_id']
                varout.GRIB_name     = var_desc[var]['GRIB_name']        
                varout.valid_range   = np.array(var_desc[var]['valid_range'],dtype=var_desc[var]['dtype'])
                varout.dataset       = dataset_tag
                varout.missing_value = missing_val
                varout.actual_range  = np.array((minval, maxval), dtype=var_desc[var]['dtype'])

                # upload the data to file
                lat[:]    = lat1d
                lev[:]    = lev1d
                time[:]   = time_output
                varout[:] = mc_ens_outarr[k,:]

            elif field_type == '1D:meridional':
                # lat
                lat = outfile.createVariable('lat', 'f', ('lat',))
                lat.description = 'latitude'
                lat.units = 'degrees_north'
                lat.long_name = 'Latitude'
                lat.standard_name = 'latitude'
                lat.axis = 'Y'
                lat.coordinate_defines = 'point'
                lat.actual_range = np.array((np.min(lat1d), np.max(lat1d)),dtype=np.float)

                # reconstructed field itself
                if (archive_type_var == 'ensemble_full') or (archive_type_var == 'ensemble_subsample'):
                    varout = outfile.createVariable(varname, var_desc[var]['dtype'], ('time','MCrun','lat','members'),
                                                    fill_value=missing_val,
                                                    zlib=True,complevel=4,fletcher32=True)
                else:
                    varout = outfile.createVariable(varname, var_desc[var]['dtype'], ('time','MCrun','lat'),
                                                    fill_value=missing_val,
                                                    zlib=True,complevel=4,fletcher32=True)
                varout.var_desc      = var_desc[var]['var_desc']
                varout.long_name     = var_desc[var]['long_name']
                varout.standard_name = var_desc[var]['standard_name']
                varout.units         = var_desc[var]['units']
                varout.level_desc    = var_desc[var]['level_desc']
                varout.statistic     = statistic[k]
                varout.GRIB_id       = var_desc[var]['GRIB_id']
                varout.GRIB_name     = var_desc[var]['GRIB_name']        
                varout.valid_range   = np.array(var_desc[var]['valid_range'],dtype=var_desc[var]['dtype'])
                varout.dataset       = dataset_tag
                varout.missing_value = missing_val
                varout.actual_range  = np.array((minval, maxval), dtype=var_desc[var]['dtype'])

                # upload the data to file
                lat[:]    = lat1d
                time[:]   = time_output
                varout[:] = mc_ens_outarr[k,:]

            elif field_type == '0D:time_series':

                if (archive_type_var == 'ensemble_full') or (archive_type_var == 'ensemble_subsample'):
                    varout = outfile.createVariable(varname, var_desc[var]['dtype'], ('time','MCrun','members'),
                                                    fill_value=missing_val,
                                                    zlib=True,complevel=4,fletcher32=True)
                    statistic = 'Ensemble Members'
                else:
                    varout = outfile.createVariable(varname, var_desc[var]['dtype'], ('time','MCrun'),
                                                    fill_value=missing_val,
                                                    zlib=True,complevel=4,fletcher32=True)

                varout.var_desc      = var_desc[var]['var_desc']
                varout.long_name     = var_desc[var]['long_name']
                varout.standard_name = var_desc[var]['standard_name']
                varout.units         = var_desc[var]['units']
                varout.level_desc    = var_desc[var]['level_desc']
                varout.statistic     = statistic[k]
                varout.GRIB_id       = var_desc[var]['GRIB_id']
                varout.GRIB_name     = var_desc[var]['GRIB_name']        
                varout.valid_range   = np.array(var_desc[var]['valid_range'],dtype=var_desc[var]['dtype'])
                varout.dataset       = dataset_tag
                varout.missing_value = missing_val
                varout.actual_range  = np.array((minval, maxval), dtype=var_desc[var]['dtype'])

                # upload the data to file
                time[:]   = time_output
                varout[:] = mc_ens_outarr[k,:]

            else:
                print('/n***WARNING: Variable of unrecognized dimensions... Skipping.')


            # Finished processing: closing the file
            outfile.close()

            if run_ncpdq:
                # See if we can compress the netcdf file some more: 
                # Use ncpdq to create a more compressed version of the newly-created netcdf file using
                # short (16-bit) integers: a lossy compression which saves about 5 significant digits.
                # If not, the netcdf files still contains some compression (zlib=True,complevel=4) and
                # error detection (fletcher32=True).
                try:
                    result = subprocess.run(['which', 'ncpdq'], stdout=subprocess.PIPE)
                    nccommand =  result.stdout.decode('utf-8').rstrip('\n')
                    if len(result.stdout.decode('utf-8')) > 0:
                        cmd = nccommand+' -6 -h -O ' + outfile_nc + ' ' + outfile_nc
                        print('Running command: %s' %(cmd))
                        os.system(cmd)
                    else:
                        print('NCO ncpdq application not found on your system.')
                        raise RuntimeError()
                except:
                    print('Additional file compression could not be carried out. Leave file as is.')


        # cleaning up before moving to next variable
        del mc_ens, mc_ens_outarr


    # -----------------------------------------------------------------------
    elapsed_time = clock.time() - begin_time
    print('\nLMR output conversion completed in %s mins' %str(elapsed_time/60.0))


# =========================================================================================

def atoi(text):
    try:
        return int(text)
    except ValueError:
        return text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''    
    return [ atoi(c) for c in re.split('([-]?\d+)', text) ]

def natural_sort(input):
    return sorted(input, key=natural_keys)


# =========================================================================================
# =========================================================================================
if __name__ == "__main__":
    main()
