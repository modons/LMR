"""
Module: LMR_convertNPZtoNETCDF_fullfield.py

Purpose: Converts LMR output from .npz files to netcdf files for every
         Monte-Carlo realization included in the experiment.  Monte-Carlo
         realizations are saved as seperate files to avoid memory errors.

Originator: Robert Tardif | Univ. of Washington, Dept. of Atmospheric Sciences
            Michael Erb   | University of Southern California
                          | June 2017

Revisions: None

"""
import glob
import numpy as np
from netCDF4 import Dataset
import os

# --- Begin section of user-defined parameters ---

# name of directory where the output of LMR experiments are located
#datadir = '/home/scec-00/lmr/erbm/LMR/archive_output'
#datadir = '/home/disk/ekman4/rtardif/LMR/output'
datadir ='/Users/hakim/data/LMR/archive'

# name of the experiment
#nexp = 'test_pages2kv2_fullfield'
nexp = 'dadt_test_fullensemble'

# Dictionary containing definitions of variables that can be handled by this code
var_desc = \
    {
        'd18O_sfc_Amon'             : ('d18O', 'delta18 oxygen isotope','permil SMOW'),
        'tas_sfc_Amon'              : ('Tsfc', 'Near surface air temperature anomaly', 'K'),
        'psl_sfc_Amon'              : ('MSLP', 'Mean sea level pressure anomaly', 'Pa'),
        'pr_sfc_Amon'               : ('PRCP', 'Precipitation rate anomaly', 'kg/m2/s1'),
        'scpdsi_sfc_Amon'           : ('scpdsi','self-calibrated Palmer Drought Severity Index', ''),
        'uas_sfc_Amon'              : ('Usfc', 'Near surface zonal wind anomaly', 'm/s'),
        'vas_sfc_Amon'              : ('Vsfc', 'Near surface meridional wind anomaly', 'm/s'),
        'zg_500hPa_Amon'            : ('H500', '500hPa geopotential height anomaly', 'm'),
        'wap_500hPa_Amon'           : ('W500', '500hPa vertical motion anomaly', 'Ps/s'),
        'wap_700hPa_Amon'           : ('W700', '700hPa vertical motion anomaly', 'Ps/s'),
        'ua_500hPa_Amon'            : ('U500', '500hPa zonal wind anomaly', 'm/s'),
        'va_500hPa_Amon'            : ('V500', '500hPa meridional wind anomaly', 'm/s'),
        'tos_sfc_Omon'              : ('tos',  'Sea surface temperature', 'K'),
        'ohcArctic_0-700m_Omon'     : ('ohcArctic_0to700m','Basin-averaged Ocean Heat Content of Arctic Ocean in 0-700m layer','J'),
        'ohcAtlanticNH_0-700m_Omon' : ('ohcAtlanticNH_0to700m','Basin-averaged Ocean Heat Content of N. Atlantic Ocean in 0-700m layer','J'),
        'ohcAtlanticSH_0-700m_Omon' : ('ohcAtlanticNH_0to700m','Basin-averaged Ocean Heat Content of S. Atlantic Ocean in 0-700m layer','J'),
        'ohcPacificNH_0-700m_Omon'  : ('ohcPacificNH_0to700m','Basin-averaged Ocean Heat Content of N. Pacific Ocean in 0-700m layer','J'),
        'ohcPacificSH_0-700m_Omon'  : ('ohcPacificSH_0to700m','Basin-averaged Ocean Heat Content of S. Pacific Ocean in 0-700m layer','J'),
        'ohcSouthern_0-700m_Omon'   : ('ohcSouthern_0to700m','Basin-averaged Ocean Heat Content of Southern Ocean in 0-700m layer','J'),
        'ohcIndian_0-700m_Omon'     : ('ohcIndian_0to700m','Basin-averaged Ocean Heat Content of Indian Ocean in 0-700m layer','J'),
        'AMOCindex_Omon'            : ('AMOCindex','AMOC index (max. overturning in region between 30N-70N and 500m-2km depth in N Atl.)','kg s-1'),
        'AMOC45N1000m_Omon'         : ('AMOC45N1000m','Meridional overturning streamfunction at 45N and 1000m depth','kg s-1'),
        'AMOC26N1000m_Omon'         : ('AMOC26N1000m','Meridional overturning streamfunction at 26N and 1000m depth','kg s-1'),
        'AMOC26Nmax_Omon'           : ('AMOC26Nmax','Maximum meridional overturning streamfunction in ocean column at 26N','kg s-1'),
        'tas_sfc_Adec'              : ('Tsfc', 'Near surface air temperature anomaly', 'K')
        }

# --- End section of user-defined parameters ---

expdir = datadir + '/'+nexp

# where the netcdf files are created 
outdir = expdir

print(('\n Getting information on Monte-Carlo realizations for ' + expdir + '...\n'))

dirs = glob.glob(expdir+"/r*")
# sorted
#dirs.sort()
# keep names of MC directories (i.r. "r...") only 
mcdirs = [item.split('/')[-1] for item in dirs]
# number of MC realizations found
niters = len(mcdirs) 

print(('mcdirs:' + str(mcdirs)))
print(('niters = ' + str(niters)))

print('\n Getting information on reconstructed variables...\n')

# look in first "mcdirs" only. It should be the same for all. 
workdir = expdir+'/'+mcdirs[0]
# look for "ensemble_full" files
listdirfiles = glob.glob(workdir+"/ensemble_full_*")

# strip directory name, keep file name only
listfiles = [item.split('/')[-1] for item in listdirfiles]

# strip everything but variable name
listvars = [(item.replace('ensemble_full_','')).replace('.npz','') for item in listfiles]

print(('Variables:', listvars, '\n'))

# Loop over variables
for var in listvars:
    print(('\n Variable:', var))
    # Loop over realizations
    for dir in mcdirs:
        fname = expdir+'/'+dir+'/ensemble_full_'+var+'.npz'
        npzfile = np.load(fname)

        # Get the reconstructed field
        field_values = npzfile['xa_ens']
        
        npzcontent = npzfile.files
        print(('  file contents:', npzcontent))
            
        # get the years in the reconstruction ... for some reason stored in an array of strings ...
        years_str =  npzfile['years']
        # convert to array of floats
        years = np.asarray([float(item) for item in years_str])

        # Determine type of variation, get spatial coordinates if present
        if 'lat' in npzcontent and 'lon' in npzcontent:
            field_type = '2D:horizontal'
            print('  field type:', field_type)
            # get lat/lon data
            lat2d = npzfile['lat']
            lon2d = npzfile['lon']
            #print '  ', lat2d.shape, lon2d.shape
            lat1d = lat2d[:,0]
            lon1d = lon2d[0,:]
            print('  nlat/nlon=', lat1d.shape, lon1d.shape)

        elif 'lat' in npzcontent and 'lev' in npzcontent:
            field_type = '2D:meridional_vertical'
            print('  field type:', field_type)
            # get lat/lev data
            lat2d = npzfile['lat']
            lev2d = npzfile['lev']
            #print '  ', lat2d.shape, lev2d.shape
            lat1d = lat2d[:,0]
            lev1d = lev2d[0,:]
            print('  nlat/nlev=', lat1d.shape, lev1d.shape)
             
        elif 'lat' not in npzcontent and 'lon' not in npzcontent and 'lev' not in npzcontent:
            # no spatial coordinate, must be a scalar (time series)
            field_type='1D:time_series'
            print('  field type:', field_type)
        else:
            print('Cannot handle this variable yet! Variable of unrecognized dimensions... Exiting!')
            exit(1)

        
        # declare master array that will contain data from all the M-C realizations 
        # (i.e. the "Monte-Carlo ensemble")
        dims = field_values.shape
        print('  xam field dimensions', dims)
        mc_ens = field_values


        # Roll array to get dims as [nens, time, nlat, nlon]
        mc_ens_outarr = np.rollaxis(mc_ens,-1,0)
        
        # Create the netcdf file for the current variable
        outfile_nc = outdir+'/'+var+'_MCiters_ensemble_full_'+dir+'.nc'
        outfile = Dataset(outfile_nc, 'w', format='NETCDF4')
        outfile.description = 'LMR climate field reconstruction for variable: %s' % var
        outfile.experiment = nexp
        outfile.comment = 'File contains values for each ensemble member for Monte-Carlo realization '+dir
        
        # define dimensions
        ntime = years.shape[0]
        nensembles = mc_ens_outarr.shape[0]

        outfile.createDimension('time', ntime)
        outfile.createDimension('ensemble_member', nensembles)

        if field_type == '2D:horizontal':
            nlat  = lat1d.shape[0]
            nlon  = lon1d.shape[0]
            outfile.createDimension('lat', nlat)
            outfile.createDimension('lon', nlon)
        elif  field_type == '2D:meridional_vertical':
            nlat  = lat1d.shape[0]
            nlev  = lev1d.shape[0]
            outfile.createDimension('lat', nlat)
            outfile.createDimension('lev', nlev)
        else:
            pass
            
        # define variables & upload the data to file

        # reconstructed field
        varinfolst = var.split('_')
        varname  = varinfolst[0]
        varlevel = None
        if len(varinfolst) > 2:
            if varinfolst[2] == 'Amon' or  varinfolst[2] == 'Omon' or varinfolst[2] == 'OImon':
                varlevel = varinfolst[1]
            
        # time
        time = outfile.createVariable('time', 'i', ('time',))
        time.description = 'time'
        time.long_name = 'year CE'

        if field_type == '2D:horizontal':
            # lat
            lat = outfile.createVariable('lat', 'f', ('lat',))
            lat.description = 'latitude'
            lat.units = 'degrees_north'
            lat.long_name = 'latitude'

            # lon
            lon = outfile.createVariable('lon', 'f', ('lon',))
            lon.description = 'longitude'
            lon.units = 'degrees_east'
            lon.long_name = 'longitude'

            varout = outfile.createVariable(varname, 'f', ('ensemble_member','time', 'lat','lon'),zlib=True,complevel=4,fletcher32=True)
            varout.description = var_desc[var][0]
            varout.long_name = var_desc[var][1]
            varout.units = var_desc[var][2]
            if varlevel: varout.level = varlevel
            
            # upload the data to file
            lat[:]    = lat1d
            lon[:]    = lon1d
            time[:]   = years
            varout[:] = mc_ens_outarr

        elif field_type == '2D:meridional_vertical':
            # lat
            lat = outfile.createVariable('lat', 'f', ('lat',))
            lat.description = 'latitude'
            lat.units = 'degrees_north'
            lat.long_name = 'latitude'

            # lev
            lev = outfile.createVariable('lev', 'f', ('lev',))
            lev.description = 'depth'
            lev.units = 'm'
            lev.long_name = 'depth'

            varout = outfile.createVariable(varname, 'f', ('ensemble_member','time', 'lat','lev'),zlib=True,complevel=4,fletcher32=True)        
            varout.description = var_desc[var][0]
            varout.long_name = var_desc[var][1]
            varout.units = var_desc[var][2]
    
            # upload the data to file
            lat[:]    = lat1d
            lev[:]    = lev1d
            time[:]   = years
            varout[:] = mc_ens_outarr

        elif field_type == '1D:time_series':
            varout = outfile.createVariable(varname, 'f', ('ensemble_member','time'),zlib=True,complevel=4,fletcher32=True)        
            varout.description = var_desc[var][0]
            varout.long_name = var_desc[var][1]
            varout.units = var_desc[var][2]
            if varlevel: varout.level = varlevel
            
            # upload the data to file
            time[:]   = years
            varout[:] = mc_ens_outarr        
            
        else:
            print('Variable of unrecognized dimensions... Exiting!')
            exit(1)
            
        
        # Closing the file
        outfile.close()

        # The initial netcdf contains some compression (zlib=True,complevel=4) and error detection (fletcher32=True).
        # Use ncpdq to create a smaller version of the newly-created netcdf file using short (16-bit) integers.
        # This is a lossy compression which saves about 5 significant digits, but takes up less storage space.
        cmd = 'ncpdq -O -P all_new ' + outfile_nc + ' ' + outfile_nc
        os.system(cmd)
