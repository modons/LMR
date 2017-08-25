"""
Module: prepare_prior_cgenie.py

Purpose: Extract data from a set of variables from a cGENIE climate model 
         simulation to generate files formatted for input into the 
         LMR data assimilation system. 

Originator: Robert Tardif - University of Washington : August 2017


"""
import os

# ==============================================================================

input_data_directory  = '/home/disk/ekman/rtardif/kalman3/LMR/data/model/cgenie_petm/orig_files/'
output_data_directory = '/home/disk/ekman/rtardif/kalman3/LMR/data/model/cgenie_petm/'


# GENIE output files, input to this script
file_2d_fields = input_data_directory+'fields_biogem_2d.nc'
file_3d_fields = input_data_directory+'fields_biogem_3d.nc'

#time_interval = 'ann' # for files with data every year
time_interval = 'dec' # for files with data every decade
#time_interval = 'cen' # for files with data every century

# ==============================================================================

FillVal = 9.96920996839e+36

# ---------------------------------------------------------------------
# 1) extract near-surface air temperature from GENIE 2D file (atm_temp)
# ---------------------------------------------------------------------
lmr_file = output_data_directory+'tas_sfc_A%s_cgenie_petm.nc' %(time_interval)
genie_variable = 'atm_temp'

# extract the variable
command = 'ncks -v %s %s %s' %(genie_variable, file_2d_fields,lmr_file)
status = os.system(command)

if status == 0:
    # rename to LMR variable
    command = 'ncrename -O -v atm_temp,tas %s' %(lmr_file)
    status = os.system(command)

    # convert deg C to Kelvins
    # mv data file to temporary file
    command = 'mv -f %s tmp.nc' %(lmr_file)
    status = os.system(command)
    # rename "missing_value" as _FillValue (recognized by NCO)
    command = 'ncrename -a .missing_value,_FillValue tmp.nc tmp2.nc'
    status = os.system(command)
    # perform conversion and put results in lmr_file
    command = 'ncap -O -s "tas=(tas+273.15)" tmp2.nc %s' %(lmr_file)
    status = os.system(command)
    # delete temporary files
    command = 'rm -f tmp.nc tmp2.nc'
    status = os.system(command)

    # re-add variable attributes
    command = 'ncatted -O -a long_name,tas,c,c,"surface air temperature" %s' %(lmr_file)
    status = os.system(command)

    command = 'ncatted -O -a units,tas,c,c,"K" %s' %(lmr_file)
    status = os.system(command)

    command = 'ncatted -a _FillValue,,m,f,%f %s' %(FillVal,lmr_file)
    status = os.system(command)
    command = 'ncatted -a missing_value,,c,f,%f %s' %(FillVal,lmr_file)
    status = os.system(command)
    
    # add necessary attributes to time variable
    command = 'ncatted -O -a calendar,time,c,c,"noleap" %s' %(lmr_file)
    status = os.system(command)

    command = 'ncatted -O -a units,time,o,c,"year mid-point" %s' %(lmr_file)
    status = os.system(command)

# ------------------------------------------------
# 2) extract SST from GENIE 2D file (ocn_sur_temp)
# ------------------------------------------------
lmr_file = output_data_directory+'tos_sfc_O%s_cgenie_petm.nc' %(time_interval)
genie_variable = 'ocn_sur_temp'

# extract the variable
command = 'ncks -v %s %s %s' %(genie_variable,file_2d_fields,lmr_file)
status = os.system(command)

if status == 0:
    # rename to LMR variable
    command = 'ncrename -O -v ocn_sur_temp,tos %s' %(lmr_file)
    status = os.system(command)

    # convert deg C to Kelvins
    # mv data file to temporary file
    command = 'mv -f %s tmp.nc' %(lmr_file)
    status = os.system(command)
    # rename "missing_value" as _FillValue (recognized by NCO)
    command = 'ncrename -a .missing_value,_FillValue tmp.nc tmp2.nc'
    status = os.system(command)
    # perform conversion and put results in lmr_file
    command = 'ncap -O -s "tos=(tos+273.15)" tmp2.nc %s' %(lmr_file)
    status = os.system(command)
    # delete temporary files
    command = 'rm -f tmp.nc tmp2.nc'
    status = os.system(command)

    # re-add variable attributes
    command = 'ncatted -O -a long_name,tos,c,c,"surface-water temp" %s' %(lmr_file)
    status = os.system(command)

    command = 'ncatted -O -a units,tos,c,c,"K" %s' %(lmr_file)
    status = os.system(command)

    command = 'ncatted -a _FillValue,,m,f,%f %s' %(FillVal,lmr_file)
    status = os.system(command)
    command = 'ncatted -a missing_value,,c,f,%f %s' %(FillVal,lmr_file)
    status = os.system(command)
    
    # add necessary attributes to time variable
    command = 'ncatted -O -a calendar,time,c,c,"noleap" %s' %(lmr_file)
    status = os.system(command)

    command = 'ncatted -O -a units,time,o,c,"year mid-point" %s' %(lmr_file)
    status = os.system(command)

# -----------------------------------------------
# 3) extract SSS from GENIE 2D file (ocn_sur_sal)
# ------------------------------------------------
lmr_file = output_data_directory+'sos_sfc_O%s_cgenie_petm.nc' %(time_interval)
genie_variable = 'ocn_sur_sal'

# extract the variable
command = 'ncks -v %s %s %s' %(genie_variable,file_2d_fields,lmr_file)
status = os.system(command)

if status == 0:
    # rename to LMR variable
    command = 'ncrename -O -v ocn_sur_sal,sos %s' %(lmr_file)
    status = os.system(command)

    # rename "missing_value" as _FillValue (recognized by NCO)
    command = 'ncrename -a .missing_value,_FillValue %s' %(lmr_file)
    status = os.system(command)    
    # re-add variable attributes
    command = 'ncatted -a _FillValue,,m,f,%f %s' %(FillVal,lmr_file)
    status = os.system(command)
    command = 'ncatted -a missing_value,,c,f,%f %s' %(FillVal,lmr_file)
    status = os.system(command)
    
    # add necessary attributes to time variable
    command = 'ncatted -O -a calendar,time,c,c,"noleap" %s' %(lmr_file)
    status = os.system(command)

    command = 'ncatted -O -a units,time,o,c,"year mid-point" %s' %(lmr_file)
    status = os.system(command)

# ---------------------------------------------------------
# 4) extract sea-ice cover from GENIE 2D file (phys_seaice)
# ---------------------------------------------------------
lmr_file = output_data_directory+'sic_sfc_OI%s_cgenie_petm.nc' %(time_interval)
genie_variable = 'phys_seaice'

# extract the variable
command = 'ncks -v %s %s %s' %(genie_variable,file_2d_fields,lmr_file)
status = os.system(command)

if status == 0:
    # rename to LMR variable
    command = 'ncrename -O -v phys_seaice,sic %s' %(lmr_file)
    status = os.system(command)

    # rename "missing_value" as _FillValue (recognized by NCO)
    command = 'ncrename -a .missing_value,_FillValue %s' %(lmr_file)
    status = os.system(command)    
    # re-add variable attributes
    command = 'ncatted -a _FillValue,,m,f,%f %s' %(FillVal,lmr_file)
    status = os.system(command)
    command = 'ncatted -a missing_value,,c,f,%f %s' %(FillVal,lmr_file)
    status = os.system(command)
    
    # add necessary attributes to time variable
    command = 'ncatted -O -a calendar,time,c,c,"noleap" %s' %(lmr_file)
    status = os.system(command)

    command = 'ncatted -O -a units,time,o,c,"year mid-point" %s' %(lmr_file)
    status = os.system(command)

    # modify units attribute to sea-ice cover
    command = 'ncatted -O -a units,sic,o,c,"percent" %s' %(lmr_file)
    status = os.system(command)

# ----------------------------------------------------------------
# 5) extract sea-ice thickness from GENIE 2D file (phys_seaice_th)
# ----------------------------------------------------------------
lmr_file = output_data_directory+'sit_sfc_OI%s_cgenie_petm.nc' %(time_interval)
genie_variable = 'phys_seaice_th'

# extract the variable
command = 'ncks -v %s %s %s' %(genie_variable,file_2d_fields,lmr_file)
status = os.system(command)

if status == 0:
    # rename to LMR variable
    command = 'ncrename -O -v phys_seaice_th,sit %s' %(lmr_file)
    status = os.system(command)

    # rename "missing_value" as _FillValue (recognized by NCO)
    command = 'ncrename -a .missing_value,_FillValue %s' %(lmr_file)
    status = os.system(command)    
    # re-add variable attributes
    command = 'ncatted -a _FillValue,,m,f,%f %s' %(FillVal,lmr_file)
    status = os.system(command)
    command = 'ncatted -a missing_value,,c,f,%f %s' %(FillVal,lmr_file)
    status = os.system(command)
    
    # add necessary attributes to time variable
    command = 'ncatted -O -a calendar,time,c,c,"noleap" %s' %(lmr_file)
    status = os.system(command)

    command = 'ncatted -O -a units,time,o,c,"year mid-point" %s' %(lmr_file)
    status = os.system(command)

