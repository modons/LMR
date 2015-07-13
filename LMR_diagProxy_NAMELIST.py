
from datetime import datetime, timedelta

# =============================================================================
# Section 1: High-level parameters of reconstruction experiment
# =============================================================================

# Name of reconstruction experiment to verify
nexp = 'Recon_ens100_allAnnualProxyTypes_pf0.5'

# Run diagnostics over this range of Monte-Carlo reconstructions
iter_range = [61,100]

# set the absolute path the experiment (could make this cwd with some os coding)
LMRpath = '/home/disk/kalman3/rtardif/LMR'
#LMRpath = '/home/disk/ekman/rtardif/nobackup/LMR'

# Input directory, where to find the reconstruction data
datadir_input  = '/home/disk/kalman3/rtardif/LMR/output'

# Reconstruction period (years)
#recon_period = [1500,2000]
#recon_period = [1000,2000]
recon_period = [1850,2000]

# =============================================================================
# Section 2: PROXIES
# =============================================================================

# Proxy data directory & file
datadir_proxy    = LMRpath+'/data/proxies';
datafile_proxy   = 'Pages2k_DatabaseS1-All-proxy-records.xlsx';
dataformat_proxy = 'CSV';

#regions = ['Arctic','Europe','Australasia']
#regions = ['Europe']
regions = ['Antarctica','Arctic','Asia','Australasia','Europe','North America','South America']

# Define proxies to be assimilated
# Use dictionary with structure: {<<proxy type>>:[... list of all measurement tags ...]}
# where "proxy type" is written as "<<archive type>>_<<measurement type>>"
# this one has it all?
proxy_verif = {'Tree ring_Width': ['Ring width','Tree ring width','Total ring width','TRW'],'Tree ring_Density': ['Maximum density','Minimum density','Earlywood density','Latewood density','MXD'],'Ice core_d18O': ['d18O'],'Ice core_d2H': ['d2H'],'Ice core_Accumulation':['Accumulation'],'Coral_d18O': ['d18O'],'Coral_Luminescence':['Luminescence'],'Lake sediment_All':['Varve thickness','Thickness','Mass accumulation rate','Particle-size distribution','Organic matter','X-ray density'],'Marine sediment_All':['Mg/Ca'],'Speleothem_All':['Lamina thickness']}
#proxy_verif = {'Tree ring_Width': ['Ring width','Tree ring width','Total ring width','TRW']}
#proxy_verif = {'Ice core_d18O': ['d18O']}
#proxy_verif = {'Coral_d18O': ['d18O']}

# Proxy temporal resolution (in yrs)
#proxy_resolution = [1.0,5.0]
proxy_resolution = [1.0]

# Source of calibration data (for PSM)
datatag_calib = 'GISTEMP'
#datatag_calib = 'HadCRUT'
#datatag_calib = 'BerkeleyEarth'
datadir_calib = LMRpath+'/data/analyses';

# Threshold correlation of linear PSM 
PSM_r_crit = 0.2

# =============================================================================
# Section 3: OUTPUT
# =============================================================================

# Output
datadir_output  = '/home/disk/kalman3/rtardif/LMR/output'
#datadir_output  = '/home/disk/ekman/rtardif/nobackup/LMR/output'
#datadir_output  = '/home/disk/ice4/hakim/svnwork/python-lib/trunk/src/ipython_notebooks/data'
