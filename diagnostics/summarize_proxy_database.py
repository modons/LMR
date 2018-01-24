"""
Module: summarize_proxy_database.py

 This script reads in the preprocessed proxy datasets stored in LMR-ready
 pandas DataFrames in pickle files and a specified PSM file and
 does two things:
  1) Produces a list of all proxy types and units.
  2) Produces individual figures of all preprocessed proxy records, 
     along with metadata.

 Notes: 
 - You'll need to make the preprocessed files (using LMR_proxy_preprocess.py)
   and a PSM file (generated with LMR_PSMbuild.py) before using this.  
 - Also, change the "data_directory" and "output_directory" to point to the
   appropriate places on your machine.
 - Only works on the LMRv2 release (merged PAGES2k phase2 + LMR-specific 
   NCDC-templated proxy records.

 author: Michael P. Erb
 date  : 4/17/2017

 Revisions:  
 - Adapted the original code to handle all records (PAGES2kv2 and NCDC) 
   considered in the LMR project.
   [R. Tardif, Univ. of Washington, May 2017]
 - Added the generation of geographical maps showing the location 
   of proxy records per proxy types considered by the data assimilation.
   [R. Tardif, Univ. of Washington, May 2017]

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists
from mpl_toolkits.basemap import Basemap


# ------------------------------------------------------------ #
# -------------- Begin: user-defined parameters -------------- #

# directory where data from reconstruction experiments are located 
#data_directory = "/home/scec-00/lmr/erbm/LMR/"
#data_directory = "/home/disk/kalman3/rtardif/LMR/"
data_directory = "/home/disk/kalman3/rtardif/LMRpy3/"

# which proxy database? PAGES2kv2 or LMRdb
#proxy_db = 'PAGES2kv1'
proxy_db = 'LMRdb'

# Version of the database to query (for proxy_db = LMRdb only)
dbversion = 'v0.2.0'

# Filter on proxy temporal resolution (range is inclusive)
temporal_resolution_range = (1,1); resolution_tag = 'annual'

# Directory where the output (figures) will be created.
# The directory needs to exist prior to running this module.
# ----------------------------------------------------------
#output_directory = "/home/scec-00/lmr/erbm/analysis/results/LMR/pages2kv2/figures/"
#output_directory = "/home/disk/kalman3/rtardif/LMR/data/proxies/PAGES2kv1/Figs/"
#output_directory = "/home/disk/kalman3/rtardif/LMR/data/proxies/LMRdb/Figs/summary_v0.2.0/"
output_directory = "/home/disk/kalman3/rtardif/LMRpy3/data/proxies/LMRdb/Figs/summary_v0.2.0/"

# Swith to indicate whether you want the figure to the produced on-screen (False)
# or save in .png files (True)
save_instead_of_plot = True


# --------------  End: user-defined parameters  -------------- #
# ------------------------------------------------------------ #

if not exists(output_directory):
    print('Output directory does not exist. Creating it now.')
    os.makedirs(output_directory)

### ----------------------------------------------------------------------------
### Proxy type definitions
### ----------------------------------------------------------------------------

if proxy_db == 'PAGES2kv1':
    proxy_def = {
                'Tree ring_Width': ['Ring width',
                                    'Tree ring width',
                                    'Total ring width',
                                    'TRW'],
                'Tree ring_Density': ['Maximum density',
                                      'Minimum density',
                                      'Earlywood density',
                                      'Latewood density',
                                      'MXD'],
                'Ice core_d18O': ['d18O'],
                'Ice core_d2H': ['d2H'],
                'Ice core_Accumulation': ['Accumulation'],
                'Coral_d18O': ['d18O'],
                'Coral_Luminescence': ['Luminescence'],
                'Lake sediment_All': ['Varve thickness',
                                      'Thickness',
                                      'Mass accumulation rate',
                                      'Particle-size distribution',
                                      'Organic matter',
                                      'X-ray density'],
                'Marine sediment_All': ['Mg/Ca'],
                'Speleothem_All': ['Lamina thickness'],
                }

elif proxy_db == 'LMRdb':
    proxy_def = {
                'Tree Rings_WidthPages2'               : ['trsgi'],
                'Tree Rings_WidthBreit'                : ['trsgi_breit'],
                'Tree Rings_WoodDensity'               : ['max_d','min_d','early_d','earl_d','late_d','MXD','density'],
                'Tree Rings_Isotopes'                  : ['d18O'],
                'Corals and Sclerosponges_d18O'        : ['d18O','delta18O','d18o','d18O_stk','d18O_int','d18O_norm','d18o_avg','d18o_ave','dO18','d18O_4'],
                'Corals and Sclerosponges_SrCa'        : ['Sr/Ca','Sr_Ca','Sr/Ca_norm','Sr/Ca_anom','Sr/Ca_int'],
                'Corals and Sclerosponges_Rates'       : ['ext','calc','calcification','calcification rate', 'composite'],
                'Ice Cores_d18O'                       : ['d18O','delta18O','delta18o','d18o','d18o_int','d18O_int','d18O_norm','d18o_norm','dO18','d18O_anom'],
                'Ice Cores_dD'                         : ['deltaD','delD','dD'],
                'Ice Cores_Accumulation'               : ['accum','accumu'],
                'Ice Cores_MeltFeature'                : ['MFP','melt'],
                'Lake Cores_Varve'                     : ['varve', 'varve_thickness', 'varve thickness', 'thickness'],
                'Lake Cores_BioMarkers'                : ['Uk37', 'TEX86', 'tex86'],
                'Lake Cores_GeoChem'                   : ['Sr/Ca', 'Mg/Ca','Cl_cont'],
                'Lake Cores_Misc'                      : ['RABD660_670','X_radiograph_dark_layer','massacum'],
                'Marine Cores_d18O'                    : ['d18O'],
                'Marine Cores_tex86'                   : ['tex86'],
                'Marine Cores_uk37'                    : ['uk37','UK37'],
                'Speleothems_d18O'                     : ['d18O'],
                'Bivalve_d18O'                         : ['d18O'],
                }


### ----------------------------------------------------------------------------
### LOAD DATA
### ----------------------------------------------------------------------------

# Load the proxy data and metadata as dataframes.

if proxy_db == 'PAGES2kv1':
    metadata = np.load(data_directory+'data/proxies/Pages2kv1_Metadata.df.pckl')
    proxies = np.load(data_directory+'data/proxies/Pages2kv1_Proxies.df.pckl')
    # Load an LMR PSM file.
    try:
        psms = np.load(data_directory+'PSM/PSMs_PAGES2kv1_GISTEMP.pckl')
    except:
        psms = []

elif proxy_db == 'LMRdb':    
    metadata = np.load(data_directory+'data/proxies/LMRdb_'+dbversion+'_Metadata.df.pckl')
    proxies = np.load(data_directory+'data/proxies/LMRdb_'+dbversion+'_Proxies.df.pckl')    
    # Load an LMR PSM file.
    try:
        psms = np.load(data_directory+'PSM/PSMs_LMRdb_'+dbversion+'_annual_GISTEMP.pckl')
    except:
        psms = []

    # include dbversion in the name
    proxy_db = '_'.join([proxy_db,dbversion])

### CALCULATIONS ---

# Count all of the different proxy types and print a list.
archive_counts = {}
archive_types = np.unique(metadata['Archive type'])
for ptype in archive_types:
    archive_counts[ptype] = np.unique(metadata['Proxy measurement'][metadata['Archive type'] == ptype],return_counts=True)

print("=================")
print(" Archive counts:")
print("=================")
for ptype in archive_types:
    for units in range(0,len(archive_counts[ptype][0])):
        print('%25s - %23s : %3d' % (ptype, archive_counts[ptype][0][units], archive_counts[ptype][1][units]))



### ----------------------------------------------------------------------------
### Maps of proxy locations, per type. 
### ----------------------------------------------------------------------------

sumnbproxies = 0
print(' ')
print('Proxy types for data assimilation -------------------')
proxy_types =  list(proxy_def.keys())
for ptype in sorted(proxy_types):
    latslons = []    
    latslons.append([(metadata['Lat (N)'][i],metadata['Lon (E)'][i]) for i in range(0,len(metadata['Proxy ID'])) \
                     if metadata['Archive type'][i] == ptype.split('_')[0] and metadata['Proxy measurement'][i] in proxy_def[ptype] \
                     and metadata['Resolution (yr)'][i] >= temporal_resolution_range[0] \
                     and metadata['Resolution (yr)'][i] <= temporal_resolution_range[1]])

    nbproxies = len(latslons[0])

    sumnbproxies = sumnbproxies + nbproxies

    if nbproxies > 0:    
        plotlist = latslons[0]

        nbunique = len(list(set(plotlist)))
        nbsamelocations = nbproxies - nbunique        
        print('%35s : %4d (same lats/lons: %3d)' %(ptype,nbproxies,nbsamelocations))

        lats = np.asarray([item[0] for item in plotlist])
        lons = np.asarray([item[1] for item in plotlist])

        fig = plt.figure(figsize=(11,9))
        #ax  = fig.add_axes([0.1,0.1,0.8,0.8])
        m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=40.  # GLOBAL

        #water = '#9DD4F0'
        #continents = '#888888'
        water = '#D3ECF8'
        continents = '#F2F2F2'
        
        m.drawmapboundary(fill_color=water)
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries()
        m.fillcontinents(color=continents,lake_color=water)
        m.drawparallels(np.arange(-80.,81.,latres))
        m.drawmeridians(np.arange(-180.,181.,lonres))

        l = []
        for k in range(len(plotlist)):
            color_dots = 'red'
            x, y = m(lons,lats)
            l.append(m.scatter(x,y,35,marker='o',color=color_dots,edgecolor='#ffe7e5',linewidth='1',zorder=4))
        plt.title('%s:%d (%s)' % (ptype,nbproxies,resolution_tag), fontweight='bold',fontsize=14)

        plt.savefig('%smap_proxies_%s_%s.png' %(output_directory,proxy_db,ptype.replace(' ','_')),bbox_inches='tight')
        #plt.show()

print('%35s : %4d' %('Total',sumnbproxies))


### ----------------------------------------------------------------------------
### FIGURES of individual records (time series)
### ----------------------------------------------------------------------------

# Save a list of all records with PSMs.
records_with_psms = []
for i in range(0,len(psms)):
    records_with_psms.append(list(psms.keys())[i][1])


plt.style.use('ggplot')

#for i in range(0,3):  # To make sample figures, use this line instead of the next line.
for i in range(0,len(metadata['Proxy ID'])):
    print("Proxy: ",i+1,"/",len(metadata['Proxy ID']), metadata['Proxy ID'][i])
    if metadata['Proxy ID'][i] in records_with_psms: has_psm = "YES"
    else: has_psm = "NO"
    
    # Make a plot of each proxy.
    plt.figure(figsize=(10,8))
    ax = plt.axes([.1,.6,.8,.3])
    plt.plot(proxies[metadata['Proxy ID'][i]],'-b',linewidth=2,alpha=.3)
    plt.plot(proxies[metadata['Proxy ID'][i]],'.',color='b')

    plt.suptitle("LMR proxy time series",fontweight='bold',fontsize=12)
    plt.title(metadata['Proxy ID'][i],fontsize=11)
    plt.xlabel("Year CE")
    plt.ylabel(metadata['Proxy measurement'][i])

    fntsize = 9
    offsetscale = 0.07
    
    # Print metadata on each figure.
    for offset, key in enumerate(metadata):
        if key != 'Proxy ID':
            plt.text(0,-.3-offsetscale*offset,key+":",transform=ax.transAxes,fontsize=fntsize)
            if key == 'Study name' or key == 'Investigators':
                if len(metadata[key][i]) > 100:
                    metadata_entry = metadata[key][i][0:100]+' ...'
                else:
                    metadata_entry = metadata[key][i]
            else:
                metadata_entry = metadata[key][i]

            if isinstance(metadata_entry,str):
                try:
                    metadata_entry = metadata_entry.encode(encoding='utf-8', errors='ignore')
                    trc = 1
                except (UnicodeEncodeError, UnicodeDecodeError):
                    try:
                        metadata_entry = metadata_entry.decode('utf-8','ignore')                        
                        trc = 2
                    except UnicodeDecodeError:
                        metadata_entry = metadata_entry.decode('iso-8859-1')
                        trc = 3

                if isinstance(metadata_entry, bytes):
                    metadata_entry = metadata_entry.decode()
                metadata_entry.encode('ascii', 'ignore')
                
            plt.text(.23,-.3-offsetscale*offset,metadata_entry,transform=ax.transAxes,fontsize=fntsize)
            
    plt.text(0,-.4-offsetscale*offset,"Proxy is in given PSM file:",transform=ax.transAxes,fontsize=fntsize)
    plt.text(.23,-.4-offsetscale*offset,has_psm,transform=ax.transAxes,fontsize=fntsize)
    
    if save_instead_of_plot:
        plt.savefig(output_directory+'ts_'+metadata['Archive type'][i].replace(" ","_")+"_"+metadata['Proxy ID'][i].replace("/","_")+".png")
    else:
        plt.show()
    plt.close()
    

