"""
 Module: plot_proxy_timeseries.py
 
   Stand-alone tool to plot the proxy timeseries from *all* records included in
   pandas Dataframes stored in .pckl files used as input by the LMR paleo-data 
   assimilation application. 
 
 Originator : Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
                            | July 2016

 Revisions: 
 - Added the some crude detection of time intervals with missing data in the 
   proxy records so that distinct data segments are not linked together on
   the plots.
   [R. Tardif, U. of Washington, March 2017]

"""

import os
import cPickle
import numpy as np
import pandas

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.basemap import Basemap


# -- -- --

datadir = '/home/disk/kalman3/rtardif/LMR/data/proxies'

proxy_database = 'NCDC_v0.1.0'
#proxy_database = 'NCDC_vDADTtest'
#proxy_database = 'NCDC_vJB.icecores'

figdir = '/home/disk/kalman3/rtardif/LMR/data/proxies/NCDC/Figs'
#figdir = '/home/disk/kalman3/rtardif/LMR/data/proxies/DADT/Figs'

# -- -- --

proxy_pandas_metafile = datadir+'/'+proxy_database+'_Metadata.df.pckl'
proxy_pandas_datafile = datadir+'/'+proxy_database+'_Proxies.df.pckl'

proxy_meta = pandas.read_pickle(proxy_pandas_metafile)
proxy_data = pandas.read_pickle(proxy_pandas_datafile)

# List of proxy sites
site_list = list(proxy_data.columns.values)

# Loop over sites
for site in site_list:

    # -- metadata --
    site_meta = proxy_meta[proxy_meta['NCDC ID'] == site]
    pid = site_meta['NCDC ID'].iloc[0]
    pmeasure = site_meta['Proxy measurement'].iloc[0]
    p_type = site_meta['Archive type'].iloc[0]

    pname = p_type.replace(" ", "_")+'_'+pmeasure.replace("/", "")+'_'+site.split(':')[0]

    print pname, p_type, pmeasure
    
    # -- data --
    site_data = proxy_data[site]
    data_valid = site_data[site_data.notnull()]
    times = data_valid.index.values
    values = data_valid.values


    # -- Detecting intervals of missing data --
    intervals = np.diff(times)
    intervals_mean = np.mean(intervals)
    intervals_std = np.std(intervals)

    # threshold on nb. of standard-deviations
    nbstd = 4. 
    inds_miss = np.where(intervals > (intervals_mean+nbstd*intervals_std))[0]
    nbmiss, = inds_miss.shape
    if nbmiss > 0:
        times_plot = np.insert(times, inds_miss[:]+1, (times[inds_miss[:]]+1.))
        values_plot = np.insert(values, inds_miss[:]+1, np.nan)
    else:
        times_plot = times
        values_plot = values
    
    # -- plot --
    plt.figure(figsize=(8,5))
    plt.plot(times_plot, values_plot, '-r',linewidth=2,alpha=.3)
    plt.plot(times_plot, values_plot, '.',color='r')
    plt.title(pname)
    plt.xlabel('Year CE')
    plt.ylabel('Proxy data')

    xmin = np.min(times_plot)
    xmax = np.max(times_plot)
    ymin = np.nanmin(values_plot)
    ymax = np.nanmax(values_plot)
    plt.axis((xmin,xmax,ymin,ymax))

    plt.savefig(figdir+'/'+'proxy_ts_%s.png' % (pname),bbox_inches='tight')

    plt.close()



    
