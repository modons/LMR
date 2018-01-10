"""

Module: summarize_assimilated_proxies.py

 This code reads the "assimilated_proxies.npy" files found in the  
 directories containing data for each Monte-Carlo iterations for a 
 specified reconstruction experiment and plots maps of the locations 
 of the assimilated proxy sites and plots the temporal distribution 
 of assimilated proxies per proxy types. 

 Figures are created in the VerifFigs subdirectory in the experiment 
 data directory.

 Originator: Robert Tardif - Univ. of Washington
             December 2016
 Revisions: 
          - Adapted to the newest set of proxy types  
            composing the LMRdb v0.2.0 proxy database            
            [R. Tardif, U of Washington - May 2017]

"""

import os
import glob
import re
import pickle
import numpy as np
from os.path import join

from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.style.use('ggplot')


# ------------------------------------------------------------ #
# -------------- Begin: user-defined parameters -------------- #

# directory where data from reconstruction experiments are located
datadir = '/home/disk/kalman3/rtardif/LMR/output'

# name of reconstruction experiment
nexp = 'test'

# range of Monte-Carlo iterations to consider
iter_range = [0,0]

# Reconstruction performed with the following proxy database:
#proxy_db = 'PAGES2kv1' # aka PAGES2k phase 1 (Hakim et al. JGR 2016)
proxy_db = 'LMRdb'     # aka PAGES2k phase 2 + LMR-specific NCDC-templated proxies

# Whether the y-axis (nb of proxy records) on the temporal plots 
# should be in log-scale or not.
logaxis_temporal_plots = False

# --------------  End: user-defined parameters  -------------- #
# ------------------------------------------------------------ #

# Dictionary to associate a symbol and color to every proxy type

if proxy_db == 'PAGES2kv1':
    # ** PAGES2kv1 **    
    proxy_symbols_color = {
        'Tree ring_Width'       : ('^','#66CC00'), 
        'Tree ring_Density'     : ('v','#FFCC00'), 
        'Coral_d18O'            : ('o','#FF8080'),
        'Coral_Luminescence'    : ('o','#FFB980'),
        'Ice core_d18O'         : ('d','#66FFFF'), 
        'Ice core_d2H'          : ('d','#B8E6E6'), 
        'Ice core_Accumulation' : ('d','#5CB8E6'), 
        'Lake sediment_All'     : ('s','#FF00FF'), 
        'Marine sediment_All'   : ('<','#00b7b6'), 
        'Speleothem_All'        : ('p','#996600'), 
    }

elif proxy_db == 'LMRdb':
    # ** LMRdb **
    proxy_symbols_color = {
        'Bivalve_d18O'                     : ('h','#FFFF00'),
        'Corals and Sclerosponges_Rates'   : ('o','#FFE6E6'),
        'Corals and Sclerosponges_SrCa'    : ('o','#E60000'),
        'Corals and Sclerosponges_d18O'    : ('o','#FF8080'),
        'Ice Cores_Accumulation'           : ('d','#5CB8E6'),
        'Ice Cores_MeltFeature'            : ('d','#0000FF'),
        'Ice Cores_d18O'                   : ('d','#66FFFF'),
        'Ice Cores_dD'                     : ('d','#B8E6E6'),
        'Lake Cores_Misc'                  : ('s','#FFB3FF'),
        'Lake Cores_Varve'                 : ('s','#FF00FF'),
        'Marine Cores_d18O'                : ('<','#00b7b6'),
        'Speleothems_d18O'                 : ('p','#D9B3FF'),
        'Tree Rings_WidthBreit'            : ('^','#B3FF66'),
        'Tree Rings_WidthPages2'           : ('^','#66CC00'),
        'Tree Rings_WoodDensity'           : ('v','#FFCC00'),
        'Tree Rings_Isotopes'              : ('*','#CCFFCC'),    
    }
else:
    raise SystemExit('ERROR: unrecognized slection for proxy database used in '
                     'reconstruction (proxy_db)')

    
proxy_types = sorted(proxy_symbols_color.keys())

expdir = join(datadir,nexp)

figdir = expdir+'/VerifFigs'
if not os.path.isdir(figdir):
    os.system('mkdir %s' % figdir)

iters = np.arange(iter_range[0], iter_range[1]+1)

# get the reconstruction years
gmt_data    = np.load(expdir+'/r'+str(iters[0])+'/'+'gmt.npz')
recon_times = gmt_data['recon_times']
nbtimes = recon_times.shape[0]
nbtypes = len(proxy_types)
nbiters = iters.shape[0]
nb_assim_proxies_type  = np.zeros(shape=[nbiters,nbtypes,nbtimes])

iters_counts_dict = []
iters_counts_dict_cumul = []

for iter in iters:
    runname = 'r'+str(iter)
    dirname = expdir+'/'+runname

    print('Reconstruction:', dirname)

    fname = dirname+'/assimilated_proxies.npy'

    # Read in the file of assimilated proxies for experiment
    assim_proxies = np.load(fname)

    sites_assim_name  = []
    sites_assim_type  = []
    sites_assim_lat   = []
    sites_assim_lon   = []
    sites_assim_years = []
    for k in range(len(assim_proxies)):
        key = list(assim_proxies[k].keys())
        sites_assim_name.append(assim_proxies[k][key[0]][0])
        sites_assim_type.append(key[0])
        sites_assim_lat.append(assim_proxies[k][key[0]][1])
        sites_assim_lon.append(assim_proxies[k][key[0]][2])
        sites_assim_years.append(assim_proxies[k][key[0]][3])

    sites_assim_type_set = sorted(list(set(sites_assim_type)))
    nb_assim_proxies = len(sites_assim_name)
    print('Assimilated proxies =>',len(sites_assim_name), 'sites')

    # ===============================================================================
    # 1) Map of assimilated proxies
    # ===============================================================================
 
    fig = plt.figure()
    ax  = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=40.

    #water = '#9DD4F0'
    #continents = '#888888'
    water = '#D3ECF8'
    continents = '#F2F2F2'

    m.drawmapboundary(fill_color=water)
    m.drawcoastlines(linewidth=0.5); m.drawcountries()
    m.fillcontinents(color=continents,lake_color=water)
    m.drawparallels(np.arange(-80.,81.,latres))
    m.drawmeridians(np.arange(-180.,181.,lonres))

    l = []
    ptype_legend = []
    for k in range(len(sites_assim_type_set)):
        marker = proxy_symbols_color[sites_assim_type_set[k]][0]
        color_dots = proxy_symbols_color[sites_assim_type_set[k]][1]
        
        inds = [i for i, j in enumerate(sites_assim_type) if j == sites_assim_type_set[k]]
        lats = np.asarray([sites_assim_lat[i] for i in inds])
        lons = np.asarray([sites_assim_lon[i] for i in inds])
        x, y = m(lons,lats)
        l.append(m.scatter(x,y,25,marker=marker,color=color_dots,edgecolor='black',linewidth='0.5',zorder=4))

        ptype_legend.append('%s (%d)' %(sites_assim_type_set[k],len(inds)))
        
    plt.title("Assimilated proxies: %s sites" % len(sites_assim_name))
    plt.legend(l,ptype_legend,
               scatterpoints=1,
               loc='lower center', bbox_to_anchor=(0.5, -0.30),
               ncol=3,
               fontsize=8)

    plt.savefig('%s/map_assim_proxies_%s.png' % (figdir,runname),bbox_inches='tight')
    plt.close()
    #plt.show()


    # ===============================================================================
    # 2) Time series of number of assimilated proxies
    # ===============================================================================

    nb_assim_proxies_total = np.zeros(shape=[nbtimes])
    for i in range(nbtimes):
        # total nb of proxies
        ind = [j for j, s in enumerate(sites_assim_years) if recon_times[i] in s]
        nb_assim_proxies_total[i] = len(ind)

    # counts per proxy type
    counts_dict = {}
    t = 0
    for p in proxy_types:
        inds  = [k for k, j in enumerate(sites_assim_type) if j == p]
        years = [sites_assim_years[k] for k in inds] 
        for i in range(nbtimes):
            ind = [j for j, s in enumerate(years) if recon_times[i] in s]
            nb_assim_proxies_type[iter,t,i] = len(ind)
        counts_dict[p] = nb_assim_proxies_type[iter,t,:]
        t += 1

    p_ordered_reverse = sorted(counts_dict, key=lambda k: np.max(counts_dict[k]), reverse=True)
    p_ordered = sorted(counts_dict, key=lambda k: np.max(counts_dict[k]))

    counts_cumul = np.zeros(shape=[nbtimes])
    counts_dict_cumul = {}
    for p in p_ordered:
        counts_cumul = counts_cumul + counts_dict[p]
        counts_dict_cumul[p] = counts_cumul

    iters_counts_dict.append(counts_dict)
    iters_counts_dict_cumul.append(counts_dict_cumul)
    
    # plot
    if logaxis_temporal_plots:
        p1 = plt.semilogy(recon_times,nb_assim_proxies_total,color='#000000',linewidth=3,label='Total')
        minval = 1
    else:
        p1 = plt.plot(recon_times,nb_assim_proxies_total,color='#000000',linewidth=3,label='Total')
        minval = 0
        
    xmin,xmax,ymin,ymax = plt.axis()
    
    for p in p_ordered_reverse:
        if any(c > 0 for c in counts_dict_cumul[p]):
            #plt.fill_between(recon_times,minval,counts_dict[p],color=proxy_symbols_color[p][1],linewidth=2,label=p)
            plt.fill_between(recon_times,minval,counts_dict_cumul[p],color=proxy_symbols_color[p][1],linewidth=2,label=p)

    plt.title(runname,fontweight='bold')
    plt.xlabel('Year CE',fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative number of proxies',fontsize=14,fontweight='bold')
    plt.legend(loc='upper left',bbox_to_anchor=(1., 1.),ncol=1,numpoints=1,fontsize=8)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.85])

    plt.savefig('%s/ts_assim_proxies_%s.png' % (figdir,runname),bbox_inches='tight')
    plt.close()
    #plt.show()


# ===============================================================================
# 3) Time series of mean number of assimilated proxies across MC iters
# ===============================================================================

nbiters = len(iters_counts_dict)

mean_counts_dict = {}
for p in proxy_types:
    tmp = np.zeros(shape=[nbiters,nbtimes])
    for i in range(nbiters):
        tmp[i,:] = iters_counts_dict[i][p]

    mean_counts_peryear = np.mean(tmp, axis=0)
    mean_counts_dict[p] = mean_counts_peryear

p_ordered_reverse = sorted(mean_counts_dict, key=lambda k: np.max(mean_counts_dict[k]), reverse=True)
p_ordered = sorted(mean_counts_dict, key=lambda k: np.max(mean_counts_dict[k]))

counts_cumul = np.zeros(shape=[nbtimes])
counts_dict_cumul = {}
for p in p_ordered:
    counts_cumul = counts_cumul + mean_counts_dict[p]
    counts_dict_cumul[p] = counts_cumul


mean_nb_assim_proxies_total = counts_cumul

if logaxis_temporal_plots:
    p1 = plt.semilogy(recon_times,mean_nb_assim_proxies_total,color='#000000',linewidth=3,label='Total')
    minval = 1
else:
    p1 = plt.plot(recon_times,mean_nb_assim_proxies_total,color='#000000',linewidth=3,label='Total')
    minval = 0
    
xmin,xmax,ymin,ymax = plt.axis()

for p in p_ordered_reverse:
        plt.fill_between(recon_times,minval,counts_dict_cumul[p],color=proxy_symbols_color[p][1],linewidth=2,label=p)

plt.xlabel('Year CE',fontsize=14, fontweight='bold')
plt.ylabel('Cumulative number of proxies',fontsize=14, fontweight='bold')
plt.legend(loc='upper left',bbox_to_anchor=(1., 1.),ncol=1,numpoints=1,fontsize=8)
plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.85])

plt.savefig('%s/ts_assim_proxies_meanMCiters.png' % (figdir),bbox_inches='tight')
plt.close()


#==========================================================================================
