
import os
import glob
import re
import cPickle
import numpy as np

from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors


expdir = '/home/disk/kalman3/rtardif/LMR/output/Experiment1'
iter_range = [0,30]

list_iters = []
iters = np.arange(iter_range[0], iter_range[1]+1)
for iter in iters:
    dirname = expdir+'/r'+str(iter)
    print iter, dirname
    list_iters.append(dirname)


for d in list_iters:

    print 'Reconstruction:', d

    figdir = d+'/DiagFigs'
    if not os.path.isdir(figdir):
        os.chdir(d)
        os.system('mkdir DiagFigs')

    fname = d+'/assimilated_proxies.npy'

    # Dictionary to associate a color to every proxy type
    proxy_color = {'Tree ring_Width'       : '#99FF33', \
                   'Tree ring_Density'     : '#FFCC00', \
                   'Coral_d18O'            : '#FF8080', \
                   'Ice core_d18O'         : '#66FFFF', \
                   'Ice core_d2H'          : '#B8E6E6', \
                   'Ice core_Accumulation' : '#FFFFFF', \
                   'Lake sediment_All'     : '#FF00FF', \
                   'Marine sediment_All'   : '#FF0000', \
                   'Speleothem_All'        : '#996600', \
                  }

    # Read in the file of assimilated proxies for experiment
    assim_proxies = np.load(fname)

    sites_assim_name  = []
    sites_assim_type  = []
    sites_assim_lat   = []
    sites_assim_lon   = []
    sites_assim_years = []
    for k in xrange(len(assim_proxies)):
        key = assim_proxies[k].keys()
        sites_assim_name.append(assim_proxies[k][key[0]][0])
        sites_assim_type.append(key[0])
        sites_assim_lat.append(assim_proxies[k][key[0]][1])
        sites_assim_lon.append(assim_proxies[k][key[0]][2])
        sites_assim_years.append(assim_proxies[k][key[0]][3])

    sites_assim_type_set = list(set(sites_assim_type))
    nb_assim_proxies = len(sites_assim_name)
    print 'Assimilated proxies =>',len(sites_assim_name), 'sites'

    # determine colors of markers w.r.t. proxy type
    color_dots = []
    for k in xrange(len(sites_assim_type)):
        color_dots.append(proxy_color[sites_assim_type[k]])

    fig = plt.figure()
    ax  = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=40.            # GLOBAL

    water = '#9DD4F0'
    continents = '#888888'
    m.drawmapboundary(fill_color=water)
    m.drawcoastlines(); m.drawcountries()
    m.fillcontinents(color=continents,lake_color=water)
    m.drawparallels(np.arange(-80.,81.,latres))
    m.drawmeridians(np.arange(-180.,181.,lonres))

    l = []
    for k in xrange(len(sites_assim_type_set)):
        color_dots = proxy_color[sites_assim_type_set[k]]
        inds = [i for i, j in enumerate(sites_assim_type) if j == sites_assim_type_set[k]]
        lats = np.asarray([sites_assim_lat[i] for i in inds])
        lons = np.asarray([sites_assim_lon[i] for i in inds])
        x, y = m(lons,lats)
        l.append(m.scatter(x,y,35,marker='o',color=color_dots,edgecolor='black',linewidth='1',zorder=4))

    plt.title("Assimilated proxies: %s sites" % len(sites_assim_name))
    plt.legend(l,sites_assim_type_set,
               scatterpoints=1,
               loc='lower center', bbox_to_anchor=(0.5, -0.25),
               ncol=3,
               fontsize=10)

    plt.savefig('%s/map_assim_proxies.png' % (figdir),bbox_inches='tight')
    plt.close()
    #plt.show()

