
#==========================================================================================
# 
#==========================================================================================

import os
import glob
import re
import cPickle
import numpy as np

from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors


#expdir = '/home/disk/kalman3/rtardif/LMR/output/ReconDevTest_100_testing_callable_GMTfix2_*'
#expdir = '/home/disk/kalman3/rtardif/LMR/output/Recon_ens100_allAnnualProxyTypes_pf0.5/r*'

expdir = '/home/disk/kalman3/rtardif/LMR/output/Recon_ens100_allAnnualProxyTypes_pf0.5'
iter_range = [0,50]

# ======================================================
# 1) Time series of global mean temperature
# ======================================================

list_iters = []
iters = np.arange(iter_range[0], iter_range[1]+1)
for iter in iters:
    dirname = expdir+'/r'+str(iter)
    print iter, dirname
    list_iters.append(dirname)

nbiters = len(list_iters)
print nbiters, list_iters

gmt_data           = np.load(list_iters[0]+'/'+'gmt.npz')
recon_times        = gmt_data['recon_times']
recon_gmt_data     = gmt_data['gmt_save']
[nbproxy, nbyears] = recon_gmt_data.shape

recon_years = np.zeros([nbiters,nbyears])
recon_gmt   = np.zeros([nbiters,nbyears])

citer = 0
for d in list_iters:

    # Read in the file of assimilated proxies
    fname = d+'/assimilated_proxies.npy'
    assim_proxies = np.load(fname)
    nb_assim_proxies = len(assim_proxies)

    # File of global mean values
    fname = d+'/gmt.npz'
    gmt_data = np.load(fname)
    recon_proxy_gmt = gmt_data['gmt_save']
    [nbproxy, nbyears] = recon_gmt_data.shape

    for k in range(nbproxy):
        print k, recon_proxy_gmt[k]

    # Final reconstruction
    recon_years[citer,:] = gmt_data['recon_times']
    recon_gmt[citer,:]   = recon_proxy_gmt[nb_assim_proxies] 
    # remove prior mean
    prior_gmt            = recon_proxy_gmt[0]
    #recon_gmt[citer,:]   = recon_gmt[citer,:] - prior_gmt
    citer = citer + 1

if nbiters > 1:
    recon_gmt_ensmean = np.mean(recon_gmt,axis=0)
    recon_gmt_enssprd = np.std(recon_gmt,axis=0)
    recon_gmt_ensmin  = np.amin(recon_gmt,axis=0)
    recon_gmt_ensmax  = np.amax(recon_gmt,axis=0)

    #figdir_root = '/home/disk/ekman/rtardif/nobackup/LMR/output'
    #figdir_root = '/home/disk/kalman3/rtardif/LMR/output'
    figdir_root = expdir
    figdir = figdir_root+'/DiagFigs'
    if not os.path.isdir(figdir):
        os.chdir(figdir_root)
        os.system('mkdir DiagFigs')
else:
    recon_gmt_ensmean = recon_gmt[0,:]
    recon_gmt_enssprd = np.zeros([nbyears])
    recon_gmt_ensmin  = recon_gmt[0,:]
    recon_gmt_ensmax  = recon_gmt[0,:]
    figdir = expdir+'/DiagFigs'
    if not os.path.isdir(figdir):
        os.chdir(expdir)
        os.system('mkdir DiagFigs')

# => plot +/- spread in the various realizations
#recon_gmt_upp = recon_gmt_ensmean + recon_gmt_enssprd
#recon_gmt_low = recon_gmt_ensmean - recon_gmt_enssprd
# => plot +/- min-max among the various realizations
recon_gmt_upp = recon_gmt_ensmax
recon_gmt_low = recon_gmt_ensmin


# ==================================
# Gridded observational T2m products
# ==================================

# *** NOAA ***
from load_gridded_data import read_gridded_data_NOAA
datadir_calib  = '/home/disk/ekman/rtardif/nobackup/LMR/data/analyses'
datafile_calib = 'er-ghcn-sst.nc'
[noaa_time,lat,lon,temp] = read_gridded_data_NOAA(datadir_calib,datafile_calib,['Tsfc'])
[ntime,nlat,nlon] = temp.shape
verif_gmt_noaa = np.zeros([ntime])
X_lat = np.zeros(shape=[nlat,nlon])
X_lon = np.zeros(shape=[nlat,nlon])
for j in xrange(0,nlat):
    for k in xrange(0,nlon):
        X_lat[j,k] = lat[j]
        X_lon[j,k] = lon[k]
lat2 = np.reshape(X_lat,(nlat,nlon))
lon2 = np.reshape(X_lon,(nlat,nlon))
lat_weight = np.cos(np.deg2rad(lat2[:,0]))
#lat_weight = np.cos(np.deg2rad(lat))
tm_lat = np.nanmean(temp,2)
for i in xrange(tm_lat.shape[0]): # loop over time dimension
    verif_gmt_noaa[i] = np.nanmean(np.multiply(lat_weight,tm_lat[i,:]))


# *** GISTEMP ***
from load_gridded_data import read_gridded_data_GISTEMP
datadir_calib  = '/home/disk/ekman/rtardif/nobackup/LMR/data/analyses'
datafile_calib = 'gistemp1200_ERSST.nc'
[gistemp_time,lat,lon,temp] = read_gridded_data_GISTEMP(datadir_calib,datafile_calib,['Tsfc'])
[ntime,nlat,nlon] = temp.shape
verif_gmt_gistemp = np.zeros([ntime])
X_lat = np.zeros(shape=[nlat,nlon])
X_lon = np.zeros(shape=[nlat,nlon])
for j in xrange(0,nlat):
    for k in xrange(0,nlon):
        X_lat[j,k] = lat[j]
        X_lon[j,k] = lon[k]
lat2 = np.reshape(X_lat,(nlat,nlon))
lon2 = np.reshape(X_lon,(nlat,nlon))
lat_weight = np.cos(np.deg2rad(lat2[:,0]))
#lat_weight = np.cos(np.deg2rad(gis_lat))
tm_lat = np.nanmean(temp,2)
for i in xrange(tm_lat.shape[0]): # loop over time dimension
    verif_gmt_gistemp[i] = np.nanmean(np.multiply(lat_weight,tm_lat[i,:]))

# *** HadCRUT ***
from load_gridded_data import read_gridded_data_HadCRUT
datadir_calib  = '/home/disk/ekman/rtardif/nobackup/LMR/data/analyses'
datafile_calib = 'HadCRUT.4.3.0.0.median.nc'
[hadcrut_time,lat,lon,temp] = read_gridded_data_HadCRUT(datadir_calib,datafile_calib,['Tsfc'])
[ntime,nlat,nlon] = temp.shape
verif_gmt_hadcrut = np.zeros([ntime])
X_lat = np.zeros(shape=[nlat,nlon])
X_lon = np.zeros(shape=[nlat,nlon])
for j in xrange(0,nlat):
    for k in xrange(0,nlon):
        X_lat[j,k] = lat[j]
        X_lon[j,k] = lon[k]
lat2 = np.reshape(X_lat,(nlat,nlon))
lon2 = np.reshape(X_lon,(nlat,nlon))
lat_weight = np.cos(np.deg2rad(lat2[:,0]))
tm_lat = np.nanmean(temp,2)
for i in xrange(tm_lat.shape[0]): # loop over time dimension
    verif_gmt_hadcrut[i] = np.nanmean(np.multiply(lat_weight,tm_lat[i,:]))


# *** BerkeleyEarth ***
from load_gridded_data import read_gridded_data_BerkeleyEarth
datadir_calib  = '/home/disk/ekman/rtardif/nobackup/LMR/data/analyses'
datafile_calib = 'Land_and_Ocean_LatLong1.nc'
[berkeley_time,lat,lon,temp] = read_gridded_data_BerkeleyEarth(datadir_calib,datafile_calib,['Tsfc'])
[ntime,nlat,nlon] = temp.shape
verif_gmt_berkeley = np.zeros([ntime])
X_lat = np.zeros(shape=[nlat,nlon])
X_lon = np.zeros(shape=[nlat,nlon])
for j in xrange(0,nlat):
    for k in xrange(0,nlon):
        X_lat[j,k] = lat[j]
        X_lon[j,k] = lon[k]
lat2 = np.reshape(X_lat,(nlat,nlon))
lon2 = np.reshape(X_lon,(nlat,nlon))
lat_weight = np.cos(np.deg2rad(lat2[:,0]))
tm_lat = np.nanmean(temp,2)
for i in xrange(tm_lat.shape[0]): # loop over time dimension
    verif_gmt_berkeley[i] = np.nanmean(np.multiply(lat_weight,tm_lat[i,:]))

# ----------------
# Time series plot
# ----------------

p1 = plt.plot(recon_years[0,:],recon_gmt_ensmean,'-r',linewidth=3, label='LMR')
plt.fill_between(recon_years[0,:], recon_gmt_low, recon_gmt_upp,facecolor='red',alpha=0.2,linewidth=0.0)

p2 = plt.plot(gistemp_time,verif_gmt_gistemp,'-g',linewidth=1, alpha = 0.6, label='GISTEMP')
p3 = plt.plot(hadcrut_time,verif_gmt_hadcrut,'-c',linewidth=1, alpha = 0.6, label='HadCRUT')
p4 = plt.plot(noaa_time,verif_gmt_noaa,'-m',linewidth=1, alpha = 0.6, label='NOAA')
p5 = plt.plot(berkeley_time,verif_gmt_berkeley,'-b',linewidth=1, alpha = 0.6, label='BerkeleyEarth')

xmin,xmax,ymin,ymax = plt.axis()
p0 = plt.plot([xmin,xmax],[0,0],'--',color='0.50',linewidth=1)
plt.title('LMR')
plt.xlabel('Year',fontsize=14)
plt.ylabel('Global-mean temperature anomaly',fontsize=14)

plt.legend( loc='lower right', numpoints = 1,fontsize=12)

plt.savefig('%s/ts_recon_gmt_compar.png' % (figdir),bbox_inches='tight')
plt.close()
#plt.show()


# ===============================================================================
# 2) Map of assimilated proxies
# ===============================================================================

citer = 0
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
    #m = Basemap(projection='npstere',boundinglat=40,lon_0=270,resolution='l', area_thresh=700.0); latres = 10.; lonres=20.   # ARCTIC
    #m = Basemap(projection='spstere',boundinglat=-40,lon_0=180,resolution='l', area_thresh=700.0); latres = 10.; lonres=20.  # ANTARCTIC

    #water = '#C2F0FF'
    #water = '#99ffff'
    #water = '#5CB8E6'
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
        #print sites_assim_type_set[k], inds, [sites_assim_type[i] for i in inds]
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


    # ===============================================================================
    # 3) Time series of number of assimilated proxies
    # ===============================================================================

    nbtimes = recon_times.shape[0]
    nbtypes = len(sites_assim_type_set)
    nb_assim_proxies_total = np.zeros(shape=[nbtimes])
    nb_assim_proxies_type  = np.zeros(shape=[nbtypes,nbtimes])

    for i in range(nbtimes):
        # total nb of proxies
        ind = [j for j, s in enumerate(sites_assim_years) if recon_times[i] in s]
        nb_assim_proxies_total[i] = len(ind)
        # nb of proxies per type
        for t in xrange(len(sites_assim_type_set)):
            inds  = [k for k, j in enumerate(sites_assim_type) if j == sites_assim_type_set[t]]
            years = [sites_assim_years[k] for k in inds]            
            ind = [j for j, s in enumerate(years) if recon_times[i] in s]
            nb_assim_proxies_type[t,i] = len(ind)


    p1 = plt.plot(recon_times,nb_assim_proxies_total,color='#000000',linewidth=3,label='Total')
    xmin,xmax,ymin,ymax = plt.axis()
    #plt.title('LMR: Assimilated proxies')
    plt.xlabel('Year',fontsize=14)
    plt.ylabel('Number of assimilated proxies',fontsize=14)
    for t in xrange(len(sites_assim_type_set)):
        plt.plot(recon_times,nb_assim_proxies_type[t,:],color=proxy_color[sites_assim_type_set[t]],linewidth=2,label=sites_assim_type_set[t])

    plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.27),ncol=3,numpoints=1,fontsize=11)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.85])

    plt.savefig('%s/ts_recon_NbAssimProxies.png' % (figdir),bbox_inches='tight')
    plt.close()
    #plt.show()


    # ===============================================================================
    # Verification stats :
    # ===============================================================================

    # ===============================================
    # Verification stats : Assimilated proxies
    # ===============================================

    PSM_Rcrit = 0.2 # eliminate sites w/ questionable PSM calibration 

    fname = d+'/reconstruction_assim_diag.pckl'
    
    if not os.path.isfile(fname):
        break

    infile     = open(fname,'rb')
    assim_data = cPickle.load(infile)
    infile.close()

    proxy_types_sites = assim_data.keys()

    sites_goodPSM       = []
    assim_lats          = []
    assim_lons          = []
    assim_stats_npts    = []
    assim_stats_me      = []
    assim_stats_rmse    = []
    assim_stats_corr    = []
    assim_stats_ce      = []
    assim_stats_EnsCal  = []
    assim_stats_PSMcorr = []

    for ts in proxy_types_sites:
        # restrict stats to "good" PSMs
        if 'PSMcorrel' in assim_data[ts].keys():
            if abs(assim_data[ts]['PSMcorrel']) >= PSM_Rcrit:
                sites_goodPSM.append(ts)
                assim_lats.append(assim_data[ts]['lat'])
                assim_lons.append(assim_data[ts]['lon'])
                assim_stats_npts.append(assim_data[ts]['NbEvalPts'])
                assim_stats_me.append(assim_data[ts]['EnsMean_MeanError'])
                assim_stats_rmse.append(assim_data[ts]['EnsMean_RMSE'])
                assim_stats_corr.append(assim_data[ts]['EnsMean_Corr'])
                assim_stats_ce.append(assim_data[ts]['EnsMean_CE'])
                assim_stats_EnsCal.append(np.power(assim_data[ts]['EnsMean_RMSE'],2)/(np.mean(np.power(assim_data[ts]['ts_EnsSpread'],2))+assim_data[ts]['PSMmse']))
                assim_stats_PSMcorr.append(assim_data[ts]['PSMcorrel'])
            
    print 'Assimilated =>',len(assim_stats_npts), 'sites'

    # ===============================================================================
    # Plots -------------------------------------------------------------------------
    # ===============================================================================

    # ========================================================
    # 4) Histogram of (recon, proxy) CORRELATION
    # ========================================================

    #  Reconstruction **correlation** w/ proxy data
    n, bins, patches = plt.hist(assim_stats_corr, histtype='stepfilled',normed=False)
    plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
    plt.title("Histogram: Correlation")
    plt.xlabel("Correlation")
    #plt.ylabel("Probability")
    plt.ylabel("Count")
    xmin,xmax,ymin,ymax = plt.axis()
    plt.axis((-1,1,ymin,ymax))

    plt.savefig('%s/recon_assim_stats_hist_corr.png' % (figdir),bbox_inches='tight')
    plt.close()


    # ========================================================
    # 5) Histogram of (recon, proxy) coeff. of efficiency (CE)
    # ========================================================

    #  Reconstruction **CE** w/ proxy data
    n, bins, patches = plt.hist(assim_stats_ce, histtype='stepfilled',normed=False)
    plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
    plt.title("Histogram: CE")
    plt.xlabel("CE")
    #plt.ylabel("Probability")
    plt.ylabel("Count")
    xmin,xmax,ymin,ymax = plt.axis()
    plt.axis((-1,1,ymin,ymax))

    plt.savefig('%s/recon_assim_stats_hist_ce.png' % (figdir),bbox_inches='tight')
    plt.close()

    # =========================================================
    # 6) Histogram of reconstruction ensemble calibration ratio
    # =========================================================

    #  Reconstruction **Ensemble calibration** w/ 
    n, bins, patches = plt.hist(assim_stats_EnsCal, histtype='stepfilled',normed=False)
    plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
    plt.title("Histogram: Ensemble calibration")
    plt.xlabel("Ensemble calibration ratio")
    #plt.ylabel("Probability")
    plt.ylabel("Count")
    xmin,xmax,ymin,ymax = plt.axis()
    plt.axis((xmin,xmax,ymin,ymax))

    plt.savefig('%s/recon_assim_stats_hist_EnsCal.png' % (figdir),bbox_inches='tight')
    plt.close()


    # ===============================================================================
    # 7) Time series: comparison of proxy values and reconstruction-estimated proxies 
    #    for every site used in verification
    #
    # 8) Scatter plots of proxy vs reconstruction (ensemble-mean)
    # ===============================================================================

    for k in xrange(len(sites_goodPSM)):

        s = sites_goodPSM[k]
        sitetype = s[0].replace (" ", "_")
        sitename = s[1]

        x  = assim_data[s]['ts_years']
        yp = assim_data[s]['ts_ProxyValues']
        yr = assim_data[s]['ts_EnsMean']
        yrlow = assim_data[s]['ts_EnsMean'] - assim_data[s]['ts_EnsSpread']
        yrupp = assim_data[s]['ts_EnsMean'] + assim_data[s]['ts_EnsSpread']

        # -----------
        # Time series
        # -----------
        p1 = plt.plot(x,yp,'-r',linewidth=2, label='Proxy',alpha=0.7)
        p2 = plt.plot(x,yr,'-b',linewidth=2, label='Reconstruction')
        plt.fill_between(x, yrlow, yrupp,alpha=0.4,linewidth=0.0)
        #plt.plot(x,yrlow,'--b',linewidth=2)
        #plt.plot(x,yrupp,'--b',linewidth=2)

        plt.title("Proxy vs reconstruction: %s" % str(s))
        plt.xlabel("Years")
        plt.ylabel("Proxy obs./estimate")
        xmin,xmax,ymin,ymax = plt.axis()
        plt.legend( loc='lower right', numpoints = 1, fontsize=11 )

        # Annotate with summary stats
        xmin,xmax,ymin,ymax = plt.axis()
        ypos = ymax-0.05*(ymax-ymin)
        xpos = xmin+0.025*(xmax-xmin)
        plt.text(xpos,ypos,'PSM corr = %s' %"{:.4f}".format(assim_stats_PSMcorr[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'ME = %s' %"{:.4f}".format(assim_stats_me[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'RMSE = %s' %"{:.4f}".format(assim_stats_rmse[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'Corr = %s' %"{:.4f}".format(assim_stats_corr[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'CE = %s' %"{:.4f}".format(assim_stats_ce[k]),fontsize=11,fontweight='bold')

        plt.savefig('%s/ts_recon_vs_proxy_assim_%s_%s.png' % (figdir, sitetype, sitename),bbox_inches='tight')
        plt.close()

        # ------------
        # Scatter plot
        # ------------
    
        minproxy = np.min(yp); maxproxy = np.max(yp)
        minrecon = np.min(yr); maxrecon = np.max(yr)
        vmin = np.min([minproxy,minrecon])
        vmax = np.max([maxproxy,maxrecon])

        plt.plot(yr,yp,'o',markersize=8,markerfacecolor='#5CB8E6',markeredgecolor='black',markeredgewidth=1)
        plt.title("Proxy vs reconstruction: %s" % str(s))
        plt.xlabel("Proxy estimates from reconstruction")
        plt.ylabel("Proxy values")
        plt.axis((vmin,vmax,vmin,vmax))
        xmin,xmax,ymin,ymax = plt.axis()
        # one-one line
        plt.plot([vmin,vmax],[vmin,vmax],'r--')

        # Annotate with summary stats
        xmin,xmax,ymin,ymax = plt.axis()
        ypos = ymax-0.05*(ymax-ymin)
        xpos = xmin+0.025*(xmax-xmin)
        plt.text(xpos,ypos,'PSM corr = %s' %"{:.4f}".format(assim_stats_PSMcorr[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'ME = %s' %"{:.4f}".format(assim_stats_me[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'RMSE = %s' %"{:.4f}".format(assim_stats_rmse[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'Corr = %s' %"{:.4f}".format(assim_stats_corr[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'CE = %s' %"{:.4f}".format(assim_stats_ce[k]),fontsize=11,fontweight='bold')

        plt.savefig('%s/scatter_recon_vs_proxy_assim_%s_%s.png' % (figdir, sitetype, sitename),bbox_inches='tight')
        plt.close()


    # =========================================================
    # Verification stats : Un-assimilated (independent) proxies
    # =========================================================

    PSM_Rcrit = 0.2 # eliminate sites w/ questionable PSM calibration 

    fname = d+'/reconstruction_verif_diag.pckl'
    
    if not os.path.isfile(fname):
        break

    infile     = open(fname,'rb')
    verif_data = cPickle.load(infile)
    infile.close()

    proxy_types_sites = verif_data.keys()

    sites_goodPSM       = []
    verif_lats          = []
    verif_lons          = []
    verif_stats_npts    = []
    verif_stats_me      = []
    verif_stats_rmse    = []
    verif_stats_corr    = []
    verif_stats_ce      = []
    verif_stats_EnsCal  = []
    verif_stats_PSMcorr = []

    for ts in proxy_types_sites:
        # restrict stats to "good" PSMs
        if 'PSMcorrel' in verif_data[ts].keys():
            if abs(verif_data[ts]['PSMcorrel']) >= PSM_Rcrit:
                sites_goodPSM.append(ts)
                verif_lats.append(verif_data[ts]['lat'])
                verif_lons.append(verif_data[ts]['lon'])
                verif_stats_npts.append(verif_data[ts]['NbEvalPts'])
                verif_stats_me.append(verif_data[ts]['EnsMean_MeanError'])
                verif_stats_rmse.append(verif_data[ts]['EnsMean_RMSE'])
                verif_stats_corr.append(verif_data[ts]['EnsMean_Corr'])
                verif_stats_ce.append(verif_data[ts]['EnsMean_CE'])
                verif_stats_EnsCal.append(np.power(verif_data[ts]['EnsMean_RMSE'],2)/(np.mean(np.power(verif_data[ts]['ts_EnsSpread'],2))+verif_data[ts]['PSMmse']))
                verif_stats_PSMcorr.append(verif_data[ts]['PSMcorrel'])
            
    print 'Verification =>',len(verif_stats_npts), 'sites'

    # ===============================================================================
    # Plots -------------------------------------------------------------------------
    # ===============================================================================

    # ========================================================
    # 9) Map of sites used in verification
    # ========================================================

    sites_verif_type = [item[0] for item in sites_goodPSM]
    sites_verif_type_set = list(set([item[0] for item in sites_goodPSM]))

    fig = plt.figure()
    ax  = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=40.            # GLOBAL
    #m = Basemap(projection='npstere',boundinglat=40,lon_0=270,resolution='l', area_thresh=700.0); latres = 10.; lonres=20.   # ARCTIC
    #m = Basemap(projection='spstere',boundinglat=-40,lon_0=180,resolution='l', area_thresh=700.0); latres = 10.; lonres=20.  # ANTARCTIC

    #water = '#C2F0FF'
    #water = '#99ffff'
    #water = '#5CB8E6'
    water = '#9DD4F0'
    continents = '#888888'
    m.drawmapboundary(fill_color=water)
    m.drawcoastlines(); m.drawcountries()
    m.fillcontinents(color=continents,lake_color=water)
    m.drawparallels(np.arange(-80.,81.,latres))
    m.drawmeridians(np.arange(-180.,181.,lonres))

    l = []
    for k in xrange(len(sites_verif_type_set)):
        color_dots = proxy_color[sites_verif_type_set[k]]
        inds = [i for i, j in enumerate(sites_verif_type) if j == sites_verif_type_set[k]]
        lats = np.asarray([verif_lats[i] for i in inds])
        lons = np.asarray([verif_lons[i] for i in inds])
        x, y = m(lons,lats)
        l.append(m.scatter(x,y,35,marker='o',color=color_dots,edgecolor='black',linewidth='1',zorder=4))

    plt.title("Verification proxies: %s sites" %len(sites_verif_type))
    plt.legend(l,sites_verif_type_set,
               scatterpoints=1,
               loc='lower center', bbox_to_anchor=(0.5, -0.25),
               ncol=3,
               fontsize=10)

    plt.savefig('%s/map_verif_proxies.png' % (figdir),bbox_inches='tight')
    plt.close()
    #plt.show()

    # ========================================================
    # 10) Histogram of (recon, proxy) CORRELATION
    # ========================================================

    #  Reconstruction **correlation** w/ proxy data
    n, bins, patches = plt.hist(verif_stats_corr, histtype='stepfilled',normed=False)
    plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
    plt.title("Histogram: Correlation")
    plt.xlabel("Correlation")
    #plt.ylabel("Probability")
    plt.ylabel("Count")
    xmin,xmax,ymin,ymax = plt.axis()
    plt.axis((-1,1,ymin,ymax))

    plt.savefig('%s/recon_verif_stats_hist_corr.png' % (figdir),bbox_inches='tight')
    plt.close()

    # ==========================================================
    # 11) Histogram of reconstruction ensemble calibration ratio
    # ==========================================================

    #  Reconstruction **Ensemble calibration** w/ 
    n, bins, patches = plt.hist(verif_stats_EnsCal, histtype='stepfilled',normed=False)
    plt.setp(patches, 'facecolor', '#5CB8E6', 'alpha', 0.75)
    plt.title("Histogram: Ensemble calibration")
    plt.xlabel("Ensemble calibration ratio")
    #plt.ylabel("Probability")
    plt.ylabel("Count")
    xmin,xmax,ymin,ymax = plt.axis()
    plt.axis((xmin,xmax,ymin,ymax))

    plt.savefig('%s/recon_verif_stats_hist_EnsCal.png' % (figdir),bbox_inches='tight')
    plt.close()


    # ================================================================================
    # 12) Time series: comparison of proxy values and reconstruction-estimated proxies 
    #     for every site used in verification
    #
    # 13) Scatter plots of proxy vs reconstruction (ensemble-mean)
    # ================================================================================

    #site = ('Tree ring_Width', 'Arc_07')

    for k in xrange(len(sites_goodPSM)):

        s = sites_goodPSM[k]
        sitetype = s[0].replace (" ", "_")
        sitename = s[1]

        x  = verif_data[s]['ts_years']
        yp = verif_data[s]['ts_ProxyValues']
        yr = verif_data[s]['ts_EnsMean']
        yrlow = verif_data[s]['ts_EnsMean'] - verif_data[s]['ts_EnsSpread']
        yrupp = verif_data[s]['ts_EnsMean'] + verif_data[s]['ts_EnsSpread']

        # -----------
        # Time series
        # -----------
        p1 = plt.plot(x,yp,'-r',linewidth=2, label='Proxy',alpha=0.7)
        p2 = plt.plot(x,yr,'-b',linewidth=2, label='Reconstruction')
        plt.fill_between(x, yrlow, yrupp,alpha=0.4,linewidth=0.0)
        #plt.plot(x,yrlow,'--b',linewidth=2)
        #plt.plot(x,yrupp,'--b',linewidth=2)

        plt.title("Proxy vs reconstruction: %s" % str(s))
        plt.xlabel("Years")
        plt.ylabel("Proxy obs./estimate")
        xmin,xmax,ymin,ymax = plt.axis()
        plt.legend( loc='lower right', numpoints = 1, fontsize=11 )

        # Annotate with summary stats
        xmin,xmax,ymin,ymax = plt.axis()
        ypos = ymax-0.05*(ymax-ymin)
        xpos = xmin+0.025*(xmax-xmin)
        plt.text(xpos,ypos,'PSM corr = %s' %"{:.4f}".format(verif_stats_PSMcorr[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'ME = %s' %"{:.4f}".format(verif_stats_me[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'RMSE = %s' %"{:.4f}".format(verif_stats_rmse[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'Corr = %s' %"{:.4f}".format(verif_stats_corr[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'CE = %s' %"{:.4f}".format(verif_stats_ce[k]),fontsize=11,fontweight='bold')

        plt.savefig('%s/ts_recon_vs_proxy_verif_%s_%s.png' % (figdir, sitetype, sitename),bbox_inches='tight')
        plt.close()

        # ------------
        # Scatter plot
        # ------------
    
        minproxy = np.min(yp); maxproxy = np.max(yp)
        minrecon = np.min(yr); maxrecon = np.max(yr)
        vmin = np.min([minproxy,minrecon])
        vmax = np.max([maxproxy,maxrecon])

        plt.plot(yr,yp,'o',markersize=8,markerfacecolor='#5CB8E6',markeredgecolor='black',markeredgewidth=1)
        plt.title("Proxy vs reconstruction: %s" % str(s))
        plt.xlabel("Proxy estimates from reconstruction")
        plt.ylabel("Proxy values")
        plt.axis((vmin,vmax,vmin,vmax))
        xmin,xmax,ymin,ymax = plt.axis()
        # one-one line
        plt.plot([vmin,vmax],[vmin,vmax],'r--')

        # Annotate with summary stats
        xmin,xmax,ymin,ymax = plt.axis()
        ypos = ymax-0.05*(ymax-ymin)
        xpos = xmin+0.025*(xmax-xmin)
        plt.text(xpos,ypos,'PSM corr = %s' %"{:.4f}".format(verif_stats_PSMcorr[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'ME = %s' %"{:.4f}".format(verif_stats_me[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'RMSE = %s' %"{:.4f}".format(verif_stats_rmse[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'Corr = %s' %"{:.4f}".format(verif_stats_corr[k]),fontsize=11,fontweight='bold')
        ypos = ypos-0.05*(ymax-ymin)
        plt.text(xpos,ypos,'CE = %s' %"{:.4f}".format(verif_stats_ce[k]),fontsize=11,fontweight='bold')

        plt.savefig('%s/scatter_recon_vs_proxy_verif_%s_%s.png' % (figdir, sitetype, sitename),bbox_inches='tight')
        plt.close()


#==========================================================================================
