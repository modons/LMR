"""
Module: LMR_verify_proxy_plot.py

Purpose: Plotting of summary statistics from proxy-based verification. Both proxy 
         chronologies that were assimilated to create reconstructions and those witheld for 
         independent verification are considered.
 
Input: Reads .pckl files containing verification data generated when running the 
       LMR_verify_proxy.py script.

Originator: Robert Tardif | Dept. of Atmospheric Sciences, Univ. of Washington
                          | October 2015

Revisions:

"""
import os
import numpy as np
import pickle    
from time import time
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.basemap import Basemap

# =========================================================================================
def roundup(x):
    if x <=100:
        n = 1
    elif 1000 > x > 100:
        n = 2
    elif 10000 > x >= 1000:
        n = 3
    else:
        n = 4      
    return int(round(x,-n))

# =========================================================================================
# START:  set user parameters here
# =========================================================================================

# ------------------------------
# Section 1: Plotting parameters
# ------------------------------

make_plots = True
make_plots_individual_sites = False
make_pdfs = True

# set the default size of the figure in inches. ['figure.figsize'] = width, height;  
plt.rcParams['figure.figsize'] = 10, 8  # that's default image size for this interactive session
plt.rcParams['axes.linewidth'] = 2.0 #set the value globally
plt.rcParams['font.weight'] = 'bold' #set the font weight globally
plt.rcParams['font.size'] = 11 #set the font size globally
#plt.rc('text', usetex=True)
plt.rc('text', usetex=False)


# Histogram plotting parameters
binwidth      = 0.05
CORRrange     = [-1,1]
CErange       = [-1,1]
CEchangerange = [-1,1]

alpha = 0.5

#fcolor = ['#5CB8E6', '#5CB8E6']
fcolor = ['blue', 'red']


# -------------------------
# Section 2: Proxy datasets
# -------------------------

proxies = 'PAGES'
#proxies = 'NCDC'


# Assign symbol to proxy types for plotting: dependent on proxy database used.
if proxies == 'PAGES':
    # PAGES proxies
    proxy_verif = {\
                   'Tree ring_Width'       :'o',\
                   'Tree ring_Density'     :'s',\
                   'Ice core_d18O'         :'v',\
                   'Ice core_d2H'          :'^',\
                   'Ice core_Accumulation' :'D',\
                   'Coral_d18O'            :'p',\
                   'Coral_Luminescence'    :'8',\
                   'Lake sediment_All'     :'<',\
                   'Marine sediment_All'   :'>',\
                   'Speleothem_All'        :'h',\
    }
elif proxies == 'NCDC':
    # NCDC proxies
    proxy_verif = {\
                   'Tree Rings_WoodDensity'        :'s',\
                   'Tree Rings_WidthPages'         :'o',\
                   'Tree Rings_WidthPages2'        :'o',\
                   'Tree Rings_WidthBreit'         :'o',\
                   'Tree Rings_Isotopes'           :'*',\
                   'Corals and Sclerosponges_d18O' :'p',\
                   'Corals and Sclerosponges_SrCa' :'8',\
                   'Corals and Sclerosponges_Rates':'D',\
                   'Ice Cores_d18O'                :'v',\
                   'Ice Cores_dD'                  :'^',\
                   'Ice Cores_Accumulation'        :'D',\
                   'Ice Cores_MeltFeature'         :'d',\
                   'Lake Cores_Varve'              :'<',\
                   'Lake Cores_BioMarkers'         :'>',\
                   'Lake Cores_GeoChem'            :'^',\
                   'Marine Cores_d18O'             :'H',\
                   'Speleothems_d18O'              :'h',\
    }
else:
    print('ERROR in the especification of the proxy dataset that will be considered. Exiting...')
    exit(1)
    
# Only keep proxy sites for which the linear PSM has a correlation >= than this value
r_crit = 0.0
#r_crit = 0.2


# ------------------------------------
# Section 3: Directories & experiments
# ------------------------------------

#datadir_input = '/home/disk/ekman4/rtardif/LMR/output'
datadir_input = '/home/disk/kalman3/rtardif/LMR/output'
#datadir_input = '/home/disk/kalman3/rtardif/LMR/output/verification_production_runs'

#nexp = 'production_gis_ccsm4_pagesall_0.75'
#nexp = 'production_mlost_ccsm4_pagesall_0.75'
nexp = 'test'

# - old -
#calib = 'MLOST'
#calib = 'GISTEMP'
# - new - 
calib = 'linear-MLOST'
#calib = 'bilinear-linear-GISTEMP-GPCC'

verif_period = [[1880,2000],[0,1879]]
#verif_period = [[1880,2000],[1759,1879]]

# Output directory, where the figs will be dumped.
#datadir_output = datadir_input # if want to keep things tidy
datadir_output = '.'          # if want local plots

# =========================================================================================
# END:  set user parameters here
# =========================================================================================


# =============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Main code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# =============================================================================
def main():

    begin_time = time()

    # =============================
    # Loading the verification data
    # =============================

    vtype = {'assim': 'Assimilated proxies', 'verif':'Non-assimilated proxies'}
    
    nbperiods = len(verif_period)
    assim_dict = [dict() for x in range(nbperiods)]
    verif_dict = [dict() for x in range(nbperiods)]

    # loop over verification periods & load data in dictionaries
    for p in range(nbperiods):
        # Read the pickle files containing summary stats
        """
        fname_assim = datadir_input+'/'+nexp+'/'+'verifProxy_PSMcalib'+calib+'_'+str(verif_period[p][0])+'to'+str(verif_period[p][1])+\
            '/reconstruction_eval_assim_proxy_summary.pckl'
        fname_verif = datadir_input+'/'+nexp+'/'+'verifProxy_PSMcalib'+calib+'_'+str(verif_period[p][0])+'to'+str(verif_period[p][1])+\
            '/reconstruction_eval_verif_proxy_summary.pckl'
        """
        fname_assim = datadir_input+'/'+nexp+'/'+'verifProxy_PSM_'+calib+'_'+str(verif_period[p][0])+'to'+str(verif_period[p][1])+\
            '/reconstruction_eval_assim_proxy_summary.pckl'
        fname_verif = datadir_input+'/'+nexp+'/'+'verifProxy_PSM_'+calib+'_'+str(verif_period[p][0])+'to'+str(verif_period[p][1])+\
            '/reconstruction_eval_verif_proxy_summary.pckl'

        
        infile_assim   = open(fname_assim,'r')
        assim_dict[p] = pickle.load(infile_assim)
        infile_assim.close()

        if os.path.isfile(fname_verif):
            infile_verif   = open(fname_verif,'r')
            verif_dict[p] = pickle.load(infile_verif)
            infile_verif.close()
            verif_data = True
        else:
            verif_data = False

    
    # ==================
    # Now creating plots
    # ==================

    if datadir_output != '.':
        figdir = datadir_output+'/figs'
        if not os.path.isdir(figdir):
            os.system('mkdir %s' % figdir)
    else:
        figdir = '.'

    # ============================================================================================================
    # 1) Histograms of (recon, proxy) CORRELATION, CE across grand ensemble for all proxy types and per proxy type
    # ============================================================================================================
    #fig = plt.figure()
    fig = plt.figure(figsize=(12,8))

    irow = 1
    for v in vtype.keys(): # "assim" & "verif" proxies

        if v == 'verif' and not verif_data:
            break
        
        ax_master = fig.add_subplot(2,1,irow)
        # Turn off axis lines and ticks of the big subplot 
        ax_master.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # Removes the white frame
        ax_master._frameon = False
        ax_master.set_title("%s\n" % vtype[v], fontsize=16, fontweight='bold')

        # ----------------------------------
        # Stats based on **all** proxy types
        # ----------------------------------

        facecolor = fcolor[0]
        if v == 'assim':
            pos = [1,2,3]
        else:
            pos = [4,5,6]

        bins_corr = np.arange(-1.-binwidth/2, 1.+binwidth/2, binwidth)
        bins_ce   = np.arange(-2.-binwidth/2, 1.+binwidth/2, binwidth)

        # 1) --- Correlation ---
        ax = fig.add_subplot(2,3,pos[0])

        prior_tmp = []
        for p in range(nbperiods):
            # pick right dict and associate to "workdict"
            dname = v+'_dict'
            workdict = eval(dname)
            sitetag = list(workdict[p].keys())
            proxy_types = list(set([item[0] for item in sitetag]))

            #tmp = [workdict[p][k]['GrandEnsCorr'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMcorrel'])>=r_crit]
            tmp = [workdict[p][k]['GrandEnsCorr'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMinfo']['corr'])>=r_crit]
            stat = [item for sublist in tmp for item in sublist] # flatten list of lists
            nbdata = len(stat)
            mean_stat = np.mean(stat)
            std_stat = np.std(stat)
            results, edges = np.histogram(stat, bins=bins_corr, normed=True)
            #plt.bar(edges[:-1]+binwidth/2,results*binwidth,binwidth,color=fcolor[p],alpha=alpha,linewidth=0,align="center",label=str(verif_period[p][0])+' to '+str(verif_period[p][1]))
            plt.bar(edges[:-1]+binwidth/2,results,binwidth,color=fcolor[p],alpha=alpha,linewidth=0,align="center",label=str(verif_period[p][0])+' to '+str(verif_period[p][1]))

            # Accumulate prior stat
            #tmp = [workdict[p][k]['PriorGrandEnsCorr'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMcorrel'])>=r_crit]
            tmp = [workdict[p][k]['PriorGrandEnsCorr'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMinfo']['corr'])>=r_crit]
            prior_tmp.append([item for sublist in tmp for item in sublist]) # flatten list of lists

        
        prior_corr = [item for sublist in prior_tmp for item in sublist]
        results, edges = np.histogram(prior_corr, bins=bins_corr, normed=True)
        #plt.plot(edges[:-1]+binwidth,results*binwidth,linewidth=1,ls='steps',color='black',label='Prior')
        plt.plot(edges[:-1]+binwidth,results,linewidth=1,ls='steps',color='black',label='Prior')

        plt.xlabel("Correlation",fontweight='bold')
        plt.ylabel("Probability density",fontweight='bold')
        xmin,xmax,ymin,ymax = plt.axis()
        ymin = 0.0
        #ymax = 0.04; nbins = 4
        #ymax = 0.05; nbins = 5 # for r_crit = 0.2
        #ymax = 0.1; nbins = 5
        ymax = 2.0; nbins = 5
        plt.axis((CORRrange[0],CORRrange[1],ymin,ymax))
        plt.locator_params(axis = 'y', nbins = nbins)
        plt.legend(loc=2,fontsize=10,frameon=False,handlelength=1)

            
        # 2) --- CE ---
        ax = fig.add_subplot(2,3,pos[1])

        prior_tmp = []
        for p in range(nbperiods):
            # pick right dict and associate to "workdict"
            dname = v+'_dict'
            workdict = eval(dname)
            sitetag = list(workdict[p].keys())
            proxy_types = list(set([item[0] for item in sitetag]))

            #tmp = [workdict[p][k]['GrandEnsCE'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMcorrel'])>=r_crit]
            tmp = [workdict[p][k]['GrandEnsCE'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMinfo']['corr'])>=r_crit]
            stat = [item for sublist in tmp for item in sublist] # flatten list of lists
            nbdata = len(stat)
            mean_stat = np.mean(stat)
            std_stat = np.std(stat)
            # Since CE is not bounded at the lower end, assign values smaller than 1st bin to value of 1st bin 
            #stat = [bins[0] if x<bins[0] else x for x in stat]
            results, edges = np.histogram(stat, bins=bins_ce, normed=True)
            #plt.bar(edges[:-1],results*binwidth,binwidth,color=fcolor[p],alpha=alpha,linewidth=0,label=str(verif_period[p][0])+' to '+str(verif_period[p][1]))
            plt.bar(edges[:-1],results,binwidth,color=fcolor[p],alpha=alpha,linewidth=0,label=str(verif_period[p][0])+' to '+str(verif_period[p][1]))

            # Accumulate prior stat
            #tmp = [workdict[p][k]['PriorGrandEnsCE'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMcorrel'])>=r_crit]
            tmp = [workdict[p][k]['PriorGrandEnsCE'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMinfo']['corr'])>=r_crit]
            prior_tmp.append([item for sublist in tmp for item in sublist]) # flatten list of lists

        prior_ce = [item for sublist in prior_tmp for item in sublist]
        # Since CE is not bounded at the lower end, assign values smaller than 1st bin to value of 1st bin 
        prior_ce = [bins_ce[0] if x<bins_ce[0] else x for x in prior_ce]
        results, edges = np.histogram(prior_ce, bins=bins_ce, normed=True)
        #plt.plot(edges[:-1]+binwidth,results*binwidth,linewidth=1,ls='steps',color='black',label='Prior')
        plt.plot(edges[:-1]+binwidth,results,linewidth=1,ls='steps',color='black',label='Prior')

        plt.xlabel("Coefficient of efficiency",fontweight='bold') 
        plt.ylabel("Probability density",fontweight='bold')
        xmin,xmax,ymin,ymax = plt.axis()
        ymin = 0.0
        #ymax = 0.45
        #ymax = 0.1 # for r_crit = 0.2
        #ymax = 0.5; nbins = 5
        ymax = 12.0; nbins = 6        

        plt.axis((CErange[0],CErange[1],ymin,ymax))
        plt.legend(loc=2,fontsize=10,frameon=False,handlelength=1)


        # 3) --- Change in CE from prior to posterior ---
        ax = fig.add_subplot(2,3,pos[2])

        prior_tmp = []
        for p in range(nbperiods):
            # pick right dict and associate to "workdict"
            dname = v+'_dict'
            workdict = eval(dname)
            sitetag = list(workdict[p].keys())
            proxy_types = list(set([item[0] for item in sitetag]))

            #tmpPost  = [workdict[p][k]['GrandEnsCE'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMcorrel'])>=r_crit]
            #tmpPrior = [workdict[p][k]['PriorGrandEnsCE'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMcorrel'])>=r_crit]
            tmpPost  = [workdict[p][k]['GrandEnsCE'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMinfo']['corr'])>=r_crit]
            tmpPrior = [workdict[p][k]['PriorGrandEnsCE'] for k in sitetag if k[0] in proxy_types and np.abs(workdict[p][k]['PSMinfo']['corr'])>=r_crit]
            statPost  = [item for sublist in tmpPost for item in sublist]  # flatten list of lists
            statPrior = [item for sublist in tmpPrior for item in sublist] # flatten list of lists

            # difference
            stat = [statPost[i]-statPrior[i] for i in range(len(statPost))]
            nbdata = len(stat)
            mean_stat = np.mean(stat)
            std_stat = np.std(stat)
            # % of positive change
            dCEplus = [stat[i] for i in range(len(stat)) if stat[i] > 0.0]
            frac = float(len(dCEplus))/float(len(stat))
            fractiondCEplus = str(float('%.2f' % frac ))
            print('CE_stats: period= ', str('%12s' %verif_period[p]), ' category= ', v, ':', str('%8s' %str(len(dCEplus))), str('%8s' %str(len(stat))), \
                ' Fraction of +change:', fractiondCEplus)

            results, edges = np.histogram(stat, bins=bins_ce, normed=True)
            plt.bar(edges[:-1],results,binwidth,color=fcolor[p],alpha=alpha,linewidth=0,label=str(verif_period[p][0])+' to '+str(verif_period[p][1]))

        plt.xlabel("Change in coefficient of efficiency",fontweight='bold') 
        plt.ylabel("Probability density",fontweight='bold')
        xmin,xmax,ymin,ymax = plt.axis()
        ymin = 0.0
        ymax = 8.0; nbins = 5        

        plt.axis((CEchangerange[0],CEchangerange[1],ymin,ymax))
        plt.legend(loc=2,fontsize=10,frameon=False,handlelength=1)



        irow = irow + 1

    fig.tight_layout()
    plt.savefig('%s/%s_verify_proxy_hist_corr_ce_Allproxies_PSM_%s.png' % (figdir,nexp,calib),bbox_inches='tight')
    if make_pdfs:
        plt.savefig('%s/%s_verify_proxy_hist_corr_ce_Allproxies_PSM_%s.pdf' % (figdir,nexp,calib),bbox_inches='tight',dpi=300, format='pdf')
    plt.close()

    # ==========================================================================
    # PART 2: MAPS of site-based verification metrics --------------------------
    # ==========================================================================

    water = '#9DD4F0'
    continents = '#888888'

    # Loop over proxy sets (assim vs verif)
    for v in vtype.keys():

        # Loop over verification periods
        for p in range(nbperiods):
            # pick right dict and associate to "workdict"
            dname = v+'_dict'
            workdict = eval(dname)
            sites = list(workdict[p].keys())
            proxy_types = list(set([item[0] for item in sitetag]))

            verif_period_label = str(verif_period[p][0])+'-'+str(verif_period[p][1])

            # ===========================================================================
            # 2) Maps with proxy sites plotted with dots colored according to correlation
            # ===========================================================================

            verif_metric = 'Correlation'

            mapcolor = plt.cm.seismic
            cbarfmt = '%4.1f'

            fmin = -1.0; fmax = 1.0
            fval = np.linspace(fmin, fmax, 100);  fvalc = np.linspace(0, fmax, 101);           
            scaled_colors = mapcolor(fvalc)
            cmap, norm = from_levels_and_colors(levels=fval, colors=scaled_colors, extend='both')
            cbarticks=np.linspace(fmin,fmax,11)

            fig = plt.figure(figsize=[8,5])
            #ax  = fig.add_axes([0.1,0.1,0.8,0.8])

            m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=40.            # GLOBAL
            
            m.drawmapboundary(fill_color=water)
            m.drawcoastlines(); m.drawcountries()
            m.fillcontinents(color=continents,lake_color=water)
            m.drawparallels(np.arange(-80.,81.,latres))
            m.drawmeridians(np.arange(-180.,181.,lonres))

            # loop over proxy sites
            l = []
            proxy_types = []
            for sitetag in sites:
                sitetype = sitetag[0]
                sitename = sitetag[1]
                sitemarker = proxy_verif[sitetype]

                lat = workdict[p][sitetag]['lat']
                lon = workdict[p][sitetag]['lon']
                x, y = m(lon,lat)
                if sitetype not in proxy_types:
                    proxy_types.append(sitetype)
                    l.append(m.scatter(x,y,35,c='white',marker=sitemarker,edgecolor='black',linewidth='1'))
                Gplt = m.scatter(x,y,35,c=workdict[p][sitetag]['MeanCorr'],marker=sitemarker,edgecolor='black',linewidth='1',zorder=4,cmap=cmap,norm=norm)

            cbar = m.colorbar(Gplt,location='right',pad="2%",size="2%",ticks=cbarticks,format=cbarfmt,extend='both')
            cbar.outline.set_linewidth(1.0)
            cbar.set_label('%s' % verif_metric,size=11,weight='bold')
            cbar.ax.tick_params(labelsize=10)
            plt.title('Period: '+verif_period_label+' : '+vtype[v],fontweight='bold')
            plt.legend(l,proxy_types,
                       scatterpoints=1,
                       loc='lower center', bbox_to_anchor=(0.5, -0.30),
                       ncol=3,
                       fontsize=9)

            plt.savefig('%s/%s_verify_proxy_map_%s_corr_%s.png' % (figdir,nexp,v,verif_period_label),bbox_inches='tight')
            if make_pdfs:
                plt.savefig('%s/%s_verify_proxy_map_%s_corr_%s.pdf' % (figdir,nexp,v,verif_period_label),bbox_inches='tight', dpi=300, format='pdf')
            plt.close()


            # ===========================================================================
            # 3) Maps with proxy sites plotted with dots colored according to CE
            # ===========================================================================

            verif_metric = 'Coefficient of efficiency'

            mapcolor = plt.cm.seismic
            cbarfmt = '%4.1f'

            fmin = -1.0; fmax = 1.0
            fval = np.linspace(fmin, fmax, 100);  fvalc = np.linspace(0, fmax, 101);           
            scaled_colors = mapcolor(fvalc)
            cmap, norm = from_levels_and_colors(levels=fval, colors=scaled_colors, extend='both')
            cbarticks=np.linspace(fmin,fmax,11)

            # Prior & Posterior
            fig = plt.figure(figsize=[8,10])

            dplot = {'Prior':'PriorMeanCE', 'Posterior':'MeanCE'}
            irow = 1
            for dd in list(dplot.keys()):

                ax = fig.add_subplot(2,1,irow)   
                m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=40.            # GLOBAL            
                m.drawmapboundary(fill_color=water)
                m.drawcoastlines(); m.drawcountries()
                m.fillcontinents(color=continents,lake_color=water)
                m.drawparallels(np.arange(-80.,81.,latres))
                m.drawmeridians(np.arange(-180.,181.,lonres))

                # loop over proxy sites
                l = []
                proxy_types = []
                for sitetag in sites:
                    sitetype = sitetag[0]
                    sitename = sitetag[1]
                    sitemarker = proxy_verif[sitetype]

                    lat = workdict[p][sitetag]['lat']
                    lon = workdict[p][sitetag]['lon']
                    x, y = m(lon,lat)
                    if sitetype not in proxy_types:
                        proxy_types.append(sitetype)
                        l.append(m.scatter(x,y,35,c='white',marker=sitemarker,edgecolor='black',linewidth='1'))

                    plot_var = dplot[dd]
                    Gplt = m.scatter(x,y,35,c=workdict[p][sitetag][plot_var],marker=sitemarker,edgecolor='black',linewidth='1',zorder=4,cmap=cmap,norm=norm)

                cbar = m.colorbar(Gplt,location='right',pad="2%",size="2%",ticks=cbarticks,format=cbarfmt,extend='both')
                cbar.outline.set_linewidth(1.0)
                cbar.set_label('%s' % verif_metric,size=11,weight='bold')
                cbar.ax.tick_params(labelsize=10)
                if irow == 1:
                    plt.title('Period: '+verif_period_label+'\n\n'+vtype[v]+' : '+ dd,fontweight='bold')
                else:
                    plt.title(vtype[v]+' : '+ dd,fontweight='bold')

                irow = irow + 1

            plt.legend(l,proxy_types,
                       scatterpoints=1,
                       loc='lower center', bbox_to_anchor=(0.5, -0.30),
                       ncol=3,
                       fontsize=9)

            #fig.tight_layout()
            plt.savefig('%s/%s_verify_proxy_map_%s_ce_%s.png' % (figdir,nexp,v,verif_period_label),bbox_inches='tight')
            if make_pdfs:
                plt.savefig('%s/%s_verify_proxy_map_%s_ce_%s.pdf' % (figdir,nexp,v,verif_period_label),bbox_inches='tight', dpi=300, format='pdf')
            plt.close()
    
            # ============================================================================
            # 4) Maps with proxy sites plotted with dots colored according to change in CE
            # ============================================================================
    
            # Change in CE from Prior to Posterior
            fig = plt.figure(figsize=[8,5])

            m = Basemap(projection='robin', lat_0=0, lon_0=0,resolution='l', area_thresh=700.0); latres = 20.; lonres=40.            # GLOBAL            
            m.drawmapboundary(fill_color=water)
            m.drawcoastlines(); m.drawcountries()
            m.fillcontinents(color=continents,lake_color=water)
            m.drawparallels(np.arange(-80.,81.,latres))
            m.drawmeridians(np.arange(-180.,181.,lonres))

            # loop over proxy sites
            l = []
            proxy_types = []
            for sitetag in sites:
                sitetype = sitetag[0]
                sitename = sitetag[1]
                sitemarker = proxy_verif[sitetype]

                lat = workdict[p][sitetag]['lat']
                lon = workdict[p][sitetag]['lon']
                x, y = m(lon,lat)
                if sitetype not in proxy_types:
                    proxy_types.append(sitetype)
                    l.append(m.scatter(x,y,35,c='white',marker=sitemarker,edgecolor='black',linewidth='1'))

                plot_var = workdict[p][sitetag]['MeanCE'] - workdict[p][sitetag]['PriorMeanCE']
                Gplt = m.scatter(x,y,35,c=plot_var,marker=sitemarker,edgecolor='black',linewidth='1',zorder=4,cmap=cmap,norm=norm)
            cbar = m.colorbar(Gplt,location='right',pad="2%",size="2%",ticks=cbarticks,format=cbarfmt,extend='both')
            cbar.outline.set_linewidth(1.0)
            cbar.set_label('Change in coefficient of efficiency',size=11,weight='bold')
            cbar.ax.tick_params(labelsize=10)
            plt.title('Period: '+verif_period_label+' : '+vtype[v],fontweight='bold')
            plt.legend(l,proxy_types,
                       scatterpoints=1,
                       loc='lower center', bbox_to_anchor=(0.5, -0.30),
                       ncol=3,
                       fontsize=9)

            #fig.tight_layout()
            plt.savefig('%s/%s_verify_proxy_map_%s_delta_ce_%s.png' % (figdir,nexp,v,verif_period_label),bbox_inches='tight')
            if make_pdfs:
                plt.savefig('%s/%s_verify_proxy_map_%s_delta_ce_%s.pdf' % (figdir,nexp,v,verif_period_label),bbox_inches='tight', dpi=300, format='pdf')
            plt.close()








    end_time = time() - begin_time
    print('=======================================================')
    print('All completed in '+ str(end_time/60.0)+' mins')
    print('=======================================================')



# =============================================================================

if __name__ == '__main__':
    main()
