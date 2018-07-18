#=================================================================================
# A function which adds proxy location markers to a map, using Basemap.  Inputs:
#    experiment_dir:  The path of the experiment directory.
#    m:               The map projection (using Basemap).
#    year:            The year for which the proxies should be plotted.  Set this
#                         to "all" to plot all proxies.
#    marker:          The marker symbol.  Set this to "proxy", "proxies", "type",
#                         or a string including one of those words to use a
#                         different symbol for each proxy.
#    size:            The size of the symbols.
#    color:           The color of the symbols.  Ignored if proxy types are used.
#    edgecolor:       The color of the symbol edges.
#    alpha:           The opacity of the symbols, 0 (invisible) to 1 (solid).
#
# Example:
#    m = Basemap(projection='robin',lon_0=180,resolution='c')
#    [Code for making a map with Basemap goes here.]
#    map_proxies.map_proxies(experiment_dir,m,'all','proxytypes',300,'b','k',1)
#
#       author: Michael Erb
#       date  : 5/17/2017
#=================================================================================

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import glob


# Plot the location of proxies on a map.  Specify a year or "all" for all years.  See the comments above for more customization.
def map_proxies(experiment_dir,m,year='all',marker='o',size=10,color='k',edgecolor='k',alpha=1):
    print("Plotting locations of assimilated proxies.")
    #
    directories = glob.glob(experiment_dir+'/r*')
    #
    proxy_names = []
    proxy_types = []
    proxy_lats = []
    proxy_lons = []
    #
    for iteration in range(0,len(directories)):
        # Load the assimilated proxies
        assimilated_proxies = np.load(directories[iteration]+'/assimilated_proxies.npy')
        #
        # Determine the names of all the assimilated proxies which exist for a given year.
        for i in range(0,len(assimilated_proxies)):
            proxy_type = list(assimilated_proxies[i].keys())[0]
            proxy_name = assimilated_proxies[i][proxy_type][0]
            #
            # Find only the proxies for the given year.  If a proxy has already been loaded, don't load it again.
            if ((year == "all") or (year in assimilated_proxies[i][proxy_type][3])) and (proxy_name not in proxy_names):
                #print assimilated_proxies[i][proxy_type][0]
                proxy_names.append(proxy_name)
                proxy_types.append(proxy_type)
                proxy_lats.append(assimilated_proxies[i][proxy_type][1])
                proxy_lons.append(assimilated_proxies[i][proxy_type][2])
    #
    # Plot the proxy locations.
    #     If the "marker" variable is set to "proxytypes," each proxy type gets a specific color and symbol.
    #     Otherwise all proxies are given the same maker.
    x,y = m(proxy_lons,proxy_lats)
    #
    # Initialize variables for the proxy counts, lons, and lats.
    #print np.unique(proxy_types)
    n_bivalve,n_borehole,n_coral,n_document,n_ice,n_hybrid,n_lake,n_marine,n_sclerosponge,n_speleothem,n_tree,n_other=0,0,0,0,0,0,0,0,0,0,0,0
    x_bivalve,x_borehole,x_coral,x_document,x_ice,x_hybrid,x_lake,x_marine,x_sclerosponge,x_speleothem,x_tree,x_other=[],[],[],[],[],[],[],[],[],[],[],[]
    y_bivalve,y_borehole,y_coral,y_document,y_ice,y_hybrid,y_lake,y_marine,y_sclerosponge,y_speleothem,y_tree,y_other=[],[],[],[],[],[],[],[],[],[],[],[]
    #
    if ("proxy" in marker) or ("proxies" in marker) or ("type" in marker):
        for i in range(0,len(proxy_names)):
            if   ("Bivalve"      in proxy_types[i]) or ("bivalve"      in proxy_types[i]): n_bivalve      += 1; x_bivalve.append(x[i]);      y_bivalve.append(y[i])
            elif ("Borehole"     in proxy_types[i]) or ("borehole"     in proxy_types[i]): n_borehole     += 1; x_borehole.append(x[i]);     y_borehole.append(y[i])
            elif ("Coral"        in proxy_types[i]) or ("coral"        in proxy_types[i]): n_coral        += 1; x_coral.append(x[i]);        y_coral.append(y[i])
            elif ("Document"     in proxy_types[i]) or ("document"     in proxy_types[i]): n_document     += 1; x_document.append(x[i]);     y_document.append(y[i])
            elif ("Ice"          in proxy_types[i]) or ("ice"          in proxy_types[i]): n_ice          += 1; x_ice.append(x[i]);          y_ice.append(y[i])
            elif ("Hybrid"       in proxy_types[i]) or ("hybrid"       in proxy_types[i]): n_hybrid       += 1; x_hybrid.append(x[i]);       y_hybrid.append(y[i])
            elif ("Lake"         in proxy_types[i]) or ("lake"         in proxy_types[i]): n_lake         += 1; x_lake.append(x[i]);         y_lake.append(y[i])
            elif ("Marine"       in proxy_types[i]) or ("marine"       in proxy_types[i]): n_marine       += 1; x_marine.append(x[i]);       y_marine.append(y[i])
            #elif ("Sclerosponge" in proxy_types[i]) or ("sclerosponge" in proxy_types[i]): n_sclerosponge += 1; x_sclerosponge.append(x[i]); y_sclerosponge.append(y[i])
            elif ("Speleothem"   in proxy_types[i]) or ("speleothem"   in proxy_types[i]): n_speleothem   += 1; x_speleothem.append(x[i]);   y_speleothem.append(y[i])
            elif ("Tree"         in proxy_types[i]) or ("tree"         in proxy_types[i]): n_tree         += 1; x_tree.append(x[i]);         y_tree.append(y[i])
            else:                                                                          n_other        += 1; x_other.append(x[i]);        y_other.append(y[i])
        #
        # Make a legend
        m.scatter(x_bivalve     ,y_bivalve     ,size,marker=(6,1,0),facecolor='Gold'        ,edgecolor=edgecolor,alpha=alpha,label='Bivalves ('+str(n_bivalve)+')')
        m.scatter(x_borehole    ,y_borehole    ,size,marker=(6,1,0),facecolor='DarkKhaki'   ,edgecolor=edgecolor,alpha=alpha,label='Boreholes ('+str(n_borehole)+')')
        m.scatter(x_coral       ,y_coral       ,size,marker='o'    ,facecolor='DarkOrange'  ,edgecolor=edgecolor,alpha=alpha,label='Corals and sclerosponges ('+str(n_coral)+')')
        m.scatter(x_document    ,y_document    ,size,marker='*'    ,facecolor='Black'       ,edgecolor=edgecolor,alpha=alpha,label='Documents ('+str(n_document)+')')
        m.scatter(x_ice         ,y_ice         ,size,marker='d'    ,facecolor='LightSkyBlue',edgecolor=edgecolor,alpha=alpha,label='Glacier ice ('+str(n_ice)+')')
        m.scatter(x_hybrid      ,y_hybrid      ,size,marker=(8,2,0),facecolor='DeepSkyBlue' ,edgecolor=edgecolor,alpha=alpha,label='Hybrid ('+str(n_hybrid)+')')
        m.scatter(x_lake        ,y_lake        ,size,marker='s'    ,facecolor='RoyalBlue'   ,edgecolor=edgecolor,alpha=alpha,label='Lake sediments ('+str(n_lake)+')')
        m.scatter(x_marine      ,y_marine      ,size,marker='s'    ,facecolor='SaddleBrown' ,edgecolor=edgecolor,alpha=alpha,label='Marine sediments ('+str(n_marine)+')')
        #m.scatter(x_sclerosponge,y_sclerosponge,size,marker='o'    ,facecolor='Red'         ,edgecolor=edgecolor,alpha=alpha,label='Sclerosponges ('+str(n_sclerosponge)+')')
        m.scatter(x_speleothem  ,y_speleothem  ,size,marker='d'    ,facecolor='DeepPink'    ,edgecolor=edgecolor,alpha=alpha,label='Speleothems ('+str(n_speleothem)+')')
        m.scatter(x_tree        ,y_tree        ,size,marker='^'    ,facecolor='LimeGreen'   ,edgecolor=edgecolor,alpha=alpha,label='Trees ('+str(n_tree)+')')
        m.scatter(x_other       ,y_other       ,size,marker='v'    ,facecolor='Black'       ,edgecolor=edgecolor,alpha=alpha,label='Other ('+str(n_other)+')')
        #
        proxylegend = plt.legend(loc=3,scatterpoints=1)
        proxylegend.get_frame().set_alpha(0)
        #
    else:
        m.scatter(x,y,size,marker=marker,color=color,edgecolor=edgecolor,alpha=alpha,label='Proxies ('+str(len(x))+')')
        proxylegend = plt.legend(loc=3,scatterpoints=1)
        proxylegend.get_frame().set_alpha(0)

        
