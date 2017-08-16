
# coding: utf-8

# In[1]:

import glob, os
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature
#get_ipython().magic(u'matplotlib inline')

# iplot = 0: none; 1: all 2: most important
iplot = 0

# figure size
#plt.rcParams["figure.figsize"] = [10,10]

# set the global colormap to identify proxis
#'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
#cmap = {'Ice:dD':'tab:blue', 'Lake:':'tab:gray', 'Coral:Rate':'tab:pink', 'Ice:d18O':'tab:brown', 'Tree:Dens':'tab:purple', 'Coral:SrCa':'tab:orange', 'Coral:d18O':'tab:red', 'Tree:Width':'tab:green'}

cmap = {'Ice:dD':'blue', 'Lake:':'gray', 'Coral:Rate':'pink', 'Ice:d18O':'brown', 'Tree:Dens':'purple', 'Coral:SrCa':'orange', 'Coral:d18O':'red', 'Tree:Width':'green'}

# In[2]:

# specify a directory with the Ye pickle file
#nexp = 'testdevMultiState'
#nexp = 'test_V2proto'
#nexp = 'dadt_test'
#nexp = 'pages2_loc12000'

# new---for file saving
#pth = '/Users/hakim/data/LMR/archive/'
#pth = '/home/disk/kalman3/hakim/LMR/'
pth = '/home/disk/kalman3/rtardif/LMR/output/'

# GH files:
#nexp = 'pages2_loc12000_pages2k2_seasonal_TorP'
#nexp = 'pages2_loc12000'
#nexp = 'pages2_loc25000_pages2k2_seasonal_TorP_nens200'
#nexp = 'pages2_loc12000_breit_seasonal_TorP/'
# RT files:
#nexp = 'p2_ccsm4LM_n100_GISTEMPseasonPSM_PAGES2kv2_pf0.75/'
#nexp = 'p2_ccsm4LM_n100_GISTEMPannual_All_pf0.75/'
#nexp = 'p2_ccsm4LM_n100_linTorP_GISTEMPGPCCseasonPSM_PAGES2kv2_pf0.75'
#nexp = 'p2_ccsm4LM_n100_linTorP_GISTEMPGPCCseasonMETA_PAGES2kv2_pf0.75'
#nexp ='p2_ccsm4LM_n100_bilin_GISTEMPGPCCseasonMETA_PAGES2kv2_pf0.75'
#nexp = 'p2_ccsm4LM_n100_bilin_GISTEMPGPCCseasonPSM_PAGES2kv2_pf0.75'
#nexp = 'p2_ccsm4LM_n100_linTorP_GISTEMPGPCCseasonPSM_PAGES2kv2Breit_pf0.75' # this one crashed
nexp = 'p2_ccsm4LM_n100_linTorPallTRW_GISTEMPGPCCseasonPSM_PAGES2kv2Breit_pf0.75_loc12k'

# get a listing of the iteration directories
workdir = pth + nexp
dirs = glob.glob(workdir+"/r*")
niters = len(dirs)
print 'number of iterations: ' + str(niters)

# dictionary and counter for directories
iterdict = {}
cdir = -1
for dir in dirs[0:10]:

    cdir = cdir + 1
    print '\n\nloading proxies in ' + dir + ' ...\n'
    filn = dir + '/analysis_Ye.pckl'
    infile = open(filn,'rb')
    Ye_data = cPickle.load(infile)
    infile.close()

    nproxies = len(Ye_data)
    print 'number of proxies:',nproxies

    # In[4]:

    # screen for known proxies, date ranges, and identify the records that are missed in the process

    print 'First pass over Ye data to screen for known proxies and determine data ranges...'
    # initialize counters to zero
    trw=trd=trd=cd18=id18=csrca=crate=idd=lake = 0
    nkeys = len(Ye_data.keys())
    # data for unclassified proxies
    miss = {}

    minyear = 2000
    maxyear = -9999

    # hard-coded proxy "groups"
    for key in Ye_data.keys():
        years = Ye_data[key]['years']
        if years[0] < minyear:
            minyear = years[0]
        if years[-1] > maxyear:
            maxyear = years[-1]
        #print key,years[0],years[-1]

        if 'Tree' in key[0] and 'Width' in key[0]:
            trw = trw + 1
        elif 'Tree' in key[0] and 'Dens' in key[0]:
            trd = trd + 1
        elif 'Coral' in key[0] and 'd18O' in key[0]:
            cd18 = cd18 + 1
        elif 'Ice' in key[0] and 'd18O' in key[0]:
            id18 = id18 + 1
        elif 'Coral' in key[0] and 'SrCa' in key[0]:
            csrca = csrca + 1
        elif 'Coral' in key[0] and 'Rate' in key[0]:
            crate = crate + 1
        elif 'Ice' in key[0] and 'dD' in key[0]:
            idd = idd + 1
        elif 'Lake'in key[0]:
            lake = lake + 1
        else:
            miss[key] = key[0]

    nyears = int(maxyear-minyear+1)
    print 'minyear: ' + str(minyear)
    print 'maxyear: ' + str(maxyear)
    print 'nyears: ' + str(nyears)

    print 'identified proxies: ' + str(trw) + ' ' + str(trd) + ' ' + str(cd18) + ' ' + str(id18) + ' ' + str(csrca)+ ' ' + str(lake)+ ' ' + str(crate) + ' ' + str(idd)
    print 'total identified: ' + str(trw+trd+cd18+id18+csrca+idd+crate+lake)
    print 'total number of proxies:' + str(nkeys)
    print 'missing from the analysis: ' + str(len(miss)) + ' : ' + str(miss)


    # In[5]:

    """

    Loop over proxies in a *defined dictionary with target lists*, then loop over the Ye data dictionary to find all of the matches 
    (cf. looping over keys in Ye estimates as above). Compute observation influence and store as follows:

    key variables:
    sens: influence for a single proxy, all years
    S: influence for a single proxy, averaged over all years


    # p counts the _total_ number of proxies in a given year (-1 since counts starts at zero)
    # Ye has shape [nproxies,Nens] for a given given proxy _type_ and a given year
    # S has the diagonal elements of the observation sensitivity matrix (nproxies,nyears)
    # sS = trace(S) "degrees of freedom from signal (proxies)"
    # bS = trace(I-S) nproxies - trace(S) "degrees of freedom from background"

    """

    # use dictionaries and lists
    proxies = {'Tree':['Width','Dens'],'Coral':['d18O','SrCa','Rate'],'Ice':['d18O','dD'],'Lake':['']}

    Syears = np.zeros([nproxies,nyears])
    missing_value = 0.

    # list for count by proxy group
    tn = []
    # order lists for sum (sn: sum(S) for each proxy group; alls)
    sn = []
    alls = []
    allp = []
    # dictionaries to save influence by proxy group
    sall = {}
    latall = {}
    lonall = {}
    Rall = {}
    # p is a counter for all Ye
    p = -1
    for rootkey in proxies.keys():
        print 'working on ' + rootkey + '...'
        for t in proxies[rootkey]:        
            # process data one proxy at a time
            # n is a counter for each Ye proxy group (e.g. TRW is one class, Coral d18O is another, etc.)
            n = 0
            S = []
            for key in Ye_data.keys():
                found = False
                if rootkey in key[0] and t in key[0]:
                    found = True
                    n = n + 1
                    p = p + 1
                    years = Ye_data[key]['years']
                    # indexing for array storage relative to the earliest year in the reconstruction (minyear)
                    isyr = int(years[0] - minyear)
                    ieyr = int(years[-1] - minyear + 1)
                    Ye = Ye_data[key]['HXa']
                    var_Ye = Ye.var(1,ddof=1)
                    R = Ye_data[key]['R']
                    # observation influence for all years (single proxy)
                    sens = var_Ye/R
                    # observation influence averaged over all years (single proxy)
                    Sm = np.mean(sens)
                    # store the time-mean in lists (S: only one proxy; alls: all proxies, with allp as the rootkey)
                    S.append(Sm)
                    alls.append(Sm)
                    allp.append(rootkey)
                    # store the influence as a function of proxies and time too
                    # skip records with missing years
                    if len(years) < ieyr-isyr:
                        Syears[p,isyr:ieyr] = missing_value
                    else:
                        Syears[p,isyr:ieyr] = sens

                    # save a dictionary with lists of S for each proxy. Need to append and replace
                    newkey = rootkey+':'+t
                    if newkey in sall:
                        # get the existing list, append the new value, and set back into the dictionary
                        svals = sall[newkey]
                        svals.append(Sm)
                        sall[newkey] = svals
                        latvals = latall[newkey]
                        latvals.append(Ye_data[key]['lat'])
                        latall[newkey] = latvals
                        lonvals = lonall[newkey]
                        lonvals.append(Ye_data[key]['lon'])
                        lonall[newkey] = lonvals
                        rvals = Rall[newkey]
                        rvals.append(R)
                        Rall[newkey] = rvals
                    else:
                        sall[newkey] = [Sm]
                        latall[newkey] = [Ye_data[key]['lat']]
                        lonall[newkey] = [Ye_data[key]['lon']]
                        Rall[newkey] = [R]
            tn.append(n)
            sn.append(np.sum(S))

    nens = np.shape(Ye)[1]
    print 'ensemble size: ' + str(nens)
    print 'total proxies classified: ' + str(np.sum(tn))
    print 'total proxies in datafile: ' + str(len(Ye_data.keys()))


    # In[6]:

    # now make plots


    # In[7]:

    sS = np.sum(Syears,0)
    bS = nproxies - sS
    print np.shape(sS)
    #print 'degrees of freedom from proxies:' + str(sS)
    #print 'degrees of freedom from background:' + str(bS)
    GAI = sS/p
    #print 'GAI (global observation influence):' + str(GAI)

    pyrs = range(int(minyear),int(maxyear)+1)
    print len(pyrs)
    len(GAI)
    if iplot > 1:
        plt.plot(pyrs,GAI)
        plt.show()

        a = np.array(alls)
        b = np.array(allp)
        c = np.argsort(a)

        nkeys = len(Ye_data.keys())
        clr = np.ones(len(a))
        print np.shape(clr)
        h = plt.plot(a[c[::-1]],'ro')
        plt.show()
        #plt.semilogy(a[c[::-1]])
        # change these plots so that dots are plotted with colors correpsonding to proxy type

        cs = np.cumsum(a[c[::-1]])
        print np.max(cs/cs[-1])
        plt.plot(cs/cs[-1])
        xl = plt.xlim()
        plt.plot(xl,([1,1]))
        plt.xlim([-10,nkeys])
        plt.show()

    # In[8]:

    #total proxy impact:
    tpi = 0.

    print '\n\n--------------------------------------------------------------------------'
    print 'experiment: ' + dir
    for key in sall.keys():
        svals = sall[key]
        nsites = len(svals)
        sums = np.sum(svals)
        print 'proxy ' + key + ' has ' + str(nsites) + ' locations // total impact: ' + '{0!s:.4}'.format(sums) + ' per proxy: ' + '{0!s:.4}'.format(sums/nsites)
        tpi = tpi + sums
        if cdir == 0:
            iterdict[key] = [[sums],[nsites]]
        else:
            tmp = iterdict[key]
            tmp[0].append(sums)
            tmp[1].append(nsites)
            iterdict[key] = tmp
            
    print 'total impact: ' + '{0!s:.4}'.format(tpi)
    print '--------------------------------------------------------------------------\n\n'


    # In[9]:

    if iplot >0:
        """Plot all proxies on one map with symbol size scaled by impact and color by proxy group"""

        # these are the keys for all proxy groups
        print sall.keys()

        #ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=-90.))
        ax.coastlines()
        #ax.add_feature(cartopy.feature.LAND,alpha=0.5,facecolor='tan')

        for key in sorted(sall):
            splot = sall[key]
            latplot = latall[key]
            lonplot = lonall[key]
            msize = np.multiply(splot,500.)
            c = cmap[key]
            ax.scatter(lonplot,latplot,marker='o',color=c,alpha=0.75,label=key,s=msize,transform=ccrs.PlateCarree(),)

        #lgnd = ax.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
        lgnd = ax.legend()
        #lgnd.legendHandles[0]._legmarker.set_markersize(6) # this doesn't set the legend markers to the same size
        #handles, labels = ax.get_legend_handles_labels()

        ax.set_global()
        plt.savefig(nexp+'_impact_global_s_markers.png',dpi=300,bbox_inches='tight')
        plt.show()


    # In[10]:

    if iplot > 1:
        for key in sall:
            print key

            # plot sorted influence by proxy group
            splot = sall[key]
            xvals = np.arange(1,len(splot)+1)
            ssort = sorted(splot,reverse=True)
            c = cmap[key]
            plt.semilogx(xvals,ssort,c=c,label=key)

        # add line for all values
        xvals = np.arange(1,len(alls)+1)
        plt.semilogx(xvals,sorted(alls,reverse=True),'k-',label='all',alpha=0.5)
        plt.ylim([0.,0.75])
        plt.legend()
        plt.show()

        # alternative sorted influence over all proxies by cumulative sum 
        plt.plot(np.cumsum(sorted(alls,reverse=True))/np.sum(alls))
        xl = plt.xlim()
        plt.plot(xl,([1,1]),'k--')
        plt.xlim([-10,nkeys])
        plt.show()


    # In[11]:

    # influence by proxy group

    # grand total for normalization
    gti = np.sum(alls)
    gai = gti/len(alls)
    print 'global total impact=' '{0!s:.4}'.format(gti) + '\n global total impact per proxy= ' '{0!s:.4}'.format(gai)

    spai = 0.
    spti = 0.
    lpti = []
    lpai = []
    rstore = []
    for key in sall:
        print key

        # total and average influence for the group
        pti = np.sum(sall[key])
        pai = pti/len(sall[key])
        lpti.append(pti)
        lpai.append(pai)
        # express as percentage contribution
        print '{0!s:.4}'.format(100.*pti/gti) + '% // per proxy: ' + '{0!s:.4}'.format(pai)
        spai = spai + pai
        spti = spti + pti
        rstore.append(np.mean(Rall[key]))

    print 'sum partial impact: ' + '{0!s:.4}'.format(spti) + '     // per proxy:  ' '{0!s:.4}'.format(spai)

    if iplot > 0:
        # total impact by group
        index = np.arange(len(lpti))
        plt.bar(index,lpti/(gti/100.))
        plt.xticks(index, sall.keys())
        plt.ylabel('Percentage of Total Impact')
        plt.title('Proxy Group Total Impact')

        plt.savefig(nexp+'_impact_proxy_total.png',dpi=300)
        plt.show()   

        # per-proxy impact by group
        index = np.arange(len(lpai))
        plt.bar(index,lpai)
        plt.xticks(index, sall.keys())
        plt.title('Proxy Group Impact per Proxy (global total impact=' + '{0!s:.4}'.format(gti) + '; per proxy= ' + '{0!s:.4}'.format(gai)+')')

        plt.savefig(nexp+'_impact_per_proxy.png',dpi=300)
        plt.show()


    # In[12]:

    print np.shape(rstore)
    rstore


    # In[13]:

    # construct full S and check properties relative to theory
    tyear = 1950 
    samp = np.zeros([nproxies,nens])
    n = -1
    for key in Ye_data.keys():
        check = True
        Ye = Ye_data[key]['HXa']
        # analyze a single year to start
        years = Ye_data[key]['years']
        #print years[0],years[-1]
        try:
            tind = np.nonzero(years==tyear)[0][0]
            #print tind
            #print np.shape(Ye[tind,:])
        except:
            check = False

        if check:
            n = n + 1
            #var_Ye = Ye[tind,:].var(ddof=1)
            R = Ye_data[key]['R']
            sens = Ye[tind,:]/R
            #print np.shape(sens)
            samp[n,:] = sens
    print 'done'


    # In[14]:

    # need to remove the sample (row) mean!
    S = np.dot(samp,np.transpose(samp))/(nens-1)
    print np.shape(S)
    print np.max(S)
    print np.min(S)
    #print np.linalg.matrix_rank(S)


    # In[15]:

    rpp = []
    for key in sall:
        print key
        prsum = np.sum(Rall[key])
        rpp.append(prsum/len(Rall[key]))

    print rpp
    index = np.arange(len(rpp))
    if iplot >2:
        plt.bar(index,np.log(rpp))
        plt.xticks(index, sall.keys())
        plt.title('log Proxy R')
        plt.show()

    # In[ ]:

    """
    to do:
    0. add iteration over iteration subdirectories. option for stats on iterations.
    1. determine which trees are moisture and temperature sensitive and break down the influence
    2. math an implementation for patterns/functionals (J)
    3. application of optimal proxy-pattern update and statistics
    """


print '\n\n--------------------------------------------------------------------------'
print 'experiment: ' + workdir
print 'number of iterations: ' +str(niters)

print '\n proxy //  number // total impact // per-proxy impact ' 
mtpi = 0.
for key in iterdict.keys():
    tmp = iterdict[key]
    sums = tmp[0]
    nsites = tmp[1]
    msums = np.mean(sums)
    mstd = np.std(sums)
    msites = np.mean(nsites)
    pps = np.divide(sums,nsites)
    mpps = np.mean(pps)
    spps = np.std(pps)
    print ' ' + key + ' // ' + str(int(msites)) + ' //  ' + '{0!s:.4}'.format(msums) + '+/-' + '{0!s:.4}'.format(mstd) + ' // ' + '{0!s:.5}'.format(mpps) + '+/-' + '{0!s:.4}'.format(spps)
    mtpi = mtpi + msums

print 'mean total impact: ' + '{0!s:.4}'.format(mtpi)
print '--------------------------------------------------------------------------\n\n'
