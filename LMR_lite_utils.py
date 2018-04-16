"""
Support functions for the 'lite" version of LMR driver for Python2.

Originator:

Greg Hakim
University of Washington
26 February 2018

Modifications:
"""

import os
import numpy as np
import sys
import yaml
import itertools
import datetime
import LMR_driver_callable as LMR
import LMR_config
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point

from LMR_utils import validate_config, ensemble_stats
import LMR_utils as Utils

from time import time
import LMR_prior
import LMR_proxy_pandas_rework
import LMR_utils
import pandas as pd

def make_gmt_figure(analysis_data,analysis_time,fsave=None):

    # work in progress...
    
    alpha = 0.5 # alpha transparency
    lw = 1 # line width
    dset_color={'LMR':'k','GIS':'r','CRU':'m','BE':'g','MLOST':'c','CON':'lime'}
    for dset in list(analysis_data.keys()):
        time = analysis_time['time']
        gm = analysis_data[dset]
        stime = time[0]
        etime =time[-1]
        if dset == 'LMR':
            lww = lw*2
            alphaa = 1.
        else:
            lww = lw
            alphaa = alpha
            
        plt.plot(time,gm,color=dset_color[dset],linewidth=lww,label=dset,alpha=alphaa)

    plt.title('Global mean temperature',weight='bold',y=1.025)
    plt.xlabel('Year CE',fontweight='bold')
    plt.ylabel('Temperature anomaly (K)',fontweight='bold')
    xl_loc = [stime,etime]
    yl_loc = [-1.,1.]

    plt.xlim(xl_loc)
    plt.ylim(yl_loc)
    plt.legend(loc=2)
    if fsave:
        print('saving to .png')
        plt.savefig(fsave+'_GMT_annual.png',dpi=300)
        
"""
    plt.fill_between(recon_times,gmt_min,gmt_max,facecolor='gray',alpha = 0.5,linewidth=0.)
    txl = xl_loc[0] + (xl_loc[1]-xl_loc[0])*.45
    tyl = yl_loc[0] + (yl_loc[1]-yl_loc[0])*.2
    offset = 0.05

    plt.text(txl,tyl,'(LMR,GISTEMP)  : r= ' + lgc.ljust(5,' ') + ' CE= ' + lgce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,HadCRUT4) : r= ' + lcc.ljust(5,' ') + ' CE= ' + lcce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,BE)       : r= ' + lbc.ljust(5,' ') + ' CE= ' + lbce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,MLOST)    : r= ' + lmc.ljust(5,' ') + ' CE= ' + lmce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,20CR-V2)  : r= ' + ltc.ljust(5,' ') + ' CE= ' + ltce.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,ERA-20C)  : r= ' + lec.ljust(5,' ') + ' CE= ' + lece.ljust(5,' '), fontsize=14, family='monospace')
    tyl = tyl-offset
    plt.text(txl,tyl,'(LMR,consensus): r= ' + loc.ljust(5,' ') + ' CE= ' + loce.ljust(5,' '), fontsize=14, family='monospace')

    plt.plot(xl_loc,[0,0],color='gray',linestyle=':',lw=2)

"""

def make_plot(vplot,grid,figsize=10,savefig=None,vmax=None):

    # add a wrap point for smooth plotting
    vplot_wrap, lon_wrap = add_cyclic_point(vplot, coord=grid.lon[0,:], axis=1)

    # figure size
    plt.rcParams["figure.figsize"] = [figsize,figsize]

    ax = plt.axes(projection=ccrs.Robinson(central_longitude=-90.))
    ax.coastlines()
    #cs = ax.contourf(lon,lat,vplot,transform=ccrs.PlateCarree(),cmap='bwr')
    if vmax:
        maxv = vmax
    else:
        maxv = np.nanmax(np.abs(vplot))
    cs = ax.pcolormesh(lon_wrap,grid.lat[:,0],vplot_wrap,transform=ccrs.PlateCarree(),cmap='bwr',shading='flat',vmin=-maxv,vmax=maxv)
    plt.colorbar(cs, extend='both', shrink=0.4)
    if savefig:
        plt.title(savefig)
        plt.savefig(savefig+'.png',dpi=300)
    plt.show()

class Grid(object):
    def __init__(self,X=None):

        if X:
        # use first variable set in config file
            var = list(X.statevars.keys())[0]

            lat = X.prior_dict[var]['lat']
            lon = X.prior_dict[var]['lon']
            nlat = np.shape(lat)[0]
            nlon = np.shape(lon)[1]

            self.lat = lat
            self.lon = lon
            self.nlat = nlat
            self.nlon = nlon
            self.Nens = X.Nens
            self.nens = X.Nens

def make_grid(X):
    # make an empty class as a handy container for grid information
    
    class Grid:
        pass
    
    # use first variable set in config file
    var = list(X.statevars.keys())[0]

    lat = X.prior_dict[var]['lat']
    lon = X.prior_dict[var]['lon']
    nlat = np.shape(lat)[0]
    nlon = np.shape(lon)[1]

    g = Grid
    g.lat = lat
    g.lon = lon
    g.nlat = nlat
    g.nlon = nlon
    g.Nens = X.Nens
    g.nens = X.Nens
    
    return g


def make_random_ensemble(Xb_one,max_ens,nens,ranseed=None):

    """
    Purpose: provide random column draws from an existing ensemble matrix
    
    Inputs:
        Xb_one: an ensemble state matrix of form (nx,max_ens)
        max_ens: the maximum number of samples that can be drawn from Xb_one
        nens: number of random draws from Xb_one
        ranseed (optional): seed the random number generator for repeatability
    
    Outputs:
        Xb_one_new: the random sample from columns of Xb_one
    """
    
    from numpy.random import sample, seed
    
    begin_time = time()

    # option to seed rng for repeatability
    if ranseed != None:
        np.random.seed(ranseed)

    # this works, but may have repeat values?
    #ens_inds = np.random.randint(0,max_ens+1,nens)
    # no repeat values
    ens_inds = np.random.choice(np.arange(max_ens),size=nens,replace=False)

    # new, random ensemble from the master read in above
    Xb_one_new = Xb_one[:,ens_inds]

    elapsed_time = time() - begin_time
    print('-----------------------------------------------------')
    print('completed in ' + str(elapsed_time) + ' seconds')
    print('-----------------------------------------------------')

    return Xb_one_new, ens_inds


def make_random_proxies(prox_manager,Ye,Ye_coords,ens_inds,max_proxies,nproxies,ranseed=None,verbose=False):

    """
    Purpose: provide random column draws from an existing ensemble matrix
    
    Inputs:
        prox_manager: proxy_manager object
        Ye: array of proxy estimates from the prior
        Ye_coords: lat,lon for Ye
        ens_inds: indices that define the ensemble members in the prior
        max_proxies: the maximum number of samples that can be drawn from Xb_one
        nproxies: number of random draws from Xb_one
        ranseed (optional): seed the random number generator for repeatability
    
    Outputs:
        Xb_one_new: the random sample from columns of Xb_one
    """
    
    from numpy.random import sample, seed

    begin_time = time()

    # option to seed rng for repeatability
    if ranseed != None:
        np.random.seed(ranseed)

    # this works, but may have repeat values?
    #prox_inds = np.random.randint(0,max_proxies+1,nproxies)
    # no repeat values
    prox_inds = np.random.choice(np.arange(max_proxies),size=nproxies,replace=False)

    # new, random proxy set
    k = -1
    vR = []
    vP = []
    vT = []
    for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
        if proxy_idx in prox_inds: 
            k = k + 1
            ob_err = Y.psm_obj.R
            vR.append(ob_err)
            vP.append(proxy_idx)
            vT.append(Y.type)

    vYe = Ye[vP,:][:,ens_inds]
    vYe_coords = Ye_coords[vP,:]
    
    elapsed_time = time() - begin_time
    if verbose:
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    return vR, vP, vT, vYe, vYe_coords


def make_proxy_group(prox_manager,pgroup,Ye,Ye_coords,ens_inds,verbose=False):
    """
    Purpose: provide single proxy group and Ye from an existing ensemble matrix
    
    Inputs:
        prox_manager: proxy_manager object
        pgroup: name of the proxy group to filter on
        Ye: array of proxy estimates from the prior
        Ye_coords: lat,lon for Ye
        ens_inds: indices that define the ensemble members in the prior
    
    Outputs:
        
    """

    begin_time = time()

    k = -1
    vR = []
    vP = []
    vT = []
    for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
        if Y.type == pgroup:
            k = k + 1
            ob_err = Y.psm_obj.R
            vR.append(ob_err)
            vP.append(proxy_idx)
            vT.append(Y.type)

    vYe = Ye[vP,:][:,ens_inds]
    vYe_coords = Ye_coords[vP,:]

    if verbose:
        elapsed_time = time() - begin_time
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    return vR, vP, vT, vYe, vYe_coords
 

def load_config(verbose=False):
    begin_time = time()

    if not LMR_config.LEGACY_CONFIG:
        if len(sys.argv) > 3:
            yaml_file = sys.argv[1]
        else:
            yaml_file = os.path.join(LMR_config.SRC_DIR, 'config_lite.yml')

        try:
            if verbose: print('Loading configuration: {}'.format(yaml_file))
            f = open(yaml_file, 'r')
            yml_dict = yaml.load(f)
            update_result = LMR_config.update_config_class_yaml(yml_dict,
                                                                LMR_config)

            # Check that all yml params match value in LMR_config
            if update_result:
                raise SystemExit(
                    'Extra or mismatching values found in the configuration yaml'
                    ' file.  Please fix or remove them.\n  Residual parameters:\n '
                    '{}'.format(update_result))

        except IOError as e:
            raise SystemExit(
                ('Could not locate {}.  If use of legacy LMR_config usage is '
                 'desired then please change LEGACY_CONFIG to True'
                 'in LMR_wrapper.py.').format(yaml_file))

    # Define main experiment output directory
    iter_range = LMR_config.wrapper.iter_range
    expdir = os.path.join(LMR_config.core.datadir_output, LMR_config.core.nexp)
    arc_dir = os.path.join(LMR_config.core.archive_dir, LMR_config.core.nexp)

    # Check if it exists, if not, create it
    if not os.path.isdir(expdir):
        os.system('mkdir {}'.format(expdir))

    # Monte-Carlo approach: loop over iterations (range of iterations defined in
    # namelist)
    MCiters = range(iter_range[0], iter_range[1]+1)
    param_iterables = [MCiters]

    # get other parameters to sweep over in the reconstruction
    param_search = LMR_config.wrapper.param_search
    if param_search is not None:
        # sort them by parameter name and combine into a list of iterables
        sort_params = list(param_search.keys())
        sort_params.sort(key=lambda x: x.split('.')[-1])
        param_values = [param_search[key] for key in sort_params]
        param_iterables = param_values + [MCiters]

    for iter_and_params in itertools.product(*param_iterables):

        iter_num = iter_and_params[-1]
        cfg_dict = Utils.param_cfg_update('core.curr_iter', iter_num)

        if LMR_config.wrapper.multi_seed is not None:
            curr_seed = LMR_config.wrapper.multi_seed[iter_num]
            cfg_dict = Utils.param_cfg_update('core.seed', curr_seed,
                                              cfg_dict=cfg_dict)
            #print ('Setting current iteration seed: {}'.format(curr_seed))

        itr_str = 'r{:d}'.format(iter_num)
        # If parameter space search is being performed then set the current
        # search space values and create a special sub-directory
        if param_search is not None:
            curr_param_values = iter_and_params[:-1]
            cfg_dict, psearch_dir = Utils.psearch_list_cfg_update(sort_params,
                                                                  curr_param_values,
                                                                  cfg_dict=cfg_dict)

            working_dir = os.path.join(expdir, psearch_dir, itr_str)
            mc_arc_dir = os.path.join(arc_dir, psearch_dir, itr_str)
        else:
            working_dir = os.path.join(expdir, itr_str)
            mc_arc_dir = os.path.join(arc_dir, itr_str)

        cfg_params = Utils.param_cfg_update('core.datadir_output', working_dir,
                                            cfg_dict=cfg_dict)

        cfg = LMR_config.Config(**cfg_params)

        proceed = validate_config(cfg)
        if not proceed:
            raise SystemExit()
        else:
            print('OK!')
            pass
    
    if verbose:
        elapsed_time = time() - begin_time
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    return cfg 


def load_prior(cfg,verbose=False):
    
    core = cfg.core
    prior = cfg.prior
    nexp = core.nexp
    workdir = core.datadir_output
    
    begin_time = time()

    # Define the number of assimilation times
    recon_times = np.arange(core.recon_period[0], core.recon_period[1]+1,core.recon_timescale)
    ntimes, = recon_times.shape

    # prior
    if verbose: print('Source for prior: ', prior.prior_source)

    # Assign prior object according to "prior_source" (from namelist)
    X = LMR_prior.prior_assignment(prior.prior_source)
    X.prior_datadir = prior.datadir_prior
    X.prior_datafile = prior.datafile_prior
    X.statevars = prior.state_variables
    X.statevars_info = prior.state_variables_info
    X.Nens = core.nens
    X.anom_reference = prior.anom_reference
    X.detrend = prior.detrend
    X.avgInterval = prior.avgInterval
    
    # Read data file & populate initial prior ensemble
    X.populate_ensemble(prior.prior_source, prior)
    Xb_one_full = X.ens

    # Prepare to check for files in the prior (work) directory (this object just
    # points to a directory)
    prior_check = np.DataSource(workdir)

    # this is a hack that skips over regridding option
    X.trunc_state_info = X.full_state_info
    Xb_one = Xb_one_full
    Xb_one_coords = X.coords
    [Nx, _] = Xb_one.shape

    # Keep dimension of pre-augmented version of state vector
    [state_dim, _] = Xb_one.shape

    if verbose:
        elapsed_time = time() - begin_time
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    return X, Xb_one


def load_proxies(cfg,verbose=True):

    core = cfg.core

    begin_time = time()

    # Build dictionaries of proxy sites to assimilate and those set aside for
    # verification
    prox_manager = LMR_proxy_pandas_rework.ProxyManager(cfg, core.recon_period)

    if verbose:
        # count the total number of proxies
        type_site_assim = prox_manager.assim_ids_by_group
        assim_proxy_count = len(prox_manager.ind_assim)
        for pkey, plist in sorted(type_site_assim.items()):
            print(('%45s : %5d' % (pkey, len(plist))))
        print(('%45s : %5d' % ('TOTAL', assim_proxy_count)))

        elapsed_time = time() - begin_time
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')
        
    return prox_manager


def get_valid_proxies(cfg,prox_manager,target_year,Ye_assim,Ye_assim_coords,prox_inds=None,verbose=False):
 
    begin_time = time()
    core = cfg.core
    recon_timescale = core.recon_timescale

    if verbose:
        print('finding proxy records for year:' + str(target_year))
        print('recon_timescale = ' + str(recon_timescale))

    tas_var = [item for item in list(cfg.prior.state_variables.keys()) if 'tas_sfc_' in item]

    start_yr = int(target_year-recon_timescale//2)
    end_yr = int(target_year+recon_timescale//2)

    vY = []
    vR = []
    vP = []
    vT = []
    for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
        # Check if we have proxy ob for current time interval
        if recon_timescale > 1:
            # exclude lower bound to not include same obs in adjacent time intervals
            Yvals = Y.values[(Y.values.index > start_yr) & (Y.values.index <= end_yr)]
        else:
            # use all available proxies from config.yml
            if prox_inds is None:
                #Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]
                Yvals = Y.values[(Y.time == target_year)]
                # use only the selected proxies (e.g., randomly filtered post-config)
            else:
                if proxy_idx in prox_inds: 
                    #Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]
                    Yvals = Y.values[(Y.time == target_year)]
                else:
                    Yvals = pd.DataFrame()
                    
        if Yvals.empty: 
            if verbose: print('no obs for this year')
            pass
        else:
            nYobs = len(Yvals)
            Yobs =  Yvals.mean()
            ob_err = Y.psm_obj.R/nYobs
 #           if (target_year >=start_yr) & (target_year <= end_yr):
            vY.append(Yobs)
            vR.append(ob_err)
            vP.append(proxy_idx)
            vT.append(Y.type)
    vYe = Ye_assim[vP,:]
    vYe_coords = Ye_assim_coords[vP,:]
   
    elapsed_time = time() - begin_time
    if verbose:
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    return vY,vR,vP,vYe,vT,vYe_coords


def Kalman_update(vY,vYe,vR,Xb_one,verbose=False):

    if verbose:
        print('solve using tradition Kalman gain...')

    begin_time = time()

    nens = Xb_one.shape[1]
    
    # solve using matrix methods only
    HBHT = np.cov(vYe,ddof=1)
    # yes, this checks previous line; keep as a record
    #Yep = vYe - vYe.mean(axis=1,keepdims=True)
    #HBHTcheck = np.dot(Yep,Yep.T)/(nens-1.)
    #print 'check on HBHT:' + str(np.max(HBHTcheck-HBHT))
    R = np.diag(vR)
    E = np.linalg.inv(HBHT + R)
    # np.cov forces broadcasting, so BH^T must be manual...
    Xbp = Xb_one - Xb_one.mean(axis=1,keepdims=True)
    Yep = vYe - vYe.mean(axis=1,keepdims=True)
    BHT = np.dot(Xbp,Yep.T)/(nens-1.)
    K = np.dot(BHT,E)
    innov = vY - vYe.mean(axis=1,keepdims=False)
    Xinc = np.dot(K,innov)
    xam = Xb_one.mean(axis=1,keepdims=False) + Xinc

    
    elapsed_time = time() - begin_time
    if verbose:
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')
    
    return xam

def Kalman_optimal(Y,vR,Ye,Xb,nsvs=None,transform_only=False,verbose=False):
    """
    Y: observation vector (p x 1)
    vR: observation error variance vector (p x 1)
    Ye: prior-estimated observation vector (p x n)
    Xbp: prior ensemble perturbation matrix (m x n) 

    Originator:

    Greg Hakim
    University of Washington
    26 February 2018

    Modifications:
    11 April 2018: Fixed bug in handling singular value matrix (rectangular, not square)
    """    
    if verbose:
        print('\n all-at-once solve...\n')

    begin_time = time()

    nobs = Ye.shape[0]
    nens = Ye.shape[1]
    ndof = np.min([nobs,nens])
    
    if verbose:
        print('number of obs: '+str(nobs))
        print('number of ensemble members: '+str(nens))
        
    # ensemble prior mean and perturbations
    xbm = Xb.mean(axis=1)
    Xbp = Xb - Xb.mean(axis=1,keepdims=True)

    R = np.diag(vR)
    Risr = np.diag(1./np.sqrt(vR))
    # (suffix key: m=ensemble mean, p=perturbation from ensemble mean; f=full value)
    # keepdims=True needed for broadcasting to work; (p,1) shape rather than (p,)
    Yem = Ye.mean(axis=1,keepdims=True)
    Yep = Ye - Yem
    Htp = np.dot(Risr,Yep)/np.sqrt(nens-1)
    Htm = np.dot(Risr,Yem)
    Yt = np.dot(Risr,Y)
    # numpy svd quirk: V is actually V^T!
    U,s,V = np.linalg.svd(Htp,full_matrices=True)
    if not nsvs:
        nsvs = len(s)
    if verbose:
        print('ndof :'+str(ndof))
        print('U :'+str(U.shape))
        print('s :'+str(s.shape))
        print('V :'+str(V.shape))
        print('recontructing using '+ str(nsvs) + ' singular values')
        
    innov = np.dot(U.T,Yt-np.squeeze(Htm))
    # Kalman gain
    Kpre = s[0:nsvs]/(s[0:nsvs]*s[0:nsvs] + 1)
    K = np.zeros([nens,nobs])
    np.fill_diagonal(K,Kpre)
    # ensemble-mean analysis increment in transformed space 
    xhatinc = np.dot(K,innov)
    # ensemble-mean analysis increment in the transformed ensemble space
    xtinc = np.dot(V.T,xhatinc)/np.sqrt(nens-1)
    if transform_only:
        xam = []
        Xap = []
    else:
        # ensemble-mean analysis increment in the original space
        xinc = np.dot(Xbp,xtinc)
        # ensemble mean analysis in the original space
        xam = xbm + xinc

        # transform the ensemble perturbations
        lam = np.zeros([nobs,nens])
        np.fill_diagonal(lam,s[0:nsvs])
        tmp = np.linalg.inv(np.dot(lam,lam.T) + np.identity(nobs))
        sigsq = np.identity(nens) - np.dot(np.dot(lam.T,tmp),lam)
        sig = np.sqrt(sigsq)
        T = np.dot(V.T,sig)
        Xap = np.dot(Xbp,T)    
        # perturbations must have zero mean
        Xap = Xap - Xap.mean(axis=1,keepdims=True)
    
    elapsed_time = time() - begin_time
    if verbose:
        print('shape of U: ' + str(U.shape))
        print('shape of s: ' + str(s.shape))
        print('shape of V: ' + str(V.shape))
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    readme = '''
    The SVD dictionary contains the SVD matrices U,s,V where V 
    is the transpose of what numpy returns. xtinc is the ensemble-mean
    analysis increment in the intermediate space; *any* state variable 
    can be reconstructed from this matrix.
    '''
    SVD = {'U':U,'s':s,'V':np.transpose(V),'xtinc':xtinc,'readme':readme}
    return xam,Xap,SVD

def Kalman_optimal_sklearn(Y,vR,Ye,Xb,mindim=None,transform_only=False,verbose=False):
    """
    THIS ROUTINE IS DEPRECATED. While it produces the right ensemble mean, it cannot produce the ensemble variance because the sklearn svd routine doesn't return null-space vectors.

    Y: observation vector (p x 1)
    vR: observation error variance vector (p x 1)
    Ye: prior-estimated observation vector (p x n)
    Xb: prior ensemble matrix (m x n) 
    mindim: number of singular values to use
    """    

    from sklearn.utils.extmath import randomized_svd

    if verbose:
        print('\n all-at-once solve...\n')

    begin_time = time()

    nobs = Ye.shape[0]
    nens = Ye.shape[1]

    # ensemble prior mean and perturbations
    xbm = Xb.mean(axis=1)
    Xbp = Xb - Xb.mean(axis=1,keepdims=True)

    R = np.diag(vR)
    Risr = np.diag(1./np.sqrt(vR))
    # ensemble-mean Hx (suffix key: m=ensemble mean, p=perturbation from ensemble mean; f=full value)
    # keepdims = True needed for broadcasting to work; (p,1) shape rather than (p,)
    Yem = Ye.mean(axis=1,keepdims=True)
    Yep = Ye - Yem
    Htp = np.dot(Risr,Yep)/np.sqrt(nens-1)
    Htm = np.dot(Risr,Yem)
    Yt = np.dot(Risr,Y)
    if not mindim:
        mindim = min(nens,nobs)
    U,s,V = randomized_svd(Htp,n_components=mindim)

    innov = np.dot(U.T,Yt-np.squeeze(Htm))
    # Kalman gain
    K = np.diag(s/(s*s + 1))
    # this is the analysis increment in smallest space (obs or nens, depending on which is smaller)
    xhatinc = np.dot(K,innov)
    # this is the analysis increment in the transformed ensemble space
    xtinc = np.dot(V.T,xhatinc)/np.sqrt(nens-1)
    # transform the ensemble perturbations
    lam = np.sqrt(1. - (1./(1. + s**2)))        
    T = np.dot(V.T,np.diag(lam))
    if transform_only:
        xam = []
        Xap = []
    else:
        # this is the ensemble-mean analysis increment in the original space
        xinc = np.dot(Xbp,xtinc)
        # ensemble mean analysis in the original space
        xam = xbm + xinc
        Xap = np.dot(Xbp,T)    
        # perturbations must have zero mean
        Xap = Xap - Xap.mean(axis=1,keepdims=True)

    elapsed_time = time() - begin_time
    if verbose:
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')


    readme = '''
    The SVD dictionary contains the SVD matrices U,s,V where V 
    is the transpose of what numpy returns. xinc is the ensemble-mean
    analysis increment in the intermediate space; *any* state variable 
    can be reconstructed from this matrix. T is the matrix that transforms
    the ensemble from the background to the analysis in the orginal space.
    '''
    SVD = {'U':U,'s':s,'V':np.transpose(V),'xtinc':xtinc,'T':T,'readme':readme}
    return xam,Xap,SVD

def Kalman_ESRF(cfg,vY,vR,vYe,Xb_in,verbose=False):
    import LMR_DA
    
    if verbose:
        print('Ensemble square root filter...')

    begin_time = time()

    # number of state variables
    nx = Xb_in.shape[0]

    # augmented state vector with Ye appended
    Xb = np.append(Xb_in, vYe, axis=0)
    
    #loc_rad = cfg.core.loc_rad
    # need to add code block to compute localization factor
    nobs = len(vY)
    if verbose: print('appended state...')
    for k in range(nobs):
        #if np.mod(k,100)==0: print k
        obvalue = vY[k]
        ob_err = vR[k]
        Ye = Xb[nx+k,:]
        Xa = LMR_DA.enkf_update_array(Xb, obvalue, Ye, ob_err, loc=None, inflate=None)   
        Xb = Xa
      
    # ensemble mean and perturbations
    Xap = Xa[0:nx,:] - Xa[0:nx,:].mean(axis=1,keepdims=True)
    xam = Xa[0:nx,:].mean(axis=1)

    elapsed_time = time() - begin_time
    if verbose:
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    return xam,Xap

def load_analyses(cfg,full_field=False,lmr_gm=None,lmr_time=None,satime=1900,eatime=1999,svtime=1880,evtime=1999):

    """Need to revise to do two things: 1) GMT for a verification interval
    and 2) send back the full data from the analyses. Add a flag and switches"""
    
    # full_field: Flag for sending back full fields instead of global means
    # --- define a reference time period for anomalies (e.g., 20th century)
    # satime: starting year of common reference time period
    # setime: ending year of common reference time 
    # --- define the time period for verification
    # svtime: starting year of the verification time period
    # evtime: ending year of the verification time period

    # check if a global-mean file has been written previously, and if yes, use it
    load = False
    if not full_field:
        try:
            filen = 'analyses'+'_'+str(satime)+'_'+str(eatime)+'_'+str(svtime)+'_'+str(evtime)+'.npz'
            npzfile = np.load(filen)
            print(filen +' exists...loading it')
            load = True
            analyses = npzfile['analyses'] 
            analysis_data = analyses[0]
            analysis_time = analyses[1]
            analysis_lat = {}
            analysis_lon = {}
        except:
            if load: print('analyses.npz exists, but error reading the file!!!')
            load = False

    if not load:

        # ==========================================
        # load GISTEMP, HadCRU, BerkeleyEarth, MLOST
        # ==========================================
        from load_gridded_data import read_gridded_data_GISTEMP
        from load_gridded_data import read_gridded_data_HadCRUT
        from load_gridded_data import read_gridded_data_BerkeleyEarth
        from load_gridded_data import read_gridded_data_MLOST
        import csv

        analysis_data = {}
        analysis_time = {}
        analysis_lat = {}
        analysis_lon = {}

        # location of the datasets from the configuration file
        datadir_calib = cfg.psm.linear.datadir_calib

        # load GISTEMP
        print('loading GISTEMP...')
        datafile_calib   = 'gistemp1200_ERSSTv4.nc'
        calib_vars = ['Tsfc']
        [gtime,GIS_lat,GIS_lon,GIS_anomaly] = read_gridded_data_GISTEMP(datadir_calib,datafile_calib,calib_vars,'annual',[satime,eatime])
        GIS_time = np.array([d.year for d in gtime])
        # fix longitude shift
        nlon_GIS = len(GIS_lon)
        nlat_GIS = len(GIS_lat)
        GIS_lon = np.roll(GIS_lon,shift=nlon_GIS//2,axis=0)
        GIS_anomaly = np.roll(GIS_anomaly,shift=nlon_GIS//2,axis=2)
        analysis_data['GIS']=GIS_anomaly
        analysis_time['GIS']=GIS_time
        analysis_lat['GIS']=GIS_lat
        analysis_lon['GIS']=GIS_lon

        # load HadCRUT
        print('loading HadCRUT...')
        datafile_calib   = 'HadCRUT.4.3.0.0.median.nc'
        calib_vars = ['Tsfc']
        [ctime,CRU_lat,CRU_lon,CRU_anomaly] = read_gridded_data_HadCRUT(datadir_calib,datafile_calib,calib_vars,'annual',[satime,eatime])
        CRU_time = np.array([d.year for d in ctime])
       # fix longitude shift
        nlon_CRU = len(CRU_lon)
        nlat_CRU = len(CRU_lat)
        CRU_lon = np.roll(CRU_lon,shift=nlon_CRU//2,axis=0)
        CRU_anomaly = np.roll(CRU_anomaly,shift=nlon_CRU//2,axis=2)
        analysis_data['CRU']=CRU_anomaly
        analysis_time['CRU']=CRU_time
        analysis_lat['CRU']=CRU_lat
        analysis_lon['CRU']=CRU_lon

        # load BerkeleyEarth
        print('loading BEST...')
        datafile_calib   = 'Land_and_Ocean_LatLong1.nc'
        calib_vars = ['Tsfc']
        [btime,BE_lat,BE_lon,BE_anomaly] = read_gridded_data_BerkeleyEarth(datadir_calib,datafile_calib,calib_vars,'annual',ref_period=[satime,eatime]
)
        BE_time = np.array([d.year for d in btime])
        # fix longitude shift
        nlon_BE = BE_lon.shape[0]
        BE_lon = np.roll(BE_lon,shift=nlon_BE//2,axis=0)
        BE_anomaly = np.roll(BE_anomaly,shift=nlon_BE//2,axis=2)
        analysis_data['BE']=BE_anomaly
        analysis_time['BE']=BE_time
        analysis_lat['BE']=BE_lat
        analysis_lon['BE']=BE_lon

        # load NOAA MLOST
        # Note: Product is anomalies w.r.t. 1961-1990 mean
        print('loading MLOST...')
        #path = datadir_calib + '/NOAA/'
        datafile_calib   = 'MLOST_air.mon.anom_V3.5.4.nc'
        calib_vars = ['Tsfc']
        [mtime,MLOST_lat,MLOST_lon,MLOST_anomaly] = read_gridded_data_MLOST(datadir_calib,datafile_calib,calib_vars,outfreq='annual',ref_period=[satime,eatime])
        MLOST_time = np.array([d.year for d in mtime])
        nlat_MLOST = len(MLOST_lat)
        nlon_MLOST = len(MLOST_lon)
        analysis_data['MLOST']=MLOST_anomaly
        analysis_time['MLOST']=MLOST_time
        analysis_lat['MLOST']=MLOST_lat
        analysis_lon['MLOST']=MLOST_lon

    if full_field:
        print('returning spatial fields...')
        return analysis_data,analysis_time,analysis_lat,analysis_lon
        
    else:
        
        if not load:
            [gis_gm,_,_] = LMR_utils.global_hemispheric_means(GIS_anomaly,GIS_lat)
            [cru_gm,_,_] = LMR_utils.global_hemispheric_means(CRU_anomaly,CRU_lat)
            [be_gm,_,_]  = LMR_utils.global_hemispheric_means(BE_anomaly,BE_lat)
            [mlost_gm,_,_]  = LMR_utils.global_hemispheric_means(MLOST_anomaly,MLOST_lat)

            # set common reference period to define anomalies
            smatch, ematch = LMR_utils.find_date_indices(GIS_time,satime,eatime)
            gis_gm = gis_gm - np.mean(gis_gm[smatch:ematch])
            smatch, ematch = LMR_utils.find_date_indices(CRU_time,satime,eatime)
            cru_gm = cru_gm - np.mean(cru_gm[smatch:ematch])
            smatch, ematch = LMR_utils.find_date_indices(BE_time,satime,eatime)
            be_gm = be_gm - np.mean(be_gm[smatch:ematch])
            smatch, ematch = LMR_utils.find_date_indices(MLOST_time,satime,eatime)
            mlost_gm = mlost_gm - np.mean(mlost_gm[smatch:ematch])

            # now pull out the time window for the verification time period
            gis_smatch, gis_ematch = LMR_utils.find_date_indices(GIS_time,svtime,evtime)
            cru_smatch, cru_ematch = LMR_utils.find_date_indices(CRU_time,svtime,evtime)
            be_smatch, be_ematch = LMR_utils.find_date_indices(BE_time,svtime,evtime)
            mlost_smatch, mlost_ematch = LMR_utils.find_date_indices(MLOST_time,svtime,evtime)
            # "consensus" global mean: average all non-LMR (obs-based) values
            consensus_gmt = np.array([gis_gm[gis_smatch:gis_ematch],cru_gm[cru_smatch:cru_ematch],be_gm[be_smatch:be_ematch],mlost_gm[mlost_smatch:mlost_ematch]])
            con_gm = np.mean(consensus_gmt,axis=0)
            CON_time = np.arange(svtime,evtime)
            CON_time = np.asarray(CON_time)

            analysis_data['GIS']=gis_gm[gis_smatch:gis_ematch]
            analysis_data['CRU']=cru_gm[cru_smatch:cru_ematch]
            analysis_data['BE']=be_gm[be_smatch:be_ematch]
            analysis_data['MLOST']=mlost_gm[mlost_smatch:mlost_ematch]
            analysis_data['CON']=con_gm
            analysis_time['CON']=CON_time
            # for global mean, there is only one common time series and no lat,lon
            analysis_time = {}
            analysis_time['time'] = CON_time
            analysis_lat = {}
            analysis_lon = {}
            # save file for use next time
            analyses = [analysis_data,analysis_time]
            readme='this files contains gmt for analysis products with anomalies relative to a reference time period'
            filen = 'analyses'+'_'+str(satime)+'_'+str(eatime)+'_'+str(svtime)+'_'+str(evtime)+'.npz'
            print('writing to:'+ filen)        
            np.savez(filen,analyses=analyses,readme=readme)

        # LMR GMT was passed to this routine for inclusion in the dictionary
        if np.any(lmr_gm):
            lmr_smatch, lmr_ematch = LMR_utils.find_date_indices(lmr_time,svtime,evtime)
            analysis_data['LMR'] = lmr_gm[lmr_smatch:lmr_ematch]
        
    # lat and lon don't inform on global means, but consistent return with full field
    print('returning global means...')
    return analysis_data,analysis_time,analysis_lat,analysis_lon


def make_obs(ob_lat,ob_lon,dat_lat,dat_lon,dat,verbose=False):

    """
    make observations from a gridded dataset given lat and lon locations
    
    Inputs:
    ob_lat, ob_lon: vector lat,lon coordinates of observations. 
    dat_lat,dat_lon: vector lat,lon coordinates of input data
    dat: array of input data from which observations are drawn. (ntimes,nlat,nlon)

    Output:
    obs: the observations [nobs,nyears]
    """
    
    nyears = dat.shape[0]
    if verbose: print('nyears: '+str(nyears))

    nobs = len(ob_lat)*len(ob_lon)
    if verbose: print('nobs: '+str(nobs))

    # initialize
    obs = np.zeros([nobs,nyears])
    obs_ind_lat = np.zeros(nobs)
    obs_ind_lon = np.zeros(nobs)
  
    k = -1
    # make the obs
    for lon in ob_lon:
        for lat in ob_lat:
            k = k + 1
            dist = LMR_utils.get_distance(lon,lat,dat_lon,dat_lat)
            jind, kind = np.unravel_index(dist.argmin(),dist.shape)
            obs[k,:] = dat[:,jind,kind]
            obs_ind_lat[k] = jind
            obs_ind_lon[k] = kind

            #print(lat,jind,kind,ob[100,k])
            
    return obs,obs_ind_lat,obs_ind_lon

# started from: stackoverflow.com/201618804
def smooth(y,box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y,box,mode='same')
    # remove endpoints with artifacts
    ii = np.int(box_pts/2.)
    y_smooth[0:ii] = np.nan
    y_smooth[-ii:] = np.nan

    return y_smooth

def smooth_121(y):
    # 1-2-1 smoother by convolution. leaves last 3 points biased
    box = np.array([1.,2.,1.])/4.
    y_smooth = np.convolve(y,box,mode='same')
    y_smooth[-2:-1] = np.nan
    # remove endpoints with artifacts
    y_smooth[0:1] = np.nan
    y_smooth[-1:] = np.nan

    return y_smooth
