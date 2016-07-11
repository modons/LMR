"""
Module: LMR_proxy.py

Purpose: Contains class definitions for the various proxy types assimilated by the LMR. 
         *** THIS CODE IS OBSOLETE *** 
         Has since replaced by LMR_proxy_pandas_rework.py

Originators: Greg Hakim & Robert Tardif | University of Washington
                                        | January 2015

Revisions: 

"""

# -------------------------------------------------------------------------------
# *** Proxy type assignment  ----------------------------------------------------
# -------------------------------------------------------------------------------
# All logic for proxy object assignment

def proxy_assignment(iproxy):
    if iproxy == 'Tree ring_Width':
        proxy_object = proxy_tree_ring_width()
    elif iproxy == 'Tree ring_Density':
        proxy_object = proxy_tree_ring_density()
    elif iproxy == 'Coral_d18O':
        proxy_object = proxy_coral_d18O()
    elif iproxy == 'Coral_Luminescence':
        proxy_object = proxy_coral_luminescence()
    elif iproxy == 'Ice core_d18O':
        proxy_object = proxy_ice_core_d18O()
    elif iproxy == 'Ice core_d2H':
        proxy_object = proxy_ice_core_d2H()
    elif iproxy == 'Ice core_Accumulation':
        proxy_object = proxy_ice_core_accumulation()
    elif iproxy == 'Lake sediment_All':
        proxy_object = proxy_lake_sediment_all()
    elif iproxy == 'Marine sediment_All':
        proxy_object = proxy_marine_sediment_all()
    elif iproxy == 'Speleothem_All':
        proxy_object = proxy_speleothem_all()
    elif iproxy == 'pseudo_proxy_temperature':
        proxy_object = pseudo_proxy_temperature()

    return proxy_object

# -------------------------------------------------------------------------------
# *** Master class for proxies --------------------------------------------------
# -------------------------------------------------------------------------------
class proxy_master(object):
    '''
    This is the master proxy class. Turn this into a metaclass so one cannot instantiate directly; 
    it is an abstract class.
    '''

    proxy_type         = 'master---do not instantiate here'

    # Define the basic proxy system model (PSM)
    def psm(self,C,X,state_info,X_coords):

        import os.path
        import numpy as np
        from scipy import stats
        import LMR_utils
        from random import sample

        try:
          self.calibrate
        except:
          calibrate_check = False
        else:
          calibrate_check = True
       
        if not calibrate_check:
            self.calibrate = True
            # do calibration here:            
            print 'Calibrating the PSM...'

            # ------------------------
            diag_output       = True
            diag_output_figs  = False
            calib_spatial_avg = False ; Npts = 9 # nb of neighboring pts used in smoothing
            # ------------------------

            # --------------------------------------------
            # Use linear model (regression) as default PSM
            # --------------------------------------------

            # Form set of collocated calibration/proxy data in time & space
            reg_x = [] # calib.
            reg_y = [] # proxy

            """ GH code block for random calibration year. not yet implemented in this new code
            GH: new: random sample of calibration period. ncalib is the sample size (max = len(self.time))
            ncalib = 75
            rantimes = sample(range(0,len(self.time)),ncalib)
            for i in rantimes:
            """

            # Look for indices of calibration grid point closest in space (in 2D) to proxy site
            dist = np.empty([C.lat.shape[0], C.lon.shape[0]])
            tmp  = np.array([LMR_utils.haversine(self.lon, self.lat, C.lon[k], C.lat[j]) for j in xrange(len(C.lat)) for k in xrange(len(C.lon))])
            dist = np.reshape(tmp,[len(C.lat),len(C.lon)])
            # indices of nearest grid pt.
            jind,kind = np.unravel_index(dist.argmin(), dist.shape) 
            
            # overlapping years
            # first, flip proxy time series if in reverse order (some chronologies may be...)
            if self.time[0] > self.time[-1]:
                self.time.reverse()
                self.value.reverse()
            sc = set(C.time); sp = set(self.time)
            common_time = list(sc.intersection(sp))
            # respective indices in calib and proxy times
            ind_c = [j for j, k in enumerate(C.time) if k in common_time]
            ind_p = [j for j, k in enumerate(self.time) if k in common_time]
            # make sure proxy time sequence matches calibration time series
            if [ C.time[m] for m in ind_c ] != [ self.time[n] for n in ind_p ]:
                ind_p = []
                for n in ind_c:
                    ind_p.append([j for j, k in enumerate(self.time) if k in common_time and k == C.time[n]][0])

            if calib_spatial_avg:                 
                C2Dsmooth = np.zeros([C.time.shape[0],C.lat.shape[0],C.lon.shape[0]])
                for m in ind_c:
                    C2Dsmooth[m,:,:] = LMR_utils.smooth2D(C.temp_anomaly[m, :, :], n=Npts)
                reg_x = [ C2Dsmooth[m,jind,kind] for m in ind_c ]
            else:
                reg_x = [ C.temp_anomaly[m,jind,kind] for m in ind_c ]

            reg_y = [ self.value[m] for m in ind_p ]
            # check for valid values
            ind_c_ok = [j for j, k in enumerate(reg_x) if np.isfinite(k)]
            ind_p_ok = [j for j, k in enumerate(reg_y) if np.isfinite(k)]
            common_ok = list(set(ind_c_ok).intersection(set(ind_p_ok)))
            # fill numpy arrays with values used in regression
            reg_xa = np.array([reg_x[k] for k in common_ok]) # calib. values
            reg_ya = np.array([reg_y[k] for k in common_ok]) # proxy values

            # -------------------------
            # Perform linear regression
            # -------------------------
            nobs = reg_ya.shape[0]
            if nobs < 10: # skip rest if insufficient overlapping data 
                return

            # START NEW (GH) 21 June 2015
            # detrend both the proxy and the calibration data
            #
            # save copies of the original data for residual estimates later
            reg_xa_all = np.copy(reg_xa)
            reg_ya_all = np.copy(reg_ya)
            # proxy detrend: (1) linear regression, (2) fit, (3) detrend
            xvar = range(len(reg_ya))
            proxy_slope, proxy_intercept, r_value, p_value, std_err = stats.linregress(xvar,reg_ya)
            proxy_fit = proxy_slope*np.squeeze(xvar) + proxy_intercept
            #reg_ya = reg_ya - proxy_fit # expt 4: no detrend for proxy when commented out
            # calibration detrend: (1) linear regression, (2) fit, (3) detrend
            xvar = range(len(reg_xa))
            calib_slope, calib_intercept, r_value, p_value, std_err = stats.linregress(xvar,reg_xa)
            calib_fit = calib_slope*np.squeeze(xvar) + calib_intercept
            #reg_xa = reg_xa - calib_fit # expt 4: no detrend for calib when commented out
            # END NEW (GH) 21 June 2015

            print 'Calib stats (x)              [min, max, mean, std]:', np.nanmin(reg_xa), np.nanmax(reg_xa), np.nanmean(reg_xa), np.nanstd(reg_xa)
            print 'Proxy stats (y:original)     [min, max, mean, std]:', np.nanmin(reg_ya), np.nanmax(reg_ya), np.nanmean(reg_ya), np.nanstd(reg_ya)
            # standardize proxy values over period of overlap with calibration data
            #reg_ya = (reg_ya - np.nanmean(reg_ya))/np.nanstd(reg_ya)
            #print 'Proxy stats (y:standardized) [min, max, mean, std]:', np.nanmin(reg_ya), np.nanmax(reg_ya), np.nanmean(reg_ya), np.nanstd(reg_ya)
            # GH: note that std_err pertains to the slope, not the residuals!!!

            self.slope, self.intercept, r_value, p_value, std_err = stats.linregress(reg_xa,reg_ya)

            # Calculate stats on regression residuals
            # GH: residuals have to be computed "by hand"
            # this is the original approach, which when detrending misses error unless the original x and y are used
            #fit = self.slope*np.squeeze(reg_xa) + self.intercept
            #residuals = fit - reg_ya
            # this is the proper way to do it, including detrending
            fit = self.slope*np.squeeze(reg_xa_all) + self.intercept
            residuals = fit - reg_ya_all
            MSE = np.mean((residuals)**2)
            self.R = MSE
            self.corr = r_value

            if diag_output:
                # Some other diagnostics
                varproxy = np.var(reg_ya); varresiduals = np.var(residuals)
                varratio = np.divide(varproxy,varresiduals)
                snr = np.absolute(r_value)/(np.sqrt(1.0-r_value**2))            
                # Diagnostic output 
                print "***PSM stats:"
                print "Nobs                 :", nobs
                print "slope                :", self.slope
                print "intercept            :", self.intercept
                print "correl.              :", r_value
                print "r-squared            :", r_value**2
                print "p_value              :", p_value
                print "std_err              :", std_err
                print "resid MSE            :", MSE
                print "var(proxy)/var(resid):", varratio
                print "SNR                  :", snr

                if diag_output_figs:
                    # Figure (scatter plot w/ summary statistics)
                    import pylab
                    line = self.slope*reg_xa+self.intercept
                    pylab.plot(reg_xa,line,'r-',reg_xa,reg_ya,'o',markersize=7,markerfacecolor='#5CB8E6',markeredgecolor='black',markeredgewidth=1)
# GH: I don't know how this ran in the first place; must exploit some global namespace                    
                    pylab.title('%s: %s' % (self.proxy_type, self.id))
                    pylab.xlabel('Calibration data')
                    pylab.ylabel('Proxy data')
                    xmin,xmax,ymin,ymax = pylab.axis()
                    # Annotate with summary stats
                    ypos = ymax-0.05*(ymax-ymin)
                    xpos = xmin+0.025*(xmax-xmin)
                    pylab.text(xpos,ypos,'Nobs = %s' %str(nobs),fontsize=12,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    pylab.text(xpos,ypos,'Slope = %s' %"{:.4f}".format(self.slope),fontsize=12,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    pylab.text(xpos,ypos,'Intcpt = %s' %"{:.4f}".format(self.intercept),fontsize=12,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    pylab.text(xpos,ypos,'Corr = %s' %"{:.4f}".format(r_value),fontsize=12,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    pylab.text(xpos,ypos,'p_value = %s' %"{:.4f}".format(p_value),fontsize=12,fontweight='bold')
                    ypos = ypos-0.05*(ymax-ymin)
                    pylab.text(xpos,ypos,'StdErr = %s' %"{:.4f}".format(std_err),fontsize=12,fontweight='bold') 
                    ypos = ypos-0.05*(ymax-ymin)
                    pylab.text(xpos,ypos,'Res.MSE = %s' %"{:.4f}".format(MSE),fontsize=12,fontweight='bold') 
                    pylab.savefig('proxy_%s_%s_LinearModel_calib.png' % (self.proxy_type.replace(" ", "_"),self.id),bbox_inches='tight')
                    pylab.close()

        if X is not None:
            # ----------------------
            # Calculate the Ye's ...
            # ----------------------

            # Looking for position of 'tas_sfc_Amon' variable in state vector 
            # (now only considering forward models calibrated against surface air temperature)
            # Check it is there! 
            if 'tas_sfc_Amon' not in state_info.keys():
                print "Needed variable not in state vector. Cannot calculate the Ye's. Exiting!"
                exit(1)

            # positions in state vector
            tas_indbegin = state_info['tas_sfc_Amon']['pos'][0]
            tas_indend   = state_info['tas_sfc_Amon']['pos'][1]
            # lat/lon column indices in X_coords 
            ind_lon = state_info['tas_sfc_Amon']['spacecoords'].index('lon')
            ind_lat = state_info['tas_sfc_Amon']['spacecoords'].index('lat')

            # Find row index of X for which [X_lat,X_lon] corresponds to closest grid point to 
            # location of proxy site [self.lat,self.lon]
            # Calclulate distances from proxy site.
            #stateDim = X.shape[0]
            varDim = (tas_indend+1) - tas_indbegin
            ensDim = X.shape[1]
            dist = np.empty(varDim)
            #rt dist = np.array([ LMR_utils.haversine(self.lon,self.lat,X_lon[k],X_lat[k]) for k in range(tas_indbegin,tas_indend+1) ])
            dist = np.array([LMR_utils.haversine(self.lon, self.lat, X_coords[k, ind_lon], X_coords[k, ind_lat]) for k in range(tas_indbegin, tas_indend + 1)])

            # row index in dist array corresponding to nearest grid pt. in prior (minimum distance)
            kind = np.unravel_index(dist.argmin(), dist.shape) 
            # corresponding index in state vector
            kind_state = tas_indbegin + kind[0]

            Ye = np.zeros(shape=ensDim)
            Ye = self.slope*np.squeeze(X[kind_state,:]) + self.intercept

        else:
            Ye = None

        return Ye


    # Define the error model for this proxy
    def error(self):
        r = 0.1
        return r

# -------------------------------------------------------------------------------
# TREE RING WIDTH: 
# -------------------------------------------------------------------------------
class proxy_tree_ring_width(proxy_master):

    proxy_type         = 'Tree ring_Width'

    def read_proxy(self,proxy_site):

        import numpy as np
        from load_proxy_data import read_proxy_data_S1csv_site

        [self.id,self.lat,self.lon,self.alt,self.time,self.value] = read_proxy_data_S1csv_site(self.proxy_datadir,self.proxy_datafile,proxy_site)

        self.nobs = len(self.value)
        print self.id, ': Number of proxy obs. uploaded:', self.nobs

        return

# OTHER TREE RING WIDTH: Suppose we get a new tree ring record from jane doe; 
# we can create a new class that inherits from an existing tree class
# we'll probably override the file I/O and maybe some other things too
#class proxy_jane_doe_tree_ring_width(proxy_tree_ring_width):
#    proxy_type = 'jane_doe_tree_ring_width'
 

# -------------------------------------------------------------------------------
# TREE RING WOOD DENSITY: 
# -------------------------------------------------------------------------------
class proxy_tree_ring_density(proxy_master):

    proxy_type         = 'Tree ring_Density'

    def read_proxy(self,proxy_site):

        import numpy as np
        from load_proxy_data import read_proxy_data_S1csv_site

        [self.id,self.lat,self.lon,self.alt,self.time,self.value] = read_proxy_data_S1csv_site(self.proxy_datadir,self.proxy_datafile,proxy_site)

        self.nobs = len(self.value)
        print self.id, ': Number of proxy obs. uploaded:', self.nobs

        return

# -------------------------------------------------------------------------------
# CORAL DELTA O18: 
# -------------------------------------------------------------------------------
class proxy_coral_d18O(proxy_master):
    
    from load_proxy_data import read_proxy_data_S1csv_site

    proxy_type         = 'Coral_d18O'

    def read_proxy(self,proxy_site):

        import numpy as np
        from load_proxy_data import read_proxy_data_S1csv_site

        [self.id,self.lat,self.lon,self.alt,self.time,self.value] = read_proxy_data_S1csv_site(self.proxy_datadir,self.proxy_datafile,proxy_site)

        self.nobs = len(self.value)
        print self.id, ': Number of proxy obs. uploaded:', self.nobs

        return

# -------------------------------------------------------------------------------
# CORAL LUMINESCENCE: 
# -------------------------------------------------------------------------------
class proxy_coral_luminescence(proxy_master):
    
    from load_proxy_data import read_proxy_data_S1csv_site

    proxy_type         = 'Coral_Luminescence'

    def read_proxy(self,proxy_site):

        import numpy as np
        from load_proxy_data import read_proxy_data_S1csv_site

        [self.id,self.lat,self.lon,self.alt,self.time,self.value] = read_proxy_data_S1csv_site(self.proxy_datadir,self.proxy_datafile,proxy_site)

        self.nobs = len(self.value)
        print self.id, ': Number of proxy obs. uploaded:', self.nobs

        return

# -------------------------------------------------------------------------------
# ICE CORE DELTA O18: 
# -------------------------------------------------------------------------------
class proxy_ice_core_d18O(proxy_master):
    
    from load_proxy_data import read_proxy_data_S1csv_site

    proxy_type         = 'Ice core_d18O'

    def read_proxy(self,proxy_site):

        import numpy as np
        from load_proxy_data import read_proxy_data_S1csv_site

        [self.id,self.lat,self.lon,self.alt,self.time,self.value] = read_proxy_data_S1csv_site(self.proxy_datadir,self.proxy_datafile,proxy_site)

        self.nobs = len(self.value)
        print self.id, ': Number of proxy obs. uploaded:', self.nobs

        return

# -------------------------------------------------------------------------------
# ICE CORE DELTA H2: 
# -------------------------------------------------------------------------------
class proxy_ice_core_d2H(proxy_master):
    
    from load_proxy_data import read_proxy_data_S1csv_site

    proxy_type         = 'Ice core_d2H'

    def read_proxy(self,proxy_site):

        import numpy as np
        from load_proxy_data import read_proxy_data_S1csv_site

        [self.id,self.lat,self.lon,self.alt,self.time,self.value] = read_proxy_data_S1csv_site(self.proxy_datadir,self.proxy_datafile,proxy_site)

        self.nobs = len(self.value)
        print self.id, ': Number of proxy obs. uploaded:', self.nobs

        return

# -------------------------------------------------------------------------------
# ICE CORE ACCUMULATION: 
# -------------------------------------------------------------------------------
class proxy_ice_core_accumulation(proxy_master):
    
    from load_proxy_data import read_proxy_data_S1csv_site

    proxy_type         = 'Ice core_Accumulation'

    def read_proxy(self,proxy_site):

        import numpy as np
        from load_proxy_data import read_proxy_data_S1csv_site

        [self.id,self.lat,self.lon,self.alt,self.time,self.value] = read_proxy_data_S1csv_site(self.proxy_datadir,self.proxy_datafile,proxy_site)

        self.nobs = len(self.value)
        print self.id, ': Number of proxy obs. uploaded:', self.nobs

        return

# -------------------------------------------------------------------------------
# LAKE SEDIMENT ALL annual types: 
# -------------------------------------------------------------------------------
class proxy_lake_sediment_all(proxy_master):
    
    from load_proxy_data import read_proxy_data_S1csv_site

    proxy_type         = 'Lake sediment_All'

    def read_proxy(self,proxy_site):

        import numpy as np
        from load_proxy_data import read_proxy_data_S1csv_site

        [self.id,self.lat,self.lon,self.alt,self.time,self.value] = read_proxy_data_S1csv_site(self.proxy_datadir,self.proxy_datafile,proxy_site)

        self.nobs = len(self.value)
        print self.id, ': Number of proxy obs. uploaded:', self.nobs

        return

# -------------------------------------------------------------------------------
# MARINE SEDIMENT ALL annual types: 
# -------------------------------------------------------------------------------
class proxy_marine_sediment_all(proxy_master):
    
    from load_proxy_data import read_proxy_data_S1csv_site

    proxy_type         = 'Marine sediment_All'

    def read_proxy(self,proxy_site):

        import numpy as np
        from load_proxy_data import read_proxy_data_S1csv_site

        [self.id,self.lat,self.lon,self.alt,self.time,self.value] = read_proxy_data_S1csv_site(self.proxy_datadir,self.proxy_datafile,proxy_site)

        self.nobs = len(self.value)
        print self.id, ': Number of proxy obs. uploaded:', self.nobs

        return

# -------------------------------------------------------------------------------
# SPELEOTHEM ALL annual types: 
# -------------------------------------------------------------------------------
class proxy_speleothem_all(proxy_master):
    
    from load_proxy_data import read_proxy_data_S1csv_site

    proxy_type         = 'Speleothem_All'

    def read_proxy(self,proxy_site):

        import numpy as np
        from load_proxy_data import read_proxy_data_S1csv_site

        [self.id,self.lat,self.lon,self.alt,self.time,self.value] = read_proxy_data_S1csv_site(self.proxy_datadir,self.proxy_datafile,proxy_site)

        self.nobs = len(self.value)
        print self.id, ': Number of proxy obs. uploaded:', self.nobs

        return


# -------------------------------------------------------------------------------
# PSEUDO-PROXY TEMPERATURE
# -------------------------------------------------------------------------------
# hard wired for CCSM4 last millenium, but can generalize if needed
class pseudo_proxy_temperature(proxy_master):

    proxy_type         = 'pseduo_proxy_temperature'

    # variable proxy_site is empty and not used, but here for compatability
    def read_proxy(self,proxy_site):
        from load_gridded_data import read_gridded_data_ccsm4_last_millenium

        # do we need self.alt? (see S1 read)
        [self.time,self.lat,self.lon,self.value] = read_gridded_data_ccsm4_last_millenium(self.proxy_datadir,self.proxy_datafile,self.statevars)

        return
