"""

 This script gives a method for pre-calculating ye values for our linear psm.
 It outputs this file in a subdirectory of the prior source directory which
 is then in turn checked for by the driver.  Files created will be unique to
 the prior, psm calibration source, proxy source, and state variable used to
 calculate it. I'm choosing to forego on the fly creation of these files for
 now.  If a user would like to create one for their current configuration  this
 script should just be an easy one-off run with no editing required.
 [ A. Perkins, U. of Washington ]

 Modified:
 - August 2016: Now calculates ye values consistent with all possible psm types
                (linear, linear_TorP, bilinear, h_interp) found in the
                LMR_config.py file as specified by the user. For a given psm type,
                ye values are calculatd for *all* proxy records in the database,
                for which a psm has been pre-calibrated (for statistical PSMs)
                [ R. Tardif, U. of Washington ]
 -  Sept. 2016: Now performs Ye calculation on the basis of either annually or
                seasonally calibrated PSMs.
                [ R. Tardif, U. of Washington ]
 -  Sept. 2016: Code now ensures that uploaded prior data is of the appropriate
                "kind" for the proxy-type dependent PSM to be used as specified
                in LMR_config. Statistical PSMs (linear, linear_TorP and bilinear)
                are typically calibrated on the basis of anomalies (temporal mean
                over reference period removed). Assimilation of isotope data using
                the h_interp forward operator should be performed on "full fields"
                if the proxy_timeseries_kind='asis' in class "proxies" in LMR_config,
                or as anomalies if proxy_timeseries_kind='anom'. The code here will
                override the setting of the prior "state_kind" in LMR_config.
                [ R. Tardif, U. of Washington ]
 -   Feb. 2017: Added functionalities allowing the calculation of proxy estimates
                for proxies with lower temporal resolution (lower than
                seasonal/annual) + activation of use of the "bayesreg_uk37" forward
                model used for marine sediment alkenone proxies.
                [ R. Tardif, U. of Washington ]
 -  March 2017: Introduced use of config.yml simplified configuration file.
                [ A. Perkins, U. of Washington ]
 -  Sept. 2017: Renamed the proxy databases to less-confusing convention. 
                'pages' renamed as 'PAGES2kv1' and 'NCDC' renamed as 'LMRdb'
                [ R. Tardif, U. of Washington ]
"""

import sys
import logging
import os
import timeit
import numpy as np
import yaml

sys.path.append('../')

from copy import deepcopy
import LMR_prior
import LMR_psms
import LMR_proxy
import LMR_config
from LMR_utils import create_precalc_ye_filename

# -------------------- logging parameters --------------------

logLevel = logging.DEBUG  # DEBUG, INFO, WARNING, ERROR
logToFile = False  # write log info into ./build_ye.log file.
logToScreen = True # write log info to screen

logFormat = '%(levelname)-5.5s : %(message)s'

# ------------------------------------------------------------

_log = logging.getLogger(__name__)

_log.setLevel(logLevel)
formatter=logging.Formatter(logFormat)
if logToFile:
    fh = logging.FileHandler('build_ye.log')
    fh.setFormatter(formatter)
    _log.addHandler(fh)
if logToScreen:
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    _log.addHandler(ch)


def main(cfgin=None, config_path=None):
    _log.info('Starting Ye calculations')
    if cfgin is not None:
        cfg = deepcopy(cfgin)  # I'm not sure that we're modifying cfg below.
    else:
        if config_path is not None:
            yaml_file = config_path
        else:
            yaml_file = os.path.join(LMR_config.SRC_DIR, 'config.yml')

        _log.debug('Loading configuration: ' + yaml_file)
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
        cfg = LMR_config.Config()

        
    masterstarttime = timeit.default_timer()

    _log.info('Starting Ye precalculation using prior data from '
           '{}'.format(cfg.prior.prior_source))
    #  Load the proxy information
    cfg.psm.linear.psm_r_crit = 0.0

    # from LMR_config, figure out all the psm types the user wants to use
    proxy_database = cfg.proxies.use_from[0]
    proxy_class = LMR_proxy.get_proxy_class(proxy_database)

    if proxy_database == 'PAGES2kv1':
        proxy_cfg = cfg.proxies.PAGES2kv1
    elif proxy_database == 'LMRdb':
        proxy_cfg = cfg.proxies.LMRdb
    elif proxy_database == 'NCDCdadt':
        proxy_cfg = cfg.proxies.NCDCdadt
    elif proxy_database == 'DAPSpseudoproxies':
        proxy_cfg = cfg.proxies.DAPSpseudoproxies
    else:
        raise KeyError('ERROR in specification of proxy database: {}'.format(proxy_database))

    proxy_types = proxy_cfg.proxy_order
    # proxy_types = list(proxy_cfg.proxy_psm_type.keys())
    psm_keys = [proxy_cfg.proxy_psm_type[p] for p in proxy_types]
    unique_psm_keys = list(set(psm_keys))
    
    # load all available proxy objects in selected proxy db
    proxy_objects = proxy_class.load_all_annual_no_filtering(cfg)
    # Number of proxy objects (will be a dim of ye_out array)
    num_proxy = len(proxy_objects)
    
    # identify the set of psms to consider
    psm_keys_proxydb_list = []
    for pobj in proxy_objects: psm_keys_proxydb_list.append(pobj.psm_obj.psm_key)
    psm_keys_proxydb = list(set(psm_keys_proxydb_list))
    # Keep only psms associated to proxies actually found in the db 
    unique_psm_keys = [item for item in unique_psm_keys if item in psm_keys_proxydb]
    
    # A quick check of availability of calibrated PSMs
    if 'linear_TorP' in unique_psm_keys:
        psm_ok = True
        # check existence of required pre-calibrated PSM files
        if not os.path.exists(cfg.psm.linear_TorP.pre_calib_datafile_T):
            _log.info(('*** linear_TorP PSM: Cannot find file of pre-calibrated PSMs for temperature:'
                       '%s' %cfg.psm.linear_TorP.pre_calib_datafile_T))
            psm_ok = False
        if not os.path.exists(cfg.psm.linear_TorP.pre_calib_datafile_P):
            _log.info(('*** linear_TorP PSM: Cannot find file of pre-calibrated PSMs for moisture:'
                   '%s' %cfg.psm.linear_TorP.pre_calib_datafile_P))
            psm_ok = False
        if not psm_ok:
            raise SystemExit

    if 'linear' in unique_psm_keys:
        if not os.path.exists(cfg.psm.linear.pre_calib_datafile):
            print('*** linear PSM: Cannot find file of pre-calibrated PSMs: %s' % cfg.psm.linear.pre_calib_datafile)
            print('Perform calibration "on-the-fly" and calculate Ye values? '
                  'This will take longer and PSM calibration parameters will not be stored in a file...')
            userinput = input('Continue (y/n)? ')
            if userinput == 'y' or userinput == 'yes':
                print('ok...continuing...')
            else:
                raise SystemExit

    if 'bilinear' in unique_psm_keys:
        if not os.path.exists(cfg.psm.bilinear.pre_calib_datafile):
            print(('*** bilinear PSM: Cannot find file of pre-calibrated PSMs: %s' %cfg.psm.bilinear.pre_calib_datafile))
            print('Perform calibration "on-the-fly" and calculate Ye values? '
                  'This will take longer and PSM calibration parameters will not be stored in a file...')
            userinput = input('Continue (y/n)? ')
            if userinput == 'y' or userinput == 'yes':
                print('ok...continuing...')
            else:
                raise SystemExit
    # Finished checking ...
    
    
    # Loop over all psm types found in the configuration
    for psm_key in sorted(unique_psm_keys):
        _log.info('\nLoading psm information for psm type:' + str(psm_key) + ' ...')

        # re-assign current psm type to all proxy records
        #  TODO: Could think of implementing filter to restrict to relevant proxy records only
        #  RT 11/18: Only do this if NOT the DADT baysreg PSM. Stricter association of PSM type and proxy type
        #  and proxy type for these, compared to the regression-based PSMs used for Last Millennium recons.
        if 'bayesreg' not in psm_key:
            for p in proxy_types: proxy_cfg.proxy_psm_type[p] = psm_key


        proxy_objects = proxy_class.load_all_annual_no_filtering(cfg)
        # Number of proxy objects (will be a dim of ye_out array)
        num_proxy = len(proxy_objects)

        
        _log.info('Calculating ye values for {:d} proxies'.format(num_proxy))

        # Define the psm-dependent required state variables
        if psm_key == 'linear':
            statevars = cfg.psm.linear.psm_required_variables
            psm_avg = cfg.psm.avgPeriod
        elif psm_key == 'linear_TorP':
            statevars = cfg.psm.linear_TorP.psm_required_variables
            psm_avg = cfg.psm.avgPeriod
        elif psm_key == 'bilinear':
            statevars = cfg.psm.bilinear.psm_required_variables
            psm_avg = cfg.psm.avgPeriod
#        elif psm_key == 'h_interp':
        elif 'h_interp' in psm_key:
            # h_interp psm class (interpolation of prior data)
            if list(cfg.prior.avgInterval)[0] == 'multiyear':
                psm_avg = 'multiyear'
            else:
                psm_avg = 'annual'

            # Define the psm-dependent required state variables
            #  check compatibility of options between prior and proxies for this psm class
            #  - proxies as 'anom' vs. 'asis' VS. prior as 'anom' vs. 'full'
            #  - possibly override definition in config.
            if proxy_cfg.proxy_timeseries_kind == 'anom':
                vkind = 'anom'
            elif proxy_cfg.proxy_timeseries_kind == 'asis':
                vkind = 'full'
            else:
                _log.error('ERROR: Unrecognized value of *proxy_timeseries_kind* attribute in proxies configuration.')
                raise SystemExit()
            statevars = {psm_key.split(',')[1]: vkind}
        elif psm_key == 'bayesreg_uk37':
            statevars = cfg.psm.bayesreg_uk37.psm_required_variables
            psm_avg = 'multiyear-season'
        elif psm_key == 'bayesreg_tex86':
            statevars = cfg.psm.bayesreg_tex86.psm_required_variables
            psm_avg = 'multiyear'
        elif psm_key in ['bayesreg_d18o_pachyderma', 'bayesreg_d18o_bulloides',
                         'bayesreg_d18o_sacculifer', 'bayesreg_d18o_ruberwhite',
                         'bayesreg_d18o_incompta', 'bayesreg_d18o_pooled']:
            statevars = cfg.psm.bayesreg_d18o.psm_required_variables
            psm_avg = 'multiyear'
            if cfg.psm.bayesreg_d18o.seasonal_seatemp:
                psm_avg = 'multiyear-season'
        elif psm_key in ['bayesreg_mgca_pachyderma_red', 'bayesreg_mgca_pachyderma_bcp',
                         'bayesreg_mgca_bulloides_red', 'bayesreg_mgca_bulloides_bcp',
                         'bayesreg_mgca_sacculifer_red', 'bayesreg_mgca_sacculifer_bcp',
                         'bayesreg_mgca_ruberwhite_red', 'bayesreg_mgca_ruberwhite_bcp',
                         'bayesreg_mgca_pooled_red', 'bayesreg_mgca_pooled_bcp']:
            statevars = cfg.psm.bayesreg_mgca.psm_required_variables
            psm_avg = 'multiyear'
            if cfg.psm.bayesreg_mgca.seasonal_seatemp:
                psm_avg = 'multiyear-season'
        else:
            _log.error('ERROR: Did not find a match to a valid psm key: {}. Exiting.'.format(psm_key))
            raise SystemExit()

        
        # Define required temporal averaging
        if psm_avg == 'annual':
            # calendar year as the only seasonality vector for all proxies
            annual = [1,2,3,4,5,6,7,8,9,10,11,12]

            # assign annual seasonality attribute to all proxy objects
            # (override metadata of the proxy record)
            if psm_key == 'bilinear':
                # seasonality in form of tuple for this psm
                season_unique = [(annual,annual)]
                for pobj in proxy_objects: pobj.seasonality = (annual,annual)
            else:
                # seasonality in form of list
                season_unique = [annual]
                for pobj in proxy_objects: pobj.seasonality = annual

            base_time_interval = 'annual'

        elif psm_avg == 'season':
            # map out all possible seasonality vectors that will have to be considered
            season_vects = []
            # Which seasonality to use? from proxy metadata or derived from psm calibration?
            # Attribute exists?
            if hasattr(cfg.psm,'season_source'):
                # which option is activated?
                if cfg.psm.season_source == 'psm_calib':
                    for pobj in proxy_objects: season_vects.append(pobj.psm_obj.seasonality)
                elif cfg.psm.season_source == 'proxy_metadata':
                    # metadata seasonality is an attribute of the proxy object itself
                    for pobj in proxy_objects: season_vects.append(pobj.seasonality)
                else:
                    raise KeyError('ERROR: Unrecognized value of *season_source* attribute in psm configuration: {}'.format(cfg.psm.season_source))
            else:
                # attribute does not exist in config., revert to proxy metadata
                for pobj in proxy_objects: season_vects.append(pobj.seasonality)

            season_unique = []
            for item in season_vects:
                if item not in season_unique:season_unique.append(item)

            base_time_interval = 'annual'

        elif psm_avg == 'multiyear-season':
            base_time_interval = 'multiyear'

            # map out all possible seasonality vectors that will have to be considered
            season_vects = []
            for pobj in proxy_objects:
                if pobj.psm_obj.psm_key == psm_key:
                    season_vects.append(pobj.seasonality)
            season_unique = []
            for item in season_vects:
                if item not in season_unique:season_unique.append(item)

        elif psm_avg == 'multiyear':
            season_unique = [cfg.prior.avgInterval['multiyear']]
            base_time_interval = 'multiyear'
            
        else:
            _log.error('ERROR in specification of averaging period: {}'.format(psm_avg))
            raise SystemExit()

        
        # Loop over seasonality definitions found in the proxy set
        firstloop = True
        for season in season_unique:

            _log.info('Calculating estimates for proxies with seasonality:' + str(season))

            # Create prior source object
            X = LMR_prior.prior_assignment(cfg.prior.prior_source)
            X.prior_datadir = cfg.prior.datadir_prior
            X.prior_datafile = cfg.prior.datafile_prior
            X.detrend = cfg.prior.detrend
            X.avgInterval = cfg.prior.avgInterval
            X.Nens = None  # None => Load entire prior
            X.statevars = statevars
            X.statevars_info = cfg.prior.state_variables_info
            X.anom_reference = cfg.core.anom_reference_period

            # Load the prior data, averaged over interval corresponding
            # to current "season" (i.e. proxy seasonality)
            # RT Nov 2018: additional (crude) logic to handle special case of DADT "seasonal" PSMs
            if psm_avg == 'multiyear-season':
                X.avgInterval = {'multiyear': cfg.prior.avgInterval['multiyear'], 'season': season}
            else:
                X.avgInterval = {base_time_interval: season}

            # populate the prior state vector
            X.populate_ensemble(cfg.prior.prior_source, cfg.prior)

            statedim = X.ens.shape[0]
            ntottime = X.ens.shape[1]


            # Calculate the Ye values
            # -----------------------
            if firstloop:
                # Declare array of ye values if first time in loop over psm types
                ye_out = np.zeros((num_proxy, ntottime))
                # initialize with nan
                ye_out[:] = np.nan
                firstloop = False

            # loop over proxies
            for i, pobj in enumerate(proxy_objects):

                if base_time_interval == 'annual':
                    # Restrict to proxy records with seasonality
                    # corresponding to current "season" loop variable
                    if cfg.psm.season_source == 'proxy_metadata':
                        if pobj.seasonality == season:
                            _log.info('{:10d} (...of {:d})'.format(i, num_proxy) + pobj.id)
                            ye_out[i] = pobj.psm(X.ens, X.full_state_info, X.coords)
                            
                    elif cfg.psm.season_source == 'psm_calib':
                        if pobj.psm_obj.seasonality == season:
                            _log.info('{:10d} (...of {:d})'.format(i, num_proxy) + pobj.id)
                            ye_out[i] = pobj.psm(X.ens, X.full_state_info, X.coords)

                else:
                    # "multiyear" PSM:
                    # Restrict to proxy records associated with current PSM (psm_key)
                    if 'bayesreg' in psm_key and pobj.psm_obj.psm_key == psm_key:
                        if psm_avg == 'multiyear-season':
                            # Restrict to proxy records with current 'season'
                            if pobj.seasonality == season:
                                _log.info('{:10d} (...of {:d})'.format(i, num_proxy) + pobj.id)
                                ye_out[i] = pobj.psm(X.ens, X.full_state_info, X.coords)
                        else:
                            # no proxy seasonality to consider
                            _log.info('{:10d} (...of {:d})'.format(i, num_proxy) + pobj.id)
                            ye_out[i] = pobj.psm(X.ens, X.full_state_info, X.coords)

                    elif 'h_interp' in psm_key and pobj.psm_obj.psm_key == psm_key:
                        # no proxy seasonality to consider
                        _log.info('{:10d} (...of {:d})'.format(i, num_proxy) + pobj.id)
                        ye_out[i] = pobj.psm(X.ens, X.full_state_info, X.coords)

        # -> here: end of loop on definitions of time averages (e.g. season)

        elapsed = timeit.default_timer() - masterstarttime
        _log.info('Elapsed time:' + str(elapsed) + ' secs')

        # Create a mapping for each proxy id to an index of the array
        pid_map = {pobj.id: idx for idx, pobj in enumerate(proxy_objects)}

        # Create filename for current experiment
        out_dir = os.path.join(cfg.core.lmr_path, 'ye_precalc_files')

        vkind = X.statevars[list(X.statevars.keys())[0]]
        out_fname = create_precalc_ye_filename(cfg,psm_key,vkind)

        assert len(out_fname) <= 255, 'Filename is too long...'

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # Write precalculated ye file
        out_full = os.path.join(out_dir, out_fname)
        _log.info('Writing precalculated ye file: {}'.format(out_full))
        np.savez(out_full,
                 pid_index_map=pid_map,
                 ye_vals=ye_out)


    elapsedtot = timeit.default_timer() - masterstarttime
    _log.info('Total elapsed time:' + str(elapsedtot/60.0)  + ' mins')


#------------------------------------------------------------------------------------
#-------------------- if not called, must be run directly ---------------------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
        main(config_path=yaml_file)
    else:
        main()
