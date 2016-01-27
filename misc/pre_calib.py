__author__ = 'frodre'
import sys
sys.path.append('../.')

import misc_config as cfg
import LMR_proxy2
import cPickle as cpckl
from os.path import join

# NOTE: Loads from misc_config, should have pre_calib_datafile set to None

calib_datatag = cfg.psm.linear.datatag_calib
fracs = [1.0]

output_dir = '/home/chaos2/wperkins/data/LMR/PSM/test_psms'

print 'Starting pre-calibration for ' + calib_datatag + ' fracs ' + str(fracs)
for frac in fracs:
    print 'Working on fraction {:1.2f}'.format(frac)
    cfg.psm.linear.min_data_req_frac = frac
    # pre_calib_fname = 'PSMs_{}_multires_{:1.2f}datfrac'.format(calib_datatag, frac)
    pre_calib_fname = 'PSMs_{}_1pt0res_{:1.2f}datfrac'.format(calib_datatag, frac)


    # Load and calibrate
    _, proxies = LMR_proxy2.ProxyPages.load_all(cfg, [1850, 2000])

    pre_calib_dict = {}
    for p in proxies:
        psm_obj = p.psm_obj
        pre_calib_dict[(p.type, p.id)] = {'PSMcorrel': psm_obj.corr,
                                          'PSMslope': psm_obj.slope,
                                          'PSMintercept': psm_obj.intercept,
                                          'PSMmse': psm_obj.R,
                                          'calib': calib_datatag,
                                          'lat': p.lat,
                                          'lon': p.lon,
                                          'nobs': psm_obj.nobs}

    with open(join(output_dir, pre_calib_fname), 'w') as f:
        cpckl.dump(pre_calib_dict, f)



