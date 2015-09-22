__author__ = 'frodre'
import sys
sys.path.append('../.')

import misc_config as cfg
import LMR_proxy2
import cPickle as cpckl
from os.path import join

calib_datatag = cfg.psm.linear.datatag_calib
output_dir = '/home/chaos2/wperkins/data/LMR/PSM'
pre_calib_fname = 'PSMs_{}_multires.pckl'.format(calib_datatag)

# Load and calibrate
proxies = LMR_proxy2.ProxyPages.load_all(cfg, [1850, 2000])

pre_calib_dict = {}
for p in proxies:
    psm_obj = p.psm_obj
    pre_calib_dict[(p.type, p.id)] = {'PSMcorrel': psm_obj.corr,
                                      'PSMslope': psm_obj.slope,
                                      'PSMintercept': psm_obj.intercept,
                                      'PSMmse': psm_obj.R,
                                      'calib': calib_datatag,
                                      'lat': psm_obj.lat,
                                      'lon': psm_obj.lon}

with open(join(output_dir, pre_calib_fname), 'w') as f:
    cpckl.dump(pre_calib_dict, f)



