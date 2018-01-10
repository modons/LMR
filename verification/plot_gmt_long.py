#
# plot long time series of GMT to compare different experiments
#
# borrowed heavily from LMR_verify_GM.py
import matplotlib
# need to do this when running remotely, and to suppress figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker

import glob, os, fnmatch
import numpy as np

expts = ['pages2_loc15000_pages2k2_seasonal_TorP_nens200',
         'pages2_noloc',
         'pages2_loc12000'
         ]

# define the time interval
stime = 1880
etime = 2000

fig = plt.figure()

xl = [stime,etime]

for nexp in expts:
    datadir_output = '/home/disk/kalman3/hakim/LMR'
    workdir = datadir_output + '/' + nexp

    # get a listing of the iteration directories
    dirs = glob.glob(workdir+"/r*")
    dirset = dirs
    niters = len(dirset)

    print('--------------------------------------------------')
    print('niters = %s' % str(niters))
    print('--------------------------------------------------')

    first = True
    kk = -1
    for dir in dirset:
        kk = kk + 1
        gmtpfile =  dir + '/gmt_ensemble.npz'
        npzfile = np.load(gmtpfile)
        npzfile.files
        gmt = npzfile['gmt_ensemble']
        nhmt = npzfile['nhmt_ensemble']
        shmt = npzfile['shmt_ensemble']
        recon_times = npzfile['recon_times']
        print(gmtpfile)
        gmt_shape = np.shape(gmt)
        nhmt_shape = np.shape(nhmt)
        shmt_shape = np.shape(shmt)
        if first:
            gmt_save = np.zeros([gmt_shape[0],gmt_shape[1],niters])
            nhmt_save = np.zeros([nhmt_shape[0],nhmt_shape[1],niters])
            shmt_save = np.zeros([shmt_shape[0],shmt_shape[1],niters])
            first = False

        gmt_save[:,:,kk] = gmt
        nhmt_save[:,:,kk] = nhmt
        shmt_save[:,:,kk] = shmt

    # average and 5-95% range
    # 1. global mean
    gmse = np.reshape(gmt_save,(gmt_shape[0],gmt_shape[1]*niters))
    sagmt = np.mean(gmse,1)
    gmt_min = np.percentile(gmse,5,axis=1)
    gmt_max = np.percentile(gmse,95,axis=1)
    # 2. NH
    nhse = np.reshape(nhmt_save,(nhmt_shape[0],nhmt_shape[1]*niters))
    sanhmt = np.mean(nhse,1)
    nhmt_min = np.percentile(nhse,5,axis=1)
    nhmt_max = np.percentile(nhse,95,axis=1)
    # 3. SH
    shse = np.reshape(shmt_save,(shmt_shape[0],shmt_shape[1]*niters))
    sashmt = np.mean(shse,1)
    shmt_min = np.percentile(shse,5,axis=1)
    shmt_max = np.percentile(shse,95,axis=1)

    lmr_gm = sagmt
    LMR_time = recon_times

    lw = 1
    plt.plot(LMR_time,lmr_gm,linewidth=lw,alpha=0.5,label=nexp)
    #plt.fill_between(recon_times,gmt_min,gmt_max,facecolor='gray',alpha = 0.5,linewidth=0.)

plt.title('Global mean temperature',weight='bold',y=1.025)
plt.xlabel('Year CE',fontweight='bold')
plt.ylabel('Temperature anomaly (K)',fontweight='bold')
plt.legend()

plt.savefig('multiexpt_GMT_'+str(xl[0])+'-'+str(xl[1])+'_annual.png')
