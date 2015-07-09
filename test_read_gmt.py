
import numpy as np

dir = '/home/disk/kalman3/hakim/LMR//test_saving_gmt_ensemble/r1/'

gmtpfile =  dir + '/gmt.npz'
npzfile = np.load(gmtpfile)
npzfile.files
gmt = npzfile['gmt_save']
print np.shape(gmt)

gmtpfile =  dir + '/gmt_ensemble.npz'
npzfile = np.load(gmtpfile)
npzfile.files
gmt_ensemble = npzfile['gmt_ensemble']
print np.shape(gmt_ensemble)

gmt_em = np.mean(gmt_ensemble,1)
print np.shape(gmt_em)

print 'new gmt from archived full ensemble...'
print gmt_em[0:10]
print 'old gmt from archived ensemble mean...'
print gmt[-1,0:10]
print 'maximum difference over the entire record...'
print np.max(gmt_em-gmt[-1,:])
