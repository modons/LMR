#
# compute climate indices from LMR grids
#

import matplotlib
# need to do this when running remotely, and to suppress figures
matplotlib.use('Agg')

import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import glob, os, fnmatch
import numpy as np
import matplotlib.font_manager as font_manager
from scipy import stats
from LMR_plot_support import *
import mpl_toolkits.basemap as bm
import cPickle
import pna
import enso
from LMR_plot_support import moving_average

bm.latlon_default = True
fsize = 14

#datadir_output = '/home/disk/kalman2/wperkins/LMR_output/archive/'
datadir_output = '/home/disk/kalman3/hakim/LMR'
#datadir_output = '../data/'

# name of exeriment
#nexp = 'production_cru_mpi_pagesall_0.75'
#nexp = 'production_cru_ccsm4_pagesall_0.75'
#nexp = 'production_mlost_20cr_pagesall_0.75'
#nexp = 'production_mlost_ccsm4_pagesall_0.75'
#nexp = 'production_gis_ccsm4_pagesall_0.75'
#nexp = 'production_gis_mpi_pagesall_0.75'
#nexp = 'production_mlost_mpi_pagesall_0.75'
#nexp = 'production_mlost_ccsm4_pagestrw_1.0'
#nexp = 'production_mlost_ccsm4_pagesnotrees_1.0'
nexp = 'pages2_loc25000_seasonal_bilinear_nens200'

# use all of the directories found from scanning the disk
workdir = datadir_output + '/' + nexp

# read 500 hPa and compute PNA
state_var = 'zg_500hPa_Amon'

# get number of mc realizations from directory count
tmp = os.listdir(workdir)
niters = 0
mcdir = []
for subdir in tmp:
    if subdir[0] == 'r' and os.path.isdir(workdir+'/'+subdir):
        niters = niters + 1
        mcdir.append(subdir)

print 'mcdir:' + str(mcdir)
print len(mcdir)

prior_filn = workdir + '/r0/ensemble_mean_' + state_var + '.npz'
npzfile = np.load(prior_filn)
lat = npzfile['lat']
lon = npzfile['lon']
years = npzfile['years']
nlat = npzfile['nlat']
nlon = npzfile['nlon']
nyears = len(years)
lat2 = np.reshape(lat,(nlat,nlon))
lon2 = np.reshape(lon,(nlat,nlon))

# only need to generate the grand ensemble mean of all years once...
z500 = np.zeros([nyears,nlat,nlon])

#for dir in mcdir:
#for dir in mcdir[0:10]: # testing
for dir in mcdir:
    ensfiln = workdir + '/' + dir + '/ensemble_mean_' + state_var + '.npz'
    print ensfiln
    npzfile = np.load(ensfiln)
    npzfile.files
    tmp = npzfile['xam']
    z500 = z500 + tmp

#z500 = z500/10. # testing
z500 = z500/len(mcdir)

# compute pna index
pna = pna.pna(lat2,lon2,z500)

# plot
plt.plot(pna)

#
# ENSO
#

t2m = np.zeros([nyears,nlat,nlon])

state_var = 'tas_sfc_Amon'

#for dir in mcdir:
#for dir in mcdir[0:10]: # testing
for dir in mcdir:
    ensfiln = workdir + '/' + dir + '/ensemble_mean_' + state_var + '.npz'
    print ensfiln
    npzfile = np.load(ensfiln)
    npzfile.files
    tmp = npzfile['xam']
    t2m = t2m + tmp

#t2m = t2m/10. # testing
t2m = t2m/len(mcdir)

# compute pna index
nino12,nino3,nino34,nino4 = enso.enso(lat2,lon2,t2m)

# plot
plt.figure()

plt.plot(nino12,label='nino1+2')
plt.plot(nino3,label='nino3')
plt.plot(nino34,label='nino3.4')
plt.plot(nino4,label='nino4')

# 10-year mean
nsyrs=30
snino34,syears = moving_average(nino34,np.arange(0,2000),nsyrs)
plt.plot(syears,snino34,'k-',label='nino3.4 10-yr smooth')
plt.legend()

# departure from 10-year mean
print syears[0],syears[-1]
print len(snino34)
tdiff = nino34[syears[0]:syears[-1]+1]-snino34
print len(tdiff),len(syears)

plt.figure()
plt.plot(syears,tdiff)
cut = 0.4
#plt.plot([syears[0],syears[-1]],[cut,cut],'r-')
#plt.plot([syears[0],syears[-1]],[-cut,-cut],'r-')
plt.xlabel('year CE',fontweight='bold',fontsize=fsize)
plt.ylabel('K',fontweight='bold',fontsize=fsize)
plt.title('NINO3.4 index with 30-year mean removed',fontweight='bold',fontsize=fsize)
plt.ylim([-1.25,1.25])
fname =nexp+'_nino3.4.png'
plt.savefig(fname, dpi=300,format='png',bbox_inches='tight')

# list of El Nino and La Nina years based on cut
#ind = np.where(tdiff>=cut)
#elnino = syears[ind]
#ind = np.where(tdiff<=-cut)
#lanina = syears[ind]

# list of El Nino and La Nina years based on sorting the top nevents events
nevents = 50
tsorted = np.argsort(tdiff) # returns array indices, not the sorted array
elnino = tsorted[-nevents:]+syears[0] # only works becaus the first year in the recon is zero!
lanina = tsorted[0:nevents]+syears[0]

print tsorted+syears[0]
print tsorted[-nevents:]+syears[0]

print 'El Nino years:'
print elnino
print 'El Nino Nino3.4 indices:'
print tdiff[tsorted[-nevents:-1]]
print 'El Nino years sorted:'
print np.sort(elnino)

print 'La Nina years:'
print lanina
print 'La Nina indices:'
print tdiff[tsorted[0:nevents]]
print 'La Nina years sorted:'
print np.sort(lanina)

plt.show()

# save indices for david
f = open('nino34.txt','w')
f.write('years,raw nino3.4 index,30-year smoothed nino3.4 index')
kk = -1
for y in years:
    kk = kk + 1
    if kk < len(snino34):
        out = y+', '+str(nino34[kk])+ ', '+str(snino34[kk])+'\n'
    else:
        out = y+','+str(nino34[kk])+', '+ 'n/a \n'
    f.write(out)
f.close()
