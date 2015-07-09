#
# read the NOAA MLOST global mean temperature time series data and plot
#

import csv
import numpy as np
import matplotlib.pyplot as plt

path = '/home/disk/ice4/hakim/svnwork/python-lib/trunk/src/ipython_notebooks/data/NOAA_MLOST/'
fname = 'NOAA_MLOST_aravg.ann.land_ocean.90S.90N.v3.5.4.201504.asc'
f = open(path+fname,'r')
dat = csv.reader(f)

MLOST_time = []
MLOST = []
for row in dat:
    # this is the year
    MLOST_time.append(int(row[0].split()[0]))

    # this is the GMT temperature anomaly
    MLOST.append(float(row[0].split()[1]))

    
plt.plot(MLOST_time,MLOST)
plt.show()
