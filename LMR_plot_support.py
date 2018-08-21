"""
Module: LMR_plot_support.py

Purpose: Contains various definitions and functions related to plotting of LMR results.

Originator: Greg Hakim | Dept. of Atmospheric Sciences, Univ. of Washington

Revisions: 
          - Added plotting of parallels & meridians in maps produced in LMR_plotter
            [R. Tardif, U. of Washington, March 2016]

"""

import numpy as np
import mpl_toolkits.basemap as bm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker

# =============================================================================
def truncate_colormap(cmap, minval=0.0,maxval=1.0,n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name,a=minval,b=maxval),
        cmap(np.linspace(minval,maxval,n)))
    return new_cmap
# =============================================================================

# this is a plotting convenience function for quick LMR plots
def LMR_plotter(data,lat,lon,cmap,nlevs,vmin=None,vmax=None,extend=None,backg=None,cbarfmt=None,nticks=None):

    """
    Inputs:
    data     : (nlat,nlon) array
    lat, lon : (nlon,nlat) arrays containing latitude and longitude values
    cmap     : string defining the colormap (e.g. 'jet' or 'bwr'; see http://matplotlib.org/examples/color/colormaps_reference.html)
    nlevs    : number of contour levels
    vmin     :
    vmax     :
    backg    : background color of map (will highlight region of missing data)
    cbarfmt  : ...
    nticks   : ...
    """

    # change default value of latlon kwarg to True.
    bm.latlon_default = True
    
    # max and min values
    maxv = np.nanmax(data)
    minv = np.nanmin(data)
    maxabs = np.nanmax(np.abs(data))
       
    # use the maximum absolute value for scaling
    if vmin == None:
        vmin = -maxabs
    if vmax == None:
        vmax = maxabs

    # set the contour values based on the data range
    cints = np.linspace(vmin, vmax, nlevs, endpoint=True)
    m = bm.Basemap(projection='robin',lon_0=0)
    cs = m.contourf(lon,lat,data,cints,cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax,extend=extend)
    m.drawmapboundary(fill_color = backg)
    m.drawcoastlines()
    cb = m.colorbar(cs,format=cbarfmt)
    #cbar = m.colorbar(ticks=cints)
    #cs.cmap.set_bad('lightgrey')
    if nticks:
        cb.ax.tick_params(labelsize=11)
        tick_locator = ticker.MaxNLocator(nbins=nticks)
        cb.locator = tick_locator
        cb.ax.yaxis.set_major_locator(ticker.AutoLocator())
        cb.update_ticks()
    # draw parallels & meridians
    parallels = np.arange(-90.,90,30.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=6,color='gray')
    meridians = np.arange(0.,360.,60.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=6,color='gray')

#
# read the Central England Temperature dataset
#
def load_HadCET():
    import numpy as np
    
    dat = np.loadtxt('HadCET_central_England_temperature_monthly.dat',skiprows=7)
    
    # these are the years
    HadCET_years = dat[:,0]
    
    # these are the annual average values
    HadCET_T = dat[:,-1]
    
    return HadCET_years,HadCET_T


def moving_average(data,xvals,window=5):

    # data is the input series 
    # window is the number of entries in data to average over (should be ODD)
    # the first value in data_smoothed is the mean of the first window values in data
    
    edge = (window-1)//2
    weigths = np.repeat(1.0, window)/window
    data_smoothed = np.convolve(data, weigths, 'valid')
    
    # also return the x values for which the data is valid
    xvals_smoothed = xvals[edge:-edge]
    
    return data_smoothed,xvals_smoothed


def plot_direction(CL,fname=''):
    
    if CL:
        plt.savefig(fname+'.png')
    else:
        plt.show()
        
    # "clear" the figure
    plt.clf()

    return None

    
def find_date_indices(time,stime,etime):

    # find start and end times that match specific values
    # input: time: an array of time values
    #        stime: the starting time
    #        etime: the ending time

    # initialize returned variables
    begin_index = None
    end_index = None
    
    smatch = np.where(time==stime)
    ematch = np.where(time==etime)

    # make sure valid integers are returned
    if type(smatch) is tuple:
        smatch = smatch[0]
        ematch = ematch[0]
        
    if type(smatch) is np.ndarray:
        try:
            smatch = smatch[0]
        except IndexError:
            pass
        try:
            ematch = ematch[0]
        except IndexError:
            pass

    
    if isinstance(smatch,(int,np.integer)):
        begin_index = smatch
    if isinstance(ematch,(int,np.integer)):
        end_index = ematch

    return begin_index, end_index

