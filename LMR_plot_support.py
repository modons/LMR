
import numpy as np
import mpl_toolkits.basemap as bm
import matplotlib.pyplot as plt


# this is a plotting convenience function for quick LMR plots
def LMR_plotter(data,lat,lon,cmap,nlevs,vmin=None,vmax=None):

    """
    Inputs:
    data : (nlat,nlon) array
    lat, lon : (nlon,nlat) arrays containing latitude and longitude values
    cmap : string defining the colormap (e.g. 'jet' or 'bwr'; see http://matplotlib.org/examples/color/colormaps_reference.html)
    nlevs: number of contour levels
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
    cs = m.contourf(lon,lat,data,cints,cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)
    m.drawcoastlines()
    m.colorbar(cs)
    #cbar = m.colorbar(ticks=cints)


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
    
    edge = (window-1)/2
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
    
    smatch = np.where(time==stime)
    ematch = np.where(time==etime)

    return smatch[0], ematch[0]
