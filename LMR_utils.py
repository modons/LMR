
#==========================================================================================
#
# 
#========================================================================================== 


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """

    from math import radians, cos, sin, asin, sqrt

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367.0 * c
    return km

# pad the year with leading zeros for file I/O
def year_fix(it):
    t = str(int(round(it)))
    if t < 10:
        ypad = '000'+str(t)
    elif t >= 10 and t < 100:
        ypad = '00'+str(t)
    elif t >= 100 and t < 1000:
        ypad = '0'+str(t)
    else:
        ypad = str(t)
    
    return ypad

def smooth2D(im, n=15):
    """
    Smooth a 2D array im by convolving with a Gaussian kernel of size n
    Input:
    im (2D array): Array of values to be smoothed
    n (int) : number of points to include in the smoothing
    Output:
    improc(2D array): smoothed array (same dimensions as the input array)
    """

    import numpy
    from scipy import signal

    # Calculate a normalised Gaussian kernel to apply as the smoothing function.
    size = int(n)
    x,y = numpy.mgrid[-n:n+1,-n:n+1]
    gk = numpy.exp(-(x**2/float(n)+y**2/float(n)))
    g = gk / gk.sum()

    #bndy = 'symm'
    bndy = 'wrap'
    improc = signal.convolve2d(im, g, mode='same', boundary=bndy)
    return(improc)
