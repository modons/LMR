import numpy as np
from scipy import special
import copy

def gaussianize(X):
    #n = X.shape[0]
    n = X[~np.isnan(X)].shape[0]  # This line counts only elements with data.

    #Xn = np.empty((n,))
    Xn = copy.deepcopy(X)  # This line retains the data type of the original data variable.
    Xn[:] = np.NAN
    nz = np.logical_not(np.isnan(X))

    index = np.argsort(X[nz])
    rank = np.argsort(index)

    CDF = 1.*(rank+1)/(1.*n) -1./(2*n)
    Xn[nz] = np.sqrt(2)*special.erfinv(2*CDF -1)
    return Xn
