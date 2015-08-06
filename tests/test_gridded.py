__author__ = 'wperkins'

"""
Test LMR_gridded classes and methods
"""

import pytest
import LMR_gridded as lmrgrid
import numpy as np

@pytest.mark.xfail
def test_gridded_data_no_abstract_instance():
    lmrgrid.GriddedVariable('tas', ['time', 'lat', 'lon'], np.zeros((3,3,3)),
                            [1.0], time=range(3), lat=range(3), lon=range(3))
