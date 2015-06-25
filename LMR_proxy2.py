__author__ = 'frodre'

"""
Module containing proxy classes
"""

from abc import ABCMeta, abstractmethod

class BaseProxyManager:
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_proxies(self):
        pass


class BaseProxyObject:
    __metaclass__ = ABCMeta

    def __init__(self, config, pid, prox_type, start_yr, end_yr, values,
                 missing=None):
        self.id = pid
        self.type = prox_type
        self.start_yr = start_yr
        self.end_yr = end_yr
        self.values = values
        self.missing = missing
        self.lat = None
        self.lon = None

    # Retrieve appropriate PSM function
    psm_obj = PSMFactory.create(config)

    @abstractmethod
    def read_proxy(self):
        return self.values[:]

    @abstractmethod
    def error(self):
        pass


class ProxyPages(BaseProxyObject):

    def read_proxy(self, ):
        pass

    def error(self):
        # Constant error for now
        return 0.1
