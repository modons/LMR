__author__ = 'frodre'

"""
Module containing proxy classes
"""

from abc import ABCMeta, abstractmethod
import LMR_psms
from load_data import load_data_frame

class ProxyManager:
    pass


class BaseProxyObject:
    __metaclass__ = ABCMeta

    def __init__(self, config, pid, prox_type, start_yr, end_yr, 
                 lat, lon, values,
                 missing=None, psm_kwargs=None):
        self.id = pid
        self.type = prox_type
        self.start_yr = start_yr
        self.end_yr = end_yr
        self.values = values
        self.missing = missing
        self.lat = lat
        self.lon = fix_lon(lon)

        # Retrieve appropriate PSM function
        psm_obj = self.get_psm_obj()
        psm_obj = psm_obj(config, self, **psm_kwargs)
        self.psm = psm_obj.psm

    # TODO: Might not need this read_proxy method, it was in consideration
    #   of different data backends which may or may not be used.
    def read_proxy(self):
        return self.values[:]

    @staticmethod
    @abstractmethod
    def get_psm_obj():
        pass

    @classmethod
    @abstractmethod
    def load_site(cls,  config, site, meta_src, data_src,
                  data_range, **psm_kwargs):
        pass

    @abstractmethod
    def error(self):
        pass


class ProxyPages(BaseProxyObject):

    @staticmethod
    def get_psm_obj():
        return LMR_psms.get_psm_class('pages')

    @classmethod
    def load_site(cls, config, site, meta_src, data_src,
                  data_range, **psm_kwargs):
        """
        Initializes proxy object from source data.  Source data
        can be a string or pandas dataframe at this point.
        """

        # TODO: Figure out if this is the correct scope to load
        meta_src = load_data_frame(meta_src)
        data_src = load_data_frame(data_src)
        start, finish = data_range

        site_meta = meta_src[meta_src['PAGES ID'] == site]
        pid = site_meta['PAGES ID']
        # TODO: Create measurement type for overall group (e.g. Tree Rings)?
        ptype = site_meta['Proxy measurement']
        start_yr = site_meta['Youngest (C.E.)']
        end_yr = site_meta['Oldest (C.E.)']
        lat = site_meta['Lat (N)']
        lon = site_meta['Lon (E)']
        # TODO: Figure out how to handle non-consecutive time series with missing
        values = data_src[start <= data_src[site] <= finish]
        try:
            return cls(config, pid, ptype, start_yr, end_yr, lat, lon, values,
                       **psm_kwargs)
        except ValueError as e:
            # TODO: logger statement
            pass

    @classmethod
    def load_all(cls, config, data_range, **psm_kwargs):

        meta_src = load_data_frame(config.proxies.pages.metafile_proxy)
        data_src = load_data_frame(config.proxies.pages.datafile_proxy)

        filters = config.proxies.pages.dat_filters

        # create index mask from filters

    def error(self):
        # Constant error for now
        return 0.1

def fix_lon(lon):
    if lon < 0:
        lon += 360.
    return lon


_proxy_classes = {'pages': ProxyPages}

def get_proxy_class(key):
    return _proxy_classes
