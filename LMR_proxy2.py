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
        pmeasure = site_meta['Proxy measurement']
        pages_type = site_meta['Archive Type']
        proxy_type = config.proxies.pages.proxy_type_mapping[(pages_type,
                                                              pmeasure)]
        start_yr = site_meta['Youngest (C.E.)']
        end_yr = site_meta['Oldest (C.E.)']
        lat = site_meta['Lat (N)']
        lon = site_meta['Lon (E)']
        # TODO: Figure out how to handle non-consecutive time series with missing
        values = data_src[start <= data_src[site] <= finish]

        try:
            return cls(config, pid, proxy_type, start_yr, end_yr, lat, lon,
                       values, **psm_kwargs)
        except ValueError as e:
            # TODO: logger statement
            return None

    @classmethod
    def load_all(cls, config, data_range, **psm_kwargs):

        meta_src = load_data_frame(config.proxies.pages.metafile_proxy)
        data_src = load_data_frame(config.proxies.pages.datafile_proxy)

        filters = config.proxies.pages.simple_filters
        proxy_order = config.proxies.pages.proxy_order
        ptype_filters = config.proxies.pages.proxy_assim2

        # initial mask all true before filtering
        useable = meta_src[meta_src.columns[0]] == 0
        useable |= True

        # Find indices matching filter specifications
        for colname, filt_list in filters.iteritems():
            simple_mask = meta_src[colname] == 0
            simple_mask &= False

            for value in filt_list:
                simple_mask |= meta_src[colname] == value

            useable &= simple_mask

        # Create proxy id lists
        proxy_id_by_type = {}
        all_proxy_ids = []

        type_col = 'Archive type'
        measure_col = 'Proxy measurement'
        for name in proxy_order:

            type_mask = meta_src[type_col] == 0
            type_mask |= True

            # Filter to proxies of a certain type
            ptype = name.split('_', 1)[0]
            type_mask &= meta_src[type_col] == ptype

            # Reduce to listed measures
            measure_mask = meta_src[measure_col] == 0
            measure_mask &= False

            for measure in ptype_filters[name]:
                measure_mask |= meta_src[measure_col] == measure

            # Extract proxy ids using mask and append to lists
            proxies = meta_src['PAGES ID'][measure_mask & type_mask].values
            proxy_id_by_type[name] = proxies.tolist()
            all_proxy_ids.append(proxies.tolist())

        # Create proxy objects list
        all_proxies = []
        for site in all_proxy_ids:
            pobj = cls.load_site(config, site, meta_src, data_src, data_range,
                                 **psm_kwargs)
            if pobj is not None:
                all_proxies.append(pobj)

        return proxy_id_by_type, all_proxies

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
