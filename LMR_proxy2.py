__author__ = 'frodre'

"""
Module containing proxy classes
"""

import LMR_psms
from load_data import load_data_frame

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from random import sample

class ProxyManager:

    def __init__(self, config, data_range):

        self.all_proxies = []
        self.all_ids_by_group = defaultdict(list)

        for proxy_class_key in config.proxies.use_from:
            pclass = get_proxy_class(proxy_class_key)

            #Get PSM kwargs
            proxy_psm_key = config.psm.use_psm[proxy_class_key]
            psm_class = LMR_psms.get_psm_class(proxy_psm_key)
            psm_kwargs = psm_class.get_kwargs(config)

            # Load proxies for current class
            ids_by_grp, proxies = pclass.load_all(config,
                                                  data_range,
                                                  **psm_kwargs)

            # Add to total lists
            self.all_proxies += proxies
            for k, v in ids_by_grp.iteritems():
                self.all_ids_by_group[k] += v

        proxy_frac = config.proxies.proxy_frac
        nsites = len(self.all_proxies)

        if proxy_frac < 1.0:
            nsites_assim = int(nsites * proxy_frac)

            self.ind_assim = sample(range(nsites), nsites_assim)
            self.ind_eval = set(range(nsites)) - set(self.ind_assim)
        else:
            self.ind_assim = range(nsites)
            self.ind_eval = None

    def proxy_obj_generator(self, indexes):
        for idx in indexes:
            yield self.all_proxies[idx]

    def sites_assim_proxy_objs(self):
        return self.proxy_obj_generator(self.ind_assim)

    def sites_eval_proxy_objs(self):
        return self.proxy_obj_generator(self.ind_eval)




class BaseProxyObject:
    __metaclass__ = ABCMeta

    def __init__(self, config, pid, prox_type, start_yr, end_yr, 
                 lat, lon, values, time,
                 missing=None, **psm_kwargs):

        if (values is None) or len(values) == 0:
            raise ValueError('No proxy data given for object initialization')

        assert len(values) == len(time), 'Time and value dimensions must match'

        self.id = pid
        self.type = prox_type
        self.start_yr = start_yr
        self.end_yr = end_yr
        self.values = values
        self.missing = missing
        self.lat = lat
        self.lon = fix_lon(lon)
        self.time = time

        # Retrieve appropriate PSM function
        psm_obj = self.get_psm_obj(config)
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

    @classmethod
    @abstractmethod
    def load_all(cls, config, data_range, **psm_kwargs):
        pass

    @abstractmethod
    def error(self):
        pass


class ProxyPages(BaseProxyObject):

    @staticmethod
    def get_psm_obj(config):
        psm_key = config.psm.use_psm['pages']
        return LMR_psms.get_psm_class(psm_key)

    @classmethod
    def load_site(cls, config, site, data_range, meta_src=None, data_src=None,
                  **psm_kwargs):
        """
        Initializes proxy object from source data.  Source data
        can be a string or pandas dataframe at this point.
        """

        # TODO: Figure out if this is the correct scope to load
        if meta_src is None:
            meta_src = load_data_frame(config.proxies.pages.metafile_proxy)
        if data_src is None:
            data_src = load_data_frame(config.proxies.pages.datafile_proxy)
        start, finish = data_range

        site_meta = meta_src[meta_src['PAGES ID'] == site]
        pid = site_meta['PAGES ID'].iloc[0]
        pmeasure = site_meta['Proxy measurement'].iloc[0]
        pages_type = site_meta['Archive type'].iloc[0]
        proxy_type = config.proxies.pages.proxy_type_mapping[(pages_type,
                                                              pmeasure)]
        start_yr = site_meta['Youngest (C.E.)'].iloc[0]
        end_yr = site_meta['Oldest (C.E.)'].iloc[0]
        lat = site_meta['Lat (N)'].iloc[0]
        lon = site_meta['Lon (E)'].iloc[0]
        site_data = data_src[site]
        values = site_data[(site_data.index >= start) &
                           (site_data.index <= finish)]
        # Might need to remove following line
        values = values[values.notnull()]
        times = values.index.values

        if len(values) == 0:
            raise ValueError('No observations in specified time range.')

        return cls(config, pid, proxy_type, start_yr, end_yr, lat, lon,
                   values, times, **psm_kwargs)

    @classmethod
    def load_all(cls, config, data_range, meta_src=None,
                 data_src=None, **psm_kwargs):

        # Load source data files
        if meta_src is None:
            meta_src = load_data_frame(config.proxies.pages.metafile_proxy)
        if data_src is None:
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
            proxies = meta_src['PAGES ID'][measure_mask & type_mask &
                                           useable].values

            # If we have ids after filtering add them to the type list
            if len(proxies) > 0:
                proxy_id_by_type[name] = proxies.tolist()

            all_proxy_ids += proxies.tolist()

        # Create proxy objects list
        all_proxies = []
        for site in all_proxy_ids:
            try:
                pobj = cls.load_site(config, site, data_range,
                                     meta_src=meta_src, data_src=data_src,
                                     **psm_kwargs)
                all_proxies.append(pobj)
            except ValueError as e:
                # Proxy had no obs or didn't meet psm r crit
                for group in proxy_id_by_type.values():
                    if site in group:
                        group.remove(site)
                        break  # Should only be one instance

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
    return _proxy_classes[key]
