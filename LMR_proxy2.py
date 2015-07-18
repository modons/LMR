"""
Module containing proxy classes.  Rewritten by AndreP to incorporate features
from the original LMR_proxy code using OOP.
"""

import LMR_psms
from load_data import load_data_frame
from LMR_utils2 import augment_docstr, class_docs_fixer

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from random import sample
from copy import deepcopy

class ProxyManager:
    """
    High-level manager to handle loading proxies from multiple sources and
    randomly sampling the proxies.

    Attributes
    ----------
    all_proxies: list(BaseProxyObject like)
        A list of all proxy objects loaded for current reconstruction
    all_ids_by_group: dict{str: list(str)}
        A dictionary holding list of proxy site ids for each proxy type loaded.
    ind_assim: list(int)
        List of indices (pretaining to all_proxies)to be assimilated during the
        reconstruction.
    ind_eval: list(int)
        List of indices of proxies withheld for verification purposes.

    Parameters
    ----------
    config: LMR_config
        Configuration module for current LMR run
    data_range: list(int)
        A two int list defining the beginning and ending time of the
        reconstruction
    """

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

        # Sample from all proxies if specified
        if proxy_frac < 1.0:
            nsites_assim = int(nsites * proxy_frac)

            self.ind_assim = sample(range(nsites), nsites_assim)
            self.ind_assim.sort()
            self.ind_eval = list(set(range(nsites)) - set(self.ind_assim))
            self.ind_eval.sort()

            # Make list of assimilated proxies by group
            self.assim_ids_by_group = deepcopy(self.all_ids_by_group)
            for idx in self.ind_eval:
                pobj = self.all_proxies[idx]
                grp = self.assim_ids_by_group[pobj.type]
                if pobj.id in grp:
                    grp.remove(pobj.id)

        else:
            self.ind_assim = range(nsites)
            self.ind_eval = None
            self.assim_ids_by_group = self.all_ids_by_group

    def proxy_obj_generator(self, indexes):
        """
        Generator to iterate over proxy objects in list at specified indexes

        Parameters
        ----------
        indexes: list(int)
            List of indices pertaining to self.all_proxies

        Returns
        -------
        generator
            A generator over all_proxies for each specified index
        """

        for idx in indexes:
            yield self.all_proxies[idx]

    def sites_assim_proxy_objs(self):
        """
        Generator over ind_assim indices.

        Yields
        ------
        BaseProxyObject like
            Proxy object from the all_proxies list.
        """

        return self.proxy_obj_generator(self.ind_assim)

    def sites_eval_proxy_objs(self):
        """
        Generator over ind_eval indices.

        Yields
        ------
        BaseProxyObject like
            Proxy object from the all_proxies list.
        """

        return self.proxy_obj_generator(self.ind_eval)


class BaseProxyObject:
    """
    Class defining attributes and methods for descendant proxy objects.

    Attributes
    ----------
    id: str
        Proxy site id
    type: str
        Proxy type
    start_yr: int
        Earliest year that data exists for this site
    end_yr: int
        Latest year that data exists for this site
    values: pandas.DataFrame
        Proxy record with time (in years) as the index
    lat: float
        Latitude of proxy site
    lon: float
        Longitude of proxy site
    time: ndarray
        List of times for which proxy contains valid data
    psm_obj: BasePSM like
        PSM for this proxy
    psm: function
        Exposed psm mapping function from the psm_obj

    Parameters
    ----------
    config: LMR_config
        Configuration module for current LMR run
    pid -> id
    prox_type -> type
    start_yr
    end_yr
    lat
    lon
    values
    time
    psm_kwargs: dict(unpacked)
        Keyword arguments to be passed to the PSM

    Notes
    -----
    All proxy object classes should descend from the BaseProxyObject abstract
    class.
    """
    __metaclass__ = ABCMeta

    def __init__(self, config, pid, prox_type, start_yr, end_yr, 
                 lat, lon, values, time, **psm_kwargs):

        if (values is None) or len(values) == 0:
            raise ValueError('No proxy data given for object initialization')

        assert len(values) == len(time), 'Time and value dimensions must match'

        self.id = pid
        self.type = prox_type
        self.start_yr = start_yr
        self.end_yr = end_yr
        self.values = values
        self.lat = lat
        self.lon = fix_lon(lon)
        self.time = time

        # Retrieve appropriate PSM function
        psm_obj = self.get_psm_obj(config)
        self.psm_obj = psm_obj(config, self, **psm_kwargs)
        self.psm = self.psm_obj.psm

    @staticmethod
    @abstractmethod
    def get_psm_obj():
        """ Retrieves PSM object class to be attached to this proxy"""
        pass

    @classmethod
    @abstractmethod
    def load_site(cls,  config, site, data_range, meta_src=None, data_src=None,
                  **psm_kwargs):
        """
        Load proxy object from single site.

        Parameters
        ----------
        config: LMR_config
            Configuration for current LMR run
        site: str
            Key to identify which site to load from source data
        meta_src: optional
            Source for proxy metadata
        data_src: optional
            Source for proxy record data (might be same as meta_src)
        data_range: iterable
            Two-item container holding beginning and end date of reconstruction
        psm_kwargs: dict(unpacked)
            Keyword arguments to be passed to the psm.

        Returns
        -------
        BaseProxyObject like
            Proxy object instance at specified site

        Notes
        -----
        If source data not specified, it should attempt to load data using
        config file information.
        """
        pass

    @classmethod
    @abstractmethod
    def load_all(cls, config, data_range, **psm_kwargs):
        """
        Load proxy objects from all sites matching filter criterion.

        Parameters
        ----------
        config: LMR_config
            Configuration for current LMR run
        meta_src: optional
            Source for proxy metadata
        data_src: optional
            Source for proxy record data (might be same as meta_src)
        data_range: iterable
            Two-item container holding beginning and end date of reconstruction
        psm_kwargs: dict(unpacked)
            Keyword arguments to be passed to the psm.

        Returns
        -------
        dict
            Dictionary of proxy types (keys) with associated site ids (values)
        list
            List of all proxy objects created

        Notes
        -----
        If source data not specified, it should attempt to load data using
        config file information.
        """
        pass

    @abstractmethod
    def error(self):
        """
        error model for proxy data
        """
        pass


@class_docs_fixer
class ProxyPages(BaseProxyObject):

    @staticmethod
    def get_psm_obj(config):
        psm_key = config.psm.use_psm['pages']
        return LMR_psms.get_psm_class(psm_key)

    @classmethod
    @augment_docstr
    def load_site(cls, config, site, data_range, meta_src=None, data_src=None,
                  **psm_kwargs):
        """%%aug%%

        Expects meta_src, data_src to be pickled pandas DataFrame objects.
        """

        #
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
    @augment_docstr
    def load_all(cls, config, data_range, meta_src=None,
                 data_src=None, **psm_kwargs):
        """%%aug%%

        Expects meta_src, data_src to be pickled pandas DataFrame objects.
        """

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
    """
    Fixes negative longitude values.

    Parameters
    ----------
    lon: ndarray like or value
        Input longitude array or single value
    """
    if lon < 0:
        lon += 360.
    return lon


_proxy_classes = {'pages': ProxyPages}

def get_proxy_class(proxy_key):
    """
    Retrieve proxy class type to be instantiated.

    Parameters
    ----------
    proxy_key: str
        Dict key to retrieve correct PSM class type.

    Returns
    -------
    BaseProxyObject like:
        Class type to be instantiated and attached to a proxy.
    """
    return _proxy_classes[proxy_key]
