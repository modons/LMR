__author__ = 'frodre'

"""
Module containing proxy classes
"""

from abc import ABCMeta, abstractmethod
import LMR_psms
from load_data import load_data_frame

class BaseProxyManager:
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_proxies(self):
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
		psm = psm_obj.psm

	# TODO: Might not need this read_proxy method, it was in consideration
	#   of different data backends which may or may not be used.
    def read_proxy(self):
        return self.values[:]
		
	@abstractmethod
	@staticmethod
	def get_psm_obj():
		pass
		
	@abstractmethod
	@classmethod
	def load_from_site(cls, site, file):
		pass

    @abstractmethod
    def error(self):
        pass


class ProxyPages(BaseProxyObject):

	def get_psm_obj():
		return LMR_psms.get_psm_class('pages')
		
	def load_from_site(cls, config, site, meta_src, data_src,
					   data_range, **psm_kwargs):
		'''
		Initializes proxy object from source data.  Source data
		can be a string or pandas dataframe at this point.
		'''
		
		# TODO: Figure out if this is the correct scope to load
		meta_src = load_data_frame(meta_src)
		data_src = load_data_frame(data_src)
		start, finish = data_range
		
		site_meta = meta_src[meta_src['PAGES ID'] == site]
		id = site_meta['PAGES ID']
		# TODO: Create measurement type for overall group?
		type = site_meta['Proxy measurement']
		start_yr = site_meta['Youngest (C.E.)']
		end_yr = site_meta['Oldest (C.E.)']
		lat = site_meta['Lat (N)']
		lon = site_meta['Lon (E)']
		# TODO: Figure out how to handle non-consecutive time series with missing
		values = data_src[data_src[site] >= start and data_src[site] <= finish]
		
		return cls(config, id, type, start_yr, end_yr, lat, lon, values,
				   **psm_kwargs)
		
    def error(self):
        # Constant error for now
        return 0.1

def fix_lon(lon):
	if lon < 0:
		lon += 360.
	return lon
