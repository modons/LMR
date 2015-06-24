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

    @abstractmethod
    def read_proxy(self):
        pass

    @abstractmethod
    def psm(self):
        pass

    @abstractmethod
    def error(self):
        pass