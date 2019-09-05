#!/usr/bin/env python
"""Filters for on-board processing
"""

import abc

import zmq
from logzero import logger


class NetworkService(object, metaclass=abc.ABCMeta):
    @classmethod
    def factory(cls, **kwargs):
        filter_type = kwargs.pop('type')
        if filter_type == 'zmq_pair':
            return ZMQPairNetworkService(**kwargs)
        else:
            raise ValueError('Unsupported filter type: {}'.format(filter_type))

    @abc.abstractmethod
    def open():
        pass

    @abc.abstractmethod
    def send(data):
        pass

    @abc.abstractmethod
    def recv():
        pass

    @abc.abstractmethod
    def close():
        pass


class ZMQPairNetworkService(object):
    def __init__(self, host, port, is_server=False):
        """ZMQ Pair Network Service
        """
        self.host = host
        self.port = port
        self.socket = None
        self._is_server = is_server

    def open(self):
        self._context = zmq.Context()
        self.socket = self._context.socket(zmq.PAIR)
        if self._is_server:
            self.socket.bind("tcp://{}:{}".format(self.host, self.port))
        else:
            self.socket.connect("tcp://{}:{}".format(self.host, self.port))

    def send(self, data):
        self.socket.send(data)

    def recv(self, flags):
        self.socket.recv(flags)

    def close(self):
        self.socket.close()
