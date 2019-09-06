#!/usr/bin/env python

# Drone Search
#
# A computer vision pipeline for live video search on drone video
# feeds leveraging edge servers.
#
# Copyright (C) 2018-2019 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
