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

"""On cloudlet processing.
"""

import configparser
import os

import cv2
import fire
from logzero import logger

from dronesearch import dronefilter, networkservice


def _start_event_loop(network_service):
    """Start pipeline of receiving results from onboard

    Args:
      source: Input source
      current_filter: Current selected filter
      filters: All loaded filter
      network_service: Network service to backend

    Returns:

    """
    network_service.open()
    filter_output = dronefilter.ImageFilterOutput()

    while True:
        try:
            filter_output_serialized = network_service.socket.recv()
            filter_output.frombytes(filter_output_serialized)

            logger.debug('received image')
            cv2.imshow('Received Image Feed', filter_output.image)
            cv2.waitKey(1)
        except KeyboardInterrupt:
            break


def start_onserver_processing(network_service='zmq_pair', server_port=9000):
    network_service = networkservice.NetworkService.factory(
        type=network_service, host='*', port=server_port, is_server=True)
    _start_event_loop(network_service)


if __name__ == "__main__":
    fire.Fire(start_onserver_processing)
