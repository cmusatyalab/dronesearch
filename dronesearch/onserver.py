#!/usr/bin/env python
"""On board processing.

Run filters on board and send them to backend
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import ConfigParser

import fire
from logzero import logger

from dronesearch import dronefilter
import dronesearch.networkservice as networkservice


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
    filter_output = dronefilter.TileFilterOutput()
    while True:
        filter_output_serialized = network_service.socket.recv()
        filter_output.frombytes(filter_output_serialized)
        logger.info('received images in tiles: {}'.format(
            filter_output.indices))


def start_onserver_processing(network_service='zmq_pair', server_port=9000):
    network_service = networkservice.NetworkService.factory(
        type=network_service, host='*', port=server_port, is_server=True)
    _start_event_loop(network_service)


if __name__ == "__main__":
    fire.Fire(start_onserver_processing)