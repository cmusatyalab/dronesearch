#!/usr/bin/env python
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
