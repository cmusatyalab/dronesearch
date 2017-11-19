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
import dronesearch.inputsource as inputsource


def _get_config_parser(config_file):
    parser = ConfigParser.SafeConfigParser()
    parser.read(config_file)
    return parser


def _get_input_source(input_source):
    input_source_type = type(input_source)
    if input_source_type != int and input_source_type != str:
        raise ValueError(
            'Cannot create input source from {}'.format(input_source))

    return inputsource.OpenCVInputSource(input_source)


def _validate_filter_config(parser):
    # validate required defaults
    required_sections = ['filter_setting']
    for section in required_sections:
        if section not in parser.sections():
            raise ValueError(
                'Invalid filter configuration. Cannot find section {}'.format(
                    section))

    # validate current_filter needs to be specified
    current_filter = parser.get('filter_setting', 'current_filter')
    if not parser.has_section(current_filter):
        raise ValueError(
            'Invalid filter configuration. '
            'Cannot find section for current_filter {}'.format(current_filter))


def _parse_filter_config(filter_config_file):
    parser = _get_config_parser(filter_config_file)
    _validate_filter_config(parser)
    current_filter_name = parser.get('filter_setting', 'current_filter')
    filter_configs = {}
    filter_names = parser.sections()
    filter_names.remove('filter_setting')
    for filter_name in filter_names:
        filter_configs[filter_name] = dict(parser.items(filter_name))
    return current_filter_name, filter_configs


def _get_filters_from_config(filter_configs):
    filters = {}
    for filter_name, filter_config in filter_configs.items():
        filters[filter_name] = dronefilter.DroneFilter.factory(
            name=filter_name, **filter_config)
    return filters


def _get_filters_from_config_file(filter_config_file):
    current_filter_name, filter_configs = _parse_filter_config(
        filter_config_file)
    filters = _get_filters_from_config(filter_configs)
    logger.info('current filter: {}'.format(current_filter_name))
    return filters[current_filter_name], filters


def _start_event_loop():
    pass


def start_onboard_processing(input_source, filter_config_file):
    source = _get_input_source(input_source)
    current_filter, filters = _get_filters_from_config_file(filter_config_file)
    # _start_event_loop


if __name__ == "__main__":
    fire.Fire(start_onboard_processing)
