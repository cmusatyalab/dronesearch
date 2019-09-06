# Overview

This repo contains a python package [dronesearch](dronesearch) for running live
video analytics on drones leveraging edge servers. It also contains our
experiment code for SEC'18 paper *[Bandwidth-efficient Live Video Analytics for
Drones via Edge Computing](https://ieeexplore.ieee.org/document/8567664)*.

## dronesearch Package

The decreasing costs of drones have made them suitable for search and rescue
tasks. Analyzing drone video feeds in real-time can greatly improve the
efficiency of search tasks. However, typical drone platforms do not have enough
computation power to do real-time video analysis onboard, especially
semantic-level vision processing, such as human survivor detection, car
detection, and animal detection. Video feeds need to be streamed to an edge
server for computer vision processing. When streaming video feeds from a swarm
of drones at the same time, judicious use of bandwidth becomes important.

This [dronesearch](dronesearch) package provides a computer vision pipeline that
selectively finds interesting frames and transmit them to edge servers for
analysis in order to save bandwidth.

### Installation

First, install [zeromq](https://zeromq.org/download/). Then,

```bash
pip install dronesearch
```

### Demo

```bash
# on drone
python -m dronesearch.onboard --input-source 0 --filter-config-file dronesearch/cfg/filter_config.ini
# on edge server
python -m dronesearch.onserver
```

## Experiments for SEC'18 paper

See [experiment-README](experiment-README.md).
