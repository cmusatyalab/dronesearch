# Overview  [![PyPI version][pypi-image]][pypi]

[pypi-image]: https://badge.fury.io/py/dronesearch.svg
[pypi]: https://pypi.org/project/dronesearch/

This repo contains a python package [dronesearch](dronesearch) for running live
video analytics on drone video feeds leveraging edge servers. It also contains
our experiment code for SEC'18 paper *[Bandwidth-efficient Live Video Analytics
for Drones via Edge Computing](https://ieeexplore.ieee.org/document/8567664)*.

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

We provide a demo that considers *computer monitors* as objects of interests.
Only video frames that are classified as *computer monitors* will be sent to an
edge server for further analysis.

To run the demo, first clone this directory. Then, issue the following commands
at the root dir of this repo. There will be a window named *Drone Feed* that
pops up showing you the feed from the input source. Once the feed captures a
computer monitor, a second window named *Received Image Feed* will pop up
showing the received frames at the edge server.

```bash
# on drone or your drone emulation platform, by default connecting to tcp://localhost:9000
# --input-source: the uri for OpenCV's VideoCapture(). 
#                 It should be a number for cameras or a file path for videos.
# --filter-config-file: a file path whose content specifies filters to run on the drone.
#                       This demo uses Tensorflow's MobileNet.
# --server-host, and --server-port specifies the edge server.
python -m dronesearch.onboard --input-source 0 --filter-config-file data/cfg/filter_config.ini

# on edge server
# --server-port specifies the listening port.
python -m dronesearch.onserver
```

## Experiments for SEC'18 paper

See [experiment-README](experiment-README.md).
