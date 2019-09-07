# Overview

Experiment code and data are structured similarly to the [paper](https://ieeexplore.ieee.org/document/8567664) organization. Below are the directories related to SEC'18 paper experiments.

## Installation

1. Clone this module with submodules.

```bash
git clone --recurse-submodules https://github.com/cmusatyalab/dronesearch
```

2. Download and extract [_processed_dataset.tar.gz_](https://storage.cmusatyalab.org/drone2018/processed_dataset.tar.gz)(md5: 2b674b365c9f81ab6be338c7c4a89ad5) and [_experiments.tar.gz_](https://storage.cmusatyalab.org/drone2018/experiments.tar.gz)(md5: 24094df83336785ecae09fa7a1cfd630).

## What's in here?

- [scripts](scripts): experiment code.
- processed_dataset: Datasets and trained models organized by dataset ([okutama](http://okutama-action.org), [stanford_campus](http://cvgl.stanford.edu/projects/uav_data), [raft, elephant](https://drive.google.com/drive/folders/1qBGLDdSxfEkTX6hT6RUouadnDJjzAGAv?usp=sharing)). Each dataset has the following directories.

  - images: images extracted from dataset videos.
  - images_448_224: resized images with resolution of 448x224. These are the inputs to finetuned mobilenet models.
  - annotations: original ground truth annotations.
  - classification_448_224_224_224_annotations: ground truth annotations for early discard experiments with 2 horizontal tiles, corresponding to the images_448_224 inputs.
  - experiments: Tensorflow training input, logs, and trained models.
    - logs_all_layers_40000: trained MobileNet classification models for early discard onboard.
    - \*.tfrecord: training data in TF record format.
    - test*inference*\*: inference results on test data.
    - random_select_and_filter: sampling with early discard results.

- experiments: Experiment data and ploting scripts. Subdirectories are listed below. (Note: Github directory only contains the plotting code. Download using the link above for experiment data.)
  - inference_speed: DNN inference speed (Fig. 3).
  - tile_inference_speed: tiling results (Fig. 8).
  - early_discard_extra_negative: early discard results (Fig. 9 and Fig. 10).
  - random_select: random sampling with early discard results (Fig. 11 and Fig. 12).
  - video_encoding_effect: video encoding results (Fig. 13).
  - jitl_added_0_to_threshold: JITL results (Fig. 14).
  - reachback: reachback results (Fig. 15).

## Compiled TF wheels for embedded platforms

Compiled Joule/Aero Drone Tensorflow Wheel with SSE2 is [here](https://drive.google.com/file/d/1WPkQ52OGUrfSsk7bq7y2kImvAzPnyxnX/view?usp=sharing)
