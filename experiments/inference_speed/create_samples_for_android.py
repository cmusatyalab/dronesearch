from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import fire
import numpy as np
from PIL import Image


def create_samples(output_dir, num_sample_per_run=100, im_h=224, im_w=224):
    samples = (np.random.randn(num_sample_per_run, im_h, im_w, 3) * 255).astype(np.uint8)
    # save samples as jpeg images
    idx = 0
    for image in samples:
        im = Image.fromarray(image)
        im.save(os.path.join(output_dir, "{:010d}.jpg".format(idx)))
        idx += 1


if __name__ == "__main__":
    fire.Fire(create_samples)
