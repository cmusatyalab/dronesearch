"""Prepare training data set from the ImageNet VID to do car/nocar detection.
"""

from __future__ import absolute_import, division, print_function

import fire
import glob
import os
import random


def symlink_images(video_list_file_path,
                   vid_base_dir,
                   output_dir,
                   max_num=2**20):
    with open(video_list_file_path, 'r') as f:
        contents = f.read().splitlines()
    frame_sequence_relative_paths = [line.split(' ')[0] for line in contents]
    random.shuffle(frame_sequence_relative_paths)

    selected_num = 0
    for frame_sequence_relative_path in frame_sequence_relative_paths:
        frame_sequence_dir_path = os.path.join(
            vid_base_dir, frame_sequence_relative_path)
        file_paths = glob.glob(os.path.join(frame_sequence_dir_path, '*'))
        skipped = False
        for file_path in file_paths:
            # give each symlink a unique name
            symlink_name = os.path.relpath(
                file_path, vid_base_dir).replace('/', '_')
            try:
                os.symlink(file_path, os.path.join(output_dir, symlink_name))
            except OSError as e:
                # A video may contain multiple objects
                # therefore appears multiple
                # times in the video_list_file_path provided
                if e.errno == 17:
                    print('skip video {} since'
                          'it has been included'.format(file_path))
                    skipped = True
                    break
                else:
                    raise e

        if not skipped:
            print('added {} files'.format(selected_num))
            selected_num += len(file_paths)
        if selected_num > max_num:
            break


if __name__ == '__main__':
    fire.Fire()
