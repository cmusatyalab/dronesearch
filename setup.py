
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

import os

import setuptools

file_dir = os.path.dirname(os.path.realpath(__file__))


with open(os.path.join(file_dir, 'README.md')) as f:
    long_description = f.read()

install_requires = [
    'setuptools>=41.0.0',  # tensorboard requirements
    'logzero',
    'fire',
    'tensorflow',
    'pyzmq',
    'opencv-python'
]

setuptools.setup(
    name='dronesearch',
    version='1.0.0.1',
    author='Junjue Wang',
    author_email='junjuew@cs.cmu.edu',
    description='A computer vision pipeline for live video search on drone video feeds leveraging edge servers.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/cmusatyalab/dronesearch',
    packages=setuptools.find_packages(),
    license='Apache License 2.0',
    install_requires=install_requires,
    python_requires='>3.5, <4',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ]
)
