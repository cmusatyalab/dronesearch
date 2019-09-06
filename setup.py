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
]

setuptools.setup(
    name='dronesearch',
    version='1.0.0',
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
