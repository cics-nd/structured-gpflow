#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages
from pkg_resources import parse_version

dependency_links = [
    "https://github.com/sdatkinson/GPflow"  # Use my fork
]

requirements = [
    "matplotlib>=2.1.2",
    "gpflow==1.1.1",
    "pytest>=3.5.0"
]

# Check for TensorFlow
# From GPflow:
# Only detect TF if not installed or outdated. If not, do not do not list as
# requirement to avoid installing over e.g. tensorflow-gpu
# To avoid this, rely on importing rather than the package name (like pip).

min_tf_version = '1.5.0'
tf_cpu = 'tensorflow>={}'.format(min_tf_version)
tf_gpu = 'tensorflow-gpu>={}'.format(min_tf_version)

try:
    # If tf not installed, import raises ImportError
    import tensorflow as tf
    if parse_version(tf.VERSION) < parse_version(min_tf_version):
        # TF pre-installed, but below the minimum required version
        raise DeprecationWarning("TensorFlow version below minimum requirement")
except (ImportError, DeprecationWarning) as e:
    # Add TensorFlow to dependencies to trigger installation/update
    requirements.append(tf_cpu)

setup(name='structured_gpflow',
    version='0.1.0',
    description='structured-gpflow - GPs with Kronecker tricks',
    author='Steven Atkinson',
    author_email='steven@atkinson.mn',
    url='https://github.com/cics-nd/structured-gpflow',
    install_requires=requirements,
    dependency_links=dependency_links,
    packages=find_packages(),
)
