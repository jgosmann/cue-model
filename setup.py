#!/usr/bin/env python

from setuptools import find_packages, setup


setup(
    name="imem",
    version='0.1-dev',
    author="Jan Gosmann",
    author_email="jgosmann@uwaterloo.ca",

    packages=find_packages(),
    provides=['imem'],
)
