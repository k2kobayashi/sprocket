#!/usr/bin/env python

from setuptools import setup, find_packages

import os
if os.path.exists('README.md'):
    README = open('README.md').read()
else:
    README = ""  # a placeholder, readme is generated on release
CHANGES = open('CHANGES.md').read()

setup(
    name="sprocket",
    version="0.18",
    description="Voice conversion software",
    url='https://github.com/k2kobayashi/sprocket',
    author='Kazuhiro Kobayashi',
    packages=find_packages(),
    long_description=(README + '\n' + CHANGES),
    license='MIT',
    install_requires=open('requirements.txt').readlines(),
)
