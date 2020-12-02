#!/usr/bin/env python

from setuptools import setup, find_packages

import os

if os.path.exists("README.md"):
    README = open("README.md").read()
else:
    README = ""  # a placeholder, readme is generated on release
CHANGES = open("CHANGES.md").read()

setup(
    # package info
    name="sprocket-vc",
    version="0.18.4",
    description="Voice conversion software",
    url="https://github.com/k2kobayashi/sprocket",
    license="MIT",
    # author details
    author="Kazuhiro Kobayashi",
    author_email="root.4mac@gmail.com",
    # package
    packages=find_packages(exclude=("docs", "tests")),
    long_description=(README + "\n" + CHANGES),
    long_description_content_type='text/markdown',
    # requirements
    python_requires=">=3.5",
    install_requires=open("requirements.txt").readlines(),
    extras_require={"develop": ["nose"],},
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
