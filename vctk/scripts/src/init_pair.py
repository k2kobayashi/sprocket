#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# init_pair.py
#   First ver.: 2017-06-05
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
create speaker pair-dependent configure file (org-tar.yml)

"""

import os
import glob
import argparse


def main():
    # Options for python
    description = 'create speaker-dependent configure file (spkr.yml)'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-nmsg', '--nmsg', default=True, action='store_false',
                        help='print no message')
    parser.add_argument('-m', '--multicore', type=int, default=1,
                        help='# of cores for multi-processing')
    parser.add_argument('org', type=str,
                        help='original speaker label')
    parser.add_argument('tar', type=str,
                        help='target speaker label')
    parser.add_argument('conf_dir', type=str,
                        help='configure directory of the speaker')
    args = parser.parse_args()

    # create list file for training

    # create list file for evaluation

    pass


if __name__ == '__main__':
    main()
