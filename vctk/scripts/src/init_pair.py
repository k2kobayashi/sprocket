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
    parser.add_argument('-m', '--multicore', type=int, default=1,
                        help='# of cores for multi-processing')
    parser.add_argument('org', type=str,
                        help='original speaker label')
    parser.add_argument('tar', type=str,
                        help='target speaker label')
    parser.add_argument('wav_dir', type=str,
                        help='wav file directory of the speaker')
    parser.add_argument('conf_dir', type=str,
                        help='configure directory of the speaker')
    args = parser.parse_args()

    # create list file for training and evaluation
    orgfiles = glob.glob(args.wav_dir + '/' + args.org + '/*.wav')
    tarfiles = glob.glob(args.wav_dir + '/' + args.tar + '/*.wav')
    assert len(orgfiles) == len(tarfiles)
    assert len(orgfiles) != 0 or len (tarfiles) != 0

    listfile = args.conf_dir + '/' + args.org + '-' + args.tar
    tf = open(listfile + '_tr.list', 'w')
    ef = open(listfile + '_ev.list', 'w')
    for (owav, twav) in zip(orgfiles, tarfiles):
        olbl, _ = os.path.splitext(args.org + '/' + os.path.basename(owav))
        tlbl, _ = os.path.splitext(args.tar + '/' + os.path.basename(twav))
        tf.write(olbl + ' ' + tlbl + '\n')
        ef.write(olbl + '\n')
    tf.close()
    ef.close()

if __name__ == '__main__':
    main()
