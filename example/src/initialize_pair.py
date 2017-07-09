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
create training and evaluation list files

"""

import os
import sys
import glob
import argparse


def main():
    # Options for python
    description = 'create training and evaluation list file (pair.yml)'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('org', type=str,
                        help='original speaker label')
    parser.add_argument('tar', type=str,
                        help='target speaker label')
    parser.add_argument('wav_dir', type=str,
                        help='wav file directory')
    parser.add_argument('pair_dir', type=str,
                        help='data directory for the speaker pair')
    args = parser.parse_args()

    # create list file for training and evaluation
    orgfiles = glob.glob(args.wav_dir + '/' + args.org + '/*.wav')
    tarfiles = glob.glob(args.wav_dir + '/' + args.tar + '/*.wav')
    assert len(orgfiles) == len(tarfiles)
    assert len(orgfiles) != 0 or len(tarfiles) != 0

    listfile = args.pair_dir + '/' + args.org + '-' + args.tar
    trlist = listfile + '_tr.list'
    evlist = listfile + '_ev.list'

    # existing training and eve file check
    if os.path.exists(trlist) and os.path.exists(evlist):
        print ("List files are already excisted.")
        sys.exit(0)

    # open files and write wave file path
    tf = open(trlist, 'w')
    ef = open(evlist, 'w')
    for (owav, twav) in zip(orgfiles, tarfiles):
        olbl, _ = os.path.splitext(args.org + '/' + os.path.basename(owav))
        tlbl, _ = os.path.splitext(args.tar + '/' + os.path.basename(twav))
        tf.write(olbl + ' ' + tlbl + '\n')
        ef.write(olbl + '\n')
    tf.close()
    ef.close()

if __name__ == '__main__':
    main()
