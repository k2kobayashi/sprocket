#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# yml.py
#   First ver.: 2017-06-10
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
yml parser

"""

import yaml


class SpeakerYML(object):

    def __init__(self, ymlf):
        # open yml file
        with open(ymlf) as yf:
            conf = yaml.safe_load(yf)

        # read parameter from yml file
        self.fs = conf['wav']['fs']
        self.shiftms = conf['wav']['shiftms']
        self.minf0 = conf['f0']['minf0']
        self.maxf0 = conf['f0']['maxf0']
        self.dim = conf['mcep']['dim']
        self.alpha = conf['mcep']['alpha']
        self.analyzer = conf['analyzer']

    def print_params(self):
        pass


class PairYML(object):

    def __init__(self, ymlf):
        # open yml file
        with open(ymlf) as yf:
            conf = yaml.safe_load(yf)

        # read parameter from yml file
        self.trlist = conf['list']['trlist']
        self.evlist = conf['list']['evlist']

        self.n_mix = conf['GMM']['n_mix']
        self.n_iter = conf['GMM']['n_iter']

    def print_params(self):
        pass

def main():
    pass


if __name__ == '__main__':
    main()
