# -*- coding: utf-8 -*-

import os
import yaml

class SpeakerYML(object):

    def __init__(self, ymlf):
        # open yml file
        with open(ymlf) as yf:
            conf = yaml.safe_load(yf)

        # read parameter from yml file
        self.fs = conf['wav']['fs']
        self.shiftms = conf['wav']['shiftms']
        self.fftl = conf['wav']['fftl']
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

        self.h5dir = conf['dir']['h5']
        self.wavdir = conf['dir']['wav']
        self.pairdir = conf['dir']['pair']

        self.n_jntiter = conf['jnt']['n_iter']

        self.n_mix = conf['GMM']['n_mix']
        self.n_iter = conf['GMM']['n_iter']
        self.covtype = conf['GMM']['covtype']

        self._read_training_list()
        self._read_evaluation_list()

    def _read_training_list(self):
        if not os.path.exists(self.trlist):
            raise('training file list does not exists.')
        # read training list
        self.trfiles = []
        with open(self.trlist, 'r') as f:
            for line in f:
                self.trfiles.append(line.rstrip().split(" "))

    def _read_evaluation_list(self):
        if not os.path.exists(self.evlist):
            raise('evaluation file list does not exists.')
        self.evfiles = []
        with open(self.evlist, 'r') as f:
            for line in f:
                self.evfiles.append(line.rstrip())

    def print_params(self):
        pass

