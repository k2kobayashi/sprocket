# -*- coding: utf-8 -*-

import os

import yaml


class SpeakerYML(object):

    def __init__(self, ymlf):
        # open yml file
        with open(ymlf) as yf:
            conf = yaml.safe_load(yf)

        # read parameter from yml file
        self.wav_fs = conf['wav']['fs']
        self.wav_bit = conf['wav']['bit']
        self.wav_fftl = conf['wav']['fftl']
        self.wav_shiftms = conf['wav']['shiftms']

        self.f0_minf0 = conf['f0']['minf0']
        self.f0_maxf0 = conf['f0']['maxf0']
        assert self.f0_minf0 < self.f0_maxf0, \
            "should be minf0 < maxf0 in yml file"

        self.mcep_dim = conf['mcep']['dim']
        self.mcep_alpha = conf['mcep']['alpha']
        self.power_threshold = conf['power']['threshold']

        self.analyzer = conf['analyzer']

    def print_params(self):
        pass


class PairYML(object):

    def __init__(self, ymlf):
        # open yml file
        with open(ymlf) as yf:
            conf = yaml.safe_load(yf)

        self.jnt_n_iter = conf['jnt']['n_iter']

        self.GMM_mcep_n_mix = conf['GMM']['mcep']['n_mix']
        self.GMM_mcep_n_iter = conf['GMM']['mcep']['n_iter']
        self.GMM_mcep_covtype = conf['GMM']['mcep']['covtype']
        self.GMM_mcep_cvtype = conf['GMM']['mcep']['cvtype']

        self.GMM_codeap_n_mix = conf['GMM']['codeap']['n_mix']
        self.GMM_codeap_n_iter = conf['GMM']['codeap']['n_iter']
        self.GMM_codeap_covtype = conf['GMM']['codeap']['covtype']
        self.GMM_codeap_cvtype = conf['GMM']['codeap']['cvtype']

        self.GV_morph_coeff = conf['GV']['morph_coeff']

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
