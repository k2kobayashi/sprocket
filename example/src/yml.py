# -*- coding: utf-8 -*-

import os

import yaml


class SpeakerYML(object):

    def __init__(self, ymlf):
        # open yml file
        with open(ymlf) as yf:
            conf = yaml.safe_load(yf)

        # read parameter from yml file
        self.wav_fs = int(conf['wav']['fs'])
        self.wav_bit = int(conf['wav']['bit'])
        self.wav_fftl = int(conf['wav']['fftl'])
        self.wav_shiftms = float(conf['wav']['shiftms'])

        self.f0_minf0 = float(conf['f0']['minf0'])
        self.f0_maxf0 = float(conf['f0']['maxf0'])
        assert self.f0_minf0 < self.f0_maxf0, \
            "should be minf0 < maxf0 in yml file"

        self.mcep_dim = int(conf['mcep']['dim'])
        self.mcep_alpha = float(conf['mcep']['alpha'])
        self.power_threshold = float(conf['power']['threshold'])

        self.analyzer = conf['analyzer']

    def print_params(self):
        pass


class PairYML(object):

    def __init__(self, ymlf):
        # open yml file
        with open(ymlf) as yf:
            conf = yaml.safe_load(yf)

        self.jnt_n_iter = int(conf['jnt']['n_iter'])

        self.GMM_mcep_n_mix = int(conf['GMM']['mcep']['n_mix'])
        self.GMM_mcep_n_iter = int(conf['GMM']['mcep']['n_iter'])
        self.GMM_mcep_covtype = str(conf['GMM']['mcep']['covtype'])
        self.GMM_mcep_cvtype = str(conf['GMM']['mcep']['cvtype'])

        self.GMM_codeap_n_mix = int(conf['GMM']['codeap']['n_mix'])
        self.GMM_codeap_n_iter = int(conf['GMM']['codeap']['n_iter'])
        self.GMM_codeap_covtype = str(conf['GMM']['codeap']['covtype'])
        self.GMM_codeap_cvtype = str(conf['GMM']['codeap']['cvtype'])

        self.GV_morph_coeff = float(conf['GV']['morph_coeff'])

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
