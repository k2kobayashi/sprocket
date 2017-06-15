#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# estimate_jnt.py
#   First ver.: 2017-06-09
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
estimate joint feature vector of the speaker pair using GMM

"""

import argparse

from sprocket.util.jnt import JointFeatureExtractor

import plotly
import plotly.graph_objs as go


def plottestfunc(mcdmatrix, path=None):
    hm = dict(
        z=mcdmatrix,
        x=range(mcdmatrix.shape[0]),
        y=range(mcdmatrix.shape[1]),
        colorscale='Electric',
        type='heatmap'
    )

    if path is not None:
        pt = dict(
            x = path[1],
            y = path[0],
        )

    data = [hm, pt]

    fig = go.Figure(data=data)
    plotly.offline.plot(fig, filename='mcdmatrix')
    return


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('org', type=str,
                        help='original speaker label')
    parser.add_argument('tar', type=str,
                        help='target speaker label')
    parser.add_argument('pair_ymlf', type=str,
                        help='yml file for the speaker pair')
    args = parser.parse_args()

    # joint feature extraction
    jnt = JointFeatureExtractor(args.pair_ymlf)
    jnt.estimate()

    # dtw to estimate twf

    return

if __name__ == '__main__':
    main()
