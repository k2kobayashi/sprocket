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
import numpy as np

from sprocket.util.yml import PairYML
from sprocket.util.hdf5 import HDF5
from sprocket.util.distance import melcd
from sprocket.util.delta import delta
from sprocket.util.extfrm import extfrm

import plotly
import plotly.graph_objs as go


def plottestfunc(mcdmatrix):
    trace = go.Heatmap(
        z=mcdmatrix,
        x=range(mcdmatrix.shape[0]),
        y=range(mcdmatrix.shape[1]),
        colorscale='Electric',
    )
    layout = go.Layout(
    )
    data = [trace]

    fig = go.Figure(data=data, layout=layout)
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

    conf = PairYML(args.pair_ymlf)

    # read tr list
    print(conf.trlist)
    with open(conf.trlist, 'r') as tr:
        for line in tr:
            line = line.rstrip().split(" ")
            print(line[0], line[1])

    # open mcep
    orgh5 = HDF5('./data/speaker/h5/' + line[0] + '.h5', mode="r")
    tarh5 = HDF5('./data/speaker/h5/' + line[1] + '.h5', mode="r")

    orgmcep = orgh5.read('mcep')
    tarmcep = tarh5.read('mcep')

    # mel-cd test
    orglen = orgmcep.shape[0]
    tarlen = tarmcep.shape[0]

    mcdmatrix = np.zeros((orglen, tarlen))
    for ot in range(orglen):
        for tt in range(tarlen):
            mcdmatrix[ot, tt] = melcd(orgmcep[ot], tarmcep[tt])
            if mcdmatrix[ot, tt] > 30:
                mcdmatrix[ot, tt] = 30

    # plot test
    plotfrag = False
    if plotfrag is True:
        print("out mcd heatmap")
        plottestfunc(mcdmatrix)

    # calculate delta
    orgsdmcep = np.c_[orgmcep, delta(orgmcep)]
    tarsdmcep = np.c_[tarmcep, delta(tarmcep)]

    # calculate extfrm
    extorgmcep = extfrm(orgh5.read('npow'), orgsdmcep)
    exttarmcep = extfrm(tarh5.read('npow'), tarsdmcep)

    # dtw to estimate twf

    return

if __name__ == '__main__':
    main()
