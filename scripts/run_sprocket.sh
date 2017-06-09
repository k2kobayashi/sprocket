#! /bin/sh
#
# run_sprocket.sh
# Copyright (C) 2017 Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
# Distributed under terms of the MIT license.
#

# arguments setting
org=clb
tar=slt

# directory setting
src_dir=./src
data_dir=./data
conf_dir=./configure
wav_dir=$data_dir/speaker/wav

# parameter setting
nproc=1

if [ 0 -eq 1 ] ; then
    cp $conf_dir/default/speaker_default.yml $conf_dir/$org.yml
    cp $conf_dir/default/speaker_default.yml $conf_dir/$tar.yml
    cp $conf_dir/default/pair_default.yml $conf_dir/$org-$tar.yml
fi

if [ 0 -eq 1 ] ; then
    # Initialization for speakers
    for spkr in $org $tar; do
        python $src_dir/init_spkr.py \
            -m $nproc \
            $spkr \
            $wav_dir \
            $conf_dir
    done
fi

if [ 1 -eq 1 ] ; then
    # Feature extraction
    for spkr in $org $tar; do
        python $src_dir/feature_extraction.py \
            -m $nproc \
            $spkr \
            $conf_dir/$spkr.yml \
            $wav_dir
    done
fi

if [ 0 -eq 1 ] ; then
    # Initilization for the speaker pair
    python $src_dir/init_pair.py \
        $org \
        $tar \
        $wav_dir \
        $conf_dir
fi

if [ 0 -eq 1 ] ; then
    # calculate speaker-dependent statistics such as F0, GV and MS
    python $src_dir/estimate_feature_stats.py \
        $org \
        $tar \
        $wav_dir \
        $conf_dir
fi

# Joint feature extraction
if [ 0 -eq 1 ] ; then
    # estimate time-aligned joint feature vector of source and target
    python $src_dir/estimate_jnt.py \
        $org \
        $tar \
        $wav_dir \
        $conf_dir
fi

# GMM train
if [ 0 -eq 1 ] ; then
    # estimate GMM parameter from joint feature vector
    python $src_dir/train_GMM.py \
        $org \
        $tar \
        $wav_dir \
        $conf_dir
fi

# Conversion based on GMM
if [ 0 -eq 1 ] ; then
    # convertsion based on the trained GMM
    python $src_dir/gmmmap.py \
        $org \
        $tar \
        $wav_dir \
        $conf_dir
fi
