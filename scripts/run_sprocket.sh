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

if [ 1 ] ; then
    cp $conf_dir/default/speaker_default.yml $conf_dir/$org.yml
    cp $conf_dir/default/speaker_default.yml $conf_dir/$tar.yml
    cp $conf_dir/default/pair_default.yml $conf_dir/$org-$tar.yml
fi

if [ 1 ] ; then
    # Initialization for speakers
    for spkr in $org $tar; do
        python $src_dir/init_spkr.py \
            -m $nproc \
            $spkr \
            $wav_dir \
            $conf_dir
    done
fi

if [ 1 ] ; then
    # Feature extraction
    for spkr in $org $tar; do
        python $src_dir/feature_extraction.py \
            -m $nproc \
            $spkr \
            $conf_dir/$spkr.yml \
            $wav_dir
    done
fi

if [ 1 ] ; then
    # Initilization for the speaker pair
    python $src_dir/init_pair.py \
        $org \
        $tar \
        $wav_dir \
        $conf_dir
fi

# Joint feature extraction

# Train

# Conversion
