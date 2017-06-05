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

# parameter setting
nproc=3

# Initialization
# python $src_dir/init_spkr.py -m $nproc $org ./data/wav/$org ./configure
# python $src_dir/init_spkr.py -m $nproc $tar ./data/wav/$tar ./configure
python $src_dir/init_pair.py $org $tar

# # Feature extraction
# python $src_dir/feature_extraction.py $org
# python $src_dir/feature_extraction.py $tar

# Joint feature extraction

# Train

# Conversion
