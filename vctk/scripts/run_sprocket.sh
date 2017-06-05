#! /bin/sh
#
# run_sprocket.sh
# Copyright (C) 2017 Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
# Distributed under terms of the MIT license.
#

org=$1
tar=$2

src_dir=./src

# Initialization
python $src_dir/init_spkr.py $org
python $src_dir/init_spkr.py $tar
python $src_dir/init_pair.py $org $tar

# Feature extraction
python $src_dir/feature_extraction.py $org
python $src_dir/feature_extraction.py $tar

# Joint feature extraction

# Train

# Conversion
