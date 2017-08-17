#!/usr/bin/env bash

# speaker setting
readonly ORG=$1
readonly TAR=$2

. variable_and_function.sh

# check list file
check_list_length $LIST_DIR/${ORG}_train.list $LIST_DIR/${TAR}_train.list || exit $?
check_list_length $LIST_DIR/${ORG}_eval.list $LIST_DIR/${TAR}_eval.list || exit $?

# check YAML file
isexist_file $CONF_DIR/speaker/${ORG}.yml || exit $?
isexist_file $CONF_DIR/speaker/${TAR}.yml || exit $?

echo "##############################################################"
echo "### 1. F0 transformation of original waveform              ###"
echo "##############################################################"
# transform F0 of waveform of original speaker
python $SRC_DIR/f0_transformation.py \
    $ORG \
    $CONF_DIR/speaker/${ORG}.yml \
    $CONF_DIR/speaker/${TAR}.yml \
    $LIST_DIR/${ORG}_train.list \
    $LIST_DIR/${ORG}_eval.list \
    $LIST_DIR/${TAR}_train.list \
    $DATA_DIR/wav
echo "# F0 transformed waveforms are generated #"
