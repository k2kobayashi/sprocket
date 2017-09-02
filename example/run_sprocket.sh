#!/usr/bin/env bash

# speaker setting
readonly ORG=$1
readonly TAR=$2

. variable_and_function.sh

# flag settings
readonly STEP1=1 # extract acoustic feature
readonly STEP2=1 # estimate acoustic feature statistics
readonly STEP3=1 # estimate time warping function and joint feature vector
readonly STEP4=1 # train GMM
readonly STEP5=1 # convert based on the trained GMM

# check list file
isexist_file $LIST_DIR/${ORG}_train.list || exit $?
isexist_file $LIST_DIR/${ORG}_eval.list || exit $?
isexist_file $LIST_DIR/${TAR}_train.list || exit $?
isexist_file $LIST_DIR/${TAR}_eval.list || exit $?
check_list_length $LIST_DIR/${ORG}_train.list $LIST_DIR/${TAR}_train.list || exit $?
check_list_length $LIST_DIR/${ORG}_eval.list $LIST_DIR/${TAR}_eval.list || exit $?

# check YAML file
isexist_file $CONF_DIR/speaker/${ORG}.yml || exit $?
isexist_file $CONF_DIR/speaker/${TAR}.yml || exit $?

mkdir -p $PAIR_DIR
if [ $STEP1 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 1. Extract acoustic features                           ###"
    echo "##############################################################"
    # Extract acoustic features consisting of F0, spc, ap, mcep, npow
    for _speaker in $ORG $TAR; do
        for _flag in train eval; do
            ymlf=$CONF_DIR/speaker/${_speaker}.yml
            listf=$LIST_DIR/${_speaker}_${_flag}.list
            python $SRC_DIR/extract_features.py \
                # --overwrite \
                $_speaker \
                $ymlf \
                $listf \
                $DATA_DIR/wav \
                $PAIR_DIR
        done
    done
fi

if [ $STEP2 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 2. Estimate acoustic feature statistics                ###"
    echo "##############################################################"
    # Estimate speaker-dependent statistics for F0 and mcep
    for _speaker in $ORG $TAR; do
        listf=$LIST_DIR/${_speaker}_train.list
        python $SRC_DIR/estimate_feature_statistics.py \
            $_speaker \
            $listf \
            $PAIR_DIR
    done
fi

# Joint feature extraction
if [ $STEP3 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 3. Estimate time warping function and jnt              ###"
    echo "##############################################################"
    # Estimate a time-aligned joint feature vector of source and target
    python $SRC_DIR/estimate_twf_and_jnt.py \
        $CONF_DIR/pair/$ORG-$TAR.yml \
        $LIST_DIR/${ORG}_train.list \
        $LIST_DIR/${TAR}_train.list \
        $PAIR_DIR
fi

# GMM train
if [ $STEP4 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 4. Train GMM                                           ###"
    echo "##############################################################"
    # estimate GMM parameter using the joint feature vector
    python $SRC_DIR/train_GMM.py \
        $CONF_DIR/pair/$ORG-$TAR.yml \
        $PAIR_DIR
fi

# Conversion based on GMM
if [ $STEP5 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 5. Conversion based on the trained models              ###"
    echo "##############################################################"
    eval_list_file=$LIST_DIR/${ORG}_eval.list
    # convertsion based on the trained GMM
    python $SRC_DIR/convert.py \
        $ORG \
        $TAR \
        $CONF_DIR/speaker/$ORG.yml \
        $CONF_DIR/pair/$ORG-$TAR.yml \
        $eval_list_file \
        $DATA_DIR/wav \
        $PAIR_DIR
    python $SRC_DIR/convert.py \
        -gmmmode diff \
        $ORG \
        $TAR \
        $CONF_DIR/speaker/$ORG.yml \
        $CONF_DIR/pair/$ORG-$TAR.yml \
        $eval_list_file \
        $DATA_DIR/wav \
        $PAIR_DIR
fi
