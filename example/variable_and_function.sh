#!/usr/bin/env bash

# directory setting
readonly CONF_DIR=./conf
readonly LIST_DIR=./list
readonly SRC_DIR=./src
readonly DATA_DIR=./data
readonly PAIR_DIR=$DATA_DIR/pair/$ORG-$TAR

function create_configure() {
    local _target_yml=$1
    local _defaul_yml=$2
    if [ -e $_target_yml ]; then
        echo "$_target_yml already exists."
    else
        echo "generate $_target_yml"
        cp $_defaul_yml $_target_yml
    fi
    return 0
}

function create_list() {
    local _target_list=$1
    local _wav_dir=$2
    if [ -e $_target_list ] ; then
        echo "$_target_list already exists."
    else
        echo "generate $_target_list"
        local wavfiles=(`ls $_wav_dir | grep -e "wav$"`)
        if [ ${#wavfiles[@]} == 0 ] ; then
            echo "wav files do not exist in $_wav_dir"
            return 1
        fi
        for _wav_file in ${wavfiles[@]} ;
        do
            echo `basename $_wav_dir`/${_wav_file%.*} >> $_target_list
        done
    fi
    return 0
}

function isexist_file() {
    local _target_file=$1
    if [ ! -e $_target_file ] ; then
        echo "ERROR: $_target_file does not exist."
        return 1
    else
        return 0
    fi
}

function check_list_length() {
    local _org_list=$1
    local _tar_list=$2

    local _org_len=(`cat $_org_list | wc -w`)
    local _tar_len=(`cat $_tar_list | wc -w`)
    if [ ! ${_org_len} == ${_tar_len} ] ; then
        echo "ERROR: lengths of following list files are different."
        echo "$_org_list, $_tar_list"
        return 1
    fi
    return 0
}
