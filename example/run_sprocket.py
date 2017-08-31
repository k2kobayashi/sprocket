#!/usr/bin/env python

"""An example script to run sprocket.

Usage: run_sprocket.py [-h] [-1] [-2] [-3] [-4] [-5] SOURCE TARGET

Options:
    -h, --help   Show the help
    -1, --step1  Execute step1 (Extraction of acoustic features)
    -2, --step2  Execute step2 (Estimation of acoustic feature statistics)
    -3, --step3  Execute step3 (Estimation of time warping function and jnt)
    -4, --step4  Execute step4 (Training of GMM)
    -5, --step5  Execute step5 (Conversion based on the trained models)

Note:
    Except for step1 are executed if no options about executing steps are not given.
"""

from __future__ import division  # , unicode_literals
from __future__ import absolute_import, print_function

import operator
import os
import shutil
import sys

import docopt
import six
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
try:
    from src import (convert, estimate_feature_statistics, estimate_twf_and_jnt,
                     extract_features, train_GMM)
except:
    raise


if six.PY2:
    # pylint: disable=unused-import, redefined-builtin, import-error, line-too-long
    from future_builtins import ascii, filter, hex, map, oct, zip
    # pylint: disable=unused-import, redefined-builtin, import-error, line-too-long
    from six.moves import range, input
if six.PY3:
    # pylint: disable=unused-import, redefined-builtin, import-error, line-too-long
    from six.moves import reduce


def list_lengths_are_all_same(first_path, *remain_paths):
    """Check whether the lengths of list files are all same.

    Parameters
    ---------
    first_path : str or path-like object
    remain_paths : list of str or path-like object

    Returns
    -------
    list_lengths_are_all_same : bool
        `True` if all of the numbers of the lengths of the given files are same.

    Notes
    ----
        The length of each file is the same as the Unix command `wc -w`.
    """

    def count_words_in_file(path):
        """Counts the number of words in a file.

        Parameters
        ---------
        path : str or path-like object
            The path of the file the number of words of which you want to know.

        Returns
        -------
        n_words : int
            The number of words in the file whose path is `path`.
        """
        with open(path) as handler:
            words = len(
                handler.read().split())  # when space in path, bug appears
        return words

    n_words_in_first_file = count_words_in_file(first_path)
    return all((count_words_in_file(path) == n_words_in_first_file
                for path in remain_paths))


LIST_EXTENSION = ".list"
USES = ("train", "eval")
LIST_SUFFIXES = {
    use: "_" + use + LIST_EXTENSION for use in USES}
YML_EXTENSION = ".yml"

EXAMPLE_ROOT_DIR = os.path.dirname(__file__)
CONF_DIR = os.path.join(EXAMPLE_ROOT_DIR, "conf")
DATA_DIR = os.path.join(EXAMPLE_ROOT_DIR, "data")
LIST_DIR = os.path.join(EXAMPLE_ROOT_DIR, "list")
WAV_DIR = os.path.join(DATA_DIR, "wav")

if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    LABELS = {label: args[label.upper()] for label in ("source", "target")}
    SOURCE_TARGET_PAIR = LABELS["source"] + "-" + LABELS["target"]
    PAIR_DIR = os.path.join(DATA_DIR, "pair",
                            SOURCE_TARGET_PAIR)
    LIST_FILES = {
        speaker_part: {
            use: os.path.join(LIST_DIR, speaker_label + LIST_SUFFIXES[use])
            for use in USES}
        for speaker_part, speaker_label in LABELS.items()}
    SPEAKER_CONF_FILES = {
        part: os.path.join(
            CONF_DIR, "speaker", label + YML_EXTENSION)
        for part, label in LABELS.items()}
    PAIR_CONF_FILE = os.path.join(
        CONF_DIR, "pair", SOURCE_TARGET_PAIR + YML_EXTENSION)

    # The first False is dummy for alignment
    #   between indexes of `args_execute_steps` and arguments
    args_execute_steps = [False] \
        + [args["--step{}".format(step_index)] for step_index in range(1, 6)]
    # The first False is also dummy because of the similar reason
    # Except for Step 1 are executed by default
    execute_steps = args_execute_steps if any(args_execute_steps[1:]) \
        else [False] * 2 + [True] * 4

    # Check the lengchs of list files for each use (training / evaluation)
    for use in USES:
        list_lengths_are_all_same(
            *[list_files_per_part[use]
              for list_files_per_part in LIST_FILES.values()])

    os.makedirs(PAIR_DIR, exist_ok=True)

    if execute_steps[1]:
        print("""\
##############################################################
### 1. Extract acoustic features                           ###
##############################################################""")
        # Extract acoustic features consisting of F0, spc, ap, mcep, npow
        for speaker_part, speaker_label in LABELS.items():
            for use in USES:
                extract_features.main(
                    speaker_label, SPEAKER_CONF_FILES[speaker_part],
                    LIST_FILES[speaker_part][use],
                    WAV_DIR, PAIR_DIR)
        exit

    if execute_steps[2]:
        print("""\
##############################################################
### 2. Estimate acoustic feature statistics                ###
##############################################################""")
        # Estimate speaker-dependent statistics for F0 and mcep
        for speaker_part, speaker_label in LABELS.items():
            LIST_FILE = os.path.join(
                LIST_DIR, "{}_train.list".format(speaker_label))
            estimate_feature_statistics.main(
                speaker_label, LIST_FILES[speaker_part]["train"],
                PAIR_DIR)

    if execute_steps[3]:
        print("""\
##############################################################
### 3. Estimate time warping function and jnt              ###
##############################################################""")
        estimate_twf_and_jnt.main(
            PAIR_CONF_FILE,
            LIST_FILES["source"]["train"],
            LIST_FILES["target"]["train"])

    if execute_steps[4]:
        print("""\
##############################################################
### 4. Train GMM                                           ###
##############################################################""")
        # estimate GMM parameter using the joint feature vector
        train_GMM.main(PAIR_CONF_FILE,
                       PAIR_DIR)

    if execute_steps[5]:
        print("""\
##############################################################
### 5. Conversion based on the trained models              ###
##############################################################""")
        EVAL_LIST_FILE = LIST_FILES["source"]["eval"]
        # convertsion based on the trained GMM
        convert.main(
            LABELS["source"], LABELS["target"],
            SPEAKER_CONF_FILES["source"],
            PAIR_CONF_FILE,
            EVAL_LIST_FILE,
            WAV_DIR)
        convert.main(
            "-gmmmode", "diff",
            LABELS["source"], LABELS["target"],
            SPEAKER_CONF_FILES["source"],
            PAIR_CONF_FILE,
            EVAL_LIST_FILE,
            WAV_DIR)
