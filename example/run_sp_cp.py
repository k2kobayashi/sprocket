#!/usr/bin/env python3

"""An example script to run sprocket.

Usage: run_sprocket.py [-h] [-1] [-2] [-3] [-4] [-5] SOURCE TARGET

Options:
    -h, --help   Show the help
    -1, --step1  Execute step1 (Extraction of acoustic features)
    -2, --step2  Execute step2 (Estimation of acoustic feature statistics)
    -3, --step3  Execute step3 (Estimation of time warping function and jnt)
    -4, --step4  Execute step4 (Training of GMM)
    -5, --step5  Execute step5 (Conversion based on the trained models)
    SOURCE         The name of speaker
                   whose voice you would like to convert from
    TARGET         The name of speaker whose voice you would like to convert to

Note:
    All steps are executed if no options from -1 to -5 are given.
"""

import os
import sys
from pathlib import Path

import docopt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))  # isort:skip
from src import (convert_cp, estimate_feature_statistics, estimate_twf_and_jnt,  # isort:skip # pylint: disable=C0413
                 extract_features, extract_features2,  train_GMM)


def list_lengths_are_all_same(first_path, *remain_paths):
    """Check whether the lengths of list files are all same.

    Parameters
    ----------
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
        ----------
        path : str or path-like object
            The path of the file the number of words of which you want to know.

        Returns
        -------
        n_words : int
            The number of words in the file whose path is `path`.
        """
        with open(str(path)) as handler:
            words = len(
                handler.read().split())  # when space in path, bug appears
        return words

    n_words_in_first_file = count_words_in_file(first_path)
    return all((count_words_in_file(path) == n_words_in_first_file
                for path in remain_paths))


USES = ("train", "eval")
LIST_SUFFIXES = {
    use: "_" + use + ".list" for use in USES}

EXAMPLE_ROOT_DIR = Path(__file__).parent
CONF_DIR = EXAMPLE_ROOT_DIR / "conf"
DATA_DIR = EXAMPLE_ROOT_DIR / "data"
LIST_DIR = EXAMPLE_ROOT_DIR / "list"
WAV_DIR = DATA_DIR / "wav"

if __name__ == "__main__":
    args = docopt.docopt(__doc__)  # pylint: disable=invalid-name

    LABELS = {label: args[label.upper()] for label in ("source", "target")}
    SOURCE_TARGET_PAIR = LABELS["source"] + "-" + LABELS["target"]
    PAIR_DIR = DATA_DIR / "pair" / SOURCE_TARGET_PAIR
    LIST_FILES = {
        speaker_part: {
            use: LIST_DIR / (speaker_label + LIST_SUFFIXES[use])
            for use in USES}
        for speaker_part, speaker_label in LABELS.items()}
    SPEAKER_CONF_FILES = {
        part:
            CONF_DIR / "speaker" / (label + ".yml")
        for part, label in LABELS.items()}
    PAIR_CONF_FILE = CONF_DIR / "pair" / (SOURCE_TARGET_PAIR + ".yml")

    # The first False is dummy for alignment
    #   between indexes of `args_execute_steps` and arguments
    # pylint: disable=invalid-name
    execute_steps = [False] \
        + [args["--step{}".format(step_index)] for step_index in range(1, 6)]

    # Execute all steps if no options on steps are given
    # Keep index #0 False in case you create a hotbed for bugs.
    if not any(execute_steps):
        execute_steps[1:] = [True] * (len(execute_steps) - 1)

    # Check the lengchs of list files for each use (training / evaluation)
    for use in USES:
        list_lengths_are_all_same(
            *[list_files_per_part[use]
              for list_files_per_part in LIST_FILES.values()])

    os.makedirs(str(PAIR_DIR), exist_ok=True)

    if execute_steps[1]:
        print("### 1. Extract acoustic features ###")
        # Extract acoustic features consisting of F0, spc, ap, mcep, npow
        for speaker_part, speaker_label in LABELS.items():
            if speaker_part == "source":
                print('source')
#               extract_features.main(
#                   speaker_label, str(SPEAKER_CONF_FILES[speaker_part]),
#                   str(LIST_FILES[speaker_part]['train']),
#                   str(WAV_DIR), str(PAIR_DIR))
            elif speaker_part == "target":
                print('target')
                extract_features2.main(
                    speaker_label, str(SPEAKER_CONF_FILES[speaker_part]),
                    str(LIST_FILES[speaker_part]['train']),
                    str(WAV_DIR), str(PAIR_DIR), str(LIST_FILES["source"]['train']))
    if execute_steps[2]:
        print("### 2. Estimate acoustic feature statistics ###")
        # Estimate speaker-dependent statistics for F0 and mcep
        for speaker_part, speaker_label in LABELS.items():
            estimate_feature_statistics.main(
                speaker_label, str(LIST_FILES[speaker_part]["train"]),
                str(PAIR_DIR))

    if execute_steps[3]:
        print("### 3. Estimate time warping function and jnt ###")
        estimate_twf_and_jnt.main(
            str(SPEAKER_CONF_FILES["source"]),
            str(SPEAKER_CONF_FILES["target"]),
            str(PAIR_CONF_FILE),
            str(LIST_FILES["source"]["train"]),
            str(LIST_FILES["target"]["train"]),
            str(PAIR_DIR))

    if execute_steps[4]:
        print("### 4. Train GMM and converted GV ###")
        # estimate GMM parameter using the joint feature vector
        train_GMM.main(
            str(LIST_FILES["source"]["train"]),
            str(PAIR_CONF_FILE),
            str(PAIR_DIR))

    if execute_steps[5]:
        print("### 5. Conversion based on the trained models ###")
        EVAL_LIST_FILE = LIST_FILES["source"]["eval"]
        # convertsion based on the trained GMM
        # not need VC
        #convert_cp.main(
        #    LABELS["source"], LABELS["target"],
        #    str(SPEAKER_CONF_FILES["source"]),
        #    str(PAIR_CONF_FILE),
        #    str(EVAL_LIST_FILE),
        #    str(WAV_DIR),
        #    str(PAIR_DIR))
        convert_cp.main(
            "-gmmmode", "diff",
            LABELS["source"], LABELS["target"],
            str(SPEAKER_CONF_FILES["source"]),
            str(PAIR_CONF_FILE),
            str(EVAL_LIST_FILE),
            str(WAV_DIR),
            str(PAIR_DIR))
