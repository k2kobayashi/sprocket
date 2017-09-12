#!/usr/bin/env python

"""Downloader of sample audio data of The Voice Conversion Challenge (VCC) 2016

About data:
    http://datashare.is.ed.ac.uk/handle/10283/2211
Article of VCC 2016:
    http://www.cstr.ed.ac.uk/downloads/publications/2016/toda2016voice.pdf

Usage:
    download_sample_data.py [-h] [-q] [-f] [-m]

Arguments:
    -h, --help      Show this help and exit
    -q, --quiet     Don't show any messages about progress
    -f, --force     Overwrite existing directories and files
    -m, --minimal   Exit when there are some directories in data/wav
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import shutil
import sys
from glob import iglob
from tempfile import TemporaryDirectory

from docopt import docopt
from six.moves import urllib, filter


def download(url, dest=None):
    """Download and store a remote file.

    Parameters
    ----------
    url : str or path-like
        The URL of the remote file.
    dest : str or path-like or None
        The path where the downloaded file is stored.
        if an existing directory is designated

    Returns
    -------
    The path of the stored file.

    Raises
    ------
    urllib.error.HTTPError
        When `the status code is not 200 or 30*.
    """
    with urllib.request.urlopen(url) as request_obj:
        real_file_name = os.path.basename(
            urllib.parse.urlparse(request_obj.geturl()).path)
        if dest is None:
            dest = real_file_name
        elif os.path.isdir(dest):
            dest = os.path.join(dest, real_file_name)
        with open(dest, "wb") as file_obj:
            shutil.copyfileobj(request_obj, file_obj)
    return dest


def iterdir(directory="."):
    """Iterate the directories.

    Parameters
    ----------
    directory : str or path-like
        The path of the directory

    Returns
    -------
    Generator of the paths of files and directories in the `directory`

    Examples
    --------
    >>> sorted(iterdir("/bin"))[:4]
    ['/bin/bash', '/bin/bunzip2', '/bin/busybox', '/bin/bzcat']
    """
    return (os.path.join(directory, path) for path in os.listdir(directory))


def stem(path):
    """Returns the part of the path of the given file without the extension and
    the path of the directory where it is in.

    Parameters
    ----------
    path : str or path-like
        The path of the file

    Returns
    -------
    The file name without the extension

    Examples
    --------
    >>> stem("/etc/ldap.conf")
    'ldap'
    """
    return os.path.splitext(os.path.basename(path))[0]


if __name__ == "__main__":
    args = docopt(__doc__)
    is_verbose = not args["--quiet"]  # Whether prints regular messages
    does_by_force = args["--force"]
    does_minimally = args["--minimal"]

    base_dir = os.path.dirname(sys.argv[0])
    wav_root_dir = os.path.join(base_dir, "data", "wav")

    if does_minimally and list(filter(os.path.isdir, iterdir(wav_root_dir))):
        if is_verbose:
            print("There are some directories in:", wav_root_dir)
        exit(0)

    with TemporaryDirectory() as working_dir:
        if is_verbose:
            print("Downloading the evaluation data...")
        eval_archive_path = download(
            "http://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/"
            "evaluation_all.zip?sequence=7&isAllowed=y", working_dir)
        if is_verbose:
            print("Unpack:", eval_archive_path)
        shutil.unpack_archive(eval_archive_path, working_dir)
        for directory in filter(
                os.path.isdir, iterdir(os.path.join(
                    working_dir, stem(eval_archive_path)))):
            if is_verbose:
                print("Move:", directory)
            dest_dir = os.path.join(wav_root_dir, os.path.basename(directory))
            os.makedirs(dest_dir, exist_ok=does_by_force)
            for wav_file in iglob(os.path.join(directory, "*.wav")):
                dest_path = os.path.join(dest_dir, os.path.basename(wav_file))
                if os.path.exists(dest_path) and does_by_force:
                    os.remove(dest_path)
                os.rename(wav_file, dest_path)

        if is_verbose:
            print("Download the training data...")
        train_archive_path = download(
            "http://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/"
            "vcc2016_training.zip?sequence=8&isAllowed=y", working_dir)
        if is_verbose:
            print("Unpack:", train_archive_path)
        shutil.unpack_archive(train_archive_path, working_dir)
        for directory in filter(
                os.path.isdir, iterdir(os.path.join(working_dir, stem(
                    train_archive_path)))):
            if is_verbose:
                print("Move:", directory)
            dest_dir = os.path.join(wav_root_dir, os.path.basename(directory))
            # exist_ok is always True because we have to append wav files in
            # existing directories
            os.makedirs(dest_dir, exist_ok=True)
            for wav_file in iglob(os.path.join(directory, "*.wav")):
                dest_path = os.path.join(dest_dir, os.path.basename(wav_file))
                if os.path.exists(dest_path) and does_by_force:
                    os.remove(dest_path)
                os.rename(wav_file, dest_path)
