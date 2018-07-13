#!/usr/bin/env python3

"""Downloader of sample audio data of The Voice Conversion Challenge (VCC) 2016

About data:
    http://datashare.is.ed.ac.uk/handle/10283/2211
Article of VCC 2016:
    http://www.cstr.ed.ac.uk/downloads/publications/2016/toda2016voice.pdf
License of VCC 2016 dataset :
    http://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/license_text?sequence=10&isAllowed=y

Usage:
    download_vcc2016data.py [-h] [-q] [-f] [-m]

Arguments:
    -h, --help      Show this help and exit
    -q, --quiet     Don't show any messages about progress
    -f, --force     Overwrite existing directories and files
    -m, --minimal   Exit when there are some directories in data/wav
"""

import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from sys import stderr
from tempfile import TemporaryDirectory

from docopt import docopt


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
        elif os.path.isdir(str(dest)):  # wrapping in str is for Python 3.5
            dest = type(dest)(os.path.join(str(dest), real_file_name))
        with open(str(dest), "wb") as file_obj:
            shutil.copyfileobj(request_obj, file_obj)
    return dest


if __name__ == "__main__":
    print(
        "Warning: This script was replaced with download_corpus_data.py and"
        " are going to be removed in the future.  Use it instead.",
        file=stderr)
    args = docopt(__doc__)
    is_verbose = not args["--quiet"]  # Whether prints regular messages
    does_by_force = args["--force"]
    does_minimally = args["--minimal"]

    base_dir = Path(__file__).parent
    wav_root_dir = base_dir / "data" / "wav"

    if does_minimally and list(filter(Path.is_dir, wav_root_dir.iterdir())):
        if is_verbose:
            print("There are some directories in:", wav_root_dir)
        exit(0)

    with TemporaryDirectory() as working_dir:
        working_dir = Path(working_dir)
        if is_verbose:
            print("Downloading the evaluation data...")
        eval_archive_path = download(
            "http://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/"
            "evaluation_all.zip?sequence=7&isAllowed=y", working_dir)
        if is_verbose:
            print("Unpack:", eval_archive_path)
        shutil.unpack_archive(str(eval_archive_path), str(working_dir))
        for directory in filter(
                Path.is_dir, (working_dir / eval_archive_path.stem).iterdir()):
            if is_verbose:
                print("Move:", directory)
            dest_dir = wav_root_dir / directory.name
            os.makedirs(str(dest_dir), exist_ok=does_by_force)
            for wav_file in directory.glob("*.wav"):
                dest_path = dest_dir / wav_file.name
                if dest_path.exists() and does_by_force:
                    dest_path.unlink()
                shutil.move(str(wav_file), str(dest_path))

        if is_verbose:
            print("Downloading the training data...")
        train_archive_path = download(
            "http://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/"
            "vcc2016_training.zip?sequence=8&isAllowed=y", working_dir)
        if is_verbose:
            print("Unpack:", train_archive_path)
        shutil.unpack_archive(str(train_archive_path), str(working_dir))
        for directory in filter(
                Path.is_dir, (working_dir / train_archive_path.stem).iterdir()):
            if is_verbose:
                print("Move:", directory)
            dest_dir = wav_root_dir / directory.name
            os.makedirs(str(dest_dir), exist_ok = True)
            for wav_file in directory.glob("*.wav"):
                dest_path=dest_dir / wav_file.name
                if dest_path.exists() and does_by_force:
                    dest_path.unlink()
                shutil.move(str(wav_file), str(dest_path))
