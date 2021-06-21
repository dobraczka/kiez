# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer

import logging
from tempfile import NamedTemporaryFile, mkstemp

__all__ = ["create_tempfile_preferably_in_dir"]


def create_tempfile_preferably_in_dir(
    suffix: str = None,
    prefix: str = None,
    directory: str = None,
    persistent: bool = False,
):
    """Create a temporary file with precedence for directory if possible, in TMP otherwise.
    For example, this is useful to try to save into /dev/shm.

    Parameters
    ---------
    suffix: str
        suffix of tempfile
    prefix: str
        prefix of tempfile
    directory: str
        directory where tempfile should preferably be created
    persistent: bool
        If True create a persistent file

    Returns
    -------
    path
        string path of tempfile
    """
    temp_file = mkstemp if persistent else NamedTemporaryFile
    try:
        handle = temp_file(suffix=suffix, prefix=prefix, dir=directory)
        warn = False
    except FileNotFoundError:
        handle = temp_file(suffix=suffix, prefix=prefix, dir=None)
        warn = True

    # Extract the path (as string)
    try:
        path = handle.name
    except AttributeError:
        _, path = handle

    if warn:
        logging.warning(
            f"Could not create temp file in {directory}. Instead, the path is {path}."
        )
    return path
