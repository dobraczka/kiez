# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# adapted from skhubness: https://github.com/VarIr/scikit-hubness/

from .base import NoHubnessReduction
from .csls import CSLS
from .dis_sim import DisSimLocal
from .local_scaling import LocalScaling
from .mutual_proximity import MutualProximity

#: Supported hubness reduction algorithms
hubness_algorithms = [
    "mp",
    "ls",
    "dsl",
    "csls",
]
hubness_algorithms_long = [
    "mutual_proximity",
    "local_scaling",
    "dis_sim_local",
    "csls",
]


__all__ = [
    "NoHubnessReduction",
    "LocalScaling",
    "MutualProximity",
    "DisSimLocal",
    "CSLS",
    "hubness_algorithms",
]
