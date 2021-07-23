# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# adapted from skhubness: https://github.com/VarIr/scikit-hubness/

from class_resolver import Resolver

from .base import HubnessReduction, NoHubnessReduction
from .csls import CSLS
from .dis_sim import DisSimLocal
from .local_scaling import LocalScaling
from .mutual_proximity import MutualProximity

hubness_reduction_resolver = Resolver.from_subclasses(
    base=HubnessReduction,
    default=NoHubnessReduction,
)

#: Supported hubness reduction algorithms
__all__ = [
    "hubness_reduction_resolver",
    "NoHubnessReduction",
    "LocalScaling",
    "MutualProximity",
    "DisSimLocal",
    "CSLS",
]
