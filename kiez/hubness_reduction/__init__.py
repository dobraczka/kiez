from class_resolver import ClassResolver

from .base import HubnessReduction, NoHubnessReduction
from .csls import CSLS
from .dis_sim import DisSimLocal
from .local_scaling import LocalScaling
from .mutual_proximity import MutualProximity

hubness_reduction_resolver = ClassResolver.from_subclasses(
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
