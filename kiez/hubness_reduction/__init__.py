from class_resolver import ClassResolver

from .base import HubnessReduction, NoHubnessReduction
from .csls import CSLS
from .dis_sim import DisSimLocal
from .local_scaling import LocalScaling
from .mutual_proximity import MutualProximity
from .new_base import NewHubnessReduction, NewNoHubnessReduction
from .new_csls import NewCSLS
from .new_dis_sim import NewDisSimLocal
from .new_local_scaling import NewLocalScaling
from .new_mutual_proximity import NewMutualProximity

hubness_reduction_resolver = ClassResolver.from_subclasses(
    base=HubnessReduction,
    default=NoHubnessReduction,
)

new_hubness_reduction_resolver = ClassResolver.from_subclasses(
    base=NewHubnessReduction,
    default=NewNoHubnessReduction,
)

#: Supported hubness reduction algorithms
__all__ = [
    "hubness_reduction_resolver",
    "NoHubnessReduction",
    "LocalScaling",
    "MutualProximity",
    "DisSimLocal",
    "CSLS",
    "NewLocalScaling",
    "NewMutualProximity",
    "NewDisSimLocal",
    "NewCSLS",
]
