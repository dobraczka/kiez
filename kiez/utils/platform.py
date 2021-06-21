# SPDX-License-Identifier: BSD-3-Clause
# adapted from skhubness https://github.com/VarIr/scikit-hubness/
import sys


def available_ann_algorithms_on_current_platform():
    """Get approximate nearest neighbor algorithms available for the current platform/OS
    Currently, the algorithms are provided by the following libraries:
        * 'hnsw': nmslib
        * 'rptree': annoy
        * 'nng': NGT
    Returns
    -------
    algorithms: Tuple[str]
        A tuple of available algorithms
    """
    # Windows
    if sys.platform == "win32":
        algorithms = (
            "hnsw",
            "rptree",
        )
    # Mac and Linux
    elif sys.platform in ["linux", "darwin"]:
        algorithms = (
            "hnsw",
            "rptree",
            "nng",
        )
    # others undefined
    else:  # pragma: no cover
        algorithms = ()

    return algorithms
