.. kiez documentation master file, created by
   sphinx-quickstart on Mon Jun 28 15:02:08 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

kiez
====

The kiez library is intended to make hubness reduction methods easily usable for the purpose of entity alignment via (knowledge graph) embeddings.

The central class of kiez serves to bundle all necessary steps to obtain nearest neighbors from source to target entities, given their embeddings:

.. code-block:: python

    from kiez import Kiez
    import numpy as np
    # create example data
    rng = np.random.RandomState(0)
    source = rng.rand(100,50)
    target = rng.rand(100,50)
    # fit and get neighbors
    k_inst = Kiez()
    k_inst.fit(source, target)
    nn_dist, nn_ind = k_inst.kneighbors()

The main feature of kiez lies in the ability to use hubness reduction methods and approximate nearest neighbor (ANN) algorithms. This enables you to profit from the speed advantage of ANN algorithms, while achieving highly accurate nearest neighbor results:

.. code-block:: python

    from kiez import Kiez
    import numpy as np
    # create example data
    rng = np.random.RandomState(0)
    source = rng.rand(100,50)
    target = rng.rand(100,50)
    # prepare algorithm and hubness reduction
    from kiez.neighbors import HNSW
    hnsw = HNSW(n_candidates=10)
    from kiez.hubness_reduction import CSLS
    hr = CSLS()
    # fit and get neighbors
    k_inst = Kiez(n_neighbors=5, algorithm=hnsw, hubness=hr)
    k_inst.fit(source, target)
    nn_dist, nn_ind = k_inst.kneighbors()

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
